"""First-result pipeline: lift the four CNT/electrolyte systems and decide
whether the PROGRAMS' STATEMENTS separate field-on from field-off.

The reviewer's binding (docs/reviews/open_items_v2.md section 5, item 3):
"Lift 20 frames with the field off and 20 with it on, for two tube
diameters. Done when the programs' statements, not the coordinates,
separate the states beyond their noise floors, and a classifier that reads
only the statements tells field on from off with at least 95% accuracy."
Each system ships 21 production frames (the 20 the reviewer named, plus
frame 0 at t = 0); all are lifted.

Stages (each stage is a function so tests/test_field_classifier.py can run
and pin them):

  read_prod_frames  read prod.dump through the repo LAMMPS reader (Y4,
                   D34): out-of-box coordinates are wrapped by the reader
                   itself and the excursion recorded in frame.info; the
                   type map comes from the BUILD data file's Masses (the
                   provenance-recorded input), never from final.data
  verify            atom counts, the molecular census, the charge sum and
                   the tube charge against provenance_*.json (the tube
                   indices come from the public nanotube_components)
  lift_one_frame    ONE call of the public API (D35, Y3): lift_frame(frame,
                   dialect, mode="auto", field=...) composes the crystal
                   tube region (chirality inference on the frame's own
                   dump-precision carbons) with the confined fluid region;
                   the composition dialect carries the solvent species view
                   (P1) and the fluid routing view (15, the largest registry
                   molecule of this solvent set)
  p1_measurements  P1 of the solvent (species view [C3H4O3, C4H8O3]) on
                   every frame, plus per-species values for the report
  noise floors     within-system pairwise |dP1| over all frame pairs,
                   floor = max(mean, P90)  (decision D15; radial components
                   are undefined for these confined cells -- see RESULTS.md)
  separation       field1-vs-field0 cross-pair |dP1| per diameter against
                   the floors (the reviewer's "beyond their noise floors")
  classifier       reads ONLY the statement texts: parses each program,
                   extracts the numeric asserts, EXCLUDES the `state field`
                   statement and its provenance note (trivial and circular),
                   and classifies field-on/off -- leave-one-out over all 84
                   frames, per diameter, and cross-diameter (train cnt10 /
                   test cnt16 and back), plus a statements-without-any-
                   field-lines control model

Every number this script prints is reproducible from the committed data:
no randomness anywhere (the lifts and CVs are deterministic).

Usage:  python research/cnt_electrolyte/analyze_first_result.py
        python research/cnt_electrolyte/analyze_first_result.py --quick   # 3
        frames per system (smoke), never for the record
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "src"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from chaord.build.molecules import molecule_census          # noqa: E402
from chaord.dialects import Dialect, load_dialect           # noqa: E402
from chaord.io.lammps import (                               # noqa: E402
    LammpsTypeMap, read_lammps_data, read_lammps_dump)
from chaord.lang.fmt import format_program                   # noqa: E402
from chaord.lift import lift_frame, nanotube_components     # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
SYSTEMS = ("cnt10_field0", "cnt10_field1", "cnt16_field0", "cnt16_field1")

# The solvent species P1 averages over (census formulas of EC and EMC),
# passed as a measurement view over the research dialect (RD4: the
# element-keyed site-charge table cannot express EC's per-site charges, so
# the carbonates join as a view; the frames carry per-atom charges, which
# win over the table). The ions Li+ / PF6- are deliberately NOT averaged:
# P1 is defined on neutral molecules only (cv/field.py).
SOLVENT_SPECIES = ["C3H4O3", "C4H8O3"]

# Production temperature of all four runs (provenance protocol.temperature_K).
T_K = 300.0

# ---------------------------------------------------------------- dialect --

# Y3 (open_items_v4.md, D35): the stack now carries CARBON -- the tube
# inference's tolerances (cn_cutoff 1.85 A, the D33 gates) live there, and
# lift_frame(mode="auto") needs them to compose tube plus liquid. Without
# carbon the molecular dialect's cn_cutoff (3.10 A, an O-O first shell)
# would shadow the C-C one.
BASE_DIALECT_NAMES = ("core", "molecular", "carbon", "research")


def research_dialect(species: list[str] | None = None) -> Dialect:
    """The research stack, optionally with `field_dipole_species` overridden
    by the caller's measurement view (RD4)."""
    dl = load_dialect(BASE_DIALECT_NAMES)
    if species is None:
        return dl
    thresholds = dict(dl._data["thresholds"])
    thresholds["field_dipole_species"] = list(species)
    return Dialect(dl.names, {**dl._data, "thresholds": thresholds},
                   dl.versions)


# ------------------------------------------------------------------- input --

def read_prod_frames(system: str) -> tuple[list, dict, LammpsTypeMap]:
    """(frames, provenance, type_map) of one system's 21 production frames.

    The Y4 reader (D34) wraps the small out-of-box excursions itself and
    records the largest one in frame.info["max_fractional_excursion"]; this
    function only reports it per system (recorded in RESULTS.md). The type
    map comes from the BUILD data file the provenance names
    (outputs.data_file) -- dumps carry integer type ids only, and the lift
    must not lean on final.data."""
    base = RUNS / system
    provenance = json.loads((base / f"provenance_{system}.json").read_text())
    build_data = HERE / provenance["outputs"]["data_file"]
    _, data_meta = read_lammps_data(build_data)
    tmap: LammpsTypeMap = data_meta["type_map"]

    frames = read_lammps_dump(base / "prod.dump", type_map=tmap)
    unknown = tmap.apply(frames, strict=True)
    assert not unknown, f"{system}: unmapped type ids {unknown}"
    provenance["_max_fractional_excursion"] = max(
        fr.info["max_fractional_excursion"] for fr in frames)
    return frames, provenance, tmap


def tube_indices(frame, dl: Dialect) -> np.ndarray:
    """Indices of the frame's nanotube (the public nanotube_components of
    D35; the composed lift identifies the tube itself, and this reports
    the same identification for verification)."""
    tubes = nanotube_components(frame, dl)
    assert len(tubes) == 1, f"expected one tube, found {len(tubes)}"
    return tubes[0].indices


def verify_against_provenance(system: str, frames: list,
                              provenance: dict, dl: Dialect) -> dict:
    """Fail-closed verification of every frame against the provenance record
    (atom count, molecular census of the electrolyte, charge sum, tube
    charge). Returns the per-frame census actually measured."""
    want_atoms = int(provenance["system_definition"]["n_atoms_total"])
    want_comp = {k: int(v) for k, v in
                 provenance["sanity"]["composition_observed"].items()}
    per_frame = []
    for fr in frames:
        assert len(fr) == want_atoms, \
            f"{system}: frame holds {len(fr)} atoms, provenance says {want_atoms}"
        q = np.asarray(fr.info["charges"], float)
        assert abs(q.sum()) < 1e-6, \
            f"{system}: cell charge {q.sum():.3e} e is not neutral"
        idx = tube_indices(fr, dl)
        assert abs(q[idx].sum()) < 1e-6, \
            f"{system}: the tube carries charge"
        keep = np.setdiff1d(np.arange(len(fr)), idx)
        from chaord.io.frames import Frame
        electrolyte = Frame(pos=fr.pos[keep], cell=fr.cell,
                            symbols=[fr.symbols[i] for i in keep],
                            pbc=fr.pbc, info={"charges": q[keep]})
        census = molecule_census(electrolyte, dl)
        got = {"EC": census.get("C3H4O3", 0), "EMC": census.get("C4H8O3", 0),
               "Li": census.get("Li", 0), "PF6": census.get("F6P", 0)}
        assert got == want_comp, \
            f"{system}: census {got} != provenance composition {want_comp}"
        per_frame.append(census)
    return {"n_frames": len(frames), "n_atoms": want_atoms,
            "composition": want_comp,
            "census_frame_stable": len({json.dumps(c, sort_keys=True)
                                        for c in per_frame}) == 1}


# -------------------------------------------------------------------- lift --

# The composed program's own statements are the record: the tube's
# chirality, radius and length are parsed back out of the lifted text (the
# research script reads only the public API's product).
_TUBE_LINE = re.compile(
    r"crystal tube : cylinder axis ([xyz]) .*\{\n"
    r"\s+nanotube \((\d+),(\d+)\) length (\d+\.\d+) A")
_TUBE_RADIUS = re.compile(
    r"crystal tube : cylinder axis [xyz] center [-\d.]+ [-\d.]+ [-\d.]+ "
    r"radius (\d+\.\d+) A")


def tube_from_text(text: str) -> dict:
    """The tube the composed program states: chirality, length, radius."""
    m = _TUBE_LINE.search(text)
    r = _TUBE_RADIUS.search(text)
    assert m and r, text[:400]
    return {"axis": m.group(1), "n": int(m.group(2)), "m": int(m.group(3)),
            "length": float(m.group(4)), "radius": float(r.group(1))}


def lift_one_frame(frame, field: dict | None, dl: Dialect | None = None
                   ) -> tuple[str, dict]:
    """The composed research program of one frame (D35) + its diagnostics.

    ONE call of the public API: lift_frame(frame, dl, mode="auto",
    field=field) identifies the tube from the frame's own dump-precision
    carbons (Y3's inference -- no final.data reference, no cell tripling),
    lifts it as the crystal tube region in its cylinder, the atoms inside
    the cylinder as the confined fluid, and asserts the blind dipole_p1
    along the tube's axis (D29)."""
    dl = dl if dl is not None else composition_dialect()
    program = lift_frame(frame, dl, T=T_K, mode="auto", field=field)
    from chaord.check.statics import run_checks
    failed = [f"{c.name}: {c.detail}" for c in
              run_checks(program, frame, dl) if not c.passed]
    text = format_program(program)
    return text, {"tube": tube_from_text(text), "failed_checks": failed}


# The fluid-routing view of the research stack (RD4): molecular.yaml's
# fluid_max_bonded_component (3 atoms, the water/CO2 class) explicitly
# names the registry's larger SPECIES templates "routing exceptions, not a
# raised bound"; the research solvent's largest molecule is EMC
# (C4H8O3, 15 atoms), so the composition sets the bound to exactly that.
# bonded_single_phase then still performs its real check (every bond-graph
# component one molecule <= 15 atoms -- it would refuse two bonded ECMs).
FLUID_MAX_BONDED_COMPONENT_VIEW = 15


def composition_dialect() -> Dialect:
    """The dialect the composed lift runs under: the research stack (with
    carbon, Y3) plus the solvent species view (P1) and the fluid routing
    view (molecules above the water-class bound)."""
    dl = research_dialect(SOLVENT_SPECIES)
    thresholds = dict(dl._data["thresholds"])
    thresholds["fluid_max_bonded_component"] = FLUID_MAX_BONDED_COMPONENT_VIEW
    return Dialect(dl.names, {**dl._data, "thresholds": thresholds},
                   dl.versions)


# ------------------------------------------------------- P1 and the floors --

def p1_of(frame, species: list[str]) -> dict:
    """P1 over `species` (a measurement view), from the frame's charges."""
    from chaord.cv.field import dipole_orientation
    return dipole_orientation(frame, research_dialect(species))


def pairwise_deltas(values: np.ndarray) -> np.ndarray:
    """All |v_i - v_j| over unordered pairs (i < j)."""
    i, j = np.triu_indices(len(values), k=1)
    return np.abs(values[i] - values[j])


def floor_of(values: np.ndarray) -> dict:
    """The repo's standing noise-floor estimator (decision D15, v2 record):
    max(mean, P90) over pair distances."""
    d = pairwise_deltas(values)
    return {"mean": float(d.mean()), "p90": float(np.percentile(d, 90)),
            "floor": float(max(d.mean(), np.percentile(d, 90))),
            "n_pairs": int(len(d))}


# --------------------------------------------------------------- classifier --

_NUM = r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"


def statement_features(text: str) -> dict:
    """Numeric statement features a classifier may read (statements only).

    Excluded BY CONSTRUCTION (the task's anti-circularity rule): the
    `state field` statement itself and every provenance note -- the regexes
    below never touch them.  The `assert dipole_p1` VALUE is a
    feature (it is the RD4 evidence, a measurement, not the metadata); the
    other numeric asserts (cn, gr_peak) and the state density are features
    too.  Returns floats or None when the statement is absent."""
    feats: dict = {"p1": None, "cn": None, "cn_tol": None, "gr_peak_r": None,
                   "gr_peak_h": None, "density": None}
    for line in text.splitlines():
        s = line.strip()
        m = re.fullmatch(rf"assert dipole_p1 {_NUM} \+- {_NUM} along "
                         r"([xyz])", s)
        if m:
            feats["p1"] = float(m.group(1))
            continue
        m = re.fullmatch(rf"assert cn {_NUM} \+- {_NUM} cutoff {_NUM} A", s)
        if m:
            feats["cn"], feats["cn_tol"] = float(m.group(1)), float(m.group(2))
            continue
        m = re.fullmatch(rf"assert gr_peak {_NUM} A height {_NUM}", s)
        if m:
            feats["gr_peak_r"], feats["gr_peak_h"] = (float(m.group(1)),
                                                      float(m.group(2)))
            continue
        m = re.fullmatch(rf"state density {_NUM} g/cm3", s)
        if m:
            feats["density"] = float(m.group(1))
    return feats


def threshold_fit(values: np.ndarray, labels: np.ndarray) -> float:
    """Midpoint between class means (the simple honest 1-D model; the P1
    physics is a mean of cosines, so the class means are the statistics)."""
    return float((values[labels == 1].mean() + values[labels == 0].mean()) / 2)


def classify_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    return (values > threshold).astype(int)


def encode_features(frames_feats: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """The classifier's design matrix: P1 assert value, absent encoded as 0.

    D29 (open_items_v4.md) made the assert blind -- every program now
    carries a measured value, field or no field -- so the absent branch is
    dead on this pipeline's frames and stays only for hand-made texts; an
    unaligned liquid's P1 is statistically 0 (cosines of uniformly sampled
    orientations), which is what the branch encodes.  The value-only
    content of the assert is the number; the control model below drops it
    entirely."""
    values, labels = [], []
    for f in frames_feats:
        values.append(f["feats"]["p1"] if f["feats"]["p1"] is not None else 0.0)
        labels.append(1 if f["field"] else 0)
    return np.asarray(values), np.asarray(labels)


def loo_accuracy(values: np.ndarray, labels: np.ndarray) -> tuple[float, list]:
    """Leave-one-out accuracy of the threshold model; returns (acc, errors)."""
    errors = []
    for k in range(len(values)):
        train = np.ones(len(values), bool)
        train[k] = False
        t = threshold_fit(values[train], labels[train])
        if int(classify_threshold(values[k:k + 1], t)[0]) != labels[k]:
            errors.append(k)
    return 1.0 - len(errors) / len(values), errors


def centroid_accuracy(feats: list[dict], keys: list[str],
                      train: np.ndarray | None = None,
                      test: np.ndarray | None = None) -> tuple[float, list]:
    """Nearest-class-mean control model over `keys` (the shared statements
    both states carry).  Parameter-free and deterministic.  Returns
    (accuracy, error indices); train/test select subsets for the
    cross-diameter runs."""
    def matrix(sel):
        rows = []
        for i in sel:
            row = [float(feats[i]["feats"][k]) for k in keys]
            if any(v is None for v in row):
                row = [0.0 if v is None else v for v in row]
            rows.append(row)
        m = np.asarray(rows, float)
        m = (m - m.mean(axis=0)) / (m.std(axis=0) + 1e-12)
        return m

    idx = np.arange(len(feats)) if train is None else train
    tst = np.arange(len(feats)) if test is None else test
    labels = np.asarray([1 if feats[i]["field"] else 0 for i in idx])
    x = matrix(idx)
    c1, c0 = x[labels == 1].mean(axis=0), x[labels == 0].mean(axis=0)
    xt = matrix(tst)
    pred = (np.linalg.norm(xt - c1, axis=1)
            <= np.linalg.norm(xt - c0, axis=1)).astype(int)
    truth = np.asarray([1 if feats[i]["field"] else 0 for i in tst])
    errors = [int(tst[k]) for k in range(len(tst)) if pred[k] != truth[k]]
    return float((pred == truth).mean()), errors


# ---------------------------------------------------------------- pipeline --

def field_spec(provenance: dict) -> dict | None:
    """The caller-supplied field of a field-on system (RD3): value from the
    provenance record, axis z from the run input's `fix efield` line."""
    e = provenance["system_definition"].get("external_field_V_per_A")
    if not e:
        return None
    return {"component": "E", "value": float(e), "unit": "V/A", "axis": "z"}


def run_pipeline(frames_per_system: int | None = None) -> dict:
    """The full first-result pipeline; every number in RESULTS.md comes from
    here.  frames_per_system caps the frame count (smoke runs only)."""
    dl = composition_dialect()
    per_system: dict[str, dict] = {}
    records: list[dict] = []
    for system in SYSTEMS:
        frames, provenance, _ = read_prod_frames(system)
        if frames_per_system:
            frames = frames[:frames_per_system]
        check = verify_against_provenance(system, frames, provenance, dl)
        field = field_spec(provenance)
        texts, p1s, n_mols = [], [], []
        ec_p1, emc_p1 = [], []
        failed_checks = []
        radii, chirs = [], []
        for fr in frames:
            text, info = lift_one_frame(fr, field, dl=dl)
            texts.append(text)
            failed_checks.extend(f"{system}: {c}" for c in info["failed_checks"])
            tube = info["tube"]
            radii.append(tube["radius"])
            chirs.append([tube["n"], tube["m"]])
            dip = p1_of(fr, SOLVENT_SPECIES)
            p1s.append(dip["P1"])
            n_mols.append(dip["n_molecules"])
            ec_p1.append(p1_of(fr, ["C3H4O3"])["P1"])
            emc_p1.append(p1_of(fr, ["C4H8O3"])["P1"])
        p1s = np.asarray(p1s)
        assert len({tuple(c) for c in chirs}) == 1, \
            f"{system}: the inferred chirality moves between frames: {chirs}"
        per_system[system] = {
            "field": bool(field), "n_frames": len(frames),
            "verification": check,
            "max_dump_excursion": provenance["_max_fractional_excursion"],
            "chirality": chirs[0],
            "chirality_matches_provenance":
                chirs[0] == [int(c) for c in
                             provenance["system_definition"]["cnt_chirality"]],
            "tube_radius_A": float(np.mean(radii)),
            "tube_radius_spread_A": float(np.ptp(radii)),
            "tube_length_A": tube["length"],
            "p1_mean": float(p1s.mean()), "p1_std": float(p1s.std(ddof=1)),
            "p1_min": float(p1s.min()), "p1_max": float(p1s.max()),
            "p1_floor": floor_of(p1s),
            "ec_p1_mean": float(np.mean(ec_p1)), "emc_p1_mean": float(np.mean(emc_p1)),
            "n_molecules": sorted(set(n_mols)),
        }
        for k, fr in enumerate(frames):
            records.append({
                "system": system, "frame": k,
                "timestep_ps": 5.0 * k,
                "field": bool(field),
                "p1_measured": float(p1s[k]),
                "ec_p1": float(ec_p1[k]), "emc_p1": float(emc_p1[k]),
                "n_molecules": int(n_mols[k]),
                "feats": statement_features(texts[k]),
                "text": texts[k],
            })
    # cross-state separation per diameter
    separation = {}
    for diam, (off, on) in {"cnt10": ("cnt10_field0", "cnt10_field1"),
                            "cnt16": ("cnt16_field0", "cnt16_field1")}.items():
        a = np.asarray([r["p1_measured"] for r in records if r["system"] == on])
        b = np.asarray([r["p1_measured"] for r in records if r["system"] == off])
        cross = np.abs(a[:, None] - b[None, :]).ravel()
        floor = max(per_system[off]["p1_floor"]["floor"],
                    per_system[on]["p1_floor"]["floor"])
        separation[diam] = {
            "cross_mean": float(cross.mean()), "cross_min": float(cross.min()),
            "floor": float(floor),
            "multiple_mean": float(cross.mean() / floor),
            "multiple_worst_pair": float(cross.min() / floor),
            "n_cross_pairs": int(len(cross)),
        }

    # classifiers (statement features only)
    values, labels = encode_features(records)
    loo_acc, loo_err = loo_accuracy(values, labels)
    idx10 = [k for k, r in enumerate(records) if r["system"].startswith("cnt10")]
    idx16 = [k for k, r in enumerate(records) if r["system"].startswith("cnt16")]
    v10, l10 = values[idx10], labels[idx10]
    v16, l16 = values[idx16], labels[idx16]
    t10 = threshold_fit(v10, l10)
    t16 = threshold_fit(v16, l16)
    acc10 = float((classify_threshold(v10, t10) == l10).mean())
    acc16 = float((classify_threshold(v16, t16) == l16).mean())
    cross_10_16 = float((classify_threshold(v16, t10) == l16).mean())
    cross_16_10 = float((classify_threshold(v10, t16) == l10).mean())
    control_keys = ["cn", "cn_tol", "gr_peak_r", "gr_peak_h", "density"]
    ctrl_loo_acc, _ = centroid_accuracy(records, control_keys)
    ctrl_10_16, _ = centroid_accuracy(records, control_keys,
                                      train=np.asarray(idx10),
                                      test=np.asarray(idx16))
    ctrl_16_10, _ = centroid_accuracy(records, control_keys,
                                      train=np.asarray(idx16),
                                      test=np.asarray(idx10))
    summary = {
        "systems": per_system,
        "separation": separation,
        "classifier": {
            "model": "single-feature threshold on the dipole_p1 assert "
                     "value (absent -> 0, the a-priori P1 of an unaligned "
                     "liquid); threshold = midpoint of class means",
            "loo_accuracy": loo_acc, "loo_errors": loo_err,
            "n_frames": len(records),
            "per_diameter": {"cnt10": {"acc": acc10, "threshold": t10},
                             "cnt16": {"acc": acc16, "threshold": t16}},
            "cross_diameter": {"train_cnt10_test_cnt16": cross_10_16,
                               "train_cnt16_test_cnt10": cross_16_10},
            "control_shared_statements_only": {
                "model": "nearest-class-mean over [cn, cn_tol, gr_peak_r, "
                         "gr_peak_h, density] -- every field-related line "
                         "excluded",
                "loo_accuracy": ctrl_loo_acc,
                "train_cnt10_test_cnt16": ctrl_10_16,
                "train_cnt16_test_cnt10": ctrl_16_10},
        },
        "failed_static_checks": sorted(set(failed_checks)),
    }
    return {"summary": summary, "records": records}


def write_outputs(result: dict) -> None:
    out = RUNS / "first_result_summary.json"
    payload = dict(result["summary"])
    payload["records"] = [{k: v for k, v in r.items() if k != "text"}
                          for r in result["records"]]
    payload["program_texts"] = {f"{r['system']}#{r['frame']}": r["text"]
                                for r in result["records"]}
    out.write_text(json.dumps(payload, indent=1))
    print(f"wrote {out}")


def print_report(result: dict) -> None:
    s = result["summary"]
    print("== per-system P1 (solvent species EC+EMC, frame charges) ==")
    for sysname, d in s["systems"].items():
        f = d["p1_floor"]
        print(f"  {sysname}: P1 = {d['p1_mean']:+.3f} +- {d['p1_std']:.3f} "
              f"[{d['p1_min']:+.3f}, {d['p1_max']:+.3f}]  n_mol={d['n_molecules']} "
              f"floor={f['floor']:.4f} (mean {f['mean']:.4f}, P90 {f['p90']:.4f}, "
              f"{f['n_pairs']} pairs)  EC={d['ec_p1_mean']:+.3f} "
              f"EMC={d['emc_p1_mean']:+.3f}")
    print("== cross-state separation (|dP1| field1 vs field0) ==")
    for diam, d in s["separation"].items():
        print(f"  {diam}: mean {d['cross_mean']:.3f}, worst pair "
              f"{d['cross_min']:.3f}, floor {d['floor']:.4f} -> "
              f"x{d['multiple_mean']:.1f} mean, x{d['multiple_worst_pair']:.1f} "
              f"worst pair ({d['n_cross_pairs']} pairs)")
    c = s["classifier"]
    print("== classifier (statements only; state field line + notes excluded) ==")
    print(f"  LOO over {c['n_frames']} frames: {c['loo_accuracy']:.4f} "
          f"(errors at {c['loo_errors']})")
    for diam, d in c["per_diameter"].items():
        print(f"  {diam}: acc {d['acc']:.4f} at threshold {d['threshold']:.3f}")
    print(f"  cross-diameter: train cnt10 -> test cnt16 "
          f"{c['cross_diameter']['train_cnt10_test_cnt16']:.4f}; "
          f"train cnt16 -> test cnt10 "
          f"{c['cross_diameter']['train_cnt16_test_cnt10']:.4f}")
    ctl = c["control_shared_statements_only"]
    print(f"  control (no field-related lines at all): LOO "
          f"{ctl['loo_accuracy']:.4f}, cnt10->cnt16 "
          f"{ctl['train_cnt10_test_cnt16']:.4f}, cnt16->cnt10 "
          f"{ctl['train_cnt16_test_cnt10']:.4f}")
    if s["failed_static_checks"]:
        print("  FAILED STATIC CHECKS:", s["failed_static_checks"])
    else:
        print("  static checks after every composed lift: all pass")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", type=int, default=None, metavar="N",
                    help="first N frames per system (smoke run; not for the record)")
    args = ap.parse_args()
    result = run_pipeline(frames_per_system=args.quick)
    print_report(result)
    write_outputs(result)


if __name__ == "__main__":
    main()
