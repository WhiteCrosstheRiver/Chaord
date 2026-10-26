# -*- coding: utf-8 -*-
"""Y8 go/no-go block analysis -- tranche 1 (docs/reviews/open_items_v4.md Y8;
decisions D31/D32 binding; floor estimator D15; block protocol of
reports/research/stage_b_plan.md).

Tranche 1 = (10,10) tube at 0.01 V/A (working field) and 0.05 V/A, 3
independent runs each (sb_cnt10_e10_r{1,2,3}, sb_cnt10_e50_r{1,2,3}), plus
the zero-field side from the DONE Stage A zero-field runs
(zf_cnt10_b{1,2,3}) -- D31: floors come from zero-field runs ONLY.

TWO INDEPENDENT ARMS (D32's second arm; computed independently, shared
nothing but the block definitions and the subsample index list):

  Arm 1 (Chaord): the Y4 reader (chaord.io.lammps.read_lammps_dump, D34)
  reads the dump with the type map of the PROVENANCE BUILD data file (the
  same convention as analyze_first_result.read_prod_frames -- never
  final.data), and every subsampled frame is lifted with IDENTICAL
  options: lift_frame(mode="auto"), NO field= (blind, D29), T=300.0 (the
  protocol temperature, metadata) -- via analyze_first_result.lift_one_frame,
  which also runs the static checks.  The statement features are parsed
  from the formatted program text (the record), region-aware, extended
  from analyze_first_result.statement_features to the Y5 statement set
  (dipole_p1, li_shell per species, ion_pairs SSIP/CIP/AGG, the n_r
  species radial profiles reduced to summary scalars, density
  (accessible), linear_density, gr_peak).  Chirality and the molecule
  census are constants here -- recorded, excluded as non-moving.

  Arm 2 (direct P1 without chaord): numpy/stdlib only (the
  orientation_check.py pattern).  Its own dump streamer reads EVERY
  production frame; P1 = mean over solvent molecules (EC 10 atoms, EMC 15
  atoms, identified by mol id -- never a chaord bond graph) of
  mu_z/|mu| with minimum-image offsets from each molecule's first atom
  and the dump's own charges.  No chaord symbol is used anywhere in the
  arm-2 functions.

BLOCK PROTOCOL (stage_b_plan.md section 2, verified against the dumps):
prod.dump = the continuous production trajectory after the 1 ns settle,
reset_timestep 0, 1 ps cadence.  sb jobs: 12,001 frames; block A =
frames 0-5999 (production t < 6000 ps), block B = frames 6000-12000
(t >= 6000 ps).  zf jobs: 2,001 frames (2 ns); two 1-ns blocks split at
frame 1000.  Block B is the PRIMARY block (stage_b_plan.md section 3.2:
block A of field-on runs retains ~27% of the switch-on transient for
cnt10); block A is disclosed as transient-biased and used in the
sensitivity checks only.

SUBSAMPLING (stated, honest): 12,001 frames x 6 sb jobs is heavy for the
lift arm; every SB_STRIDE-th frame is lifted (201 frames per job = 100
in block A + 101 in block B).  The zf jobs are subsampled at
ZF_STRIDE = 10 -> 201 frames per job (100 + 101), so both sides carry
the same number of block frames.  Arm 2 additionally computes P1 on
EVERY frame (the "dense" variant), which quantifies what the
subsampling costs.  The subsampling error enters both sides' block
means symmetrically and the floors are derived from the same-cadence zf
blocks, so the D15 floors already include it.

FLOORS (D15 estimator, D31 pair set): per statement, the values are the
zf BLOCK MEANS; the pair set is every pair of zf blocks from DIFFERENT
runs (3 builds x 2 blocks = 6 blocks -> 12 cross-run pairs); floor =
max(mean |delta|, P90 |delta|).  Choice stated: zf blocks are 1 ns
while the sb primary blocks are 6 ns; a block-mean's variance scales
~ 2 tau/L, so the 1-ns zf floors OVERESTIMATE the 6-ns block noise by
~ sqrt(6) -- conservative (harder to detect), never looser.

DETECTION: per field, |mean_on - mean_off| vs 3 x floor for EVERY
statement; mean_on = mean over the 3 runs of their block-B means (sem
over the 3 runs = the primary error bar, D31); mean_off = mean over the
6 zf blocks.  Secondary: a Gaussian z with the two sems (indicative
only -- blocks within a run are correlated; runs are the units).

CLASSIFIER: threshold on the dipole_p1 statement value (the RD4
evidence, the same quantity arm 2 measures directly), units = blocks,
threshold = midpoint of class means (analyze_first_result.threshold_fit),
scored by leave-one-RUN-out: each fold holds out one run entirely (both
classes contribute folds), refits the threshold on the other runs'
blocks, and classifies the held-out run's blocks.  Primary variant:
block-B blocks on the field-on side (1 per run) vs all 6 zf blocks.
Sensitivity: all blocks (A+B) on both sides.

LINEAR RESPONSE: P1 at 0 (zf), 0.01 and 0.05 V/A; two slopes
(P1(0.01)-P1(0))/0.01 and (P1(0.05)-P1(0))/0.05 compared within their
run-level errors.  Two field points + the zero: what can and cannot be
concluded is stated in the report, not here.

Every number is deterministic (no randomness anywhere).  Output:
tranche1_results.json next to this file, plus a console summary.

Usage (from the repository root):
    python research/cnt_electrolyte/analysis/y8_block_analysis.py
    python research/cnt_electrolyte/analysis/y8_block_analysis.py --smoke
        # 3 frames per job, pipeline test only, never for the record
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parents[1]          # research/cnt_electrolyte
for _p in (str(ROOT / "src"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Arm 1 imports (chaord).  Arm 2 uses only numpy/stdlib -- see its section.
from analyze_first_result import (                    # noqa: E402
    T_K, composition_dialect, lift_one_frame)
from chaord.io.lammps import read_lammps_data, read_lammps_dump  # noqa: E402
from chaord.lang.fmt import format_program            # noqa: E402

OUT_JSON = Path(__file__).resolve().parent / "tranche1_results.json"

# ---- tranche-1 constants (verified against the dumps at run time) ----
TUBE = 10                                   # the (10,10) tranche
SB_FIELDS = {0.01, 0.05}                    # V/A, the first tranche
SB_FRAMES, SB_SPLIT_PS, SB_STRIDE = 12001, 6000, 60
ZF_FRAMES, ZF_SPLIT_PS, ZF_STRIDE = 2001, 1000, 10
DT_PS = 1.0                                 # production dump cadence
CHARGE_TOL = 1e-6                           # cell neutrality (e)
EXPECTED_CENSUS = {"EC": 8, "EMC": 5, "PF6": 2, "Li": 2}   # asserted, per frame
ATOMS_PER = {"EC": 10, "EMC": 15, "PF6": 7, "Li": 1}       # stageA convention
N_RUNS = 3

_NUM = r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"


# ---------------------------------------------------------------------------
# provenance: which jobs, which build data file (type map source)
# ---------------------------------------------------------------------------

def discover_jobs() -> tuple[list[dict], list[dict]]:
    """(sb jobs, zf jobs) of the cnt10 tranche, from the queue's own
    provenance records.  An sb job counts as DONE only with a non-smoke
    DONE marker (smoke dumps hold no production)."""
    sb = []
    for man_path in sorted((HERE / "stageB" / "manifests").glob("*.json")):
        man = json.loads(man_path.read_text())
        name = man["job"]
        if not re.fullmatch(rf"sb_cnt{TUBE}_e\d+_r\d", name):
            continue
        done = HERE / "stageB" / "jobs" / name / "DONE"
        if not done.is_file():
            continue
        assert not json.loads(done.read_text())["smoke"], name
        assert man["tube"] == TUBE, name
        assert man["protocol"]["n_blocks"] == 2, name
        sb.append({"name": name, "field": float(man["field"]),
                   "build": man["build"], "kind": "sb"})
    zf_index = json.loads((HERE / "stageA" / "jobs_index.json").read_text())
    zf = []
    for e in zf_index:
        # b1m (the mirrored copy of b1) is the symmetry instrument, NOT an
        # independent start (stage_b_plan.md section 2) -- excluded
        if not re.fullmatch(rf"zf_cnt{TUBE}_b\d", e["name"]):
            continue
        assert e["kind"] == "zf" and e["field"] == 0.0, e["name"]
        zf.append({"name": e["name"], "field": 0.0,
                   "build": e["build"], "kind": "zf",
                   "source": e["source"]})
    assert len(sb) == 2 * N_RUNS and len(zf) == N_RUNS, (len(sb), len(zf))
    fields = sorted({j["field"] for j in sb})
    assert set(fields) == SB_FIELDS, fields

    # type map source: the BUILD data file of the run's own build (D34 --
    # the provenance-recorded input; for sb jobs via the zf build it
    # started from, whose jobs_index entry names the data file)
    zf_src = {j["build"]: j["source"] for j in zf}
    for j in sb:
        j["build_data"] = HERE / zf_src[j["build"]]
    for j in zf:
        j["build_data"] = HERE / j["source"]
    for j in sb + zf:
        assert j["build_data"].is_file(), j
    return sb, zf


def type_map_of(build_data: Path):
    """The Y4 type map from the build data file (the first-result
    convention; never final.data)."""
    _, meta = read_lammps_data(str(build_data))
    return meta["type_map"]


# ---------------------------------------------------------------------------
# Arm 2 -- direct P1 without chaord (numpy/stdlib only; the
# orientation_check.py pattern).  No chaord symbol below this line.
# ---------------------------------------------------------------------------

def stream_frames(path: Path):
    """Yield (frame_index, timestep, box(3,2), rows[N,10]) for EVERY frame
    of a LAMMPS custom dump (id mol type q x y z ix iy iz).  Line-based,
    no chaord."""
    with open(path) as f:
        k = -1
        while True:
            line = f.readline()
            if not line:
                return
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            k += 1
            t = int(f.readline())
            hdr = f.readline()
            assert hdr.startswith("ITEM: NUMBER OF ATOMS"), hdr
            n = int(f.readline())
            f.readline()                                    # BOX BOUNDS
            box = np.array([f.readline().split()[:2] for _ in range(3)],
                           float)
            f.readline()                                    # ATOMS header
            rows = np.empty((n, 10), float)
            for i in range(n):
                rows[i] = f.readline().split()[:10]
            yield k, t, box, rows[rows[:, 0].argsort()]


def build_file_census(path: Path) -> tuple[int, dict, dict]:
    """(n_atoms, electrolyte census, type->mass map) from the BUILD data
    file, chaord-free (the orientation_check.atoms pattern).  Molecules
    are mol ids; sizes map to species (EC 10, EMC 15, PF6 7, Li 1); the
    one pure-carbon molecule (the tube) is expected and excluded from the
    census -- any other unclassifiable molecule is an error."""
    L = Path(path).read_text().splitlines()
    i = next(k for k, l in enumerate(L) if l.startswith("Masses"))
    type_mass = {}
    for l in L[i + 1:]:
        p = l.split()
        if len(p) >= 2:
            try:
                type_mass[int(p[0])] = float(p[1])
            except ValueError:
                break
        elif not l.strip():
            if type_mass:
                break
    i = next(k for k, l in enumerate(L) if l.startswith("Atoms"))
    rows = []
    for l in L[i + 2:]:
        if not l.strip():
            break
        rows.append(l.split()[:7])
    a = np.array(rows, float)
    census = {}
    tube_mols = 0
    for m in np.unique(a[:, 1].astype(int)):
        idx = np.where(a[:, 1].astype(int) == m)[0]
        size = len(idx)
        sp = {v: k for k, v in ATOMS_PER.items()}.get(size)
        if sp is None:
            assert all(abs(type_mass[int(t)] - 12.011) < 0.01
                       for t in a[idx, 2]), \
                f"molecule of {size} atoms in {path.name} is not the carbon tube"
            tube_mols += 1
            continue
        census[sp] = census.get(sp, 0) + 1
    assert tube_mols == 1, f"{path.name}: {tube_mols} non-electrolyte molecules"
    return len(a), census, type_mass


def classify_frame_census(mol_ids: np.ndarray, types: np.ndarray,
                          type_mass: dict) -> dict:
    """Per-frame electrolyte census (same rule as build_file_census)."""
    census = {}
    tube_mols = 0
    for m in np.unique(mol_ids):
        idx = np.where(mol_ids == m)[0]
        sp = {v: k for k, v in ATOMS_PER.items()}.get(len(idx))
        if sp is None:
            assert all(abs(type_mass[int(t)] - 12.011) < 0.01
                       for t in types[idx]), \
                f"molecule of {len(idx)} atoms is not the carbon tube"
            tube_mols += 1
            continue
        census[sp] = census.get(sp, 0) + 1
    assert tube_mols == 1, f"{tube_mols} non-electrolyte molecules in frame"
    return census


def direct_arm(path: Path, build_data: Path, subsample: list[int],
               is_sb: bool, dense_cap: int | None = None) -> dict:
    """Arm 2 on one job: P1 of the solvent on EVERY frame (dense) plus the
    per-frame verification; returns dense and subsampled series.
    dense_cap (smoke only) stops the stream early and lifts the full-file
    frame-count assert -- never for the record."""
    want_atoms, want_census, type_mass = build_file_census(build_data)
    p1, p1_ec, p1_emc = [], [], []
    n_frames = 0
    t_first = t_last = None
    for k, t, box, a in stream_frames(path):
        if n_frames == 0:
            t_first = t
        t_last = t
        assert len(a) == want_atoms, (path.name, k, len(a), want_atoms)
        assert abs(a[:, 3].sum()) < CHARGE_TOL, (path.name, k, "charge")
        census = classify_frame_census(a[:, 1].astype(int), a[:, 2],
                                       type_mass)
        assert census == want_census, (path.name, k, census)
        L = box[:, 1] - box[:, 0]
        cos, cos_ec, cos_emc = [], [], []
        for m in np.unique(a[:, 1].astype(int)):
            idx = np.where(a[:, 1].astype(int) == m)[0]
            n_at = len(idx)
            if n_at not in (ATOMS_PER["EC"], ATOMS_PER["EMC"]):
                continue
            d = a[idx, 4:7] - a[idx[0], 4:7]
            d -= L * np.round(d / L)
            mu = (a[idx, 3][:, None] * d).sum(0)
            c = float(mu[2] / np.linalg.norm(mu))
            cos.append(c)
            (cos_ec if n_at == ATOMS_PER["EC"] else cos_emc).append(c)
        p1.append(float(np.mean(cos)))
        p1_ec.append(float(np.mean(cos_ec)))
        p1_emc.append(float(np.mean(cos_emc)))
        n_frames += 1
        if dense_cap is not None and n_frames >= dense_cap:
            break
    if dense_cap is None:
        assert n_frames in (SB_FRAMES, ZF_FRAMES), (path.name, n_frames)
    else:
        n_frames = dense_cap                  # smoke: capped stream
    dt_ps = (t_last - t_first) / max(n_frames - 1, 1) / 1000.0
    assert abs(dt_ps - DT_PS) < 1e-9, (path.name, dt_ps)
    split_ps = SB_SPLIT_PS if is_sb else ZF_SPLIT_PS
    split_frames = split_ps                   # 1 frame per ps
    return {
        "n_frames": n_frames, "t_first": t_first, "t_last": t_last,
        "split_ps": split_ps, "split_frames": split_frames,
        "census": want_census, "n_atoms": want_atoms,
        "p1_dense": np.array(p1), "p1_ec_dense": np.array(p1_ec),
        "p1_emc_dense": np.array(p1_emc),
        "p1_sub": np.array(p1)[subsample],
    }


# ---------------------------------------------------------------------------
# Arm 1 -- the Chaord lift (Y4 reader -> blind lift -> statement features)
# ---------------------------------------------------------------------------

def y8_statement_features(text: str) -> tuple[dict, dict]:
    """Region-aware numeric statement features of one composed program,
    extended from analyze_first_result.statement_features to the Y5 set.

    Returns (feats, consts): feats maps flat names to floats -- the
    system-level statements carry no prefix, the per-region statements
    are prefixed with the region name (fluid / outside / ...).  n_r
    profiles (arrays) are reduced to three scalars per species, stated
    here: the profile's peak value, the peak bin's radius, and the
    density-weighted mean radius (NaN when the profile is empty).
    consts records the non-moving statements (chirality, census, cell).
    """
    feats: dict = {}
    consts: dict = {"molecules": {}}
    region = None
    for raw in text.splitlines():
        line = raw.strip()
        if line == "system {":
            region = "system"
            continue
        m = re.match(r"(?:crystal|liquid) (\S+) :", line)
        if m and line.endswith("{"):
            region = m.group(1)
            continue
        if line == "}":
            region = None
            continue
        if region is None:
            continue
        pre = "" if region == "system" else region + "."
        m = re.fullmatch(rf"assert dipole_p1 {_NUM} \+- {_NUM} along ([xyz])",
                         line)
        if m:
            feats["dipole_p1"] = float(m.group(1))
            consts["p1_axis"] = m.group(3)
            continue
        m = re.fullmatch(rf"state density {_NUM} g/cm3( accessible)?", line)
        if m:
            feats[pre + "density"] = float(m.group(1))
            consts["density_variant"] = ("accessible" if m.group(2)
                                         else "region_volume")
            continue
        m = re.fullmatch(rf"state linear_density {_NUM} A\^-1", line)
        if m:
            feats[pre + "linear_density"] = float(m.group(1))
            continue
        m = re.fullmatch(rf"assert li_shell (\S+) {_NUM} \+- {_NUM} "
                         rf"(\S+) {_NUM} A", line)
        if m:
            feats[f"{pre}li_shell.{m.group(1)}"] = float(m.group(2))
            continue
        m = re.fullmatch(rf"assert ion_pairs SSIP (-?\d+) CIP (-?\d+) "
                         rf"AGG (-?\d+)", line)
        if m:
            for cls, v in zip(("SSIP", "CIP", "AGG"), m.groups()):
                feats[f"{pre}ion_pairs.{cls}"] = float(v)
            continue
        m = re.fullmatch(rf"state n_r (\S+) (.+?) bin {_NUM} A", line)
        if m:
            vals = np.array([float(x) for x in m.group(2).split()])
            w = float(m.group(3))
            r = (np.arange(len(vals)) + 0.5) * w
            tot = vals.sum()
            feats[f"{pre}nr.{m.group(1)}.peak"] = float(vals.max())
            feats[f"{pre}nr.{m.group(1)}.peak_r"] = float(r[int(np.argmax(vals))])
            feats[f"{pre}nr.{m.group(1)}.mean_r"] = (
                float((vals * r).sum() / tot) if tot > 0 else float("nan"))
            continue
        m = re.fullmatch(rf"assert gr_peak {_NUM} A height {_NUM}", line)
        if m:
            feats[pre + "gr_peak_r"] = float(m.group(1))
            feats[pre + "gr_peak_h"] = float(m.group(2))
            continue
        m = re.fullmatch(rf"nanotube \((\d+),(\d+)\) length {_NUM} A", line)
        if m:
            consts["chirality"] = [int(m.group(1)), int(m.group(2))]
            consts["tube_length_A"] = float(m.group(3))
            continue
        m = re.fullmatch(r"molecules (\S+) (\d+)", line)
        if m:
            consts["molecules"][m.group(1)] = int(m.group(2))
            continue
        m = re.fullmatch(rf"cell {_NUM} {_NUM} {_NUM}", line)
        if m:
            consts["cell"] = [float(m.group(i)) for i in (1, 2, 3)]
            continue
    return feats, consts


def chaord_arm(job: dict, subsample: list[int], dl) -> dict:
    """Arm 1 on one job: Y4-read the subsampled frames, lift each with
    identical blind options, parse the statement features."""
    tmap = type_map_of(job["build_data"])
    base = HERE / ("stageB" if job["kind"] == "sb" else "stageA") / "jobs"
    path = base / job["name"] / "prod.dump"
    frames = read_lammps_dump(str(path), frame_indices=subsample,
                              type_map=tmap)
    unknown = tmap.apply(frames, strict=True)
    assert not unknown, f"{job['name']}: unmapped type ids {unknown}"
    assert len(frames) == len(subsample)
    feats_list, consts_list, failed = [], [], []
    p1_text = []
    for k, fr in zip(subsample, frames):
        assert fr.info["timestep"] == k * 1000, (job["name"], k)
        assert abs(np.asarray(fr.info["charges"]).sum()) < CHARGE_TOL
        text, info = lift_one_frame(fr, None, dl=dl)   # T=300, blind, auto
        feats, consts = y8_statement_features(text)
        feats_list.append(feats)
        consts_list.append(json.dumps(consts, sort_keys=True))
        if info["failed_checks"]:
            failed.append({"frame": k, "checks": info["failed_checks"]})
        p1_text.append(feats["dipole_p1"])
    if failed:
        print(f"[y8] WARNING {job['name']}: {len(failed)} lifted frame(s) "
              f"with failed static checks -- recorded in the JSON",
              file=sys.stderr)
    assert len(set(consts_list)) == 1, \
        f"{job['name']}: the program's non-moving statements move between frames"
    keys = sorted(set().union(*[set(f) for f in feats_list]))
    counts = {k2: sum(k2 in f for f in feats_list) for k2 in keys}
    assert all(c == len(frames) for c in counts.values()), \
        f"{job['name']}: statement presence varies across frames: {counts}"
    return {
        "n_frames_dump": None,       # arm 2 asserts the frame count
        "n_lifted": len(frames),
        "frame_indices": list(subsample),
        "timesteps_ps": [fr.info["timestep"] / 1000.0 for fr in frames],
        "consts": json.loads(consts_list[0]),
        "features": {k2: [f[k2] for f in feats_list] for k2 in keys},
        "failed_checks": failed,
        "p1_text": p1_text,
    }


# ---------------------------------------------------------------------------
# blocks, floors, detection, classifier, linear response
# ---------------------------------------------------------------------------

def blocks_of(values: np.ndarray, sub_idx: np.ndarray, split_ps: float
              ) -> dict:
    """Block means over the subsampled frames: A = t < split, B = t >= split.
    values are aligned with sub_idx."""
    a = values[np.asarray(sub_idx) * DT_PS < split_ps]
    b = values[np.asarray(sub_idx) * DT_PS >= split_ps]
    return {"A": float(np.nanmean(a)), "B": float(np.nanmean(b)),
            "nA": int(np.sum(~np.isnan(a))), "nB": int(np.sum(~np.isnan(b)))}


def cross_run_block_values(zf_results: list[dict], key: str, which: str
                           ) -> tuple[np.ndarray, list[str]]:
    """The zf block means of one statement: (values, run ids), 2 blocks per
    run.  which in {"A+B" (all blocks), "B"}."""
    vals, runs = [], []
    for r in zf_results:
        for blk in (r["blocks"][key]["A"], r["blocks"][key]["B"]):
            if which == "B" and blk is not r["blocks"][key]["B"]:
                continue
            vals.append(blk)
            runs.append(r["name"])
    return np.array(vals), runs


def d15_floor(values: np.ndarray, run_ids: list[str]) -> dict:
    """The D15 estimator max(mean, P90) over |delta| between blocks from
    DIFFERENT runs (D31's pair set; 12 pairs for 3 runs x 2 blocks)."""
    d = []
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            if run_ids[i] != run_ids[j]:
                d.append(abs(values[i] - values[j]))
    d = np.array(d)
    return {"mean": float(d.mean()), "p90": float(np.percentile(d, 90)),
            "floor": float(max(d.mean(), np.percentile(d, 90))),
            "n_pairs": int(len(d)), "deltas": [float(x) for x in d]}


def state_level(sb_results: list[dict], key: str, which: str) -> dict:
    """mean/sem over the runs of one field (D31: runs are the units).
    which="B": one block-B mean per run.  which="A+B": each run's
    frame-weighted all-blocks mean."""
    per_run = []
    for r in sb_results:
        if which == "B":
            per_run.append(r["blocks"][key]["B"])
        else:
            nA, nB = r["blocks"][key]["nA"], r["blocks"][key]["nB"]
            per_run.append((nA * r["blocks"][key]["A"]
                            + nB * r["blocks"][key]["B"]) / (nA + nB))
    v = np.array(per_run, float)
    return {"mean": float(v.mean()),
            "sem": float(v.std(ddof=1) / math.sqrt(len(v))),
            "per_run": [float(x) for x in v]}


def zf_level(zf_results: list[dict], key: str) -> dict:
    """mean over the 6 zf blocks + sem over the 3 run means."""
    per_run = []
    for r in zf_results:
        nA, nB = r["blocks"][key]["nA"], r["blocks"][key]["nB"]
        per_run.append((nA * r["blocks"][key]["A"]
                        + nB * r["blocks"][key]["B"]) / (nA + nB))
    v = np.array(per_run, float)
    blocks, runs = cross_run_block_values(zf_results, key, "A+B")
    return {"mean": float(blocks.mean()),
            "sem": float(v.std(ddof=1) / math.sqrt(len(v))),
            "per_run": [float(x) for x in v]}


def loro_classifier(on_vals: np.ndarray, on_runs: list[str],
                    off_vals: np.ndarray, off_runs: list[str]) -> dict:
    """Threshold classifier on block values, leave-one-RUN-out folds over
    both classes; threshold = midpoint of the training class means."""
    units = [(float(v), r, 1) for v, r in zip(on_vals, on_runs)] + \
            [(float(v), r, 0) for v, r in zip(off_vals, off_runs)]
    folds = []
    for run in sorted({u[1] for u in units}):
        train = [u for u in units if u[1] != run]
        test = [u for u in units if u[1] == run]
        t = (np.mean([u[0] for u in train if u[2] == 1])
             + np.mean([u[0] for u in train if u[2] == 0])) / 2
        for v, r, lab in test:
            folds.append({"held_out_run": r, "value": v, "label": lab,
                          "pred": int(v > t), "threshold": float(t)})
    n_ok = sum(f["pred"] == f["label"] for f in folds)
    return {"accuracy": n_ok / len(folds), "n_blocks": len(folds),
            "n_correct": n_ok,
            "on_blocks_correct": sum(f["pred"] == f["label"]
                                     for f in folds if f["label"] == 1),
            "off_blocks_correct": sum(f["pred"] == f["label"]
                                      for f in folds if f["label"] == 0),
            "folds": folds}


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------

def subsample_indices(n_frames: int, stride: int) -> list[int]:
    return list(range(0, n_frames, stride))


def run(smoke: bool = False) -> dict:
    t0 = time.perf_counter()
    sb_jobs, zf_jobs = discover_jobs()
    dl = composition_dialect()
    results: dict = {"meta": {}, "jobs": {}, "zf_jobs": {}, "arm2": {},
                     "arm2_zf": {}}
    sb_res, zf_res = [], []

    for job in sb_jobs + zf_jobs:
        is_sb = job["kind"] == "sb"
        n_frames = SB_FRAMES if is_sb else ZF_FRAMES
        stride = SB_STRIDE if is_sb else ZF_STRIDE
        sub = subsample_indices(n_frames, stride)
        if smoke:
            sub = sub[:3]
        print(f"[y8] {job['name']} (field {job['field']} V/A): "
              f"arm 2 dense stream...", flush=True)
        d = direct_arm(HERE / ("stageB/jobs" if is_sb else "stageA/jobs")
                       / job["name"] / "prod.dump", job["build_data"], sub,
                       is_sb=is_sb, dense_cap=(max(sub) + 1 if smoke else None))
        print(f"[y8]   {d['n_frames']} frames read, census+charge verified "
              f"every frame", flush=True)
        print(f"[y8]   arm 1: Y4 read + blind lift of {len(sub)} frames...",
              flush=True)
        c = chaord_arm(job, sub, dl)
        assert not c["failed_checks"], (job["name"], c["failed_checks"])
        split = d["split_ps"]
        sub_idx = np.array(c["frame_indices"], float)
        c["blocks"] = {k: blocks_of(np.array(v), sub_idx, split)
                       for k, v in c["features"].items()}
        # arm agreement: the printed statement vs arm 2 on the same frames
        agree = np.abs(np.array(c["p1_text"]) - d["p1_sub"])
        c["p1_text_vs_direct_max_abs"] = float(agree.max())
        print(f"[y8]   lift ok; dipole_p1 printed-vs-direct max |diff| "
              f"{agree.max():.2e}", flush=True)
        store = results["jobs"] if is_sb else results["zf_jobs"]
        store[job["name"]] = {
            "field": job["field"], "build": job["build"],
            "build_data": str(job["build_data"]),
            "consts": c["consts"], "n_lifted": c["n_lifted"],
            "frame_indices": c["frame_indices"],
            "timesteps_ps": c["timesteps_ps"],
            "features": c["features"], "blocks": c["blocks"],
            "failed_checks": c["failed_checks"],
            "p1_text_vs_direct_max_abs": c["p1_text_vs_direct_max_abs"],
        }
        (results["arm2"] if is_sb else results["arm2_zf"])[job["name"]] = {
            "n_frames": d["n_frames"], "split_ps": d["split_ps"],
            "census": d["census"], "n_atoms": d["n_atoms"],
            "p1_dense": [round(x, 6) for x in d["p1_dense"]],
            "p1_ec_dense": [round(x, 6) for x in d["p1_ec_dense"]],
            "p1_emc_dense": [round(x, 6) for x in d["p1_emc_dense"]],
            "p1_sub": [round(x, 6) for x in d["p1_sub"]],
        }
        (sb_res if is_sb else zf_res).append(
            {"name": job["name"], "field": job["field"],
             "build": job["build"], "blocks": c["blocks"],
             "direct": d, "sub_idx": sub_idx})

    # ---------------- floors, detection, classifier ----------------
    sb_keys = sorted(set().union(*[set(r["blocks"]) for r in sb_res]))
    zf_keys = sorted(set().union(*[set(r["blocks"]) for r in zf_res]))
    assert sb_keys == zf_keys, "the two arms' statement sets differ"
    keys = sb_keys

    def zf_block_vals(key, which):
        vals, runs = [], []
        for r in zf_res:
            blks = ([r["blocks"][key]["A"], r["blocks"][key]["B"]]
                    if which == "A+B" else [r["blocks"][key]["B"]])
            for b in blks:
                vals.append(b)
                runs.append(r["name"])
        return np.array(vals), runs

    floors = {k: d15_floor(*zf_block_vals(k, "A+B")) for k in keys}

    fields = sorted({r["field"] for r in sb_res})
    detection = {}
    for f in fields:
        rs = [r for r in sb_res if r["field"] == f]
        det = {}
        for k in keys:
            on = state_level(rs, k, "B")
            off = zf_level(zf_res, k)
            floor = floors[k]["floor"]
            delta = abs(on["mean"] - off["mean"])
            det[k] = {
                "mean_on_B": on["mean"], "sem_on_B": on["sem"],
                "per_run_B": on["per_run"],
                "mean_off": off["mean"], "sem_off": off["sem"],
                "per_run_off": off["per_run"],
                "delta": delta, "floor": floor,
                "floor_zero": bool(floor == 0),
                "ratio": delta / floor if floor > 0 else float("inf"),
                "pass_3x": bool(delta > 3 * floor),
                "z": delta / math.hypot(on["sem"], off["sem"])
                if (on["sem"] or off["sem"]) else float("inf"),
                "mean_on_allblocks": state_level(rs, k, "A+B")["mean"],
            }
        detection[f"{f:g}"] = det

    # SECONDARY DIAGNOSTICS -- disclosed, never the gate (the ordered gate
    # is the unpaired 3x-floor test above):
    #
    # (a) paired-by-build contrast.  Each sb run rK starts from zf build
    #     bK's final state, so the long-lived zero-field orientation
    #     offset of build K is shared by the pair; the paired difference
    #     removes it.  Only valid to the extent the offset survives into
    #     the field-on run (a model, not the ordered test).
    paired = {}
    for f in fields:
        rs = [r for r in sb_res if r["field"] == f]
        rows = {}
        for k in keys:
            zf_by_build = {}
            for r in zf_res:
                b = r["blocks"][k]
                zf_by_build[r["build"]] = (
                    (b["A"] * b["nA"] + b["B"] * b["nB"]) / (b["nA"] + b["nB"]))
            pairs = [r["blocks"][k]["B"] - zf_by_build[r["build"]]
                     for r in rs]
            v = np.array(pairs)
            rows[k] = {"per_build": [float(x) for x in v],
                       "mean": float(v.mean()),
                       "sem": float(v.std(ddof=1) / math.sqrt(len(v)))
                       if len(v) > 1 else 0.0}
        paired[f"{f:g}"] = rows

    # (b) floor decomposition: the D15 pair set excludes same-run pairs;
    #     the within-run |A-B| deltas (3 pairs) estimate the floor the
    #     zf blocks would have WITHOUT the cross-build zero-field offsets.
    floor_decomp = {}
    for k in keys:
        within = [abs(r["blocks"][k]["A"] - r["blocks"][k]["B"])
                  for r in zf_res]
        floor_decomp[k] = {
            "within_run_deltas": [float(x) for x in within],
            "within_run_mean": float(np.mean(within)),
            "cross_run_mean": floors[k]["mean"],
        }

    # statement-move list (D32 step 7): beyond 2x floor at or below the
    # working field (0.01), dipole_p1 excluded; e50 listed for the record.
    # A statement MOVES only if delta > 0 as well: floor-0 statements whose
    # delta is also 0 (the structural constants density / linear_density,
    # and ion_pairs.AGG when no aggregation occurs anywhere) have a 0/0
    # ratio and did not move.
    working = 0.01
    moves = {}
    for f, det in detection.items():
        m2 = sorted(k for k, v in det.items()
                    if k != "dipole_p1" and v["ratio"] > 2 and v["delta"] > 0)
        m3 = sorted(k for k, v in det.items()
                    if k != "dipole_p1" and v["pass_3x"])
        structural = sorted(k for k, v in det.items()
                            if v["floor_zero"] and v["delta"] == 0)
        moves[f] = {"beyond_2x_floor": m2, "beyond_3x_floor": m3,
                    "constant_by_construction": structural,
                    "at_or_below_working_field":
                        float(f) <= working}

    # classifier: dipole_p1 block values, LORO over runs, per contrast.
    # Chaord arm: the blind statement's printed value; direct arm: the
    # same quantity computed without chaord, on the same blocks (both the
    # dense all-frames blocks and the shared subsample's blocks).
    def zf_dense_means(which):
        vals, runs = [], []
        for r in zf_res:
            n = r["direct"]["n_frames"]
            sf = r["direct"]["split_frames"]
            if which == "all":
                vals += [r["direct"]["p1_dense"][:sf].mean(),
                         r["direct"]["p1_dense"][sf:].mean()]
                runs += [r["name"], r["name"]]
            else:
                vals.append(r["direct"]["p1_dense"][sf:].mean())
                runs.append(r["name"])
        return np.array(vals), runs

    def zf_sub_means():
        vals, runs = [], []
        for r in zf_res:
            si = np.array(r["sub_idx"])
            sf = r["direct"]["split_frames"]
            for sel in (si < sf, si >= sf):
                vals.append(r["direct"]["p1_sub"][sel].mean())
                runs.append(r["name"])
        return np.array(vals), runs

    classifier = {}
    for f in fields:
        rs = [r for r in sb_res if r["field"] == f]
        on_runs = [r["name"] for r in rs]
        on_b = np.array([r["blocks"]["dipole_p1"]["B"] for r in rs])
        on_ab, on_ab_runs = [], []
        for r in rs:
            for w in ("A", "B"):
                on_ab.append(r["blocks"]["dipole_p1"][w])
                on_ab_runs.append(r["name"])
        off_ab, off_runs = zf_block_vals("dipole_p1", "A+B")
        on_dense = np.array(
            [r["direct"]["p1_dense"][r["direct"]["split_frames"]:].mean()
             for r in rs])
        on_dense_all, _ = [], []
        for r in rs:
            sf = r["direct"]["split_frames"]
            on_dense_all += [r["direct"]["p1_dense"][:sf].mean(),
                             r["direct"]["p1_dense"][sf:].mean()]
        on_sub = np.array(
            [r["direct"]["p1_sub"][np.array(r["sub_idx"])
                                   >= r["direct"]["split_frames"]].mean()
             for r in rs])
        classifier[f"{f:g}"] = {
            "feature": "dipole_p1 (the blind statement's value; the direct "
                       "arm uses the same quantity without chaord)",
            "primary_blockB_on_vs_zf_allblocks": loro_classifier(
                on_b, on_runs, off_ab, off_runs),
            "sensitivity_allblocks": loro_classifier(
                np.array(on_ab), on_ab_runs, off_ab, off_runs),
            "direct_p1_dense": loro_classifier(
                on_dense, on_runs, *zf_dense_means("all")),
            "direct_p1_dense_allblocks_on": loro_classifier(
                np.array(on_dense_all), [r["name"] for r in rs for _ in (0, 1)],
                *zf_dense_means("all")),
            "direct_p1_subsampled": loro_classifier(
                on_sub, on_runs, *zf_sub_means()),
        }

    # linear response (both arms, block-B primary on the field-on side;
    # the zero-field point is the zf runs -- no transient there, all their
    # blocks count on both arms)
    linresp = {}
    zf_st = zf_level(zf_res, "dipole_p1")
    zf_dense = [r["direct"]["p1_dense"].mean() for r in zf_res]
    for arm, getter in (
            ("chaord_dipole_p1_statement",
             lambda r: r["blocks"]["dipole_p1"]["B"]),
            ("direct_p1_dense_blockB",
             lambda r: r["direct"]["p1_dense"]
             [r["direct"]["split_frames"]:].mean())):
        pts = {}
        if arm.startswith("chaord"):
            pts[0.0] = {"mean": zf_st["mean"], "sem": zf_st["sem"]}
        else:
            pts[0.0] = {"mean": float(np.mean(zf_dense)),
                        "sem": float(np.std(zf_dense, ddof=1)
                                     / math.sqrt(len(zf_dense)))}
        for f in fields:
            rs = [r for r in sb_res if r["field"] == f]
            vals = [getter(r) for r in rs]
            pts[f] = {"mean": float(np.mean(vals)),
                      "sem": float(np.std(vals, ddof=1)
                                   / math.sqrt(len(vals)))}
        slopes = {}
        for f in fields:
            num = pts[f]["mean"] - pts[0.0]["mean"]
            den = math.hypot(pts[f]["sem"], pts[0.0]["sem"]) / f
            slopes[f"{f:g}"] = {"slope": num / f, "sem": den}
        a = slopes[f"{fields[0]:g}"]
        b = slopes[f"{fields[1]:g}"]
        linresp[arm] = {
            "P1_vs_field": {f"{f:g}": pts[f] for f in sorted(pts)},
            "slopes_per_V_per_A": slopes,
            "slopes_agree_within_2sigma":
                bool(abs(a["slope"] - b["slope"])
                     <= 2 * math.hypot(a["sem"], b["sem"])),
        }

    # block A/B transient check (stage_b_plan.md section 3.3)
    ab_check = {}
    for f in fields:
        rs = [r for r in sb_res if r["field"] == f]
        rows = {}
        for k in keys:
            mA = float(np.mean([r["blocks"][k]["A"] for r in rs]))
            mB = float(np.mean([r["blocks"][k]["B"] for r in rs]))
            run_sem = state_level(rs, k, "A+B")["sem"]
            rows[k] = {"mean_A": mA, "mean_B": mB,
                       "abs_offset_over_2x_run_sem":
                           abs(mA - mB) / (2 * run_sem) if run_sem > 0
                           else float("inf"),
                       "offset_exceeds_2x_run_sem":
                           bool(run_sem > 0 and abs(mA - mB) > 2 * run_sem)}
        ab_check[f"{f:g}"] = rows

    wall = time.perf_counter() - t0
    results["meta"] = {
        "work_order": "docs/reviews/open_items_v4.md Y8 (Review 10)",
        "decisions": "D31 (ensembles/floors/LORO), D32 (two arms), "
                     "D15 (floor estimator), D29 (blind dipole_p1)",
        "tranche": "1 of the stage-B ladder: (10,10) at 0.01 and 0.05 V/A, "
                   "3 runs each; zero-field side = stage-A zf_cnt10_b{1,2,3}",
        "lift_options": "lift_frame(mode=\"auto\"), NO field= (blind), "
                        "T=300.0, one composition dialect for every frame",
        "subsampling": {"sb_stride": SB_STRIDE,
                        "sb_frames_per_job": SB_FRAMES,
                        "sb_subsampled_per_job": len(subsample_indices(SB_FRAMES, SB_STRIDE)),
                        "zf_stride": ZF_STRIDE,
                        "zf_frames_per_job": ZF_FRAMES,
                        "zf_subsampled_per_job": len(subsample_indices(ZF_FRAMES, ZF_STRIDE)),
                        "note": "arm 2 reads every frame (dense); arm 1 "
                                "lifts the stride subsample; identical "
                                "subsample indices on both arms"},
        "block_protocol": {"sb_split_ps": SB_SPLIT_PS,
                           "zf_split_ps": ZF_SPLIT_PS,
                           "primary": "block B (stage_b_plan.md section 3.2)",
                           "zf_blocks": "2 x 1 ns per run (the zf runs are "
                                        "2 ns; floors from cross-run pairs)"},
        "floor_estimator": "D15 max(mean, P90) over |delta| between zf "
                           "block pairs from different runs (12 pairs)",
        "smoke": bool(smoke),
        "wall_seconds": round(wall, 1),
    }
    results["floors"] = floors
    results["detection"] = detection
    results["statement_moves"] = moves
    results["paired_build_contrast_secondary"] = paired
    results["floor_decomposition"] = floor_decomp
    results["classifier"] = classifier
    results["linear_response"] = linresp
    results["block_AB_check"] = ab_check

    if not smoke:
        OUT_JSON.write_text(json.dumps(results, indent=1))
        print(f"[y8] wrote {OUT_JSON}")
    print(f"[y8] wall {wall:.1f} s")
    return results


def print_summary(res: dict) -> None:
    floors = res["floors"]
    print("\n== D15 floors (zf cross-run block pairs, 12 pairs) ==")
    for k in ("dipole_p1", "fluid.density", "fluid.linear_density",
              "fluid.li_shell.C3H4O3", "fluid.li_shell.C4H8O3",
              "fluid.li_shell.F6P", "fluid.ion_pairs.CIP",
              "fluid.nr.Li+.peak", "fluid.nr.Li+.mean_r"):
        if k in floors:
            fl = floors[k]
            print(f"  {k:34s} floor {fl['floor']:.4f} "
                  f"(mean {fl['mean']:.4f}, P90 {fl['p90']:.4f})")
    print("\n== detection: |mean_on(B) - mean_off| vs 3x floor ==")
    for f, det in res["detection"].items():
        p1 = det["dipole_p1"]
        print(f"  field {f} V/A: dipole_p1 on {p1['mean_on_B']:+.4f} "
              f"+- {p1['sem_on_B']:.4f} vs off {p1['mean_off']:+.4f} "
              f"+- {p1['sem_off']:.4f}; delta {p1['delta']:.4f} = "
              f"x{p1['ratio']:.2f} floor ({p1['floor']:.4f}) -> "
              f"{'PASS' if p1['pass_3x'] else 'fail'} at 3x (z {p1['z']:.1f})")
        m2 = res["statement_moves"][f]
        tag = " [at/below working field]" if m2["at_or_below_working_field"] else ""
        print(f"    moves beyond 2x floor (dipole_p1 excluded){tag}: "
              f"{m2['beyond_2x_floor'] or 'none'}")
        print(f"    moves beyond 3x floor: {m2['beyond_3x_floor'] or 'none'}")
    print("\n== classifier (dipole_p1 blocks, leave-one-run-out) ==")
    for f, c in res["classifier"].items():
        pr = c["primary_blockB_on_vs_zf_allblocks"]
        print(f"  field {f} V/A vs zf: chaord statement "
              f"{pr['n_correct']}/{pr['n_blocks']} blocks "
              f"(on {pr['on_blocks_correct']}/3, off "
              f"{pr['off_blocks_correct']}/6); direct P1 subsampled "
              f"{c['direct_p1_subsampled']['n_correct']}/"
              f"{c['direct_p1_subsampled']['n_blocks']}; direct dense "
              f"{c['direct_p1_dense']['n_correct']}/"
              f"{c['direct_p1_dense']['n_blocks']}")
    print("\n== linear response (block B, per run sems) ==")
    for arm, lr in res["linear_response"].items():
        sl = lr["slopes_per_V_per_A"]
        p10 = lr["P1_vs_field"]["0"]
        print(f"  {arm}: P1(0) {p10['mean']:+.4f} "
              f"+- {p10['sem']:.4f}; "
              + "; ".join(f"slope@{f} {s['slope']:.2f}+-{s['sem']:.2f}"
                          for f, s in sl.items())
              + f"; agree within 2 sigma: {lr['slopes_agree_within_2sigma']}")

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true",
                    help="3 frames per job; pipeline test, never for the "
                         "record (no JSON written)")
    args = ap.parse_args()
    res = run(smoke=args.smoke)
    print_summary(res)


if __name__ == "__main__":
    main()
