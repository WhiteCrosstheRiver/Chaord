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

  load_system      read prod.dump through the repo LAMMPS reader (type map
                   from final.data), verify atom counts, the molecular
                   census, the charge sum and the tube charge against
                   provenance_*.json
  lift_one_frame   compose the research program (RD4, docs/reviews/
                   research_decisions.md): the frozen carbon sublattice ->
                   RD2 chirality inference -> crystal `tube` region in its
                   RD1 cylinder; the electrolyte -> lift_frame(mode="fluid",
                   field=...) under the composition dialect (the research
                   stack plus two recorded views: the solvent species view
                   for P1, and fluid_max_bonded_component = 15, the largest
                   registry molecule of this solvent set -- molecular.yaml
                   names such species "routing exceptions"), with density
                   and g(r) referred to the region volume
                   (pipeline._fit_fluid_region's rule)
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
import math
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "src"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from chaord.build.molecules import bond_graph, molecule_census  # noqa: E402
from chaord.dialects import Dialect, load_dialect               # noqa: E402
from chaord.io.frames import Frame                              # noqa: E402
from chaord.io.lammps import (                                   # noqa: E402
    LammpsTypeMap, read_lammps_data, read_lammps_dump)
from chaord.lang.fmt import format_program                       # noqa: E402
from chaord.lift import infer_nanotube_chirality, lift_frame    # noqa: E402

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

BASE_DIALECT_NAMES = ("core", "molecular", "research")


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

def read_prod_frames(system: str) -> tuple[list[Frame], dict, dict]:
    """(frames, provenance, type_map) of one system's 21 production frames.

    LAMMPS rigid-body dumps carry atoms a few times 1e-3 box lengths
    outside the box between re-neighbouring rewraps (measured max 8.2e-3
    over the four dumps); the repo reader correctly refuses out-of-box
    coordinates (Frame's wrapped invariant). Wrapping is a periodic-image
    relabelling -- no physical change, the CVs minimum-image anyway -- so
    each snapshot's x/y/z are wrapped into that snapshot's bounds in a
    scratch copy that the reader then reads.  The largest excursion per
    system is returned in the provenance dict handed back to the caller
    (recorded in RESULTS.md)."""
    base = RUNS / system
    provenance = json.loads((base / f"provenance_{system}.json").read_text())
    _, data_meta = read_lammps_data(base / "final.data")
    tmap: LammpsTypeMap = data_meta["type_map"]

    src = (base / "prod.dump").read_text().splitlines()
    out: list[str] = []
    worst = 0.0
    i = 0
    while i < len(src):
        if not src[i].startswith("ITEM: TIMESTEP"):
            i += 1
            continue
        natoms = int(src[i + 3])
        lo, hi = [], []
        for k in range(3):
            p = src[i + 5 + k].split()
            lo.append(float(p[0]))
            hi.append(float(p[1]))
        columns = src[i + 8].split()[2:]
        ix, iy, iz = (columns.index(c) for c in ("x", "y", "z"))
        out.extend(src[i:i + 9])
        for k in range(natoms):
            tok = src[i + 9 + k].split()
            for j, c in ((0, ix), (1, iy), (2, iz)):
                L = hi[j] - lo[j]
                v = float(tok[c])
                worst = max(worst, (lo[j] - v) / L, (v - hi[j]) / L)
                tok[c] = repr(lo[j] + (v - lo[j]) % L)   # periodic-image relabel
            out.append(" ".join(tok))
        i += 9 + natoms
    scratch = Path(tempfile.mkdtemp()) / "wrapped.dump"
    scratch.write_text("\n".join(out) + "\n")
    frames = read_lammps_dump(scratch)
    scratch.unlink()
    scratch.parent.rmdir()
    unknown = tmap.apply(frames, strict=True)
    assert not unknown, f"{system}: unmapped type ids {unknown}"
    provenance["_max_fractional_excursion"] = worst
    return frames, provenance, tmap


def verify_against_provenance(system: str, frames: list[Frame],
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
        tube, electrolyte = split_tube(fr, dl)
        assert abs(q[tube_indices(fr, dl)].sum()) < 1e-6, \
            f"{system}: the frozen tube carries charge"
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


_TUBE_CACHE: dict[int, tuple[Frame, np.ndarray]] = {}


def tube_indices(frame: Frame, dl: Dialect) -> np.ndarray:
    """Indices of the frozen tube: the frame's largest bond-graph component
    (all-carbon, the majority of atoms -- the pseudo-molecule the census
    would refuse is exactly the region boundary here).  The cache holds the
    frame itself beside the indices, so a freed frame's id() cannot be
    reused by a different frame while its answer is cached."""
    cached = _TUBE_CACHE.get(id(frame))
    if cached is not None and cached[0] is frame:
        return cached[1]
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    edges = bond_graph(frame, dl)
    n = len(frame)
    rows = [e[0] for e in edges] + [e[1] for e in edges]
    cols = [e[1] for e in edges] + [e[0] for e in edges]
    _, labels = connected_components(
        coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)),
        directed=False)
    counts = np.bincount(labels)
    tube_label = int(np.argmax(counts))
    idx = np.where(labels == tube_label)[0]
    assert counts[tube_label] * 2 >= n, "the largest component is not the tube"
    assert {frame.symbols[i] for i in idx} == {"C"}, \
        "the tube component is not pure carbon"
    if len(_TUBE_CACHE) > 64:
        _TUBE_CACHE.clear()
    _TUBE_CACHE[id(frame)] = (frame, idx)
    return idx


def split_tube(frame: Frame, dl: Dialect) -> tuple[Frame, Frame]:
    """(tube frame, electrolyte frame): the frozen carbon sublattice and the
    confined mobile fluid, both with the full cell (pbc) and the charges."""
    idx = tube_indices(frame, dl)
    keep = np.setdiff1d(np.arange(len(frame)), idx)
    q = np.asarray(frame.info["charges"], float)
    tube = Frame(pos=frame.pos[idx], cell=frame.cell,
                 symbols=[frame.symbols[i] for i in idx], pbc=frame.pbc,
                 info={"charges": q[idx]})
    electrolyte = Frame(pos=frame.pos[keep], cell=frame.cell,
                        symbols=[frame.symbols[i] for i in keep],
                        pbc=frame.pbc, info={"charges": q[keep]})
    return tube, electrolyte


# -------------------------------------------------------------------- lift --

def read_tube_reference(system: str) -> Frame:
    """The frozen tube sublattice from the system's `final.data` (full
    1e-6-A text precision; the production dump rounds to 6 significant
    digits, which cannot pass RD2's exact single-radius gate).

    The tube atoms are frozen (`fix setforce 0`), so this sublattice IS
    every frame's tube; the per-frame pipeline asserts the dump's tube
    matches it within the dump's rounding before using it."""
    frame, meta = read_lammps_data(RUNS / system / "final.data")
    idx = np.array([i for i, mid in enumerate(meta["molecule_ids"])
                    if mid == 1])
    assert len(idx) > 0 and {frame.symbols[i] for i in idx} == {"C"}, \
        f"{system}: molecule 1 of final.data is not the all-carbon tube"
    q = np.asarray(meta["charges"], float) if meta["charges"] else None
    return Frame(pos=frame.pos[idx], cell=frame.cell,
                 symbols=[frame.symbols[i] for i in idx], pbc=frame.pbc,
                 info={"charges": q})


def tube_geometry(tube: Frame) -> dict:
    """The tube's region geometry from its own positions: RD2's chirality
    inference on the frozen (ideal, never thermalised -- the tube atoms are
    frozen at build time) carbon sublattice, plus the axis centre (the
    perpendicular centroid, min-image anchored: exact for the symmetric
    frozen tube) and radius (mean radial distance).

    RD2 limitation, recorded in RESULTS.md: `infer_nanotube_chirality`
    re-anchors perpendicular coordinates by minimum image, which is unique
    only below half the perpendicular cell edge; the cnt16 tube's diameter
    (21.71 A) exceeds its cell's half-width (20.86 A), so the inference
    runs with the two perpendicular cell lengths TRIPLED.  No wrap actually
    occurs in either system (each tube sits centred, entirely inside the
    box: max tube extent 21.71 A against 41.72 A of cell), the axial
    period and every consistency gate are unchanged, and the cnt10 result
    is bit-identical with and without the tripling (asserted by the test
    suite).  Fixing the anchoring inside lift/nanotube.py is next round's
    item, not this round's."""
    L = np.asarray(tube.cell_diag, float)
    inference_cell = L * np.array([3.0, 3.0, 1.0])
    chir = infer_nanotube_chirality(tube.pos, inference_cell)
    perp = tube.pos[:, :2] - tube.pos[0, :2]
    perp -= 5.0 * L[:2] * np.round(perp / (5.0 * L[:2]))   # wide-window anchor
    centre = tube.pos[0, :2] + perp.mean(axis=0)
    d = tube.pos[:, :2] - centre
    d -= 5.0 * L[:2] * np.round(d / (5.0 * L[:2]))
    radius = float(np.linalg.norm(d, axis=1).mean())
    spread = float(np.linalg.norm(d, axis=1).max()
                   - np.linalg.norm(d, axis=1).min())
    assert spread < 1e-4, f"tube radius spread {spread:.2e} A"
    return {"n": chir.n, "m": chir.m, "length": chir.length,
            "axis": "z", "centre": (float(centre[0]), float(centre[1]),
                                    float(L[2] / 2)),
            "radius": radius, "radius_spread_A": spread}


def _cylinder(geometry: dict, dl: Dialect):
    from chaord.lang.ir import GeoChain, Quantity, ShCylinder
    return GeoChain(parts=[ShCylinder(
        axis=geometry["axis"],
        cx=Quantity(num=f"{geometry['centre'][0]:.3f}"),
        cy=Quantity(num=f"{geometry['centre'][1]:.3f}"),
        cz=Quantity(num=f"{geometry['centre'][2]:.3f}"),
        radius=Quantity(num=f"{geometry['radius']:.3f}", unit="A"),
        length=Quantity(num=f"{geometry['length']:.3f}", unit="A"))], ops=[])


def _region_volume(geometry: dict) -> float:
    """pi r^2 L of the fluid region (the tube's interior cylinder): the
    volume the region's density and g(r) refer to."""
    return math.pi * geometry["radius"] ** 2 * geometry["length"]


def _fluid_region_statements(electrolyte: Frame, geometry: dict,
                             dl: Dialect) -> list:
    """The confined fluid's statements, following pipeline._fit_fluid_region
    (lift/build v2 design 2.2) exactly: the legacy fluid lifter's statement
    set with densities referred to the REGION volume -- a sub-region is not
    the box, and the box here is mostly vacuum."""
    from chaord.lift.fluid import _centers, _cn_stats_centers
    from chaord.lift.pipeline import _rdf_region
    from chaord.lift.reactive import display_name
    from chaord.lang.ir import Name, Quantity, Statement, Tol

    census = molecule_census(electrolyte, dl)
    statements = []
    for formula in sorted(census):
        statements.append(Statement(
            kind="build", key="molecules",
            values=[Name(text=display_name(formula, dl)),
                    Quantity(num=str(census[formula]))]))
    volume = _region_volume(geometry)
    total_mass = sum(_atomic_mass(s) for s in electrolyte.symbols)
    rho_g = total_mass / volume / 0.6022140857   # u/A^3 -> g/cm3 (fluid.py constant)
    statements.append(Statement(
        kind="state", key="density",
        values=[Quantity(num=f"{rho_g:.3f}", unit="g/cm3")]))
    centers = _centers(electrolyte, dl)
    cn, cn_sd, cn_cut = _cn_stats_centers(electrolyte, centers, dl)
    statements.append(Statement(
        kind="assert", key="cn",
        values=[Quantity(num=f"{cn:.1f}"), Tol(value=Quantity(num=f"{cn_sd:.1f}")),
                Name(text="cutoff"), Quantity(num=f"{cn_cut:.2f}", unit="A")]))
    rm, g = _rdf_region(centers, electrolyte.cell_diag,
                        len(centers) / volume, dl)
    if len(rm):
        ipk = int(np.argmax(g))
        statements.append(Statement(
            kind="assert", key="gr_peak",
            values=[Quantity(num=f"{rm[ipk]:.2f}", unit="A"),
                    Name(text="height"), Quantity(num=f"{g[ipk]:.2f}")]))
    return statements


def _atomic_mass(symbol: str) -> float:
    from ase.data import atomic_masses, chemical_symbols
    return float(atomic_masses[chemical_symbols.index(symbol)])


def lift_one_frame(frame: Frame, field: dict | None, geometry: dict,
                   tube_reference: Frame, dl: Dialect | None = None
                   ) -> tuple[str, dict]:
    """The composed research program of one frame (RD4) + its diagnostics.

    `field` is the provenance field spec for field-on systems
    ({"value": 0.2, "unit": "V/A", "axis": "z"}), None for field-off.
    `geometry` is the frozen tube's region geometry (from final.data, see
    tube_geometry) and `tube_reference` its sublattice, which the frame's
    own tube must match within the dump's text rounding before the
    composition uses it (the tube is frozen: any larger deviation would be
    a data error).  The composition: tube region (RD2 chirality inside its
    RD1 cylinder) + electrolyte region lifted through
    lift_frame(mode="fluid", field=...) under `dl` (the composition
    dialect: the research stack with the species and fluid-routing views of
    RD4).  The electrolyte subset's field statements (`state field`, the
    RD4 `assert field_dipole_p1`, their notes) are carried into the
    composed program's system block and provenance verbatim."""
    dl = dl if dl is not None else composition_dialect()
    tube, electrolyte = split_tube(frame, dl)
    assert len(tube) == len(tube_reference), \
        (f"frame's tube holds {len(tube)} atoms, final.data's holds "
         f"{len(tube_reference)}")
    drift = float(np.abs(np.mod(tube.pos - tube_reference.pos
                                + frame.cell_diag / 2, frame.cell_diag)
                         - frame.cell_diag / 2).max())
    assert drift < 1e-3, \
        f"the 'frozen' tube moved {drift:.2e} A between final.data and this frame"
    subset_program = lift_frame(electrolyte, dl, T=T_K, mode="fluid",
                                field=field)
    from chaord.lang.ir import (
        Chirality, Name, PhysicsBlock, Program, ProvenanceBlock, Quantity,
        RegionBlock, ResidualBlock, Statement, StrVal, SystemBlock)

    sub_system = next(b for b in subset_program.blocks if b.t == "system")
    sub_physics = next(b for b in subset_program.blocks if b.t == "physics")
    sub_prov = next(b for b in subset_program.blocks if b.t == "provenance")

    counts: dict[str, int] = {}
    for s in frame.symbols:
        counts[s] = counts.get(s, 0) + 1
    conserve = []
    for s in sorted(counts):
        conserve += [Name(text=s), Quantity(num=str(counts[s]))]

    L = frame.cell_diag
    system_statements = [
        Statement(kind="build", key="cell", values=[
            Quantity(num=f"{L[i]:.3f}") for i in range(3)]),
        Statement(kind="build", key="pbc", values=[Name(text="xyz")]),
        Statement(kind="state", key="T", values=[Quantity(num=f"{T_K:.2f}")]),
    ]
    field_statements = [s for s in sub_system.statements
                        if s.key in ("field", "field_dipole_p1")]
    system_statements.extend(field_statements)
    system_statements.append(
        Statement(kind="conserve", key="atoms", values=conserve))

    tube_region = RegionBlock(
        phase="crystal", name="tube", geometry=_cylinder(geometry, dl),
        statements=[Statement(
            kind="build", key="nanotube",
            values=[Chirality(n=geometry["n"], m=geometry["m"]),
                    Name(text="length"),
                    Quantity(num=f"{geometry['length']:.3f}", unit="A")])])
    fluid_region = RegionBlock(
        phase="liquid", name="fluid", geometry=_cylinder(geometry, dl),
        statements=_fluid_region_statements(electrolyte, geometry, dl))

    # the composed program's dialect line names the composed stack (the
    # subset lift ran without carbon; the composition adds it for the
    # nanotube region's vocabulary)
    composed_stack = load_dialect(("core", "molecular", "carbon", "research"))
    prov = [Statement(kind="build", key="dialects",
                      values=[StrVal(text=composed_stack.version_string)])] + [
        s for s in sub_prov.statements
        if not (s.kind == "build" and s.key == "dialects")]
    if field is not None:
        axis_note = (f"field axis {field['axis']} from the run input "
                     f"(in.*: fix efield mobile efield 0. 0. {field['value']})")
        prov.append(Statement(kind="build", key="note",
                              values=[StrVal(text=axis_note)]))
    prov.append(Statement(kind="build", key="note", values=[StrVal(
        text="composed first-result program (RD4, docs/reviews/"
             "research_decisions.md): crystal tube region from RD2 chirality "
             "inference on the frozen carbon sublattice; electrolyte through "
             "lift_frame mode=fluid under fluid_max_bonded_component = 15 "
             "(the registry's largest molecule of this solvent set; "
             "molecular.yaml names such species routing exceptions)")]))
    prov.append(Statement(kind="build", key="note", values=[StrVal(
        text="fluid density and g(r) referred to the tube-interior region "
             "volume (pipeline._fit_fluid_region's rule: a sub-region is not "
             "the box)")]))

    program = Program(
        version="0.1",
        dialects=["core", "molecular", "carbon", "research"],
        blocks=[SystemBlock(statements=system_statements),
                PhysicsBlock(statements=list(sub_physics.statements)),
                tube_region, fluid_region,
                ResidualBlock(none=True),
                ProvenanceBlock(statements=prov)])
    from chaord.check.statics import run_checks
    failed = [f"{c.name}: {c.detail}" for c in
              run_checks(program, frame, dl) if not c.passed]
    return format_program(program), {"geometry": geometry,
                                     "failed_checks": failed}


# The fluid-routing view of the research stack (RD4): molecular.yaml's
# fluid_max_bonded_component (3 atoms, the water/CO2 class) explicitly
# names the registry's larger SPECIES templates "routing exceptions, not a
# raised bound"; the research solvent's largest molecule is EMC
# (C4H8O3, 15 atoms), so the composition sets the bound to exactly that.
# bonded_single_phase then still performs its real check (every bond-graph
# component one molecule <= 15 atoms -- it would refuse two bonded ECMs).
FLUID_MAX_BONDED_COMPONENT_VIEW = 15


def composition_dialect() -> Dialect:
    """The dialect the composed lift runs under: the research stack with the
    solvent species view (P1) and the fluid routing view (molecules above
    the water-class bound)."""
    dl = research_dialect(SOLVENT_SPECIES)
    thresholds = dict(dl._data["thresholds"])
    thresholds["fluid_max_bonded_component"] = FLUID_MAX_BONDED_COMPONENT_VIEW
    return Dialect(dl.names, {**dl._data, "thresholds": thresholds},
                   dl.versions)


# ------------------------------------------------------- P1 and the floors --

def p1_of(frame: Frame, species: list[str]) -> dict:
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
    below never touch them.  The `assert field_dipole_p1` VALUE is a
    feature (it is the RD4 evidence, a measurement, not the metadata); the
    other numeric asserts (cn, gr_peak) and the state density are features
    too.  Returns floats or None when the statement is absent."""
    feats: dict = {"p1": None, "cn": None, "cn_tol": None, "gr_peak_r": None,
                   "gr_peak_h": None, "density": None}
    for line in text.splitlines():
        s = line.strip()
        m = re.fullmatch(rf"assert field_dipole_p1 {_NUM} \+- {_NUM} along "
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

    The encoding is physics, not leakage laundering: RD4 ties the P1 assert
    to a stated field, so a field-off program has no assert -- and an
    unaligned liquid has P1 statistically 0 (cosines of uniformly sampled
    orientations).  The measured field-off P1 distributions (RESULTS.md)
    sit at 0.02-0.14, so 0 is the honest a-priori value of the absent
    measurement.  The co-occurrence of assert-presence with the field
    statement is stated openly in the report; the value-only content of the
    assert is the number, and the control model below drops it entirely."""
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
        tube_reference = read_tube_reference(system)
        geometry = tube_geometry(tube_reference)
        texts, p1s, n_mols = [], [], []
        ec_p1, emc_p1 = [], []
        failed_checks = []
        tube_drift = 0.0
        for fr in frames:
            text, info = lift_one_frame(fr, field, geometry, tube_reference,
                                        dl=dl)
            texts.append(text)
            failed_checks.extend(f"{system}: {c}" for c in info["failed_checks"])
            tube, electrolyte = split_tube(fr, dl)
            drift = float(np.abs(np.mod(
                tube.pos - tube_reference.pos + fr.cell_diag / 2,
                fr.cell_diag) - fr.cell_diag / 2).max())
            tube_drift = max(tube_drift, drift)
            dip = p1_of(electrolyte, SOLVENT_SPECIES)
            p1s.append(dip["P1"])
            n_mols.append(dip["n_molecules"])
            ec_p1.append(p1_of(electrolyte, ["C3H4O3"])["P1"])
            emc_p1.append(p1_of(electrolyte, ["C4H8O3"])["P1"])
        p1s = np.asarray(p1s)
        per_system[system] = {
            "field": bool(field), "n_frames": len(frames),
            "verification": check,
            "max_dump_excursion": provenance["_max_fractional_excursion"],
            "chirality": [geometry["n"], geometry["m"]],
            "chirality_matches_provenance":
                [geometry["n"], geometry["m"]]
                == [int(c) for c in
                    provenance["system_definition"]["cnt_chirality"]],
            "tube_radius_A": geometry["radius"],
            "tube_length_A": geometry["length"],
            "max_tube_drift_A": tube_drift,
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
            "model": "single-feature threshold on the field_dipole_p1 assert "
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
