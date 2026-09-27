"""WP21 acceptance: Chaord-Bench generation and ground-truth recovery.

The bench generator (bench/generate.py) is invoked into a tmp directory; every
case must land on disk with five independent frames, and the lift/census
machinery must recover the planted ground truth from the stored frames:
prototype + lattice parameter for a sample of the (thermally displaced)
crystal cases through the noise-robust defect path, vacancy/antisite counts
for the defect cases, an exact molecular census for every fluid and reactive
frame, the stated density for the glass, the planted two-phase labels for the
interface case (>= 95% bulk accuracy, interface band excluded) and the
adsorption geometry for the surface case.
"""
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from chaord.build.defects import typical_neighbor_distance
from chaord.build.molecules import molecule_census
from chaord.dialects import load_dialect
from chaord.io.frames import read_frame
from chaord.lang.fmt import format_program
from chaord.lift import lift_frame
from chaord.lift.segment import label_accuracy, phase_labels

ROOT = Path(__file__).parent.parent

EXPECTED_CASES = {
    "crystals/fcc_cu", "crystals/bcc_fe", "crystals/rocksalt_nacl",
    "crystals/l12_ni3al", "crystals/hcp_mg", "crystals/diamond_si",
    "crystals/perovskite_srtio3", "crystals/fcc_crconi",
    "defects/fcc_cu_vacancies", "defects/l12_ni3al_vac_antisite",
    "defects/rocksalt_nacl_vna",
    "fluid/water_box15", "fluid/ar_gas_box25", "fluid/n2_box22",
    "reactive/water_oh_h_box20",
    "glass/lj_glass_rho085",
    "interface/lj_solid_liquid",
    "surface/ni111_o_top",
}
FRAMES_PER_CASE = 5

# crystal cases lifted in the acceptance check. The frames carry a thermal
# displacement of thermal_test_amplitude x d_NN, so the exact spglib path is
# out (it needs sub-symprec positions) and the check goes through the
# noise-robust Wigner-Seitz fitter. Unary bcc and diamond degenerate to their
# exact half-subset lattices (sc / fcc) under that fitter, so the sample takes
# the species-resolved hosts plus the two fcc-based solutions.
CRYSTAL_LIFT_SAMPLE = [
    "crystals/fcc_cu",
    "crystals/rocksalt_nacl",
    "crystals/l12_ni3al",
    "crystals/perovskite_srtio3",
    "crystals/fcc_crconi",
]
# noise floor of the fitted lattice constant under the thermal displacement
# (observed |delta a| stays below 0.04 A on these hosts)
A_TOLERANCE = 0.05


@pytest.fixture(scope="module")
def bench_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("bench")
    r = subprocess.run(
        [sys.executable, str(ROOT / "bench" / "generate.py"), "--out", str(out)],
        capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    return out


@pytest.fixture(scope="module")
def cases(bench_dir):
    """id -> {'frames': [Path], 'gt': dict} for every case in the manifest."""
    manifest = json.loads((bench_dir / "index.json").read_text(encoding="utf-8"))
    out = {}
    for entry in manifest["cases"]:
        gt = json.loads((bench_dir / entry["ground_truth"]).read_text(encoding="utf-8"))
        out[entry["id"]] = {"frames": [bench_dir / f for f in entry["frames"]],
                            "gt": gt}
    return out


def _region_stmts(program):
    region = next(b for b in program.blocks if b.t == "region")
    return {s.key: s for s in region.statements if s.kind == "build"}


def _system_cell(program):
    system = next(b for b in program.blocks if b.t == "system")
    cell = next(s for s in system.statements if s.key == "cell")
    return [float(v.num) for v in cell.values]


def _counts(symbols) -> dict:
    c = Counter(symbols)
    return {k: int(v) for k, v in sorted(c.items())}


# ------------------------------------------------------------------ presence --

def test_manifest_lists_all_cases(bench_dir, cases):
    manifest = json.loads((bench_dir / "index.json").read_text(encoding="utf-8"))
    assert manifest["n_cases"] == len(cases) == len(EXPECTED_CASES)
    assert manifest["n_cases"] >= 15, "PLAN WP21 asks for a wider bench"
    assert set(cases) == EXPECTED_CASES


def test_every_case_has_five_independent_frames(cases):
    for case_id, entry in cases.items():
        gt = entry["gt"]
        assert gt["n_frames"] == FRAMES_PER_CASE, case_id
        assert len(entry["frames"]) == FRAMES_PER_CASE, case_id
        assert len(gt["frames"]) == FRAMES_PER_CASE, case_id
        seeds = [rec["seed"] for rec in gt["frames"]]
        assert len(set(seeds)) == FRAMES_PER_CASE, f"{case_id}: frames share a seed"
        for k, (path, rec) in enumerate(zip(entry["frames"], gt["frames"])):
            assert path.name == f"frame_{k}.npz" and path.is_file(), case_id
            frame = read_frame(path)
            assert len(frame) == rec["n_atoms"], f"{case_id} frame {k}"
            assert _counts(frame.symbols) == rec["counts"], f"{case_id} frame {k}"
        # per-frame invariants must agree across the whole macrostate
        assert all(rec["n_atoms"] == gt["expected"]["n_atoms"]
                   for rec in gt["frames"]), case_id
        assert all(rec["counts"] == gt["expected"]["counts"]
                   for rec in gt["frames"]), case_id
        # the cell is a stated (frame-independent) property of the case
        if "cell" in gt["expected"]:
            cell = np.asarray(gt["expected"]["cell"])
            for path in entry["frames"]:
                with np.load(path) as z:
                    assert np.allclose(z["L"], cell, atol=1e-9), case_id


# ------------------------------------------------------------------ crystals --

@pytest.mark.parametrize("frame_index", [0, FRAMES_PER_CASE - 1])
@pytest.mark.parametrize("case_id", CRYSTAL_LIFT_SAMPLE,
                         ids=lambda c: c.split("/")[-1])
def test_crystal_prototype_and_a_recovered(case_id, frame_index, cases):
    dialect = load_dialect(("core", "metal"))
    entry = cases[case_id]
    gt = entry["gt"]["expected"]
    frame = read_frame(entry["frames"][frame_index])

    program = lift_frame(frame, dialect, mode="defects")
    stmts = _region_stmts(program)

    proto = stmts.get("prototype") or stmts.get("lattice")
    assert proto is not None, "no prototype/lattice statement in lifted program"
    assert proto.values[0].text == gt["prototype"], \
        f"{case_id} frame {frame_index}: lifted {proto.values[0].text}, " \
        f"expected {gt['prototype']}"
    if gt["composition"]:
        assert stmts["composition"].values[0].text == gt["composition"]
    if "occupancy" in gt:
        vals = stmts["occupancy"].values
        lifted = {vals[i].text: vals[i + 1].num for i in range(0, len(vals), 2)}
        assert lifted == gt["occupancy"], \
            f"{case_id}: lifted occupancy {lifted}, expected {gt['occupancy']}"

    a = float(stmts["a"].values[0].num)
    assert abs(a - gt["a"]) <= A_TOLERANCE, \
        f"{case_id} frame {frame_index}: lifted a={a}, expected {gt['a']}"

    # the stated supercell must be an integer tiling of the conventional cell
    cell = _system_cell(program)
    reps = gt["reps"]
    for i in range(3):
        assert abs(cell[i] - gt["a"] * reps[i]) <= 2e-3


# ------------------------------------------------------------------- defects --

@pytest.mark.parametrize("frame_index", [0, FRAMES_PER_CASE - 1])
@pytest.mark.parametrize("case_id", sorted(c for c in EXPECTED_CASES
                                           if c.startswith("defects/")),
                         ids=lambda c: c.split("/")[-1])
def test_defect_cases_recover_defects(case_id, frame_index, cases):
    dialect = load_dialect(("core", "metal"))
    entry = cases[case_id]
    gt = entry["gt"]["expected"]
    frame = read_frame(entry["frames"][frame_index])

    text = format_program(lift_frame(frame, dialect, mode="defects"))
    found = {tok: int(n) for tok, n in
             re.findall(r"defect (\S+) count (\d+)", text)}
    assert found == gt["defects"], \
        f"{case_id} frame {frame_index}: lifted defects {found}, " \
        f"expected {gt['defects']}"
    # never drop an atom: the conserve line must state the frame as built
    line = next(l for l in text.splitlines() if l.strip().startswith("conserve atoms"))
    toks = line.split()[2:]          # after 'conserve' 'atoms'
    stated = {toks[i]: int(toks[i + 1]) for i in range(0, len(toks), 2)}
    assert stated == gt["counts"], \
        f"{case_id} frame {frame_index}: conserve {stated} != {gt['counts']}"


# --------------------------------------------------------------------- fluid --

@pytest.mark.parametrize("case_id", sorted(c for c in EXPECTED_CASES
                                           if c.startswith(("fluid/", "reactive/"))),
                         ids=lambda c: c.split("/")[-1])
def test_fluid_and_reactive_census_exact_every_frame(case_id, cases):
    dialect = load_dialect(("core", "molecular"))
    entry = cases[case_id]
    gt = entry["gt"]
    for k, path in enumerate(entry["frames"]):
        frame = read_frame(path)
        census = molecule_census(frame, dialect)
        assert census == gt["frames"][k]["census"] == gt["expected"]["census"], \
            f"{case_id} frame {k}: census {census} != ground truth"
        # every planted molecule is one connected component: the census
        # accounts for all of them and for every atom in the frame
        assert sum(census.values()) == sum(gt["molecules"].values()), \
            f"{case_id} frame {k}: census counts {sum(census.values())} components"
        assert _counts(frame.symbols) == gt["frames"][k]["counts"]


# --------------------------------------------------------------------- glass --

def test_glass_density_and_counts_every_frame(cases):
    entry = cases["glass/lj_glass_rho085"]
    gt = entry["gt"]["expected"]
    for k, path in enumerate(entry["frames"]):
        frame = read_frame(path)
        rho = len(frame) / float(np.prod(frame.cell_diag))
        assert rho == pytest.approx(gt["density"], abs=1e-9), \
            f"frame {k}: density {rho} != {gt['density']}"
        assert _counts(frame.symbols) == gt["counts"] == {gt["species"]: gt["n_atoms"]}


# ----------------------------------------------------------------- interface --

@pytest.mark.parametrize("frame_index", range(FRAMES_PER_CASE))
def test_interface_segmentation_accuracy(frame_index, cases):
    """Planted solid/liquid labels recovered outside the interface band."""
    dialect = load_dialect(("core", "lj"))
    entry = cases["interface/lj_solid_liquid"]
    gt = entry["gt"]["expected"]
    frame = read_frame(entry["frames"][frame_index])
    L = frame.cell_diag

    truth = frame.pos[:, 2] > gt["boundary_z"]   # True = liquid (upper half)
    labels = phase_labels(frame, dialect)
    # atoms within 2 x d_NN of the planted boundary belong to the interface
    # object, not to either phase: the criterion is evaluated outside that
    # zone, and away from the periodic wrap at z = 0 / z = Lz
    d_nn = typical_neighbor_distance(frame)
    z = frame.pos[:, 2]
    core = ((np.abs(z - gt["boundary_z"]) > 2.0 * d_nn)
            & (z > 2.0 * d_nn) & (z < L[2] - 2.0 * d_nn))
    assert core.sum() >= 50, "interface band exclusion left too few core atoms"
    acc = label_accuracy(labels[core], (~truth)[core])
    assert acc >= 0.95, f"frame {frame_index}: bulk label accuracy {acc:.3f}"


# ------------------------------------------------------------------- surface --

def test_surface_geometry_every_frame(cases):
    from scipy.spatial import cKDTree
    dialect = load_dialect(("core", "metal", "surface"))
    entry = cases["surface/ni111_o_top"]
    gt = entry["gt"]["expected"]
    height = float(dialect.threshold("adsorbate_height"))
    vacuum = float(dialect.threshold("surface_default_vacuum"))
    for k, path in enumerate(entry["frames"]):
        frame = read_frame(path)
        L = frame.cell_diag
        ni = np.array([p for s, p in zip(frame.symbols, frame.pos) if s == "Ni"])
        ox = np.array([p for s, p in zip(frame.symbols, frame.pos) if s == "O"])
        assert len(ni) == gt["counts"]["Ni"]
        assert len(ox) == gt["adsorb"]["count"] == entry["gt"]["frames"][k]["adsorb_o_count"]
        top_z = ni[:, 2].max()
        assert L[2] - top_z >= vacuum - 0.1, "vacuum gap missing"
        assert np.allclose(ox[:, 2] - top_z, height, atol=1e-6)
        dist, _ = cKDTree(ni[ni[:, 2] > top_z - 0.5][:, :2]).query(ox[:, :2])
        assert dist.max() < 0.1  # directly above surface atoms (top sites)


# ------------------------------------------------------------------ reproducible --

def test_generation_is_deterministic(bench_dir, tmp_path):
    r = subprocess.run(
        [sys.executable, str(ROOT / "bench" / "generate.py"), "--out", str(tmp_path)],
        capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    manifest = json.loads((bench_dir / "index.json").read_text(encoding="utf-8"))
    for entry in manifest["cases"]:
        for rel in entry["frames"]:
            with np.load(bench_dir / rel) as a, np.load(tmp_path / rel) as b:
                assert np.array_equal(a["r"], b["r"]), (entry["id"], rel)
                assert np.array_equal(a["L"], b["L"]), (entry["id"], rel)
                assert np.array_equal(a["symbols"], b["symbols"]), (entry["id"], rel)
