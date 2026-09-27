"""WP21 acceptance: Chaord-Bench generation and ground-truth recovery.

The bench generator (bench/generate.py) is invoked into a tmp directory; every
case must land on disk, and the lift/census machinery must recover the planted
ground truth from the stored frames: prototype + lattice parameter for the
crystals, vacancy count for the defect case, atom conservation for the fluid,
and an exact molecular census for the reactive mixture.
"""
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from chaord.build.molecules import molecule_census
from chaord.dialects import load_dialect
from chaord.io.frames import read_frame
from chaord.lang.fmt import format_program
from chaord.lift import lift_frame

ROOT = Path(__file__).parent.parent

EXPECTED_CASES = {
    "crystals/fcc_cu", "crystals/bcc_fe", "crystals/rocksalt_nacl",
    "crystals/l12_ni3al", "defects/fcc_cu_vacancies", "fluid/water_box15",
    "reactive/water_oh_h_box20",
}
CRYSTAL_CASES = sorted(c for c in EXPECTED_CASES if c.startswith("crystals/"))


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
    """id -> {'npz': Path, 'gt': dict} for every case in the manifest."""
    manifest = json.loads((bench_dir / "index.json").read_text(encoding="utf-8"))
    out = {}
    for entry in manifest["cases"]:
        gt = json.loads((bench_dir / entry["ground_truth"]).read_text(encoding="utf-8"))
        out[entry["id"]] = {"npz": bench_dir / entry["npz"], "gt": gt}
    return out


def _region_stmts(program):
    region = next(b for b in program.blocks if b.t == "region")
    return {s.key: s for s in region.statements if s.kind == "build"}


def _system_cell(program):
    system = next(b for b in program.blocks if b.t == "system")
    cell = next(s for s in system.statements if s.key == "cell")
    return [float(v.num) for v in cell.values]


# ------------------------------------------------------------------ presence --

def test_manifest_lists_all_cases(cases):
    assert set(cases) == EXPECTED_CASES


def test_all_case_files_exist(bench_dir, cases):
    for case_id, entry in cases.items():
        assert entry["npz"].is_file(), f"{case_id}: missing {entry['npz']}"
    manifest = json.loads((bench_dir / "index.json").read_text(encoding="utf-8"))
    for entry in manifest["cases"]:
        gt = bench_dir / entry["ground_truth"]
        assert gt.is_file(), f"missing {gt}"
        frame = read_frame(bench_dir / entry["npz"])
        gt_data = json.loads(gt.read_text(encoding="utf-8"))
        assert len(frame) == gt_data["expected"]["n_atoms"]
        counts = Counter(frame.symbols)
        assert {k: int(v) for k, v in sorted(counts.items())} == \
            gt_data["expected"]["counts"]


# ------------------------------------------------------------------ crystals --

@pytest.mark.parametrize("case_id", CRYSTAL_CASES, ids=lambda c: c.split("/")[-1])
def test_crystal_prototype_and_a_recovered(case_id, cases):
    dialect = load_dialect(("core", "metal"))
    entry = cases[case_id]
    gt = entry["gt"]["expected"]
    frame = read_frame(entry["npz"])

    program = lift_frame(frame, dialect, mode="crystal")
    stmts = _region_stmts(program)

    proto = stmts.get("prototype") or stmts.get("lattice")
    assert proto is not None, "no prototype/lattice statement in lifted program"
    assert proto.values[0].text == gt["prototype"], \
        f"{case_id}: lifted {proto.values[0].text}, expected {gt['prototype']}"
    if gt["composition"]:
        assert stmts["composition"].values[0].text == gt["composition"]

    a = float(stmts["a"].values[0].num)
    assert abs(a - gt["a"]) <= 1e-3, f"{case_id}: lifted a={a}, expected {gt['a']}"

    # the stated supercell must be an integer tiling of the conventional cell
    cell = _system_cell(program)
    reps = gt["reps"]
    for i in range(3):
        assert abs(cell[i] - gt["a"] * reps[i]) <= 1e-3


# ------------------------------------------------------------------- defects --

def test_defect_case_recovers_vacancies(cases):
    dialect = load_dialect(("core", "metal"))
    entry = cases["defects/fcc_cu_vacancies"]
    gt = entry["gt"]["expected"]
    frame = read_frame(entry["npz"])

    text = format_program(lift_frame(frame, dialect, mode="defects"))
    found = {tok: int(n) for tok, n in
             re.findall(r"defect (\S+) count (\d+)", text)}
    for token, count in gt["defects"].items():
        assert found.get(token) == count, \
            f"lifted defects {found}, expected {gt['defects']}"
    assert sum(1 for s in frame.symbols if s == "Cu") == gt["counts"]["Cu"]


# --------------------------------------------------------------------- fluid --

def test_fluid_case_conserves_atoms(cases):
    dialect = load_dialect(("core", "molecular"))
    entry = cases["fluid/water_box15"]
    gt = entry["gt"]["expected"]
    frame = read_frame(entry["npz"])

    counts = Counter(frame.symbols)
    assert counts["H"] == 120
    assert counts["O"] == 60
    assert {k: int(v) for k, v in sorted(counts.items())} == gt["counts"]
    assert molecule_census(frame, dialect) == gt["census"] == {"H2O": 60}


# ------------------------------------------------------------------ reactive --

def test_reactive_case_census_exact(cases):
    dialect = load_dialect(("core", "molecular"))
    entry = cases["reactive/water_oh_h_box20"]
    gt = entry["gt"]["expected"]
    frame = read_frame(entry["npz"])

    census = molecule_census(frame, dialect)
    assert census == gt["census"], \
        f"census {census} != ground truth {gt['census']}"
    # the census must account for every planted molecule and every atom
    assert sum(census.values()) == sum(gt["molecules"].values()) == 68
    counts = Counter(frame.symbols)
    assert {k: int(v) for k, v in sorted(counts.items())} == gt["counts"]
    assert counts["H"] == 118 and counts["O"] == 59


# ------------------------------------------------------------------ reproducible --

def test_generation_is_deterministic(bench_dir, tmp_path):
    r = subprocess.run(
        [sys.executable, str(ROOT / "bench" / "generate.py"), "--out", str(tmp_path)],
        capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    import numpy as np
    manifest = json.loads((bench_dir / "index.json").read_text(encoding="utf-8"))
    for entry in manifest["cases"]:
        with np.load(bench_dir / entry["npz"]) as a, np.load(tmp_path / entry["npz"]) as b:
            assert np.array_equal(a["r"], b["r"]), entry["id"]
            assert np.array_equal(a["L"], b["L"]), entry["id"]
            assert np.array_equal(a["symbols"], b["symbols"]), entry["id"]
