"""F1 root-cause tests (red team, reports/redteam_findings.md S1): the
lift/build CELL CONTRACT and rotation stability on the bench's own THERMAL
crystal frames (bench/data/crystals/*/frame_*.npz + solutions/cuau_random),
not on perfect rebuilds.

Pre-fix classification of the 18 thermal frames the finding measured
(9 crystal-lifting cases x frames 0-1; reproduced 2026-09-30):

  * 8 x CELL-MISMATCH -- the lift prints a fitted `a` from one source and
    copies `cell` from the frame box; the builder refuses its own program
    ("system cell [7.23] is not an integer multiple of the fcc lattice
    vectors [3.608]", tolerance 0.01 A): fcc_cu f0/f1, bcc_fe f1,
    rocksalt f0, diamond f1, perovskite f1, crconi f1, cuau f1.
  * 2 x F2 (other class) -- hcp_mg f0/f1 misroute to the fluid lifter
    (prototype recognition of the orthohexagonal setting fails and the
    cascade swallows it; reports/redteam_findings.md F2 owns that chain).
    Category test below is xfail-waiting for the F2 stream.
  * 8 x build OK but lift->build->lift text byte-identical on 1/18 only
    (l12_ni3al f0); every other frame drifts on the `cell` line because the
    builder rebuilds the box as reps x a instead of the stated box.

Rotation: fit_crystal's scanned lattice constant moves with the frame's
orientation (fcc_cu f0: a 3.608 -> 3.575 / 3.495 A under one rigid transform,
-0.9 % / -3.1 %; the finding measured -4.3 %).

The contract under test (one definition, both sides): a program's stated
`cell` IS its build box -- the builder tiles the conventional cell `reps`
times and requires the stated box to be that integer multiple within the
dialect's `lattice_match_tolerance` (A12), so the lift must emit `a` and
`cell` that already satisfy it, invariantly under rigid transforms.

Test-side constants (not pass code): A_DRIFT_TOL is the review target
(fitted a constant to < 0.5 % across rigid transforms).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from chaord.build import build_program                     # noqa: E402
from chaord.dialects import load_dialect                   # noqa: E402
from chaord.io.frames import read_frame                    # noqa: E402
from chaord.lang.errors import ChaordError                 # noqa: E402
from chaord.lang.fmt import format_program                 # noqa: E402
from chaord.lang.parser import parse_text                  # noqa: E402
from chaord.lift import lift_frame                         # noqa: E402
from tools import acceptance as acc                        # noqa: E402

METAL = ("core", "metal")
A_DRIFT_TOL = 0.005          # fitted a must stay within 0.5 % under transforms
N_TRANSFORMS = 2

CRYSTAL_CASES = [
    "crystals/fcc_cu", "crystals/bcc_fe", "crystals/rocksalt_nacl",
    "crystals/l12_ni3al", "crystals/diamond_si",
    "crystals/perovskite_srtio3", "crystals/fcc_crconi",
    "solutions/cuau_random",
]
FRAMES = (0, 1)              # the 18-frame evidence set of the finding

_dl = load_dialect(METAL)


def _lift_text(frame):
    return format_program(lift_frame(frame, _dl))


def _case_frame(cid, k):
    case = acc.case_by_id(cid)
    return read_frame(case["frames"][k])


def _stmt_values(text, key):
    """Numeric values of the first statement line starting with `key`."""
    for line in text.splitlines():
        parts = line.strip().split()
        if parts and parts[0] == key:
            return [float(v) for v in parts[1:4] if v[0].isdigit()]
    return []


def _a_of(text):
    vals = _stmt_values(text, "a")
    assert vals, f"no `a` statement in lifted text:\n{text}"
    return vals[0]


def _cell_of(text):
    vals = _stmt_values(text, "cell")
    assert len(vals) == 3, f"no 3-value `cell` statement in text:\n{text}"
    return np.array(vals)


# ---- category 1: the stated cell is an integer tiling of the stated a =======

@pytest.mark.parametrize("cid", CRYSTAL_CASES)
@pytest.mark.parametrize("fk", FRAMES)
def test_crystal_thermal_cell_is_integer_tiling_of_stated_a(cid, fk):
    """The lift's own program text must satisfy the builder's cell contract:
    for each axis there is an integer rep count with |reps x a - cell| within
    the dialect's lattice_match_tolerance. Pre-fix: 8/16 frames violate it by
    0.012-0.028 A (the builder's refusal message of F1)."""
    tol = float(_dl.threshold("lattice_match_tolerance"))
    text = _lift_text(_case_frame(cid, fk))
    a, cell = _a_of(text), _cell_of(text)
    reps = np.rint(cell / a).astype(int)
    off = np.abs(reps * a - cell)
    assert np.all(off <= tol), (
        f"{cid} frame_{fk}: stated cell {cell.tolist()} is not an integer "
        f"tiling of the stated a {a} (reps {reps.tolist()}, mismatch "
        f"{off.tolist()} > tolerance {tol}) -- the program its own lift "
        f"emits would be refused by its own builder (F1)")


# ---- category 2: lift -> build -> lift on the thermal frames =================

@pytest.mark.parametrize("cid", CRYSTAL_CASES)
@pytest.mark.parametrize("fk", FRAMES)
def test_crystal_thermal_round_trip_builds_and_text_is_stable(cid, fk):
    """lift -> build must succeed on every thermal crystal frame (pre-fix the
    builder refuses 8/16) and lift -> build -> lift -> build -> lift must be
    byte-stable: t2 == t1 (pre-fix 1/18 frames over the finding's set) and
    t3 == t2 (the second generation must already be a fixed point)."""
    frame = _case_frame(cid, fk)
    t1 = _lift_text(frame)
    rebuilt1 = build_program(parse_text(t1), _dl, rng=np.random.default_rng(5))
    t2 = _lift_text(rebuilt1)
    assert t2 == t1, (
        f"{cid} frame_{fk}: lift->build->lift text differs "
        f"(cell {_cell_of(t1).tolist()} -> {_cell_of(t2).tolist()}, "
        f"a {_a_of(t1)} -> {_a_of(t2)})")
    rebuilt2 = build_program(parse_text(t2), _dl, rng=np.random.default_rng(5))
    t3 = _lift_text(rebuilt2)
    assert t3 == t2, (
        f"{cid} frame_{fk}: the round trip is not a fixed point "
        f"(t3 != t2; a {_a_of(t2)} -> {_a_of(t3)})")


# ---- category 3: rotation stability (A2's law extended to thermal frames) ====

@pytest.mark.parametrize("cid", CRYSTAL_CASES)
@pytest.mark.parametrize("fk", FRAMES)
def test_crystal_thermal_rigid_transform_invariant_text_and_a(cid, fk):
    """Rule 3 on the bench's own thermal frames: rotation + translation +
    re-imaging + re-ordering (the verifier's own rigid_transform) must lift to
    byte-identical text, and the fitted lattice constant must not drift by
    more than 0.5 % (pre-fix: fcc_cu f0 drifts -0.9 %/-3.1 %, the finding
    measured -4.3 %; the scanned `a` followed the frame's orientation)."""
    frame = _case_frame(cid, fk)
    t_ref = _lift_text(frame)
    a_ref = _a_of(t_ref)
    rng = np.random.default_rng(acc._stable_seed(f"{cid}|f{fk}"))
    for k in range(N_TRANSFORMS):
        g = acc.rigid_transform(frame, rng)
        t = _lift_text(g)
        a = _a_of(t)
        drift = abs(a - a_ref) / a_ref
        assert t == t_ref, (
            f"{cid} frame_{fk} transform {k}: text differs under a rigid "
            f"transform (a {a_ref} -> {a})")
        assert drift <= A_DRIFT_TOL, (
            f"{cid} frame_{fk} transform {k}: fitted a drifts {drift:.3%} "
            f"({a_ref} -> {a}) under a rigid transform -- the lattice fit is "
            f"orientation-dependent (F1)")


# ---- category 4 (other class): hcp_mg is the F2 misroute, not a cell bug ====

@pytest.mark.xfail(strict=False, reason="F2 (reports/redteam_findings.md): "
                                        "orthohexagonal hcp prototype "
                                        "recognition + cascade misroute; owned "
                                        "by the F2 stream, not the cell "
                                        "contract fixed here")
@pytest.mark.parametrize("fk", FRAMES)
def test_hcp_thermal_frame_lifts_as_crystal_program(fk):
    """The hcp_mg thermal frames are the OTHER failure class of the 18: the
    crystal engine cannot recognise the orthohexagonal setting, the cascade
    swallows the error and the fluid lifter emits semantic garbage
    (`molecules Mg32`) that the builder refuses (`unknown species 'Mg32'`).
    The cell-contract fix must not paper over this: the honest crystal lift
    (mode='crystal') should succeed once F2 lands."""
    frame = _case_frame("crystals/hcp_mg", fk)
    program = lift_frame(frame, _dl, mode="crystal")
    text = format_program(program)
    assert "crystal" in text
    build_program(parse_text(text), _dl, rng=np.random.default_rng(5))
