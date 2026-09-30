"""A2/A3 -- RED: the invariance and round-trip criteria are evidenced on
PERECT rebuilt crystals, never on the bench's own stored THERMAL frames.

tools/acceptance.py check_a2/check_a3 call _rebuild_crystal(case) (a perfect
frame from ground truth) instead of case["frames"].  On the stored frames:

  * rotation invariance (rule 3) breaks: crystals/fcc_cu frame_0 lifts to
    'a 3.608 A'; after one rigid transform of the SAME frame the lift prints
    'a 3.452 A' (-4.3%).  l12_ni3al: 3.572 -> 3.518 (-1.5%).
  * lift -> build fails outright on 10/18 stored frames ("system cell is not
    an integer multiple of the lattice vectors": the lift prints a fitted
    'a' inconsistent with the copied 'cell'), and is byte-identical on only
    1/18, against the reported "9/9 text byte-identical".

Root cause direction: the lattice fit (spglib idealisation of the jittered
frame) and the printed cell come from different sources; the builder's
integer-multiple tolerance (0.01) is tighter than the lift's own rounding.
The hcp_mg misroute is covered in its own file.
"""
import numpy as np
import pytest

from chaord.dialects import load_dialect
from chaord.io.frames import read_frame
from chaord.lang.errors import ChaordError
from chaord.lang.parser import parse_text
from chaord.build import build_program
from chaord.lift import lift_frame
from tools import acceptance as acc

METAL = ("core", "metal")


def _lift_text(cid, frame, names=METAL):
    return acc.format_program_text(lift_frame(frame, load_dialect(names)))


@pytest.mark.parametrize("cid", ["crystals/fcc_cu", "crystals/l12_ni3al"])
def test_a2_thermal_frame_rigid_transform_invariant(cid):
    case = acc.case_by_id(cid)
    frame = read_frame(case["frames"][0])
    t_ref = _lift_text(cid, frame)
    rng = np.random.default_rng(acc._stable_seed(cid))
    a_ref = [l.split()[1] for l in t_ref.splitlines() if l.strip().startswith("a ")]
    for _ in range(2):
        g = acc.rigid_transform(frame, rng)
        t = _lift_text(cid, g)
        a = [l.split()[1] for l in t.splitlines() if l.strip().startswith("a ")]
        assert t == t_ref, (
            f"A2 RED: the stored thermal frame of {cid} lifts to different "
            f"text under rotation+translation+re-imaging+re-ordering "
            f"(a {a_ref} -> {a}); the A2 evidence set uses perfect "
            "_rebuild_crystal frames and never sees this"
        )


def test_a3_thermal_frame_round_trip_builds():
    """lift -> build must at least succeed on the bench's own crystal frames
    (A3's 'exact round trip on every crystal case')."""
    fails = []
    for cid in ("crystals/fcc_cu", "crystals/bcc_fe", "crystals/rocksalt_nacl",
                "crystals/diamond_si"):
        case = acc.case_by_id(cid)
        dl = load_dialect(METAL)
        frame = read_frame(case["frames"][0])
        t1 = _lift_text(cid, frame)
        try:
            build_program(parse_text(t1), dl, rng=np.random.default_rng(5))
        except ChaordError as e:
            fails.append(f"{cid}: {str(e)[:60]}")
    assert not fails, (
        "A3 RED: the lifted text of stored thermal crystal frames is "
        "rejected by the builder itself (the acceptance check round-trips "
        "perfect _rebuild_crystal frames instead): " + "; ".join(fails)
    )


def test_a3_thermal_frame_round_trip_text():
    """Where the build succeeds, lift->build->lift must be byte-identical
    (A3 text requirement) -- measured today: 1/18 stored frames; the lone
    survivor is l12_ni3al frame_0, so use the next one that builds."""
    dl = load_dialect(METAL)
    case = acc.case_by_id("crystals/fcc_crconi")
    frame = read_frame(case["frames"][0])
    t1 = _lift_text(case["id"], frame)
    rebuilt = build_program(parse_text(t1), dl, rng=np.random.default_rng(5))
    t2 = _lift_text(case["id"], rebuilt)
    assert t2 == t1, (
        "A3 RED: lift->build->lift text differs on the stored thermal frame "
        "of fcc_crconi (the acceptance evidence uses perfect rebuilds; "
        "measured today only 1/18 stored crystal frames round-trips "
        "byte-identically)"
    )
