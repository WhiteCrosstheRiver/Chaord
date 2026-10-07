"""Unit tests for the tutor demo's pure parts (the MD acts are covered by a
slow end-to-end run; the harness itself is the same acceptance code path)."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from tutor_demo import set_temperature, verdict_line  # noqa: E402


def test_set_temperature_rewrites_the_state_line():
    prog = "chaord 0.1\n\nsystem {\n  state T 0.72\n}\n"
    out = set_temperature(prog, 0.576)
    assert "state T 0.576" in out
    assert "0.72" not in out


def test_set_temperature_requires_a_state_line():
    with pytest.raises(ValueError):
        set_temperature("chaord 0.1\n\nsystem {\n  pbc xyz\n}\n", 0.5)


def test_verdict_line_flags_mismatches():
    line_ok = verdict_line("physics off", True, True, "numbers")
    assert "as required" in line_ok and "UNEXPECTED" not in line_ok
    line_bad = verdict_line("T = 0.576", False, True, "numbers")
    assert "UNEXPECTED" in line_bad


# ---- act 0's translation invariance (v1 O6 / W12 item 1) ---------------------

def test_shifted_copy_wraps_into_the_cell_and_keeps_the_frame():
    """The helper: a translation by t plus wrap, nothing else moved."""
    from chaord.io.frames import Frame
    from tutor_demo import ACT0_SHIFT, shifted_copy
    frame = Frame(pos=np.array([[0.1, 0.2, 0.3], [7.1, 7.0, 7.15]]),
                  cell=np.diag([7.23] * 3), symbols=["Cu", "Cu"],
                  pbc=(True,) * 3)
    g = shifted_copy(frame, ACT0_SHIFT)
    assert g.symbols == frame.symbols and g.pbc == frame.pbc
    assert np.allclose(g.cell_diag, frame.cell_diag)
    assert len(g) == len(frame)
    assert ((g.pos >= 0) & (g.pos < 7.23)).all(), "shifted copy must wrap"
    d = g.pos - frame.pos
    d -= 7.23 * np.round(d / 7.23)   # minimum image
    assert np.allclose(d, np.asarray(ACT0_SHIFT)), \
        "the copy differs from the original by exactly the shift"


def test_act0_shifted_cu_frame_lifts_to_the_same_text():
    """v1 O6 (W12 item 1): act 0 must demonstrate that a rigidly TRANSLATED
    copy of the stored thermal fcc Cu frame lifts to byte-identical program
    text -- the program describes the macrostate, not one box origin. Before
    W12 the demo had no shifted copy at all (AttributeError), so the claim
    was never shown live."""
    from chaord.dialects import load_dialect
    from chaord.io.frames import read_frame
    from chaord.lift import lift_frame
    from tools import acceptance as acc
    from tutor_demo import shifted_copy

    dl = load_dialect(("core", "metal"))
    cu = read_frame(ROOT / "bench" / "data" / "crystals" / "fcc_cu" /
                    "frame_0.npz")
    t_ref = acc.format_program_text(lift_frame(cu, dl))
    t_shift = acc.format_program_text(lift_frame(shifted_copy(cu), dl))
    assert t_shift == t_ref, (
        "act 0's shifted Cu copy lifts to different text: the lift follows "
        "the box origin")


@pytest.mark.slow
def test_tutor_demo_end_to_end():
    """The three acts, really run: ordered byte round trip INCLUDING the
    shifted-copy translation invariance, liquid statistical round trip inside
    the floor, physics-off and 0.8x-temperature rebuilds both exceeding their
    gates. Exit code 0 exactly when all hold (~5 min)."""
    import subprocess
    r = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "tutor_demo.py")],
        capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "ALL THREE ACTS AS REQUIRED" in r.stdout
    assert "shifted copy" in r.stdout and "as required" in r.stdout.split(
        "shifted copy")[1].splitlines()[0], (
        "act 0 must show the translated copy lifting to the same text")
