"""Unit tests for the tutor demo's pure parts (the MD acts are covered by a
slow end-to-end run; the harness itself is the same acceptance code path)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
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


@pytest.mark.slow
def test_tutor_demo_end_to_end():
    """The three acts, really run: ordered byte round trip, liquid statistical
    round trip inside the floor, physics-off and 0.8x-temperature rebuilds
    both exceeding their gates. Exit code 0 exactly when all hold (~5 min)."""
    import subprocess
    r = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "tutor_demo.py")],
        capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "ALL THREE ACTS AS REQUIRED" in r.stdout
