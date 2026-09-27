"""Shortest-program controller (greedy deletion search) and `roundtrip --samples`.

Covers the two deliverables:
- ``shortest_program`` with a ``reference_later`` frame runs the real search:
  categories are deleted one at a time, each deletion is rebuilt and kept only
  inside `factor` x the measured noise floor, otherwise rolled back;
- the old signature (no reference) is the conservative mode and must stay
  equivalent to the previous three-variant controller (full text, nothing
  deleted, conservation intact);
- ``chaord roundtrip --samples N`` rebuilds N times with independent seeds and
  reports per-sample observable distances plus mean +- std vs the noise floor.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from chaord.build.fluid import build_fluid
from chaord.check.shortest import shortest_program
from chaord.dialects import load_dialect
from chaord.io.frames import Frame
from chaord.lang.fmt import format_program
from chaord.lang.parser import parse_text
from chaord.lift import lift_frame

ROOT = Path(__file__).parent.parent

LIQUID_TEXT = """chaord 0.1
dialect core + lj

system {
  cell 7.000 7.000 7.000
  pbc xyz
  conserve atoms X 200
}

physics {
  backend lj
}

liquid fluid : all {
  state density 0.583
}
"""


@pytest.fixture(scope="module")
def dialect():
    return load_dialect(("core", "lj"))


def _liquid_frame(dialect, seed):
    """One microstate of the liquid macrostate (packed, physics off)."""
    prog = parse_text(LIQUID_TEXT)
    return build_fluid(prog, dialect, np.random.default_rng(seed), physics=False)


def _build_fn(dialect):
    return lambda p, r: build_fluid(p, dialect, r, physics=False)


def test_shortest_search_with_reference(dialect):
    """With a reference frame the search really deletes and shortens."""
    frame = _liquid_frame(dialect, 11)
    program = lift_frame(frame, dialect)
    later = _liquid_frame(dialect, 99)  # independent frame of the same macrostate

    name, text, removals = shortest_program(
        frame, program, dialect, _build_fn(dialect),
        rng_seed=5, reference_later=later)

    full = format_program(program)
    assert len(text) < len(full)           # the search shortened the program
    assert "conserve atoms X 200" in text  # conservation never deleted
    assert removals                        # the deletion list is reported
    assert "assert" not in text            # asserts are the first category tried


def test_shortest_rolls_back_outside_tolerance(dialect):
    """A deletion that leaves the noise floor is rolled back."""
    frame = _liquid_frame(dialect, 11)
    program = lift_frame(frame, dialect)
    later = _liquid_frame(dialect, 99)

    name, text, removals = shortest_program(
        frame, program, dialect, _build_fn(dialect),
        rng_seed=5, reference_later=later,
        factor=0.0)  # impossible tolerance: nothing may be certified

    assert (name, text, removals) == ("full", format_program(program), [])


def test_shortest_conservative_mode_is_old_behaviour(dialect):
    """Old signature (no reference): conservative mode, old result."""
    rng = np.random.default_rng(7)
    frame = Frame(pos=rng.uniform(0, 7, (200, 3)), cell=np.diag([7.0] * 3),
                  symbols=["X"] * 200)
    program = lift_frame(frame, dialect)
    name, text, removals = shortest_program(
        frame, program, dialect, _build_fn(dialect))
    assert name == "full"                   # old acceptance: exact statement set only
    assert text == format_program(program)  # nothing deleted without a floor
    assert removals == []
    assert "conserve atoms X 200" in text


def test_roundtrip_samples_cli():
    r = subprocess.run(
        [sys.executable, "-m", "chaord.cli", "roundtrip",
         str(ROOT / "prototype" / "snap.npz"), "--samples", "2", "--no-physics"],
        capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "sample" in r.stdout            # per-sample lines exist
    assert "mean" in r.stdout              # the mean +- std / floor line exists
