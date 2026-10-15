"""D29 (docs/reviews/open_items_v4.md, work order Y2): the lift is blind.

The dipole-orientation statement `assert dipole_p1 <v> +- <tol> along
<axis>` is measured for EVERY frame whose dialect names dipolar species
(`field_dipole_species`), whether or not the caller passed `field=`;
`field=` adds ONLY the `state field` line and its assumed-value note.  The
key is `dipole_p1` (renamed from `field_dipole_p1`): the observable is a
property of the frame against an analysis axis, not of the caller's
metadata.

Axis resolution (Y2 step 2): the field's axis when one is given; else the
nanotube region's axis when the program holds one; else the dialect's
`dipole_axis_default`.

The slow test lifts all 84 committed production frames of the first-result
pilot with NO field= (the analysis helpers of
research/cnt_electrolyte/analyze_first_result.py, read-only use); every
program must carry the assert.  Power mutation (Y2 step 1): re-gating the
statement on `field` must fail it -- demonstrated in the work order's
report by patching `_attach_dipole_p1` back behind the field gate.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src"), str(ROOT / "research" / "cnt_electrolyte")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from chaord.build.molecules import TEMPLATES               # noqa: E402
from chaord.dialects import load_dialect                   # noqa: E402
from chaord.io.frames import Frame                         # noqa: E402
from chaord.lang.fmt import format_program                 # noqa: E402
from chaord.lift import lift_frame                        # noqa: E402

import analyze_first_result as A                           # noqa: E402

# `assert dipole_p1 <v> +- <tol> along <axis>` -- the statement every blind
# research lift must carry (the value and tolerance are measured; only the
# presence and shape are pinned here).
_P1_LINE = re.compile(r"^  assert dipole_p1 -?\d+\.\d{3} \+- \d+\.\d{3} "
                      r"along [xyz]$")

# SPC/E site charges of the water template, template atom order (the
# research dialect's own table; Berendsen, Grigera & Straatsma 1987).
_H2O_Q = [-0.8476, 0.4238, 0.4238]


def _water_frame(rotations, centres, L=(12.0, 12.0, 18.0)):
    """A frame of registry water molecules, one per (rotation, centre),
    carrying per-atom charges (the LAMMPS reader's convention)."""
    t = TEMPLATES["H2O"]
    rel = np.asarray(t["rel"], float)
    pos = np.vstack([np.asarray(c, float) + rel @ R.T
                     for R, c in zip(rotations, centres)])
    return Frame(pos=pos, cell=np.diag(np.asarray(L, float)),
                 symbols=list(t["symbols"]) * len(rotations),
                 pbc=(True, True, True),
                 info={"charges": np.tile(_H2O_Q, len(rotations))})


def _research():
    return load_dialect(("core", "molecular", "research"))


def _p1_lines(text):
    return [l.strip() for l in text.splitlines() if "dipole_p1" in l]


# ------------------------------------------------------ the 84-frame blind lift --

@pytest.mark.slow
def test_all_84_production_frames_carry_blind_dipole_p1():
    """D29: lifting the committed pilot frames with NO field= still asserts
    the orientation statement -- the lift measures it from the frame, it
    does not wait for the caller to confess a field.

    Y3 (D35) ported this to the composed lift: lift_frame(mode="auto") on
    the FULL dump frame -- no split_tube, no final.data reference -- composes
    the tube and the confined liquid and asserts the blind statement along
    the tube's own axis."""
    dl = A.composition_dialect()
    n_frames = 0
    for system in A.SYSTEMS:
        frames, _, _ = A.read_prod_frames(system)
        for fr in frames:
            text = format_program(lift_frame(fr, dl, T=A.T_K,
                                             mode="auto"))
            lines = _p1_lines(text)
            assert len(lines) == 1, (system, lines)
            assert _P1_LINE.match("  " + lines[0]), (system, lines[0])
            assert lines[0].endswith("along z"), (system, lines[0])
            assert "state field" not in text, (system, "blind lift printed "
                                                "a field statement")
            assert "field" not in text, (system, "blind lift mentions the "
                                         "field:\n" + text)
            n_frames += 1
    assert n_frames == 84, n_frames


# ------------------------------------------------------------- blind == fielded --

def test_blind_and_fielded_differ_only_in_the_field_line_and_its_note():
    """The Y2 done-when on a template frame: the same dipole_p1 line with
    and without field=0; the only extra lines of the fielded text are the
    `state field` statement and its assumed-value note."""
    frame = _water_frame([np.eye(3)] * 3,
                         [[6.0, 6.0, 3.0], [6.0, 6.0, 9.0], [6.0, 6.0, 15.0]])
    dl = _research()
    blind = format_program(lift_frame(frame, dl, T=300.0, mode="fluid"))
    fielded = format_program(lift_frame(
        frame, dl, T=300.0, mode="fluid", field={"value": 0.0, "axis": "z"}))
    assert _p1_lines(blind) == _p1_lines(fielded) and len(_p1_lines(blind)) == 1
    extra = [l for l in fielded.splitlines() if l not in blind.splitlines()]
    assert len(extra) == 2, extra
    assert extra[0].strip() == "state field E 0 V/A along z", extra
    assert "field assumed" in extra[1] and "not measured" in extra[1], extra


def test_no_field_means_no_field_statement_but_the_p1_stays():
    """D29 reversed the old absence rule: without field= there is still no
    `state field` line (there is no honest default field), but the measured
    assert stays."""
    frame = _water_frame([np.eye(3)] * 3,
                         [[6.0, 6.0, 3.0], [6.0, 6.0, 9.0], [6.0, 6.0, 15.0]])
    text = format_program(lift_frame(frame, _research(), T=300.0,
                                     mode="fluid"))
    assert "field" not in text
    assert _P1_LINE.match("  " + _p1_lines(text)[0])


# ------------------------------------------------------------ axis resolution --

def test_field_axis_wins_over_the_dialect_default():
    frame = _water_frame([np.eye(3)] * 3,
                         [[6.0, 6.0, 3.0], [6.0, 6.0, 9.0], [6.0, 6.0, 15.0]])
    dl = _research()
    text = format_program(lift_frame(
        frame, dl, T=300.0, mode="fluid",
        field={"value": 0.1, "axis": "x"}))
    assert _p1_lines(text)[0].endswith("along x")


def test_dialect_default_axis_is_used_without_a_field():
    frame = _water_frame([np.eye(3)] * 3,
                         [[6.0, 6.0, 3.0], [6.0, 6.0, 9.0], [6.0, 6.0, 15.0]])
    text = format_program(lift_frame(frame, _research(), T=300.0,
                                     mode="fluid"))
    assert _p1_lines(text)[0].endswith("along z")   # dipole_axis_default


def test_nanotube_region_axis_beats_the_dialect_default():
    """A program holding a nanotube region resolves the analysis axis from
    it before the dialect default (Y2 step 2; the end-to-end composition is
    Y3's work order, the resolution rule is pinned here)."""
    from chaord.lang.ir import (
        Chirality, GeoChain, Name, PhysicsBlock, Program, ProvenanceBlock,
        Quantity, RegionBlock, ResidualBlock, ShCylinder, Statement, StrVal,
        SystemBlock)
    from chaord.lift import _resolve_dipole_axis

    tube = RegionBlock(
        phase="crystal", name="tube",
        geometry=GeoChain(parts=[ShCylinder(
            axis="y", cx=Quantity(num="10"), cy=Quantity(num="10"),
            cz=Quantity(num="20"), radius=Quantity(num="6.78", unit="A"),
            length=Quantity(num="39.36", unit="A"))], ops=[]),
        statements=[Statement(
            kind="build", key="nanotube",
            values=[Chirality(n=10, m=10), Name(text="length"),
                    Quantity(num="39.36", unit="A")])])
    program = Program(version="0.1", dialects=["core", "carbon", "research"],
                      blocks=[SystemBlock(statements=[]),
                              PhysicsBlock(statements=[]),
                              tube, ResidualBlock(none=True),
                              ProvenanceBlock(statements=[
                                  Statement(kind="build", key="dialects",
                                            values=[StrVal(text="x")])])])
    assert _resolve_dipole_axis(program, _research(), field_axis=None) == \
        ("y", "the nanotube region's axis")
    # the field's axis still wins over the tube's
    assert _resolve_dipole_axis(program, _research(), field_axis="x") == \
        ("x", "the field's axis")


def test_dialect_without_species_is_silent_blind():
    """A dialect that names no dipolar species (core + lj) promises no
    orientation observable: the blind lift adds no assert and no note."""
    rng = np.random.default_rng(0)
    frame = Frame(pos=rng.uniform(0.0, 25.0, size=(32, 3)),
                  cell=np.diag([25.0, 25.0, 25.0]), symbols=["Ne"] * 32,
                  pbc=(True, True, True))
    text = format_program(lift_frame(
        frame, load_dialect(("core", "lj")), mode="fluid"))
    assert "dipole_p1" not in text
    assert "not asserted" not in text
