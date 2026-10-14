"""RD3 (docs/reviews/research_decisions.md): an external-field statement.

Red-first suite: written against the decision record before the code. The
statement is state metadata (`state field E 0.10 V/A along z`); the lift
prints it only when the caller supplies the value, flagged as assumed the
same way fluid.AssumedT flags an unstated temperature (W11)."""
import numpy as np
import pytest

from chaord.dialects import load_dialect
from chaord.io.frames import Frame
from chaord.lang.fmt import format_program
from chaord.lang.ir import ir_equal
from chaord.lang.parser import parse_text
from chaord.lift import lift_frame

NL = chr(10)


# ------------------------------------------------------------ parse / format --

def test_field_statement_parses_and_round_trips():
    text = (f"chaord 0.1{NL}dialect core + carbon{NL}{NL}"
            f"system {{{NL}  pbc xyz{NL}  state T 300 K{NL}"
            f"  state field E 0.10 V/A along z{NL}}}{NL}")
    program = parse_text(text, units=load_dialect(("core", "carbon")).units)
    stmt = program.blocks[0].statements[-1]
    assert stmt.kind == "state"
    assert stmt.key == "field"
    vals = stmt.values
    assert vals[0].t == "n" and vals[0].text == "E"
    assert vals[1].t == "q" and vals[1].num == "0.10" and vals[1].unit == "V/A"
    assert vals[2].t == "n" and vals[2].text == "along"
    assert vals[3].t == "n" and vals[3].text == "z"
    assert format_program(program) == text
    assert ir_equal(parse_text(format_program(program),
                               units=load_dialect(("core", "carbon")).units),
                    program)


def test_field_statement_idempotent_without_unit_word():
    text = (f"chaord 0.1{NL}{NL}system {{{NL}"
            f"  state field E 0.5 along x{NL}}}{NL}")
    program = parse_text(text)
    assert format_program(program) == text
    assert ir_equal(parse_text(format_program(program)), program)


def test_field_key_declared_in_carbon_dialect():
    assert load_dialect(("core", "carbon")).key_allowed("system", "field")


# ---------------------------------------------------------------------- lift --

def _gas_frame(seed=0):
    """A sparse noble-gas frame: the cheapest honest lift target. Neon: its
    covalent radius keeps the bond graph empty under these draws, so the
    fluid arm stays atomic (argon's larger radius calls the closest pair a
    molecule and routes the lift into the bonded path, which core+lj does
    not serve)."""
    rng = np.random.default_rng(seed)
    n = 32
    pos = rng.uniform(0.0, 25.0, size=(n, 3))
    return Frame(pos=pos, cell=np.diag([25.0, 25.0, 25.0]),
                 symbols=["Ne"] * n, pbc=(True, True, True))


def test_lift_prints_caller_supplied_field_as_assumed():
    program = lift_frame(_gas_frame(), load_dialect(("core", "lj")),
                         mode="fluid",
                         field={"component": "E", "value": 0.10,
                                "unit": "V/A", "axis": "z"})
    text = format_program(program)
    assert "state field E 0.1 V/A along z" in text
    prov = next(b for b in program.blocks if b.t == "provenance")
    notes = [s.values[0].text for s in prov.statements
             if s.key == "note" and s.values]
    assert any("field assumed" in t and "not measured" in t for t in notes), notes


def test_lift_uses_field_defaults_for_component_and_unit():
    program = lift_frame(_gas_frame(), load_dialect(("core", "lj")),
                         mode="fluid", field={"value": 0.10, "axis": "z"})
    text = format_program(program)
    assert "state field E 0.1 V/A along z" in text


def test_lift_without_field_prints_nothing():
    program = lift_frame(_gas_frame(), load_dialect(("core", "lj")),
                         mode="fluid")
    text = format_program(program)
    assert "field" not in text


def test_lift_refuses_unknown_field_axis():
    with pytest.raises(Exception):
        lift_frame(_gas_frame(), load_dialect(("core", "lj")), mode="fluid",
                   field={"value": 0.10, "axis": "q"})


# --------------------------------------------------------------------- build --

def test_field_statement_is_build_metadata():
    """A program carrying a field statement builds unchanged (RD2 program)."""
    from chaord.build import build_program
    text = (f"chaord 0.1{NL}dialect core + carbon{NL}{NL}"
            f"system {{{NL}  cell 33.564 33.564 39.36 A{NL}  pbc xyz{NL}"
            f"  state field E 0.10 V/A along z{NL}"
            f"  conserve atoms C 640{NL}}}{NL}{NL}"
            f"physics {{{NL}  backend mlp{NL}}}{NL}{NL}"
            f"crystal tube : cylinder axis z center 0. 0. 19.68 "
            f"radius 6.782 A length 39.36 A {{{NL}"
            f"  nanotube (10,10) length 39.36 A{NL}}}{NL}{NL}"
            f"residual none{NL}")
    program = parse_text(text, units=load_dialect(("core", "carbon")).units)
    frame = build_program(program, load_dialect(("core", "carbon")),
                          rng=np.random.default_rng(0), physics=False)
    assert len(frame) == 640
