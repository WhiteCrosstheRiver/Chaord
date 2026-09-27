"""Property-based tests: fmt is idempotent and parse(print(ir)) == ir.

A program generator builds IR directly from safe vocabulary (never reserved
words, never unit words used as names), prints it, parses it back and compares.
The PR-cadence test runs 200 examples; the slow (nightly) one runs 10,000.
"""
from pathlib import Path

import pytest
from hypothesis import given, settings, strategies as st

from chaord.lang.fmt import format_program
from chaord.lang.ir import (
    At, Arrow, Direction, Eq, Family, GeoChain, InterfaceBlock, KVDefect, Name,
    Plane, Plus, Program, ProvenanceBlock, Quantity, RangeVal, RegionBlock,
    ResidualBlock, ShAll, ShBox, ShCylinder, ShRest, ShSlab, ShSphere, SpecDef,
    SpeciesBlock, Statement, StrVal, SystemBlock, PhysicsBlock, Tol, Wood,
    ir_equal,
)
from chaord.lang.parser import parse_text
from chaord.dialects import load_dialect

UNITS = sorted(load_dialect(("core",)).units)
RESERVED = {
    "chaord", "dialect", "system", "physics", "provenance", "species",
    "residual", "interface", "none", "state", "constrain", "assert", "history",
    "conserve", "crystal", "amorphous", "liquid", "gas", "fluid", "cluster",
    "vacuum", "all", "rest", "slab", "box", "sphere", "cylinder", "and", "or",
    "minus", "center", "radius", "axis", "molecule", "ion", "atom", "smiles",
    "file", "x", "y", "z",
} | set(UNITS)

SAFE_NAMES = ["alpha", "beta", "gamma", "kappa", "omega", "Ni3Al", "rutile",
              "matrix", "water", "gap", "film", "core", "shell", "solute",
              "region", "B", "D", "sub", "top", "mid"]

KEYS = ["seed", "cell", "pbc", "a", "c", "lattice", "prototype", "backend",
        "epsilon", "sigma", "cutoff", "defect", "molecules", "coverage",
        "termination", "note", "source", "count", "depth"]

PHASES = ["crystal", "amorphous", "liquid", "gas", "fluid", "cluster", "vacuum"]

NUMBERS = ["0", "7", "42", "3.572", "0.5", "27.50", "+0.9", "-0.2", "1/3",
           "2/5", "1e12", "2.5e-3"]

KVS = ["V_Ni", "Al_Ni", "V_O^..", "Ni_i", "V_Ni^''", "V_C"]


st_names = st.sampled_from(SAFE_NAMES)
st_keys = st.sampled_from(KEYS)
st_kinds = st.sampled_from(["build", "state", "constrain", "assert", "history", "conserve"])
st_nums = st.sampled_from(NUMBERS)
st_units = st.sampled_from(UNITS)
st_strings = st.sampled_from(["abc", "NiAl.eam.alloy", "mace-mp-0", 'q"z', "a\\b"])
st_kvs = st.sampled_from(KVS)
st_comments = st.sampled_from(["a note", "another one", "0.32 fixed", "Kröger–Vink"])

st_quantity = st.builds(lambda n, u: Quantity(num=n, unit=u), st_nums,
                        st.one_of(st.none(), st_units))
st_range = st.builds(lambda lo, hi: RangeVal(lo=lo, hi=hi), st_quantity, st_quantity)
st_tol = st.builds(lambda q: Tol(value=q), st_quantity)

st_value = st.one_of(
    st_quantity,
    st_range,
    st_tol,
    st.builds(Name, text=st_names),
    st.builds(StrVal, text=st_strings),
    st.builds(Direction, text=st.sampled_from(["[001]", "[1-10]", "[2 1 0]"])),
    st.builds(Family, text=st.sampled_from(["<110>", "<100>", "<1 1 1>"])),
    st.builds(Plane, text=st.sampled_from(["(110)", "(1-11)", "(0 0 1)"])),
    st.builds(Wood, text=st.sampled_from(["p(2x1)", "c(4x2)", "(r3xr3)R30"])),
    st.builds(KVDefect, text=st_kvs),
    st.just(Arrow()), st.just(At()), st.just(Plus()), st.just(Eq()),
)

st_statement = st.builds(
    lambda kind, key, values, comment: Statement(
        kind=kind, key=key, values=values, comment=comment),
    st_kinds, st.one_of(st_keys, st_names),
    st.lists(st_value, max_size=6),
    st.one_of(st.none(), st_comments),
)

@st.composite
def st_specdef(draw):
    k = draw(st.sampled_from(["molecule", "ion", "atom"]))
    name = draw(st_names)
    if draw(st.booleans()):
        source = draw(st.sampled_from(["smiles", "file"]))
        ref = draw(st_strings)
    else:
        source = ref = None
    return SpecDef(k=k, name=name, source=source, ref=ref)

st_shape = st.one_of(
    st.just(ShAll()),
    st.just(ShRest()),
    st.builds(lambda rng: ShSlab(axis="z", rng=rng), st_range),
    st.builds(lambda xs, ys, zs: ShBox(xs=xs, ys=ys, zs=zs), st_range, st_range, st_range),
    st.builds(lambda cx, cy, cz, r: ShSphere(
                  cx=Quantity(num=cx), cy=Quantity(num=cy), cz=Quantity(num=cz), radius=r),
              st_nums, st_nums, st_nums, st_quantity),
    st.builds(lambda cx, cy, r: ShCylinder(
                  axis="x", cx=Quantity(num=cx), cy=Quantity(num=cy), radius=r),
              st_nums, st_nums, st_quantity),
)
@st.composite
def st_geometry(draw):
    parts = draw(st.lists(st_shape, min_size=1, max_size=3))
    if len(parts) > 1:
        ops = draw(st.lists(st.sampled_from(["and", "or", "minus"]),
                            min_size=len(parts) - 1, max_size=len(parts) - 1))
    else:
        ops = []
    return GeoChain(parts=parts, ops=ops)

st_block = st.one_of(
    st.builds(SystemBlock, statements=st.lists(st_statement, max_size=4)),
    st.builds(PhysicsBlock, statements=st.lists(st_statement, max_size=4)),
    st.builds(ProvenanceBlock, statements=st.lists(st_statement, max_size=4)),
    st.builds(SpeciesBlock, defs=st.lists(st_specdef(), max_size=3)),
    st.builds(RegionBlock, phase=st.sampled_from(PHASES), name=st_names,
              geometry=st_geometry(), statements=st.lists(st_statement, max_size=4)),
    st.builds(InterfaceBlock, a=st_names, b=st_names,
              statements=st.lists(st_statement, max_size=3)),
    st.builds(lambda none, stmts: ResidualBlock(
        none=none, statements=[] if none else stmts),
        st.booleans(), st.lists(st_statement, max_size=3)),
)

st_program = st.builds(
    lambda dialects, blocks: Program(version="0.1", dialects=dialects, blocks=blocks),
    st.lists(st.sampled_from(["core", "lj", "metal"]), max_size=3),
    st.lists(st_block, max_size=6),
)


def _laws(program):
    text = format_program(program)
    reparsed = parse_text(text)
    assert ir_equal(reparsed, program), text
    text2 = format_program(reparsed)
    assert text2 == text


@given(st_program)
@settings(max_examples=200, deadline=None)
def test_fmt_laws(program):
    _laws(program)


@pytest.mark.slow
@given(st_program)
@settings(max_examples=10000, deadline=None)
def test_fmt_laws_ten_thousand(program):
    _laws(program)
