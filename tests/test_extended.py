"""M6 acceptance: extended defects and reactive census.

Exit criteria (PLAN M6): census exact on planted cases; Burgers vectors and
Sigma values correct on the benchmark constructions.
"""
import numpy as np
import pytest

from chaord.build.extended import build_dislocation, build_grain_boundary
from chaord.build.molecules import molecule_census, pack_molecules
from chaord.dialects import load_dialect
from chaord.io.frames import Frame
from chaord.lang.fmt import fmt_statement
from chaord.lang.ir import (
    GeoChain, Name, Quantity, RegionBlock, ShAll, Statement,
)
from chaord.lift.extended import burgers_vector, grain_boundary_sigma
from chaord.lift.reactive import dissociation_statement


@pytest.fixture(scope="module")
def metal():
    return load_dialect(("core", "metal"))


def _region(*stmts):
    return RegionBlock(phase="crystal", name="bulk",
                       geometry=GeoChain(parts=[ShAll()], ops=[]),
                       statements=list(stmts))


def test_reactive_census_exact():
    d = load_dialect(("core", "molecular"))
    rng = np.random.default_rng(3)
    f = pack_molecules({"H2O": 50, "OH": 9, "H": 9}, [20.0] * 3, rng, d)
    census = molecule_census(f, d)
    assert census["H2O"] == 50
    assert census.get("OH", census.get("HO")) == 9
    assert census["H"] == 9
    # element conservation
    n_h = sum(1 for s in f.symbols if s == "H")
    n_o = sum(1 for s in f.symbols if s == "O")
    assert (n_h, n_o) == (50 * 2 + 9 + 9, 50 + 9)


def test_dissociation_statement():
    d = load_dialect(("core", "molecular"))
    rng = np.random.default_rng(3)
    f = pack_molecules({"H2O": 50, "OH": 9, "H": 9}, [20.0] * 3, rng, d)
    line = fmt_statement(dissociation_statement(f, d))
    assert line == "dissociate H2O -> OH @ surface + H @ surface count 9"


def test_dissociation_zero_without_fragments():
    d = load_dialect(("core", "molecular"))
    rng = np.random.default_rng(3)
    f = pack_molecules({"H2O": 20}, [18.0] * 3, rng, d)
    assert dissociation_statement(f, d) is None


def test_burgers_family(metal):
    frame = build_dislocation(_region(
        Statement(kind="build", key="lattice", values=[Name(text="fcc")]),
        Statement(kind="build", key="a", values=[Quantity(num="3.615")]),
        Statement(kind="build", key="dislocation", values=[Name(text="edge")])),
        {}, metal, np.random.default_rng(0))
    result = burgers_vector(frame, metal)
    assert result is not None
    assert result["family"] == "<110>"
    # unrelaxed construction: magnitude within the family detection band
    assert abs(result["magnitude"] - 3.615 / np.sqrt(2)) < 0.35 * 3.615 / np.sqrt(2)


def test_no_burgers_in_perfect_crystal(metal):
    from chaord.build.crystal import build_conventional
    frame = build_conventional("fcc", {"a": 3.615}, ("Cu",), (6, 6, 4))
    assert burgers_vector(frame, metal) is None


def test_sigma5_detected(metal):
    frame = build_grain_boundary(_region(
        Statement(kind="build", key="lattice", values=[Name(text="fcc")]),
        Statement(kind="build", key="a", values=[Quantity(num="3.615")]),
        Statement(kind="build", key="grain_boundary",
                  values=[Name(text="sigma"), Quantity(num="5")])),
        {}, metal, np.random.default_rng(0))
    assert grain_boundary_sigma(frame, metal) == 5


def test_no_sigma_in_single_crystal(metal):
    from chaord.build.crystal import build_conventional
    frame = build_conventional("fcc", {"a": 3.615}, ("Cu",), (8, 8, 6))
    assert grain_boundary_sigma(frame, metal) is None
