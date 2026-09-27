"""A12 acceptance: charge conservation (species-block ions + conserve charge).

Charge sources, lowest priority first: the ionic dialect's `formal_charges`
table and the program's species-block `ion` definitions (template charge when
the ion is a known molecule, else the trailing sign run in the name: `Li+` +1,
`PF6-` -1). `conserve charge N` compares N with the frame's total; a mismatch
FAILs with expected/actual detail. Without the statement an ionic system still
passes (WARN level: the detail suggests adding it). Neutral programs pass
quietly and skip the molecular census entirely.
"""
import numpy as np
import pytest

from chaord.build.molecules import (
    TEMPLATES, _register_mol, molecule_census, pack_molecules,
)
from chaord.check.statics import charge_check, run_checks
from chaord.dialects import load_dialect
from chaord.io.frames import Frame
from chaord.lang.parser import parse_text


@pytest.fixture(scope="module", autouse=True)
def _pf6_template():
    """PF6- as an exact test template: octahedral, P-F 1.58 A, charge -1.

    Registered here because the core ships templates only for its benchmark
    molecules (molecules.py is out of scope for this work package)."""
    if "PF6-" not in TEMPLATES:
        _register_mol("PF6-", ["P"] + ["F"] * 6,
                      [[0.0, 0.0, 0.0],
                       [1.58, 0.0, 0.0], [-1.58, 0.0, 0.0],
                       [0.0, 1.58, 0.0], [0.0, -1.58, 0.0],
                       [0.0, 0.0, 1.58], [0.0, 0.0, -1.58]],
                      charge=-1)
    return TEMPLATES["PF6-"]


@pytest.fixture(scope="module")
def mol_dialect():
    return load_dialect(("core", "molecular"))


@pytest.fixture(scope="module")
def electrolyte(mol_dialect):
    """45 Li+ + 45 PF6- packed in a 26 A box (seed 11 keeps the Li-Li distances
    outside the bond-graph threshold, so the census sees intact ions)."""
    rng = np.random.default_rng(11)
    frame = pack_molecules({"Li+": 45, "PF6-": 45}, [26.0] * 3, rng, mol_dialect)
    assert len(frame) == 45 + 45 * 7
    # physical sanity of the fixture: ions come out as intact molecular units
    assert molecule_census(frame, mol_dialect) == {"Li": 45, "F6P": 45}
    return frame


def _electrolyte_program(conserve: str = "conserve charge 0"):
    """Hand-written electrolyte macrostate: species block + charge statement."""
    line = f"  {conserve}\n" if conserve is not None else ""
    return parse_text(
        "chaord 0.1\ndialect core + molecular\n\n"
        "system {\n"
        "  cell 26 26 26\n"
        "  pbc xyz\n"
        "  seed 11\n"
        "  state T 300 K\n"
        "  conserve atoms Li 45 P 45 F 270\n"
        + line +
        "}\n\n"
        "physics {\n"
        "  backend mlp\n"
        "  model \"mace-mp-0\"\n"
        "}\n\n"
        "species {\n"
        "  ion Li+ = smiles \"[Li+]\"\n"
        "  ion PF6- = smiles \"F[P-](F)(F)(F)(F)F\"\n"
        "}\n\n"
        "liquid electrolyte : all {\n"
        "  molecules Li+ 45 PF6- 45\n"
        "}\n\n"
        "residual none\n")


def test_neutral_electrolyte_passes(electrolyte, mol_dialect):
    result = charge_check(_electrolyte_program(), electrolyte, mol_dialect)
    assert result.name == "charge"
    assert result.passed, result.detail
    assert "Li+ 45 x +1" in result.detail
    assert "PF6- 45 x -1" in result.detail


def test_wrong_conserve_charge_fails(electrolyte, mol_dialect):
    result = charge_check(_electrolyte_program("conserve charge 1"),
                          electrolyte, mol_dialect)
    assert not result.passed
    assert "expected 1" in result.detail
    assert "actual 0" in result.detail


def test_missing_conserve_charge_only_warns(electrolyte, mol_dialect):
    """No `conserve charge` statement in an ionic system: WARN, not FAIL."""
    result = charge_check(_electrolyte_program(conserve=None),
                          electrolyte, mol_dialect)
    assert result.passed, result.detail
    assert "conserve charge 0" in result.detail   # the suggested statement


def test_neutral_system_passes_quietly(mol_dialect):
    rng = np.random.default_rng(5)
    frame = Frame(pos=rng.uniform(0, 20, (50, 3)), cell=np.diag([20.0] * 3),
                  symbols=["Ar"] * 50)
    program = parse_text(
        "chaord 0.1\ndialect core + molecular\n\n"
        "system {\n  cell 20 20 20\n  pbc xyz\n}\n")
    result = charge_check(program, frame, mol_dialect)
    assert result.passed, result.detail
    assert result.detail == "total charge 0 (no ionic species)"
    assert "conserve charge" not in result.detail


def test_charge_runs_after_conservation_in_run_checks(electrolyte, mol_dialect):
    names = [c.name for c in run_checks(_electrolyte_program(), electrolyte,
                                        mol_dialect)]
    assert "charge" in names
    assert names.index("conservation") < names.index("charge")


def test_formal_charge_table_fallback():
    """No species block: monovalent ions draw their charge from the ionic
    dialect's formal_charges table (K+ and Br- have no shipped template, so
    the names themselves carry the sign)."""
    dialect = load_dialect(("core", "ionic"))
    rng = np.random.default_rng(7)
    frame = Frame(pos=rng.uniform(0, 18, (60, 3)), cell=np.diag([18.0] * 3),
                  symbols=["K"] * 30 + ["Br"] * 30)
    head = ("chaord 0.1\ndialect core + ionic\n\n"
            "system {{\n  cell 18 18 18\n  pbc xyz\n{}}}")
    balanced = charge_check(
        parse_text(head.format("  conserve charge 0\n")), frame, dialect)
    assert balanced.passed, balanced.detail
    assert "K+ 30 x +1" in balanced.detail
    assert "Br- 30 x -1" in balanced.detail
    off = charge_check(parse_text(head.format("  conserve charge -1\n")),
                       frame, dialect)
    assert not off.passed
    assert "expected -1" in off.detail and "actual 0" in off.detail
