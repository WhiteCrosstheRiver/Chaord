"""A6/A12 -- RED: conservation and charge checkers have two measured blind
spots.

1. Slab/interface programs: region arithmetic is never audited.  Take the
   lifted cu_water program and change its region statement
   'molecules H2O 136' -> 'molecules H2O 106' (30 waters, 18% of the water,
   removed from the region's own accounting) while the conserve line keeps
   the true frame counts: derive_counts returns None ('non-trivial region
   geometry') and the A6 row passes counts_ok=derivation_ok=True.  A census
   bug in the multi-species interface lift is invisible to A6's three-way
   claim.

2. Multivalent ions: every charge table in play is monovalent-only.
   _ion_charge('Ca2+') = +1 (the trailing sign run; the '2' is read as
   stoichiometry by _ion_composition -> the molecule Ca2), and the
   acceptance's ION_CHARGES has no Ca at all.  Demonstrated false pass:
   program 'molecules Ca2+ 5 Cl- 5, conserve charge 0' with a frame of 5
   bonded Ca2 pairs + 5 Cl -> charge_check passes narrating
   'total charge +0 (Ca2+ 5 x +1, Cl- 5 x -1)'; conservation passes; the
   acceptance row passes (-5 == -5); the physical charge is +5.
"""
import pytest
import numpy as np

from chaord.check.statics import charge_check, conservation_check, _ion_charge
from chaord.dialects import load_dialect
from chaord.io.frames import read_frame, Frame
from chaord.lang.ir import (
    GeoChain, Name, PhysicsBlock, Program as IRProgram, Quantity, RegionBlock,
    ShAll, SpecDef, SpeciesBlock, Statement, SystemBlock,
)
from chaord.lift import lift_frame
from tools import acceptance as acc


# ---------------------------------------------------------------- 1. slab ----
def test_a6_slab_region_arithmetic_is_audited():
    case = acc.case_by_id("interfaces/cu_water")
    dl = load_dialect(("core", "metal"))
    frame = read_frame(case["frames"][0])
    text = acc.format_program_text(lift_frame(frame, dl))
    import re
    m = re.search(r"(?m)^(\s*molecules\s+)(\S+\s+\d+)(.*)$", text)
    assert m, "expected a molecules line in the cu_water lift"
    name, n = m.group(2).split()[0], int(m.group(2).split()[1])
    doctored = text[:m.start(2)] + f"{name} {n - 30}" + text[m.end(2):]
    row = acc._conservation_row("cu_water", 0, frame.symbols, doctored)
    assert not (row["counts_ok"] and row["derivation_ok"]), (
        f"A6 RED: a cu_water program whose region statement lost 30 of 136 "
        f"water molecules passes the conservation row ({row['derivation']}); "
        "the three-way claim degrades to two-way for slab geometries"
    )


# --------------------------------------------------------------- 2. charge ----
@pytest.mark.parametrize("name,want", [("Ca2+", 2), ("Mg2+", 2), ("Zn2+", 2),
                                       ("Fe3+", 3), ("O2-", -2), ("SO42-", -2)])
def test_a12_multivalent_ion_charge(name, want):
    got = _ion_charge(name)
    assert got == want, (
        f"A12/A6 RED: the species name {name} carries formal charge {got:+d} "
        f"in the charge bookkeeping; standard notation says {want:+d} "
        "(the digits are parsed as stoichiometry and only the trailing sign "
        "run is counted)"
    )


def _ca_program():
    return IRProgram(
        version="0.1", dialects=["core", "molecular"],
        blocks=[
            SpeciesBlock(defs=[SpecDef(k="ion", name="Ca2+"),
                               SpecDef(k="ion", name="Cl-")]),
            SystemBlock(statements=[
                Statement(kind="build", key="pbc", values=[Name(text="xyz")]),
                Statement(kind="conserve", key="charge",
                          values=[Quantity(num="0")]),
                Statement(kind="conserve", key="atoms",
                          values=[Name(text="Ca"), Quantity(num="10"),
                                  Name(text="Cl"), Quantity(num="5")])]),
            PhysicsBlock(statements=[Statement(kind="build", key="backend",
                                               values=[Name(text="classical")])]),
            RegionBlock(phase="liquid", name="e",
                        geometry=GeoChain(parts=[ShAll()], ops=[]),
                        statements=[Statement(kind="build", key="molecules",
                            values=[Name(text="Ca2+"), Quantity(num="5"),
                                    Name(text="Cl-"), Quantity(num="5")])]),
        ])


def _ca_frame():
    pos = []
    for i in range(5):                    # 5 bonded Ca-Ca pairs (2.5 A)
        c = np.array([8.0 * i + 4.0, 4.0, 4.0])
        pos += [c, c + [2.5, 0, 0]]
    for i in range(5):                    # 5 lone chlorides
        pos.append([8.0 * i + 4.0, 20.0, 4.0])
    return Frame(pos=np.array(pos), cell=np.diag([45.0] * 3),
                 symbols=["Ca"] * 10 + ["Cl"] * 5, pbc=(True,) * 3)


def test_a12_charge_balanced_but_physically_wrong_is_caught():
    mol = load_dialect(("core", "molecular"))
    cc = charge_check(_ca_program(), _ca_frame(), mol)
    cv = conservation_check(_ca_program(), _ca_frame())
    assert not (cc.passed and cv.passed), (
        f"A12/A6 RED: 5 x Ca2+ (+2) + 5 x Cl- (-1) = +5, but the program "
        f"'conserve charge 0' passes charge_check ({cc.detail}) and "
        f"conservation ({cv.detail[:60]})"
    )


def test_a06_acceptance_charge_arithmetic_covers_multivalent_elements():
    for el in ("Ca", "Mg", "Fe", "O"):
        assert el in acc.ION_CHARGES or any(
            el in k for k in acc.ION_CHARGES), (
            f"A6 RED: element {el} is absent from the verifier's ION_CHARGES "
            "table, so any frame containing it passes the acceptance charge "
            "check unaudited (both sides of the comparison read 0 for it)"
        )
