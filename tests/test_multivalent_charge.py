"""F5 (red-team reports/redteam_findings.md, 2026-09-30): multivalent-ion
charges were outside every charge table.

`_ion_charge('Ca2+')` read only the trailing sign run (+1) because the digits
before the sign were parsed as stoichiometry (`_ion_composition('Ca2+')` was
the molecule 'Ca2', two Ca atoms). Both directions were wrong:

* a physically CORRECT CaCl2 program (5 Ca2+ + 10 Cl-, conserve charge 0)
  FAILED the charge check (5 x +1 + 10 x -1 = -5) and the conservation check
  (the molecules line implied 10 Ca);
* a physically ABSURD one (5 Ca2+ + 5 Cl-, conserve charge 0; the physical
  charge is +5) PASSED both.

The tests below were red before the fix (verified 2026-09-30). Notation rule
they pin: in an ion name, the digit run immediately before the trailing sign
run is the charge magnitude (Ca2+ = +2, SO42- = sulfate with -2), and the
charge-stripped rest is the composition (SO42- -> S + 4 O)."""
import numpy as np
import pytest

from chaord.check.statics import (
    _ion_charge, _ion_composition, charge_check, conservation_check,
)
from chaord.dialects import load_dialect
from chaord.io.frames import Frame
from chaord.lang.parser import parse_text

IONIC = ("core", "ionic")


@pytest.mark.parametrize("name,want", [
    ("Ca2+", 2), ("Mg2+", 2), ("Zn2+", 2), ("Fe3+", 3),
    ("O2-", -2), ("SO42-", -2), ("Al3+", 3), ("Ti4+", 4),
    # monovalent and template names keep their reading (regression pins)
    ("Li+", 1), ("Na+", 1), ("Cl-", -1), ("PF6-", -1), ("H2O", 0),
])
def test_charge_magnitude_digits_are_the_valence(name, want):
    got = _ion_charge(name)
    assert got == want, (
        f"F5: species name {name} carries formal charge {got:+d} in the "
        f"charge bookkeeping; standard ion notation says {want:+d}")


@pytest.mark.parametrize("name,want", [
    ("Ca2+", ("Ca",)), ("O2-", ("O",)), ("Fe3+", ("Fe",)),
    ("Hg22+", ("Hg", "Hg")),            # mercurous dimer: 2 Hg, total +2
    ("SO42-", ("S", "O", "O", "O", "O")),
    ("H2O", ("O", "H", "H")),            # template: no sign, digits stay
                                         # stoichiometry (template symbol order)
    ("PF6-", ("P",) + ("F",) * 6),       # untemplated polyatomic: digits are
                                         # stoichiometry, charge -1 (the
                                         # monatomic rule does not claim it)
])
def test_charge_digits_are_not_stoichiometry(name, want):
    got = _ion_composition(name)
    assert got == want, (
        f"F5: the element multiset behind {name} parsed as {got}; expected "
        f"{want} (digits before the sign run are the charge magnitude)")


def _solution_frame(pairs, box=33.0):
    """Atoms on a 3 A-spaced grid: every pair far outside any contact."""
    pos, symbols = [], []
    n = 0
    for sym, count in pairs:
        for _ in range(count):
            pos.append([(n % 6) * 3.0 + 4.0, ((n // 6) % 6) * 3.0 + 4.0,
                        (n // 36) * 3.0 + 4.0])
            symbols.append(sym)
            n += 1
    return Frame(pos=np.array(pos), cell=np.diag([box] * 3), symbols=symbols,
                 pbc=(True,) * 3)


def _cacl2_program(conserve_atoms="Ca 5 Cl 10", molecules="Ca2+ 5 Cl- 10",
                   conserve_charge="0", species_block=True):
    species = "species {\n  ion Ca2+\n  ion Cl-\n}\n\n" if species_block else ""
    return parse_text(
        "chaord 0.1\ndialect core + ionic\n\n"
        "system {\n  cell 33 33 33\n  pbc xyz\n"
        f"  conserve atoms {conserve_atoms}\n"
        f"  conserve charge {conserve_charge}\n"
        "}\n\n"
        + species +
        "liquid s : all {\n"
        f"  molecules {molecules}\n"
        "}\n\n"
        "residual none\n")


def test_correct_cacl2_solution_passes_the_charge_check():
    """5 x Ca2+ (+2) + 10 x Cl- (-1) = 0 with a matching frame: the honest
    program. Before the fix this FAILED both checks (charge read -5; the
    molecules line implied 10 Ca against the stated 5)."""
    frame = _solution_frame([("Ca", 5), ("Cl", 10)])
    program = _cacl2_program()
    cc = charge_check(program, frame, load_dialect(IONIC))
    assert cc.passed, cc.detail
    assert "Ca2+ 5 x +2" in cc.detail and "Cl- 10 x -1" in cc.detail
    cv = conservation_check(program, frame)
    assert cv.passed, cv.detail


def test_charge_imbalanced_cacl2_is_caught():
    """5 x Ca2+ + 5 x Cl- is +5, not 0: the red team's demonstrated false
    pass. Before the fix charge_check narrated 'total charge +0 (Ca2+ 5 x
    +1, Cl- 5 x -1) equals conserve charge 0' while the physical charge is
    +5. (The atom census is not the witness here -- 5 Ca2+ ions ARE 5 Ca
    atoms, and the conserve line agrees -- so the charge check must be the
    one that refuses.)"""
    frame = _solution_frame([("Ca", 5), ("Cl", 5)])
    program = _cacl2_program(conserve_atoms="Ca 5 Cl 5",
                             molecules="Ca2+ 5 Cl- 5")
    cc = charge_check(program, frame, load_dialect(IONIC))
    cv = conservation_check(program, frame)
    assert not (cc.passed and cv.passed), (
        f"F5: an ionic program whose physical charge is +5 passes both "
        f"checks (charge: {cc.detail}; conservation: {cv.detail})")
    assert not cc.passed
    assert "Ca2+ 5 x +2" in cc.detail
    # the conservation side of the OLD bug is gone too: the molecules line no
    # longer implies a Ca2 dimer census (10 Ca) against the stated 5
    assert cv.passed or "imply" not in cv.detail


def test_multivalent_fallback_without_species_block():
    """No species block: fixed-valent multivalent ions draw their charge from
    the common-valence fallback table shipped with the check (the multivalent
    extension of the ionic dialect's monovalent formal_charges). Before the
    fix Ca was unaudited (charge 0) and the CORRECT CaCl2 program failed with
    'actual -10'."""
    frame = _solution_frame([("Ca", 5), ("Cl", 10)])
    program = _cacl2_program(species_block=False)
    cc = charge_check(program, frame, load_dialect(IONIC))
    assert cc.passed, cc.detail
    assert "Ca2+ 5 x +2" in cc.detail
    # and the same fallback does not bless an unbalanced frame
    frame_bad = _solution_frame([("Ca", 5), ("Cl", 5)])
    program_bad = _cacl2_program(conserve_atoms="Ca 5 Cl 5",
                                 molecules="Ca2+ 5 Cl- 5", species_block=False)
    assert not charge_check(program_bad, frame_bad,
                            load_dialect(IONIC)).passed


def test_variable_valence_stays_program_decided():
    """Fe is deliberately absent from the fallback table (ionic.yaml's design
    note): without a species block an Fe-containing frame is not silently
    assigned Fe2+/Fe3+ -- the charge check reports no source for it rather
    than guessing."""
    from chaord.check.statics import _charge_sources
    sources = _charge_sources(_cacl2_program(species_block=False),
                              load_dialect(IONIC))
    assert not any(info["element"] == "Fe" for info in sources.values()), (
        "F5: variable-valence elements must stay program-decided, not enter "
        "the charge bookkeeping through a fixed table")
