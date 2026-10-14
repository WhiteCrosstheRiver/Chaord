"""W12 (leftovers): EC and PF6- as general SPECIES templates of the core
registry (src/chaord/build/molecules.py TEMPLATES), needed by
solutions/lipf6_ec and later the PC/EMC/EC/PF6/Li electrolyte research case.

Until W12 these two templates existed only as local registrations inside
bench/generate.py (and PF6- a second time as a test fixture in
test_charge.py), so a hand-written or lifted program naming them built only
in processes that had run the bench generator first. The tests pin:

  * both templates exist with the registry's documented keys;
  * exact stoichiometry (EC = C3H4O3 neutral, 10 atoms; PF6- = P + 6 F,
    charge -1, 7 atoms) and the published internal geometry;
  * a census of one placed molecule of each sees ONE intact molecule -- the
    geometry is contact-consistent with the molecular dialect's bond window;
  * a full `molecules EC/Li+/PF6-` program BUILDS and produces exactly the
    right atom counts and census composition.

Numbers and their sources (cited in the registration comments too):
  P-F 1.58 A    crystallographic hexafluorophosphate (LiPF6 structures);
                the value the frozen lipf6_ec frames and the A12 charge
                tests were generated with (bench/generate.py)
  ring C-O 1.43 A, carbonyl C=O 1.20 A, C-H 1.09 A
                standard organic bond lengths (Allen, Kennard, Watson,
                Galloy, Baalham & Brammer, J. Chem. Soc. Perkin Trans. 2
                (1987) S1-S19: ester C-O 1.43, ketone C=O 1.21, sp3 C-H
                1.09); the exact geometry bench/generate.py generated the
                frozen lipf6_ec frames with, promoted verbatim so the
                species template, the bench frames and the census agree.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from chaord.build import build_program                      # noqa: E402
from chaord.build.molecules import (                        # noqa: E402
    TEMPLATES, molecule_census, pack_molecules, species_mass,
)
from chaord.dialects import load_dialect                    # noqa: E402
from chaord.io.frames import Frame                          # noqa: E402
from chaord.lang.parser import parse_text                   # noqa: E402


@pytest.fixture(scope="module")
def mol_dialect():
    return load_dialect(("core", "molecular"))


# ------------------------------------------------------------- existence ----

def test_ec_template_exists_with_documented_keys():
    """W12: EC (ethylene carbonate) is a core species template."""
    assert "EC" in TEMPLATES, (
        "no 'EC' species template in the core registry; known: "
        f"{sorted(TEMPLATES)}")
    t = TEMPLATES["EC"]
    for key in ("symbols", "rel", "charge", "radius", "pack_radius"):
        assert key in t, f"the EC template misses the documented key {key!r}"
    assert t["symbols"] == ["C", "O", "C", "C", "O", "O", "H", "H", "H", "H"]
    assert len(t["rel"]) == 10
    assert int(t["charge"]) == 0


def test_pf6_template_exists_with_documented_keys():
    """W12: PF6- (hexafluorophosphate) is a core species template."""
    assert "PF6-" in TEMPLATES, (
        "no 'PF6-' species template in the core registry; known: "
        f"{sorted(TEMPLATES)}")
    t = TEMPLATES["PF6-"]
    for key in ("symbols", "rel", "charge", "radius", "pack_radius"):
        assert key in t, f"the PF6- template misses the documented key {key!r}"
    assert t["symbols"] == ["P"] + ["F"] * 6
    assert int(t["charge"]) == -1


# ------------------------------------------------------------- geometry -----

def test_pf6_octahedral_geometry():
    """P at the centre, six F at 1.58 A, all F-P-F angles 90/180 deg."""
    rel = np.asarray(TEMPLATES["PF6-"]["rel"], float)
    assert np.allclose(rel.mean(axis=0), 0.0, atol=1e-12)
    d_pf = np.linalg.norm(rel[1:] - rel[0], axis=1)
    assert np.allclose(d_pf, 1.58, atol=1e-6), d_pf
    v = rel[1:] - rel[0]
    cos = (v @ v.T) / np.outer(np.linalg.norm(v, axis=1),
                               np.linalg.norm(v, axis=1))
    off = np.abs(cos[np.triu_indices(6, 1)])
    assert np.all((np.isclose(off, 0.0, atol=1e-9)
                   | np.isclose(off, 1.0, atol=1e-9)),), \
        "octahedral F-P-F angles must be 90 or 180 deg"


def test_ec_published_internal_geometry():
    """EC stoichiometry C3H4O3 and the promoted bench geometry: 4 x C-H
    1.09 A, 1 x carbonyl C=O 1.20 A, 5 x ring edges 1.43 A, plus the two
    geminal H-H pairs at 2 x 1.09 x sin(35 deg) = 1.2504 A of the
    generator's 70-deg H-C-H spread (the simplified ring geometry the frozen
    lipf6_ec frames carry; documented in their ground truth)."""
    t = TEMPLATES["EC"]
    assert len(t["symbols"]) == 10
    from collections import Counter
    assert Counter(t["symbols"]) == {"C": 3, "H": 4, "O": 3}
    rel = np.asarray(t["rel"], float)
    assert np.allclose(rel.mean(axis=0), 0.0, atol=1e-12)
    d = np.linalg.norm(rel[:, None, :] - rel[None, :, :], axis=-1)
    iu = np.triu_indices(10, 1)
    bonds = sorted(round(x, 2) for x in d[iu] if x < 1.45)
    assert bonds == sorted([1.09] * 4 + [1.20] + [1.25] * 2 + [1.43] * 5), \
        bonds
    assert (d[iu] >= 1.09 - 1e-6).all(), \
        "no EC atom pair may sit inside the shortest published bond"
    # and nothing else hides under 1.45 A (every short pair is one of the
    # documented bonds/contacts above)
    assert sum(x < 1.45 for x in d[iu]) == 12


def test_species_mass_of_the_new_templates():
    assert species_mass("EC") == pytest.approx(88.06, abs=0.02)
    assert species_mass("PF6-") == pytest.approx(144.96, abs=0.02)


# ------------------------------------------------- contact consistency ------

def _one_molecule_frame(name):
    """One template molecule at the centre of a box, diluted with Ar atoms on
    an 8 A grid (Ar never bonds: the census sees it as monatomic filler; the
    grid's shortest Ar-Ar distance, 4 A across the periodic edge, sits outside
    the 1.25 x (1.06 + 1.06) = 2.65 A Ar bond window). The dilution keeps the
    molecule from being the frame's majority, which the census's
    extended-component guard refuses by design -- the property under test is
    the geometry, not the majority rule."""
    t = TEMPLATES[name]
    pos = [np.asarray(t["rel"], float) + 18.0]
    syms = list(t["symbols"])
    for x in (2.0, 10.0, 18.0, 26.0, 34.0):
        for y in (2.0, 18.0, 34.0):
            for z in (2.0, 18.0, 34.0):
                if 8.0 < x < 32.0 and y == 18.0 and z == 18.0:
                    continue          # keep the centre line for the molecule
                pos.append([[x, y, z]])
                syms.append("Ar")
    return Frame(pos=np.vstack(pos), cell=np.diag([36.0] * 3), symbols=syms,
                 pbc=(True,) * 3)


@pytest.mark.parametrize("name,formula", [("EC", "C3H4O3"), ("PF6-", "F6P")])
def test_census_sees_one_intact_molecule(name, formula, mol_dialect):
    """The shipped geometry must survive the dialect's own bond window: the
    placed EC/PF6- is censused as exactly one molecule among inert filler
    (contact-consistent internal geometry: every intramolecular bond present,
    no intermolecular bond to the filler)."""
    frame = _one_molecule_frame(name)
    n_ar = frame.symbols.count("Ar")
    assert n_ar == 45 - 3, "5x3x3 grid minus the 3 centre-line points"
    assert molecule_census(frame, mol_dialect) == {formula: 1, "Ar": n_ar}


# ---------------------------------------------------------- build (A13) -----

def test_electrolyte_program_builds_the_right_composition(mol_dialect):
    """A hand-written electrolyte program naming the new species builds and
    conserves: 4 EC + 4 Li+ + 4 PF6- -> 40 + 4 + 28 = 72 atoms, census
    {C3H4O3: 4, Li: 4, F6P: 4} (Li+ is a solvation ion: never bonded)."""
    program = parse_text(
        "chaord 0.1\ndialect core + molecular\n\n"
        "system {\n  cell 32 32 32\n  pbc xyz\n  seed 3\n  state T 300 K\n"
        "  conserve atoms C 12 F 24 H 16 Li 4 O 12 P 4\n"
        "  conserve charge 0\n}\n\n"
        "physics {\n  backend mlp\n  model \"mace-mp-0\"\n}\n\n"
        "species {\n  ion Li+ = smiles \"[Li+]\"\n"
        "  ion PF6- = smiles \"F[P-](F)(F)(F)(F)F\"\n}\n\n"
        "liquid electrolyte : all {\n  molecules EC 4 Li+ 4 PF6- 4\n"
        "}\n\n"
        "residual none\n")
    frame = build_program(program, mol_dialect, rng=np.random.default_rng(3),
                          physics=False)
    assert len(frame) == 4 * 10 + 4 * 1 + 4 * 7
    assert molecule_census(frame, mol_dialect) == \
        {"C3H4O3": 4, "Li": 4, "F6P": 4}


def test_packing_mixed_electrolyte_composition(mol_dialect):
    """The pack primitive itself: EC + PF6- + Li+ pack as intact molecular
    units (the packing radii keep the census exact)."""
    rng = np.random.default_rng(5)
    frame = pack_molecules({"EC": 4, "PF6-": 4, "Li+": 4}, [32.0] * 3, rng,
                           mol_dialect)
    assert len(frame) == 72
    assert molecule_census(frame, mol_dialect) == \
        {"C3H4O3": 4, "Li": 4, "F6P": 4}


# ===========================================================================
# Post-gate research phase (Gate A' declared; docs/reviews/open_items_v2.md
# section 5): the carbonate solvents of the CNT research case. PC and EMC as
# core SPECIES templates, same registry pattern as the W12 EC/PF6- pair, so
# the research frames' generator, the builder and the census share ONE
# geometry. The tests mirror the EC/PF6- block above:
#
#   * both templates exist with the registry's documented keys;
#   * exact stoichiometry (PC = C4H6O3 neutral, 13 atoms; EMC = C4H8O3
#     neutral, 15 atoms) and the cited internal geometry;
#   * a census of one placed molecule of each sees ONE intact molecule;
#   * a full PC/EMC/EC/Li+/PF6- electrolyte program BUILDS with exactly the
#     right atom counts and census composition.
#
# Numbers and their sources (cited in the registration comments too):
#   C=O 1.20 A, ester C-O 1.43 A, sp3 C-C 1.51 A, sp3 C-H 1.09 A
#                 standard organic bond lengths (F. H. Allen, O. Kennard,
#                 D. G. Watson, L. Brammer, A. G. Orpen, R. Taylor,
#                 J. Chem. Soc. Perkin Trans. 2 (1987) S1-S19: ketone C=O
#                 1.21 -> the registry's EC value 1.20 kept, ester C-O 1.43,
#                 sp3 C-C 1.51, sp3 C-H 1.09)
#   O-C(=O) 1.35 A
#                 carbonate O-C(=O) (Allen et al.'s ester C(=O)-O class,
#                 1.33-1.35 A; the cyclic-carbonate value)
#   O=C-O 126 deg, O-C-O 108 deg
#                 the EC template's carbonyl-centre angles (one registry-wide
#                 carbonate convention: 360 - 2 x 126 = 108)
#   sp3 angles 109.47 deg
#                 ideal tetrahedral geometry (arccos(-1/3))
# ===========================================================================


def test_pc_template_exists_with_documented_keys():
    """PC (propylene carbonate) is a core species template."""
    assert "PC" in TEMPLATES, (
        "no 'PC' species template in the core registry; known: "
        f"{sorted(TEMPLATES)}")
    t = TEMPLATES["PC"]
    for key in ("symbols", "rel", "charge", "radius", "pack_radius"):
        assert key in t, f"the PC template misses the documented key {key!r}"
    assert t["symbols"] == ["C", "O", "C", "C", "O", "O", "C"] + ["H"] * 6
    assert len(t["rel"]) == 13
    assert int(t["charge"]) == 0


def test_emc_template_exists_with_documented_keys():
    """EMC (ethyl methyl carbonate) is a core species template."""
    assert "EMC" in TEMPLATES, (
        "no 'EMC' species template in the core registry; known: "
        f"{sorted(TEMPLATES)}")
    t = TEMPLATES["EMC"]
    for key in ("symbols", "rel", "charge", "radius", "pack_radius"):
        assert key in t, f"the EMC template misses the documented key {key!r}"
    assert t["symbols"] == ["C", "O", "C", "O", "O", "C", "C"] + ["H"] * 8
    assert len(t["rel"]) == 15
    assert int(t["charge"]) == 0


def _dihedral(p0, p1, p2, p3):
    """Signed dihedral p0-p1-p2-p3 in degrees (test-side helper)."""
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 /= np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))


def test_pc_published_internal_geometry():
    """PC stoichiometry C4H6O3 and the cited geometry: 5-membered cyclic
    carbonate ring O=C(-O)-O-CH(CH3)-CH2 with ring O-C(=O) 1.35 A x 2,
    ester C-O 1.43 A x 2, ring C-C 1.51 A, carbonyl C=O 1.20 A, methyl
    C-C 1.51 A and 6 x C-H 1.09 A. The ring closes exactly at the cited
    sp3 C-C length (the O2-C3-C4 angle it implies, 103.64 deg, matches the
    squeezed O-C-C angles of five-membered cyclic carbonates); the methyl
    and the ring H occupy the two ideal-tetrahedral sites flanking the ring
    plane (109.47 deg to both ring bonds)."""
    t = TEMPLATES["PC"]
    assert len(t["symbols"]) == 13
    from collections import Counter
    assert Counter(t["symbols"]) == {"C": 4, "H": 6, "O": 3}
    rel = np.asarray(t["rel"], float)
    assert np.allclose(rel.mean(axis=0), 0.0, atol=1e-12)
    d = np.linalg.norm(rel[:, None, :] - rel[None, :, :], axis=-1)
    iu = np.triu_indices(13, 1)
    bonds = sorted(round(x, 2) for x in d[iu] if x < 1.55)
    assert bonds == sorted([1.09] * 6 + [1.20] + [1.35] * 2
                           + [1.43] * 2 + [1.51] * 2), bonds
    assert (d[iu] >= 1.09 - 1e-6).all(), \
        "no PC atom pair may sit inside the shortest published bond"
    # nothing else hides under 1.55 A (every short pair is one of the 13
    # documented bonds; the closest non-bonded contacts are the geminal
    # H-H pairs at 1.78 A)
    assert sum(x < 1.55 for x in d[iu]) == 13
    # the two ring C-O(=C) bonds and the ring closure edge exactly as cited
    assert d[0, 1] == pytest.approx(1.35, abs=1e-6)   # C1-O2
    assert d[0, 4] == pytest.approx(1.35, abs=1e-6)   # C1-O5
    assert d[2, 3] == pytest.approx(1.51, abs=1e-6)   # ring C3-C4
    assert d[0, 5] == pytest.approx(1.20, abs=1e-6)   # C1=O6
    # the methyl sits on the ring CH (C3), out of the ring plane
    assert abs(rel[6, 2]) > 0.5, "the PC methyl must leave the ring plane"


def test_emc_published_internal_geometry():
    """EMC stoichiometry C4H8O3 and the cited geometry: the linear carbonate
    CH3-O-C(=O)-O-CH2-CH3 with carbonate O-C(=O) 1.35 A x 2, ester C-O
    1.43 A x 2, carbonyl C=O 1.20 A, ethyl C-C 1.51 A and 8 x C-H 1.09 A.
    The heavy-atom backbone is the extended anti/anti zigzag (every chain
    dihedral C1-O2-C3-O5-C6-C7 exactly 180 deg; the alkyls syn to the
    carbonyl oxygen, the extended carbonate backbone), so the terminal
    carbons sit 4.54 A apart -- no folded intramolecular contact."""
    t = TEMPLATES["EMC"]
    assert len(t["symbols"]) == 15
    from collections import Counter
    assert Counter(t["symbols"]) == {"C": 4, "H": 8, "O": 3}
    rel = np.asarray(t["rel"], float)
    assert np.allclose(rel.mean(axis=0), 0.0, atol=1e-12)
    d = np.linalg.norm(rel[:, None, :] - rel[None, :, :], axis=-1)
    iu = np.triu_indices(15, 1)
    bonds = sorted(round(x, 2) for x in d[iu] if x < 1.55)
    assert bonds == sorted([1.09] * 8 + [1.20] + [1.35] * 2
                           + [1.43] * 2 + [1.51]), bonds
    assert (d[iu] >= 1.09 - 1e-6).all(), \
        "no EMC atom pair may sit inside the shortest published bond"
    # nothing else hides under 1.55 A (every short pair is one of the 14
    # documented bonds; the closest non-bonded contacts are the geminal
    # H-H pairs at 1.78 A)
    assert sum(x < 1.55 for x in d[iu]) == 14
    # the extended anti/anti backbone, pinned exactly
    assert d[0, 1] == pytest.approx(1.43, abs=1e-6)   # C1-O2
    assert d[1, 2] == pytest.approx(1.35, abs=1e-6)   # O2-C3(=O)
    assert d[2, 4] == pytest.approx(1.35, abs=1e-6)   # C3(=O)-O5
    assert d[2, 3] == pytest.approx(1.20, abs=1e-6)   # C3=O4
    assert d[5, 6] == pytest.approx(1.51, abs=1e-6)   # C6-C7
    assert _dihedral(rel[0], rel[1], rel[2], rel[4]) == \
        pytest.approx(180.0, abs=1e-6)                # C1-O2-C3-O5 anti
    assert _dihedral(rel[1], rel[2], rel[4], rel[5]) == \
        pytest.approx(180.0, abs=1e-6)                # O2-C3-O5-C6 anti
    assert _dihedral(rel[2], rel[4], rel[5], rel[6]) == \
        pytest.approx(180.0, abs=1e-6)                # C3-O5-C6-C7 anti
    assert d[0, 5] == pytest.approx(4.54, abs=0.01), \
        "the extended backbone keeps the terminal carbons apart"


def test_species_mass_of_pc_and_emc():
    assert species_mass("PC") == pytest.approx(102.09, abs=0.02)
    assert species_mass("EMC") == pytest.approx(104.10, abs=0.02)


# ------------------------------------------------- contact consistency ------


@pytest.mark.parametrize("name,formula", [("PC", "C4H6O3"), ("EMC", "C4H8O3")])
def test_census_sees_one_intact_pc_emc_molecule(name, formula, mol_dialect):
    """The shipped PC/EMC geometry must survive the dialect's own bond
    window: the placed molecule is censused as exactly one molecule among
    inert Ar filler (contact-consistent internal geometry: every
    intramolecular bond present, no intermolecular bond to the filler)."""
    frame = _one_molecule_frame(name)
    n_ar = frame.symbols.count("Ar")
    assert n_ar == 45 - 3, "5x3x3 grid minus the 3 centre-line points"
    assert molecule_census(frame, mol_dialect) == {formula: 1, "Ar": n_ar}


# ---------------------------------------------------- build (research) -----


def test_carbonate_electrolyte_program_builds_the_right_composition(
        mol_dialect):
    """The research-case electrolyte program naming all five species builds
    and conserves: 3 PC + 3 EMC + 3 EC + 3 Li+ + 3 PF6- ->
    39 + 45 + 30 + 3 + 21 = 138 atoms, census {C4H6O3: 3, C4H8O3: 3,
    C3H4O3: 3, Li: 3, F6P: 3} (Li+ is a solvation ion: never bonded)."""
    program = parse_text(
        "chaord 0.1\ndialect core + molecular\n\n"
        "system {\n  cell 32 32 32\n  pbc xyz\n  seed 7\n  state T 300 K\n"
        "  conserve atoms C 33 F 18 H 54 Li 3 O 27 P 3\n"
        "  conserve charge 0\n}\n\n"
        "physics {\n  backend mlp\n  model \"mace-mp-0\"\n}\n\n"
        "species {\n  ion Li+ = smiles \"[Li+]\"\n"
        "  ion PF6- = smiles \"F[P-](F)(F)(F)(F)F\"\n}\n\n"
        "liquid electrolyte : all {\n"
        "  molecules PC 3 EMC 3 EC 3 Li+ 3 PF6- 3\n"
        "}\n\n"
        "residual none\n")
    frame = build_program(program, mol_dialect, rng=np.random.default_rng(7),
                          physics=False)
    assert len(frame) == 3 * 13 + 3 * 15 + 3 * 10 + 3 * 1 + 3 * 7
    assert molecule_census(frame, mol_dialect) == \
        {"C4H6O3": 3, "C4H8O3": 3, "C3H4O3": 3, "Li": 3, "F6P": 3}


def test_packing_pc_emc_electrolyte_composition(mol_dialect):
    """The pack primitive itself: the five research species pack as intact
    molecular units (the packing radii keep the census exact)."""
    rng = np.random.default_rng(11)
    frame = pack_molecules({"PC": 3, "EMC": 3, "EC": 3, "Li+": 3, "PF6-": 3},
                           [32.0] * 3, rng, mol_dialect)
    assert len(frame) == 138
    assert molecule_census(frame, mol_dialect) == \
        {"C4H6O3": 3, "C4H8O3": 3, "C3H4O3": 3, "Li": 3, "F6P": 3}
