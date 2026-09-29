"""Water-model statements: the language and the toolchain distinguish rigid
water models (TIP4P vs SPC/E).

Root cause (acceptance run 2026-09-29, A5 reference/nacl_aq, gr_rms x46 the
noise floor): the reference frames are rigid SPC/E water (O-H 1.0000 A,
H-O-H 109.47 deg) + Joung-Cheatham SPC/E ions, but the rebuild chain packed
and rigidly constrained the H2O template's TIP4P geometry (O-H 0.9572 A,
104.52 deg). Under rigid MD the O-H bond length is a conserved quantity, so
the intramolecular g(r) peaks sat one bin off (O-H 0.95 vs 1.05, H-H 1.55 vs
1.65) in every rebuilt frame. The molecular dialect now carries a
``water_models`` table; the fluid lift measures the frame's median O-H and
emits a physics ``model`` statement; the builder packs the model's template;
the classical realization runs the model's published potential.

These tests fail before the fix: the lifted nacl_aq program carries no model
statement and the physics rebuild's median O-H is 0.9572 A, not 1.0000 A.
"""
import numpy as np
import pytest
from pathlib import Path

from chaord.build.molecules import TEMPLATES, molecule_census
from chaord.dialects import load_dialect
from chaord.io.frames import Frame, read_frame
from chaord.lang.fmt import format_program
from chaord.lift import lift_frame

ROOT = Path(__file__).parent.parent
NACL_FRAME = ROOT / "bench" / "reference" / "nacl_aq" / "frame_0.npz"
WATER_FRAME = ROOT / "bench" / "reference" / "water_tip4p" / "frame_0.npz"

# The water_tip4p program must stay byte-identical to its pre-model-statement
# form apart from the one added `model tip4p` line (Review 2 golden).
WATER_TIP4P_PROGRAM = """chaord 0.1
dialect core + molecular

system {
  cell 19.731 19.731 19.731
  pbc xyz
  conserve atoms H 512 O 256
}

physics {
  backend classical
  model tip4p
}

liquid fluid : all {
  molecules H2O 256
  state density 0.997 g/cm3
  assert cn 3.6 +- 1.0 cutoff 3.10 A
  assert gr_peak 2.85 A height 2.64
}

residual none

provenance {
  dialects "core 0.1.0 + molecular 0.1.0"
  lift_version "0.1.0"
}
"""


@pytest.fixture(scope="module")
def mol_dialect():
    return load_dialect(("core", "molecular"))


def _oh_median(frame) -> float:
    """Median minimum-image O-H length over the frame's OHH water triples."""
    L = frame.cell_diag
    syms = np.asarray(frame.symbols)
    o = np.where(syms == "O")[0]
    d1 = frame.pos[o + 1] - frame.pos[o]
    d2 = frame.pos[o + 2] - frame.pos[o]
    d1 -= L * np.round(d1 / L)
    d2 -= L * np.round(d2 / L)
    return float(np.median(np.concatenate(
        [np.linalg.norm(d1, axis=1), np.linalg.norm(d2, axis=1)])))


# --------------------------------------------------------------- the table --

def test_water_models_table_is_published_geometry(mol_dialect):
    """The dialect table is the single authority: classification geometry is
    the published pair (TIP4P: Jorgensen 1983; SPC/E: Berendsen 1987) and the
    windows are disjoint, with a declared default model."""
    table = mol_dialect.threshold("water_models")
    assert table["default"] == "tip4p"
    tip4p, spce = table["tip4p"], table["spce"]
    assert tip4p["r_oh_A"] == pytest.approx(0.9572, abs=1e-6)
    assert tip4p["hoh_angle_deg"] == pytest.approx(104.52, abs=0.01)
    assert spce["r_oh_A"] == pytest.approx(1.0, abs=1e-6)
    assert spce["hoh_angle_deg"] == pytest.approx(109.47, abs=0.01)
    # published SPC/E potential parameters (Berendsen 1987)
    assert spce["sigma_O_A"] == pytest.approx(3.166, abs=1e-6)
    assert spce["epsilon_O_K"] == pytest.approx(78.22, abs=0.01)
    assert spce["q_O_e"] == pytest.approx(-0.8476, abs=1e-6)
    assert spce["q_H_e"] == pytest.approx(0.4238, abs=1e-6)
    # realization truncation: the nacl_aq reference's recorded DSF protocol
    # (Fennell & Gezelter 2006)
    assert spce["dsf_alpha_inv_A"] == pytest.approx(0.2, abs=1e-9)
    assert spce["realization_rc_A"] == pytest.approx(9.0, abs=1e-9)
    # citations recorded with the parameters
    assert "Berendsen" in spce["citation"]
    # the classification windows must not overlap
    w = float(tip4p["classify_half_width_A"])
    assert tip4p["r_oh_A"] + w < spce["r_oh_A"] - float(spce["classify_half_width_A"])


def test_classify_water_model_by_median_oh(mol_dialect):
    """The shared classifier (one definition, used by lift and realize):
    each model claims a half-width window around its published r_OH; the
    midpoint between TIP4P and SPC/E claims nothing."""
    from chaord.build.molecules import classify_water_model
    assert classify_water_model(mol_dialect, 0.9572) == "tip4p"
    assert classify_water_model(mol_dialect, 1.0000) == "spce"
    mid = 0.5 * (0.9572 + 1.0000)
    assert classify_water_model(mol_dialect, mid) is None


def test_water_templates_per_model(mol_dialect):
    """H2O keeps the TIP4P template (the dialect default, byte-for-byte the
    old geometry); the model-qualified SPC/E template carries the published
    SPC/E geometry and the same atoms/mass."""
    from chaord.build.molecules import molecular_mass, packing_radius
    tip4p = TEMPLATES["H2O"]
    spce = TEMPLATES["H2O/spce"]
    assert spce["symbols"] == ["O", "H", "H"]
    for name, r_oh, angle in (("H2O", 0.9572, 104.52), ("H2O/spce", 1.0, 109.47)):
        rel = TEMPLATES[name]["rel"]
        d1 = np.linalg.norm(rel[1] - rel[0])
        d2 = np.linalg.norm(rel[2] - rel[0])
        assert d1 == pytest.approx(r_oh, abs=1e-3), name
        assert d2 == pytest.approx(r_oh, abs=1e-3), name
        v1, v2 = rel[1] - rel[0], rel[2] - rel[0]
        cosang = np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2)
        assert np.degrees(np.arccos(cosang)) == pytest.approx(angle, abs=0.2), name
    assert molecular_mass("H2O/spce") == pytest.approx(molecular_mass("H2O"))
    # the wider SPC/E geometry packs with the wider radius
    assert packing_radius("H2O/spce", mol_dialect) > packing_radius("H2O", mol_dialect)


# ------------------------------------------------------------------- lift --

def test_nacl_aq_lift_states_spce(mol_dialect):
    """Root cause, reproduced: the nacl_aq reference frame is SPC/E water;
    its lifted program must say so (and name the model's water template, so
    the rebuild packs SPC/E geometry). Failed before the fix: no model
    statement at all."""
    frame = read_frame(NACL_FRAME)
    text = format_program(lift_frame(frame, mol_dialect))
    assert "model spce" in text
    assert "molecules H2O/spce 540" in text
    assert "molecules Na+ 10" in text and "molecules Cl- 10" in text


def test_water_tip4p_lift_is_the_golden_program_plus_model_line(mol_dialect):
    """water_tip4p lifts to exactly its previous text plus `model tip4p`
    (lineage guard: the added statement changes nothing else)."""
    frame = read_frame(WATER_FRAME)
    text = format_program(lift_frame(frame, mol_dialect))
    assert text == WATER_TIP4P_PROGRAM


def test_unclassifiable_water_emits_no_model_but_notes_geometry(mol_dialect):
    """Water at the window midpoint is no known rigid model: no model
    statement (default tip4p applies) and the measured median goes to
    provenance."""
    from chaord.build.molecules import pack_molecules
    rng = np.random.default_rng(5)
    frame = pack_molecules({"H2O": 16}, [9.0] * 3, rng, mol_dialect)
    L = frame.cell_diag
    # move every H radially to the unclassified midpoint distance
    mid = 0.5 * (0.9572 + 1.0)
    pos = frame.pos.copy()
    for h, o in ((1, 0), (2, 0)):
        d = pos[h::3] - pos[o::3]
        d -= L * np.round(d / L)
        pos[h::3] = pos[o::3] + d / np.linalg.norm(d, axis=1)[:, None] * mid
    frame = Frame(pos=pos, cell=frame.cell, symbols=frame.symbols,
                  pbc=frame.pbc)
    text = format_program(lift_frame(frame, mol_dialect, mode="fluid"))
    assert "model " not in text
    assert "molecules H2O 16" in text            # default template name
    assert f"{mid:.4f}" in text or f"{mid:.3f}" in text  # provenance median


# ----------------------------------------------------------------- realize --

def _spce_water_frame(n, box, seed):
    """Clean rigid SPC/E waters (whole molecules, no overlaps)."""
    rng = np.random.default_rng(seed)
    pos = np.zeros((n * 3, 3))
    th = np.radians(109.47 / 2)
    r = 1.0
    centers = []
    for m in range(n):
        for _ in range(100000):
            o = rng.uniform(1.5, box - 1.5, 3)
            if all(np.linalg.norm(o - c) > 3.0 for c in centers):
                centers.append(o)
                break
        else:
            raise RuntimeError("spce water generator jammed; enlarge the box")
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        v = rng.normal(size=3)
        v -= u * (v @ u)
        v /= np.linalg.norm(v)
        pos[3 * m] = o
        pos[3 * m + 1] = o + r * (np.cos(th) * u + np.sin(th) * v)
        pos[3 * m + 2] = o + r * (np.cos(th) * u - np.sin(th) * v)
    return Frame(pos=pos, cell=np.diag([box] * 3), symbols=["O", "H", "H"] * n,
                 pbc=(True, True, True))


def _spce_pair_reference(pos, syms, table, rc, alpha):
    """Independent brute-force energy of one SPC/E + ion configuration, the
    reference family's own recorded scheme (bench/reference/nacl_aq
    provenance): Lorentz-Berthelot LJ energy-shifted at rc (force unshifted)
    between LJ sites, damped-shifted-force (Wolf/DSF) electrostatics between
    all charged sites, intramolecular water pairs excluded."""
    from scipy.special import erfc
    from ase import units
    kc = units.Hartree * units.Bohr
    lj = {"O": (float(table["sigma_O_A"]),
                float(table["epsilon_O_K"]) * units.kB)}
    # Joung-Cheatham (Table 3, SPC/E set), as recorded in the provenance
    lj["Na"] = (2.159542, 0.3526418 * units.kcal / units.mol)
    lj["Cl"] = (4.830486, 0.0300147 * units.kcal / units.mol)
    q = {"O": float(table["q_O_e"]), "H": float(table["q_H_e"]),
         "Na": 1.0, "Cl": -1.0}
    mol = np.arange(len(pos)) // 3  # waters are OHH triples, ions singletons
    for k in range(len(pos) // 3 * 3, len(pos)):
        mol[k] = k
    f_rc = erfc(alpha * rc) / rc
    g_rc = f_rc / rc + 2 * alpha / np.sqrt(np.pi) * np.exp(-(alpha * rc) ** 2) / rc
    e = 0.0
    for a in range(len(pos)):
        for b in range(a + 1, len(pos)):
            if mol[a] == mol[b]:
                continue
            d = np.linalg.norm(pos[b] - pos[a])
            if d >= rc:
                continue
            if syms[a] in lj and syms[b] in lj:
                sig = 0.5 * (lj[syms[a]][0] + lj[syms[b]][0])
                eps = np.sqrt(lj[syms[a]][1] * lj[syms[b]][1])
                s6, s6c = (sig / d) ** 6, (sig / rc) ** 6
                e += 4 * eps * (s6 * s6 - s6 - (s6c * s6c - s6c))
            f_r = erfc(alpha * d) / d
            e += kc * q[syms[a]] * q[syms[b]] * (f_r - f_rc + (d - rc) * g_rc)
    return e


def test_spce_water_kernel_matches_bruteforce_published_potential(mol_dialect):
    """The SPC/E realization kernel is the published three-site model in the
    reference family's own truncation: LJ (sigma 3.166 A, epsilon/kB 78.22 K,
    energy-shifted at rc) and DSF electrostatics on q_O -0.8476 e /
    q_H +0.4238 e (Berendsen 1987; DSF scheme Fennell & Gezelter 2006, the
    recorded protocol of the nacl_aq reference). Verified against independent
    brute-force arithmetic on a fixed configuration with ions (all pairs
    inside the cutoff and half the box)."""
    from ase import Atoms
    from chaord.realize.ase_backend import TIP4PSolution
    table = mol_dialect.threshold("water_models")["spce"]
    rc = float(table["realization_rc_A"])
    alpha = float(table["dsf_alpha_inv_A"])
    r, th = 1.0, np.radians(109.47 / 2)
    centers = [np.array(c, float) for c in
               ((10.0, 10.0, 10.0), (12.5, 10.3, 10.1),
                (11.0, 12.4, 10.2), (10.4, 11.0, 12.6))]
    pos, syms = [], []
    for c in centers:
        pos += [c, c + r * np.array([np.cos(th), np.sin(th), 0.0]),
                c + r * np.array([np.cos(th), -np.sin(th), 0.0])]
        syms += ["O", "H", "H"]
    pos += [np.array([12.2, 11.8, 11.5]), np.array([9.4, 11.2, 9.2])]
    syms += ["Na", "Cl"]
    pos = np.array(pos)
    at = Atoms(syms, positions=pos, cell=np.diag([24.0] * 3), pbc=True)
    at.calc = TIP4PSolution(dialect=mol_dialect, model="spce")
    got = at.get_potential_energy()
    want = _spce_pair_reference(pos, syms, table, rc, alpha)
    assert got == pytest.approx(want, rel=1e-10)


def test_spce_solution_forces_are_exact_derivatives(mol_dialect):
    """The SPC/E + Joung-Cheatham kernel must be the exact gradient of its
    own energy (finite differences over every atom and axis)."""
    from ase import Atoms
    from chaord.realize.ase_backend import TIP4PSolution
    f = _spce_water_frame(8, 10.5, seed=11)
    rng = np.random.default_rng(2)
    pos = f.pos
    while len(pos) < len(f) + 2:
        p = rng.uniform(1.5, 9.0, 3)
        if np.linalg.norm(pos - p[None], axis=1).min() > 2.4:
            pos = np.vstack([pos, p])
    syms = f.symbols + ["Na", "Cl"]
    atoms = Atoms(syms, positions=pos, cell=f.cell, pbc=True)
    atoms.calc = TIP4PSolution(rc=5.0, width=1.0, dialect=mol_dialect,
                               model="spce")
    f0 = atoms.get_forces()
    base = atoms.positions.copy()
    h = 1e-6
    for i in range(len(atoms)):
        for axis in range(3):
            p = base.copy()
            p[i, axis] += h
            atoms.positions = p
            ep = atoms.get_potential_energy()
            p = base.copy()
            p[i, axis] -= h
            atoms.positions = p
            em = atoms.get_potential_energy()
            num = (ep - em) / (2 * h)
            assert num == pytest.approx(-f0[i, axis], rel=2e-5, abs=1e-6), \
                (i, axis)


def test_backend_resolves_model_from_water_geometry(mol_dialect):
    """The frozen builder call sites pass only (cell, symbols, backend,
    dialect): the model is resolved from the water geometry itself (rigid MD
    conserves O-H, the same measurement the lift makes). Packed TIP4P
    geometry -> tip4p forces; packed SPC/E geometry -> SPC/E forces."""
    from chaord.build.molecules import pack_molecules
    from chaord.realize.ase_backend import ASEBackend
    tip = pack_molecules({"H2O": 12}, [9.0] * 3, np.random.default_rng(4),
                         mol_dialect)
    spce = pack_molecules({"H2O/spce": 12}, [9.0] * 3, np.random.default_rng(4),
                          mol_dialect)
    b_tip = ASEBackend(tip.cell_diag, tip.symbols, "classical", dialect=mol_dialect)
    b_spce = ASEBackend(spce.cell_diag, spce.symbols, "classical", dialect=mol_dialect)
    b_tip.forces(np.mod(tip.pos, tip.cell_diag))      # one energy call resolves
    b_spce.forces(np.mod(spce.pos, spce.cell_diag))
    assert b_tip.model == "tip4p"
    assert b_spce.model == "spce"


def test_unknown_water_template_is_a_static_error(mol_dialect):
    """A program naming a water template the registry does not ship raises
    the species error (never silently packs another geometry)."""
    from chaord.build import build_program
    from chaord.lang.errors import ChaordError
    from chaord.lang.parser import parse_text
    text = ("chaord 0.1\ndialect core + molecular\n\n"
            "system {\n  cell 9 9 9\n  pbc xyz\n  state T 300 K\n}\n\n"
            "physics {\n  backend classical\n}\n\n"
            "liquid water : all {\n  molecules H2O/tip5p 20\n}\n")
    with pytest.raises(ChaordError, match="unknown species|no template"):
        build_program(parse_text(text), mol_dialect,
                      rng=np.random.default_rng(1), physics=True, md_steps=1)


def test_unknown_explicit_model_is_rejected(mol_dialect):
    """An explicit model= the dialect table does not parameterize is a
    static error of the calculator (no silent wrong potential)."""
    from chaord.lang.errors import ChaordError
    from chaord.realize.ase_backend import TIP4PSolution
    with pytest.raises(ChaordError, match="unknown water model"):
        TIP4PSolution(dialect=mol_dialect, model="tip5p")


# ----------------------------------------------------------------- rebuild --

@pytest.mark.slow
def test_nacl_aq_rebuild_uses_spce_geometry(mol_dialect):
    """Root cause, full chain: lift the nacl_aq frame, rebuild with physics
    on -- the rebuilt water must keep the SPC/E geometry (median O-H 1.0000
    A). Failed before the fix: the rebuild packed and rigidly constrained
    TIP4P water, median O-H 0.9572 A, and the intramolecular g(r) peaks sat
    one bin off (A5 gr_rms x46 the noise floor)."""
    from chaord.build import build_program
    frame = read_frame(NACL_FRAME)
    prog = lift_frame(frame, mol_dialect)
    rebuilt = build_program(prog, mol_dialect, rng=np.random.default_rng(3),
                            physics=True, md_steps=20)
    assert len(rebuilt) == len(frame) == 1640
    assert molecule_census(rebuilt, mol_dialect) == {
        "H2O": 540, "Na": 10, "Cl": 10}
    assert _oh_median(rebuilt) == pytest.approx(1.0, abs=0.01), \
        f"rebuilt median O-H {_oh_median(rebuilt):.4f} A is not SPC/E"


@pytest.mark.slow
def test_water_tip4p_rebuild_keeps_tip4p_geometry(mol_dialect):
    """No-regression guard: the water_tip4p reference still lifts and
    rebuilds with its TIP4P geometry."""
    from chaord.build import build_program
    frame = read_frame(WATER_FRAME)
    prog = lift_frame(frame, mol_dialect)
    rebuilt = build_program(prog, mol_dialect, rng=np.random.default_rng(3),
                            physics=True, md_steps=20)
    assert len(rebuilt) == 768
    assert molecule_census(rebuilt, mol_dialect) == {"H2O": 256}
    assert _oh_median(rebuilt) == pytest.approx(0.9572, abs=0.01)
