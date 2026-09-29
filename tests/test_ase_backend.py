"""ASE realization backend acceptance: TIP4P water, ion solutions, LJ
molecular fluids, tabulated EAM.

Review 2: `backend classical` (what every molecular lift states) had no core
realization -- build_fluid raised "no realize backend for 'classical'". These
tests pin the closing of that gap with ASE's own calculators:

* the wrapper reproduces ase.calculators.tip4p.TIP4P / eam.EAM / lj numbers,
* rigid water stays rigid through the relaxation (FixBondLengths-equivalent
  constraint; TIP4P virtual sites live inside the vectorized kernel whose
  numbers match ASE's calculator to machine precision),
* the ion terms of the solution calculator and the site-wise LJ terms of the
  molecular-fluid calculator are the exact derivative of their energy
  (finite-difference checks),
* `backend classical` water and NaCl-solution programs build with physics on
  and conserve every atom, and the shipped bench reference cases round trip;
* A5 regressions: the LJ molecular fluids (Ar gas, N2) realize classically
  from the dialect's published-parameter table (no-potential species still
  raise), and the 1640-atom nacl_aq rebuild fits the acceptance budget --
  one force evaluation times the protocol's own step count (fast check), the
  whole physics rebuild (slow check), both deterministic under one seed.
"""
import numpy as np
import pytest
from pathlib import Path

from ase import Atoms

from chaord.build.molecules import molecule_census, pack_molecules
from chaord.dialects import load_dialect
from chaord.io.frames import Frame, read_frame
from chaord.lang.api import load
from chaord.lang.parser import parse_text

ROOT = Path(__file__).parent.parent
CU_EAM = ROOT / "bench" / "reference" / "potentials" / "Cu_u3.eam"


@pytest.fixture(scope="module")
def mol_dialect():
    return load_dialect(("core", "molecular"))


def _water_frame(n, box, seed):
    """Packed rigid water (OHH per molecule) in a cubic box."""
    return pack_molecules({"H2O": n}, [box] * 3, np.random.default_rng(seed),
                          load_dialect(("core", "molecular")))


def _clean_water(n, box, seed):
    """Random rigid waters with whole molecules (no boundary straddling) and
    no hard overlaps -- the configuration ASE's own TIP4P asserts on, for
    equivalence tests."""
    from ase.calculators.tip4p import angleHOH, rOH
    rng = np.random.default_rng(seed)
    pos = np.zeros((n * 3, 3))
    th = np.radians(angleHOH / 2)
    centers = []
    for m in range(n):
        for _ in range(100000):
            o = rng.uniform(1.5, box - 1.5, 3)
            if all(np.linalg.norm(o - c) > 3.0 for c in centers):
                centers.append(o)
                break
        else:
            raise RuntimeError(
                f"clean-water generator jammed at {m}/{n} molecules; "
                f"enlarge the box")
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        v = rng.normal(size=3)
        v -= u * (v @ u)
        v /= np.linalg.norm(v)
        pos[3 * m] = o
        pos[3 * m + 1] = o + rOH * (np.cos(th) * u + np.sin(th) * v)
        pos[3 * m + 2] = o + rOH * (np.cos(th) * u - np.sin(th) * v)
    syms = ["O", "H", "H"] * n
    return Frame(pos=pos, cell=np.diag([box] * 3), symbols=syms,
                 pbc=(True, True, True))


def _oh_lengths(pos, L=None):
    d = lambda a, b: pos[a::3] - pos[b::3]
    if L is not None:
        L = np.asarray(L, float)
        d = lambda a, b: pos[a::3] - pos[b::3] - L * np.round(
            (pos[a::3] - pos[b::3]) / L)
    return (np.linalg.norm(d(1, 0), axis=1),
            np.linalg.norm(d(2, 0), axis=1))


def _hoh_angle(pos, L=None):
    v1 = pos[1::3] - pos[0::3]
    v2 = pos[2::3] - pos[0::3]
    if L is not None:
        L = np.asarray(L, float)
        v1 = v1 - L * np.round(v1 / L)
        v2 = v2 - L * np.round(v2 / L)
    cosang = np.sum(v1 * v2, 1) / (np.linalg.norm(v1, axis=1)
                                   * np.linalg.norm(v2, axis=1))
    return np.degrees(np.arccos(np.clip(cosang, -1, 1)))


# ------------------------------------------------------------- water kernel --

def test_tip4p_forces_match_ase_calculator():
    """The backend wraps ase.calculators.tip4p.TIP4P, it does not re-derive
    it: energies and forces must agree to machine precision."""
    from ase.calculators.tip4p import TIP4P
    from chaord.realize.ase_backend import ASEBackend
    f = _clean_water(32, 12.0, seed=5)
    rc = 5.0  # 2*rc must stay inside the box for the ASE calculator
    ref = Atoms(f.symbols, positions=f.pos, cell=f.cell, pbc=True)
    ref.calc = TIP4P(rc=rc, width=1.0)
    backend = ASEBackend(f.cell_diag, f.symbols, "tip4p", rc=rc)
    assert backend.energy(f.pos) == pytest.approx(ref.get_potential_energy(),
                                                  abs=1e-9)
    assert np.abs(backend.forces(f.pos) - ref.get_forces()).max() < 1e-8


def test_solution_ion_forces_are_exact_derivatives():
    """The ion-ion and ion-water LJ+Coulomb kernel must be the exact gradient
    of its own energy (finite differences over every atom and axis)."""
    from chaord.realize.ase_backend import TIP4PSolution
    f = _clean_water(16, 10.5, seed=11)
    rng = np.random.default_rng(2)
    pos = f.pos
    # append 4 ions at sane solvation distances (no overlap with the water)
    while len(pos) < len(f) + 4:
        p = rng.uniform(1.5, 9.0, 3)
        if np.linalg.norm(pos - p[None], axis=1).min() > 2.4:
            pos = np.vstack([pos, p])
    syms = f.symbols + ["Na", "Na", "Cl", "Cl"]
    atoms = Atoms(syms, positions=pos, cell=f.cell, pbc=True)
    atoms.calc = TIP4PSolution(rc=5.0, width=1.0)
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
            # 1e-5 relative: central differences at h=1e-6 A (pairs sitting in
            # the smooth-cutoff switch add an O(h) truncation term); a wrong
            # term (a sign or a factor) shows up orders above this
            assert num == pytest.approx(-f0[i, axis], rel=2e-5, abs=1e-6), \
                (i, axis)


def test_solution_without_ions_is_tip4p():
    """Zero ions: the solution calculator must reduce to plain ASE TIP4P."""
    from ase.calculators.tip4p import TIP4P
    from chaord.realize.ase_backend import TIP4PSolution
    f = _clean_water(24, 11.0, seed=7)
    rc = 5.0
    ref = Atoms(f.symbols, positions=f.pos, cell=f.cell, pbc=True)
    ref.calc = TIP4P(rc=rc, width=1.0)
    at = Atoms(f.symbols, positions=f.pos, cell=f.cell, pbc=True)
    at.calc = TIP4PSolution(rc=rc, width=1.0)
    assert at.get_potential_energy() == pytest.approx(
        ref.get_potential_energy(), abs=1e-9)
    assert np.abs(at.get_forces() - ref.get_forces()).max() < 1e-8


# ---------------------------------------------------- rigid-water constraint --

def test_rigid_water_constraint_preserves_geometry():
    """The vectorized constraint satisfies the SHAKE/RATTLE contract of
    ase.constraints.FixBondLengths: every bond length restored (and momenta
    projected) after an arbitrary displacement.

    (Compared head-to-head, ASE's per-bond FixBondLengths loop does not
    converge on this 16-water displacement within its 500-iteration budget
    on this machine -- the per-bond Python loop is the very slowness the
    vectorized sweep exists to fix -- so the contract is asserted directly:
    restored lengths plus momentum and centre-of-mass conservation.)"""
    from chaord.realize.ase_backend import RigidWater
    f = _clean_water(16, 12.0, seed=3)
    rng = np.random.default_rng(9)
    d1, d2 = _oh_lengths(f.pos)
    hh = np.linalg.norm(f.pos[2::3] - f.pos[1::3], axis=1)

    atoms = Atoms(f.symbols, positions=f.pos, cell=f.cell, pbc=True)
    constraint = RigidWater(np.arange(48))
    new = atoms.positions + rng.normal(scale=0.15, size=f.pos.shape)
    p = rng.normal(size=f.pos.shape)
    p0 = p.sum(0)
    constraint.adjust_positions(atoms, new)
    constraint.adjust_momenta(atoms, p)
    n1, n2 = _oh_lengths(new)
    nhh = np.linalg.norm(new[2::3] - new[1::3], axis=1)
    assert np.abs(n1 - d1).max() < 1e-8
    assert np.abs(n2 - d2).max() < 1e-8
    assert np.abs(nhh - hh).max() < 1e-8
    assert np.abs(p.sum(0) - p0).max() < 1e-8


def test_relaxed_water_stays_rigid():
    """FIRE + Langevin relaxation keeps the TIP4P geometry: O-H at the
    template bond length, H-O-H at the template angle, atoms conserved."""
    from chaord.realize.ase_backend import ASEBackend
    f = _water_frame(32, 12.0, seed=13)
    backend = ASEBackend(f.cell_diag, f.symbols, "tip4p", rc=5.0)
    r = backend.relax(f.pos, rng=np.random.default_rng(17), T_K=300.0,
                      md_steps=30)
    assert len(r) == len(f)
    # packed frames straddle the boundary: measure min-image lengths
    d1, d2 = _oh_lengths(r, f.cell_diag)
    assert np.abs(d1 - 0.9572).max() < 0.02
    assert np.abs(d2 - 0.9572).max() < 0.02
    ang = _hoh_angle(r, f.cell_diag)
    assert np.abs(ang - 104.52).max() < 1.0
    assert np.isfinite(r).all()


# ------------------------------------------------------------------ EAM, LJ --

@pytest.mark.skipif(not CU_EAM.exists(), reason="shipped Cu_u3.eam potential")
def test_eam_backend_matches_ase_eam():
    """The eam flavor wraps ase.calculators.eam.EAM on the shipped tabulated
    potential (bench/reference/potentials/Cu_u3.eam)."""
    from ase.calculators.eam import EAM
    from ase.build import bulk
    from chaord.realize.ase_backend import ASEBackend
    ref = bulk("Cu", "fcc", a=3.615, cubic=True).repeat((2, 2, 1))
    ref.rattle(stdev=0.15, seed=4)
    ref.calc = EAM(potential=str(CU_EAM))
    backend = ASEBackend(ref.cell.diagonal(), ref.get_chemical_symbols(),
                         "eam", eam_potential=str(CU_EAM))
    assert backend.energy(ref.positions) == pytest.approx(
        ref.get_potential_energy(), abs=1e-9)
    assert np.abs(backend.forces(ref.positions) - ref.get_forces()).max() < 1e-8


def test_lj_flavor_matches_ase_lj():
    """The lj flavor wraps ase.calculators.lj.LennardJones."""
    from ase.calculators.lj import LennardJones
    from chaord.realize.ase_backend import ASEBackend
    rng = np.random.default_rng(8)
    pos = rng.uniform(0, 6, (30, 3))
    syms = ["Ar"] * 30
    ref = Atoms(syms, positions=pos, cell=[6, 6, 6], pbc=True)
    ref.calc = LennardJones(epsilon=0.01, sigma=3.0, rc=5.0)
    backend = ASEBackend([6.0] * 3, syms, "lj", epsilon=0.01, sigma=3.0,
                         rc=5.0)
    assert backend.energy(pos) == pytest.approx(ref.get_potential_energy(),
                                                abs=1e-10)
    assert np.abs(backend.forces(pos) - ref.get_forces()).max() < 1e-8


@pytest.mark.skipif(not CU_EAM.exists(), reason="shipped Cu_u3.eam potential")
def test_eam_backend_runs_with_run_md():
    """Constraint-free flavors plug into the shared BAOAB integrator through
    the LJ-class build(r)/forces(r) interface (the reuse contract)."""
    from ase.build import bulk
    from chaord.realize.ase_backend import ASEBackend
    from chaord.realize.lj import run_md
    rng = np.random.default_rng(21)
    ref = bulk("Cu", "fcc", a=3.615, cubic=True).repeat((2, 2, 2))
    ref.rattle(stdev=0.12, seed=6)
    L = ref.cell.diagonal().copy()
    backend = ASEBackend(L, ref.get_chemical_symbols(), "eam",
                         eam_potential=str(CU_EAM))
    r = np.asarray(ref.positions, float)
    v = np.zeros_like(r)
    r2, _ = run_md(r, v, L, 30, 0.002, 0.03, 2.0, rng, lj=backend)
    assert len(r2) == len(r)
    assert np.isfinite(r2).all()
    # forces are not zero: the rattled lattice is off equilibrium
    assert np.abs(backend.forces(r2)).max() > 1e-4


# --------------------------------------------------- build wiring (Review 2) --

_WATER_PROGRAM = ("chaord 0.1\ndialect core + molecular\n\n"
                  "system {{\n  cell auto cubic\n  pbc xyz\n"
                  "  state T 300 K\n}}\n\n"
                  "physics {{\n  backend classical\n}}\n\n"
                  "liquid water : all {{\n  molecules H2O {n}\n"
                  "  state density 0.95 g/cm3\n}}\n")


def test_classical_water_builds_with_physics(mol_dialect):
    """Review 2 root cause, reproduced: `backend classical` raised
    'no realize backend' before this backend existed. It must now relax the
    packed frame and conserve every atom."""
    from chaord.build import build_program
    text = _WATER_PROGRAM.format(n=60)
    frame = build_program(parse_text(text), mol_dialect,
                          rng=np.random.default_rng(3), physics=True,
                          md_steps=25)
    assert len(frame) == 180
    assert molecule_census(frame, mol_dialect) == {"H2O": 60}
    d1, d2 = _oh_lengths(frame.pos, frame.cell_diag)
    assert np.abs(d1 - 0.9572).max() < 0.02
    assert np.abs(d2 - 0.9572).max() < 0.02


_NACL_PROGRAM = ("chaord 0.1\ndialect core + molecular\n\n"
                 "system {\n  cell auto cubic\n  pbc xyz\n"
                 "  state T 300 K\n}\n\n"
                 "physics {\n  backend classical\n}\n\n"
                 "liquid brine : all {\n  molecules Cl- 2\n"
                 "  molecules H2O 100\n  molecules Na+ 2\n"
                 "  state density 1.05 g/cm3\n}\n")


def test_classical_solution_builds_with_physics(mol_dialect):
    """Water + Joung-Cheatham ions: the classical backend realizes the
    solvent as rigid TIP4P and the ions with the JC parameters."""
    from chaord.build import build_program
    frame = build_program(parse_text(_NACL_PROGRAM), mol_dialect,
                          rng=np.random.default_rng(5), physics=True,
                          md_steps=25)
    assert len(frame) == 304
    assert molecule_census(frame, mol_dialect) == {"H2O": 100, "Na": 2, "Cl": 2}


def test_classical_unknown_species_is_an_error(mol_dialect, tmp_path):
    """No silent fall-back: molecules the classical backend has no potential
    for raise (physics=False stays the explicit opt-out). O2 has a packing
    template but no classical_pair_potentials entry, so it still pins the
    gate now that Ar and N2 are parameterized."""
    from chaord.build import build_program
    from chaord.lang.errors import ChaordError
    text = _WATER_PROGRAM.format(n=60).replace("molecules H2O 60",
                                               "molecules O2 40")
    path = tmp_path / "o2.chaord"
    path.write_text(text)
    with pytest.raises(ChaordError, match="no ASE classical potential"):
        build_program(load(path), mol_dialect, rng=np.random.default_rng(1))


def test_unknown_backend_still_raises(mol_dialect, tmp_path):
    """The Review 2 no-silent-skip gate: unimplemented backends still raise."""
    from chaord.build import build_program
    from chaord.lang.errors import ChaordError
    text = _WATER_PROGRAM.format(n=60).replace("classical", "mlp")
    path = tmp_path / "mlp.chaord"
    path.write_text(text)
    with pytest.raises(ChaordError, match="no realize backend for 'mlp'"):
        build_program(load(path), mol_dialect, rng=np.random.default_rng(1))


# ------------------------------------------------ shipped reference cases ---

def test_reference_water_tip4p_round_trip(mol_dialect):
    """Review 2 acceptance: the water_tip4p reference frame lifts to a
    `backend classical` program that builds back with physics on, conserving
    every atom and the exact H2O census."""
    from chaord.build import build_program
    from chaord.lift import lift_frame
    f = read_frame(ROOT / "bench" / "reference" / "water_tip4p" / "frame_0.npz")
    prog = lift_frame(f, mol_dialect)
    rebuilt = build_program(prog, mol_dialect, rng=np.random.default_rng(3),
                            physics=True, md_steps=20)
    assert len(rebuilt) == len(f) == 768
    assert molecule_census(rebuilt, mol_dialect) == {"H2O": 256}


@pytest.mark.slow
def test_reference_nacl_aq_round_trip(mol_dialect):
    """Review 2 acceptance: the nacl_aq solution reference frame lifts and
    rebuilds with physics on, conserving every atom and the exact census."""
    from chaord.build import build_program
    from chaord.lift import lift_frame
    f = read_frame(ROOT / "bench" / "reference" / "nacl_aq" / "frame_0.npz")
    prog = lift_frame(f, mol_dialect)
    rebuilt = build_program(prog, mol_dialect, rng=np.random.default_rng(3),
                            physics=True, md_steps=20)
    assert len(rebuilt) == len(f) == 1640
    assert molecule_census(rebuilt, mol_dialect) == {
        "H2O": 540, "Na": 10, "Cl": 10}


# ------------------------------------------ A5 regression: LJ molecular fluids --

_AR_PROGRAM = ("chaord 0.1\ndialect core + molecular\n\n"
                "system {\n  cell 25.0 25.0 25.0\n  pbc xyz\n"
                "  conserve atoms Ar 50\n}\n\n"
                "physics {\n  backend classical\n}\n\n"
                "liquid gas : all {\n  state density 0.003\n"
                "  assert gr_peak 3.35 height 2.66\n}\n")

_N2_PROGRAM = ("chaord 0.1\ndialect core + molecular\n\n"
               "system {\n  cell 22.0 22.0 22.0\n  pbc xyz\n"
               "  conserve atoms N 160\n}\n\n"
               "physics {\n  backend classical\n}\n\n"
               "liquid fluid : all {\n  molecules N2 80\n"
               "  state density 0.349 g/cm3\n"
               "  assert gr_peak 3.55 A height 1.89\n}\n")


def test_classical_ar_gas_builds_with_physics(mol_dialect):
    """A5 regression, reproduced: the lifted ar_gas_box25 program states
    `backend classical`, and after Review 2/T5 wired that name to the ASE
    backend its 50 Ar atoms raise 'no ASE classical potential for Ar' -- a
    hard build failure where the pre-T5 program was (honestly) degraded to
    physics=False. The molecular dialect's published LJ table must realize
    monatomic Ar: atom-centred LJ, every atom conserved."""
    from chaord.build import build_program
    frame = build_program(parse_text(_AR_PROGRAM), mol_dialect,
                          rng=np.random.default_rng(7), physics=True)
    assert len(frame) == 50
    assert molecule_census(frame, mol_dialect) == {"Ar": 50}
    assert np.isfinite(frame.pos).all()


def test_classical_n2_builds_with_physics(mol_dialect):
    """A5 regression, reproduced: n2_box22 (160 N = 80 rigid N2) raised 'no
    ASE classical potential for N'. The two-site realization keeps the N2
    template bond rigid through the relaxation and the census exact."""
    from chaord.build import build_program
    frame = build_program(parse_text(_N2_PROGRAM), mol_dialect,
                          rng=np.random.default_rng(7), physics=True)
    assert len(frame) == 160
    assert molecule_census(frame, mol_dialect) == {"N2": 80}
    # rigid diatomic: the min-image N-N distance stays at the template bond
    L = frame.cell_diag
    d = frame.pos[1::2] - frame.pos[0::2]
    d -= L * np.round(d / L)
    assert np.abs(np.linalg.norm(d, axis=1) - 1.10).max() < 0.01
    assert np.isfinite(frame.pos).all()


def test_lj_fluid_energy_is_published_lj(mol_dialect):
    """The LJ-fluid kernel uses the dialect's published parameters in physical
    units: two Ar atoms well inside the cutoff interact with exactly
    4 epsilon [(sigma/r)^12 - (sigma/r)^6], epsilon = epsilon_K * kB."""
    from ase import units
    from chaord.realize.ase_backend import ASEBackend
    table = mol_dialect.threshold("classical_pair_potentials")
    sig, eps_K = table["Ar"]["sigma_A"], table["Ar"]["epsilon_K"]
    r = 3.8
    box = 10.0
    pos = np.array([[1.0, 1.0, 1.0], [1.0 + r, 1.0, 1.0]])
    backend = ASEBackend([box] * 3, ["Ar", "Ar"], "classical", dialect=mol_dialect,
                         rc=6.0, width=0.0)
    u_ref = 4 * eps_K * units.kB * ((sig / r) ** 12 - (sig / r) ** 6)
    assert backend.energy(pos) == pytest.approx(u_ref, rel=1e-10)
    f = backend.forces(pos)
    # symmetric pair force, zero on the perpendicular axes
    assert f[0, 0] == pytest.approx(-f[1, 0], abs=1e-12)
    assert np.abs(f[:, 1:]).max() == pytest.approx(0.0, abs=1e-12)
    assert f[0, 0] < 0  # r < 2^(1/6) sigma: the pair repels


def test_lj_fluid_forces_are_exact_derivatives(mol_dialect):
    """The Ar + N2 site-wise LJ kernel (Lorentz-Berthelot cross terms,
    intramolecular pair excluded) must be the exact gradient of its energy:
    finite differences over every atom and axis."""
    from chaord.build.molecules import pack_molecules
    frame = pack_molecules({"Ar": 8, "N2": 10}, [9.0] * 3,
                           np.random.default_rng(4), mol_dialect)
    from chaord.realize.ase_backend import ASEBackend
    backend = ASEBackend(frame.cell_diag, frame.symbols, "classical",
                         dialect=mol_dialect, rc=4.5, width=0.0)
    pos = frame.pos
    f0 = backend.forces(pos)
    h = 1e-6
    for i in range(len(pos)):
        for axis in range(3):
            p = pos.copy()
            p[i, axis] += h
            ep = backend.energy(p)
            p = pos.copy()
            p[i, axis] -= h
            em = backend.energy(p)
            num = (ep - em) / (2 * h)
            assert num == pytest.approx(-f0[i, axis], rel=2e-5, abs=1e-7), (i, axis)


def test_water_plus_lj_gas_is_an_error(mol_dialect):
    """No silent fall-back: the classical backend realizes water (TIP4P +
    Joung-Cheatham ions) or an LJ molecular fluid, and mixing the two
    families is a static error naming the offending species."""
    from chaord.build import build_program
    from chaord.lang.errors import ChaordError
    text = _N2_PROGRAM.replace("molecules N2 80",
                               "molecules H2O 20 Ar 10").replace(
        "state density 0.349 g/cm3", "state density 0.118 g/cm3")
    with pytest.raises(ChaordError, match="no ASE classical potential"):
        build_program(parse_text(text), mol_dialect,
                      rng=np.random.default_rng(1), physics=True)


def test_fluid_rebuild_is_deterministic(mol_dialect):
    """Rule 10: same program + seed + backend -> identical coordinates, for
    both the monatomic and the two-site classical realization."""
    from chaord.build import build_program
    for text in (_AR_PROGRAM, _N2_PROGRAM):
        a = build_program(parse_text(text), mol_dialect,
                          rng=np.random.default_rng(7), physics=True, md_steps=40)
        b = build_program(parse_text(text), mol_dialect,
                          rng=np.random.default_rng(7), physics=True, md_steps=40)
        assert np.array_equal(a.pos, b.pos), text.splitlines()[3]


# -------------------------------------------- A5 regression: nacl_aq rebuild time

_NACL_FRAME = ROOT / "bench" / "reference" / "nacl_aq" / "frame_0.npz"


def test_nacl_kernel_fits_acceptance_budget(mol_dialect):
    """A5 regression, reproduced (fast form): the nacl_aq rebuild timed out
    because one force evaluation of the 1640-atom solution cost ~0.25 s, and
    the protocol runs fire + fast + relax = 820 of them (~205 s against the
    150 s per-case acceptance budget). Instrumented bound: measured per-call
    cost times the protocol's own step count must fit the budget."""
    import time as _time
    from chaord.realize.ase_backend import ASEBackend
    f = read_frame(_NACL_FRAME)
    backend = ASEBackend(f.cell_diag, f.symbols, "classical", dialect=mol_dialect)
    r = np.mod(f.pos, f.cell_diag)
    backend.build(r)                       # warm up caches
    n = 3
    t0 = _time.perf_counter()
    for i in range(n):
        backend.forces(r + 1e-6 * i)
    per_call = (_time.perf_counter() - t0) / n
    steps = sum(int(backend.md[k]) for k in
                ("fire_steps", "fast_steps", "relax_steps"))
    est = per_call * steps
    assert est < 150.0, (
        f"force evaluation costs {per_call * 1000:.0f} ms; x{steps} protocol "
        f"steps = {est:.0f} s, over the 150 s acceptance budget")


@pytest.mark.slow
def test_nacl_aq_full_rebuild_under_budget(mol_dialect):
    """A5 regression, full form: the whole physics rebuild of nacl_aq (RSA
    packing + FIRE + Langevin, the exact acceptance path) must finish in
    under 130 s of the 150 s per-case acceptance budget (the molecular
    dialect's ase_md protocol sizes the relax stage to that budget:
    ~125 s measured; the 130 s gate leaves the child-process startup and
    packing margin, coordinator authorization 2026-09-29)."""
    import time as _time
    from chaord.build import build_program
    from chaord.lift import lift_frame
    f = read_frame(_NACL_FRAME)
    prog = lift_frame(f, mol_dialect)
    t0 = _time.perf_counter()
    rebuilt = build_program(prog, mol_dialect, rng=np.random.default_rng(7),
                            physics=True)
    dt = _time.perf_counter() - t0
    assert len(rebuilt) == 1640
    assert molecule_census(rebuilt, mol_dialect) == {
        "H2O": 540, "Na": 10, "Cl": 10}
    assert dt < 130.0, f"nacl_aq rebuild took {dt:.0f} s (budget 130 s)"


@pytest.mark.slow
def test_nacl_aq_rebuild_is_deterministic(mol_dialect):
    """Rule 10 at solution scale: same program + seed -> identical
    coordinates from the vectorized solution backend."""
    from chaord.build import build_program
    from chaord.lift import lift_frame
    f = read_frame(_NACL_FRAME)
    prog = lift_frame(f, mol_dialect)
    a = build_program(prog, mol_dialect, rng=np.random.default_rng(7),
                      physics=True, md_steps=30)
    b = build_program(prog, mol_dialect, rng=np.random.default_rng(7),
                      physics=True, md_steps=30)
    assert np.array_equal(a.pos, b.pos)
