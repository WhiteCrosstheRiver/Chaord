"""Generate the Chaord reference data for disordered systems (bench/reference).

Six cases, each >= 5 decorrelated equilibrium frames, produced with ASE as an
independent MD engine and published potentials:

  lj_liquid          Lennard-Jones liquid          rho*=0.85, T*=0.72
  lj_glass           Lennard-Jones glass           melt T*=2.0 -> quench -> anneal
  lj_solid_liquid    LJ solid/liquid interface     fcc bottom half + melted top
  water_tip4p        rigid TIP4P water             rho=0.997 g/cm3, 300 K
  nacl_aq            1 M NaCl in rigid SPC/E       Joung-Cheatham ions, Wolf/DSF
  cu_solid_liquid    fcc Cu solid/liquid interface FBD-1986 EAM (NIST Cu_u3)

Frames are stored as frame_<k>.npz with arrays r (N,3), L (3,) and symbols
(U8); every case directory also holds provenance.json recording engine,
potential (name, citation, parameters), unit mapping, protocol, seeds, the
sampling step of every frame, and the sanity targets that
check_sanity.py enforces.

Generator code MUST NOT import chaord (circular-validation ban, AGENTS.md).
Run:  python bench/reference/generate_reference.py [--out DIR] [--only CASE]

Unit mapping for the LJ cases (verified below and recorded in every LJ
provenance): epsilon = 1 eV, sigma = 1 A, mass = 1 amu make the ASE unit
system identical to LJ reduced units, with the time unit
tau = sqrt(m sigma^2 / epsilon) = 10.180506 fs; T* = kB T / epsilon, so
T* = 0.72 is T = 8355.4 K and dt* = 0.005 is dt = 0.0509 fs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import ase
import scipy
from ase import Atoms, units
from ase.build import bulk
from ase.calculators.lj import LennardJones
from ase.calculators.tip4p import angleHOH as TIP4P_ANGLE
from ase.calculators.tip4p import rOH as TIP4P_ROH
from ase.constraints import FixAtoms, FixBondLengths
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import Stationary, thermalize_momenta
from ase.optimize import FIRE

sys.path.insert(0, str(Path(__file__).parent))
from fast_calculators import (JCSPCEWolf, FastEAM, FastTIP4P)  # noqa: E402
import ref_common as rc                                                   # noqa: E402

HERE = Path(__file__).parent
POT_DIR = HERE / "potentials"
CU_EAM = POT_DIR / "Cu_u3.eam"

N_FRAMES = 5
WATER_MASS_AMU = 18.01528
KB = units.kB                      # eV/K
AMU_PER_A3_PER_G_CM3 = 1.66053906660


# ------------------------------------------------------------- provenance --

def engine_block(integrator="ase.md.langevin.Langevin",
                 minimizer=None) -> dict:
    b = {"name": "ASE", "version": ase.__version__,
         "python": platform.python_version(), "numpy": np.__version__,
         "scipy": scipy.__version__, "platform": platform.platform(),
         "integrator": integrator}
    if minimizer:
        b["minimizer"] = minimizer
    return b


def lj_units_block(temperatures_star: dict) -> dict:
    """LJ reduced units <-> ASE units mapping, verified from constants."""
    tau_si = float(np.sqrt(1.66053906660e-27 * 1e-20 / 1.602176634e-19) * 1e15)
    tau_ase = float(1.0 / units.fs)
    assert abs(tau_si - tau_ase) < 1e-9 * tau_ase, "unit mapping violated"
    temps_K = {k: round(v / KB, 3) for k, v in temperatures_star.items()}
    return {
        "mapping": ("epsilon = 1 eV, sigma = 1 A, mass = 1 amu: the ASE unit "
                    "system (A, eV, amu) coincides with LJ reduced units; "
                    "tau = sqrt(m sigma^2/epsilon) is the ASE time unit"),
        "tau_fs": round(tau_ase, 6),
        "tau_fs_from_SI_constants": round(tau_si, 6),
        "kB_eV_per_K": float(KB),
        "Tstar_to_K": "T_K = T* * epsilon / kB",
        "temperatures_K": temps_K,
        "dt_reduced": 0.005,
        "dt_fs": round(0.005 * tau_ase, 6),
    }


def write_case(out: Path, case: str, frames, steps, provenance: dict,
               wall_s: float):
    """frames: list of (positions, L, symbols) sampled at equilibrium."""
    d = out / case
    d.mkdir(parents=True, exist_ok=True)
    records = []
    for k, (pos, L, symbols) in enumerate(frames):
        f = d / f"frame_{k}.npz"
        rc.save_frame(f, pos, L, symbols)
        records.append({"file": f.name, "step": int(steps[k])})
    provenance["case"] = case
    provenance["wall_clock_s"] = round(wall_s, 1)
    provenance["frames"] = records
    (d / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=False), encoding="utf-8")


# ------------------------------------------------------------------ setup --

def lj_fcc(nx, ny, nz, rho_star):
    """fcc lattice at reduced density rho_star; symbols X, masses 1 amu."""
    a0 = (4.0 / rho_star) ** (1.0 / 3.0)
    at = bulk('X', 'fcc', a=a0, cubic=True).repeat((nx, ny, nz))
    at.set_masses(np.full(len(at), 1.0))
    return at, a0


def run_langevin(atoms, steps, T_K, dt, friction, rng, frozen=None,
                 keep_constraints=False, label=""):
    """Langevin NVT stage; clears constraints unless told to keep them."""
    t0 = time.time()
    if frozen is not None:
        atoms.set_constraint(FixAtoms(mask=frozen))
    elif not keep_constraints:
        atoms.set_constraint()
    dyn = Langevin(atoms, dt, temperature_K=T_K, friction=friction,
                   rng=rng, fixcm=False)
    dyn.run(steps)
    print(f"    {label}: {steps} steps in {time.time()-t0:.0f}s "
          f"(T now {atoms.get_temperature():.0f} K)")


def thermalize(atoms, T_K, rng):
    thermalize_momenta(atoms, temperature_K=T_K, rng=rng)
    Stationary(atoms)


def water_bonds(nmol):
    return [(3 * i + j, 3 * i + (j + 1) % 3) for i in range(nmol)
            for j in range(3)]


def make_water_molecules(nmol, L, r_oh, angle, rng, grid=None):
    """O sites on a cubic grid subset (or random), random rigid orientations;
    returns positions (nmol*3, 3) in OHH order."""
    if grid is None:
        O = rng.uniform(0, L, (nmol, 3))
    else:
        g = grid
        xyz = (np.array(np.meshgrid(*[np.arange(g)] * 3, indexing='ij'))
               .reshape(3, -1).T * (L / g) + L / (2 * g))
        O = xyz[rng.choice(len(xyz), nmol, replace=False)]
    th = np.radians(angle / 2)
    pos = np.zeros((nmol * 3, 3))
    for m in range(nmol):
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        v = rng.normal(size=3)
        v -= u * (v @ u)
        v /= np.linalg.norm(v)
        pos[3 * m] = O[m]
        pos[3 * m + 1] = O[m] + r_oh * (np.cos(th) * u + np.sin(th) * v)
        pos[3 * m + 2] = O[m] + r_oh * (np.cos(th) * u - np.sin(th) * v)
    return pos


def relax_fire(atoms, fmax, steps, label):
    t0 = time.time()
    opt = FIRE(atoms, logfile=None)
    opt.run(fmax=fmax, steps=steps)
    print(f"    FIRE {label}: {opt.get_number_of_steps()} steps to fmax<={fmax}"
          f" in {time.time()-t0:.0f}s")


def _interface_expand(atoms, frozen, s, Lz_old):
    """Stretch the mobile (upper) half about the interface plane z=Lz/2."""
    zmid = Lz_old / 2.0
    pos = atoms.positions
    upper = ~frozen
    zu = zmid + np.mod(pos[upper, 2] - zmid, Lz_old - zmid)
    pos[upper, 2] = zmid + (zu - zmid) * s
    Lz_new = zmid + (Lz_old - zmid) * s
    cell = atoms.cell.array.copy()
    cell[2, 2] = Lz_new
    atoms.set_cell(cell)
    return zmid, Lz_new


# ------------------------------------------------------------------ cases --

def case_lj_liquid(out: Path, seed: int):
    t0 = time.time()
    n, rho_star, t_star = 500, 0.85, 0.72
    T_K = t_star / KB
    atoms, a0 = lj_fcc(5, 5, 5, rho_star)
    L = np.array(atoms.cell.lengths())
    atoms.calc = LennardJones(epsilon=1.0, sigma=1.0, rc=2.5, smooth=False)
    thermalize(atoms, T_K, np.random.default_rng([seed, "vel", 0]))
    equil, stride = 2000, 1000
    run_langevin(atoms, equil, T_K, 0.005, 0.5,
                 np.random.default_rng([seed, "md", 0]), label="equil")
    frames, steps = [], []
    for k in range(N_FRAMES):
        run_langevin(atoms, stride, T_K, 0.005, 0.5,
                     np.random.default_rng([seed, "md", k + 1]),
                     label=f"sample {k}")
        steps.append(equil + (k + 1) * stride)
        frames.append((atoms.positions.copy(), L, ["X"] * n))
    prov = {
        "engine": engine_block(),
        "potential": {
            "name": "Lennard-Jones 12-6",
            "implementation": "ase.calculators.lj.LennardJones",
            "citation": ("Lennard-Jones potential as implemented in ASE "
                         "(https://ase-lib.org); original: J. E. Jones, "
                         "Proc. R. Soc. Lond. A 106, 463 (1924)"),
            "parameters": {"epsilon_eV": 1.0, "sigma_A": 1.0,
                           "rc_sigma": 2.5, "smooth": False,
                           "truncation": "energy-shifted at rc=2.5 sigma, "
                                         "forces truncated (ASE default)"},
        },
        "units": lj_units_block({"equilibrium": t_star}),
        "protocol": {
            "description": (
                "N=500 fcc start at rho*=0.85, Maxwell velocities at T*, "
                "Langevin NVT (gamma*=0.5/tau) equilibration for 2000 steps "
                "(10 tau), then 5 frames at 1000-step (5 tau) intervals; "
                "dt*=0.005"),
            "ensemble": "NVT (Langevin, ase.md.langevin, fixcm=False)",
            "steps": {"equilibration": equil, "sampling_stride": stride},
            "friction_per_tau": 0.5,
        },
        "seed": seed,
        "sanity": {
            "density": [{"region": "bulk", "target": rho_star,
                         "tolerance_pct": 2.0,
                         "note": "atom number density, sigma units"}],
            "min_pairs": [{"elements": ["X"], "floor": 0.8,
                           "note": "0.8 sigma hard core"}],
            "gr_peaks": [{"elements": ["X"], "window": [1.05, 1.12],
                          "rmax": 2.5,
                          "note": "LJ liquid at rho*=0.85, T*~0.7: literature "
                                  "first peak 1.05-1.12 sigma"}],
        },
    }
    write_case(out, "lj_liquid", frames, steps, prov, time.time() - t0)


def case_lj_glass(out: Path, seed: int):
    t0 = time.time()
    n, rho_star = 500, 0.85
    t_melt, t_end, t_anneal = 2.0, 0.01, 0.01
    atoms, a0 = lj_fcc(5, 5, 5, rho_star)
    L = np.array(atoms.cell.lengths())
    atoms.calc = LennardJones(epsilon=1.0, sigma=1.0, rc=2.5, smooth=False)
    thermalize(atoms, t_melt / KB, np.random.default_rng([seed, "vel", 0]))

    melt, quench, anneal_eq, stride = 1500, 3000, 1000, 500
    run_langevin(atoms, melt, t_melt / KB, 0.005, 0.5,
                 np.random.default_rng([seed, "melt", 0]), label="melt")
    # linear quench: lower the Langevin bath linearly every step
    t1 = time.time()
    atoms.set_constraint()
    rng_q = np.random.default_rng([seed, "quench", 0])
    for s in range(quench):
        T_s = t_melt + (t_end - t_melt) * (s + 1) / quench
        Langevin(atoms, 0.005, temperature_K=T_s / KB, friction=0.5,
                 rng=rng_q, fixcm=False).run(1)
    print(f"    quench: {quench} steps in {time.time()-t1:.0f}s")
    run_langevin(atoms, anneal_eq, t_anneal / KB, 0.005, 0.5,
                 np.random.default_rng([seed, "anneal", 0]), label="anneal")
    frames, steps = [], []
    step0 = melt + quench + anneal_eq
    for k in range(N_FRAMES):
        run_langevin(atoms, stride, t_anneal / KB, 0.005, 0.5,
                     np.random.default_rng([seed, "anneal", k + 1]),
                     label=f"sample {k}")
        steps.append(step0 + (k + 1) * stride)
        frames.append((atoms.positions.copy(), L, ["X"] * n))
    rate = (t_melt - t_end) / (quench * 0.005)
    prov = {
        "engine": engine_block(),
        "potential": {
            "name": "Lennard-Jones 12-6",
            "implementation": "ase.calculators.lj.LennardJones",
            "citation": ("Lennard-Jones potential as implemented in ASE; "
                         "original: J. E. Jones, Proc. R. Soc. Lond. A 106, "
                         "463 (1924)"),
            "parameters": {"epsilon_eV": 1.0, "sigma_A": 1.0,
                           "rc_sigma": 2.5, "smooth": False},
        },
        "units": lj_units_block({"melt": t_melt, "quench_end": t_end,
                                 "anneal": t_anneal}),
        "protocol": {
            "description": (
                "N=500 fcc start at rho*=0.85; melt at T*=2.0 (1500 steps); "
                "linear quench T*=2.0 -> 0.01 over 3000 steps (rate "
                f"{rate:.4f} T*/tau); anneal at T*=0.01 for 1000 steps; 5 "
                "frames at 500-step intervals of the anneal segment; "
                "dt*=0.005, Langevin gamma*=0.5"),
            "ensemble": "NVT (Langevin, fixcm=False)",
            "quench_rate_Tstar_per_tau": round(rate, 5),
            "steps": {"melt": melt, "quench": quench,
                      "anneal_equilibration": anneal_eq,
                      "sampling_stride": stride},
            "friction_per_tau": 0.5,
        },
        "seed": seed,
        "sanity": {
            "density": [{"region": "bulk", "target": rho_star,
                         "tolerance_pct": 2.0,
                         "note": "atom number density, sigma units"}],
            "min_pairs": [{"elements": ["X"], "floor": 0.8,
                           "note": "0.8 sigma hard core"}],
            "gr_peaks": [{"elements": ["X"], "window": [1.05, 1.16],
                          "rmax": 2.5,
                          "note": "LJ glass first peak ~1.1 sigma"}],
        },
    }
    write_case(out, "lj_glass", frames, steps, prov, time.time() - t0)


def case_lj_solid_liquid(out: Path, seed: int):
    t0 = time.time()
    rho_s, rho_l, t_melt, t_run = 0.96, 0.845, 2.0, 0.65
    atoms, a0 = lj_fcc(4, 6, 9, rho_s)
    n = len(atoms)
    L = np.array(atoms.cell.lengths())
    frozen = atoms.positions[:, 2] < L[2] / 2
    atoms.calc = LennardJones(epsilon=1.0, sigma=1.0, rc=2.5, smooth=False)
    thermalize(atoms, t_melt / KB, np.random.default_rng([seed, "vel", 0]))
    # stage 1: melt the upper half with the lower half frozen
    t1 = time.time()
    atoms.set_constraint(FixAtoms(mask=frozen))
    dyn = Langevin(atoms, 0.005, temperature_K=t_melt / KB, friction=0.5,
                   rng=np.random.default_rng([seed, "melt", 0]), fixcm=False)
    dyn.run(1500)
    print(f"    melt: 1500 steps in {time.time()-t1:.0f}s")
    # stage 2: expand the liquid half to rho_l along z (prototype protocol)
    atoms.set_constraint()
    zmid, Lz_new = _interface_expand(atoms, frozen, rho_s / rho_l, L[2])
    L = np.array(atoms.cell.lengths())
    # stage 3: free interface at T*=0.65
    thermalize(atoms, t_run / KB, np.random.default_rng([seed, "vel", 1]))
    run_langevin(atoms, 2000, t_run / KB, 0.005, 0.5,
                 np.random.default_rng([seed, "md", 0]), label="equil")
    frames, steps = [], []
    step0, stride = 1500 + 2000, 500
    for k in range(N_FRAMES):
        run_langevin(atoms, stride, t_run / KB, 0.005, 0.5,
                     np.random.default_rng([seed, "md", k + 1]),
                     label=f"sample {k}")
        steps.append(step0 + (k + 1) * stride)
        frames.append((atoms.positions.copy(), L, ["X"] * n))
    buf = 2.0     # sigma kept away from both interfaces (zmid and z=0/Lz)
    prov = {
        "engine": engine_block(),
        "potential": {
            "name": "Lennard-Jones 12-6",
            "implementation": "ase.calculators.lj.LennardJones",
            "citation": ("Lennard-Jones potential as implemented in ASE; "
                         "original: J. E. Jones, Proc. R. Soc. Lond. A 106, "
                         "463 (1924)"),
            "parameters": {"epsilon_eV": 1.0, "sigma_A": 1.0,
                           "rc_sigma": 2.5, "smooth": False},
        },
        "units": lj_units_block({"melt": t_melt, "coexistence": t_run}),
        "protocol": {
            "description": (
                "Chaord prototype protocol (prototype/make_snapshot.py) at "
                f"N=864: fcc(4x6x9) at rho*={rho_s} with the lower half "
                "frozen; melt the upper half at T*=2.0 (Langevin, 1500 "
                f"steps); stretch the liquid half along z to rho*={rho_l} "
                "about the interface plane; release all atoms, equilibrate "
                f"the whole cell at T*={t_run} (2000 steps) and sample 5 "
                "frames at 500-step (2.5 tau) intervals; dt*=0.005"),
            "geometry": {"a0_sigma": round(a0, 6),
                         "zmid_sigma": round(zmid, 4),
                         "Lz_sigma": round(Lz_new, 4), "buffer_sigma": buf},
            "steps": {"melt": 1500, "equilibration": 2000,
                      "sampling_stride": 500},
            "friction_per_tau": 0.5,
        },
        "seed": seed,
        "sanity": {
            "density": [
                {"region": "solid", "z": [buf, zmid - buf],
                 "target": rho_s, "tolerance_pct": 2.0,
                 "note": "atom number density, sigma units"},
                {"region": "liquid", "z": [zmid + buf, Lz_new - buf],
                 "target": rho_l, "tolerance_pct": 2.0,
                 "note": "atom number density, sigma units"},
            ],
            "min_pairs": [{"elements": ["X"], "floor": 0.8,
                           "note": "0.8 sigma hard core"}],
            "gr_peaks": [
                {"elements": ["X"], "region": "solid", "z": [buf, zmid - buf],
                 "window": [1.08, 1.20], "rmax": 2.5,
                 "note": "fcc nearest-neighbour distance a0/sqrt(2)=1.140"},
                {"elements": ["X"], "region": "liquid",
                 "z": [zmid + buf, Lz_new - buf],
                 "window": [1.03, 1.18], "rmax": 2.5,
                 "note": "broad liquid first peak ~1.1"},
            ],
        },
    }
    write_case(out, "lj_solid_liquid", frames, steps, prov, time.time() - t0)


def case_water_tip4p(out: Path, seed: int):
    t0 = time.time()
    nmol = 256
    rho_g_cc = 0.997
    vol = nmol * WATER_MASS_AMU / (rho_g_cc / AMU_PER_A3_PER_G_CM3)  # A^3
    L = vol ** (1.0 / 3.0)
    rng = np.random.default_rng([seed, "build"])
    pos = make_water_molecules(nmol, L, TIP4P_ROH, TIP4P_ANGLE, rng, grid=7)
    atoms = Atoms('OH2' * nmol, positions=pos, cell=[L, L, L], pbc=True)
    atoms.calc = FastTIP4P(rc=8.0, width=1.0)
    atoms.set_constraint(FixBondLengths(water_bonds(nmol)))
    relax_fire(atoms, fmax=0.05, steps=500, label="water")
    thermalize(atoms, 300.0, np.random.default_rng([seed, "vel"]))
    run_langevin(atoms, 1500, 350.0, 1 * units.fs, 0.05,
                 np.random.default_rng([seed, "hot", 0]),
                 keep_constraints=True, label="hot 350K")
    run_langevin(atoms, 1500, 300.0, 1 * units.fs, 0.05,
                 np.random.default_rng([seed, "equil", 0]),
                 keep_constraints=True, label="equil")
    frames, steps = [], []
    stride = 1000                      # 1 ps between frames
    step0 = 3000
    for k in range(N_FRAMES):
        run_langevin(atoms, stride, 300.0, 1 * units.fs, 0.05,
                     np.random.default_rng([seed, "samp", k]),
                     keep_constraints=True, label=f"sample {k}")
        steps.append(step0 + (k + 1) * stride)
        frames.append((atoms.positions.copy(), np.array([L, L, L]),
                       ["O", "H", "H"] * nmol))
    validation = FastTIP4P.validate_against_ase()
    prov = {
        "engine": engine_block(minimizer="ase.optimize.FIRE"),
        "potential": {
            "name": "TIP4P rigid water",
            "implementation": (
                "bench/reference/fast_calculators.py:FastTIP4P — vectorized "
                "evaluation of ase.calculators.tip4p.TIP4P (same parameters, "
                "virtual sites, O-O smooth truncation and force "
                "redistribution), validated against the ASE calculator"),
            "citation": ("W. L. Jorgensen, J. Chandrasekhar, J. D. Madura, "
                         "R. W. Impey, M. L. Klein, J. Chem. Phys. 79, 926 "
                         "(1983), doi:10.1063/1.445869"),
            "parameters": {
                "qH_e": 0.52, "sigma_O_A": 3.153578,
                "epsilon_O_eV": 0.00672324, "d_OM_A": 0.15,
                "rOH_A": 0.9572, "angleHOH_deg": 104.52,
                "rc_OO_A": 8.0, "switch_width_A": 1.0,
                "truncation": ("smooth switching function on the O-O center "
                               "distance (ASE tip4p scheme)"),
                "validation_vs_ase_tip4p": validation,
            },
        },
        "units": {"length": "A", "time": "fs", "temperature": "K",
                  "note": "physical units; rigid water constrained with "
                          "ase.constraints.FixBondLengths (O-H, O-H, H-H per "
                          "molecule) following the ASE water tutorial"},
        "protocol": {
            "description": (
                f"N={nmol} TIP4P waters at rho={rho_g_cc} g/cm3 "
                f"(L={L:.4f} A): O sites on a 7^3 grid subset with random "
                "rigid orientations; FIRE relaxation (fmax 0.05 eV/A); "
                "Langevin NVT dt=1 fs, friction 0.05 per ASE time unit: "
                "1.5 ps at 350 K (melt-in), 1.5 ps at 300 K, then 5 frames "
                "at 1 ps intervals at 300 K"),
            "ensemble": "NVT (Langevin, rigid constraints, fixcm=False)",
            "steps": {"hot_350K": 1500, "equil_300K": 1500,
                      "sampling_stride": 1000},
            "dt_fs": 1.0,
        },
        "seed": seed,
        "sanity": {
            "density": [{"region": "bulk", "target": 3 * nmol / vol,
                         "tolerance_pct": 2.0,
                         "note": f"atom number density; {rho_g_cc} g/cm3"}],
            "min_pairs": [
                {"elements": ["O", "O"], "floor": 2.4,
                 "note": "hard core for O-O"},
                {"elements": ["O", "H"], "floor": 1.5,
                 "exclude_intramolecular": True},
                {"elements": ["H", "H"], "floor": 1.5,
                 "exclude_intramolecular": True},
            ],
            "gr_peaks": [
                {"elements": ["O", "O"], "window": [2.75, 2.90], "rmax": 5.0,
                 "note": "TIP4P O-O first peak at 300 K"},
            ],
        },
    }
    write_case(out, "water_tip4p", frames, steps, prov, time.time() - t0)


def case_nacl_aq(out: Path, seed: int):
    t0 = time.time()
    nion = 10
    # final composition: each Na+ replaces one water molecule, each Cl-
    # replaces two (its site and the nearest neighbour site, so the big
    # anion starts with a ~3.2 A cavity instead of a 2.83 A grid hole)
    nw = 570 - nion - 2 * nion      # 540 waters
    rho_g_cc = 1.037                # 1 M aqueous NaCl at 25 C
    mass = nw * WATER_MASS_AMU + nion * (22.98976928 + 35.4532)
    vol = mass / (rho_g_cc / AMU_PER_A3_PER_G_CM3)
    L = vol ** (1.0 / 3.0)
    rc_wolf = round(min(9.0, 0.49 * L), 3)
    rng = np.random.default_rng([seed, "build"])
    nw_build = 570
    pos = make_water_molecules(nw_build, L, JCSPCEWolf.R_OH,
                               JCSPCEWolf.ANGLE_HOH, rng, grid=9)
    o_pos = pos[0::3].copy()
    active = np.ones(nw_build, bool)
    na_sites, cl_sites = [], []
    # Na+: replace single waters
    for _ in range(nion):
        cands = np.where(active)[0]
        k = int(rng.choice(cands))
        active[k] = False
        na_sites.append(o_pos[k])
    # Cl-: remove a water and its nearest neighbour, sit at the midpoint
    for _ in range(nion):
        cands = np.where(active)[0]
        k = int(rng.choice(cands))
        d = o_pos[cands] - o_pos[k]
        d -= L * np.round(d / L)
        order = np.argsort(np.linalg.norm(d, axis=1))
        m = int(cands[order[1]])          # nearest active neighbour
        active[k] = active[m] = False
        cl_sites.append(0.5 * (o_pos[k] + o_pos[m] + L * np.round((o_pos[k]
                                                                  - o_pos[m]) / L)))
    water_mols = [m for m in range(nw_build) if active[m]]
    keep = np.array([i for m in water_mols for i in (3 * m, 3 * m + 1,
                                                     3 * m + 2)])
    water_pos = pos[keep]
    symbols = (["O", "H", "H"] * len(water_mols) + ["Na"] * nion
               + ["Cl"] * nion)
    positions = np.vstack([water_pos, np.array(na_sites),
                           np.array(cl_sites)])
    atoms = Atoms(''.join(symbols), positions=positions,
                  cell=[L, L, L], pbc=True)
    atoms.calc = JCSPCEWolf(rc=rc_wolf, alpha=0.2)
    atoms.set_constraint(FixBondLengths(water_bonds(nw)))
    relax_fire(atoms, fmax=0.05, steps=800, label="solution")
    thermalize(atoms, 300.0, np.random.default_rng([seed, "vel"]))
    run_langevin(atoms, 1500, 350.0, 1 * units.fs, 0.05,
                 np.random.default_rng([seed, "hot", 0]),
                 keep_constraints=True, label="hot 350K")
    run_langevin(atoms, 2000, 300.0, 1 * units.fs, 0.05,
                 np.random.default_rng([seed, "equil", 0]),
                 keep_constraints=True, label="equil")
    frames, steps = [], []
    stride = 1000
    step0 = 3500
    for k in range(N_FRAMES):
        run_langevin(atoms, stride, 300.0, 1 * units.fs, 0.05,
                     np.random.default_rng([seed, "samp", k]),
                     keep_constraints=True, label=f"sample {k}")
        steps.append(step0 + (k + 1) * stride)
        frames.append((atoms.positions.copy(), np.array([L, L, L]), symbols))
    conc = nion / (nw / 55.508)     # mol/L equivalent
    selftest = JCSPCEWolf(rc=rc_wolf, alpha=0.2).selftest()
    prov = {
        "engine": engine_block(minimizer="ase.optimize.FIRE"),
        "potential": {
            "name": "SPC/E water + Joung-Cheatham Na+/Cl- (Wolf/DSF)",
            "implementation": (
                "bench/reference/fast_calculators.py:JCSPCEWolf — LJ "
                "(Lorentz-Berthelot, energy-shifted at rc) between O and "
                "ion sites plus damped-shifted-force (Wolf) electrostatics "
                "on all charged sites; intramolecular water pairs excluded"),
            "citation": {
                "water": ("H. J. C. Berendsen, J. R. Grigera, T. P. "
                          "Straatsma, J. Phys. Chem. 91, 6269 (1987), "
                          "SPC/E model"),
                "ions": ("I. S. Joung, T. E. Cheatham III, J. Phys. Chem. B "
                         "112, 9020 (2008), Table 3, SPC/E parameter set"),
                "electrostatics": ("C. J. Fennell, J. D. Gezelter, J. Chem. "
                                   "Phys. 124, 234104 (2006), damped "
                                   "shifted force method"),
            },
            "parameters": {
                "water": {"qO_e": JCSPCEWolf.Q_O, "qH_e": JCSPCEWolf.Q_H,
                          "sigma_O_A": JCSPCEWolf.SIGMA_O,
                          "epsilon_O_kcal_mol": JCSPCEWolf.EPS_O,
                          "rOH_A": JCSPCEWolf.R_OH,
                          "angleHOH_deg": JCSPCEWolf.ANGLE_HOH},
                "Na": {"sigma_A": JCSPCEWolf.SIGMA_NA,
                       "epsilon_kcal_mol": JCSPCEWolf.EPS_NA, "charge_e": 1},
                "Cl": {"sigma_A": JCSPCEWolf.SIGMA_CL,
                       "epsilon_kcal_mol": JCSPCEWolf.EPS_CL, "charge_e": -1},
                "mixing": "Lorentz-Berthelot",
                "wolf_dsf": {
                    "alpha_inv_A": 0.2, "rc_A": rc_wolf,
                    "formula": ("V(r) = q_i q_j k [erfc(a r)/r - "
                                "erfc(a rc)/rc + (r-rc)(erfc(a rc)/rc^2 + "
                                "(2a/sqrt(pi)) exp(-a^2 rc^2)/rc)]"),
                    "selftest": selftest},
                "k_eV_A_per_e2": round(units.Hartree * units.Bohr, 6),
            },
        },
        "units": {"length": "A", "time": "fs", "temperature": "K",
                  "note": "rigid SPC/E water on the same machinery as "
                          "water_tip4p; SPC/E is the solvent the "
                          "Joung-Cheatham ion parameters were fitted for"},
        "protocol": {
            "description": (
                f"{nw} rigid SPC/E waters + {nion} Na+ + {nion} Cl- "
                f"({conc:.2f} M), rho={rho_g_cc} g/cm3 (L={L:.4f} A): water "
                "O on a 9^3 grid subset with random orientations; each Na+ "
                "replaces one grid water, each Cl- replaces two (its site "
                "and the nearest neighbour, placed at the midpoint so the "
                "anion starts with a ~3.2 A cavity); FIRE (fmax 0.05 eV/A); "
                "Langevin NVT dt=1 fs, friction 0.05 per ASE time unit: "
                "1.5 ps at 350 K, 2 ps at 300 K, then 5 frames at 1 ps "
                "intervals at 300 K"),
            "ensemble": "NVT (Langevin, rigid water, fixcm=False)",
            "steps": {"hot_350K": 1500, "equil_300K": 2000,
                      "sampling_stride": 1000},
            "dt_fs": 1.0,
        },
        "seed": seed,
        "sanity": {
            "density": [{"region": "bulk", "target": len(atoms) / vol,
                         "tolerance_pct": 2.0,
                         "note": f"atom number density; {rho_g_cc} g/cm3"}],
            "min_pairs": [
                {"elements": ["O", "O"], "floor": 2.4,
                 "note": "0.8 x ~3 A O-O core"},
                {"elements": ["Na", "O"], "floor": 2.1,
                 "note": "0.8 x sigma_LB(Na,O) = 0.8 x 2.663"},
                {"elements": ["Cl", "O"], "floor": 2.8,
                 "note": "LJ wall ~0.72 sigma_LB(Cl,O); epsilon_Cl is tiny, "
                         "0.8 sigma sits inside the equilibrium shell"},
                {"elements": ["Na", "Na"], "floor": 1.7,
                 "note": "0.8 x sigma_Na"},
                {"elements": ["Cl", "Cl"], "floor": 3.4,
                 "note": "LJ wall ~0.7 sigma_Cl"},
                {"elements": ["Na", "Cl"], "floor": 2.6,
                 "note": "Coulomb attraction pulls the contact pair inside "
                         "sigma; floor clears the 10 kBT wall at 0.72 sigma"},
                {"elements": ["O", "H"], "floor": 1.5,
                 "exclude_intramolecular": True},
                {"elements": ["H", "H"], "floor": 1.5,
                 "exclude_intramolecular": True},
                {"elements": ["H", "Na"], "floor": 1.9},
                {"elements": ["H", "Cl"], "floor": 1.9},
            ],
            "gr_peaks": [
                {"elements": ["O", "O"], "window": [2.70, 2.95], "rmax": 5.0,
                 "note": "SPC/E O-O first peak"},
                {"elements": ["Cl", "O"], "window": [3.00, 3.40], "rmax": 5.5,
                 "note": "Joung-Cheatham report Cl-O ~3.19 A in SPC/E"},
                {"elements": ["Na", "O"], "window": [2.20, 2.50], "rmax": 4.5,
                 "note": "Joung-Cheatham report Na-O ~2.35 A in SPC/E"},
            ],
        },
    }
    write_case(out, "nacl_aq", frames, steps, prov, time.time() - t0)


def case_cu_solid_liquid(out: Path, seed: int):
    t0 = time.time()
    a0 = 3.615                      # FBD Cu lattice constant (potential file)
    rho_s = 4 * 63.546 / (a0 ** 3 * 0.6022140857)      # g/cm3
    rho_l = 7.96                    # liquid Cu at Tm (experiment)
    t_melt, t_eq = 1700.0, 0.8 * 1330.0
    atoms = bulk('Cu', 'fcc', a=a0, cubic=True).repeat((4, 6, 9))
    n = len(atoms)
    L = np.array(atoms.cell.lengths())
    frozen = atoms.positions[:, 2] < L[2] / 2
    validation = FastEAM.validate_against_ase(str(CU_EAM))
    atoms.calc = FastEAM(potential=str(CU_EAM))
    thermalize(atoms, t_melt, np.random.default_rng([seed, "vel", 0]))
    t1 = time.time()
    atoms.set_constraint(FixAtoms(mask=frozen))
    dyn = Langevin(atoms, 2 * units.fs, temperature_K=t_melt, friction=0.05,
                   rng=np.random.default_rng([seed, "melt", 0]), fixcm=False)
    dyn.run(1200)
    print(f"    melt: 1200 steps in {time.time()-t1:.0f}s")
    atoms.set_constraint()
    zmid, Lz_new = _interface_expand(atoms, frozen, rho_s / rho_l, L[2])
    L = np.array(atoms.cell.lengths())
    thermalize(atoms, t_eq, np.random.default_rng([seed, "vel", 1]))
    run_langevin(atoms, 2000, t_eq, 2 * units.fs, 0.05,
                 np.random.default_rng([seed, "md", 0]), label="equil")
    frames, steps = [], []
    step0, stride = 1200 + 2000, 400
    for k in range(N_FRAMES):
        run_langevin(atoms, stride, t_eq, 2 * units.fs, 0.05,
                     np.random.default_rng([seed, "md", k + 1]),
                     label=f"sample {k}")
        steps.append(step0 + (k + 1) * stride)
        frames.append((atoms.positions.copy(), L, ["Cu"] * n))
    buf = 5.1                       # ~2 nearest-neighbour distances
    sha = hashlib.sha256(CU_EAM.read_bytes()).hexdigest()
    prov = {
        "engine": engine_block(),
        "potential": {
            "name": "Foiles-Baskes-Daw 1986 Cu EAM (universal-3)",
            "implementation": (
                "bench/reference/fast_calculators.py:FastEAM — vectorized "
                "evaluation of the spline tables parsed by "
                "ase.calculators.eam.EAM from the published funcfl file "
                "(validated against the ASE calculator, numbers below)"),
            "citation": ("S. M. Foiles, M. I. Baskes, M. S. Daw, Phys. Rev. "
                         "B 33, 7983 (1986); Cu_u3.eam from the NIST "
                         "Interatomic Potentials Repository"),
            "url": ("https://www.ctcms.nist.gov/potentials/entry/"
                    "1986--Foiles-S-M-Baskes-M-I-Daw-M-S--Cu/"),
            "file": "potentials/Cu_u3.eam",
            "sha256": sha,
            "parameters": {"a0_A": 3.615, "cutoff_A": 4.95,
                           "format": "LAMMPS funcfl (single-element setfl)",
                           "validation_vs_ase_eam": validation},
        },
        "units": {"length": "A", "time": "fs", "temperature": "K",
                  "note": "Tm(FBD Cu) ~1330 K; equilibrium T = 0.8*Tm "
                          f"= {t_eq:.0f} K"},
        "protocol": {
            "description": (
                f"N={n} fcc Cu (4x6x9 cells, a0={a0} A, rho_s={rho_s:.3f} "
                "g/cm3) with the lower half frozen; melt the upper half at "
                f"{t_melt:.0f} K (Langevin, 1200 steps, dt=2 fs); stretch "
                f"the liquid half along z to rho_l={rho_l} g/cm3 about the "
                "interface plane; release all atoms, equilibrate at "
                f"{t_eq:.0f} K (2000 steps) and sample 5 frames at 400-step "
                "(0.8 ps) intervals"),
            "geometry": {"a0_A": a0, "zmid_A": round(zmid, 4),
                         "Lz_A": round(Lz_new, 4), "buffer_A": buf},
            "steps": {"melt": 1200, "equilibration": 2000,
                      "sampling_stride": 400},
            "dt_fs": 2.0, "friction_per_ASE_time_unit": 0.05,
        },
        "seed": seed,
        "sanity": {
            "density": [
                {"region": "solid", "z": [buf, zmid - buf],
                 "target": 4 / a0 ** 3, "tolerance_pct": 2.0,
                 "note": f"atom number density; {rho_s:.3f} g/cm3"},
                {"region": "liquid", "z": [zmid + buf, Lz_new - buf],
                 "target": 4 * rho_l / rho_s / a0 ** 3, "tolerance_pct": 2.0,
                 "note": f"atom number density; {rho_l} g/cm3"},
            ],
            "min_pairs": [{"elements": ["Cu"], "floor": 2.0,
                           "note": "hard core for Cu-Cu"}],
            "gr_peaks": [
                {"elements": ["Cu"], "region": "solid", "z": [buf, zmid - buf],
                 "window": [2.49, 2.62], "rmax": 4.5,
                 "note": "fcc a0/sqrt(2) = 2.556 A"},
                {"elements": ["Cu"], "region": "liquid",
                 "z": [zmid + buf, Lz_new - buf],
                 "window": [2.35, 2.62], "rmax": 4.5,
                 "note": "liquid/supercooled-liquid Cu first shell ~2.5 A"},
            ],
        },
    }
    write_case(out, "cu_solid_liquid", frames, steps, prov, time.time() - t0)


CASES = {
    "lj_liquid": case_lj_liquid,
    "lj_glass": case_lj_glass,
    "lj_solid_liquid": case_lj_solid_liquid,
    "water_tip4p": case_water_tip4p,
    "nacl_aq": case_nacl_aq,
    "cu_solid_liquid": case_cu_solid_liquid,
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--only", choices=sorted(CASES), action="append")
    ap.add_argument("--seed", type=int, default=20260928)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    todo = args.only or sorted(CASES)
    t_all = time.time()
    for name in todo:
        print(f"== {name}")
        CASES[name](out, args.seed)
    print(f"total wall clock: {time.time()-t_all:.0f}s")


if __name__ == "__main__":
    main()
