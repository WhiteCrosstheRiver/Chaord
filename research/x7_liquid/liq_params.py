"""Single source of truth for the X7 item-2 liquid-floor runs (chaord-free).

CHAORD-FREE BY RULE (D23 / AGENTS.md): reference-data generators must not
import chaord.  Every constant below is cross-checked against the repo's own
records, cited inline.

Order: docs/reviews/open_items_v3.md X7 item 2 ("Liquid floors. At least 30
decorrelated frame pairs per liquid case (two or three independent
trajectories), with bootstrap 95% intervals on every floor. Today each floor
rests on 10-12 pairs, so its P90 is in effect the maximum."), unchanged by
open_items_v5.md (Review 11; S3: "the rest of X7 in idle time").

Cases (the liquid cases whose floors rest on 10-12 pairs, reports/
noise_floors.json): lj_liquid (avg floor 10 pairs) and lj_liquid_large
(avg floor 11 pairs; frame-pair floors 21 pairs each).  The molecular water/
NaCl cases keep their own floors (out of scope here); the CNT electrolyte
e0 floors are a separate concern (Stage B, Z3).

  potential  single-species LJ 12-6, epsilon = sigma = m = 1, rc = 2.5 sigma,
             energy-shifted at rc, forces truncated -- exactly the committed
             reference potential (bench/reference/lj_liquid/provenance.json
             potential.parameters; ase.calculators.lj.LennardJones rc=2.5
             smooth=False == LAMMPS lj/cut 2.5 + pair_modify shift yes, the
             validated ka_convergence mapping) and the lj dialect state point
             (src/chaord/dialects/lj.yaml: cn_cutoff 1.50, gr_rmax 2.5).
  geometry   lj_liquid: N = 500, fcc 5x5x5 at rho* = 0.85 -> L = 8.3788
             (a0 = (4/rho*)^(1/3) = 1.6758, generate_reference.py:lj_fcc;
             committed frame L = 8.37883606, checked).
             lj_liquid_large: N = 2048, fcc 8x8x8, same rho* -> L = 13.4061
             (committed frame L = 13.40613769; the provenance text's
             "L=13.51" is a typo in its prose -- the stored frames carry
             13.4061, and lj_fcc is what generated them).
  state      T* = 0.72 (provenance units.temperatures_K.equilibrium 8355.256 K
             = 0.72 epsilon/kB), NVT Langevin gamma* = 0.5/tau
             (friction_per_tau 0.5), dt* = 0.005 (units.dt_reduced).
  protocol   per-case committed protocol with NEW seeds only:
             lj_liquid  equil 2000 steps (10 tau), then frames at 1000-step
                        (5 tau) stride (provenance protocol.steps);
             lj_liquid_large  equil 4000 steps (20 tau; 2x lj_liquid because
                        the fcc start must melt homogeneously in the 4x
                        larger cell), same stride (case_lj_liquid_large).
             Both cases: 45 sampling frames per trajectory (vs 10 and 5
             committed) -- 225 tau of sampling per trajectory, enough for
             >= 30 decorrelated pairs many times over (X7 item 2) and for a
             bootstrap on every floor.
  sampling lag  LAG_MIN = 4 frame indices = 20 tau, the committed lj_liquid
             frame-pair floor convention ("lag >= 4, the decorrelated half",
             noise_floors.json note; frames 5 tau apart).  Applied uniformly
             to both new cases -- conservative vs lj_liquid_large's committed
             lag >= 2 (10 tau), which generate_reference.py justified with a
             measured lag-decay ("distances are flat in lag from 2.5 tau on",
             in.lj_liquid_large protocol note).  analyze_floors.py reports
             the measured lag-decay of the new trajectories next to this
             choice (distances at lags 1..8, so the choice is checked, not
             assumed).
  pair sets  frame-pair floor: cross-trajectory pairs with both frame
             indices >= LAG_MIN (the committed lj_liquid_large convention
             restricted cross pairs to "the decorrelated halves") POOLED
             with within-trajectory pairs at index lag >= LAG_MIN (the
             committed pooling; an equilibrated liquid carries no
             preparation memory -- cross sits +1%/+10% above within,
             measured 2026-09-30, noise_floors.json note).
             averaged floor: the committed group-size conventions, resampled
             over many independent blocks:
               lj_liquid  rotated disjoint 5v5 half-splits of 10-frame
                          blocks -- each trajectory cut into 4 disjoint
                          10-frame blocks (frames 0-9/10-19/20-29/30-39;
                          5 rotations x 2 orderings = 10 pairs per block),
                          plus position-matched block-vs-block pairs across
                          trajectories (the committed large-case
                          trajectory-vs-trajectory mean pair, one per
                          unordered trajectory pair per block position).
               lj_liquid_large  the committed mixed convention, resampled:
                          per 5-frame window, the 5 cyclic rotations of the
                          2v3 split x 2 orderings (the committed 10 pairs
                          per 5-frame trajectory) over 9 disjoint windows
                          per trajectory, plus position-matched 5v5
                          window-vs-window pairs across trajectories (the
                          committed single 5v5 cross pair).
  estimator  D15: floor = max(mean, P90) of the pair distances
             (reports/history/gate_a_prime_v1_2026-10-09.md D15 "kept";
             acceptance A5 effective floor: max(floor mean, P90)).
             P90 = np.percentile(values, 90), the repo's convention
             (research/ka_convergence/analyze.py p90 fields).
  bootstrap  95% CI on every floor: resample the pair list with replacement,
             NBOOT = 1000 draws (the stream brief for this item), D15 per
             draw, CI = np.percentile(draws, [2.5, 97.5]) -- the
             ka_convergence bootstrap_ci convention (RNG.integers index
             resampling + percentile ends), NBOOT 1000 per this item's brief
             (ka used 10000 on n=10 samples; here n >= 120 so 1000 draws
             resolve the 2.5% tail).  One generator, seed BOOT_SEED, floors
             evaluated in the fixed order recorded in the output JSON.
             Caveat stated in the report: pairs share frames (each frame
             enters many pairs), so the pair bootstrap is the brief's
             convention, not an independent-draw CI.
  LAMMPS mapping  units lj; per-pair coefficients, pair_modify shift yes;
             velocity create ... mom yes rot no dist gaussian == ASE
             MaxwellBoltzmannDistribution + Stationary (COM momentum zeroed,
             which is what thermalize() does); fix nve + fix langevin
             T T damp = 1/gamma* SEED zero yes -- the ka_convergence input
             template (research/ka_convergence/gen_inputs.py), whose V1
             check measured LAMMPS-vs-ASE potential energy agreement to
             3.5e-5 per atom on the glass protocol
             (research/ka_convergence/validation/report.json).  The langevin
             "zero yes" (bath force COM-neutral) vs ASE fixcm=False
             (uncorrected COM) difference concerns only centre-of-mass
             motion, which no floor observable sees (g(r) and cn are
             translation-invariant under PBC).
  seeds      NEW trajectory seed bases 20261101 + 10*case_idx + traj_idx
             (case 0 = lj_liquid, 1 = lj_liquid_large; traj 0..2), sub-seeds
             x10 + {1 velocity, 2 langevin}, ka_params.py seed convention.
             Collision-checked: distinct from the committed bench seeds
             (20260928, 20260929) and every ka_convergence seed base
             (20261025 + 100r + q in 20261025..20261054, actual LAMMPS seeds
             202610251..202610546); no overlap with 20261101x..202611032x.
"""
from __future__ import annotations

import numpy as np

# LAMMPS executables: cluster module build (stream brief: the ONLY working
# build on the cluster) and the local anaconda build (ka_params.py precedent)
# for smoke tests.
LMP_CLUSTER = "lmp"                    # after: module load lammps/3Mar20-icc.2022
LMP_LOCAL = r"C:/ProgramData/anaconda3/Scripts/lmp.exe"

# --- potential (bench/reference/lj_liquid/provenance.json; lj.yaml) ---------
EPS = 1.0
SIGMA = 1.0
RC = 2.5                               # x sigma, energy-shifted

# --- observables (dialect core+lj thresholds, lj.yaml lines 81-83/97) -------
GR_RMAX, GR_BINS, CN_CUT, GR_RMIN = 2.5, 60, 1.5, 0.8

# --- state point (both cases, provenance) ----------------------------------
RHO_STAR = 0.85
T_STAR = 0.72
GAMMA = 0.5                            # Langevin friction, 1/tau
DT = 0.005                             # tau/step
LANGEVIN_DAMP = 1.0 / GAMMA            # 2.0 tau (LAMMPS damp)

# --- per-case protocol ------------------------------------------------------
CASES = {
    0: {
        "tag": "lj500",                # run-id prefix
        "name": "lj_liquid",
        "n_atoms": 500,
        "fcc": (5, 5, 5),
        "equil_steps": 2000,           # 10 tau (provenance protocol.steps)
        "committed_pairs": {"frame_pair": 21, "avg": 10},
    },
    1: {
        "tag": "lj2048",               # run-id prefix
        "name": "lj_liquid_large",
        "n_atoms": 2048,
        "fcc": (8, 8, 8),
        "equil_steps": 4000,           # 20 tau (case_lj_liquid_large)
        "committed_pairs": {"frame_pair": 21, "avg": 11},
    },
}
STRIDE_STEPS = 1000                    # 5 tau (committed sampling_stride)
N_TRAJ = 3                             # X7 item 2: two or three -> three
N_FRAMES = 45                          # per trajectory, 225 tau of sampling
LAG_MIN = 4                            # frame indices; 20 tau (see docstring)
N_BLOCKS_AVG = 4                       # disjoint 10-frame blocks per trajectory
                                       # for the lj_liquid avg statistic
                                       # (45 = 4 x 10 + 5 unused tail frames)
WINDOW_AVG = 5                         # 5-frame windows for the large case
                                       # committed 2v3/5v5 statistic (9/traj)
NBOOT = 1000                           # bootstrap draws (stream brief)
BOOT_SEED = 20261101                   # recorded bootstrap generator seed

AVG_BLOCK_LEN = 10                     # 10-frame blocks (lj_liquid statistic)


def fcc_start(nx: int, ny: int, nz: int, rho: float):
    """fcc lattice at reduced density rho (generate_reference.py:lj_fcc).

    a0 = (4/rho)^(1/3); 4 atoms per cell; returns (positions (N,3), L).
    Same construction as the committed frames (verified in
    validate_conventions.py V0 against the stored L values).
    """
    a0 = (4.0 / rho) ** (1.0 / 3.0)
    basis = np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.5],
                      [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]) * a0
    cells = np.array([[i, j, k] for i in range(nx) for j in range(ny)
                      for k in range(nz)]) * a0
    pos = (cells[:, None, :] + basis[None, :, :]).reshape(-1, 3)
    L = np.array([nx, ny, nz], float) * a0
    return pos, L


def seed_base(case_idx: int, traj: int) -> int:
    return 20261101 + 10 * case_idx + traj


def seeds(case_idx: int, traj: int) -> dict:
    """Deterministic seed set for one trajectory (AGENTS.md rule 32)."""
    base = seed_base(case_idx, traj)
    return {
        "velocity": base * 10 + 1,     # velocity all create
        "langevin": base * 10 + 2,     # fix langevin (equil + sampling)
    }


def run_id(case_idx: int, traj: int) -> str:
    return f"{CASES[case_idx]['tag']}_t{traj:02d}"


def protocol_record(case_idx: int) -> dict:
    c = CASES[case_idx]
    nx, ny, nz = c["fcc"]
    _, L = fcc_start(nx, ny, nz, RHO_STAR)
    return {
        "case": c["name"], "n_atoms": c["n_atoms"], "rho_star": RHO_STAR,
        "box_L": float(L[0]), "T_star": T_STAR,
        "equil_steps": c["equil_steps"], "stride_steps": STRIDE_STEPS,
        "n_frames": N_FRAMES, "n_traj": N_TRAJ, "lag_min": LAG_MIN,
        "dt_tau_per_step": DT, "gamma_per_tau": GAMMA,
        "langevin_damp": LANGEVIN_DAMP,
        "sampling_tau": N_FRAMES * STRIDE_STEPS * DT,
        "frame_steps": [c["equil_steps"] + (k + 1) * STRIDE_STEPS
                        for k in range(N_FRAMES)],
        "total_md_steps": c["equil_steps"] + N_FRAMES * STRIDE_STEPS,
    }
