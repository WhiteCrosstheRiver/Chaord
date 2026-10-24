"""Single source of truth for the X7 Kob-Andersen rate-convergence runs.

CHAORD-FREE BY RULE (D23 / AGENTS.md): reference-data generators must not
import chaord.  Every constant below is cross-checked against the repo's own
records, cited inline:

  potential  src/chaord/dialects/lj_mixtures.yaml (kob_andersen block),
             bench/reference/generate_reference.py (KA_EPSILON/KA_SIGMA/
             KA_RC_FACTOR) and bench/reference/ka_glass/provenance.json
             (identical numbers); published source W. Kob & H. C. Andersen,
             Phys. Rev. E 51, 4626 (1995), Table I (80:20, rho* = 1.2).
  geometry   N = 2000 (1600 A + 400 B), rho* = 1.2, fcc 8x8x8 start in the
             target box with 48 random deletions, random 80:20 species
             assignment -- bench/reference/ka_glass/provenance.json protocol
             description, verbatim construction.
  dt / dt-unit convention  dt* = 0.005 tau/step (provenance.json
             units.dt_reduced; glass.yaml md.relax_dt).
  melt/anneal counts  glass.yaml D7 block at the reference size
             (glass_protocol_ref_n = 2000 -> scale factor 1.0):
             glass_melt_T 2.0 / glass_melt_steps 2000 / glass_anneal_T 0.1 /
             glass_anneal_steps 4000.  NOTE the bench reference itself used a
             20000-step melt (the frozen W7 protocol); the X7 order (docs/
             reviews/open_items_v3.md X7 + the stream brief) pins the melt/
             anneal to the glass.yaml D7 block scaled at N=2000.
  quench counts  exact from the ordered rates: quench_steps =
             (T_melt - T_end)/rate/dt  (20000 / 200000 / 2000000 steps).
  sampling  5 frames per quench at 800-step stride, written AFTER the anneal
             (one stride of dynamics between the anneal end and frame 0) --
             bench ka_glass provenance steps convention
             (steps.anneal_equilibration 4000, sampling_stride 800,
             frames_per_quench 5; stored frames at 44800..48000 for the
             20000/20000 run).
  thermostat  D19(c) / X5 (open_items_v3.md item 2; glass.yaml ramp_gamma_*):
             Langevin gamma* = max(relax_gamma 0.5,
                                   10 / stage_tau)  [tau-multiple floor]
             and, on a ramp,  max(..., 500 x |dT/dt|)  [tracking floor];
             the anneal keeps the 10/tau_stage floor.  This is a DELIBERATE,
             documented deviation from the bench reference (which ran
             gamma* = 0.5 everywhere and lags: X5 measured kinetic T* 0.280
             at the 0.1 bath at quench end on the reference protocol).
  LAMMPS mapping  units lj (epsilon_AA = sigma_AA = m = 1), pair_style
             lj/cut with PER-PAIR cutoffs 2.5 sigma_ab (2.5 / 2.0 / 2.2
             sigma_AA) and pair_modify shift yes == the repo's per-pair
             cut-and-energy-shifted convention (forces plain truncated).
"""
from __future__ import annotations

import numpy as np

# LAMMPS executable (stream brief; anaconda distribution build,
# stable_22Jul2025_update4, serial per-process, run N jobs in parallel)
LMP = r"C:/ProgramData/anaconda3/Scripts/lmp.exe"

# --- potential (lj_mixtures.yaml kob_andersen / KA_* in generate_reference.py)
EPS = {"AA": 1.0, "AB": 1.5, "BB": 0.5}
SIG = {"AA": 1.0, "AB": 0.8, "BB": 0.88}
RC_FACTOR = 2.5                       # r_c,ab = 2.5 sigma_ab, per pair
RC = {k: RC_FACTOR * v for k, v in SIG.items()}     # 2.5 / 2.0 / 2.2
MASS = {"A": 1.0, "B": 1.0}

# --- state point
N_ATOMS, N_B = 2000, 400              # 1600 A + 400 B (80:20)
RHO_STAR = 1.2
L_BOX = (N_ATOMS / RHO_STAR) ** (1.0 / 3.0)

# --- protocol
DT = 0.005                            # tau/step (provenance dt_reduced)
T_MELT, T_END = 2.0, 0.1
MELT_STEPS = 2000                     # glass.yaml glass_melt_steps @ N=2000
ANNEAL_STEPS = 4000                   # glass.yaml glass_anneal_steps @ N=2000
STRIDE_STEPS = 800                    # bench ka_glass sampling_stride
N_FRAMES = 5                          # bench ka_glass frames_per_quench
RATES = [0.019, 0.0019, 0.00019]      # X7 order, T*/tau
RATE_TAGS = ["r0", "r1", "r2"]        # fast, mid, slow (1 decade apart)
N_SEEDS = 10                          # X7 order: 10 independent quenches/rate

# D19(c) thermostat gammas (see module docstring for the formula and cites)
RELAX_GAMMA = 0.5                     # glass.yaml md.relax_gamma
RAMP_GAMMA_TAU_MULT = 10.0            # glass.yaml ramp_gamma_tau_multiple
RAMP_GAMMA_RATE_MULT = 500.0          # glass.yaml ramp_gamma_rate_multiple


def stage_gamma(stage: str, rate: float | None = None) -> float:
    """D19(c) Langevin gamma* for one stage (glass.yaml rule, see docstring)."""
    if stage == "melt":
        tau_s = MELT_STEPS * DT
        return max(RELAX_GAMMA, RAMP_GAMMA_TAU_MULT / tau_s)
    if stage == "quench":
        assert rate is not None
        tau_s = (T_MELT - T_END) / rate          # stage duration in tau
        return max(RELAX_GAMMA, RAMP_GAMMA_TAU_MULT / tau_s,
                   RAMP_GAMMA_RATE_MULT * rate)
    if stage == "anneal":
        tau_s = ANNEAL_STEPS * DT
        return max(RELAX_GAMMA, RAMP_GAMMA_TAU_MULT / tau_s)
    if stage == "sample":
        # the bench reference sampled at the anneal's own gamma* = 0.5
        # (provenance protocol: "5 frames at 800-step intervals" under the
        # same Langevin bath); keep it -- a 10/tau_stride reading of the
        # rule would be a silent protocol change
        return max(RELAX_GAMMA, RAMP_GAMMA_TAU_MULT / (ANNEAL_STEPS * DT))
    raise ValueError(stage)


def quench_steps(rate: float) -> int:
    """Exact integer step count of the linear ramp (the X7 order formula)."""
    ns = (T_MELT - T_END) / rate / DT
    ni = int(round(ns))
    assert abs(ns - ni) < 1e-6, (rate, ns)
    return ni


def seeds(rate_idx: int, q: int) -> dict:
    """Deterministic seed set for one run (AGENTS.md: seed all randomness).

    base = 20261025 (stream start date) + 100*rate_idx + q: distinct across
    all 30 runs; sub-seeds x10+{1..6} split construction vs MD streams.
    """
    base = 20261025 + 100 * rate_idx + q
    return {
        "delete": base * 10 + 1,      # 48 fcc-site deletions
        "species": base * 10 + 2,     # 80:20 assignment
        "velocity": base * 10 + 3,    # LAMMPS velocity create
        "langevin_melt": base * 10 + 4,
        "langevin_quench": base * 10 + 5,
        "langevin_anneal": base * 10 + 6,   # also the sampling stage
    }


def build_start(seed_del: int, seed_spec: int):
    """fcc 8x8x8 start in the TARGET box, 48 random deletions, 80:20 species.

    Returns (positions (N,3) in [0,L), is_b (N,) bool, L).  Construction
    mirrors bench/reference/generate_reference.py:case_ka_glass verbatim
    (2048 sites at rho* x 2048/2000, delete 48 -> exactly rho*=1.2).
    """
    rng_del = np.random.default_rng(seed_del)
    rng_spec = np.random.default_rng(seed_spec)
    a0 = L_BOX / 8.0
    basis = np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.5],
                      [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]) * a0
    cells = np.array([[i, j, k] for i in range(8) for j in range(8)
                      for k in range(8)]) * a0
    sites = (cells[:, None, :] + basis[None, :, :]).reshape(-1, 3)
    n_sites = len(sites)                          # 2048
    keep = np.ones(n_sites, bool)
    keep[rng_del.choice(n_sites, n_sites - N_ATOMS, replace=False)] = False
    pos = sites[keep]
    is_b = np.zeros(N_ATOMS, bool)
    is_b[rng_spec.choice(N_ATOMS, N_B, replace=False)] = True
    return pos, is_b, L_BOX


def run_id(rate_idx: int, q: int) -> str:
    return f"{RATE_TAGS[rate_idx]}_q{q:02d}"


def protocol_record(rate: float) -> dict:
    qs = quench_steps(rate)
    gq = stage_gamma("quench", rate)
    return {
        "rate_Tstar_per_tau": rate,
        "melt_steps": MELT_STEPS, "melt_T": T_MELT,
        "quench_steps": qs,
        "quench_tau": (T_MELT - T_END) / rate,
        "anneal_steps": ANNEAL_STEPS, "anneal_T": T_END,
        "sampling_stride": STRIDE_STEPS, "frames_per_quench": N_FRAMES,
        "dt_tau_per_step": DT,
        "gamma_melt": stage_gamma("melt"),
        "gamma_quench": gq,
        "gamma_anneal": stage_gamma("anneal"),
        "gamma_sample": stage_gamma("sample"),
        "total_md_steps": MELT_STEPS + qs + ANNEAL_STEPS
                          + N_FRAMES * STRIDE_STEPS,
        "frame_steps": [MELT_STEPS + qs + ANNEAL_STEPS
                        + (k + 1) * STRIDE_STEPS for k in range(N_FRAMES)],
    }
