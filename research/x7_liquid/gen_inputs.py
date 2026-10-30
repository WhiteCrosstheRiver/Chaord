"""Generate LAMMPS inputs for the X7 item-2 liquid-floor runs (chaord-free).

Usage (from research/x7_liquid/):
    python gen_inputs.py            # all 6 runs (2 cases x 3 trajectories)

Writes per run under runs/<id>/: data.<id> (fcc start config), in.<id>
(LAMMPS script), and inputs_manifest.json.  Protocol: liq_params.py (cited
there); conventions validated bit-exact by validate_conventions.py (8/8).

The committed protocols are reproduced with NEW seeds only (the
lj_liquid_large second-trajectory precedent: "2 independent trajectories
(identical protocol, distinct seeds)"); the new trajectories extend the
sampling to 45 frames at the committed 1000-step (5 tau) stride.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import liq_params as K

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"


def write_data(path: Path, pos, L, case_idx: int):
    with open(path, "w", newline="\n") as f:
        f.write(f"LAMMPS data file LJ liquid start (X7 item 2, "
                f"research/x7_liquid) N={len(pos)} rho*={K.RHO_STAR} "
                f"L={float(L[0]):.10f}\n\n")
        f.write(f"{len(pos)} atoms\n1 atom types\n\n")
        for d, l in zip("xyz", np.broadcast_to(np.asarray(L, float), (3,))):
            f.write(f"0.0 {l:.10f} {d}lo {d}hi\n")
        f.write("\nMasses\n\n1 1.0\n\nAtoms # atomic\n\n")
        for i, p in enumerate(pos, 1):
            f.write(f"{i} 1 {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}\n")


IN_TEMPLATE = """\
# {rid}: single-species LJ liquid, X7 item 2 liquid floors
# (research/x7_liquid; case {case}, trajectory {traj} of {ntraj}, new seeds,
# protocol otherwise identical to the committed bench reference)
# params: liq_params.py (cited there); conventions validated bit-exact in
# validate_conventions.py (8/8 vs reports/noise_floors.json)
units           lj
dimension       3
boundary        p p p
atom_style      atomic
read_data       data.{rid}

# committed reference potential (bench/reference/{case}/provenance.json):
# LJ 12-6, eps=sigma=m=1, cut 2.5 sigma, energy-shifted at rc (pair_modify
# shift yes), forces truncated.  LAMMPS damp = 1/gamma* = {damp}.
pair_style      lj/cut {rc}
pair_modify     shift yes
pair_coeff      1 1 {eps} {sigma} {rc}

neighbor        0.3 bin
neigh_modify    delay 0 every 1 check yes
timestep        {dt}

thermo          1000
thermo_style    custom step temp pe press etotal
thermo_modify   flush yes

# == ASE thermalize(): Maxwell velocities at T*, COM momentum zeroed
velocity        all create {tstar} {vseed} mom yes rot no dist gaussian

# ---- equilibration: T*={tstar}, {equil} steps ({equiltau} tau), gamma*={gamma}
fix             int all nve
fix             bath all langevin {tstar} {tstar} {damp} {lseed} zero yes
run             {equil}
print           "STAGE_END equil T*=$(temp:%.4f) PE=$(pe:%.8f)"

# ---- sampling: {nframes} frames at {stride}-step ({stridetau} tau) stride
#      (committed protocol.steps.sampling_stride; 45 frames vs 10/5 committed)
{sample_block}
unfix           bath
unfix           int
print           "RUN_END T*=$(temp:%.4f) PE=$(pe:%.8f)"
"""


def sample_block() -> str:
    lines = []
    for k in range(K.N_FRAMES):
        lines.append(f"run             {K.STRIDE_STEPS}")
        lines.append(f'print           "FRAME {k} PE=$(pe:%.8f)"')
        lines.append(f"write_dump      all custom frame_{k}.lammpstrj "
                     f"id type x y z modify sort id")
    return "\n".join(lines)


def gen_one(case_idx: int, traj: int) -> dict:
    c = K.CASES[case_idx]
    rid = K.run_id(case_idx, traj)
    sd = K.seeds(case_idx, traj)
    prot = K.protocol_record(case_idx)
    nx, ny, nz = c["fcc"]
    pos, L = K.fcc_start(nx, ny, nz, K.RHO_STAR)
    d = RUNS / rid
    d.mkdir(parents=True, exist_ok=True)
    write_data(d / f"data.{rid}", pos, L, case_idx)
    in_txt = IN_TEMPLATE.format(
        rid=rid, case=c["name"], traj=traj, ntraj=K.N_TRAJ,
        eps=K.EPS, sigma=K.SIGMA, rc=K.RC, dt=K.DT,
        tstar=K.T_STAR, damp=K.LANGEVIN_DAMP, gamma=K.GAMMA,
        equil=c["equil_steps"], equiltau=c["equil_steps"] * K.DT,
        vseed=sd["velocity"], lseed=sd["langevin"],
        nframes=K.N_FRAMES, stride=K.STRIDE_STEPS,
        stridetau=K.STRIDE_STEPS * K.DT,
        sample_block=sample_block())
    (d / f"in.{rid}").write_text(in_txt, newline="\n")
    return {
        "run_id": rid, "case_idx": case_idx, "case": c["name"],
        "traj_idx": traj, "seeds": sd, "seed_base": K.seed_base(case_idx, traj),
        "protocol": prot, "n_atoms": c["n_atoms"],
        "lammps": K.LMP_CLUSTER,
        "files": [f"data.{rid}", f"in.{rid}"],
    }


def main(argv):
    made = []
    for ci in K.CASES:
        for t in range(K.N_TRAJ):
            made.append(gen_one(ci, t))
    man = {
        "generator": "research/x7_liquid/gen_inputs.py",
        "params_module": "research/x7_liquid/liq_params.py",
        "order": "docs/reviews/open_items_v3.md X7 item 2 (liquid floors)",
        "records": {
            "potential": "bench/reference/lj_liquid/provenance.json "
                         "(single-species LJ 12-6, eps=sigma=m=1, rc 2.5 "
                         "sigma energy-shifted); src/chaord/dialects/lj.yaml",
            "protocol": "per-case committed protocol, new seeds only "
                        "(bench/reference/lj_liquid{,_large}/provenance.json)",
            "lag": "LAG_MIN 4 frame indices = 20 tau (committed lj_liquid "
                   "frame-pair floor convention), applied to both cases; "
                   "measured lag-decay reported in analyze_floors output",
            "seeds": "base 20261101 + 10*case + traj, sub-seeds x10+{1,2} "
                     "(ka_params.py convention; collision-checked vs bench "
                     "20260928-29 and ka 202610251..202610546)",
        },
        "runs": {m["run_id"]: m for m in made},
    }
    (RUNS / "inputs_manifest.json").write_text(json.dumps(man, indent=1))
    print(f"generated {len(made)} run input sets -> "
          f"{RUNS / 'inputs_manifest.json'}")
    for m in made:
        p = m["protocol"]
        print(f"  {m['run_id']}: N={m['n_atoms']} equil {p['equil_steps']} "
              f"+ {p['n_frames']}x{p['stride_steps']} steps, seeds "
              f"{m['seeds']['velocity']}/{m['seeds']['langevin']}")


if __name__ == "__main__":
    main(sys.argv)
