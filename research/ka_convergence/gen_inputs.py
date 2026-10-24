"""Generate LAMMPS inputs for the X7 KA rate-convergence runs (chaord-free).

Usage (from research/ka_convergence/):
    python gen_inputs.py                # all 30 runs (3 rates x 10 seeds)
    python gen_inputs.py r0             # one rate
    python gen_inputs.py r0_q03         # one run

Writes per run under runs/<id>/: data.<id> (start config), in.<id> (LAMMPS
script), and appends inputs_manifest.json.  The protocol is ka_params.py;
see that module's docstring for the record citations.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import ka_params as K

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
LMP = r"C:/ProgramData/anaconda3/Scripts/lmp.exe"


def write_data(path: Path, pos, is_b, L):
    types = np.where(is_b, 2, 1)
    with open(path, "w", newline="\n") as f:
        f.write(f"LAMMPS data file KA 80:20 start (X7, research/ka_convergence) "
                f"N={len(pos)} rho*={K.RHO_STAR} L={float(L):.10f}\n\n")
        f.write(f"{len(pos)} atoms\n2 atom types\n\n")
        for d, l in zip("xyz", [l for l in np.broadcast_to(np.asarray(L, float), (3,))]):
            f.write(f"0.0 {l:.10f} {d}lo {d}hi\n")
        f.write("\nMasses\n\n1 1.0\n2 1.0\n\nAtoms # atomic\n\n")
        for i, (p, t) in enumerate(zip(pos, types), 1):
            f.write(f"{i} {t} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}\n")


IN_TEMPLATE = """\
# {rid}: Kob-Andersen 80:20 LJ glass, X7 rate sweep (research/ka_convergence)
# rate {rate} T*/tau | seed base {seedbase} | params: ka_params.py (cited there)
units           lj
dimension       3
boundary        p p p
atom_style      atomic
read_data       data.{rid}

# KA potential (lj_mixtures.yaml kob_andersen == bench provenance):
# per-pair eps/sigma, each pair cut at 2.5 sigma_ab, energy-shifted at its
# own cutoff (pair_modify shift yes); forces plain truncated, the repo
# convention.  LAMMPS damp = 1/gamma*.
pair_style      lj/cut 2.5
pair_modify     shift yes
pair_coeff      1 1 {eAA} {sAA} {rcAA}
pair_coeff      1 2 {eAB} {sAB} {rcAB}
pair_coeff      2 2 {eBB} {sBB} {rcBB}

neighbor        0.3 bin
neigh_modify    delay 0 every 1 check yes
timestep        {dt}

thermo          {thermo}
thermo_style    custom step temp pe press etotal
thermo_modify   flush yes

velocity        all create {tmelt} {vseed} mom yes rot no dist gaussian

# ---- melt: T*={tmelt} for {melt} steps ({melttau} tau), gamma*={gmelt} (D19c: 10/tau_stage)
fix             int all nve
fix             bathm all langevin {tmelt} {tmelt} {dmelt} {smelt} zero yes
run             {melt}
print           "STAGE_END melt T*=$(temp:%.4f) PE=$(pe:%.8f)"
unfix           bathm

# ---- quench: linear {tmelt} -> {tend} over {qsteps} steps (rate {rate} T*/tau), gamma*={gq}
fix             bathq all langevin {tmelt} {tend} {dq} {sq} zero yes
run             {qsteps}
print           "STAGE_END quench T*=$(temp:%.4f) PE=$(pe:%.8f)"
unfix           bathq

# ---- anneal: T*={tend} for {anneal} steps, gamma*={ga} (10/tau_stage floor)
fix             batha all langevin {tend} {tend} {da} {sa} zero yes
run             {anneal}
print           "STAGE_END anneal T*=$(temp:%.4f) PE=$(pe:%.8f)"

# ---- sampling: {nframes} frames at {stride}-step stride (bench ka_glass convention)
{sample_block}
print           "STAGE_END sample T*=$(temp:%.4f) PE=$(pe:%.8f)"
unfix           batha
unfix           int

# ---- inherent structure (D20): FIRE minimisation of the final frame
min_style       fire
min_modify      integrator eulerimplicit
minimize        1e-10 1e-10 100000 200000
print           "IS_RESULT T*=$(temp:%.6f) PE=$(pe:%.10f) F=$(fnorm:%.6e) step=$(step)"
write_dump      all custom is_frame.lammpstrj id type x y z modify sort id
"""


def sample_block() -> str:
    lines = []
    for k in range(K.N_FRAMES):
        lines.append(f"run             {K.STRIDE_STEPS}")
        lines.append(f'print           "FRAME {k} PE=$(pe:%.8f)"')
        lines.append(f"write_dump      all custom frame_{k}.lammpstrj "
                     f"id type x y z modify sort id")
    return "\n".join(lines)


def gen_one(rate_idx: int, q: int) -> dict:
    rid = K.run_id(rate_idx, q)
    rate = K.RATES[rate_idx]
    sd = K.seeds(rate_idx, q)
    prot = K.protocol_record(rate)
    pos, is_b, L = K.build_start(sd["delete"], sd["species"])
    d = RUNS / rid
    d.mkdir(parents=True, exist_ok=True)
    write_data(d / f"data.{rid}", pos, is_b, L)
    in_txt = IN_TEMPLATE.format(
        rid=rid, rate=rate, seedbase=20261025 + 100 * rate_idx + q,
        eAA=K.EPS["AA"], sAA=K.SIG["AA"], rcAA=K.RC["AA"],
        eAB=K.EPS["AB"], sAB=K.SIG["AB"], rcAB=K.RC["AB"],
        eBB=K.EPS["BB"], sBB=K.SIG["BB"], rcBB=K.RC["BB"],
        dt=K.DT, thermo=max(1, prot["quench_steps"] // 400),
        tmelt=K.T_MELT, tend=K.T_END,
        melt=K.MELT_STEPS, melttau=K.MELT_STEPS * K.DT,
        gmelt=prot["gamma_melt"], dmelt=1.0 / prot["gamma_melt"],
        smelt=sd["langevin_melt"], vseed=sd["velocity"],
        qsteps=prot["quench_steps"], gq=prot["gamma_quench"],
        dq=1.0 / prot["gamma_quench"], sq=sd["langevin_quench"],
        anneal=K.ANNEAL_STEPS, ga=prot["gamma_anneal"],
        da=1.0 / prot["gamma_anneal"], sa=sd["langevin_anneal"],
        nframes=K.N_FRAMES, stride=K.STRIDE_STEPS,
        sample_block=sample_block())
    (d / f"in.{rid}").write_text(in_txt, newline="\n")
    return {
        "run_id": rid, "rate_idx": rate_idx, "seed_idx": q, "rate": rate,
        "seed_base": 20261025 + 100 * rate_idx + q, "seeds": sd,
        "protocol": prot, "n_atoms": K.N_ATOMS, "n_B": K.N_B,
        "rho_star": K.RHO_STAR, "box_L": L,
        "lammps": LMP,
        "files": [f"data.{rid}", f"in.{rid}"],
    }


def main(argv):
    want = argv[1] if len(argv) > 1 else "all"
    made = []
    for r in range(len(K.RATES)):
        for q in range(K.N_SEEDS):
            rid = K.run_id(r, q)
            if want in ("all", K.RATE_TAGS[r], rid):
                made.append(gen_one(r, q))
    man_path = RUNS / "inputs_manifest.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else {
        "generator": "research/ka_convergence/gen_inputs.py",
        "params_module": "research/ka_convergence/ka_params.py",
        "records": {
            "potential": "src/chaord/dialects/lj_mixtures.yaml kob_andersen; "
                         "bench/reference/generate_reference.py KA_*; "
                         "bench/reference/ka_glass/provenance.json",
            "melt_anneal": "src/chaord/dialects/glass.yaml D7 block at "
                           "glass_protocol_ref_n=2000 (scale 1.0)",
            "thermostat": "D19(c)/X5: glass.yaml ramp_gamma_tau_multiple 10, "
                          "ramp_gamma_rate_multiple 500, relax_gamma 0.5",
            "sampling": "bench/reference/ka_glass/provenance.json steps "
                        "(anneal_equilibration 4000, stride 800, 5 frames)",
            "order": "docs/reviews/open_items_v3.md X7 item 1",
        },
        "runs": {}}
    for m in made:
        man["runs"][m["run_id"]] = m
    man_path.write_text(json.dumps(man, indent=1))
    print(f"generated {len(made)} run input sets -> {man_path}")
    for m in made:
        p = m["protocol"]
        print(f"  {m['run_id']}: rate {m['rate']} gamma_q {p['gamma_quench']} "
              f"quench {p['quench_steps']} steps, total {p['total_md_steps']}")


if __name__ == "__main__":
    main(sys.argv)
