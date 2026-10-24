# -*- coding: utf-8 -*-
"""
Write the Stage B LAMMPS run inputs (Review 10, Y7: tau_M-limited blocks).

Protocol owner: reports/research/stage_b_plan.md. Everything here is a
direct implementation of that plan; every number's reason is stated there.

Jobs (42 = 2 tubes x 7 fields x 3 runs), name sb_<tube>_e<fieldmV>_r<run>:

  Field ladder (D30 working-field context), V/A:
      0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2   ->  e0, e5, e10, e20, e50,
                                                e100, e200
  Start states: the DONE Stage A zero-field finals
      stageA/jobs/zf_cntN_b{1,2,3}/final.data
    (each is 100 ps 10->300 K ramp + 900 ps equilibration + 2 ns
    zero-field production - exactly the equilibrated zero-field starts
    Stage B needs). Run rK starts from build bK (b1, b2, b3 = independent
    builds; the mirrored b1m is the symmetry instrument, NOT counted as an
    independent start).
  Per job: field on at t = 0 (or stays 0 for e0), 1000 ps settle (dump
    every 10 ps), then 2 x 6000 ps production blocks dumped every 1 ps
    (blocks are split at 6000 ps in the analysis; the input is one
    continuous trajectory). 13000 ps total, write_data final.data.
  Block length 6 ns = 2 x tau_M(cnt16 @0.01 V/A) = 2 x 3.01 ns, the
    slower tube; for cnt10 (tau_M = 2.67 ns) a block is 2.25 tau_M.
    tau_M measured in Stage A (stageA/analysis/summary.json).

Physics, thermostat and I/O conventions are IDENTICAL to stageA_inputs.py
(same pilot model, fix rigid/nve molecule langevin + fix nve/fix langevin
Li+ with 1 ps damping, dt 1 fs, dumps every 1 ps with %.10g + image
flags, thermo every 1 ps). One thermostat seed pair per job, distinct
across all 42 jobs (formula below, all < 2^31).

Every input accepts the smoke self-test:  lmp -var smoke 1 -in in.lammps
runs 500-step versions of each stage.

STANDALONE: does not import chaord (house rule, AGENTS.md).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import params as P  # noqa: E402  (parent dir; standalone module, no chaord)

HERE = os.path.dirname(os.path.abspath(__file__))          # .../stageB
ROOT = os.path.dirname(HERE)                                # .../cnt_electrolyte
JOBS = os.path.join(HERE, "jobs")

# ---- protocol constants (reasons in stage_b_plan.md §2) -------------------
FIELD_LADDER_V_PER_A = (0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2)
RUN_BUILDS = {1: "b1", 2: "b2", 3: "b3"}   # run rK <- zf build bK
SETTLE_PS = 1000.0     # field-on settle before block A (10-ps dumps)
BLOCK_PS = 6000.0      # >= 2 tau_M (tau_M = 2.67 / 3.01 ns; stage_a.md)
N_BLOCKS = 2
# first tranche (owner order 2026-10-25): working field + first mid field,
# cnt10 first - results flow earliest into the D32 comparison
TRANCHE_1 = (10, (0.01, 0.05))

# measured serial throughput, atom*ps/min: median over the 8 wf/sf DONE
# files per tube (stageA/jobs/*/DONE, runs of 2026-10-17/18):
#   cnt10 (971 atoms):  41,398 - 42,529, median 42,077
#   cnt16 (2058 atoms): 21,348 - 22,774, median 22,398
MEASURED_THROUGHPUT = {10: 42077.0, 16: 22398.0}


def field_tag(field):
    """0.01 V/A -> 'e10' (field in mV/A); 0.005 -> 'e5'; 0.2 -> 'e200'."""
    return "e%d" % round(field * 1000)


def seeds(tube, field, run):
    """One (bodies, li) Langevin seed pair per job, distinct across all 42
    jobs and disjoint from every Stage A seed range (Stage A thermostat
    seeds live near 3.1e6 / 5.1e6; these live in 21.5-22.1e6). All < 2^31
    (LAMMPS random seeds are 32-bit ints)."""
    base = 20267000 + tube * 100000 + round(field * 1000) * 1000 + run * 10
    return base + 1, base + 2


HEADER = """\
# LAMMPS input: {name}  (Stage B / Y7: field ladder, tau_M-limited blocks)
# Engine: LAMMPS 22 Jul 2025 Update 4. Units: real (A, kcal/mol, fs, V/A).
# Model: pilot model (owner_inputs.md D37 note): OPLS-AA-based rigid
#   molecules from Nutakki 2023 Tables II-III, Werder sp2-C tube carbons,
#   Lorentz-Berthelot mixing (recorded deviation), full +-1 ion charges.
# Thermostat: ONE Langevin scheme for every mobile atom, damping 1 ps:
#   fix rigid/nve molecule langevin (multi-atom bodies) + fix nve +
#   fix langevin (Li+); no Nose-Hoover anywhere (same rule as Stage A).
#   NOTE rigid/nve (not the /small variant), same as Stage A (the local
#   conda build's fix rigid/nve/small molecule corrupted body data at
#   step 1 for 3 of 4 cnt16 builds - stageA_plan.md addendum 2026-10-16).
# Dumps: every 1 ps (settle: every 10 ps), dump_modify format float
#   %.10g, with image flags. Thermo: every 1 ps.
# Stage B protocol (reports/research/stage_b_plan.md):
#   start = {source} final.data (equilibrated zero-field state: 100 ps
#   10->300 K ramp + 900 ps equil + 2 ns zero-field production);
#   settle {settle_ps:g} ps (field on from t = 0), then {nblocks} x
#   {block_ps:g} ps production blocks (one continuous dumped trajectory;
#   the analysis splits blocks at {block_ps:g} ps). Block length
#   {block_ns:g} ns = 2 x tau_M(cnt16 @0.01 V/A) = 2 x 3.01 ns;
#   2.25 x tau_M for cnt10 (2.67 ns) - tau_M measured in Stage A
#   (reports/research/stage_a.md, stageA/analysis/summary.json).
{extra}#

units           real
atom_style      full
boundary        p p p

read_data       system.data

pair_style      lj/cut/coul/long {cutoff}
pair_modify     mix arithmetic tail no
kspace_style    pppm {pppm_acc}
{pair_coeffs}bond_style       harmonic
bond_coeff      1 0.0 1.0
special_bonds   lj/coul 0.0 0.0 0.0

neighbor        2.0 bin
neigh_modify    every 1 delay 0 check yes exclude molecule/intra all

group           tube type 1
group           mobile subtract all tube
group           li type 9
group           bodies subtract mobile li
fix             frz tube setforce 0.0 0.0 0.0

# honest mobile-subsystem temperature: 2*KE/(DOF*kB), real units
# (the default thermo temp is diluted by the frozen tube atoms)
compute         keb bodies ke
compute         keli li ke
variable        Tmob equal 2.0*(c_keb+c_keli)/({dof}*0.0019872041)

timestep        {dt}

# smoke self-test:  lmp -var smoke 1 -in in.lammps
variable        smoke index 0
if "${{smoke}} == 1" then "{smoke_vars}" &
else "{real_vars}"

thermo          {thermo}
thermo_style    custom step atoms v_Tmob temp pe etotal press vol
"""

DUMP_CMD = """\
dump            {did} all custom {every} {fname} id mol type q x y z ix iy iz
dump_modify     {did} sort id format float %.10g
"""


def pair_coeff_lines():
    out = []
    for t in sorted(P.ATOM_TYPES):
        _, _, sig, eps, _, _src = P.ATOM_TYPES[t]
        out.append(f"pair_coeff      {t} {t} {eps:.6f} {sig:.4f}")
    return "\n".join(out) + "\n\n"


def sb_input(name, tube, field, run, source, dof, sb, sl):
    settle = int(SETTLE_PS * 1000 / P.TIMESTEP_FS)
    block = int(BLOCK_PS * 1000 / P.TIMESTEP_FS)
    if field > 0.0:
        efield = (f"# external field along +z switched ON at t = 0 "
                  f"(V/A, real units)\n"
                  f"fix             efield mobile efield 0.0 0.0 {field}\n")
        note = (f"ladder field {field} V/A, run {run} of 3, from {source} "
                f"final state.")
    else:
        efield = ("# field 0: the null-field ladder point (identical "
                  "protocol; the\n"
                  "# switch-on transient is absent by construction)\n")
        note = (f"null-field control, run {run} of 3, from {source} "
                f"final state (continued zero-field sampling).")
    body = f"""\
{efield}\
# velocities continue from the read-in zero-field final state (no re-draw)
fix             thb bodies rigid/nve molecule langevin 300.0 300.0 {P.STAGEA['langevin_damp_fs']:g} {sb}
fix             thl li nve
fix             thla li langevin 300.0 300.0 {P.STAGEA['langevin_damp_fs']:g} {sl}

# long job: flush the log/screen buffers at every thermo output
thermo_modify    flush yes

# --- stage 1: {SETTLE_PS:g} ps settle (field already on), dump every 10 ps
{DUMP_CMD.format(did="ed", every=10000, fname="equil.dump")}run             ${{nsettle}}
undump          ed

# --- stage 2: {N_BLOCKS} x {BLOCK_PS:g} ps production blocks, dump every 1 ps
#     (one continuous trajectory; block A = 0-{BLOCK_PS:g} ps,
#      block B = {BLOCK_PS:g}-{2 * BLOCK_PS:g} ps, split in the analysis)
reset_timestep  0
{DUMP_CMD.format(did="pd", every=1000, fname="prod.dump")}run             ${{nblockA}}
run             ${{nblockB}}

write_data      final.data nocoeff
"""
    smoke = (f'variable nsettle equal 500" "variable nblockA equal 500" '
             f'"variable nblockB equal 500')
    real = (f"variable nsettle equal {settle}\" \"variable nblockA equal "
            f"{block}\" \"variable nblockB equal {block}")
    txt = HEADER.format(
        name=name, desc=note, source=source, settle_ps=SETTLE_PS,
        block_ps=BLOCK_PS, block_ns=BLOCK_PS / 1000.0, nblocks=N_BLOCKS,
        extra=f"# {note}\n",
        cutoff=P.CUTOFF_A, pppm_acc=f"{P.PPPM_REL_ACC:g}",
        pair_coeffs=pair_coeff_lines(), dof=dof, dt=P.TIMESTEP_FS,
        smoke_vars=smoke, real_vars=real,
        thermo=int(P.STAGEA["thermo_every_ps"] * 1000 / P.TIMESTEP_FS))
    return txt + "\n" + body


def main():
    index = []
    for tube in (10, 16):
        comp = P.COMPOSITION[tube]
        dof = (6 * (comp["n_EC"] + comp["n_EMC"] + comp["n_LiPF6"])
               + 3 * comp["n_LiPF6"] - 3)
        atoms = {10: 971, 16: 2058}[tube]
        for field in FIELD_LADDER_V_PER_A:
            for run, build in RUN_BUILDS.items():
                name = f"sb_cnt{tube}_{field_tag(field)}_r{run}"
                zf = f"zf_cnt{tube}_{build}"
                src = f"stageA/jobs/{zf}/final.data"
                sb_seed, sl_seed = seeds(tube, field, run)
                d = os.path.join(JOBS, name)
                os.makedirs(d, exist_ok=True)
                with open(os.path.join(d, "in.lammps"), "w") as f:
                    f.write(sb_input(name, tube, field, run, zf, dof,
                                     sb_seed, sl_seed))
                tranche = ((tube == TRANCHE_1[0] and field in TRANCHE_1[1])
                           or None)
                index.append(dict(
                    name=name, kind="sb", tube=tube, build=build, run=run,
                    field=field, field_mv=round(field * 1000),
                    source=src, depends=zf, atoms=atoms,
                    settle_ps=SETTLE_PS, block_ps=BLOCK_PS,
                    n_blocks=N_BLOCKS,
                    steps_ps=SETTLE_PS + N_BLOCKS * BLOCK_PS, dof=dof,
                    seeds=dict(bodies=sb_seed, li=sl_seed),
                    tranche=1 if tranche else None,
                    est_wall_min=round(
                        atoms * (SETTLE_PS + N_BLOCKS * BLOCK_PS)
                        / MEASURED_THROUGHPUT[tube], 1),
                ))
    with open(os.path.join(HERE, "jobs_index.json"), "w") as f:
        json.dump(index, f, indent=2)
    total = sum(j["est_wall_min"] for j in index)
    t1 = sum(j["est_wall_min"] for j in index if j["tranche"])
    print(f"[stageB-inputs] wrote {len(index)} job inputs under {JOBS}")
    print(f"[stageB-inputs] estimated serial wall: total {total:.0f} min "
          f"= {total / 1440:.1f} days; tranche 1 {t1:.0f} min "
          f"= {t1 / 60:.1f} h (throughput: measured per-tube medians "
          f"{MEASURED_THROUGHPUT})")


if __name__ == "__main__":
    main()
