# -*- coding: utf-8 -*-
"""Write the R13 reservoir-filling LAMMPS run inputs (Review 13, T1-T2;
r13/PLAN.md). Extends confirmatory_inputs.py's verified conventions with
the ONE difference D55 orders: the composition is the corrected open-tube
uptake (EC 125 / EMC 58 / 24 LiPF6 per 98.4 A tube; PLAN.md section 1) --
everything else (protocol, seeds plan shape, dumps, thermostat) is the
committed Z3 protocol verbatim.

Jobs (18 = 3 fields x 6 builds), name r13_e<fieldmV>_b<buildK>.

Seed plan (PLAN.md section 5): velocity-create + thermalization pairs per
BUILD (20291150+K; ramp 20291200+20K+1/+2, equil +3/+4 -> identical
ladder-t=0 state per build); ladder pair per JOB
(20291400 + field_mv*100 + 10*K, +1/+2).

Every input accepts the smoke self-test:  lmp -var smoke 1 -in in.lammps.

STANDALONE: does not import chaord.
"""
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import params as P  # noqa: E402  (parent dir; standalone module, no chaord)

HERE = os.path.dirname(os.path.abspath(__file__))          # .../r13
JOBS = os.path.join(HERE, "jobs")
DATA = os.path.join(HERE, "data")

# ---- protocol constants (identical to the committed Z3 protocol) -----------
FIELDS_V_PER_A = (0.0, 0.01, 0.02)
BUILDS = tuple(f"r13_b{k}" for k in range(1, 7))
RAMP_PS = 100.0
EQUIL_PS = 900.0
SETTLE_PS = 1000.0
BLOCK_PS = 6000.0
N_BLOCKS = 2
R13_COMPOSITION_16 = dict(n_EC=125, n_EMC=58, n_LiPF6=24)
ATOMS = 4872
DOF = 6 * (R13_COMPOSITION_16["n_EC"] + R13_COMPOSITION_16["n_EMC"]
           + R13_COMPOSITION_16["n_LiPF6"]) + 3 * R13_COMPOSITION_16["n_LiPF6"] - 3

CLUSTER_ATOMNS_PER_MIN = 26754.0 / 188.0     # measured, 8 ranks (Stage B)


def field_tag(field):
    return "e%d" % round(field * 1000)


def seeds(build_k, field_mv):
    tb = 20291200 + 20 * build_k
    lb = 20291400 + field_mv * 100 + 10 * build_k
    return (20291150 + build_k, tb + 1, tb + 2, tb + 3, tb + 4,
            lb + 1, lb + 2)


def pair_coeff_lines():
    out = []
    for t in sorted(P.ATOM_TYPES):
        _, _, sig, eps, _, _src = P.ATOM_TYPES[t]
        out.append(f"pair_coeff      {t} {t} {eps:.6f} {sig:.4f}")
    return "\n".join(out) + "\n\n"


HEADER = """\
# LAMMPS input: {name}  (R13 reservoir-filling test, D55/PLAN.md: field {fieldtag_desc})
# Engine: LAMMPS 3 Mar 2020 (cluster module lammps/3Mar20-icc.2022, ime001;
#   8-rank MPI). ALL 18 R13 jobs run on this ONE build (T1).
#   Local smoke validation only: 22 Jul 2025 Update 4, lmp -var smoke 1.
# Model: pilot model (owner_inputs.md D37 note): OPLS-AA-based rigid
#   molecules from Nutakki 2023 Tables II-III, Werder sp2-C tube carbons,
#   Lorentz-Berthelot mixing (recorded deviation), full +-1 ion charges.
# System: r13_{bk} -- armchair (16,16) CNT, 98.4 A (ncell 40), packed pore
#   at the CORRECTED open-tube uptake: 125 EC / 58 EMC / 24 LiPF6 (Q2 per-A
#   fc8bec2 x 98.4; salt neutralized at the rounded mean of the measured
#   Li+ 26.3 / PF6- 20.9 -- the imbalance is a Q2 finding, not the build;
#   PLAN.md sections 1).
# Thermostat: ONE Langevin scheme for every mobile atom, damping 1 ps:
#   fix rigid/nve molecule langevin (NON-distributed variant per f9cae3f)
#   + fix nve + fix langevin (Li+); no Nose-Hoover.
# Protocol (the committed Z3 protocol, r13/PLAN.md section 4 verbatim):
#   fresh packed build -> {ramp:g} ps 10->300 K ramp (zero field) +
#   {equil:g} ps zero-field equilibration [per-build seeds: the build's
#   three field jobs reach the IDENTICAL ladder-t=0 state] -> field on at
#   ladder t = 0, {settle:g} ps settle (dump every 10 ps) -> 2 x {block:g} ps
#   production blocks dumped every 1 ps (block A = 0-{block:g} ps,
#   block B = {block:g}-{block2:g} ps) -> write_data final.data.
#   13 ns ladder + 1 ns thermalization = 14 ns. Block 6 ns = 2 x tau_M
#   (cnt16 @0.01 V/A) = 2 x 3.01 ns (Stage A).
# Seeds (PLAN.md section 5): velocity create {vseed}; ramp {rb}/{rl};
#   equil {eb}/{el} (per-build); ladder {lb1}/{lb2} (per-job).
# Dumps: every 1 ps (settle: every 10 ps), dump_modify format float
#   %.10g, with image flags. Thermo: every 1 ps, flushed.
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
thermo_modify   flush yes

velocity        mobile create 10.0 {vseed} mom yes rot yes dist gaussian

# --- stage 1: {ramp:g} ps ramp 10 -> 300 K, zero field (per-build seeds)
fix             thb bodies rigid/nve molecule langevin 10.0 300.0 {damp:g} {rb}
fix             thl li nve
fix             thla li langevin 10.0 300.0 {damp:g} {rl}
run             ${{nramp}}

# --- stage 2: {equil:g} ps equilibration at 300 K, zero field
unfix           thb
unfix           thl
unfix           thla
fix             thb bodies rigid/nve molecule langevin 300.0 300.0 {damp:g} {eb}
fix             thl li nve
fix             thla li langevin 300.0 300.0 {damp:g} {el}
run             ${{nequil}}

# --- ladder protocol starts here (Z3 protocol verbatim):
#     {efield}
{efield_fix}\
# velocities continue from the thermalized state (no re-draw)
fix             thb bodies rigid/nve molecule langevin 300.0 300.0 {damp:g} {lb1}
fix             thl li nve
fix             thla li langevin 300.0 300.0 {damp:g} {lb2}

# --- stage 3: {settle:g} ps settle (field already on), dump every 10 ps
{settle_dump}run             ${{nsettle}}
undump          ed

# --- stage 4: {nblocks} x {block:g} ps production blocks, dump every 1 ps
reset_timestep  0
{prod_dump}run             ${{nblockA}}
run             ${{nblockB}}

write_data      final.data nocoeff
"""

DUMP_CMD = """\
dump            {did} all custom {every} {fname} id mol type q x y z ix iy iz
dump_modify     {did} sort id format float %.10g
"""


def r13_input(name, field, build_k):
    (vseed, rb, rl, eb, el, lb1, lb2) = seeds(build_k, round(field * 1000))
    ramp = int(RAMP_PS * 1000 / P.TIMESTEP_FS)
    equil = int(EQUIL_PS * 1000 / P.TIMESTEP_FS)
    settle = int(SETTLE_PS * 1000 / P.TIMESTEP_FS)
    block = int(BLOCK_PS * 1000 / P.TIMESTEP_FS)
    if field > 0.0:
        efield_txt = ("external field along +z switched ON at ladder t = 0 "
                      "(V/A, real units)")
        efield_fix = (f"fix             efield mobile efield 0.0 0.0 "
                      f"{field}\n")
    else:
        efield_txt = ("field 0: the null-field ladder point (identical "
                      "protocol; the\n#     switch-on transient is absent "
                      "by construction)")
        efield_fix = ""
    smoke = (f'variable nramp equal 500" "variable nequil equal 500" '
             f'"variable nsettle equal 500" "variable nblockA equal 500" '
             f'"variable nblockB equal 500')
    real = (f"variable nramp equal {ramp}\" \"variable nequil equal "
            f"{equil}\" \"variable nsettle equal {settle}\" "
            f"\"variable nblockA equal {block}\" \"variable nblockB equal "
            f"{block}")
    txt = HEADER.format(
        name=name, bk=f"r13_b{build_k}", fieldtag_desc=field_tag(field),
        ramp=RAMP_PS, equil=EQUIL_PS, settle=SETTLE_PS, block=BLOCK_PS,
        block2=2 * BLOCK_PS, nblocks=N_BLOCKS,
        extra=("# R13 pre-registration: the D39 analysis call and the 0.01 "
               "V/A\n# decision rule are fixed in r13/PLAN.md (written "
               "before any job ran).\n"),
        cutoff=P.CUTOFF_A, pppm_acc=f"{P.PPPM_REL_ACC:g}",
        pair_coeffs=pair_coeff_lines(), dof=DOF, dt=P.TIMESTEP_FS,
        damp=P.STAGEA["langevin_damp_fs"],
        smoke_vars=smoke, real_vars=real,
        thermo=int(P.STAGEA["thermo_every_ps"] * 1000 / P.TIMESTEP_FS),
        vseed=vseed, rb=rb, rl=rl, eb=eb, el=el, lb1=lb1, lb2=lb2,
        efield=efield_txt, efield_fix=efield_fix,
        settle_dump=DUMP_CMD.format(did="ed", every=10000,
                                    fname="equil.dump"),
        prod_dump=DUMP_CMD.format(did="pd", every=1000, fname="prod.dump"),
    )
    return txt + "\n"


def main():
    index = []
    for field in FIELDS_V_PER_A:
        fm = round(field * 1000)
        for bk, build in enumerate(BUILDS, start=1):
            name = f"r13_{field_tag(field)}_b{bk}"
            vseed, rb, rl, eb, el, lb1, lb2 = seeds(bk, fm)
            d = os.path.join(JOBS, name)
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "in.lammps"), "w",
                      encoding="utf-8", newline="\n") as f:
                f.write(r13_input(name, field, bk))
            shutil.copyfile(os.path.join(DATA, build + ".data"),
                            os.path.join(d, "system.data"))
            est_min = (ATOMS * (RAMP_PS + EQUIL_PS + SETTLE_PS
                                + N_BLOCKS * BLOCK_PS) / 1000.0
                       / CLUSTER_ATOMNS_PER_MIN)
            index.append(dict(
                name=name, kind="r13_reservoir", tube=16, tube_len_A=98.4,
                build=build, build_k=bk, build_seed=20291101 + bk - 1,
                field=field, field_mv=fm,
                source=f"r13/data/{build}.data", atoms=ATOMS, dof=DOF,
                composition=dict(R13_COMPOSITION_16),
                ramp_ps=RAMP_PS, equil_ps=EQUIL_PS, settle_ps=SETTLE_PS,
                block_ps=BLOCK_PS, n_blocks=N_BLOCKS,
                steps_total=RAMP_PS + EQUIL_PS + SETTLE_PS
                + N_BLOCKS * BLOCK_PS,
                seeds=dict(velocity=vseed, ramp_bodies=rb, ramp_li=rl,
                           equil_bodies=eb, equil_li=el,
                           ladder_bodies=lb1, ladder_li=lb2),
                est_cluster_hours=round(est_min / 60.0, 1),
            ))
    with open(os.path.join(HERE, "jobs_index.json"), "w",
              encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    total_h = sum(j["est_cluster_hours"] for j in index)
    print(f"[r13-inputs] wrote {len(index)} job inputs under {JOBS}")
    print(f"[r13-inputs] estimated 8-rank cluster wall per job: "
          f"{index[0]['est_cluster_hours']} h; aggregate {total_h:.0f} h")


if __name__ == "__main__":
    main()
