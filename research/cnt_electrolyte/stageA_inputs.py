# -*- coding: utf-8 -*-
"""
Write the Stage A LAMMPS run inputs (Review 10, work order Y6).

Jobs (34 total), all dumped every 1 ps with 10 significant digits and
image flags, thermo every 1 ps, dt 1 fs, ONE Langevin scheme (damping
1 ps) for every mobile atom:

  fill_cntN            open tube in a bulk bath, 50 ps ramp + 5 ns NPT
                       (1 atm, 300 K); fix nph ... dilate mobile is the
                       pure barostat (no thermostat) so the one-Langevin
                       rule survives the NPT.
  zf_cntN_bX           zero field: 100 ps ramp (10->300 K) + 900 ps
                       equilibration + 2 ns production; X in b1,b2,b3 and
                       b1m (mirrored copy of b1).
  wf_cntN_bX /         switch-on from the zf final state: field 0.01
  sf_cntN_bX           (D30 working field) / 0.2 V/A, 2 ns.
  off_cntN_bX          switch-off from the sf final state, field 0, 2 ns.

Each job directory stageA/jobs/<job>/ holds in.lammps, which reads
"system.data" from its own directory (the queue runner stages that file:
a copy of data/<build>.data for zf/fill jobs, or the dependency's
final.data for wf/sf/off). Paths inside the inputs are therefore all
relative -> cluster-portable; the lmp invocation is a queue-runner
concern (--lmp), never baked into the input.

Every input accepts a smoke self-test:  lmp -var smoke 1 -in in.lammps
runs 500-step versions of each stage (the "variable smoke index 0" line
is skipped when the variable already exists from the command line).

STANDALONE: does not import chaord.
Run AFTER stageA_build.py (reads data/stageA_build_manifest.json).
"""
import json
import os

import params as P

HERE = os.path.dirname(os.path.abspath(__file__))
JOBS = os.path.join(HERE, "stageA", "jobs")

# measured local serial throughput (pilot README: 2058 atoms x 550 ps in
# 51.7 min), used only for the honest cost column of the job index
ATOM_PS_PER_MIN = 22000.0

# thermostat seeds: derived from each build's velocity seed; distinct for
# every run, every fix; kept < 2^31 (LAMMPS random seeds are 32-bit ints -
# found on first smoke execution)
def _seeds(base, code):
    b = (base - 20260000) * 1000 + code * 100   # < 2^31 for our seeds
    return dict(bodies=b + 1, li=b + 2, bodies2=b + 3, li2=b + 4)


HEADER = """\
# LAMMPS input: {name}  (Stage A / Y6: {desc})
# Engine: LAMMPS 22 Jul 2025 Update 4. Units: real (A, kcal/mol, fs, V/A).
# Model: pilot model (owner_inputs.md D37 note): OPLS-AA-based rigid
#   molecules from Nutakki 2023 Tables II-III, Werder sp2-C tube carbons,
#   Lorentz-Berthelot mixing (recorded deviation), full +-1 ion charges.
# Thermostat: ONE Langevin scheme for every mobile atom, damping 1 ps:
#   fix rigid/nve molecule langevin (multi-atom bodies) + fix nve +
#   fix langevin (Li+); no Nose-Hoover anywhere (Y6 settings).
#   NOTE rigid/nve (not the /small variant): the local conda build's
#   fix rigid/nve/small molecule corrupted body data at the first
#   integration step for 3 of 4 cnt16 fresh builds (heap-layout
#   dependent; atoms teleported out of the PPPM brick at step 1 with
#   sane step-0 forces/velocities - see stageA_plan.md addendum
#   2026-10-16). Same Langevin scheme, same damping, deterministic
#   body bookkeeping.
# Dumps: every 1 ps, dump_modify format float %.10g, with image flags.
# Thermo: every 1 ps.
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
thermo_style    custom step atoms v_Tmob temp pe etotal press vol{dens_col}
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


def zf_input(name, build, field_note, dof, seeds, vseed, s):
    ramp = int(s["ramp_ps"] * 1000 / P.TIMESTEP_FS)
    equil = int(s["equil_ps"] * 1000 / P.TIMESTEP_FS)
    prod = int(s["prod_ps"] * 1000 / P.TIMESTEP_FS)
    body = f"""\
velocity        mobile create 10.0 {vseed} mom yes rot yes dist gaussian

# --- stage 1: ramp 10 -> 300 K over {s['ramp_ps']:g} ps (Langevin only)
fix             thb bodies rigid/nve molecule langevin 10.0 300.0 {s['langevin_damp_fs']:g} {seeds['bodies']}
fix             thl li nve
fix             thla li langevin 10.0 300.0 {s['langevin_damp_fs']:g} {seeds['li']}
run             ${{nramp}}

# --- stage 2: {s['equil_ps']:g} ps equilibration at 300 K
unfix           thb
unfix           thl
unfix           thla
fix             thb bodies rigid/nve molecule langevin 300.0 300.0 {s['langevin_damp_fs']:g} {seeds['bodies2']}
fix             thl li nve
fix             thla li langevin 300.0 300.0 {s['langevin_damp_fs']:g} {seeds['li2']}
{DUMP_CMD.format(did="ed", every=100000, fname="equil.dump")}
run             ${{nequil}}
undump          ed

# --- stage 3: {s['prod_ps']:g} ps production, dump every {s['dump_every_ps']:g} ps
reset_timestep  0
{DUMP_CMD.format(did="pd", every=1000, fname="prod.dump")}
run             ${{nprod}}

write_data      final.data nocoeff
"""
    smoke = (f"variable nramp equal 500\" \"variable nequil equal 500\" "
             f"\"variable nprod equal 500")
    real = (f"variable nramp equal {ramp}\" \"variable nequil equal {equil}\" "
            f"\"variable nprod equal {prod}")
    txt = HEADER.format(
        name=name, desc=f"zero field, build {build}. {field_note}",
        extra="", cutoff=P.CUTOFF_A, pppm_acc=f"{P.PPPM_REL_ACC:g}",
        pair_coeffs=pair_coeff_lines(), dof=dof, dt=P.TIMESTEP_FS,
        smoke_vars=smoke, real_vars=real,
        thermo=int(s["thermo_every_ps"] * 1000 / P.TIMESTEP_FS), dens_col="")
    return txt + "\n" + body


def field_input(name, kind, build, field, source_job, dof, seeds, s):
    prod = int(s["prod_ps"] * 1000 / P.TIMESTEP_FS)
    if field > 0.0:
        efield = (f"# external field along +z switched ON at t = 0 "
                  f"(V/A, real units)\n"
                  f"fix             efield mobile efield 0.0 0.0 {field}\n")
        note = f"switch-on, field {field} V/A, from {source_job} final state."
    else:
        efield = "# field OFF (switch-off from the saturating-field state)\n"
        note = f"switch-off (field 0), from {source_job} final state."
    body = f"""\
{efield}\
# velocities continue from the read-in final state (no re-draw)
fix             thb bodies rigid/nve molecule langevin 300.0 300.0 {s['langevin_damp_fs']:g} {seeds['bodies']}
fix             thl li nve
fix             thla li langevin 300.0 300.0 {s['langevin_damp_fs']:g} {seeds['li']}

# --- {s['prod_ps']:g} ps production, dump every {s['dump_every_ps']:g} ps
reset_timestep  0
{DUMP_CMD.format(did="pd", every=1000, fname="prod.dump")}
run             ${{nprod}}

write_data      final.data nocoeff
"""
    smoke = "variable nprod equal 500"
    real = f"variable nprod equal {prod}"
    txt = HEADER.format(
        name=name, desc=note, extra="",
        cutoff=P.CUTOFF_A, pppm_acc=f"{P.PPPM_REL_ACC:g}",
        pair_coeffs=pair_coeff_lines(), dof=dof, dt=P.TIMESTEP_FS,
        smoke_vars=smoke, real_vars=real,
        thermo=int(s["thermo_every_ps"] * 1000 / P.TIMESTEP_FS), dens_col="")
    return txt + "\n" + body


def fill_input(name, dof, seeds, vseed, s):
    ramp = int(s["fill_ramp_ps"] * 1000 / P.TIMESTEP_FS)
    prod = int(s["fill_prod_ps"] * 1000 / P.TIMESTEP_FS)
    pdamp = s["fill_baro_damp_fs"]
    p0 = s["fill_press_anneal_start_atm"]
    body = f"""\
velocity        mobile create 10.0 {vseed} mom yes rot yes dist gaussian

# --- stage 1 ({s['fill_ramp_ps']:g} ps): temperature ramp 10 -> 300 K and
#     pressure anneal. The bath is BUILT at {int(s['block_pack_frac'] * 100)}% of the
#     target bulk density (bulk-density sequential packing jams in the
#     small periodic build block - recorded in the build manifest), so the
#     barostat target ramps {p0:g} atm -> 1 atm over this stage and
#     compresses the bath before production. The barostat (fix nph, pure
#     pressure control - NO thermostat, so the single Langevin scheme
#     covers every mobile atom) is the fix-rigid docs' rigid-bodies-plus-
#     point-particles hybrid; dilate mobile never rescales the frozen
#     tube; the NPT pressure is the global thermo one.
fix             thb bodies rigid/nve molecule langevin 10.0 300.0 {s['langevin_damp_fs']:g} {seeds['bodies']}
fix             thl li nph iso {p0:g} {s['baro_press_atm']:g} {pdamp:g} dilate mobile
fix             thla li langevin 10.0 300.0 {s['langevin_damp_fs']:g} {seeds['li']}
run             ${{nrampf}}

# --- stage 2: {s['fill_prod_ps']:g} ps NPT production (1 atm, 300 K),
#     dump every {s['dump_every_ps']:g} ps
unfix           thb
unfix           thl
unfix           thla
fix             thb bodies rigid/nve molecule langevin 300.0 300.0 {s['langevin_damp_fs']:g} {seeds['bodies2']}
fix             thl li nph iso {s['baro_press_atm']:g} {s['baro_press_atm']:g} {pdamp:g} dilate mobile
fix             thla li langevin 300.0 300.0 {s['langevin_damp_fs']:g} {seeds['li2']}
reset_timestep  0
{DUMP_CMD.format(did="pd", every=1000, fname="prod.dump")}
run             ${{nprodf}}

write_data      final.data nocoeff
"""
    smoke = (f"variable nrampf equal 500\" \"variable nprodf equal 500")
    real = (f"variable nrampf equal {ramp}\" \"variable nprodf equal {prod}")
    txt = HEADER.format(
        name=name, desc="filling: open CNT in a bulk 1M LiPF6/EC:EMC bath; "
                        f"{s['fill_ramp_ps']:g} ps T+P anneal (10->300 K, "
                        f"barostat target {s['fill_press_anneal_start_atm']:g}"
                        "->1 atm, Pdamp "
                        f"{s['fill_baro_damp_fs']:g} fs, compressing the "
                        "sub-target-density packed bath) then 5 ns NPT "
                        "1 atm 300 K; tube starts empty (imbibition)",
        extra="", cutoff=P.CUTOFF_A, pppm_acc=f"{P.PPPM_REL_ACC:g}",
        pair_coeffs=pair_coeff_lines(), dof=dof, dt=P.TIMESTEP_FS,
        smoke_vars=smoke, real_vars=real,
        thermo=int(s["thermo_every_ps"] * 1000 / P.TIMESTEP_FS),
        dens_col=" density")
    return txt + "\n" + body


def main():
    s = P.STAGEA
    with open(os.path.join(HERE, "data", "stageA_build_manifest.json")) as f:
        bm = json.load(f)

    index = []

    def emit(name, txt, **kw):
        d = os.path.join(JOBS, name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "in.lammps"), "w") as f:
            f.write(txt)
        kw["name"] = name
        kw["est_wall_min"] = round(kw["atoms"] * kw["steps_ps"]
                                   / ATOM_PS_PER_MIN, 1)
        index.append(kw)

    for n in (10, 16):
        # ---- filling
        name = f"fill_cnt{n}"
        info = bm[name]
        comp = info["composition"]
        nb = comp["n_EC"] + comp["n_EMC"] + comp["n_LiPF6"]
        dof = 6 * nb + 3 * comp["n_LiPF6"] - 3
        vseed = 20265100 + n // 16
        seeds = _seeds(vseed, 1)
        emit(name, fill_input(name, dof, seeds, vseed, s),
             kind="fill", tube=n, build=None, field=None,
             source=f"data/{name}.data", depends=None,
             atoms=info["n_atoms_total"],
             steps_ps=s["fill_ramp_ps"] + s["fill_prod_ps"], dof=dof,
             seeds=dict(bodies=seeds["bodies"], li=seeds["li"],
                        bodies2=seeds["bodies2"], li2=seeds["li2"],
                        velocity=vseed))
        # ---- zero field + chained field runs
        for build in s["builds"]:
            sysname = f"cnt{n}_{build}"
            comp = P.COMPOSITION[n]
            dof = 6 * (comp["n_EC"] + comp["n_EMC"] + comp["n_LiPF6"]) \
                + 3 * comp["n_LiPF6"] - 3
            atoms = bm[f"cnt{n}_b1"]["n_atoms_total"]
            vseed = P.SEEDS[sysname]["velocity"]
            note = ("MIRRORED copy of b1 (z -> Lz - z, every atom)."
                    if build == "b1m" else "independent build (new seed).")
            zf = f"zf_{sysname}"
            seeds = _seeds(vseed, 1)
            emit(zf, zf_input(zf, build, note, dof, seeds, vseed, s),
                 kind="zf", tube=n, build=build, field=0.0,
                 source=f"data/{sysname}.data", depends=None, atoms=atoms,
                 steps_ps=s["ramp_ps"] + s["equil_ps"] + s["prod_ps"], dof=dof,
                 seeds=dict(bodies=seeds["bodies"], li=seeds["li"],
                            bodies2=seeds["bodies2"], li2=seeds["li2"],
                            velocity=vseed))
            for tag, field in (("wf", s["field_wf"]), ("sf", s["field_sf"])):
                job = f"{tag}_{sysname}"
                seeds = _seeds(vseed, 2 if tag == "wf" else 3)
                emit(job, field_input(job, tag, build, field, zf, dof,
                                      seeds, s),
                     kind=tag, tube=n, build=build, field=field,
                     source=f"jobs/{zf}/final.data", depends=zf, atoms=atoms,
                     steps_ps=s["prod_ps"], dof=dof,
                     seeds=dict(bodies=seeds["bodies"], li=seeds["li"]))
            job = f"off_{sysname}"
            seeds = _seeds(vseed, 4)
            emit(job, field_input(job, "off", build, 0.0, f"sf_{sysname}",
                                  dof, seeds, s),
                 kind="off", tube=n, build=build, field=0.0,
                 source=f"jobs/sf_{sysname}/final.data",
                 depends=f"sf_{sysname}", atoms=atoms,
                 steps_ps=s["prod_ps"], dof=dof,
                 seeds=dict(bodies=seeds["bodies"], li=seeds["li"]))

    with open(os.path.join(HERE, "stageA", "jobs_index.json"), "w") as f:
        json.dump(index, f, indent=2)
    total = sum(j["est_wall_min"] for j in index)
    print(f"[stageA-inputs] wrote {len(index)} job inputs under {JOBS}")
    print(f"[stageA-inputs] estimated total serial wall time "
          f"{total:.0f} min = {total / 60:.1f} h "
          f"(throughput {ATOM_PS_PER_MIN:g} atom*ps/min)")


if __name__ == "__main__":
    main()
