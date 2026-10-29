# -*- coding: utf-8 -*-
"""
Write the Stage A LAMMPS run inputs (Review 10, work order Y6).

Jobs (34 total), all dumped every 1 ps with 10 significant digits and
image flags, thermo every 1 ps, dt 1 fs, ONE Langevin scheme (damping
1 ps) for every mobile atom:

  fill_cntN            open tube in a bulk bath, staged decompression
                       (10 K NVT settle + 10->300 K NVT warm-up at fixed
                       volume, then a 5000->1000->100->1 atm stepped NPT
                       descent with Pdamp 25 fs, then 5 ns NPT production
                       at 1 atm, 300 K); fix nph ... dilate mobile is the
                       pure barostat (no thermostat) so the one-Langevin
                       rule survives the NPT. (2026-10-20 repair of the
                       single-stage T+P anneal that blew up - see
                       stageA_plan.md addendum 2026-10-20.)
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
    """Staged decompression (2026-10-20 repair). The 2026-10-19 protocol
    turned the barostat on at step 0 with a -2000 atm target against the
    packed bath's actual +20,000 atm jam pressure and Pdamp 2 ps; the box
    blew up x26 within 7 ps (fill_cnt10 thermo: step 0 P=+20,869 atm,
    V=2.07e5 -> step 7000 Tmob 705 K, V=3.92e7). The repair: fixed-volume
    relaxation first, weak stepped barostat last - the box never meets
    the jam. Reasons for every number in stageA_plan.md addendum
    2026-10-20.

    seeds: 7 (bodies, li) seed pairs, one per thermostat fix pair
    (settle, warm-up, 4 pressure steps, production)."""
    setl = int(s["fill_settle_ps"] * 1000 / P.TIMESTEP_FS)
    warm = int(s["fill_warm_ps"] * 1000 / P.TIMESTEP_FS)
    baro = int(s["fill_baro_stage_ps"] * 1000 / P.TIMESTEP_FS)
    prod = int(s["fill_prod_ps"] * 1000 / P.TIMESTEP_FS)
    pdamp = s["fill_baro_damp_fs"]
    p_desc = " -> ".join(f"{p:g}" for p in s["fill_baro_stages_atm"])
    damp = s["langevin_damp_fs"]
    anneal_ps = (s["fill_settle_ps"] + s["fill_warm_ps"]
                 + s["fill_baro_stage_ps"] * len(s["fill_baro_stages_atm"]))

    def lang(t0, t1, sb, sl, li_fix):
        # one Langevin scheme per atom subset: rigid bodies + free Li+;
        # li_fix is "nve" (NVT stages) or the nph line (NPT stages), in
        # which fix nph itself time-integrates the Li+ (no double
        # integration) - the fix-rigid docs' hybrid, kept from the
        # 2026-10-19 input.
        return (f"fix             thb bodies rigid/nve molecule langevin "
                f"{t0:g} {t1:g} {damp:g} {sb}\n"
                f"fix             thl li {li_fix}\n"
                f"fix             thla li langevin {t0:g} {t1:g} {damp:g} "
                f"{sl}\n")

    (sb1, sl1), (sb2, sl2) = seeds[0], seeds[1]
    baro_stages = []
    for k, (ptarget, (sb, sl)) in enumerate(
            zip(s["fill_baro_stages_atm"], seeds[2:]), start=1):
        head = ("The 5000 atm head is a conservative UPPER bound on the "
                "post-NVT residual pressure, so the barostat's opening "
                "action is neutral-to-compressive: it cannot explode by "
                "construction." if k == 1 else
                "Each ~5x pressure step keeps the per-stage volume "
                "excursion small; the 300 K liquid re-wets the frozen "
                "tube between steps (each dilation by dilate mobile "
                "opens a thin sleeve at the un-rescaled tube, which the "
                "liquid closes again before the next step).")
        baro_stages.append(f"""
# --- stage 3.{k}: NPT at 300 K, barostat target {ptarget:g} atm, held
#     {s['fill_baro_stage_ps']:g} ps (stepped decompression {p_desc} atm,
#     Pdamp {pdamp:g} fs - deliberately ~25x the liquid-typical 1 ps, a
#     very weak barostat: the volume must still grow ~25% to the model's
#     own equation of state, and the 2026-10-19 failure was box-dynamics
#     overshoot at Pdamp 2 ps, so the box is made quasi-static).
#     {head}
unfix           thb
unfix           thl
unfix           thla
{lang(300.0, 300.0, sb, sl, f"nph iso {ptarget:g} {ptarget:g} {pdamp:g} dilate mobile")}run             ${{nbaro{k}}}
""")

    sbp, slp = seeds[2 + len(s["fill_baro_stages_atm"])]
    body = f"""\
velocity        mobile create 10.0 {vseed} mom yes rot yes dist gaussian

# force a log/screen flush at every thermo output: these jobs run for
# days; without it the redirected screen.log buffers ~4 KB and shows
# nothing for hours (found while monitoring the 2026-10-20 verification)
thermo_modify    flush yes

# --- stage 1 ({s['fill_settle_ps']:g} ps): NVT settle at 10 K, FIXED volume.
#     The packed bath is a cold jam carrying ~+20,000 atm of contact
#     stress (failed-run thermo: P = +20,869 / +11,042 atm at step 0)
#     because pack-then-relax leaves the contact network AT the packing
#     floors (0.85 sigma) even though the density is only ~87% of the
#     CRC target. At fixed volume there is no box degree of freedom to
#     run away: the Langevin bath (1 ps damping) simply drains the
#     contact energy released as neighbouring molecules push apart into
#     local free volume. A Tmob spike like the zf builds' 642 K
#     step-1000 transient is expected and harmless at fixed V.
{lang(10.0, 10.0, sb1, sl1, "nve")}run             ${{nsettle}}

# --- stage 2 ({s['fill_warm_ps']:g} ps): NVT warm-up ramp 10 -> 300 K,
#     still fixed volume. Melt the jam into a normal liquid BEFORE any
#     barostat sees it, so the first pressure the barostat reads is
#     ordinary liquid stress, not jam stress. (The failed runs heated
#     AND barostatted from step 0 - exactly the coupling that ran away.)
unfix           thb
unfix           thl
unfix           thla
{lang(10.0, 300.0, sb2, sl2, "nve")}run             ${{nwarm}}
{''.join(baro_stages)}
# --- stage 4: {s['fill_prod_ps']:g} ps NPT production (1 atm, 300 K,
#     Pdamp {pdamp:g} fs - kept weak so the box keeps drifting
#     adiabatically through the 5 ns), dump every {s['dump_every_ps']:g} ps
unfix           thb
unfix           thl
unfix           thla
{lang(300.0, 300.0, sbp, slp, f"nph iso {s['baro_press_atm']:g} {s['baro_press_atm']:g} {pdamp:g} dilate mobile")}reset_timestep  0
{DUMP_CMD.format(did="pd", every=1000, fname="prod.dump")}
run             ${{nprodf}}

write_data      final.data nocoeff
"""

    def if_vars(vals):
        # LAMMPS if/then command list: a" "b" "c inside the outer quotes
        return '" "'.join(f"variable {v} equal {x}" for v, x in vals)

    n_baro = len(s["fill_baro_stages_atm"])
    var_names = (["nsettle", "nwarm"]
                 + [f"nbaro{k}" for k in range(1, n_baro + 1)]
                 + ["nprodf"])
    var_vals = ([("nsettle", setl), ("nwarm", warm)]
                + [(f"nbaro{k}", baro) for k in range(1, n_baro + 1)]
                + [("nprodf", prod)])
    smoke = if_vars((v, 500) for v in var_names)
    real = if_vars(var_vals)
    txt = HEADER.format(
        name=name, desc="filling: open CNT in a bulk 1M LiPF6/EC:EMC bath; "
                        f"staged decompression: {s['fill_settle_ps']:g} ps "
                        f"10 K NVT settle + {s['fill_warm_ps']:g} ps 10->300 K "
                        f"NVT warm-up (fixed volume), then a "
                        f"{p_desc} atm stepped NPT descent "
                        f"({anneal_ps:g} ps total, Pdamp "
                        f"{s['fill_baro_damp_fs']:g} fs), then 5 ns NPT "
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
        # (bodies, li) seed pairs, one per thermostat fix pair: settle,
        # warm-up, one per pressure step, production (a distinct seed per
        # thermostat fix instance)
        n_pairs = 2 + len(s["fill_baro_stages_atm"]) + 1
        seeds = [(_seeds(vseed, c)["bodies"], _seeds(vseed, c)["li"])
                 for c in range(1, n_pairs + 1)]
        emit(name, fill_input(name, dof, seeds, vseed, s),
             kind="fill", tube=n, build=None, field=None,
             source=f"data/{name}.data", depends=None,
             atoms=info["n_atoms_total"],
             steps_ps=(s["fill_settle_ps"] + s["fill_warm_ps"]
                       + s["fill_baro_stage_ps"]
                       * len(s["fill_baro_stages_atm"])
                       + s["fill_prod_ps"]),
             dof=dof,
             seeds=dict(velocity=vseed, stage_bodies_li=seeds))
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
