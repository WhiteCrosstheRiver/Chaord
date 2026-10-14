# -*- coding: utf-8 -*-
"""
Generate the LAMMPS run inputs in.<system> for each of the four systems.

Protocol (identical for all systems except the field and the seeds):
  1. read data file (rigid molecules as topology-free atom groups + zero-force
     bonds used only for 1-2/1-3 special_bonds exclusions)
  2. velocities: gaussian at 300 K, seed per system, zero net momentum
  3. 50 ps ramp 10 -> 300 K (Nose-Hoover on rigid bodies), field already on
     for the field systems (the 'on' state is prepared under the field)
  4. 400 ps NVT equilibration at 300 K
  5. 100 ps production, dump every 5 ps -> 21 frames (>= 20, >= 1 ps apart)

Engine: LAMMPS 22 Jul 2025 Update 4 (serial, Windows).
STANDALONE: does not import chaord.
"""
import os

import params as P

HERE = os.path.dirname(os.path.abspath(__file__))

TEMPLATE = """\
# LAMMPS input: {name}  (LiPF6/EC:EMC in ({n},{n}) armchair CNT, field {field})
# Engine: LAMMPS 22 Jul 2025 Update 4. Units: real (A, kcal/mol, fs, V/A).
# Potential: OPLS-AA-based rigid molecules, LJ + point charges
#   (parameters + citations in params.py / provenance JSON).
# Combining rules: Lorentz-Berthelot (documented deviation from OPLS
#   geometric sigma; see provenance).
# Rigid molecules: fix rigid/nvt/small (Nose-Hoover, body DOFs).
#   Intra-body nonbonded terms are constants of the motion (see provenance);
#   real-space 1-2/1-3 excluded via zero-K bonds + special_bonds 0 0 0 and
#   neigh_modify exclude molecule/intra.
#
# POST-RESUME EDITS (2026-10-14, recorded as deviations in provenance):
#   * velocities created at 10 K (was 300 K) to match the documented
#     10 -> 300 K thermostat ramp (the original 300 K create + 10 K ramp
#     start was an internal inconsistency: it quenched then reheated).
#   * equilibration dump every 25 ps added (equil.dump, same format as the
#     production dump) to record the inside-tube density plateau used as
#     equilibration evidence.
#   * thermo 500 (0.5 ps resolution) for the potential-energy plateau record.
#   * Li+ single-atom bodies: BOTH fix rigid and fix rigid/nvt/small reject
#     single-particle rigid bodies (found on first execution; the original
#     plan had never been run). Li+ atoms are now integrated/thermostatted
#     by their own Nose-Hoover fix nvt on a disjoint group (same Tstart/
#     Tstop/Tdamp), multi-atom molecules by fix rigid/nvt/small molecule.

units           real
atom_style      full
boundary        p p p

read_data       {datafile}

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

# honest mobile-subsystem temperature: 2*KE/(DOF*kB), real-units
# kB = 0.0019872041 kcal/mol/K; DOF = 6*N_rigid_bodies + 3*N_Li - 3
# (the default thermo 'temp' is diluted by the 800/1280 frozen,
# zero-velocity tube atoms and is kept only for reference)
compute         keb bodies ke
compute         keli li ke
variable        Tmob equal 2.0*(c_keb+c_keli)/({dof}*0.0019872041)

velocity        mobile create 10.0 {vseed} mom yes rot yes dist gaussian

timestep        {dt}

# external electric field along +z (acts on charges; tube is neutral)
{field_line}
# --- stage 1: gentle thermalization ramp 10 -> 300 K, {ramp_ps} ps
#     (multi-atom rigid bodies: Nose-Hoover on body DOFs; Li+ single atoms:
#     Nose-Hoover fix nvt; identical targets and damping)
fix             thb bodies rigid/nvt/small molecule temp 10.0 300.0 {tdamp}
fix             thli li nvt temp 10.0 300.0 {tdamp}
thermo          500
thermo_style    custom step atoms v_Tmob temp pe etotal evdwl ecoul elong press vol
run             {ramp_steps}

# --- stage 2: NVT equilibration at 300 K, {equil_ps} ps
#     (dump every 25 ps: inside-tube density plateau evidence)
unfix           thb
unfix           thli
fix             thb bodies rigid/nvt/small molecule temp 300.0 300.0 {tdamp}
fix             thli li nvt temp 300.0 300.0 {tdamp}
dump            ed all custom {equil_dump_every} equil.dump id mol type q x y z
dump_modify     ed sort id
run             {equil_steps}
undump          ed

# --- stage 3: production {prod_ps} ps, dump every {dump_every_ps:g} ps
reset_timestep  0
dump            pd all custom {dump_every} prod.dump id mol type q x y z
dump_modify     pd sort id
run             {prod_steps}

write_data      final.data
"""


def main():
    for name, spec in P.SYSTEMS.items():
        n_cnt = spec["n"]
        comp = P.COMPOSITION[n_cnt]
        dof = 6 * (comp["n_EC"] + comp["n_EMC"] + comp["n_LiPF6"]) \
            + 3 * comp["n_LiPF6"] - 3
        field_on = spec["field"] == "on"
        field_line = ""
        if field_on:
            field_line = (f"fix efield mobile efield 0.0 0.0 "
                          f"{P.FIELD_ON_V_PER_A}")
        else:
            field_line = "# (field off state: no fix efield)"
        coeffs = []
        for t in sorted(P.ATOM_TYPES):
            _, _, sig, eps, _, _src = P.ATOM_TYPES[t]
            coeffs.append(f"pair_coeff      {t} {t} {eps:.6f} {sig:.4f}")
        txt = TEMPLATE.format(
            name=name, n=spec["n"], field=spec["field"],
            datafile=os.path.join(HERE, "data", f"{name}.data").replace("\\", "/"),
            cutoff=P.CUTOFF_A, ppcm="", pppm_acc=f"{P.PPPM_REL_ACC:g}",
            pair_coeffs="\n".join(coeffs) + "\n\n",
            temp=P.TEMPERATURE_K, vseed=P.SEEDS[name]["velocity"],
            dof=dof,
            equil_dump_every=int(25 * 1000 / P.TIMESTEP_FS),
            dt=P.TIMESTEP_FS, field_line=field_line,
            ramp_ps=P.RAMP_PS, ramp_steps=int(P.RAMP_PS * 1000 / P.TIMESTEP_FS),
            equil_ps=P.EQUIL_PS, equil_steps=int(P.EQUIL_PS * 1000 / P.TIMESTEP_FS),
            prod_ps=P.PROD_PS, prod_steps=int(P.PROD_PS * 1000 / P.TIMESTEP_FS),
            tdamp=P.THERMO_DAMP_FS,
            dump_every=int(P.DUMP_EVERY_PS * 1000 / P.TIMESTEP_FS),
            dump_every_ps=P.DUMP_EVERY_PS,
        )
        path = os.path.join(HERE, f"in.{name}")
        with open(path, "w") as f:
            f.write(txt)
        print("wrote", path)


if __name__ == "__main__":
    main()
