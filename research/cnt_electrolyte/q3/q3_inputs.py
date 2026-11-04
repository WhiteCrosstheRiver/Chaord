"""Q3 confirmatory inputs (Review 12, D49 trigger): the 6 jobs
q3_<fieldmv>_b<1..3> — fields 0 and 0.01 V/A x 3 builds, the Z3 protocol
verbatim (1 ns in-job zero-field thermalization + 1 ns settle + 2 x 6 ns
production blocks, 12 ps thinned record), ONE cluster LAMMPS build
(module lammps/3Mar20-icc.2022, 3 Mar 2020). Seeds per PLAN section 4:
the stageB_inputs.seeds() pattern with the Q3 base 20281600+ (disjoint
from every prior range: stageA 2026xxxx, stageB 2126xxxx-2127xxxx,
confirmatory 20273xxx, q3 builds 20281xxx).

STANDALONE: no chaord import. Reads the q3 builds (data/q3_b{1,2,3}.data).
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CLUSTER_BASE = "/vol02/hdd1/ime001/chaord_q3/jobs"
FIELDS_MV = (0, 10)                      # 0 and 0.01 V/A
BUILDS = (1, 2, 3)
Q3_BASE = 20281600   # seeds land at ~2.03e7, the stageB scale


def seeds(build_k, field_mv):
    """Per-job seed pair, the stageB pattern: base + field*10 + build, so
    every (build, field) pair has its own pair and all are disjoint."""
    base = Q3_BASE + field_mv * 10 + build_k
    return dict(bodies=base * 10 + 1, li=base * 10 + 2)


def conf_input(name, field, build_k, build_seed):
    v = seeds(build_k, int(field * 1000))
    fm = int(field * 1000)
    efield = f"fix             efield mobile efield 0.0 0.0 {field}" if field else \
             "fix             efield mobile efield 0.0 0.0 0.0"
    return f"""# LAMMPS input: {name}  (Q3 confirmatory: the RESERVOIR filling
# (88 EC / 55 EMC / 15 LiPF6 per 98.4 A tube, Q2-measured and neutralized,
# q3/PLAN.md section 3) at {field} V/A -- the D49 filling-sensitivity check
# of the Z4 verdict. Protocol = Z3's: 1 ns in-job zero-field thermalization
# + 1 ns settle + 2 x 6 ns production blocks; 12 ps thinned record.
# One LAMMPS build for all Q3 jobs: module lammps/3Mar20-icc.2022.
units           real
atom_style      full
boundary        p p p
read_data       system.data

pair_style      lj/cut/coul/long 12.0
pair_modify     shift yes
kspace_style    pppm 0.0001
{pair_coeff_lines()}

group           tube type 1
group           mobile subtract all tube
group           li type 9
group           bodies subtract mobile li
fix             frz tube setforce 0.0 0.0 0.0
{efield}

compute         keb mobile temp
compute         keli li temp
variable        smoke index 0
if "${smoke} == 1" then "variable nthermal equal 10000" "variable nsettle equal 10000" "variable nblock equal 60000" "jump SELF skip-rest"
variable        nthermal equal 1000000
variable        nsettle equal 1000000
variable        nblock equal 6000000
label           skip-rest
variable        Tmob equal 2.0*(c_keb+c_keli)/(8301*0.0019872041)
variable        dofequal equal 8301

# ---- 1 ns in-job zero-field thermalization (the build's velocities are
# fresh; this stage equilibrates the RESERVOIR filling at 300 K).
# NOTE: no fix ave/time anywhere -- the 3 Mar 2020 build rejects
# reset_timestep while an ave/time fix is alive (its first output
# window would be skipped); the stage temperatures come from the
# regular thermo output instead.
velocity        mobile create 300.0 {v['bodies']} mom yes rot yes dist gaussian
fix             thb bodies rigid/nve molecule langevin 300.0 300.0 1000 {v['bodies']}
fix             thl li nve
fix             thla li langevin 300.0 300.0 1000 {v['li']}
timestep        1.0
run             ${nthermal}
unfix           thb
unfix           thla

# ---- 1 ns settle (the Z3 stage-2) ----
run             ${nsettle}
write_data      settle.data nocoeff

# ---- production: 2 x 6 ns blocks, dump every 1 ps, %.10g + image flags ----
reset_timestep  0
dump            ed all custom 1000 equil.dump id mol type q x y z ix iy iz
dump_modify     ed sort id format float %.10g
dump            pd all custom 1000 prod.dump id mol type q x y z ix iy iz
dump_modify     pd sort id format float %.10g
thermo          1000
thermo_style    custom step atoms v_Tmob temp pe etotal press vol
run             ${nblock}
undump          ed
undump          pd
write_data      blockA.data nocoeff
reset_timestep  0
dump            pd all custom 1000 prod2.dump id mol type q x y z ix iy iz
dump_modify     pd sort id format float %.10g
run             ${nblock}
undump          pd
write_data      final.data nocoeff
"""


def pair_coeff_lines():
    # the full 11-type self-coeff matrix (copied verbatim from the verified
    # stageB template sb_cnt10_e0_r1/in.lammps); pair_modify mix arithmetic
    # supplies the cross terms, exactly as every stageA/stageB job ran
    return """pair_modify     mix arithmetic tail no
pair_coeff      1 1 0.085300 3.4000
pair_coeff      2 2 0.104995 3.7500
pair_coeff      3 3 0.170005 3.0000
pair_coeff      4 4 0.065989 3.5000
pair_coeff      5 5 0.209990 2.9600
pair_coeff      6 6 0.015010 2.4200
pair_coeff      7 7 0.029995 2.5000
pair_coeff      8 8 0.066000 3.5000
pair_coeff      9 9 0.165057 1.5100
pair_coeff      10 10 0.200000 3.7400
pair_coeff      11 11 0.060994 3.1180

bond_style       harmonic
bond_coeff      1 0.0 1.0
special_bonds   lj/coul 0.0 0.0 0.0"""


def main():
    index = []
    jobs_dir = os.path.join(HERE, "jobs")
    for fm in FIELDS_MV:
        for bk in BUILDS:
            name = f"q3_e{fm:02d}_b{bk}" if fm else f"q3_e0_b{bk}"
            d = os.path.join(jobs_dir, name)
            os.makedirs(d, exist_ok=True)
            build_seed = 20281101 + bk if bk == 1 else (
                20281101 + bk if bk == 3 else 20281703)  # q3_b2's successful 6th attempt
            with open(os.path.join(d, "in.lammps"), "w",
                      encoding="utf-8", newline="\n") as f:
                f.write(conf_input(name, fm / 1000.0, bk, build_seed))
            shutil_copy(os.path.join(HERE, "data", f"q3_b{bk}.data"),
                        os.path.join(d, "system.data"))
            v = seeds(bk, fm)
            index.append(dict(
                name=name, kind="q3", tube=16, tube_len_A=98.4,
                build=f"q3_b{bk}", build_k=bk, field=fm / 1000.0,
                field_mv=fm, source=f"q3/data/q3_b{bk}.data",
                atoms=4385, seeds=v,
                protocol="Z3 verbatim: 1 ns thermalization + 1 ns settle "
                         "+ 2 x 6 ns blocks, 12 ps thinned record"))
    out = os.path.join(HERE, "jobs_index.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    print(f"[q3-inputs] wrote {len(index)} jobs under {jobs_dir}")
    print("[q3-inputs] 8-rank cluster wall per job ~7 h; 2 waves of 3-6")


def shutil_copy(src, dst):
    import shutil
    shutil.copyfile(src, dst)


if __name__ == "__main__":
    main()
