# -*- coding: utf-8 -*-
"""Staged R13 packing: insert at a reduced floor, then relax to the TRUE floor
with rigid-body MD (T1 order: "insert at reduced LJ radius, then relax to the
floor ... do not lower the counts"; r13/PLAN.md section 5).

Why: Packer.relax() moves centres of mass only. At the reservoir density the
flat EC rings cannot clear the floors without rotating (r13/HANDOFF.md). Here
the relaxation runs in LAMMPS with fix rigid/nve, so bodies translate AND
rotate, under a purely repulsive soft pair potential whose range is set by
the true floor matrix, plus a radial wall for the same wall floor that
Packer.verify() checks. No Coulomb, no LJ: this stage only removes overlaps.

The floors are never relaxed for acceptance: `finish` applies the unmodified
Packer.verify() thresholds and validate_data_file(), and fails closed.

Steps (one build at a time):
    python r13_staged.py prepack r13_b1      # reduced-floor insertion (serial)
    python r13_staged.py input   r13_b1      # writes relax/r13_b1/in.relax
    mpirun lmp_mpi -in in.relax              # in relax/r13_b1
    python r13_staged.py finish  r13_b1      # verify, write data/r13_b1.data

STANDALONE: imports r13_build/build_systems/params only, never chaord.
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r13_build as R       # noqa: E402
B, P = R.B, R.P

FLOOR_SCALE = 0.80                # lowest-effort scale that inserts everything
PREPACK_FINAL_SWEEPS = 300        # best effort only; MD does the real relaxing
RC_OVER_FLOOR = 1.12              # soft-pair range / true floor
WALL_OVER_FLOOR = 1.04            # radial wall onset / true wall floor
WALL_K = 60.0                     # kcal/mol/A^2
A_END = 100.0                     # soft-pair amplitude, kcal/mol
DT_FS = 0.5
RELAX_SEEDS = {f"r13_b{k}": 20293100 + k for k in range(1, 7)}   # outside every
# range in PLAN.md section 5 (which ends at 20292062); Langevin pairs add
# +10/+20/+30/+40, so builds 1-6 stay distinct

# Recorded from r13/HANDOFF.md: the earlier r13_b1 ladder (same build seed).
PRIOR_B1_ATTEMPTS = [
    dict(floor_scale=s, outcome="jam on an EC insertion",
         source="r13/HANDOFF.md") for s in (1.0, 0.90, 0.85)
] + [
    dict(floor_scale=s, outcome="inserted; translation-only relax() could "
         "not clear the true floors", source="r13/HANDOFF.md")
    for s in (0.80, 0.75, 0.70)
] + [
    dict(floor_scale=0.80, route="T1 staged: rigid soft relaxation",
         wall_over_floor=1.01, wall_k=20.0,
         outcome="wall gate failed: min pair/floor 1.1005, min wall/floor "
                 "0.9985 (threshold 0.999), slurm 233779; threshold kept, "
                 "wall stiffened")
]


def pre_path(name):
    return os.path.join(HERE, "prepack", "data", f"{name}.data")


def relax_dir(name):
    return os.path.join(HERE, "relax", name)


def tube_radius():
    return P.GRAPHENE_A * 16 * math.sqrt(3.0) / (2.0 * math.pi)


def prepack(name):
    """Stock build_system at FLOOR_SCALE, with the final strict verify
    deferred to `finish`. The insertion path is r13_build's own staged path."""
    P.COMPOSITION[16] = dict(R.R13_COMPOSITION_16)
    P.SYSTEMS[name] = dict(n=16, ncell=R.NCELL, field="zf")
    P.SEEDS[name] = dict(build=R.BUILD_SEEDS[name],
                         velocity=R.VELOCITY_SEEDS[name])
    B.HERE = os.path.join(HERE, "prepack")
    R._TRUE_MATRIX = B.MIN_ALLOWED
    R._SCALED_MATRIX = B.MIN_ALLOWED * FLOOR_SCALE
    B.Packer._ok = R._ok_staged
    B.Packer._candidate = R._candidate_staged
    B.Packer.RELAX_SWEEPS_FINAL = PREPACK_FINAL_SWEEPS
    B.Packer.verify = lambda self: True     # deferred: `finish` is the gate
    info, path = B.build_system(name)
    stats = B.validate_data_file(path, floor_ratio=0.0)
    out = os.path.join(HERE, "prepack", f"{name}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dict(info=info, stats=stats, floor_scale=FLOOR_SCALE), f,
                  indent=2)
    print(f"[prepack] {name}: {info['n_tolerant_inserts']} tolerant inserts, "
          f"min_inter {stats['min_intermolecular_A']} A, "
          f"min_tube {stats['min_mol_tube_A']} A -> {path}", flush=True)


IN_RELAX = """\
# {name}: rigid-body overlap removal (r13_staged.py). No charges, no LJ.
units           real
atom_style      full
boundary        p p p
read_data       {pre}

pair_style      soft {global_cut:.4f}
{pair_lines}
bond_style      harmonic
bond_coeff      1 0.0 1.0
special_bonds   lj/coul 0.0 0.0 0.0
neighbor        2.0 bin
neigh_modify    every 1 delay 0 check yes exclude molecule/intra all

group           tube type 1
group           mobile subtract all tube
group           li type 9
group           bodies subtract mobile li
fix             frz tube setforce 0.0 0.0 0.0

region          pore cylinder z 0.0 0.0 {R:.6f} INF INF side in
{wall_lines}
timestep        {dt}
thermo          2000
thermo_style    custom step temp pe ke press
thermo_modify   flush yes
velocity        mobile create 50.0 {seed} mom yes rot yes dist gaussian

# stage 1: 20 ps, amplitude 1 -> 30, 300 K (rotations unlock the rings)
variable        A1 equal ramp(1.0,30.0)
fix             ad all adapt 1 pair soft a * * v_A1
fix             thb bodies rigid/nve molecule langevin 300.0 300.0 100.0 {s1}
fix             thl li nve
fix             thla li langevin 300.0 300.0 100.0 {s2}
run             40000
unfix           ad
unfix           thb
unfix           thl
unfix           thla

# stage 2: 20 ps hold at amplitude 30, 300 K
variable        A2 equal 30.0
fix             ad all adapt 1 pair soft a * * v_A2
fix             thb bodies rigid/nve molecule langevin 300.0 300.0 100.0 {s3}
fix             thl li nve
fix             thla li langevin 300.0 300.0 100.0 {s4}
run             40000
unfix           ad
unfix           thb
unfix           thl
unfix           thla

# stage 3: 40 ps, amplitude 30 -> {a_end:g}, 300 -> 20 K
variable        A3 equal ramp(30.0,{a_end:g})
fix             ad all adapt 1 pair soft a * * v_A3
fix             thb bodies rigid/nve molecule langevin 300.0 20.0 100.0 {s1}
fix             thl li nve
fix             thla li langevin 300.0 20.0 100.0 {s2}
run             80000
unfix           ad
unfix           thb
unfix           thl
unfix           thla

# stage 4: 10 ps quench at amplitude {a_end:g}, 20 K
variable        A4 equal {a_end:g}
fix             ad all adapt 1 pair soft a * * v_A4
fix             thb bodies rigid/nve molecule langevin 20.0 20.0 100.0 {s3}
fix             thl li nve
fix             thla li langevin 20.0 20.0 100.0 {s4}
run             20000

write_dump      all custom relaxed.dump id mol type q xu yu zu modify sort id format float %.10g
"""


def write_input(name):
    d = relax_dir(name)
    os.makedirs(d, exist_ok=True)
    M = B.MIN_ALLOWED
    types = sorted(P.ATOM_TYPES)
    pairs, rcmax = [], 0.0
    for a, i in enumerate(types):
        for j in types[a:]:
            rc = RC_OVER_FLOOR * float(M[i, j])
            rcmax = max(rcmax, rc)
            pairs.append(f"pair_coeff      {i} {j} 1.0 {rc:.4f}")
    t_tube = B.TYPE_OF["C_tube"]
    walls = []
    for t in types:
        if t == t_tube:
            continue
        walls.append(f"group           w{t} type {t}")
        walls.append(f"fix             w{t} w{t} wall/region pore harmonic "
                     f"{WALL_K:g} 1.0 {WALL_OVER_FLOOR * float(M[t, t_tube]):.4f}")
    s = RELAX_SEEDS[name]
    txt = IN_RELAX.format(
        name=name, pre=os.path.relpath(pre_path(name), d).replace("\\", "/"),
        global_cut=rcmax + 0.05, pair_lines="\n".join(pairs),
        R=tube_radius(), wall_lines="\n".join(walls), dt=DT_FS, seed=s,
        s1=s + 10, s2=s + 20, s3=s + 30, s4=s + 40, a_end=A_END)
    with open(os.path.join(d, "in.relax"), "w", encoding="utf-8",
              newline="\n") as f:
        f.write(txt)
    print(f"[input] {name}: {d}/in.relax", flush=True)


def read_relaxed(path):
    with open(path) as f:
        L = f.readlines()
    k = next(i for i, l in enumerate(L) if l.startswith("ITEM: ATOMS"))
    cols = L[k].split()[2:]
    if cols != ["id", "mol", "type", "q", "xu", "yu", "zu"]:
        raise RuntimeError(f"unexpected dump columns {cols}")
    a = np.array([l.split() for l in L[k + 1:]], float)
    return a[np.argsort(a[:, 0])]


def split_data(path):
    with open(path) as f:
        L = f.read().split("\n")
    i0 = next(i for i, l in enumerate(L) if l.startswith("Atoms")) + 2
    i1 = next(i for i in range(i0, len(L)) if not L[i].strip())
    return L, i0, i1


def finish(name):
    """Rewrite the prepack data file with relaxed coordinates, then gate."""
    d = relax_dir(name)
    L, i0, i1 = split_data(pre_path(name))
    rows = [l.split() for l in L[i0:i1]]
    pre = np.array([[float(x) for x in r] for r in rows])
    pre = pre[np.argsort(pre[:, 0])]
    rel = read_relaxed(os.path.join(d, "relaxed.dump"))
    if pre.shape[0] != rel.shape[0] or np.any(pre[:, :3] != rel[:, :3]) \
            or np.any(np.abs(pre[:, 3] - rel[:, 3]) > 1e-9):
        raise RuntimeError("atom id/mol/type/charge changed during relaxation")
    zline = next(l for l in L if l.endswith("zlo zhi"))
    Lz = float(zline.split()[1]) - float(zline.split()[0])
    mol, typ = rel[:, 1].astype(int), rel[:, 2].astype(int)
    X = rel[:, 4:7].copy()

    # tube frozen; every molecule rigid to dump precision
    tube = mol == 1
    tube_move = float(np.abs(X[tube] - pre[tube, 4:7]).max())
    if tube_move > 1e-4:
        raise RuntimeError(f"tube moved by {tube_move} A")
    worst_rigid = 0.0
    for m in np.unique(mol[~tube]):
        ix = np.flatnonzero(mol == m)
        if len(ix) == 1:
            continue
        a, b = X[ix], pre[ix, 4:7]
        da = np.linalg.norm(a[:, None] - a[None], axis=-1)
        db = np.linalg.norm(b[:, None] - b[None], axis=-1)
        worst_rigid = max(worst_rigid, float(np.abs(da - db).max()))
        X[ix, 2] -= Lz * math.floor(a[:, 2].mean() / Lz)   # wrap whole body
    if worst_rigid > 1e-3:
        raise RuntimeError(f"rigid geometry drifted by {worst_rigid} A")

    # strict gate: the Packer.verify() thresholds, unmodified
    M = B.MIN_ALLOWED
    t_tube = B.TYPE_OF["C_tube"]
    mob = np.flatnonzero(~tube)
    P_, T_, O_ = X[mob], typ[mob], mol[mob]
    dv = P_[:, None] - P_[None]
    dv[..., 2] -= Lz * np.round(dv[..., 2] / Lz)
    dist = np.sqrt((dv ** 2).sum(-1))
    thr = M[T_][:, T_]
    ratio = dist / thr
    ratio[O_[:, None] == O_[None, :]] = np.inf
    min_pair_ratio = float(ratio.min())
    gap = tube_radius() - np.hypot(P_[:, 0], P_[:, 1])
    min_wall_ratio = float((gap / M[T_, t_tube]).min())
    n_bad = int((ratio < 0.999).sum() // 2)
    print(f"[finish] {name}: min pair/floor {min_pair_ratio:.4f}, "
          f"min wall/floor {min_wall_ratio:.4f}, pairs under floor {n_bad}, "
          f"rigid drift {worst_rigid:.2e} A", flush=True)
    # written as "not (>=)" so a NaN fails closed rather than passing
    if not (np.isfinite(X).all() and min_pair_ratio >= 0.999
            and min_wall_ratio >= 0.999):
        raise RuntimeError(f"{name}: true floors NOT cleared; not writing "
                           f"data/{name}.data")

    # write with the stock format, tube rows byte-identical to the build
    out = list(L)
    by_id = {int(i): k for k, i in enumerate(rel[:, 0])}
    for n, r in enumerate(rows):
        k = by_id[int(r[0])]
        if mol[k] == 1:
            continue
        out[i0 + n] = (f"{r[0]} {r[1]} {r[2]} {r[3]} {X[k, 0]:.6f} "
                       f"{X[k, 1]:.6f} {X[k, 2]:.6f}")
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    path = os.path.join(HERE, "data", f"{name}.data")
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(out))

    guard = B.validate_data_file(path)
    head = out[:8]
    natoms = int(next(l for l in head if "atoms" in l).split()[0])
    nbonds = int(next(l for l in head if "bonds" in l).split()[0])
    if natoms != R.EXPECTED_ATOMS or nbonds != R.EXPECTED_BONDS:
        os.remove(path)
        raise RuntimeError(f"{name}: counts {natoms}/{nbonds}")

    with open(os.path.join(HERE, "prepack", f"{name}.json"),
              encoding="utf-8") as f:
        pp = json.load(f)
    attempts = list(PRIOR_B1_ATTEMPTS) if name == "r13_b1" else []
    attempts.append(dict(
        floor_scale=FLOOR_SCALE, route="T1 staged: rigid soft relaxation",
        outcome="true floors cleared", min_pair_over_floor=round(min_pair_ratio, 4),
        min_wall_over_floor=round(min_wall_ratio, 4)))
    entry = dict(
        pp["info"], guard=guard, velocity_seed=R.VELOCITY_SEEDS[name],
        ncell=R.NCELL, packing_attempts=attempts,
        prepack_stats=pp["stats"],
        jam_policy="T1 staged packing: insert at 0.80 x MIN_ALLOWED, then "
                   "LAMMPS rigid-body (translation + rotation) relaxation "
                   "under a soft repulsion ranged on the true floors, then "
                   "strict floor gate; counts never lowered",
        relax=dict(engine="LAMMPS 3 Mar 2020 (lammps/3Mar20-icc.2022)",
                   seed=RELAX_SEEDS[name], input=f"relax/{name}/in.relax",
                   rc_over_floor=RC_OVER_FLOOR, a_end=A_END, dt_fs=DT_FS,
                   ps=90, rigid_drift_A=worst_rigid),
        composition_source="corrected open-tube uptake (fc8bec2) x 98.4 A: "
                           "EC 125 / EMC 58 / 24 LiPF6 (PLAN.md section 1)")
    frag = os.path.join(HERE, "data", f"build_manifest_{name}.json")
    with open(frag, "w", encoding="utf-8") as f:
        json.dump({name: entry}, f, indent=2)
    print(f"[finish] {name}: PASS {natoms} atoms, {nbonds} bonds, "
          f"guard min_inter {guard['min_intermolecular_A']} A, "
          f"min_tube {guard['min_mol_tube_A']} A -> {path}", flush=True)


def main():
    if len(sys.argv) != 3 or sys.argv[2] not in R.BUILD_SEEDS:
        sys.exit("usage: r13_staged.py prepack|input|finish r13_bK")
    {"prepack": prepack, "input": write_input,
     "finish": finish}[sys.argv[1]](sys.argv[2])


if __name__ == "__main__":
    main()
