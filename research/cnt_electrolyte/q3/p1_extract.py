"""Q3 confirmatory P1 extraction (Review 12 Q3, D39/D49): run on the CLUSTER
via sbatch — per the owner's directive, batch compute goes on the cluster,
never the local box.

Parses the 6 q3_e{0,10}_b{1,2,3} prod.dump/prod2.dump files (block A / B,
1 ps cadence, 6000 frames each), computes per-frame solvent P1 (dipole
cos w.r.t. z) via numpy over mol-id groups, and writes block means.

Parallel: multiprocessing Pool over the 12 dump files, 6 workers.

Usage (on the cluster):
    sbatch -p cpu -N 1 -n 28 -t 01:00:00 -o p1out-%j.out -J q3p1 \
        --wrap="module load lammps/3Mar20-icc.2022; python3 p1_extract.py"
    (or plain python3 on a compute node)
"""
import json
import os
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

BASE = "/vol02/hdd1/ime001/chaord_q3/jobs"
OUT = "/vol02/hdd1/ime001/chaord_q3/p1_blocks.json"
JOBS = [f"q3_e{f}_b{b}" for f in (0, 10) for b in (1, 2, 3)]
DUMPS = [("A", "prod.dump"), ("B", "prod2.dump")]
BOXZ = 98.4


def parse_one(args):
    path, tag = args
    frames = []
    with open(path, errors="replace") as f:
        while True:
            line = f.readline()
            if not line:
                break
            if "ITEM: TIMESTEP" in line:
                step = int(f.readline())
                f.readline()                       # NUMBER OF ATOMS
                n = int(f.readline())
                f.readline()                       # BOX BOUNDS header
                f.readline(); f.readline(); f.readline()  # 3 bound lines
                hdr = f.readline().split()[2:]
                cols = {c: i for i, c in enumerate(hdr)}
                atoms = np.empty((n, len(hdr)))
                for k in range(n):
                    parts = f.readline().split()
                    for ci, cn in enumerate(cols):
                        atoms[k, ci] = float(parts[ci])
                frames.append((step, atoms, cols))
    p1s = []
    for step, atoms, cols in frames:
        mol = defaultdict(list)
        for k in range(atoms.shape[0]):
            mol[int(atoms[k, 0])].append(k)
        cos = []
        for m, ks in mol.items():
            if len(ks) < 4:                        # solvent only (EC 10/EMC 15)
                continue
            xs = np.array([atoms[k, cols["x"]] for k in ks])
            ys = np.array([atoms[k, cols["y"]] for k in ks])
            zs = np.array([atoms[k, cols["z"]] for k in ks])
            qs = np.array([atoms[k, cols["q"]] for k in ks])
            z0 = zs[0]
            zs = z0 + (zs - z0 + BOXZ / 2) % BOXZ - BOXZ / 2
            cx, cy, cz = xs.mean(), ys.mean(), zs.mean()
            muz = (qs * (zs - cz)).sum()
            nmu = np.sqrt(((qs * (xs - cx)).sum()) ** 2 +
                          ((qs * (ys - cy)).sum()) ** 2 + muz ** 2)
            if nmu > 1e-6:
                cos.append(muz / nmu)
        if cos:
            p1s.append(float(np.mean(cos)))
    return tag, (float(np.mean(p1s)) if p1s else None), len(p1s)


def main():
    tasks = [(f"{BASE}/{name}/{dumpf}", f"{name}_{blk}")
             for name in JOBS for blk, dumpf in DUMPS]
    with Pool(6) as pool:
        results = pool.map(parse_one, tasks)
    out = {}
    for tag, mean, n in results:
        key, blk = tag.rsplit("_A", 1) if tag.endswith("_A") else tag.rsplit("_B", 1), None
        # rebuild: tag is f"{name}_{blk}"
        name, blk = tag.rsplit("_", 1)
        out.setdefault(name, {})[f"P1_{blk}_mean"] = mean
        out[name][f"P1_{blk}_n"] = n
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {OUT}")
    for name, d in sorted(out.items()):
        print(name, {k: (round(v, 4) if isinstance(v, float) else v)
                     for k, v in d.items()})


if __name__ == "__main__":
    main()
