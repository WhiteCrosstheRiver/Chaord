"""Q3 D39 analysis on the cluster (Review 12): per-frame P1 from the raw
prod.dump / prod2.dump files (direct arm, no chaord), block means, Welch
test. Run on the ime001 login node: python3 p1_fixed.py

Dump line format: id mol type q x y z ix iy iz
We pack (q, x, y, z, mol) into the local array."""
import numpy as np
import json
import sys
from collections import defaultdict
from scipy import stats


def read_frames(path):
    frames = []
    with open(path, errors="replace") as f:
        while True:
            line = f.readline()
            if not line:
                break
            if "ITEM: TIMESTEP" not in line:
                continue
            step = int(f.readline())
            f.readline()          # ITEM: NUMBER OF ATOMS
            f.readline()          # n
            f.readline()          # ITEM: BOX BOUNDS
            f.readline()          # x
            f.readline()          # y
            f.readline()          # z
            f.readline()          # ITEM: ATOMS header
            alines = []
            while True:
                aline = f.readline()
                if not aline or aline.startswith("ITEM:"):
                    break
                alines.append(aline)
            # columns: q=3, x=4, y=5, z=6, mol=1
            atoms = np.empty((len(alines), 5))
            for k, al in enumerate(alines):
                parts = al.split()
                atoms[k, 0] = float(parts[3])   # q
                atoms[k, 1] = float(parts[4])   # x
                atoms[k, 2] = float(parts[5])   # y
                atoms[k, 3] = float(parts[6])   # z
                atoms[k, 4] = float(parts[1])   # mol id
            frames.append((step, atoms))
    return frames


def frame_p1(atoms):
    """Per-frame P1: per-molecule dipole (q-weighted, relative to the
    molecule's charge centre) z-component / |dipole|, averaged over
    multi-atom molecules (ions with |mu| ~ 0 are skipped)."""
    mol = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0])  # q*x, q*y, q*z, sum q, n
    for k in range(atoms.shape[0]):
        m = int(atoms[k, 4])
        mol[m][0] += atoms[k, 0] * atoms[k, 1]
        mol[m][1] += atoms[k, 0] * atoms[k, 2]
        mol[m][2] += atoms[k, 0] * atoms[k, 3]
        mol[m][3] += atoms[k, 0]
        mol[m][4] += 1.0
    cos = []
    for m, v in mol.items():
        if v[4] < 3:
            continue            # single atoms / diatomics: no meaningful dipole
        cx = v[3]               # net charge == centre-of-charge x (q-weighted)
        # dipole about the charge centre: mu = sum q_i (r_i - r_c)
        mux = v[0] - v[3] * cx  # = 0 by construction; use a fixed reference
        # simpler: dipole about the first atom
    # redo with first-atom reference
    cos = []
    seen = {}
    for k in range(atoms.shape[0]):
        m = int(atoms[k, 4])
        if m not in seen:
            seen[m] = [atoms[k, 1], atoms[k, 2], atoms[k, 3]]
    for m, v in mol.items():
        if v[4] < 3 or m not in seen:
            continue
        mux = v[0] - v[3] * seen[m][0]
        muy = v[1] - v[3] * seen[m][1]
        muz = v[2] - v[3] * seen[m][2]
        nrm = np.sqrt(mux * mux + muy * muy + muz * muz)
        if nrm < 1e-8:
            continue
        cos.append(muz / nrm)
    return float(np.mean(cos)) if cos else float("nan")


def main():
    results = {}
    for tag, base in (("e0", "/vol02/hdd1/ime001/chaord_q3/jobs/q3_e0_b"),
                      ("e10", "/vol02/hdd1/ime001/chaord_q3/jobs/q3_e10_b")):
        for b in (1, 2, 3):
            key = f"{tag}_b{b}"
            vals = {"A": [], "B": []}
            for blk, dumpf in (("A", "prod.dump"), ("B", "prod2.dump")):
                frames = read_frames(f"{base}{b}/{dumpf}")
                for step, atoms in frames:
                    vals[blk].append(frame_p1(atoms))
            results[key] = dict(A_mean=float(np.nanmean(vals["A"])),
                                B_mean=float(np.nanmean(vals["B"])),
                                A_n=len(vals["A"]), B_n=len(vals["B"]))
            print(key, "A", results[key]["A_mean"], "B", results[key]["B_mean"],
                  file=sys.stderr)

    off = [results[f"e0_b{b}"]["B_mean"] for b in (1, 2, 3)]
    on = [results[f"e10_b{b}"]["B_mean"] for b in (1, 2, 3)]
    r = stats.ttest_ind(on, off, equal_var=False)
    shift = np.mean(on) - np.mean(off)
    print(json.dumps(dict(shift=float(shift), t=float(r.statistic),
                          p=float(r.pvalue), e0_blockB=off, e10_blockB=on,
                          detected=bool(r.pvalue < 0.05)), indent=1))


if __name__ == "__main__":
    main()
