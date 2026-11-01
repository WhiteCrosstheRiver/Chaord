"""Q2 (Review 12): the Y6 reservoir-filling analysis.

Per fill (fill_cnt10, fill_cnt16 -- open CNT in a bulk 1M LiPF6/EC:EMC
bath, NPT 1 atm 300 K, thinned prod series at 5 ps): count molecules and
ions per Angstrom in the MIDDLE of the tube (more than 1 nm from either
mouth), block means and errors, and compare per species with the pilot
compositions per Angstrom:
  (10,10): EC 8, EMC 5, LiPF6 2 per 49.2 A
  (16,16): EC 37, EMC 24, LiPF6 6 per 49.2 A

Molecule identification: the dump carries `id mol type q x y z ix iy iz`;
type 1 is the tube carbon (group tube type 1), the Li+ is the lone 6.94
amu type (9), PF6- the P type (10) + F types (11); EC/EMC are identified
by their bond-graph molecules through the mol id (each solvent molecule
is one rigid body = one mol id; EC 10 atoms, EMC 15 atoms -- the census
comes from the body sizes in the data file convention: bodies of 10 atoms
with 3 C are EC... robustly: per mol id, count atoms and whether it
contains the P type; solvent bodies with a carbonyl-pattern C/O count and
10 atoms are EC, 15 atoms EMC).

Run:  .venv/Scripts/python.exe research/cnt_electrolyte/analysis/filling_analysis.py
Writes the numbers into reports/research/filling.md's data block and
research/cnt_electrolyte/analysis/filling_results.json.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
FILLS = ROOT / "research/cnt_electrolyte/fills_cluster"
PILOT = {  # per 49.2 A of tube (Stage B ladder composition)
    10: dict(EC=8, EMC=5, LiPF6=2),
    16: dict(EC=37, EMC=24, LiPF6=6),
}
PILOT_LEN = 49.2          # A, the ladder tubes
MOUTH_MARGIN = 10.0       # A, "more than 1 nm from either mouth" (Q2)
BLOCK_PS = 2500.0         # two production blocks of the 5 ns run
DT_PS = 5.0               # the thinned series' spacing


def parse_types(system_data_path):
    """type -> (mass, element-ish) from the Masses section."""
    masses = {}
    in_masses = False
    for line in Path(system_data_path).read_text().splitlines():
        if line.strip() == "Masses":
            in_masses = True
            continue
        if in_masses:
            if not line.strip():
                if masses:
                    break
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    masses[int(parts[0])] = float(parts[1])
                except ValueError:
                    break
    return masses


def molecule_bodies(system_data_path):
    """mol id -> (atom count, has_P, has_Li) from the Atoms section (full
    style: id mol type q x y z ...)."""
    bodies = defaultdict(lambda: dict(n=0, has_P=False, has_Li=False,
                                      has_C=False, q=0.0))
    in_atoms = False
    for line in Path(system_data_path).read_text().splitlines():
        if line.startswith("Atoms"):
            in_atoms = True
            continue
        if in_atoms:
            if not line.strip():
                if bodies:
                    break
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            mid = int(parts[1])
            t = int(parts[2])
            q = float(parts[3])
            b = bodies[mid]
            b["n"] += 1
            b["q"] += q
            if t == 10:
                b["has_P"] = True
            if t == 9:
                b["has_Li"] = True
            if t in (2, 4, 8):      # the carbon types of the pilot model
                b["has_C"] = True
    return bodies


def classify(bodies):
    """mol id -> species string, via body size + composition."""
    out = {}
    for mid, b in bodies.items():
        if b["has_Li"]:
            out[mid] = "Li+"
        elif b["has_P"]:
            out[mid] = "PF6-"
        elif b["n"] == 10:
            out[mid] = "EC"
        elif b["n"] == 15:
            out[mid] = "EMC"
        else:
            out[mid] = "tube"       # the frozen carbon body
    return out


def snapshots(dump_path):
    """Yield (zlo, zhi, {mid: (type, q, x, y, z)}) per frame; the fill tubes
    are periodic along z and the box is orthogonal."""
    frames = []
    cur = None
    box = None
    natoms = 0
    with open(dump_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("ITEM: NUMBER OF ATOMS"):
                natoms = int(next(f))
            elif line.startswith("ITEM: BOX BOUNDS"):
                rows = [next(f).split() for _ in range(3)]
                box = [(float(r[0]), float(r[1])) for r in rows]
            elif line.startswith("ITEM: ATOMS"):
                cur = np.empty((natoms, 7))
                idx = {}
                header = line.split()[2:]
                cols = {c: i for i, c in enumerate(header)}
                for k in range(natoms):
                    parts = next(f).split()
                    i = int(parts[cols["id"]])
                    cur[k] = (int(parts[cols["mol"]]), int(parts[cols["type"]]),
                              float(parts[cols["q"]]),
                              float(parts[cols["x"]]), float(parts[cols["y"]]),
                              float(parts[cols["z"]]),
                              float(parts[cols["ix"]]))
                    idx[i] = k
                cur = cur[np.argsort(cur[:, 0])]
                frames.append((box, cur))
    return frames


def com_per_mol(atoms, bodies_species):
    """mol id -> (x, y, z) of the molecule's centre of mass. Every molecule
    is whole in the wrapped coordinates (the fill guard asserts it), so the
    plain mean of its atoms is its COM; the tube carbons span the box so
    they are excluded by the caller via bodies_species."""
    per = defaultdict(list)
    for row in atoms:
        per[int(row[0])].append((row[3], row[4], row[5]))
    out = {}
    for mid, pts in per.items():
        sp = bodies_species.get(mid)
        if sp:
            arr = np.array(pts)
            out[mid] = tuple(float(v) for v in arr.mean(axis=0))
    return out


def count_middle(com_per_mol_map, species, tube_lo, tube_hi,
                 tube_axis_xy, tube_radius, radial_margin=0.0):
    """Count per species of molecules whose COM lies in the tube's middle
    axial window (MOUTH_MARGIN inside both mouths) AND inside the tube bore
    (radial distance from the axis < tube_radius - radial_margin).

    The bore convention is r < R (the wall position): the narrow (10,10)
    tube's bore is entirely wall-proximate -- an r < R-3.2 core criterion
    would exclude nearly every ion (a Li+ with its solvation shell sits
    off-axis) and read a false -100% Li+ depletion. The wall criterion
    counts every molecule the tube contains, which is what the pilot
    composition counts too."""
    counts = defaultdict(int)
    lo = tube_lo + MOUTH_MARGIN
    hi = tube_hi - MOUTH_MARGIN
    ax, ay = tube_axis_xy
    r_max = tube_radius - radial_margin
    for mid, (x, y, z) in com_per_mol_map.items():
        sp = species[mid]
        if not (lo <= z <= hi):
            continue
        r = ((x - ax) ** 2 + (y - ay) ** 2) ** 0.5
        if r <= r_max:
            counts[sp] += 1
    return dict(counts)


def block_stats(counts_series, dt_ps=DT_PS):
    a = np.array(counts_series, float)
    half = len(a) // 2
    blocks = [a[:half], a[half:]]
    means = [b.mean() for b in blocks]
    sem = a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0
    return dict(mean=float(a.mean()), sem=float(sem),
                block_means=[float(m) for m in means],
                n_frames=len(a))


def main():
    results = {}
    for name, tube in (("fill_cnt10", 10), ("fill_cnt16", 16)):
        d = FILLS / name
        masses = parse_types(d / "final.data")
        bodies = molecule_bodies(d / "final.data")
        species = classify(bodies)
        frames = snapshots(d / "prod_thinned.dump")
        # the tube's z extent: the carbons' min/max (type 1)
        zc = np.concatenate([a[a[:, 1] == 1][:, 5] for _b, a in frames[:1]])
        tube_lo, tube_hi = float(zc.min()), float(zc.max())
        # tube axis (x, y) = the mean of the type-1 carbons; radius = the
        # mean radial distance of the carbons from that axis
        c_atoms = frames[0][1][frames[0][1][:, 1] == 1]
        ax, ay = float(c_atoms[:, 3].mean()), float(c_atoms[:, 4].mean())
        tube_radius = float(np.mean(((c_atoms[:, 3] - ax) ** 2
                                     + (c_atoms[:, 4] - ay) ** 2) ** 0.5))
        # the wall criterion (radial_margin 0) is the Q2 convention; the
        # R-3.2 core count is disclosed as the secondary number
        core_radius = tube_radius - 3.2
        series = defaultdict(list)
        core_series = defaultdict(list)
        for box, atoms in frames:
            cpm = com_per_mol(atoms, species)
            c = count_middle(cpm, species, tube_lo, tube_hi,
                             (ax, ay), tube_radius)
            c_core = count_middle(cpm, species, tube_lo, tube_hi,
                                  (ax, ay), core_radius)
            for sp in ("EC", "EMC", "Li+", "PF6-"):
                series[sp].append(c.get(sp, 0))
                core_series[sp].append(c_core.get(sp, 0))
        tube_len = tube_hi - tube_lo
        per_A = {sp: dict(
            mean=v["mean"] / tube_len,
            sem=v["sem"] / tube_len,
            count_mean=v["mean"], count_sem=v["sem"])
            for sp, v in ((sp, block_stats(s)) for sp, s in series.items())}
        pilot_per_A = {sp: PILOT[tube][sp] / PILOT_LEN
                       for sp in PILOT[tube]}
        diff = {}
        for sp, pilot_name in (("EC", "EC"), ("EMC", "EMC"),
                               ("Li+", "LiPF6"), ("PF6-", "LiPF6")):
            if sp not in per_A:
                continue
            pilot = pilot_per_A[pilot_name]
            fill = per_A[sp]["mean"]
            diff[sp] = dict(pilot_per_A=pilot, fill_per_A=fill,
                            pct=(fill - pilot) / pilot * 100.0
                            if pilot > 0 else float("nan"))
        # charge-neutrality sanity: Li+ and PF6- in-tube counts must match
        # over the run's mean (a mismatch would mean counter-ions in the
        # reservoir carrying the difference -- legitimate but worth stating)
        neutrality = (series["Li+"], series["PF6-"])
        results[name] = dict(tube=tube, tube_len_A=tube_len,
                             n_frames=len(frames), per_species=per_A,
                             vs_pilot=diff,
                             ion_neutrality=dict(
                                 li_mean=float(np.mean(neutrality[0])),
                                 pf6_mean=float(np.mean(neutrality[1]))),
                             raw_counts={sp: block_stats(s)
                                         for sp, s in series.items()},
                             core_counts={sp: block_stats(s)
                                          for sp, s in core_series.items()})
    out = ROOT / "research/cnt_electrolyte/analysis/filling_results.json"
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    for name, r in results.items():
        print(f"== {name} (tube {r['tube']}, {r['tube_len_A']:.1f} A, "
              f"{r['n_frames']} frames)")
        for sp, d in r["vs_pilot"].items():
            print(f"  {sp:5s}: fill {d['fill_per_A']:.4f}/A vs pilot "
                  f"{d['pilot_per_A']:.4f}/A  -> {d['pct']:+.1f}%")
    print("wrote", out)


if __name__ == "__main__":
    main()
