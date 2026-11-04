import numpy as np, json
from collections import defaultdict

def analyze(name, path, pilot):
    counts = {"EC": [], "EMC": [], "Li": [], "F6P": []}
    with open(path, errors="replace") as f:
        while True:
            line = f.readline()
            if not line: break
            if "ITEM: TIMESTEP" not in line: continue
            f.readline(); f.readline()
            f.readline(); f.readline(); f.readline()
            f.readline()
            mol = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0])
            while True:
                aline = f.readline()
                if not aline or aline.startswith("ITEM:"): break
                p = aline.split()
                m = int(p[1]); t = int(p[2]); q = float(p[3])
                mol[m][0] += q*float(p[4]); mol[m][1] += q*float(p[5])
                mol[m][2] += q*float(p[6]); mol[m][3] += q
                mol[m][4] += 1
                if t == 1: mol[m][5] = 1  # tube carbon
            c = defaultdict(int)
            for m, v in mol.items():
                n = v[4]
                if v[5] == 1: sp = "tube"
                elif n == 10: sp = "EC"
                elif n == 15: sp = "EMC"
                elif n == 1 and abs(v[3] - 1.0) < 0.01: sp = "Li"
                elif n == 7 and v[3] < -1.0: sp = "F6P"
                else: sp = "other"
                c[sp] += 1
            for sp in ("EC", "EMC", "Li", "F6P"):
                counts[sp].append(c.get(sp, 0))
    out = {}
    for sp, s in counts.items():
        a = np.array(s, float)
        out[sp] = dict(mean=float(a.mean()), sem=float(a.std(ddof=1)/np.sqrt(len(a))))
    return out

results = {}
for name, pilot in (("fill_cnt10", dict(EC=8, EMC=5, LiPF6=2)),
                    ("fill_cnt16", dict(EC=37, EMC=24, LiPF6=6))):
    path = f"/vol02/hdd1/ime001/chaord_fills_v2/{name}/prod_thinned.dump"
    r = analyze(name, path, pilot)
    results[name] = dict(counts=r, pilot=pilot)
    print(name)
    for sp in ("EC", "EMC", "Li", "F6P"):
        d = r[sp]
        pk = {"Li": "LiPF6", "F6P": "LiPF6"}.get(sp, sp)
        print(f"  {sp}: {d['mean']:.2f} +- {d['sem']:.2f} | pilot {pilot[pk]}")

with open("/vol02/hdd1/ime001/chaord_q3/q2_counts.json", "w") as f:
    json.dump(results, f, indent=1)
