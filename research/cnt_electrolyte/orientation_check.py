# Chaord-free: does each solvent molecule's axial dipole relax during the 500 ps after the ramp?
import numpy as np
RUN = "research/cnt_electrolyte/runs/"

def atoms(path):                       # LAMMPS data file, Atoms # full: id mol type q x y z
    L = open(path).read().splitlines()
    i = next(k for k, l in enumerate(L) if l.startswith("Atoms"))
    rows = []
    for l in L[i + 2:]:
        if not l.strip():
            break
        rows.append(l.split()[:7])
    a = np.array(rows, float)
    return a[np.argsort(a[:, 0])]

def snapshots(path):                   # LAMMPS dump: id mol type q x y z
    out, f = [], open(path)
    for line in f:
        if line.startswith("ITEM: NUMBER OF ATOMS"):
            n = int(next(f)); next(f)
            box = np.array([next(f).split()[:2] for _ in range(3)], float); next(f)
            a = np.array([next(f).split() for _ in range(n)], float)
            out.append((box[:, 1] - box[:, 0], a[np.argsort(a[:, 0])]))
    return out

for name in ["cnt10_field0", "cnt16_field0", "cnt10_field1", "cnt16_field1"]:
    top = atoms(RUN + name + "/final.data")
    mol, q = top[:, 1].astype(int), top[:, 3]
    solvent = [np.where(mol == m)[0] for m in np.unique(mol) if (mol == m).sum() in (10, 15)]  # EC, EMC
    cos = []
    for L, a in snapshots(RUN + name + "/equil.dump") + snapshots(RUN + name + "/prod.dump")[1:]:
        row = []
        for idx in solvent:
            d = a[idx, 4:7] - a[idx[0], 4:7]
            d -= L * np.round(d / L)
            mu = (q[idx, None] * d).sum(0)
            row.append(mu[2] / np.linalg.norm(mu))
        cos.append(row)
    c = np.array(cos)                  # frames 50..550 ps
    never = int((~((c > 0.5).any(0) & (c < -0.5).any(0))).sum())
    print(f"{name}: P1 range {c.mean(1).min():+.2f}..{c.mean(1).max():+.2f}; "
          f"{never} of {c.shape[1]} molecules never reverse; "
          f"per-molecule 500 ps mean cos, sd {c.mean(0).std():.2f}")
# prints:
# cnt10_field0: P1 range -0.24..+0.25; 7 of 13 molecules never reverse; per-molecule 500 ps mean cos, sd 0.61
# cnt16_field0: P1 range +0.03..+0.20; 38 of 61 molecules never reverse; per-molecule 500 ps mean cos, sd 0.57
# cnt10_field1: P1 range +0.55..+0.84; 10 of 13 molecules never reverse; per-molecule 500 ps mean cos, sd 0.38
# cnt16_field1: P1 range +0.46..+0.71; 27 of 61 molecules never reverse; per-molecule 500 ps mean cos, sd 0.30

# --- Census and value-only check quoted in RESULTS.md section 5 (Review 10,
# docs/reviews/open_items_v4.md, work order Y1: its first block, and the
# measured-P1 threshold corollary). Every number RESULTS.md quotes that does
# not come from analyze_first_result.py is printed above or below. numpy and
# stdlib only, no chaord; run from the repository root.
import json
S = json.load(open(RUN + "first_result_summary.json"))
for on in (False, True):
    texts = [t for k, t in S["program_texts"].items() if S["systems"][k.split("#")[0]]["field"] == on]
    print("field on: " if on else "field off:", sum("field_dipole_p1" in t for t in texts), "of", len(texts), "programs carry the P1 line")
# prints:
# field off: 0 of 42 programs carry the P1 line
# field on:  42 of 42 programs carry the P1 line

R = S["records"]
p1 = np.array([r["p1_measured"] for r in R])
field = np.array([r["field"] for r in R])
d10 = np.array([r["system"].startswith("cnt10") for r in R])

def correct(train, test):              # threshold = midpoint of class means, as in RESULTS.md section 5
    t = 0.5 * (p1[train][field[train]].mean() + p1[train][~field[train]].mean())
    return int(((p1[test] > t) == field[test]).sum())

loo = sum(correct(np.arange(len(R)) != i, np.array([i])) for i in range(len(R)))
print(f"threshold on records[].p1_measured: leave-one-out {loo}/{len(R)}; "
      f"cnt10->cnt16 {correct(d10, ~d10)}/{int((~d10).sum())}; "
      f"cnt16->cnt10 {correct(~d10, d10)}/{int(d10.sum())}; "
      f"measured P1 ranges field0 {p1[~field].min():+.3f}..{p1[~field].max():+.3f}, "
      f"field1 {p1[field].min():+.3f}..{p1[field].max():+.3f}")
# prints:
# threshold on records[].p1_measured: leave-one-out 84/84; cnt10->cnt16 42/42; cnt16->cnt10 42/42; measured P1 ranges field0 -0.245..+0.191, field1 +0.460..+0.838
