# Open items and decisions, v7 (external Review 13, 2026-11-05)

Read this file before any work. It supersedes v6 where they conflict; D1–D53
stand unless amended here. Reviewed HEAD: `f818df4`. The code block below ran on
`f818df4` from the repository root (scipy needed) and printed what it says.

## 0. What the reviewer re-checked

- Q1 is done (Stage A, Y6, Z3 evidence in git; `corrections.md`; CI check).
- Q2's final report (`fc8bec2`) is correct: per Å in (16,16) the open tube holds
  EC 1.2701, EMC 0.5917, Li⁺ 0.2673, PF6⁻ 0.2127 — far above the pilot packing.
- Q3's plan (`dc2607a`, two minutes after `fc8bec2`) used EC 0.8984/Å, EMC
  0.5578/Å, Li⁺ 0.1541/Å from the uncorrected draft (`caf8b78`). Q3 tested neither
  the pilot nor the reservoir filling.
- Q3 has two reported results (`36daad6`/`go_no_go.md` §14: shift +0.008, p 0.78;
  `0f63a06`: shift +0.035, p 0.1285) and no results file, final data or logs in
  git.
- Independent mass check on the final snapshots:

```python
import numpy as np
def load(path):
    L = open(path).read().splitlines()
    mi = next(k for k, l in enumerate(L) if l.startswith("Masses"))
    masses = {}
    for l in L[mi + 2:]:
        if not l.strip():
            break
        t = l.split(); masses[int(t[0])] = float(t[1])
    ai = next(k for k, l in enumerate(L) if l.startswith("Atoms"))
    rows = []
    for l in L[ai + 2:]:
        if not l.strip():
            break
        rows.append(l.split()[:7])
    a = np.array(rows, float)
    return a[:, 2].astype(int), a[:, 4:7], masses
def tube_density(path, z0=None, z1=None):
    typ, x, masses = load(path)
    m = np.array([masses[t] for t in typ]); tube = typ == 1
    cx, cy = x[tube, 0].mean(), x[tube, 1].mean()
    r = np.hypot(x[:, 0] - cx, x[:, 1] - cy); R = r[tube].mean()
    if z0 is None:
        z0, z1 = x[tube, 2].min(), x[tube, 2].max() + 2.46 / 2
    sel = (~tube) & (r < R) & (x[:, 2] > z0) & (x[:, 2] < z1)
    return m[sel].sum() / 6.02214e23 / (np.pi * R**2 * (z1 - z0) * 1e-24)
print(f"pilot (16,16): {tube_density('research/cnt_electrolyte/data/cnt16_field0.data', 0.0, 49.2):.3f} g/cm3 over the full tube radius")
print(f"reservoir fill (16,16), middle window: {tube_density('research/cnt_electrolyte/fills_cluster/fill_cnt16/final.data', 40.0, 68.0):.3f} g/cm3")
# prints:
# pilot (16,16): 0.596 g/cm3 over the full tube radius
# reservoir fill (16,16), middle window: 0.936 g/cm3
```

## 1. Decisions (binding, 2026-11-05)

| ID | Decision |
| --- | --- |
| D54 | Q3 is void as a filling test. Add it to `docs/records/corrections.md` with both reported numbers and which script produced each; commit Q3's final data, logs and one `q3_results.json`; replace `go_no_go.md` §14 with "void: composition from the uncorrected Q2 draft". |
| D55 | The reservoir filling is the only filling for physics claims. One pre-registered re-run of the Z3 design at it (T1–T3). |
| D56 | Stage B and Z4 stay in the paper, labelled "packed below open-tube uptake". The (10,10) tube is not re-run. |
| D57 | Writing starts today, independent of D55. Outline Nov 6; methods + filling-independent results Nov 9; full draft Nov 16 (replaces Nov 13). |
| D58 | The Nov 6 owner fallbacks (D46, D53) trigger as ruled if `owner_inputs.md` is still unchanged. |
| D59 | One agent per work order (no parallel agents on the same order). Every reported number comes from one committed results file, and the commit message quotes that file's value. |

## 2. Work orders

### T1 · Pre-register the reservoir-filling test (P0, S3, today)
`research/cnt_electrolyte/r13/PLAN.md`, committed before any build:
- (16,16) armchair, 98.4 Å (40 periods), EC 125 / EMC 58 / 24 LiPF6 (Q2 per-Å values
  × 98.4 Å; salt neutralised at the rounded mean of Li⁺ 26.3 and PF6⁻ 20.9). State the
  in-tube Li⁺/PF6⁻ imbalance Q2 measured as a finding, not as the build.
- 6 independent builds, seeds disjoint from all earlier ranges. If the packer jams
  at this density, pack in stages (insert at reduced LJ radius, then relax to the
  floor) and record every attempt; do not lower the counts.
- Fields 0, 0.01, 0.02 V/Å; the Z3 protocol (1 ns zero-field thermalisation, 1 ns
  settle, 2 × 6 ns blocks); one LAMMPS build (cluster 3 Mar 2020).
- The test: D39 Welch, two-sided, α = 0.05, zero-field run means (blocks A+B)
  against block B, n = 6 per group, with the 95% interval (D48). One test per field;
  nothing added after the data.

### T2 · Build and run (P0, S3)
18 jobs. Done when every job's `final.data`, `log.lammps` and thinned series
manifest are in git and the thinned series are in the archive.

### T3 · The one test (P0, S3 or S5, half a day)
Run the committed PLAN script; write `r13_results.json`; add `go_no_go.md` §15 with
the result either way, beside Z4's pilot-filling result.

### T4 · Close Q3 (P1, S2, half a day)
As D54.

### T5 · The manuscript (P0, S5)
- T5a by Nov 6: `docs/paper/outline.md` — claims table (claim → evidence file →
  script) and figures: (1) round-trip and acceptance table, (2) blind P1 against
  field (Stage B, Z4, and the T3 points when available, filling labelled),
  (3) fidelity: Chaord statement against direct calculation, all lifts, (4) an
  example program for a tube frame, (5) the filling chart (pilot vs open-tube uptake).
- T5b by Nov 9: methods; results for the round trip, blind lifting and fidelity.
- T5c: the field section after T3. Limits stated: pilot force field, filling,
  narrow-tube non-ergodicity, cluster floating-point non-reproducibility, 6 runs per
  field.
- Done when `docs/paper/draft.md` is complete by 2026-11-16 and every number in it
  is printed by a committed script.

### T6 · Freeze and archive (P1, S2)
v6 Q4, unchanged: `research/cnt_electrolyte/paper_dataset.json` plus the archive.

### T7 · Last X7 items (P2, S3, idle cluster time only)
v6 Q6, unchanged.

## 3. Streams

| Stream | Work orders | Starts |
| --- | --- | --- |
| S3 Cluster | T1 → T2 → T3, then T7 | now |
| S5 Writing | T5a → T5b → T5c | now |
| S2 Records | T4 → T6 | now |
