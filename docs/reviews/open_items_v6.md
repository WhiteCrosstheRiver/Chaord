# Open items and decisions, v6 (external Review 12, 2026-11-01)

Read this file before any work. It supersedes v5 where they conflict; D1–D47
stand unless amended here. Reviewed HEAD: `4de4d14`. The code block below ran on
`4de4d14` from the repository root (scipy needed) and printed what it says.

## 0. What the reviewer re-checked

- Z1, Z2, Z3, Z4 and X7 item 2 are done. PLAN.md (`fb50f3f`, 2026-10-31 11:44)
  predates the first submission (`1113f15`, 12:05) and was never edited after.
- Z4 reproduces from `confirmatory_results.json`:

```python
import json, numpy as np
from scipy import stats
J = json.load(open("research/cnt_electrolyte/analysis/confirmatory_results.json"))["jobs"]
off = np.array([np.mean([j["direct_p1_blocks"]["A"], j["direct_p1_blocks"]["B"]]) for j in J.values() if j["field"] == 0])
for E in (0.01, 0.02):
    on = np.array([j["direct_p1_blocks"]["B"] for j in J.values() if j["field"] == E])
    r = stats.ttest_ind(on, off, equal_var=False)
    print(f"{E}: shift {on.mean() - off.mean():+.3f}  t {r.statistic:.2f}  p {r.pvalue:.4f}")
# prints:
# 0.01: shift +0.107  t 2.85  p 0.0246
# 0.02: shift +0.187  t 7.97  p 0.0000
```

- Records gap: `b994426` says "final.data + log.lammps committed" but adds only
  `fills_cluster/fill_series_manifest.json`; `1957a08` adds only
  `confirmatory/series_manifest.json` and a `.gitignore`. No Z3 or Y6 run evidence
  is in git.
- The Y6 reservoir-filling analysis has not been run.
- `owner_inputs.md` is unchanged since 2026-10-15.

## 1. Decisions (binding, 2026-11-01)

| ID | Decision |
| --- | --- |
| D48 | Accept Z4: 0.01 V/Å detected in (16,16). Report shift +0.107 with a 95% confidence interval (Welch–Satterthwaite), p = 0.025, n = 6 per group, and the linear-response slopes (10.7 and 9.4 per V/Å from 0.01 and 0.02). Never write "detected" without the interval. |
| D49 | Experiment freeze for the methods paper. New simulations only for (a) Q3, if Q2 finds a species' count per Å differs from the pilot's by more than 5%, and (b) Q6. |
| D50 | Every job's `final.data`, `log.lammps` and manifest go into git. Thinned series go into one archive (GitHub release assets or Zenodo) with sha256. Over-claiming commit messages are listed in `docs/records/corrections.md`; history is not rewritten. A commit message may only name files the commit contains (add a check to the pre-commit or CI). |
| D51 | The manuscript is a Chaord methods paper for a methods journal (J. Chem. Theory Comput. or Digital Discovery). Draft in `docs/paper/` by 2026-11-13. It uses the pilot force field and says so. |
| D52 | MLP-paper fallback (D46) stands for 2026-11-06. |
| D53 | Without a DFT-set answer by 2026-11-06, the Laya loop moves to the MLP project; the methods paper claims nothing about DFT labels. |

## 2. Work orders

### Q1 · Evidence into git (P1, S2, half a day)
Commit `final.data`, `log.lammps` and manifests for every Z3 (18) and Y6 (2) job,
and any Stage A/B job missing them. Write `docs/records/corrections.md` listing
`b994426`, `1957a08` and any other commit whose message names files it lacks.
Add the D50 check. Done when `git ls-files` shows the files and the check fails
on a deliberately over-claiming test commit.

### Q2 · Reservoir filling (P1, S3, 1 day)
From the fill series: molecules and ions per Å in the middle of each tube (more
than 1 nm from either mouth), block means and errors. Compare per species with the
pilot compositions per Å ((10,10): EC 8, EMC 5, LiPF6 2 per 49.2 Å; (16,16): EC 37,
EMC 24, LiPF6 6 per 49.2 Å). Report in `reports/research/filling.md`.

### Q3 · Filling check (conditional, S3)
Only if Q2 finds a difference above 5% for any species in (16,16): 3 builds at the
reservoir filling, fields 0 and 0.01 V/Å, the Z3 protocol, the D39 test written
into a PLAN before submission. Report the shift beside Z4's.

### Q4 · Freeze and archive (P1, S2, 1 day, after Q1)
`research/cnt_electrolyte/paper_dataset.json`: every run used in a figure, with
build, LAMMPS version, sha256 and archive URL. Relative paths only. Analysis code
refuses to run while `QUEUE_RUNNING` exists.

### Q5 · The manuscript (P0, S5)
- Q5a by 2026-11-03: `docs/paper/outline.md` with the claims table (claim →
  evidence file → script) and the figure list.
- Q5b: four figures from committed results only: (1) round-trip and acceptance
  table, (2) blind P1 against field for both tubes with D39 markers and the
  confirmatory points, (3) fidelity: Chaord statement against direct calculation
  for all 60,060 lifts, (4) one example program text for a tube frame.
- Q5c: methods and results first, then introduction and discussion. Limits
  stated: pilot force field, narrow-tube non-ergodicity, cluster floating-point
  non-reproducibility, 3–6 runs per field.
- Done when `docs/paper/draft.md` is complete by 2026-11-13 and every number in it
  is printed by a committed script.

### Q6 · Last X7 items (P2, S3)
Relaxed defects in Cu (EAM, 300 K and 0.8 Tm, 20 frames each) for A4, and
temperature resolution (ΔT 5/10/20% at N 2048 and 8192) for A5, as specified in v3
X7.

## 3. Streams

| Stream | Work orders | Starts |
| --- | --- | --- |
| S2 Records | Q1 → Q4 | now |
| S5 Writing | Q5a → Q5b → Q5c | now |
| S3 Checks | Q2 → (Q3) → Q6 | now |

## 4. Owner questions (in `docs/reviews/owner_inputs.md`, by 2026-11-06)

1. Working field strength(s) in V/Å, or "don't know".
2. Moltemplate model files or path, or "don't know".
3. A labelled DFT set for the CNT electrolyte: yes (frames, code, functional,
   field or not) or no.
