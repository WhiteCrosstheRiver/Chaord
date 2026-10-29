# Open items and decisions, v5 (external Review 11, 2026-10-30)

Read this file before any work. It supersedes v4 and older briefs where they
conflict. D1–D38 stand unless amended here. `pending_rulings.md` R1–R4 are ruled
below (D39–D44); mark them `RULED: see open_items_v5.md`.

- Reviewed HEAD: `ca1a796`.
- The code block below ran on `ca1a796` from the repository root and printed what
  it says (scipy needed). Run it before Z1 and keep it as the D39 reference.

## 0. What the reviewer re-checked on `ca1a796`

- Every X and Y work order from v3/v4 has merged. Stage A and the full Stage B
  ladder (2 tubes × 7 fields × 3 runs, 42,042 blind lifts) are committed.
- Fidelity holds: the blind `dipole_p1` equals the direct numpy arm to the print
  quantum, and both arms give the same E_min under every criterion.
- The registered criterion A (3 × D15 cross-run floor) tests per-block separation,
  not a mean shift. Under it even 0.2 V/Å fails in (10,10). That was the
  reviewer's error in D31/D32, fixed by D39.
- The secondary movers do not hold (block B, per run, (16,16)):
  `ion_pairs.AGG` = [0, 0.17, 0.09] at 0; [0, 1.82, 0] at 0.005; [0, 0, 0.04] at
  0.01; [2.23, 0, 1.27] at 0.02 — a 0-or-2 event with 6 Li⁺, not a dose–response.
  `nr.C4H8O3.peak` = 0.010 / 0.010 / 0.012 / 0.011 from 0 to 0.02 V/Å.
- Stage B departed from Y7: 49.2 Å tubes (not 98.4 Å), 3 runs (not 5) at 0 and
  0.01 V/Å, pilot filling (reservoir fills still running), two LAMMPS builds mixed
  inside the (10,10) ladder (cluster 3 Mar 2020 for 0.005/0.02/0.2, local 22 Jul
  2025 for 0/0.01/0.05/0.1).
- `owner_inputs.md` still says working field "unknown" and model
  "not-yet-provided".

```python
import json, numpy as np
from scipy import stats
J = json.load(open("research/cnt_electrolyte/analysis/full_ladder_results.json"))["jobs"]
for t in (10, 16):
    off = np.array([np.mean([j["direct_p1_blocks"]["A"], j["direct_p1_blocks"]["B"]])
                    for j in J.values() if j["tube"] == t and j["field"] == 0])
    for E in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2):
        on = np.array([j["direct_p1_blocks"]["B"] for j in J.values() if j["tube"] == t and j["field"] == E])
        r = stats.ttest_ind(on, off, equal_var=False)
        print(f"({t},{t}) {E}: shift {on.mean() - off.mean():+.3f}  t {r.statistic:4.1f}  p {r.pvalue:.4f}")
# prints:
# (10,10) 0.005: shift -0.159  t -1.0  p 0.4167
# (10,10) 0.01: shift -0.058  t -0.4  p 0.6895
# (10,10) 0.02: shift -0.039  t -0.4  p 0.6954
# (10,10) 0.05: shift +0.078  t  1.3  p 0.2913
# (10,10) 0.1: shift +0.303  t  4.9  p 0.0164
# (10,10) 0.2: shift +0.549  t  8.1  p 0.0019
# (16,16) 0.005: shift +0.068  t  0.9  p 0.4285
# (16,16) 0.01: shift +0.110  t  2.8  p 0.0859
# (16,16) 0.02: shift +0.229  t  5.3  p 0.0109
# (16,16) 0.05: shift +0.328  t  8.7  p 0.0104
# (16,16) 0.1: shift +0.454  t 12.1  p 0.0059
# (16,16) 0.2: shift +0.580  t 15.5  p 0.0035
```

## 1. Decisions (binding, 2026-10-30)

| ID | Decision |
| --- | --- |
| D39 (R1) | Detection of a field = a test of the mean shift with independent runs as the unit: Welch's t, two-sided, α = 0.05, zero-field run means (blocks A and B averaged) against field-on block-B values. The build-paired contrast is reported as secondary. The D15 3×-floor rule remains only for per-block classification claims. Pre-stated now, before Z3's data exist. |
| D40 | Go/no-go: GO for Chaord as an instrument (fidelity met; E_min Chaord = E_min direct). E_min = 0.02 V/Å for (16,16) and 0.1 V/Å for (10,10) under D39. 0.01 V/Å is open until Z4. R4 (Laya loop) is unblocked, subject to D47. |
| D41 (R2) | The 3-run, 2 × 6 ns ladder is ratified as evidence. One confirmatory set (Z3): (16,16) only; fields 0, 0.01, 0.02 V/Å; 6 new builds each; 98.4 Å tubes; reservoir filling from Y6 when available, else the pilot filling stated; one LAMMPS build; same 1 ns settle + 2 × 6 ns protocol. (10,10) is reported as non-ergodic at weak field (zero-field run means +0.11, −0.01, +0.17; τ_M ≈ 2.7 ns), not chased with plain MD. |
| D42 | A non-P1 statement counts as moved only with (a) a monotone dose–response over at least 3 consecutive fields, (b) Holm-corrected significance across all statements × fields tested, and (c) at least 10 ions of each kind involved for ion-pair statements. Until then, no secondary-mover claim in any report. |
| D43 (R3) | The verdict is recorded on 2026-10-30. The reservoir fills are a separate Y6 deliverable. |
| D44 | (1) One spec example per key stands; A14 is 41/41, close it. (2) `dipole_axis_default: z` ratified. (3) Regenerate the 5 pinned molecular cases physically, one commit each, P2. (4) Frame-charge convention stands until the owner's model arrives. (5) Cluster record accepted. |
| D45 | One frozen paper dataset: one LAMMPS build per figure, a frozen run list in `research/cnt_electrolyte/paper_dataset.json`, no queue writing while an analysis reads (check and refuse), relative paths only in every committed results file. |
| D46 | The Chaord methods paper may use the pilot model, stated as such. Runs for the owner's MLP paper use the owner's Moltemplate model; if it is not in `owner_inputs.md` by 2026-11-06, they use a published, cited carbonate force field chosen by S3 and recorded there. |
| D47 | If the owner reports a labelled DFT set, start R4 retrospectively: pool-based active learning on that set, NEP retrained on subsets chosen at random, by MLP uncertainty and by Chaord-statement novelty, 3 seeds each; report test force error against labels. |

## 2. Work orders

### Z1 · Rewrite the verdict (P0, S2, half a day)
`reports/research/go_no_go.md`: verdict GO for the instrument under D39/D40, with
the table printed by the block above, E_min 0.02 / 0.1 V/Å, criterion A kept as a
labelled per-block result, and the D32→D39 change stated as the reviewer's
correction. Done when the banner no longer says PENDING and every number comes
from a committed script (move the block above into
`research/cnt_electrolyte/analysis/d39_welch.py`).

### Z2 · Withdraw the secondary movers and code D42 (P1, S2, 1 day)
Remove "AGG grows monotonically" and the EMC-peak claim from commit-referenced
reports; add `d42_movers.py` implementing D42 with a test that the current data
yield no mover. Done when the test passes and the reports say "none established".

### Z3 · Confirmatory set (P0, S3, inputs today, ~4–5 days)
As D41. Write the run list and the D39 analysis call into
`research/cnt_electrolyte/confirmatory/PLAN.md` and commit it BEFORE the first job
starts (pre-registration). Done when all 18 jobs are DONE with manifests.

### Z4 · Decide 0.01 V/Å (P0, S5, half a day, after Z3)
Run exactly the committed PLAN.md analysis. Report the result either way; no
other test is added after seeing the data. Done when `go_no_go.md` gains a
"confirmatory" section.

### Z5 · Freeze the paper dataset (P1, S2, 1 day)
As D45. Stop the local Stage B queue first. Done when `paper_dataset.json` lists
every run used in a figure with its build, LAMMPS version and sha256, and a test
refuses analysis while `QUEUE_RUNNING` exists.

### Z6 · The MLP-paper model (P1, S3, owner-gated)
As D46. Done when `owner_inputs.md` names the model, its citation and charge
scaling.

### Z7 · Laya loop, retrospective (P1, S4, owner-gated)
As D47. Done when a report gives error-per-label curves for the three selectors.

### Z8 · Methods-paper outline (P2, S2, after Z1, Z2, Z4)
`docs/paper/outline.md`: claims table, figure list (round-trip table, P1 ladder
with D39 markers, fidelity scatter Chaord vs direct), and the stated limits (pilot
model, narrow-tube non-ergodicity). Done by 2026-11-13.

## 3. Streams

| Stream | Work orders | Starts |
| --- | --- | --- |
| S2 Records | Z1 → Z2 → Z5 → Z8 | now |
| S3 Cluster | Z3 inputs now, then the rest of X7 in idle time; Z6 when the owner answers | now |
| S5 Analysis | Z4 | after Z3 |
| S4 Research | Z7 | when the owner reports a DFT set |

Merge rules (D5) stand: one work order per merge, green CI with the run URL.

## 4. Owner questions (answer in `docs/reviews/owner_inputs.md`)

1. Working field strength(s) in V/Å.
2. The Moltemplate model files or path, and their charge scaling.
3. A labelled DFT set for the CNT electrolyte: yes or no; if yes, number of frames,
   DFT code and functional, with or without field.
