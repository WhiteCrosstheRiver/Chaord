# Go/no-go — the D39 gate on the FULL stage-B field ladder

**VERDICT (Review 11, D40, recorded 2026-10-30): GO for Chaord as an
instrument. Fidelity is met — the blind Chaord statements equal the direct
numpy arm to the print quantum on every lift, and both arms give the same
E_min under every criterion tested. Under D39 (detection = a test of the
mean shift with independent runs as the unit: Welch's t, two-sided,
alpha = 0.05), E_min = 0.02 V/Å for (16,16) and 0.1 V/Å for (10,10);
0.01 V/Å is open until the Z3 confirmatory set (D41) reports. The D39
criterion supersedes D32's: the registered 3×-floor rule tested per-block
SEPARATION, not a mean shift — under it even 0.2 V/Å fails in (10,10).
That was the reviewer's own error in D31/D32, corrected in D39
(docs/reviews/open_items_v5.md), pre-stated before the Z3 confirmatory
data exist. The 3×-floor result is kept below as a labelled per-block
classification result, not as the detection gate.**

Analysis stream Y8, Review 10, run on 2026-10-29/30. Everything below is
produced by `research/cnt_electrolyte/analysis/y8_full_ladder.py`
(deterministic, no randomness; the tranche-1 machinery — arms, statement
parser, D15 estimator, LORO classifier — is imported from
`y8_block_analysis.py` unchanged). All numbers land in
`research/cnt_electrolyte/analysis/full_ladder_results.json`.

## 1. What was analysed

| state | jobs | canonical record |
|---|---|---|
| field on + zero, (10,10) | 21: `sb_cnt10_e{0,5,10,20,50,100,200}_r{1,2,3}` | e0/e10/e50/e100 LOCAL full-density (`stageB/jobs/…/prod.dump`, 12,001 frames at 1 ps, manifests + non-smoke DONE, LAMMPS 22 Jul 2025); e5/e20/e200 CLUSTER thinned (`stageB/sb_*/prod_thinned.dump`, every 12th production frame = 12 ps, 1,001 frames, LAMMPS 3 Mar 2020) |
| field on + zero, (16,16) | 21: `sb_cnt16_e{0,5,10,20,50,100,200}_r{1,2,3}` | all CLUSTER thinned (12 ps, 1,001 frames per job) |

Every run is 1 ns settle + 2 × 6 ns blocks (13 ns simulated), started
from the DONE Stage-A zero-field finals (`zf_cnt<tube>_b{1,2,3}`), one
seed pair per job from `stageB/jobs_index.json`. The canonical
assignment is FROZEN as of the work order (the 33 `sb_*` cluster dirs of
which cnt10 e100's canonical copy is the local full-density run, plus
the 12 local cnt10 jobs). Inventory: **174,042 production frames read**
(every frame of every canonical dump, by the arm-2 streamer), **42,042
blind lifts** (1,001 per job — every frame of the 12-ps record cadence),
census (cnt10: EC 8, EMC 5, PF6 2, Li 2 + the 800-C tube; cnt16: EC 37,
EMC 24, PF6 6, Li 6 + the 1600-C tube) and cell charge neutrality
verified on every one of the 174,042 frames, **zero failed static
checks** on any lifted frame.

**Anomaly, disclosed (concurrent local queue).** While this analysis
ran, a still-active local stageB queue (`stageB/QUEUE_RUNNING`, pid
44668) finished `sb_cnt10_e5_r{1,2,3}` locally (new DONE markers +
manifests appeared mid-run) and started local e20/e200 runs (in flight,
in no record here). Per the freeze, e5's canonical record stays the
cluster thinned copy; the fresh local full-density copies were analysed
as duplicates (below). Any re-run of this analysis after the queue
finishes must re-freeze the assignment explicitly.

## 2. Method (the choices, stated)

**Two independent arms (D32's second arm), as in tranche 1.** Arm 1
(Chaord): Y4 reader with the type map of the provenance BUILD data file
(`data/cnt<tube>_bK.data` — never `final.data`), ONE blind lift per
frame (`lift_frame(mode="auto")`, no `field=` kwarg, T = 300.0, one
composition dialect for all 42,042 frames), statement features parsed
from the formatted program text: `dipole_p1`, `li_shell` per species,
`ion_pairs` SSIP/CIP/AGG, `n_r` species radial profiles reduced to peak
/ peak-radius / mean-radius scalars, `density` (accessible),
`linear_density`, `gr_peak` — 23 moving-candidate statements. Arm 2
(direct, no chaord): numpy/stdlib only, molecules by mol id, P1 = mean
solvent μ_z/|μ| with minimum-image offsets, computed on EVERY frame
(dense) plus the 12-ps cadence the lift arm sees.

**Lift coverage (measured, not assumed).** The task brief priced lifts
at 20–30 s; measured cost on this box is 0.10 s (cnt10, 971 atoms) to
0.17 s (cnt16, 2,058 atoms), so FULL coverage is affordable and used:
every frame of the committed record at the 12-ps thinned cadence — 500
frames in block A + 501 in block B per job per arm; no analysis
subsampling beyond the record's own thinning. The 12 local jobs carry
12× denser records; their full density is used by arm 2 (6,000/6,001
frames per block) and the dense-vs-cadence agreement is recorded: worst
block-mean difference 0.0027 over all 12 jobs — the 12-ps cadence costs
nothing visible. Every block mean rests on ≥ 500 frames.

**Blocks** (stage_b_plan.md section 2, verified against the dumps:
thinned timesteps 0, 12,000, …, 12,000,000; local 0, 1,000, …): block A
= production t < 6,000 ps, block B = t ≥ 6,000 ps. Block B is PRIMARY
(section 3.2: block A of a field-on run retains ~27–31 % of the
switch-on transient); block A appears in the sensitivity checks only.
The e0 runs get identical treatment (field stays 0, both blocks clean).

**Floors (D15 estimator, D31 pair set) — now like-for-like.** Per tube
and statement, over THIS batch's e0 block means (6 ns × 2 blocks × 3
runs = 6 blocks, 12 cross-run pairs per tube): floor = max(mean |Δ|,
P90 |Δ|). These supersede Stage A's 1-ns floors (the tranche-1 finding:
short-block scatter inflated them); e0 blocks have exactly the length
and cadence of the field-on blocks, so no length correction remains.

**Detection — two criteria, both computed (the choice is the owner's,
see §8).**

- **A (ordered, as registered in D31/D32 and tranche 1):**
  |mean_on(B) − mean_off| > 3 × floor; mean_on = mean over the 3 runs
  of their block-B means (sem over runs = the primary error bar);
  mean_off = mean over the 6 e0 blocks.
- **B (build-paired secondary, tranche-1 §6.3):** per build, on(B) −
  the SAME build's e0 all-block mean; detected when |mean| > 3 × its
  sem over the 3 builds (2 σ also recorded). Removes the shared
  per-build zero-field offset; assumes that offset survives into the
  field-on run — a model, not the ordered test.

**Classifier (Y8 step 4):** threshold on the `dipole_p1` block value,
midpoint of class means, scored leave-one-RUN-out over both classes (9
folds: 3 field-on + 6 e0 blocks), refitting the threshold per fold.
Variants: all-blocks on both sides; direct-P1 dense and cadence
variants. Runs are the units (stage_b_plan §3.1).

**Linear response (Y8 step 6):** slopes (P1(E) − P1(0))/E at 0.005,
0.01, 0.02 V/Å compared pairwise within 2 σ.

## 3. The two arms agree — D32's comparison is well posed

Max |P1(statement text) − P1(direct)| over all 42,042 lifted frames:
**5.00e-04 — exactly the statement's 3-decimal print quantum** (per-job
maxima 1.9e-04 … 5.00e-04). The blind statement and the direct quantity
are the same measurement to print precision, so E_min(Chaord) =
E_min(direct) by identity at every field and every criterion below;
every number is simultaneously both arms' result.

**Determinism cross-checks (6 duplicate pairs, disclosed).** Two jobs
exist in both chains — cnt10 e100 (cluster copy of the canonical local
run) and cnt10 e5 (fresh local copies of the canonical cluster run).
Same seeds, bit-identical step-0 thermo, different LAMMPS builds
(22 Jul 2025 vs 3 Mar 2020) → chaotically diverging trajectories:
block-B P1 differs by 0.07–0.17 between copies (e.g. e100 r2: cluster
+0.620 vs local +0.447; e5 r2: local −0.186 vs cluster −0.329), frame-wise
max |ΔP1| up to 0.56 over 13 ns. Both copies are valid samples of the
same protocol; the canonical copy is used and the duplicates
independently corroborate the run-to-run spread that sets the floors.
This is a property of cross-build LAMMPS reproduction, not a data error.

## 4. Floors

| statement | cnt10 floor (mean \|Δ\|, P90) | cnt16 floor | within-run \|A−B\| mean |
|---|---|---|---|
| `dipole_p1` | **0.2658** (0.1362, 0.2658) | **0.1401** (0.0851, 0.1401) | 0.115 (cnt10) / 0.032 (cnt16) |
| `fluid.li_shell.C3H4O3` | 0.688 | 1.104 | — |
| `fluid.li_shell.C4H8O3` | 0.698 | 0.429 | — |
| `fluid.li_shell.F6P` | 0.354 | 0.779 | — |
| `fluid.ion_pairs.SSIP`/`.CIP` | 1.000 | 1.732 | — |
| `fluid.ion_pairs.AGG` | 0 (never aggregates in 36 ns of cnt10 e0) | 0.168 (small aggregates in cnt16 e0 r2/r3) | — |
| `fluid.nr.Li+.peak` | 0.0054 | 0.0004 | — |
| `fluid.density`, `fluid.linear_density` | ≤ 2.2e-16 (one ulp) | ≤ 5.6e-17 | — |

Supersession check (cnt10 `dipole_p1`): Stage-A 1-ns floor 0.3935 →
like-for-like 6-ns e0 floor 0.2658 (× 0.68, not the naive √6 = 0.41 —
the offsets are longer-lived than 1 ns). Composition shift: within-run
|A−B| (0.115 mean) and cross-run |Δ| (0.136 mean) are now comparable —
the 6-ns e0 blocks average out much of the metastable per-build
orientation, but the floor stays P90-dominated by the b2 offset (cnt10
e0 block-B means: b1 +0.190, b2 +0.052, b3 +0.205; cnt16: −0.041,
−0.064, +0.019 — cnt16's zero-field offsets are ~3× tighter, which is
why its floor is half of cnt10's).

## 5. The full E-response curve (block B, mean ± sem over 3 runs)

**cnt10 (blind statement ≡ direct P1 to print precision):**

| E (V/Å) | P1 | per-run block-B |
|---|---|---|
| 0 | +0.0917 ± 0.0557 | (all-block +0.114, −0.014, +0.175) |
| 0.005 | −0.0675 ± 0.1548 | (−0.080, −0.329, +0.207) |
| 0.01 (working) | +0.0342 ± 0.1192 | (+0.095, −0.196, +0.204) |
| 0.02 | +0.0525 ± 0.0744 | (−0.077, +0.054, +0.180) |
| 0.05 | +0.1706 ± 0.0201 | (+0.143, +0.159, +0.210) |
| 0.1 | +0.3952 ± 0.0287 | (+0.349, +0.448, +0.389) |
| 0.2 (saturating) | +0.6410 ± 0.0391 | (+0.610, +0.594, +0.719) |

**cnt16:**

| E (V/Å) | P1 | per-run block-B |
|---|---|---|
| 0 | −0.0351 ± 0.0371 | (all-block −0.044, −0.095, +0.033) |
| 0.005 | +0.0325 ± 0.0648 | (−0.053, −0.009, +0.160) |
| 0.01 (working) | +0.0746 ± 0.0122 | (+0.092, +0.081, +0.051) |
| 0.02 | +0.1934 ± 0.0213 | (+0.151, +0.220, +0.209) |
| 0.05 | +0.2928 ± 0.0072 | (+0.281, +0.306, +0.292) |
| 0.1 | +0.4192 ± 0.0051 | (+0.416, +0.412, +0.429) |
| 0.2 (saturating) | +0.5453 ± 0.0053 | (+0.544, +0.537, +0.555) |

Shape, honestly read: cnt16 is a clean saturating-response curve —
tight from 0.01 up (the field overwrites the build offsets), linear
through 0.02, bending above it. cnt10 is flat within errors through
0.02 (b2's metastable negative orientation dominates the spread), then
rises steeply. Stage-A anchor: the 2-ns saturating runs gave P1
+0.46…+0.62 (cnt10) and +0.53…+0.54 (cnt16); the 6-ns block-B values
(+0.641, +0.545) are consistent. **0.2 V/Å caveat (stage_b_plan §3):**
τ_M at the saturating field was measured > 4.6 ns (3/4 fits, cnt16), so
6-ns blocks there may be < 2 τ_M_effective; 17/23 cnt16 statements show
block-A/B offsets beyond 2 × the run sem at 0.2 (dipole_p1 itself does
not: A +0.542 vs B +0.545). The e200 points are the saturation anchor,
not slope points, and their block-B magnitudes are the cleaner read.

## 6. Detection per field (criterion A, the ordered gate; block B)

| tube | E (V/Å) | delta | ratio to floor | z | 3× test |
|---|---|---|---|---|---|
| cnt10 | 0.005 | 0.159 | 0.60× | 1.0 | fail |
| cnt10 | 0.01 (working) | 0.058 | 0.22× | 0.4 | fail |
| cnt10 | 0.02 | 0.039 | 0.15× | 0.4 | fail |
| cnt10 | 0.05 | 0.079 | 0.30× | 1.3 | fail |
| cnt10 | 0.1 | 0.304 | 1.14× | 4.8 | fail |
| cnt10 | 0.2 | 0.549 | 2.07× | 8.1 | **fail** |
| cnt16 | 0.005 | 0.068 | 0.48× | 0.9 | fail |
| cnt16 | 0.01 (working) | 0.110 | 0.78× | 2.8 | fail |
| cnt16 | 0.02 | 0.229 | 1.63× | 5.3 | fail |
| cnt16 | 0.05 | 0.328 | 2.34× | 8.7 | fail |
| cnt16 | 0.1 | 0.454 | 3.24× | 12.1 | **PASS** |
| cnt16 | 0.2 | 0.580 | 4.14× | 15.5 | **PASS** |

Criterion A detects: **cnt16 at 0.1 and 0.2 V/Å only; cnt10 at NO field
on the ladder — even the saturating field peaks at 2.07× the floor**
(cnt10's zero-field metastability keeps 3 × floor at 0.797 while the
full saturation response over e0 is 0.549; this is the tranche-1 §6.4
prediction realised). The direct arm gives identical verdicts (ratios
0.60/0.22/0.15/0.29/1.15/2.08 and 0.48/0.78/1.63/2.34/3.24/4.14).

**Criterion B (build-paired), mean ± sem over the 3 builds:**

| tube | E | paired Δ | z | 2σ | 3σ |
|---|---|---|---|---|---|
| cnt10 | 0.005 | −0.159 ± 0.102 | −1.6 | — | — |
| cnt10 | 0.01 | −0.058 ± 0.064 | −0.9 | — | — |
| cnt10 | 0.02 | −0.039 ± 0.078 | −0.5 | — | — |
| cnt10 | 0.05 | +0.079 ± 0.047 | 1.7 | — | — |
| cnt10 | 0.1 | +0.304 ± 0.079 | 3.8 | yes | **yes** |
| cnt10 | 0.2 | +0.549 ± 0.032 | 17.0 | yes | **yes** |
| cnt16 | 0.005 | +0.068 ± 0.040 | 1.7 | — | — |
| cnt16 | 0.01 (working) | +0.110 ± 0.047 | 2.3 | **yes** | — |
| cnt16 | 0.02 | +0.229 ± 0.044 | 5.2 | yes | **yes** |
| cnt16 | 0.05 | +0.328 ± 0.041 | 8.0 | yes | yes |
| cnt16 | 0.1 | +0.454 ± 0.032 | 14.2 | yes | yes |
| cnt16 | 0.2 | +0.580 ± 0.032 | 18.2 | yes | yes |

Criterion B (3σ) detects: **cnt16 from 0.02 V/Å, cnt10 from 0.1 V/Å.**
At 2σ it additionally detects cnt16 at exactly the working field
(z 2.3 — the on side is tight, +0.075 ± 0.012, and the off side carries
the zero-field width). At the working field the weak response is real
and consistent on cnt16 (all three runs positive, +0.051…+0.092) but
its evidence is 0.78× the ordered floor / 2.3σ paired — suggestive, not
detected, under every registered rule.

**LORO classifier (dipole_p1, leave-one-run-out, primary variant
on-block-B vs all e0 blocks; identical for the direct arm in every
variant):**

| E (V/Å) | 0.005 | 0.01 | 0.02 | 0.05 | 0.1 | 0.2 |
|---|---|---|---|---|---|---|
| cnt10 | 2/9 | 5/9 | 4/9 | 6/9 | **9/9** | **9/9** |
| cnt16 | 5/9 | 7/9 | **9/9** | **9/9** | **9/9** | **9/9** |

Perfect separation: cnt16 from 0.02 V/Å, cnt10 from 0.1 V/Å. At the
working field cnt16 scores 7/9 (on 3/3, off 4/6 — the misclassified
zero-field blocks are again the offset-carrying builds), cnt10 5/9.
All-blocks sensitivity: same thresholds (12/12 from cnt16 0.02, cnt10
0.1).

## 7. E_min per tube and per arm (smallest ladder field detected)

| criterion | cnt10 Chaord | cnt10 direct | cnt16 Chaord | cnt16 direct |
|---|---|---|---|---|
| A: ordered 3× cross-run floor | not detected on 0–0.2 | not detected | 0.1 | 0.1 |
| B: build-paired ≥ 3σ | 0.1 | 0.1 | 0.02 | 0.02 |
| B: build-paired ≥ 2σ | 0.1 | 0.1 | **0.01** | **0.01** |
| LORO perfect separation | 0.1 | 0.1 | 0.02 | 0.02 |

## 8. The verdict history: D32 (per-block criterion, superseded) → D39 (mean-shift)

**D32 condition 1** (Chaord's blind statements detect at the same
smallest field as direct P1 without chaord, within one ladder step):
**met exactly** — the two arms are the same measurement to print
precision (§3), so E_min(Chaord) = E_min(direct) at every criterion.

**D32 condition 2** (that field at or below the working field, 0.01
V/Å): **depends on the detection criterion — the owner's call**, as
flagged in the tranche-1 report (§6.4). Both result sets are present;
neither is assumed; nothing here loosens the registered rule. Which
fields each criterion detects, plainly:

| ruling ("detect" =) | cnt10 detects | cnt16 detects | E_min(cnt10) | E_min(cnt16) | condition 2 met? |
|---|---|---|---|---|---|
| A: 3× cross-run e0 floor (the registered rule) | none, 0–0.2 | 0.1, 0.2 | not detected | 0.1 | **NO** |
| B: build-paired ≥ 3σ | 0.1, 0.2 | 0.02, 0.05, 0.1, 0.2 | 0.1 | 0.02 | **NO** |
| B: build-paired ≥ 2σ (loosest, not registered anywhere before this report) | 0.1, 0.2 | 0.01, 0.02, 0.05, 0.1, 0.2 | 0.1 | **0.01** | cnt16 only, exactly at the working field |
| paired-D15 strict (3× the paired deltas' own cross-build floor) | 0.2 | 0.1, 0.2 | 0.2 | 0.1 | **NO** |
| LORO classifier, perfect 9/9 | 0.1, 0.2 | 0.02, 0.05, 0.1, 0.2 | 0.1 | 0.02 | **NO** |

- Under criterion A (the only rule every prior document registered):
  **NO-GO.** Nothing detected at or below the working field on either
  tube; on cnt10 nothing on the whole ladder.
- Under criterion B at 3σ, or under the LORO reading: **NO-GO** (E_min
  one to three ladder steps above the working field).
- Under criterion B at 2σ: the answer CHANGES for cnt16 only —
  detection at exactly 0.01 V/Å on a marginal z = 2.3 with 3/3 on-side
  runs but 4/6 off-side LORO; cnt10 still at 0.1 V/Å. A split verdict
  (cnt16 yes / cnt10 no) that the current D32 text — one gate field —
  does not contemplate.
- The strict paired-D15 variant (floor = D15 over the paired deltas'
  own cross-build spread — the conservative analogue of criterion B,
  which asks the response to be large against the build-to-build
  variation of the response itself) is MORE conservative than the 3σ
  reading: it detects cnt16 at 0.1 and 0.2 (ratios 4.5, 5.8) and cnt10
  at 0.2 only (ratio 5.4); ratios ≤ 2.6 everywhere else. Nothing at or
  below the working field under it either.

**Superseded record (kept for the audit trail).** Under D32's per-block
criterion as written, the answer was PENDING-OWNER-RULING: if the owner ruled that
"detect" means the ordered 3×-floor test, the paired test at 3σ, or the
LORO classifier, the D32 verdict is **NO-GO**: Chaord stays the builder
and recorder for the owner's MLP paper, off its critical path, and R4
does not start. If the owner rules the paired 2σ contrast carries
detection, cnt16 detects at the working field marginally while cnt10
does not — and the owner must then also decide what a tube-split
verdict means for D32. The conservative registered reading is NO-GO.

## 9. Secondary movers — NONE ESTABLISHED (D42)

**Withdrawn per Review 11 (D42) and the reviewer's own recheck on
ca1a796:** the candidate movers below do not survive the D42 bar
(monotone dose-response over >= 3 consecutive fields + Holm-corrected
significance across all statements x fields + >= 10 ions of each kind
for ion-pair statements):

* `fluid.ion_pairs.AGG` per run at (16,16): [0, 0.17, 0.09] at 0 V/A;
  [0, 1.82, 0] at 0.005; [0, 0, 0.04] at 0.01; [2.23, 0, 1.27] at 0.02 —
  a 0-or-2 event with 6 Li+, not a dose-response.
* `fluid.nr.C4H8O3.peak` = 0.010 / 0.010 / 0.012 / 0.011 from 0 to
  0.02 V/A — flat, not rising (the tranche-1 "EMC peak rises" reading
  does not hold on the full ladder).

**No secondary-mover claim is established in any report.** The
per-block ratios below are kept as the superseded tranche-1 record
(audit trail only; they are per-block classification observations, not
established movers):

<details><summary>superseded tranche-1 table</summary>

| E | statement | on vs off | floor | ratio | z | paired z | reading |
|---|---|---|---|---|---|---|---|
| 0.005 | `fluid.ion_pairs.AGG` | 0.607 ± 0.607 vs 0.061 | 0.168 | 3.26× | 0.90 | 0.96 | one run of three aggregates — sporadic |
| 0.005 | `fluid.nr.C3H4O3.peak_r` | 6.42 ± 0.36 vs 7.01 Å | 0.275 | 2.16× | 1.61 | −1.45 | driven by r1 |
| 0.005 | `fluid.nr.Li+.peak` | 0.0076 ± 0.0043 vs 0.0031 | 0.0004 | 11.8× | 1.04 | 1.06 | quantized peak, tiny floor |
| 0.01 | `fluid.nr.C3H4O3.peak_r` | 6.42 ± 0.69 vs 7.01 Å | 0.275 | 2.18× | 0.86 | −0.95 | not coherent |
| 0.01 | `fluid.nr.C4H8O3.peak` | 0.0114 ± 0.0003 vs 0.0102 | 0.0006 | 2.00× | 3.91 | 3.18 | withdrawn: flat on the full ladder |

</details>

D42's implementing script: `research/cnt_electrolyte/analysis/d42_movers.py`
(reports "none established" on the current data; see its test).

## 10. Linear response over 0.005–0.02 V/Å (Y8 step 6)

Slopes (P1(E) − P1(0))/E, per V/Å, with run-level errors:

| tube | @0.005 | @0.01 | @0.02 | all pairs within 2σ |
|---|---|---|---|---|
| cnt16 | 13.5 ± 14.9 | 11.0 ± 3.9 | 11.4 ± 2.1 | **yes** |
| cnt10 | −31.9 ± 32.9 | −5.8 ± 13.2 | −2.0 ± 4.7 | yes (all consistent with 0) |

cnt16: P1 ∝ E holds through the low-ladder region with a consistent
slope ~11 per V/Å; above 0.02 the slope falls (6.6 at 0.05, 4.5 at 0.1,
2.9 at 0.2) — the approach to saturation, as expected. cnt10: the
weak-field slope is unconstrained (b2's metastable orientation
dominates; its 1σ band includes zero and ±4× the cnt16 slope). What CAN
be concluded: no contradiction with linear response on either tube.
What CANNOT: a slope measurement at or below the working field on cnt10
— that would need the run-length upgrade (5 τ_M blocks), not more
fields.

## 11. Block-protocol checks and anomalies

- **Block A vs B:** dipole_p1 offsets stay small through 0.05 (≤ 0.05
  absolute); at 0.1 the cnt10 block-A mean retains ~8 % transient
  (+0.367 vs +0.395); at 0.2 the A/B agreement degrades broadly
  (17/23 cnt16 statements beyond 2 × run sem — the stage_b_plan §3
  saturation caveat, τ_M > 4.6 ns there). Block B is primary
  everywhere; e200/e100 are anchors, not slope points.
- **Dense vs 12-ps cadence (arm 2):** worst block-mean difference
  0.0027 over all 12 local jobs — the record cadence costs nothing.
- **Provenance:** 12 local jobs with manifests (sha256 of every
  input/output); all 42 jobs in `stageB/jobs_index.json` with builds,
  seeds and zf sources; cluster jobs verified finished
  (`Total wall time`, 1,001 frames at exactly 12 ps, timesteps
  0…12,000,000). Every lifted program's non-moving statements identical
  across each job's 1,001 lifts; zero failed static checks.
- **Anomalies:** (1) the concurrent local queue (§1) — e5 local
  duplicates cross-checked, e20/e200 local runs excluded (in flight);
  (2) cross-build trajectory divergence between duplicate runs (§3) —
  disclosed, floors price such spread; (3) cnt16 e0 contains small
  ion-pair aggregates (AGG 0.06–0.17), so cnt16's AGG floor is nonzero
  while cnt10's is exactly zero.

## 12. Reproduce

    python research/cnt_electrolyte/analysis/y8_full_ladder.py

reads all 42 canonical jobs (174,042 frames dense, both arms), lifts
42,042 frames blind, and writes
`research/cnt_electrolyte/analysis/full_ladder_results.json` (floors,
deltas, ratios, per-run and per-build values, paired contrasts,
classifier folds, linear-response fits, block checks, duplicate
cross-checks, provenance and the freeze/anomaly statement). Per-job
progress is checkpointed to `full_ladder_checkpoint.jsonl` (one JSON
line per job), so an interrupted run resumes; wall ≈ 1.7 h on the
development box (the record run was interrupted externally at job
31/42 and resumed from the checkpoint without recomputation).
`--smoke` runs 4 frames per job as a pipeline test (never for the
record); `--fresh` ignores the checkpoint.

Provenance chain per job: `stageB/jobs_index.json` (field, build, seeds,
zf source) → `stageA/jobs_index.json` (zf build → `data/cnt<tube>_bK.data`
type map, D34) → inputs generated by `stageB/stageB_inputs.py` from
`params.py` physics; local jobs additionally
`stageB/manifests/<job>.json` (sha256). Analysis decisions: D31
(ensembles, zero-field floors, LORO), D15 (floor estimator), D32 (two
arms, verdict conditions), D29 (blind dipole statement), D30 (working
field 0.01 V/Å), stage_b_plan.md sections 2–3 (block protocol, block-B
primary, saturation caveat). Tranche-1 record:
`analysis/y8_block_analysis.py` + `tranche1_results.json` (its cnt10
e10/e50 numbers reproduce here at the 12-ps cadence: +0.034 ± 0.119 and
+0.171 ± 0.020 vs tranche-1's +0.036 ± 0.120 and +0.173 ± 0.021).
