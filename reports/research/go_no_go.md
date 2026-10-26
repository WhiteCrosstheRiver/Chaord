# Go/no-go — the D32 second arm on the tranche-1 field ladder

**DRAFT — PRELIMINARY. One tube (10,10), one tranche (0.01 and 0.05 V/Å,
3 runs each) of the stage-B ladder. This file does NOT declare the D32
gate: per the work order (docs/reviews/open_items_v4.md Y8, decision
D32) the verdict is recorded when the ladder completes. What is recorded
here is the full Y8 analysis of the first tranche, honestly, with its
negative result at the working field and the structural reason for it.**

Analysis stream Y8, Review 10, run on 2026-10-26. Everything below is
produced by `research/cnt_electrolyte/analysis/y8_block_analysis.py`
(deterministic, no randomness) from the committed run data; the numbers
land in `research/cnt_electrolyte/analysis/tranche1_results.json`.

## 1. What was analysed

| state | jobs | field | per job | source |
|---|---|---|---|---|
| field on, working | `sb_cnt10_e10_r{1,2,3}` | 0.01 V/Å | 1 ns settle + 2 x 6 ns blocks, 1 ps dumps (12,001 frames) | stage-B tranche 1, started from `zf_cnt10_b{1,2,3}` finals (manifests) |
| field on | `sb_cnt10_e50_r{1,2,3}` | 0.05 V/Å | same | same |
| field off (zero) | `zf_cnt10_b{1,2,3}` | 0 | 2 ns production, 1 ps dumps (2,001 frames) | stage A, independent builds b1/b2/b3 |

The mirrored build `zf_cnt10_b1m` is excluded (stage_b_plan.md section 2:
the symmetry instrument, not an independent start). The stage-B e0 jobs
are queued, not run (smoke dumps only) — the zero-field side is the
stage-A zf runs, per D31.

Inventory: **78,009 production frames read** (every frame of all 9 jobs,
by the arm-2 streamer), **1,809 blind lifts** (9 jobs x 201 subsampled
frames), zero failed static checks on any lifted frame, cell charge
neutral and molecular census (EC 8, EMC 5, PF6 2, Li 2, tube 800 C)
verified on every one of the 78,009 frames.

## 2. Method (the choices, stated)

**Two independent arms (D32's second arm).** Arm 1 (Chaord): the Y4
reader (`chaord.io.lammps.read_lammps_dump`, D34) with the type map of
the provenance BUILD data file (`data/cnt10_bK.data`, the zf build the
run started from — the first-result convention, never `final.data`);
the reader parses every frame of each dump (it has no skipping path)
and returns the stride subsample for lifting, then ONE blind lift per
frame: `lift_frame(mode="auto")`, **no `field=` kwarg**, T = 300.0
(protocol metadata), one composition dialect for all 1,809 frames.
Statement features are parsed from the formatted program text (the
record), region-aware, extended to the Y5 set: `dipole_p1`, `li_shell`
per species, `ion_pairs` SSIP/CIP/AGG, `n_r` species radial profiles,
`density` (accessible), `linear_density`, `gr_peak`. Arm 2 (direct, no
chaord): numpy/stdlib only, the `orientation_check.py` pattern — its
own dump streamer (which is what touches every one of the 78,009
frames), molecules by mol id (never a chaord bond graph), P1 = mean
over the solvent (EC, EMC) of mu_z/|mu| with minimum-image offsets and
the dump's own charges. The `n_r` arrays are reduced to three scalars
per species, stated here: the profile's peak value, the peak bin's
radius, and the density-weighted mean radius.

**Subsampling (stated).** The lift arm lifts every SB_STRIDE-th frame:
stride 60 for the 12,001-frame sb dumps (201 frames per job = 100 in
block A + 101 in block B), stride 10 for the 2,001-frame zf dumps (201
per job = 100 + 101), so both sides carry the same frame count per
block. Arm 2 additionally computes P1 on EVERY frame (the dense
variant), which is what quantifies the subsampling cost. The
subsampling noise enters both sides symmetrically and the floors are
derived from same-cadence zf blocks, so the D15 floors include it.

**Blocks** (stage_b_plan.md section 2, verified against the dumps:
prod.dump timesteps 0..12,000,000 ps after `reset_timestep 0`): block A
= production t < 6,000 ps (frames 0-5999), block B = t >= 6,000 ps
(frames 6000-12000). Block B is the primary block (section 3.2 of the
plan: block A of a field-on run retains ~27% of the switch-on transient
for cnt10); block A is disclosed as transient-biased and appears only in
sensitivity checks. zf blocks: two 1-ns blocks per run (split at 1,000
ps).

**Floors (D15 estimator, D31 pair set).** Per statement, over the zf
BLOCK MEANS: every pair of blocks from DIFFERENT runs (3 builds x 2
blocks -> 12 pairs), floor = max(mean |Delta|, P90 |Delta|). Choice
stated: zf blocks are 1 ns while the sb primary blocks are 6 ns; a
block-mean's variance scales ~ 2 tau/L, so the 1-ns zf floors
OVERESTIMATE the 6-ns block noise — conservative, never looser.

**Detection (the ordered gate).** Per field and statement:
|mean_on(B) - mean_off| against 3 x floor; mean_on = mean over the 3
runs of their block-B means (sem over the 3 runs, D31); mean_off = mean
over the 6 zf blocks. Secondary: the Gaussian z with the two run-level
sems. **Classifier**: threshold on the `dipole_p1` block value (the RD4
evidence — the same quantity arm 2 measures), threshold = midpoint of
class means, scored by leave-one-RUN-out over both classes (6 folds;
each fold refits the threshold without the held-out run and scores its
blocks).

## 3. The two arms agree — the D32 comparison is well posed

Max |P1(statement text) − P1(direct)| over the shared frames of each
job: 4.92e-04 to 5.00e-04 — exactly the statement's 3-decimal print
quantum. The blind statement and the direct quantity are the same
measurement to print precision. Independent cross-check: arm 2's dense
all-frame zf P1 means (+0.0567, −0.2479, +0.1343 for b1/b2/b3) reproduce
stage A's committed `stageA/analysis/per_job.json` values digit for
digit. Consequently the two arms cannot disagree about E_min on this
tranche: D32's "same smallest field" condition is met by identity, and
every number below is simultaneously the Chaord result and the direct
result (the direct dense variant, on all frames, agrees with the
subsampled one to < 0.01 in every block mean).

## 4. Floors

| statement | floor (D15) | mean \|Delta\| | P90 | within-run mean \|A−B\| |
|---|---|---|---|---|
| `dipole_p1` | **0.3935** | 0.2591 | 0.3935 | 0.0749 |
| `fluid.gr_peak_h` | 0.8338 | 0.5120 | 0.8338 | — |
| `fluid.gr_peak_r` | 0.8648 | 0.5661 | 0.8648 | — |
| `fluid.li_shell.C3H4O3` | 1.5004 | 0.9078 | 1.5004 | — |
| `fluid.li_shell.C4H8O3` | 1.4248 | 0.8201 | 1.4248 | — |
| `fluid.li_shell.F6P` | 0.4473 | 0.2868 | 0.4473 | — |
| `fluid.ion_pairs.SSIP` / `.CIP` | 0.9970 | 0.6050 | 0.9970 | — |
| `fluid.nr.Li+.mean_r` / `.peak_r` | 2.03 / 2.18 | 1.34 / 1.45 | — | — |
| `fluid.density`, `fluid.linear_density`, `fluid.ion_pairs.AGG` | 0 | 0 | 0 | — |

The structural constants (density, linear_density: fixed mass in a fixed
accessible volume; AGG: no aggregated ion pair anywhere) have zero floor
and zero delta to any meaningful precision (linear_density and AGG are
bit-exact 0; density's floor is one ulp, 2.2e-16, from float summation
order) — they cannot move by construction, like the chirality.

The `dipole_p1` floor is dominated by the **long-lived per-build
zero-field orientation offsets**: within-run |A−B| averages 0.075
([0.134, 0.036, 0.055] for b1/b2/b3) while cross-run |Delta| averages
0.259 — the cross-build offsets contribute ~ 0.25 in quadrature, 3.4x
the within-run spread. Stage A already knew this shape (b2's whole 2-ns
production sits at P1 ≈ −0.25, b3's at +0.13, and the mirror build
flips the sign): the zero-field orientation is metastable on scales
longer than the 2-ns zf production. The D15 floor, as ordered, prices
that metastability into every detection threshold.

## 5. Detection at the two tranche fields (block B primary)

| field | statement | mean on | mean off | delta | floor | ratio | z | 3x test |
|---|---|---|---|---|---|---|---|---|
| 0.01 V/Å | `dipole_p1` | +0.0364 ± 0.1197 | −0.0185 ± 0.1175 | 0.0549 | 0.3935 | 0.14x | 0.33 | **fail** |
| 0.05 V/Å | `dipole_p1` | +0.1732 ± 0.0209 | −0.0185 ± 0.1175 | 0.1917 | 0.3935 | 0.49x | 1.61 | **fail** |

No statement other than `dipole_p1` passes either: the largest ratios
anywhere are 0.89x (`li_shell.F6P` at 0.05) and 0.83x
(`ion_pairs.CIP`/`.SSIP` at 0.05). The all-blocks sensitivity (block A +
block B) changes nothing (on-side means +0.033 at 0.01, +0.177 at 0.05).

**E_min(Chaord) = E_min(direct P1) = not reached on this tranche**:
neither 0.01 nor 0.05 V/Å is detected, so E_min > 0.05 V/Å under the
ordered 3x-floor rule, i.e. **not at or below the working field**. The
classifier (LORO) does not separate either: 4/9 blocks at 0.01 (on 2/3,
off 2/6), 6/9 at 0.05 (on 3/3, off 3/6); the all-blocks sensitivity
variant gives 6/12 and 9/12 (at 0.05 the on-side is 6/6 but the
zero-field side stays at 3/6) — identical for both arms in every
variant. The classifier's off-side errors are exactly the
positive-offset zf blocks (b3: +0.107/+0.162, b1: +0.126): a metastable
zero-field orientation is indistinguishable from a weak field response
to a threshold on P1.

## 6. Why the gate fails — the honest structure

1. **At the working field the response is genuinely small.** P1(0.01) =
   +0.036 ± 0.120 over three independent runs; stage A's 2-ns wf runs
   said +0.087 ± 0.09. Six nanoseconds per block does not resolve a
   ~ 0.05-0.1 response against run-to-run spreads of ~ 0.12.
2. **At 0.05 V/Å the response is consistent but the floor blocks the
   ordered test.** The on-side is tight (+0.142/+0.164/+0.213 per run,
   sem 0.021 — the field has overwritten the build offsets) while the
   zero-field side carries the metastability width (sem 0.118). The
   delta, 0.19, is 1.6 z but only 0.49 x the D15 floor.
3. **Secondary diagnostic, NOT the gate (disclosed, not loosened): the
   paired-by-build contrast.** Each sb run rK starts from zf build bK's
   final, so the pair shares build K's offset. Paired deltas — at
   0.01 V/Å: [+0.038, +0.055, +0.072], mean **+0.0550 ± 0.0096**; at
   0.05 V/Å: [+0.084, +0.413, +0.079], mean **+0.1918 ± 0.1108**. The
   weak field rides on top of the metastable orientation (a consistent
   +0.055 response per build); the 0.05 field erases the offsets, so
   the paired view there inherits the zf spread. This is why the
   unpaired ordered test loses the 0.01 response in the zf-side width
   and still fails at 0.05 against the offset-inflated floor. The
   pairing assumption (offset inheritance into the field-on run) is a
   model; the ordered gate stays the unpaired 3x-floor test.
4. **Extrapolation (why the e0 point matters).** With this floor,
   3 x floor = 1.18 on `dipole_p1` — stage A's saturating-field P1
   (~ 0.6-0.8 at 0.2 V/Å) would give a delta of ~ 0.65-0.82 and STILL
   fail the 3x test. As it stands, no field on this ladder can pass the
   3x-floor criterion against the 1-ns zf floor. The queued stage-B e0
   runs (6-ns zero-field blocks, same protocol) are the direct
   empirical test of whether a like-for-like floor shrinks the
   metastability contribution; and the classifier criterion may
   separate from e100 up while the 3x criterion does not. Which
   criterion carries "detect" in D32 — or whether the like-for-like
   e0-based floor replaces the 1-ns one — is the owner's call, and it
   decides the verdict. Both are recorded here, neither assumed.

## 7. The D32 statement-move list (step 7)

**Empty at both fields** — no statement other than `dipole_p1` moves
beyond 2 x its floor (nothing reaches even 1x). Constants by
construction (recorded, excluded): `density`, `linear_density`,
`ion_pairs.AGG`, chirality, census. Secondary z-view at 0.05 V/Å (not
the gate): `ion_pairs.CIP`/`.SSIP` z = 2.49, `li_shell.F6P` z = 1.98,
`nr.F6P.peak_r` z = 1.90, `nr.F6P.mean_r` z = 1.65 — a physically
coherent picture (fewer Li-PF6 contact pairs, PF6 pushed outward) that
the ordered thresholds do not confirm at this sampling.

## 8. Linear response (2 field points + the zero)

P1: −0.0185 ± 0.1175 (zf), +0.0364 ± 0.1197 (0.01), +0.1732 ± 0.0209
(0.05). Slopes: 5.5 ± 16.8 and 3.8 ± 2.4 per V/Å — consistent with each
other and with P1 proportional to E through the origin, but the
working-field slope is unconstrained (its 1-sigma band includes zero
and 4x the 0.05 slope). What CAN be concluded: no contradiction with
linear response. What CANNOT: any linearity verification at or below
the working field — that is the ladder's 0.005/0.02 points with their
own runs, not this tranche.

## 9. Block-protocol checks (all pass)

- **Block A vs B** (the plan's transient model): `dipole_p1` offsets
  0.03x (0.01) and 0.16x (0.05) of 2 x the run-level sem — block B is
  clean, block A adds nothing; only 3 minor n_r profile scalars exceed
  2 x run sem at 0.05 (max 2.02x).
- **Dense vs subsampled** (arm 2): block means agree to < 0.01; the
  subsampling costs nothing visible at these block sizes.
- **Provenance**: every frame of every job passed census + neutrality
  checks; every lifted program's non-moving statements (chirality
  (10,10), census, cell) are identical across all 1,809 lifts; zero
  failed static checks.

## 10. D32 status

- Arms: E_min(Chaord) = E_min(direct) — trivially the same field,
  since the blind statement IS the direct quantity to print precision.
- Detection at or below the working field: **no** (nothing detected at
  0.01 or 0.05 under the ordered 3x-floor rule; classifier 4/9 and 6/9).
- Move list: empty. Linear response: consistent, unconstrained at the
  working field.
- **No GO evidence from tranche 1, and no NO-GO either**: the verdict
  is recorded when the ladder completes (D32). The decisive items the
  rest of the ladder settles: the like-for-like e0 floor (6-ns
  zero-field blocks) against the 1-ns zf floor used here; the ladder
  points e5/e20 (is anything detectable near the working field?); the
  e100/e200 points, where the classifier criterion and the 3x-floor
  criterion are likely to DISAGREE (section 6.4) — that disagreement,
  not the lift, is the open question this tranche exposes.

## 11. Reproduce

    python research/cnt_electrolyte/analysis/y8_block_analysis.py

reads all 9 jobs (78,009 frames, two independent arms), lifts 1,809
frames blind, and writes
`research/cnt_electrolyte/analysis/tranche1_results.json` (all floors,
deltas, per-run and per-build values, classifier folds, linear-response
fit, block checks). Wall time ~ 7 min on the development box. `--smoke`
runs 3 frames per job as a pipeline test (never for the record).

Provenance chain per job: `stageB/manifests/<job>.json` (field, build,
sha256 of every input/output) -> `stageA/jobs_index.json` (zf build ->
`data/cnt10_bK.data` type map, D34) -> `stageB` inputs generated by
`stageB_inputs.py` from `params.py` physics. Analysis decisions: D31
(ensembles, zero-field floors, LORO), D15 (floor estimator), D32 (two
arms), D29 (blind dipole statement), stage_b_plan.md (block protocol,
block-B primary).
