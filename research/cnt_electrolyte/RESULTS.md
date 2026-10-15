# Pilot at a saturating field (0.2 V/Å, about 10 kT per EC dipole)

**Verdict: pilot at a saturating field (0.2 V/Å, about 10 kT per EC
dipole).** What this run shows: the pipeline works end to end. Four
systems were built, run and lifted; all 84 production frames (2
diameters x field on/off x 21 frames, 0-100 ps every 5 ps) passed every
static check; within each diameter the field1-vs-field0
statement-observable distance exceeds the per-system noise floor by
x3.3-x4.8 (mean) and x2.1-x3.3 (worst pair); the field-on and field-off
P1 ranges never overlap (cnt10: [-0.245, +0.005] vs [+0.585, +0.838];
cnt16: [+0.092, +0.191] vs [+0.460, +0.708]); and a classifier that
reads only the statement texts classifies all 84 frames correctly
(100%), leave-one-out and cross-diameter both ways. The P1 physics is
the only statement that separates: a control classifier denied every
field-related line falls to 64% / chance — the honest negative that
shows the separation is the field's, not a side effect of composition
asserts.

What this run does NOT show: weak-field sensitivity (only 0 and 0.2 V/A
were run, and 0.2 V/A saturates — the title's about-10-kT-per-EC-dipole
field; nothing here probes the working field or the linear-response
regime); an equilibrated zero-field orientation (section 3: the
zero-field runs had 500 ps after the ramp, 7 of 13 and 38 of 61
molecules never reverse their axial dipole, and each molecule's 500 ps
mean of cos theta spreads by 0.57-0.61); independent runs (one
trajectory per system — the section 4 floors are frame pairs within
single trajectories, not ensembles; D31 judges ensembles, not frames,
and the field ladder of D30 is what would answer the missing
questions).

The bar this pilot answered (docs/reviews/open_items_v2.md, section 5,
item 3): "Lift 20 frames with the field off and 20 with it on, for two
tube diameters. Done when the programs' statements, not the
coordinates, separate the states beyond their noise floors, and a
classifier that reads only the statements tells field on from off with
at least 95% accuracy." That bar is met at this field, with the numbers
below — and only at this field.

Every number in this file is produced by
`research/cnt_electrolyte/analyze_first_result.py` (the lift, the
statement inventory, the P1 table and the classifier) or by
`research/cnt_electrolyte/orientation_check.py` (the orientation
relaxation, the P1-line census and the measured-P1 threshold; numpy and
stdlib only, no chaord) from the committed run data; the classifier
numbers are pinned by `tests/test_field_classifier.py` (slow). The full
record, including every program text, is
`research/cnt_electrolyte/runs/first_result_summary.json`.

Reproduce (from the repository root):

    python research/cnt_electrolyte/analyze_first_result.py
    python research/cnt_electrolyte/orientation_check.py

## 1. Inputs and verification (fail-closed, before any lifting)

Read with the repo LAMMPS reader (`chaord.io.lammps`): `prod.dump` frames
(columns `id mol type q x y z`; charges land in `frame.info["charges"]`),
type map from the BUILD data file each system's provenance names
(`outputs.data_file`) -- the provenance-recorded input; the pipeline never
reads `final.data`.

| system | frames | atoms | census (every frame) | cell charge | chirality inferred | matches provenance |
|---|---|---|---|---|---|---|
| cnt10_field0 | 21 | 971 | EC 8, EMC 5, Li 2, PF6 2 | 0 (|q| < 1e-6 e) | (10,10), 49.200 A, R 6.781 A | yes |
| cnt10_field1 | 21 | 971 | EC 8, EMC 5, Li 2, PF6 2 | 0 | (10,10), 49.200 A, R 6.781 A | yes |
| cnt16_field0 | 21 | 2058 | EC 37, EMC 24, Li 6, PF6 6 | 0 | (16,16), 49.200 A, R 10.850 A | yes |
| cnt16_field1 | 21 | 2058 | EC 37, EMC 24, Li 6, PF6 6 | 0 | (16,16), 49.200 A, R 10.850 A | yes |

The tube's charge is 0 in every frame. Every frame is verified against
`provenance_*.json` before any lifting: atom count, electrolyte census,
total charge, tube charge, and the inferred chirality must not move
between the 21 frames of a system. Two input facts were handled and
recorded, not hidden:

- **Dump coordinates sit outside the box.** Rigid-body dumps carry atoms
  up to 8.2e-3 box lengths (0.28 A) beyond a boundary between
  re-neighbouring rewraps; the reader itself wraps them (a
  periodic-image relabelling, no physical change) and records the
  largest excursion per frame (max 8.20e-3, cnt10_field0).
- **The chirality is inferred by the lift, on the dump.** The composed
  lift (one `lift_frame(mode="auto")` call per frame, D35/Y3) finds the
  all-carbon three-coordinated component, infers (n, m) and the radius
  on the frame's own dump-precision carbons -- the D33 inference
  (bond-graph unwrap with minimum-image bond vectors, Kasa circle fit,
  gates calibrated on noisy and thermal frames in `carbon.yaml` 0.2.0)
  -- and states the tube region together with the confined fluid. No
  `final.data` read and no tripled perpendicular cell: the first pilot
  pipeline's workaround for dump precision and cnt16's tube-wider-than-
  half-the-cell anchoring, retired with the numbers unchanged (every
  inferred value in the table above is identical through the public
  path).

## 2. What the programs say

One composed program per frame (RD4, docs/reviews/research_decisions.md):
a `crystal tube` region carrying RD2's chirality inside RD1's cylinder,
and a `liquid fluid` region for the electrolyte (composition, region-
referred density, cn and g(r) first-peak asserts). Field-on programs carry
RD3's `state field` line plus RD4's measured consequence. Field-off
programs carry no field-related line at all. Example (cnt10_field1,
frame 0):

```
system {
  cell 33.563 33.563 49.200
  pbc xyz
  state T 300.00
  state field E 0.2 V/A along z
  assert field_dipole_p1 0.763 +- 0.277 along z
  conserve atoms C 844 F 12 H 72 Li 2 O 39 P 2
}
...
crystal tube : cylinder axis z center 16.781 16.781 24.600 radius 6.781 A length 49.200 A {
  nanotube (10,10) length 49.200 A
}
liquid fluid : cylinder ... {
  molecules C3H4O3 8 / C4H8O3 5 / F6P 2 / Li+ 2
  state density 0.357 g/cm3
  assert cn 0.0 +- 0.0 cutoff 3.10 A
  assert gr_peak 3.35 A height 10.46
}
```

The static checks (`chaord.check.statics.run_checks`: overlap,
conservation, charge, residual) pass after every one of the 84 composed
lifts.

Statement inventory across the 84 texts: cell/pbc/T/conserve (identical
within a system), chirality+cylinder (identical within a system), the
fluid composition (identical within a system), region density (0.357
g/cm3 cnt10, 0.609 g/cm3 cnt16 — diameter, not field), cn (0.0-0.1,
noise), gr_peak position 3.15-5.45 A and height 2.2-10.5 (diameter-
dependent, state-overlapping), and — only in field-on programs — the
`field_dipole_p1` assert. Nothing else in the statement set separates the
states (see the control classifier in section 5).

## 3. The P1 distributions (the physics)

P1 = <cos theta> of every solvent molecule's dipole against z (the field
axis), from the frames' own charges; species view [C3H4O3, C4H8O3] (all
solvent molecules: the molecular dialect's ion-solvation rule keeps Li+
out of the bond graph, so n = 13 and 61, not just the "free" subset an
earlier probe without the molecular stack measured).

| system | P1 mean +- std (21 frames) | range | EC mean | EMC mean | n molecules |
|---|---|---|---|---|---|
| cnt10_field0 | -0.121 +- 0.076 | [-0.245, +0.005] | +0.014 | -0.336 | 13 |
| cnt10_field1 | **+0.725 +- 0.078** | [+0.585, +0.838] | +0.939 | +0.382 | 13 |
| cnt16_field0 | +0.152 +- 0.030 | [+0.092, +0.191] | +0.246 | +0.007 | 61 |
| cnt16_field1 | **+0.606 +- 0.053** | [+0.460, +0.708] | +0.743 | +0.395 | 61 |

Reading. By symmetry, the equilibrium zero-field expectation is P1 = 0:
an achiral armchair tube is invariant under z -> -z, so no confined
equilibrium state can carry a nonzero mean axial orientation. The
measured field-off means (-0.121, +0.152) are therefore not a
confinement effect to read physics from — they are the unrelaxed
remnant of the packed starting orientations, and
`orientation_check.py` (chaord-free, per molecule over equil + prod
dumps, frames 50-550 ps) shows the orientation had not equilibrated in
the 500 ps available:

    cnt10_field0: P1 range -0.24..+0.25; 7 of 13 molecules never reverse; per-molecule 500 ps mean cos, sd 0.61
    cnt16_field0: P1 range +0.03..+0.20; 38 of 61 molecules never reverse; per-molecule 500 ps mean cos, sd 0.57
    cnt10_field1: P1 range +0.55..+0.84; 10 of 13 molecules never reverse; per-molecule 500 ps mean cos, sd 0.38
    cnt16_field1: P1 range +0.46..+0.71; 27 of 61 molecules never reverse; per-molecule 500 ps mean cos, sd 0.30

With the field on, the solvent is saturated: the cnt10 zero-field P1
swings from -0.24 to +0.25 across the same 500 ps window, while the
field-on per-frame P1 stays within +0.55..+0.84 / +0.46..+0.71, and in
the production window the two states' RANGES do not overlap in either
diameter (cnt10: +0.005 vs +0.585; cnt16: +0.191 vs +0.460). Even
saturated, the per-molecule 500 ps mean cos keeps an sd of 0.38/0.30:
the field fixes the mean, not every molecule.

## 4. Noise floors and separation

Floor = max(mean, P90) of |dP1| over all within-system frame pairs
(210 pairs per system; decision D15's standing estimator). Separation =
|dP1| over all field1 x field0 cross pairs (441 per diameter).

| diameter | floor (field0) | floor (field1) | cross mean | cross worst pair | multiple (mean) | multiple (worst pair) |
|---|---|---|---|---|---|---|
| cnt10 | 0.176 (mean 0.088, P90 0.176) | 0.173 (mean 0.091, P90 0.173) | 0.845 | 0.580 | **x4.8** | **x3.3** |
| cnt16 | 0.071 (mean 0.034, P90 0.071) | 0.129 (mean 0.058, P90 0.129) | 0.454 | 0.269 | **x3.5** | **x2.1** |

The statements separate the states beyond their noise floors in both
diameters: even the closest cross pair is more than double the (state-
maximal) floor. The radial observable is NOT part of this table — see
"Honest negatives".

## 5. The classifier (reads only the statement texts)

Features: the numeric statements of each program text — the P1 assert
value when present, cn, cn tolerance, gr_peak position and height, the
region density — parsed by `statement_features` (regexes that touch
statement lines only). **Excluded by construction: the `state field`
statement itself and every provenance note** (using the stated field
would be trivial and circular — the note regexes never see them). Model:
a single-feature threshold on P1 (midpoint of class means), the simple
honest model for a mean of cosines; a field-off program carries no P1
assert and is encoded as P1 = 0 — the a-priori value of an unaligned
liquid (the measured field-off distributions above sit at -0.12 / +0.15,
and no field-off frame comes near the field-on range).

| evaluation | accuracy |
|---|---|
| leave-one-out, all 84 frames | **1.000** (0 errors) |
| cnt10 only (42 frames, threshold 0.362) | 1.000 |
| cnt16 only (42 frames, threshold 0.303) | 1.000 |
| train cnt10 -> test cnt16 | **1.000** |
| train cnt16 -> test cnt10 | **1.000** |
| control: nearest-class-mean over [cn, cn tol, gr_peak r, gr_peak h, density], every field-related line excluded | 0.643 (LOO); 0.52 / 0.57 cross-diameter |

The control is the honest negative: the statements the two states share
do NOT separate them (chance-level across diameters). The separation is
the field's own statement.

**Caveat stated plainly:** RD4 ties the P1 assert to a supplied field, so
the assert's PRESENCE co-occurs with the `state field` line by
construction; a classifier could read presence instead of value. The
value-only content is what the numbers above carry, and the underlying
physics is section 3-4: the measured P1 of every field-on frame exceeds
every field-off frame's, in both diameters, beyond the floors.

The caveat, quantified (`orientation_check.py`): the P1 line's presence
co-occurs with the field by construction —

    field off: 0 of 42 programs carry the P1 line
    field on:  42 of 42 programs carry the P1 line

— but the value alone carries the separation: a threshold on each
frame's MEASURED P1 (`records[].p1_measured`, recorded for all 84
frames whether or not the program asserts it) also classifies 100% at
this field — leave-one-out 84/84, cnt10 -> cnt16 42/42, cnt16 -> cnt10
42/42, over measured ranges field0 -0.245..+0.191 vs field1
+0.460..+0.838.

## 6. Honest negatives, limitations and open items

1. **The radial observable is unusable on these cells as defined.**
   `inner_outer_ratio` splits the cell-centred cylindrical profile at half
   the shorter transverse box edge; the outer shell of a tube-in-vacuum
   cell is empty, so `cv/field.py:280` divides by a zero outer mean and
   raises (ZeroDivisionError, not the docstring's promised inf). The
   floors and separations rest on the P1 component of the field distance
   only. Fixing cv/field.py (or an r_max caller override) is next round's
   item — the file is frozen for this work package.
2. **The shared statements do not separate the states** (control model at
   chance, above). Composition/density/gr asserts are diameter, not
   field, information.
3. **The cn assert is degenerate** (0.0 +- 0.0 in 55/84 frames): the
   molecule-centre coordination at the dialect's 3.10 A cutoff counts no
   neighbours. Honest as measured; a confinement-aware cutoff is a
   dialect question, not tuned here.
4. **RD2's inference limitations are retired** (section 1): the first
   pilot pipeline worked around the exact single-radius gate (dump text
   precision) and the one-anchor minimum-image break (cnt16's tube
   wider than half the perpendicular cell edge) by reading `final.data`
   with the perpendicular cell lengths tripled. The D33/Y3 rewrite made
   both moot -- the bond-graph unwrap has no one-anchor minimum image,
   and the gates in `carbon.yaml` (0.2.0) are calibrated on noisy and
   thermal frames -- and the workaround was dropped; every pilot number
   reproduced exactly through the public `lift_frame` path (4da59c5).
5. **The `field_dipole_p1` assert fires only when a field is supplied**
   (RD4's design; the task's binding). A field-off program therefore
   states no P1 — the classifier encodes the absence as the a-priori 0,
   and the caveat in section 5 applies.
6. **Routing views are pipeline configuration, not dialect changes.**
   The composition lifts the electrolyte under
   `fluid_max_bonded_component = 15` (EMC, the registry's largest
   molecule of this solvent) and measures P1 under the species view
   [C3H4O3, C4H8O3]. Both are recorded in RD4; neither changes any
   numeric threshold in a dialect file, and `bonded_single_phase` still
   performs its real check. `mode="auto"` is the one call the pipeline
   makes per frame (D35/Y3: it finds the tube and composes the tube and
   confined-fluid regions; the routing views ride the same stack).

## 7. Provenance of every number

- Pipeline: `research/cnt_electrolyte/analyze_first_result.py` (this
  directory), deterministic, no randomness anywhere.
- Orientation, reversal counts, P1-line census and the measured-P1
  threshold: `research/cnt_electrolyte/orientation_check.py` (this
  directory; numpy + stdlib only, no chaord; run from the repository
  root).
- Raw record with every program text and per-frame P1:
  `research/cnt_electrolyte/runs/first_result_summary.json` (written by
  the pipeline run of 2026-10-14/15 on the committed `runs/` data;
  `prod.dump` sha256 in each `provenance_*.json`).
- Pinned by: `tests/test_field_classifier.py` (@pytest.mark.slow, runs
  the same pipeline end-to-end on the committed data).
- P1 definition and charges: `src/chaord/cv/field.py::dipole_orientation`
  with the frames' own per-atom charges; site-charge table never consulted
  (frames carry charges).
- Floor rule: open_items_v2.md decision D15 (max(mean, P90)).
- Tolerance printed in the P1 assert: 1/sqrt(n) (Popoviciu, Mathematica
  (Cluj) 9, 129 (1935) variance bound on [-1,1]) — an a-priori bound,
  not a floor; the floors in section 4 come from frame pairs instead.
