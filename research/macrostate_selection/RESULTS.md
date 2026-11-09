# Macrostate selection: RESULTS

Run 2026-11-09 on one gerui compute node (24 processes, 1,900 s wall) by
`matrix.py`, from an export of commit 8deeff2, which registers `PLAN.md`. Seeds
from 1001 and 2001, as registered. Every number is in `results.json`; the screen
log is `results_run.out`. `PLAN.md` is not edited.

## Registered verdict: NOT SUPPORTED

| Condition | Result |
|---|---|
| `no_refusals` | true |
| `references_valid` | true |
| `diagonal` | **false** |
| `off_diagonal` | true |
| `control_liquid` | true |
| `control_not_source` | true |

Five of the six conditions hold. `diagonal` fails because the slab's full
programs do not match the slab reference (2.5 times the floor; the gate is 1.5).

## The full table

Largest of the three distance ratios (g(r), coordination, common-neighbour
classes) against the reference's floor; * marks a match (all three at most 1.5).

Full programs (rows source, columns reference):

| | liquid | fcc | hcp | coex |
|---|---|---|---|---|
| liquid | 0.4* | 17.5 | 21.9 | 7.4 |
| fcc | 1240.6 | 0.7* | 14.9 | 12.4 |
| hcp | 1161.5 | 14.1 | 1.4* | 12.7 |
| coex | 362.0 | 13.4 | 19.2 | 2.5 |

Control programs (cell, T, atom count, density, backend only):

| | liquid | fcc | hcp | coex |
|---|---|---|---|---|
| liquid | 0.9* | 17.5 | 21.7 | 7.5 |
| fcc | 0.5* | 17.5 | 21.3 | 7.4 |
| hcp | 0.5* | 17.4 | 22.2 | 7.4 |
| coex | 0.4* | 17.5 | 21.6 | 7.4 |

Program lengths in characters: liquid 375, fcc 363, hcp 346, slab 689, 726 and
373; control 203 to 205.

## Why the slab missed

This section explains the failed condition. It does not change it.

Of the three slab programs, two were lifted as a crystal slab plus a liquid and
one as a homogeneous liquid. The third source frame had the smallest crystal of
those lifted (final-frame fcc fraction 0.047; references 0 and 1 had 0.099 and
0.076), small enough to pass `is_single_phase`. Final-frame fcc fractions of the
three rebuilds were 0.089, 0.065 and 0.000, so the two slab programs rebuilt a
slab and the liquid program did not. Per-rebuild observables were not stored, so
the two slab programs cannot be scored separately from this run, and no such
score is claimed. This is the limit the plan's pilot record names: a crystal that
is a small fraction of the box is not detected by the lift.

## What the data say

1. **Without structural text, physics gives one answer.** The control matched
   the liquid reference from all four sources and matched no other reference
   (7.4 to 22.2 times the floor).
2. **With the text, three of four macrostates were recovered and all twelve
   wrong identifications were rejected.** Liquid, fcc and hcp programs matched
   their own reference (0.4, 0.7 and 1.4 times the floor) under identical MD.
   The smallest off-diagonal entry is 7.4.
3. **The slab is a partial result.** At 2.5 times the floor it is far closer to
   its reference than the control is (7.4), but it fails the registered gate.
4. **hcp passed narrowly** (coordination 1.45 times the floor).

## What this does and does not support

It supports: for states that physics alone does not select within the run
length, a lifted program can carry the choice of macrostate, for homogeneous
crystals and liquids at this state point. It does not support the registered
four-state claim. It covers one component, one potential, and one state point
chosen in pilots, and the test does not address molecular systems.

Pilots found two compiler defects, both fixed before registration
(`docs/records/corrections.md`). A follow-up on slab detection would be new work
with its own plan; this run is not re-analysed.
