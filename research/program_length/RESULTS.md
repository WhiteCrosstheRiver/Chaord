# Program length against physics fidelity: RESULTS

Run 2026-11-07 on one gerui compute node (27 processes, 156 s wall) by
`curve.py` at commit ea65781, exactly as `PLAN.md` (registered in 5b434b8)
fixes the grid, the ladder and the A5 gate. Every number below is in
`results.json`; the screen log is `results_run.out`. PLAN.md is not edited;
everything in this file is reporting, and the interpretation section is
explicitly flagged.

## The full table (PLAN: reported either way)

Ratio = A5 distance / effective floor (gate: both <= 1.5). angle_tv is the
held-out bond-angle check, shown relative to the P2/S0 cell of the same case.
Lengths are characters of the canonical (`chaord fmt`) text.

| case | physics | program | length | gr_rms ratio | cn_tv ratio | pass | angle_tv / P2-S0 |
|---|---|---|---|---|---|---|---|
| lj_liquid | P0 | S0 | 201 | 13.38 | 5.47 | no | 17.6 |
| lj_liquid | P0 | S1 | 240 | 13.38 | 5.47 | no | 17.6 |
| lj_liquid | P0 | S2 | 277 | 13.42 | 6.49 | no | 19.2 |
| lj_liquid | P1 | S0 | 201 | 1.21 | 1.17 | **yes** | 1.65 |
| lj_liquid | P1 | S1 | 240 | 1.21 | 1.17 | yes | 1.65 |
| lj_liquid | P1 | S2 | 277 | 4.72 | 2.76 | no | 7.5 |
| lj_liquid | P2 | S0 | 201 | 0.94 | 0.46 | **yes** | 1.00 |
| lj_liquid | P2 | S1 | 240 | 0.94 | 0.46 | yes | 1.00 |
| lj_liquid | P2 | S2 | 277 | 9.15 | 4.51 | no | 13.2 |
| lj_liquid_large | P0 | S0 | 205 | 18.72 | 13.64 | no | 27.5 |
| lj_liquid_large | P0 | S1 | 244 | 18.72 | 13.64 | no | 27.5 |
| lj_liquid_large | P0 | S2 | 281 | 19.50 | 17.45 | no | 32.0 |
| lj_liquid_large | P1 | S0 | 205 | 1.27 | 1.17 | **yes** | 2.45 |
| lj_liquid_large | P1 | S1 | 244 | 1.27 | 1.17 | yes | 2.45 |
| lj_liquid_large | P1 | S2 | 281 | 6.98 | 6.13 | no | 11.8 |
| lj_liquid_large | P2 | S0 | 205 | 0.59 | 0.65 | **yes** | 1.00 |
| lj_liquid_large | P2 | S1 | 244 | 0.59 | 0.65 | yes | 1.00 |
| lj_liquid_large | P2 | S2 | 281 | 19.25 | 17.45 | no | 31.6 |

No restraint refused in any cell. Shortest passing program: P0 none,
P1 S0, P2 S0, for both cases.

## The decision rule, and an ambiguity it left open

PLAN: SUPPORTED iff, for both cases, the shortest passing length is
non-increasing from P0 to P1 to P2, **and at least one physics level needs a
strictly longer program than P2**. PLAN defines the shortest passing program
as "none" when S2 fails, but does not say how "none" compares with a length.

- **Reading A (the one `curve.py` implements): "none" is longer than any
  program.** P0 then "needs" a longer program than P2, and the rule returns
  **SUPPORTED** for both cases. This ordering was written into the runner
  after the PLAN was registered (though before any grid data existed), so it
  is an interpretation, not part of the registration.
- **Reading B: "none" is not a program length.** No physics level that has a
  passing program needs a longer one than P2 (every pass is at S0), so the
  second clause fails and the rule returns **NOT SUPPORTED**.

Because the registration does not settle this, neither reading is reported
as the registered verdict. **The registered verdict is undetermined between
SUPPORTED (A) and NOT SUPPORTED (B), and it is the substance below, not the
label, that we report.**

## What the data say

1. **Physics alone accounts for every pass.** At P1 and P2 the minimal
   program S0 passes on both liquids; at P0 nothing in the ladder passes.
   The original design claim, that a richer program compensates for weaker
   physics, is not demonstrated: no restraint level ever turned a failing
   cell into a passing one.
2. **S1 is inert by construction.** The ladder copies the lifted
   `assert cn 12.0 +- 1.1` verbatim, and the printed ±1.1 is the per-atom
   spread of coordination, not a tolerance on its mean. A random packing
   already satisfies it, so the restraint stops before its first move and
   S1 reproduces S0 exactly in all six physics cells.
3. **S2 is harmful, and most harmful after full MD.** Restraining the g(r)
   peak height to the lifted value within the dialect's ±0.15, applied after
   the physics stage, drives an already-equilibrated liquid away from the
   reference: gr_rms rises from ×0.6–0.9 to ×9–19 of the floor at P2, and the
   held-out bond-angle distance rises 13–32×. The restraint hits its target
   number while distorting the rest of the structure; the held-out check
   catches exactly that.
4. **A by-product: the A5 MD prior is generous.** P1 (700 steps, 10% of the
   dialect's 7000) already passes both liquids, at ×1.17–1.27, inside the
   1.5× gate; P2 passes with more margin. Not a claim the PLAN set out to
   test; recorded as observed.

## What this does and does not support

It supports: with physics on, Chaord's minimal lifted liquid program
reproduces the reference structure within the A5 gate, even with 10% of the
default MD. It does not support: that programs can trade length for physics
fidelity, because the two restraints the compiler has either do nothing (S1,
as the ladder states it) or damage the structure (S2). A test of the trade-off
would need restraints that act on the distribution rather than one number,
or a tolerance on the mean taken from the measured floor, and it would have
to be pre-registered as new work. This run is not re-analysed under any such
change.
