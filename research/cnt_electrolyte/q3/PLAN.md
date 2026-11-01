# Q3 confirmatory set — PRE-REGISTRATION (Review 12, D49 trigger)

Written 2026-11-02 by stream S3, BEFORE any Q3 job was built, run or
submitted. The D39 analysis call below is fixed now; when the data exist,
the analysis runs exactly this call and reports the result either way,
adding no other test after seeing the data.

## Trigger (from Q2, reports/research/filling.md)

The Y6 reservoir filling differs from the pilot composition by more than
5% for all four species in (16,16): EC +68.9%, EMC +21.3%, Li+ +119.2%,
PF6- +74.4% (per-A, the fill tubes measured at 49.2 A = the pilot length,
Y6's "as the pilot, so the numbers compare"). D49 makes the filling-
sensitivity check binding: the Z4 verdict (0.01 V/A CONFIRMED at the
PILOT filling) must be re-tested at the RESERVOIR filling to show the
detection is not an artifact of the starting composition.

## Scope

(16,16) only; fields 0 and 0.01 V/A; 3 builds (Q3's wording), NEW seeds;
98.4 A tubes (the Z3 geometry: ncell 40, twice the pilot); **the
reservoir filling** — the molecule counts Q2 measured in the 49.2 A fill
bores, scaled to the 98.4 A tube (per-A from Q2 x 98.4, recorded in
section 3); ONE LAMMPS build (the cluster's module
lammps/3Mar20-icc.2022, 3 Mar 2020 — the same binary every Z3 job ran);
the Z3 protocol verbatim (1 ns in-job zero-field thermalization + 1 ns
settle + 2 x 6 ns production blocks; 12 ps thinned record).

## 1. Run list (6 jobs)

`q3_<fieldmv>_b<1..3>` — 2 fields x 3 builds:

| field (V/A) | jobs |
| --- | --- |
| 0 | q3_e0_b1 .. q3_e0_b3 |
| 0.01 | q3_e10_b1 .. q3_e10_b3 |

## 2. The D39 analysis call (verbatim)

Zero-field side: the 3 e0 run means (blocks A and B averaged). Field-on
side: each field's 3 block-B values. Welch's t, two-sided, alpha = 0.05,
n = 3 per group. The pre-registered question: does 0.01 V/A still show
p < 0.05 at the reservoir filling? Report the shift beside Z4's
(+0.107 at the pilot filling). No other test is added after seeing the
data.

## 3. Filling composition (from Q2, filling_results.json)

Reservoir counts in the 49.2 A fill bores (the wall criterion), scaled
to 98.4 A: EC 0.8984/A x 98.4 = 88.4 -> 88; EMC 0.5578/A x 98.4 = 54.9
-> 55; Li+ 0.1541/A x 98.4 = 15.2 -> 15; PF6- 0.1263/A x 98.4 = 12.4 ->
12. Charge neutrality: 15 Li+ vs 12 PF6- is NOT neutral — the Q2 bore
counts are instantaneous samples with fluctuation; neutralizing means
Li+ = PF6- = 13-15. DECIDED: 15 LiPF6 formula units (15 Li+ + 15 PF6-,
matching the Li+ measurement; the extra 3 anions' mass is 0.5% of the
cell, recorded). Solvent: EC 88, EMC 55.

## 4. Builds and seeds

3 new builds q3_b1..b3, seeds disjoint from all prior ranges (base
20281101+, step 100); the velocity/thermostat seeds per job derive from
the stageB_inputs.seeds() pattern with the Q3 base.

## 5. Manifests and 只算不存

Per-job manifests (sha256 of in.lammps + system.data); after completion
the 12-ps thinned series come back, raw dumps die on the cluster.
