# Pending rulings — Chaord (for the external reviewer)

Collected 2026-10-30. **ALL FOUR RULED 2026-10-30 by external Review 11 —
RULED: see open_items_v5.md (D39–D44).** The record below is preserved as
the ruling's input; the rulings' operational text lives in open_items_v5.md
section 1.

## R1 · The detection criterion for the go/no-go (D32's operational core)

RULED: see open_items_v5.md -- D39 (Welch's t on run means, alpha 0.05; the
D15 3x-floor rule stays only for per-block classification claims).

**The question.** "Chaord detects the field" — measured how? Two candidate
criteria give DIFFERENT answers at 0.05 V/Å:

- **(A) Cross-run floors, as D31 is written.** Floor = D15 estimator
  (max(mean, P90)) over |Δ| between blocks from different INDEPENDENT runs.
  The dipole_p1 floor at (10,10) came out 0.394 because it is dominated by
  long-lived per-build zero-field orientation offsets (within-run |Δ| mean
  0.075 vs cross-run 0.259): under (A), even the SATURATING field (~0.7)
  would fail 3 × 0.394. E_min > 0.05 V/Å under (A) in the tranche-1 data.
- **(B) Build-paired deltas.** Each field run starts from the equilibrated
  zf final of its OWN build; comparing each run to its own build's zero-field
  baseline cancels the per-build offset. Under (B): +0.0550 ± 0.0096 at
  0.01 V/Å and +0.1918 ± 0.1108 at 0.05 (paired z 2.49 on ion_pairs CIP/SSIP
  at 0.05 as corroborating physics). 0.01 V/Å already significant at (B).

**The arguments.**
- For (A): it is the criterion as ruled (D31, "floors from zero-field runs
  only"); it cannot be tuned after seeing the data; it is the honest
  blind-test standard.
- For (B): the zero-field offset is a measured initial-condition remnant,
  not equilibrium order (Stage A's mirrored-start test: the cnt16 offset
  FLIPS sign under z → −z; cnt10's does not but sits within ~2 sem of
  zero) — a symmetric system's equilibrium zero-field P1 is 0, so the
  cross-run scatter is noise whose magnitude depends on how long the runs
  are, not on physics; D31's intent ("a program is a macrostate") is
  arguably satisfied by build-paired blocks from independent runs.
- Middle options: (C) rule (A) but extend the zero-field runs until the
  cross-run floor shrinks (6 ns sb e0 blocks exist; the τ_M = 2.7-3.0 ns
  measured means even 6 ns blocks may not decorrelate fully — cost scales);
  (D) rule that BOTH must agree for a "detection" claim (strictest).

**What it changes.** E_min, the go/no-go verdict (D32), and how the result
would be stated in the paper. Evidence and both result sets:
`reports/research/go_no_go.md` (tranche-1 PRELIMINARY; the full-ladder
update lands with the cluster batch).

**Status: OPEN — needed before the go/no-go verdict is written.**

## R2 · Stage B run length vs the measured τ_M (D30/Y7 gate, already triggered)

RULED: see open_items_v5.md -- D41 (the 3-run 2x6 ns ladder ratified;
one confirmatory set Z3 on (16,16) at 98.4 A with 6 new builds per field).

Measured τ_M at the working field: 2.7 ± 1.1 ns (cnt10), 3.0 ± 0.9 ns
(cnt16) — both > 1 ns, which per the brief pauses Stage B for the owner's
run-length decision. The runs that exist (Stage B cluster batch) use 6 ns
blocks = 2 τ_M with the statistical consequences stated in
`reports/research/stage_b_plan.md` (block A carries 27-31% of the switch-on
gap as transient bias → block-B-only primary statistics; "more runs, not
longer runs" as the upgrade path). The 20 τ_M full protocol would cost
~55-60 ns/run × ~50 runs (~1.1 µs aggregate) — cluster-scale.

**The question.** Ratify Option (b) (2 τ_M blocks, 3 runs per field, already
run) as the go/no-go evidence base, or require the 20 τ_M protocol before
the verdict counts?

**Status: OPEN, but superseded in practice by R1** — under (A) no field is
detected at any block length available; under (B) the tranche data already
carry the claim and the full ladder refines it.

## R3 · The go/no-go date (2026-10-29, D32)

RULED: see open_items_v5.md -- D43 (the verdict is recorded 2026-10-30;
the reservoir fills are a separate Y6 deliverable).

The date has passed with the verdict pending on R1/R2 and the fills still
running on the cluster. Propose: the go/no-go is recorded when (i) the fills
finish (Y6's reservoir filling), (ii) the full-ladder Y8 analysis is
committed, and (iii) R1 is ruled — currently projected ~2 days after the
fills complete (~2026-11-01), or immediately on an R1 ruling if the owner
accepts the fills as a separate Y6 deliverable not blocking the field-
detection claim.

**Status: OPEN — pick one of {immediate-on-R1, ~11-01, a stated date}.**

## R4 · Minor rulings carried from the review trail (all documented, none blocking)

RULED: see open_items_v5.md -- D44 (A14 closed at 41/41; dipole_axis_default
z ratified; the 5 pinned cases to regenerate physically, P2; frame-charge
convention stands; cluster record accepted).

1. **A14's 9 example-less keys** (atom, chirality, dialects, lift_version,
   nanotube, note, reconstruction, source, stacking): chirality/nanotube
   now have examples (11/15_nanotube_chirality.chaord); the remaining
   examples are X9's work — confirm the spec-example bar (one per key) or
   accept documented groups.
2. **`dipole_axis_default: z`** (research dialect, Y2): the blind lift's
   axis fallback order is field axis → nanotube axis → dialect default.
   Ratify the default or name another.
3. **The W8 contact-floor tripwire pins** (5 molecular cases flagged beyond
   the review's prediction: ar_gas_box25, n2_box22, nacl_aq, lipf6_ec,
   cu_water): regenerate them physically (each its own commit) or leave
   pinned as recorded failures.
4. **PC/EMC/EC site charges** for the field observables: currently only H2O
   carries a cited charge model (SPC/E); the carbonates' P1 statements use
   frame charges (the LAMMPS dumps' own q column). If the owner's Moltemplate
   model fixes their site charges, hand them over; otherwise the frame-charge
   convention stands (D36-consistent).
5. **`docs/reviews/owner_inputs.md` cluster section**: now records the real
   cluster (slurm, 9 × 112-core, module lammps/3Mar20-icc.2022 — the default
   lmp_mpi lacks RIGID/KSPACE/MOLECULE), the 只算不存 policy as implemented
   (/dev/shm is LOGIN-NODE-ONLY — compute nodes cannot see it; jobs run on
   /vol02 shared storage with post-job extraction and raw-dump deletion), and
   the 28-rank-per-job ceiling the owner set.

## Settled this round (for the record, no ruling needed)

X1 fake-adatom fix; X2 lift-side snap (94/94, FOILED emptied); X5 history
semantics (rates per time unit, thermostat follows the ramp, fail-closed
cap, one protocol runner); X6 per-pair glass statistics (first-minimum
cutoffs, network-glass gate); X3 records (gate report rewritten, table
generated from the artifact); X4 acceptance draws (20/20×20 + per-arm A11
table); X9 leftovers (slab warnings as branches, A14 at 41/41);
Y1 pilot downgrade; Y2 blind lift (dipole_p1 measured unconditionally);
Y3 real-frame tube inference (20/20 on every D33 case; tube+liquid via the
public API); Y4 reader (real dumps, ambiguous-mass raise); Y5 electrolyte
statements (Li shells, ion pairs, accessible-volume density);
Y9 EC template (Nutakki geometry, bench-v1 preserved);
X7 item 1 (KA rates: IS energy resolves adjacent decades decisively,
structure does not; the bench reference's thermostat lag independently
confirmed); Stage A (32 tube jobs + τ_M/τ_rot/mirrored-start report);
Stage B (the full 7-field ladder, cluster).
