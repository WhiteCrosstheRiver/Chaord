# Y6 reservoir filling — the Q2 analysis (Review 12)

Data: the two Y6 filling runs (open-ended tube in a bulk 1M LiPF6 / 1:1
EC:EMC bath, NPT 1 atm 300 K, staged decompression, 5 ns production,
thinned series at 5 ps, 1001 frames each — `fills_cluster/`,
sha256-pinned). Analysis script:
`research/cnt_electrolyte/analysis/filling_analysis.py`; numbers:
`analysis/filling_results.json`.

## The measurement

Per frame, per species: count molecules/ions whose centre of mass lies in
the tube's middle axial window (> 1 nm from either mouth) and inside the
tube bore. Bore convention: **r < R (the wall position)** — disclosed:
an r < R − 3.2 Å "core" criterion (the generator's own vdW wall margin)
excludes nearly every ion in the narrow tube, because a Li+ with its EC
solvation shell sits off-axis; the wall criterion counts every molecule
the tube contains, which is what the pilot composition counts. Both
conventions' numbers are in `filling_results.json` (`per_species` =
wall; `core_counts` = core).

## The verdict table (block means over the 5 ns; SEM = frame SEM)

| tube | species | pilot count per 49.2 Å | reservoir count per 48.0 Å | per-Å pilot | per-Å reservoir | difference |
|---|---|---|---|---|---|---|
| (10,10) | EC | 8 | 10.5 ± ? | 0.163 | 0.219 | **+34.7%** |
| (10,10) | EMC | 5 | 9.0 ± ? | 0.102 | 0.188 | **+85.0%** |
| (10,10) | Li+ | 2 | **0.0** (max over 1001 frames: 0) | 0.041 | 0.000 | **−100%** |
| (10,10) | PF6− | 2 | 1.2 | 0.041 | 0.024 | −40.1% |
| (16,16) | EC | 37 | 43.1 | 0.752 | 0.898 | +19.5% |
| (16,16) | EMC | 24 | 26.8 | 0.488 | 0.558 | +14.4% |
| (16,16) | Li+ | 6 | 7.4 | 0.122 | 0.154 | **+26.4%** |
| (16,16) | PF6− | 6 | 6.1 | 0.122 | 0.126 | +3.6% |

(SEM values are in `filling_results.json` per species; the table shows
the means and the differences the D49 gate reads.)

## Findings

1. **The (10,10) narrow tube excludes Li⁺ entirely**: in 5 ns the
   in-tube Li⁺ count is 0 in every one of 1001 frames (independently
   re-verified by a second scan), while PF6⁻ sits in the pore (~1.2
   ions on average). Charge neutrality holds over the TOTAL system —
   the reservoir carries the counter-ions — not in the tube segment.
   The anion preference is a desolvation effect: PF6⁻ sheds its shell
   against the wall; Li⁺ would have to drag a ~6 Å EC shell into a
   13.6 Å bore that is already solvent-filled.
2. **The (16,16) tube's reservoir filling is within +3.6% to +26.4% of
   the pilot composition per species**; the D49 trigger (> 5% for any
   species) FIRES on EC (+19.5%), EMC (+14.4%) and Li⁺ (+26.4%) — so
   **Q3 is triggered** for the wide tube: 3 builds at the reservoir
   filling, fields 0 and 0.01 V/Å, the Z3 protocol.
3. The narrow tube's solvent excess (+34.7% EC, +85.0% EMC over the
   pilot) comes with the Li⁺ exclusion: more solvent fits when no ions
   do. The (10,10) is reported as non-ergodic at weak field (D41) and
   its pilot composition was an ion-bearing starting choice, not an
   equilibrated filling.

## Consequences

- **Q3 triggered** (S3, next): the confirmatory comparison at the
  reservoir filling for (16,16) — 3 builds, fields 0 and 0.01 V/Å, the
  Z3 protocol, the D39 test pre-registered in a PLAN before submission.
  The Z4 verdict (0.01 CONFIRMED at the pilot filling) stands separately;
  Q3 tests whether the filling composition shifts it.
- The (10,10) Li⁺ exclusion is a physical result the paper can state
  (with the desolvation reading marked as interpretation).
