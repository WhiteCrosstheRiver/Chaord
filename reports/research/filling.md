# Y6 reservoir filling — the Q2 analysis (Review 12, D49 binding)

Contract (open_items_v6.md Q2, verbatim): "From the fill series: molecules
and ions per Å in the middle of each tube (more than 1 nm from either
mouth), block means and errors. Compare per species with the pilot
compositions per Å ((10,10): EC 8, EMC 5, LiPF6 2 per 49.2 Å; (16,16):
EC 37, EMC 24, LiPF6 6 per 49.2 Å)."

Everything below is printed by
`research/cnt_electrolyte/analysis/filling_analysis.py` (committed), which
also writes `research/cnt_electrolyte/analysis/filling_results.json`
(per-frame series for all 1001 frames of both fills + every statistic
quoted here + the validation records).

Inputs: `research/cnt_electrolyte/fills_cluster/fill_cnt{10,16}/
{final.data, log.lammps, prod_thinned.dump}`, sha256-verified against
`fills_cluster/fill_series_manifest.json` before parsing (V0).

## Geometry as found (Q2 step 1 — recorded, not assumed)

- The fill cell is NOT "tube continuous through a periodic box": the tube
  is a finite open segment, z ∈ [30.0, 79.2] Å (builder record
  `tube_z_span_A` in `data/stageA_build_manifest.json`; 20 unit cells ×
  2.46 Å = 49.2 Å — the same length as the pilot tubes, per Y6 "as the
  pilot, so the numbers compare"), centred in a box with a 30 Å reservoir
  on each side (log read_data box z = 0 → 109.2).  The tube is molecule 1,
  frozen (`fix setforce`, never dilated by `fix nph ... dilate mobile`):
  its coordinates are bit-identical in all 1001 frames of both series (V5).
  The outermost tube ATOMS sit at z = [30.0, 77.97] (the top ring is
  1.23 Å inside the declared top mouth plane).
- Middle window: [40.0, 69.2] Å (more than 1 nm from either declared
  mouth plane), length 29.2 Å.  Sensitivity: window from the atom extent,
  [40.0, 67.97] Å, is reported alongside.
- A molecule/ion is counted iff its mass-weighted centre of mass
  (unwrapped with the dump image flags) is in the window AND radially
  inside the geometric tube cylinder, r_com < R (R measured from the
  frozen tube atoms: 6.7813 Å for (10,10), 10.8502 Å for (16,16); spread
  < 1e-6 Å).  The pilot counts (everything packed inside the periodic
  tube) satisfy the same criterion trivially.  Without the radial
  criterion the axial window alone counts BATH molecules — the tube and
  the bath share the same z range.
- Normalisation (where a per-Å comparison can silently go wrong): the
  WINDOW count is a density in 29.2 Å and is divided by 29.2 Å; the pilot
  count is a whole-tube count over 49.2 Å and is divided by 49.2 Å.  A
  window count divided by the full tube length understates the reservoir
  per-Å density by the factor 29.2/49.2 = 0.593 (a draft of this analysis
  made exactly that error and reported +3.6%…+26.4% where the correct
  values are +21%…+119%).
- Species: Li+ = the single 6.94 amu atom (type 9, q +1); PF6− = P + 6 F
  (types 10/11, q +1.70/−0.45); EC (10 atoms) and EMC (15 atoms) by the
  per-molecule TYPE census of the build templates — validated three ways
  below.

## Validation (all recorded in filling_results.json)

- V0 sha256 of final.data / log.lammps / prod_thinned.dump ==
  fill_series_manifest.json (both fills).
- V1 census by mol id == census by bond-graph connected components, for
  every mobile molecule; the tube body is written WITHOUT bonds (solvent
  bond counts match the data-file headers exactly: 14,620 and 19,892), so
  it is asserted separately: exactly one body, all type 1.
- V2 SPOT-CHECK AGAINST THE KNOWN FILL COMPOSITION: the data-file census
  is {EC 814, EMC 420, Li+ 100, PF6− 100, tube 1} for fill_cnt10 and
  {EC 1132, EMC 551, Li+ 143, PF6− 143, tube 1} for fill_cnt16 ==
  the builder's own record (`stageA_build_manifest.json` composition;
  atom totals 16,040 / 22,009 check).  Per-molecule charges: EC/EMC 0,
  Li+ +1, PF6− −1; whole cell neutral to 1e-13.
- V3/V4 every one of the 1001 frames per fill: atom ids 1..N sorted, and
  the per-atom type and mol id arrays are IDENTICAL to the data file —
  so the dump's per-molecule species map is the data file's by
  construction, in every frame.
- V5 tube frozen (max deviation 0.0e+00 Å both fills); tube axis on the
  box centre (max 3.6e-17 Å).
- V6 log.lammps cross-checks: read_data box == build manifest box;
  "atoms in group tube/li" == 800/100 and 1280/143; production dump
  cadence 1000 steps; final.data written at step 5,000,000 == last dump
  timestep.
- V7 the series is exactly the 1/5-thinned 1 ps production cadence:
  timesteps 0..5,000,000 at exactly 5000-step (5 ps) spacing, 1001
  frames.
- Independent recount: a separate brute-force parser (different code
  path) recomputes the LAST frame of each fill from raw dump text —
  counts match the stored series exactly (cnt10: EC 8, EMC 7, Li+ 0,
  PF6− 1; cnt16: EC 38, EMC 17, Li+ 8, PF6− 7).

## Blocks

Production = the 5 ns NPT at 1 atm after the 650 ps staged decompression
(log: `reset_timestep 0`, `run 5000000`, dt 1 fs from the "every 1000
steps = 1 ps" dump cadence).  Per the Q2 brief ("2 blocks of the
production half"): block A = t ∈ [2500, 3750) ps (250 frames), block
B = t ∈ [3750, 5000] ps (251 frames).  Central value = per-frame mean of
the half; error = between-block SEM = std([A,B], ddof=1)/√2 (with
n_blocks = 2 this equals |A−B|/2 exactly).  HONEST CAVEAT: Y8 asks for
blocks of ≥ 5 τ_M; stage_a.md measures τ_M ≈ 2.7–3.0 ns at the working
field for these tubes, so a 1.25 ns block is ~0.4–0.5 τ_M and the
between-block SEM is a LOWER bound on the true uncertainty.  The
per-frame SEM of the half is also recorded (it is smaller, i.e. more
optimistic; the correlation is real).

## The verdict table

Mean counts in the middle window [40, 69.2] Å over the production half;
per-Å = count / 29.2 Å; pilot per-Å = pilot count / 49.2 Å; %diff =
(fill − pilot)/pilot.  "sens." = %diff recomputed on the atom-extent
window [40, 67.97] Å (27.97 Å).

### (10,10) — fill_cnt10

| species | pilot /49.2 Å | pilot /Å | fill count (A / B) | ± block SEM | fill /Å | %diff (sens.) |
|---|---|---|---|---|---|---|
| EC | 8 | 0.1626 | 8.50 (8.68 / 8.32) | 0.179 | 0.2910 | **+79.0%** (+84.9%) |
| EMC | 5 | 0.1016 | 6.57 (6.57 / 6.56) | 0.005 | 0.2249 | **+121.3%** (+115.6%) |
| Li+ | 2 | 0.0407 | 0.00 (0.00 / 0.00) | 0.000 | 0.0000 | **−100.0%** (−100.0%) |
| PF6− | 2 | 0.0407 | 1.00 (1.00 / 1.00) | 0.000 | 0.0342 | **−15.8%** (−12.0%) |

Whole declared tube (mouths included), half means: EC 15.12, EMC 11.01,
Li+ 0.09, PF6− 1.00.

### (16,16) — fill_cnt16

| species | pilot /49.2 Å | pilot /Å | fill count (A / B) | ± block SEM | fill /Å | %diff (sens.) |
|---|---|---|---|---|---|---|
| EC | 37 | 0.7520 | 37.09 (36.85 / 37.32) | 0.235 | 1.2701 | **+68.9%** (+70.2%) |
| EMC | 24 | 0.4878 | 17.28 (16.97 / 17.58) | 0.305 | 0.5917 | **+21.3%** (+26.3%) |
| Li+ | 6 | 0.1220 | 7.81 (7.64 / 7.97) | 0.166 | 0.2673 | **+119.2%** (+128.9%) |
| PF6− | 6 | 0.1220 | 6.21 (6.35 / 6.07) | 0.140 | 0.2127 | **+74.4%** (+74.2%) |

Whole declared tube (mouths included), half means: EC 64.21, EMC 29.92,
Li+ 8.34, PF6− 9.00.  Cross-check of the normalisation: the whole-tube
per-Å density (e.g. EC 64.21/49.2 = 1.305/Å) agrees with the window
per-Å density (1.2701/Å) to 3% — the pore is uniformly filled, and BOTH
are ~70% above the pilot's 0.752/Å.

## D49 gate — Q3 TRIGGERS

Every species in (16,16) differs from the pilot by far more than 5%
(smallest |%diff|: EMC +21.3%): **Q3 is triggered** — a new confirmatory
set at the reservoir filling (3 builds, fields 0 and 0.01 V/Å, the Z3
protocol, the D39 test pre-registered in a PLAN before submission) is now
required, and its shift is to be reported beside Z4's.  (10,10) is not
the gate tube, but it fails the same comparison (all four species
≥ 15.8% off, Li+ −100%).

## Filling dynamics — equilibrated at the count level, with caveats

- The production series starts POST-IMBIBITION: at production t = 0 the
  window already holds its plateau content (cnt16: EC 37, EMC 17, Li+ 7,
  PF6− 6 vs half means 37.1 / 17.3 / 7.8 / 6.2; cnt10: EC 9, EMC 6, 0,
  1 vs 8.5 / 6.6 / 0 / 1).  The filling itself happened during the
  650 ps staged anneal that precedes `reset_timestep 0`, so the 5 ns
  series cannot show the imbibition transient.
- Plateau (trailing 250 ps rolling mean within 10% of the half mean) is
  reached by t ≈ 245 ps for every species except cnt10 PF6− (305 ps):
  that "ion" is a single trapped PF6− whose COM moves between the middle
  window and the mouth region (window series takes only the values 0
  and 1; whole-tube count is exactly 1 in every frame).
- Residual drift over the half (OLS slope): cnt10 EC −0.40/ns, EMC
  −0.02/ns, Li+ 0, PF6− 0; cnt16 EC +0.34/ns, EMC +0.46/ns, Li+ +0.31/ns
  (still slowly accumulating: +4%/ns of 7.8), PF6− −0.14/ns.  Block A vs
  B differ by ≤ ~0.5 counts everywhere except nothing.  With τ_M ≈
  2.7–3.0 ns, 5 ns of production is < 2 τ_M: COUNT-level stationarity is
  what can be claimed (drift ≤ a few %/ns), not full equilibration of
  the slow orientational degrees of freedom.

## What the comparison means (observations, marked as such)

1. The pilot compositions are NOT the reservoir equilibrium of this
   model.  Both pores hold substantially more solvent per Å than the
   pilot's packed counts (cnt16: EC +69%, EMC +21%; cnt10: EC +79%, EMC
   +121%).  Expected direction: the pilot numbers were the densest
   PACKABLE configurations (the pack-then-relax packing ceilings recorded
   in params.py), not equilibrated uptakes.
2. Ions: the (16,16) pore ACCUMULATES ions above the pilot (Li+ +119%,
   PF6− +74%); the (10,10) pore EXCLUDES them — 0 Li+ in the tube in all
   1001 frames (max whole-tube count 1) and exactly one trapped PF6−.
   The pilot's 2 pairs per 49.2 Å in the narrow pore were placed by
   construction; a reservoir at equilibrium does not maintain them.
   The cnt10 ion statistics rest on 0–1 ions — qualitative only.
3. Bath caveat: the achieved bath is 0.84 M (not 1 M) and EC-rich
   (EC:EMC = 814:420 ≈ 1.94:1 by number vs 1.56:1 for the 1:1-by-volume
   target) because EMC insertions were skipped at packing (build
   manifest, `skipped`).  The cnt16 pore EC:EMC ratio (37.1:17.3 =
   2.14) tracks the bath ratio (1.94) more than the pilot's in-tube
   ratio (1.6) — part of the EC excess is bath composition, not pore
   thermodynamics.  A like-for-like re-run of the pilot packing at the
   bath's composition is NOT done here; the D49 gate reads the raw
   per-species differences, as written.

## Which filling the confirmatory set used (honest statement)

- The Z3 confirmatory builds (cf16_b1..b6, 98.4 Å tubes) were composed
  from the PILOT composition doubled to the tube length (74 EC / 48 EMC
  / 12 LiPF6): `confirmatory/data/build_manifest_cf16_b1.json` records
  `composition_source = "pilot 37/24/6 x2 for the same molarity at 2x
  pore volume (PLAN.md section 6)"` (commit 6f889bd).  That was the only
  defensible choice at build time: the reservoir-filling counts that
  could have replaced the pilot scaling are THIS analysis, which ran
  2026-11-02 — after Z3's submission (1113f15, 2026-10-31 12:05) and
  after Z4 (4de4d14).
- PLAN.md (fb50f3f) closes with "the confirmatory set uses the reservoir
  filling, not the pilot", which misdescribes the build composition
  (the build-level record says pilot-scaled).  Note the fill RUNS
  themselves had completed on the cluster before submission (final.data
  mtimes 2026-10-30 22:54 / 10-31 05:23; "both COMPLETED" already in
  fb50f3f) — what did not exist was their ANALYSIS.  A corrections.md
  entry for the PLAN sentence is suggested (Q1/S2's file; not edited
  here).
- Q2's verdict makes the distinction material: the reservoir equilibrium
  differs from the pilot by ≫ 5% for every species, which is exactly why
  Q3 triggers.  Z4's verdict (0.01 V/Å CONFIRMED at the pilot filling)
  stands as measured; Q3 tests it at the reservoir filling.
