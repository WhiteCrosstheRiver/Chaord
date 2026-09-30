# Independent verification report — Gate A′ — 2026-09-28

**Verifier role.** Fresh agent with no authorship of any `src/`, `tests/` or
`tools/` code in this repository. Scope: rerun the acceptance suite, rerun the
mutation canaries and the fast test suite, and independently spot-check the
noise-floor honesty. No repo file was modified by the verifier except this
report; nothing was committed.

**Environment.** Windows 10 (Git Bash), repo at the repository root
(local working path removed post Review 3 — reports must not carry
machine-local paths), interpreter `.venv/Scripts/python.exe`.

## 1. Commands executed and top-line results

| # | Command | Result |
| --- | --- | --- |
| 1 | `.venv/Scripts/python.exe tools/acceptance.py -o /tmp/acc_verify.json --details /tmp/acc_verify_details.md` | **9/14 criteria pass**; runner exit code 1 (expected whenever any criterion fails); duration within the predicted 15–20 min envelope (subprocess rebuilds included) |
| 2 | `.venv/Scripts/python.exe -m pytest tests/acceptance -q` | **17/17 passed** in 46.5 s (16 test functions; one A12 stub test parametrized ×2) |
| 3 | `.venv/Scripts/python.exe -m pytest tests/ -q -m "not slow" --tb=no` | **382 passed, 1 skipped, 7 deselected** in 178 s; the single skip is benign (`tests/test_defects.py:71 — unary host has no antisites`); rerun reproduced 382 passed identically |
| 4 | Independent g(r) recomputation (verifier's own numpy code, below) | JSON noise-floor values reproduced to **0.2 %** relative |

## 2. Per-criterion verdicts (from the verifier's own run, `/tmp/acc_verify.json`)

| ID | Verdict | Key numbers (verifier's run) |
| --- | --- | --- |
| A1 parse & format | **PASS** | 9/9 spec examples parse; parse==IR and fmt-idempotent hold on 10,000 generated programs |
| A2 canonical invariance | **FAIL** | 7/8 crystal cases; 14/16 rigid transforms byte-identical; only `crystals/fcc_crconi` fails (0/2) |
| A3 exact round trip | **FAIL** | 7/8 lift-build-lift byte-identical and StructureMatcher-fitting; `fcc_crconi` text differs (extra `constrain sro alpha1 Ni-Ni +0.18 ± 0.02`), matcher no fit |
| A4 defect recovery | **FAIL** | recall ≥ 0.95 in 22/22 cells; precision ≥ 0.95 in 18/22; worst precision 0.09 (L12-NiAl / interstitial / 0.8 Tm, 30 false frenkel pairs vs 3 planted) |
| A5 statistical round trip | **FAIL** | with floors on record: 0/3 pass (fluid 0/1, interface 0/1, glass 0/1; targets 90 %/90 %/80 %). Distances vs floor: lj_liquid gr_rms 0.684 vs 0.092 (×7.4), cn_tv 0.214 vs 0.060 (×3.6); lj_glass gr_rms ×12.7; lj_solid_liquid gr_rms ×27.4. Plus 4 build failures: `nacl_aq` (ValueError 'H10NaO5' is not in list), `water_tip4p` (ChaordError: cannot place 256 H2O at this density), `lj_solid_liquid` bench case (rebuild time budget), `cu_water` (ValueError 'Cu384' is not in list) |
| A6 conservation | **PASS** | 126/126 lifts: atom counts == conserve line, charge consistent; verifier-side region arithmetic pins/cross-checks 86 programs |
| A7 phase segmentation | **PASS** | 10/10 interface frames ≥ 95 % correct labels; worst 0.999 (cu_water frame 3); interface band of 2×d_NN excluded |
| A8 reactive census | **PASS** | 2 independently constructed planted cases exact: {H2O 40, HO 9, H 9} and {H2O 25, HO 5, H 7} |
| A9 compression | **FAIL** | 0 raw bench frames ≥ 1,000 atoms (largest 792) → criterion not demonstrable on this bench; worst ratio 34.29 % (bcc_fe, 16 atoms); supplementary tiled 2×2 L12 case reaches 1,012 atoms at 0.74 % — honestly not counted |
| A10 determinism | **PASS** | repeated lift byte-identical; same-seed builds identical coordinates |
| A11 speed | **PASS** | 100,000-atom fluid lift in 3.9 s (target ≤ 120 s); program states the exact atom count |
| A12 static checks | **PASS** | all four seeded errors caught: lattice_mismatch, impossible_density, charge_imbalance, overlap 0.1σ |
| A13 no crashes | **PASS** | 125/125 bench frames lift with zero unhandled exceptions; 0 atoms in residual blocks |
| A14 documentation | **PASS** | 35/35 dialect statement keys covered by `docs/reference.md`; worked examples found for 26/35 keys |

**Total: 9/14 PASS, 5 FAIL (A2, A3, A4, A5, A9).**
These numbers coincide with the authors' checked-in `reports/acceptance.json`
on every criterion except A5: the checked-in file predates the population of
`reports/noise_floors.json` (its A5 says "no floor on record"); this rerun is
the first to exercise the A5 gate with floors on record, and the gate fails
0/3 — an honest, newly measured failure, not a regression introduced by the
verifier.

## 3. Independent g(r) honesty spot-check

The verifier wrote a from-scratch numpy g(r) (minimum-image convention, all
pairs, ideal-gas shell normalisation; no `chaord` import) and applied it to
`bench/reference/lj_liquid/frame_{0,1,2}.npz` (N = 500, cubic box
L = 8.37883606, number density 0.8500).

* Physics sanity of the frames: g(r) first peak at r = 1.10 σ, height 3.16,
  g(r→r_max) ≈ 1.02 — a textbook LJ-liquid radial distribution.
* Verifier's own binning (r_max = 4.0, 100 bins, all bins): gr_rms(0,1) =
  **0.073**; (r_max = 4.0, 80 bins): 0.062 — same order of magnitude as the
  JSON value.
* Reproducing the LJ dialect's published regime (r_max = 2.5 σ, 60 bins, RMS
  restricted to r ≥ 0.8 σ): verifier gr_rms(0,1) = **0.10062** vs JSON
  **0.10042** (relative difference 0.2 %); gr_rms(0,2) = **0.10753** vs JSON
  **0.10732** (0.2 %).

Conclusion: the `gr_rms` entries in `reports/noise_floors.json` are genuinely
computed from the actual reference frames; no fabrication detected. The
independent numbers above are the verifier's own.

## 4. Verifier opinion

**Reproducibility.** Every PASS claim in the authors' report reproduced
exactly under an independent rerun, the mutation canaries prove the acceptance
checker itself flips on 17/17 seeded faults, and the noise-floor table survived
an independent recomputation to 0.2 %. The measurement infrastructure is
trustworthy; the FAILs below are therefore credible measurements, not
measurement artifacts.

**Honest physics gaps (failures of capability, correctly reported):**

* **A5 (deepest gap).** Two separable problems. (i) Four cases cannot rebuild
  at all — two `ValueError` species-key bugs (`'H10NaO5'`, `'Cu384'`), one
  packing failure (water_tip4p), one time-budget timeout; these are code
  defects the authors should fix. (ii) The three cases that do rebuild sit
  7–27× above their noise floors: the generator cannot yet reproduce liquid
  structure from a lifted program. That is a genuine physics gap and the
  single most important open problem for v1.0.
* **A4 (partial).** Recall is perfect (22/22); precision fails only for
  interstitials at 0.8 T_m, where thermal disorder makes planted
  interstitials and near-frenkel pairs hard to separate (worst 3 planted vs
  30 reported frenkel pairs). Honest gap: defect classification at high
  temperature, needs author work (e.g. displaced-atom classification with
  uncertainty), not a data problem.
* **A9 (bench design, not code).** No raw bench frame reaches 1,000 atoms, so
  the 2 % criterion cannot be demonstrated. The supplementary tiled
  measurement (1,012 atoms at 0.74 %) suggests the code would pass once
  ≥ 1,000-atom reference cases exist; adding them is an author action.

**Narrow, fixable code gaps:** A2/A3 fail on exactly one case
(`fcc_crconi`, an SRO-constrained random alloy whose lifted program adds an
SRO constraint that breaks byte-invariance and the round trip). All other
crystal prototypes are byte-perfect. This is a well-localised lifter gap.

**Bookkeeping note:** `reports/acceptance.json` in the repo is one step stale
(A5 evidence predates the noise floors); updating it with the current
A5-measured output would keep the paper trail consistent.

Overall: the 9/14 result is an accurate, independently confirmed snapshot.
The five FAILs are real; none is cosmetic, and the A5 build failures plus the
fcc_crconi round-trip bug are the items most clearly on the authors' critical
path, ahead of the harder physics (statistical fidelity of rebuilt fluids,
high-T defect precision).

---

Independently verified by a fresh agent (no code authorship); 2026-09-28
