# X7 item 1 — Kob–Andersen quench-rate convergence (Review 10, convergence stream)

30 independent LAMMPS quenches of the Kob–Andersen 80:20 binary LJ glass:
10 seeds at each of three rates one decade apart (0.019, 0.0019, 0.00019
T*/tau), N = 2000, rho* = 1.2, inherent-structure (IS) energy per quench
(D20), energy and structure differences per decade with confidence
intervals, judged against the repo's own cross-quench floors.  All numbers
below come from the committed artifacts under
`research/ka_convergence/` (this report's JSON twin: `ka_rates.json`).

**Answer up front.** The IS energy separates the rates decisively: every
decade costs 0.023–0.040 epsilon/atom of IS energy with 95% CIs 5–10x
smaller than the shift, i.e. 6–10x the cross-quench IS spread — A5 with the
D20 observable tells adjacent decades apart.  The structure observables do
not: rate-mean g(r)/cn distances are 0.5–1.0x the (own) cross-quench floor
at one decade; only the full 100x rate ratio exceeds the bench floor
threshold, and not the honest 10-quench floor measured here.

## 1. Protocol and parameter cross-check (every value cited to repo records)

| Parameter | Value used | Repo record |
|---|---|---|
| eps AA/AB/BB | 1.0 / 1.5 / 0.5 | `src/chaord/dialects/lj_mixtures.yaml` kob_andersen; `bench/reference/generate_reference.py` KA_EPSILON; bench `ka_glass/provenance.json` (identical); Kob–Andersen PRE 51, 4626 (1995) Table I |
| sigma AA/AB/BB | 1.0 / 0.8 / 0.88 | same three records |
| masses | 1.0 both | same |
| cutoffs | 2.5 sigma_ab per pair (2.5 / 2.0 / 2.2 sigma_AA) | `rc_factor_sigma_ab 2.5`, energy-shifted at its own cutoff, forces plain truncated (provenance; lj_mixtures.yaml) — implemented as `pair_style lj/cut 2.5` + per-pair `pair_coeff ... <rc>` + `pair_modify shift yes` |
| N / composition | 2000 = 1600 A + 400 B | provenance protocol |
| rho* | 1.2 | provenance; lj_mixtures.yaml reference_point |
| start config | fcc 8x8x8 in target box, 48 random deletions, random 80:20 assignment | provenance protocol, verbatim construction (`case_ka_glass`) |
| dt | 0.005 tau/step | provenance `units.dt_reduced`; glass.yaml `md.relax_dt` |
| melt | T* = 2.0, 2000 steps | glass.yaml D7 block at `glass_protocol_ref_n` 2000 (scale 1.0): `glass_melt_T 2.0`, `glass_melt_steps 2000` |
| quench | linear 2.0 -> 0.1; steps = 1.9/rate/0.005 = 20000 / 200000 / 2000000 (exact integers) | X7 order formula |
| anneal | T* = 0.1, 4000 steps | glass.yaml `glass_anneal_steps 4000` |
| sampling | 5 frames, 800-step stride, first frame one stride after anneal end | bench ka_glass provenance `steps` block (frames 44800..48000 for the 20000/20000 run) |
| thermostat | Langevin NVT, gamma* = max(0.5, 10/tau_stage) and on ramps max(..., 500 x \|dT/dt\|): melt 1.0; quench 9.5 / 0.95 / 0.5; anneal + sampling 0.5 (LAMMPS `fix langevin`, `zero yes`, damp = 1/gamma) | D19(c)/X5: glass.yaml `ramp_gamma_tau_multiple 10`, `ramp_gamma_rate_multiple 500`, `md.relax_gamma 0.5` |

Stage-end kinetic temperatures, all 30 runs (D19c provenance requirement,
in each `runs/<id>/summary.json`): quench end 0.0981–0.1043 vs stated 0.1;
anneal end 0.0969–0.1029.  The thermostat tracks the ramp; compare the
reference protocol's lag (kinetic T* 0.280 at the 0.1 bath, open_items_v3
X5 item 2).

Sanity inherited from the bench recipe: per-quench 5-frame energy flatness
max–min 0.0022–0.0110 epsilon/atom over all 30 quenches, inside the
recorded bound 0.015 (provenance `energy_flatness_per_quench`).

Documented deviations from the bench reference protocol (both ordered by
the X7 brief / D19c, none silent):

1. **Thermostat**: bench reference ran gamma* = 0.5 throughout (and lags);
   these runs use the D19(c) gammas above.
2. **Melt length**: bench reference 20000 steps (frozen W7 protocol);
   X7 brief pins the melt/anneal to the glass.yaml D7 block scaled at
   N = 2000 -> 2000/4000 steps.
3. **Engine**: LAMMPS 22Jul2025-update4 (`units lj`) instead of ASE
   Langevin — D23 expressly allows this.  Validated below, not assumed.

## 2. Validation against the repo's own records (`validation/report.json`)

- **V1 potential**: LAMMPS pe/atom of the 6 most-annealed bench frames
  (3,4,8,9,13,14) matches the values recorded in
  `reports/ka_glass_d7_sweep.json` to 3.5e-5 (recorded are rounded to 4
  decimals; means -7.4603244098 vs -7.4603244099).  The `pair_coeff` +
  `shift yes` setup is exactly the repo's cut-and-shifted KobAndersenLJ.
  (Note `units lj` normalises thermo `pe` per atom — recorded as such.)
- **V2 inherent structure (D20)**: FIRE minimisation of the bench
  quench-final frames gives -7.6197 / -7.6110 / -7.5985 per atom vs the
  values recorded in open_items_v3.md X7 (-7.615 / -7.611 / -7.598);
  max deviation 0.0047 (different minimiser detail, same basins).
- **V3 observables**: the numpy g(r)/partial-g/cn implementation
  (`ka_obs.py`) recomputes the 12 recorded cross-quench floor pairs of
  `reports/noise_floors.json` ka_glass with **rel dev 0.0** on every
  statistic (gr_rms, cn_tv, and the A-A/A-B/B-B partials) — the analysis
  convention is bit-identical to `chaord.cv.noise`/`rich` at the ka_glass
  dialect thresholds (gr_rmax 2.5 / 60 bins / cn cutoff 1.5 / gr_rmin 0.8).
- **IS robustness** (`validation/is_robustness.json`): a fire->hftn->fire
  cascade to machine-precision forces reproduces all 30 single-FIRE IS
  energies within 5.8e-5 per atom (mean 5e-6).  The few stalled single-FIRE
  runs (residual fnorm up to 1.1) were already in their basins.

## 3. Measured cost (not guessed)

- Throughput, N = 2000 LJ reduced units, one serial `lmp.exe` process:
  880–1020 steps/s alone, ~916 steps/s with 10 processes in parallel
  (32-core box) — 1.8–2.0e6 atom-steps/s per process.
- Steps per run: 30,000 (r0) / 210,000 (r1) / 2,010,000 (r2) MD steps +
  ~0.7–2k FIRE iterations.
- Measured wall: r0 batch 36 s (10 parallel), r1 batch 263 s, r2 batch
  2356 s — **~44 min total** for 2.25e7 MD steps = 4.5e10 atom-steps
  (~6.6 h serial-equivalent).  Far under the 4-day bound, so all 30
  quenches ran; nothing queued.

## 4. Results

### 4.1 Inherent-structure energy (D20) — the rate signal

Per-rate mean over 10 independent quenches (epsilon/atom; FIRE on the
final sampled frame; bootstrap 95% CI over quenches):

| rate (T*/tau) | IS energy | sd | 95% CI | cross-quench \|dIS\| mean / P90 / max |
|---|---|---|---|---|
| 0.019    | -7.5987 | 0.0062 | [-7.6021, -7.5947] | 0.0069 / 0.0163 / 0.0181 |
| 0.0019   | -7.6391 | 0.0049 | [-7.6421, -7.6363] | 0.0055 / 0.0100 / 0.0173 |
| 0.00019  | -7.6626 | 0.0037 | [-7.6648, -7.6604] | 0.0042 / 0.0083 / 0.0132 |

Differences per decade (Welch 95% CI; positive = slower rate is deeper):

| comparison | decades | dIS (epsilon/atom) | 95% CI | significant |
|---|---|---|---|---|
| 0.019 vs 0.0019   | 1 | +0.0404 | [+0.0352, +0.0457] | yes |
| 0.0019 vs 0.00019 | 1 | +0.0234 | [+0.0194, +0.0275] | yes |
| 0.019 vs 0.00019  | 2 | +0.0639 | [+0.0590, +0.0688] | yes |

The thermal (un-minimised) potential energy per atom moves identically
(-7.4487 / -7.4896 / -7.5129 last-two-frame means; decade diffs +0.0409
[+0.0352,+0.0466] and +0.0233 [+0.0186,+0.0280]).  The IS shifts are 6–10x
the cross-quench IS pair spread (0.0042–0.0069 mean) and the CIs exclude
zero by >10 SE.  There is no plateau by 10^4 tau: the KA glass at this
state point keeps gaining ~0.02–0.04 epsilon/atom per decade of slower
cooling, with diminishing returns (first decade > second).

For scale: the bench reference's 3 quenches span 0.017 epsilon/atom in IS
energy (open_items_v3 X7) — the same order as one decade of rate here, so
rate is a first-order knob on IS energy, seed noise is not.

### 4.2 Structure — below the floors at one decade

Distances between rate-mean observables (per-quench observables averaged
over the last two most-annealed frames, then over the 10 quenches — the
floor's own convention; 2000-resample bootstrap 95% CI over quenches):

| comparison | gr_rms | 95% CI | gr_rms A-A | A-B | B-B | cn_tv | 95% CI |
|---|---|---|---|---|---|---|---|
| 0.019 vs 0.0019   | 0.0278 | [0.0235, 0.0347] | 0.0372 | 0.0467 | 0.0654 | 0.0100 | [0.0077, 0.0184] |
| 0.0019 vs 0.00019 | 0.0170 | [0.0151, 0.0245] | 0.0216 | 0.0332 | 0.0508 | 0.0128 | [0.0092, 0.0237] |
| 0.019 vs 0.00019  | 0.0424 | [0.0385, 0.0487] | 0.0566 | 0.0747 | 0.0941 | 0.0176 | [0.0116, 0.0260] |

Floors (gr_rms / cn_tv):

| floor | gr_rms | cn_tv |
|---|---|---|
| bench ka_glass pair floor (12 pairs, 3 quenches) | 0.0378 | 0.0310 |
| bench ka_glass avg floor (recorded, A5's reference) | 0.0258 (P90 0.0273) | 0.0233 (P90 0.0286) |
| this work, own cross-quench floor, r0 (45 pairs) | 0.0268 (P90 0.0323) | 0.0243 (P90 0.0341) |
| same, r1 | 0.0291 (P90 0.0348) | 0.0261 (P90 0.0373) |
| same, r2 | 0.0323 (P90 0.0402) | 0.0278 (P90 0.0382) |

A5 criterion (1.5 x max(mean, P90) of the cross-quench floor):

- **One decade** (0.019 vs 0.0019): gr_rms 0.0278 vs thresholds 0.0603
  (own) / 0.0410 (bench avg) — **not told apart**; cn_tv 0.0100 vs 0.0573 /
  0.0429 — not told apart.  Second decade likewise (smaller still).
- **100x ratio** (0.019 vs 0.00019): gr_rms 0.0424 > bench-avg threshold
  0.0410 (marginal, 1.03x) and A-A 0.0566 > 0.0553, but below the honest
  own-floor thresholds (0.0603 / 0.0689) — structure is **borderline by
  the bench floor, within noise by the 10-quench floor**.
- The partial B-B channel shows the largest relative movement (0.0654 ->
  0.0941 over 1–2 decades; the B-B peak sharpens from 1.59 to 1.77 in
  height as the frustrated B sublattice orders slightly) but sits under a
  B-B floor of 0.13–0.18.

Mean structure moves in the right direction but is small: AA first-peak
height 5.40 -> 5.32 -> 5.29, B-B first peak 1.59 -> 1.63 -> 1.77, mean
coordination (cutoff 1.5) 14.94 -> 14.92 -> 14.91; peak positions pinned
(1.0625 / 0.8958 sigma_AA), matching the bench-recorded windows.

### 4.3 Bridge to the bench reference

Our rate-0.019 rate-mean sits gr_rms 0.0136 / cn_tv 0.0095 from the bench
ka_glass 6-frame mean — inside the bench cross-quench floor, i.e. the same
nominal-rate family despite the D19c thermostat change.  The bench
quenches' IS energies (-7.615..-7.598, mean ~-7.608) fall between our r0
(-7.5987) and r1 (-7.6391): the gamma*-0.5 reference quench, whose kinetic
temperature lags the bath high through the ramp (X5's 0.280 at bath 0.1),
relaxes like a tracked quench slower than 0.019 but faster than 0.0019 —
thermostat lag acts as extra effective annealing time, consistent with the
X5/D19c analysis.

### 4.4 Can A5 tell the rates apart?

- **With the D20 IS-energy observable (this stream's addition): yes,
  cleanly, one decade is enough** (shift 0.023–0.040 vs IS cross-quench
  spread 0.004–0.007; every CI excludes zero by an order of magnitude).
- **With the legacy structure observables alone: no** at one decade
  (distances at or below the floor), and only marginally at 100x against
  the bench-recorded floor — the honest 10-quench floors measured here say
  no.  This is precisely the case D20 was opened for.

## 5. File layout

```
research/ka_convergence/
  ka_params.py            single-source protocol/params (cited inline)
  gen_inputs.py           data.+in. generator, inputs_manifest.json
  run_jobs.py             parallel queue runner (DONE markers, idempotent)
  parse_runs.py           log parser, npz converter, sha256 manifests
  ka_obs.py               numpy floor observables (validated bit-exact, V3)
  validate_against_bench.py   V1/V2/V3 validation vs repo records
  is_robustness.py        fire->hftn->fire cascade check on all 30
  analyze.py              statistics, CIs, verdicts -> ka_rates.json
  validation/report.json, is_robustness.json, logs + bench frame checks
  runs/<r{0,1,2}_q{00..09}>/
    data.<id> in.<id> log.lammps DONE
    frame_{0..4}.lammpstrj + .npz (bench r/L/symbols convention)
    is_frame.lammpstrj + .npz   summary.json   sha256.txt
  runs/inputs_manifest.json, energy_flatness.json
reports/convergence/ka_rates.json   (numbers behind every table above)
reports/convergence/ka_rates.md     (this file)
```

Generators/analysis import numpy/scipy only — no chaord import (D23).
Per-run `sha256.txt` covers every artifact (data, input, log, raw dumps,
npz frames, IS frame); `summary.json` records the protocol, stage-end
kinetic temperatures, frame energies and the IS result per quench.

Reproduce: `python gen_inputs.py && python run_jobs.py r0 r1 r2 --jobs 10
&& python parse_runs.py && python analyze.py` (then
`python validate_against_bench.py && python is_robustness.py`).
