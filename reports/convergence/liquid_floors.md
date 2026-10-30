# X7 item 2 — liquid noise floors re-grounded on ≥ 30 decorrelated pairs
(Review 10/11 convergence stream, 2026-10-31)

The two LJ liquid cases' floors — `lj_liquid` (averaged floor 10 pairs) and
`lj_liquid_large` (averaged floor 11 pairs, frame-pair floor 21 pairs) —
were re-grounded on 3 new independent LAMMPS trajectories per case
(7 626 decorrelated frame pairs and 132/162 averaged pairs per case), with a
seeded bootstrap 95% interval on every floor. This report's JSON twin:
`liquid_floors.json`. Code, runs and manifests: `research/x7_liquid/`
(chaord-free, D23). Order: `docs/reviews/open_items_v3.md` X7 item 2,
unchanged by v5 (S3: "the rest of X7 in idle time").

**Answer up front.** The committed floors largely hold, and the X7 item's
diagnosis is confirmed in both directions:

- `gr_rms` floors move at most ±12%: the frame-pair floors TIGHTEN slightly
  (lj_liquid 0.1204 → 0.1102, x0.92; large 0.0528 → 0.0545, x1.03) and the
  averaged floors hold (large 0.0335 → 0.0326, x0.97) or widen slightly
  (lj_liquid 0.0472 → 0.0530, x1.12). With ≥ 132 pairs the P90 is no longer
  the maximum; the committed 10–21-pair P90s were max-dominated — too high
  for lj_liquid's frame-pair floor, slightly low for its averaged floor.
- `cn_tv` floors for `lj_liquid_large` widen substantially (frame-pair
  0.0293 → 0.0400, x1.37; averaged 0.0167 → 0.0262, x1.57): the committed
  21 pairs from 10 frames under-sampled the coordination-histogram
  fluctuation tail (the new 5 043-pair q90 0.0400 exceeds the committed
  21-pair max 0.0347). The three-way check below shows the new frames are
  the same equilibrium distribution as the committed ones, so this is
  sampling, not physics.
- Verdict per floor: frame-pair lj_liquid gr/cn TIGHTEN x0.92/x0.93;
  averaged lj_liquid gr WIDENS x1.12, cn HOLDS x1.02; frame-pair large gr
  HOLDS x1.03, cn WIDENS x1.37; averaged large gr HOLDS x0.97, cn WIDENS
  x1.57. If the gate (A5) adopts the re-grounded values, the effective
  change is: lj_liquid's gate floor (the averaged one) moves x1.12 (gr) /
  x1.02 (cn) — marginally looser; large's averaged floor moves x0.97 (gr) /
  x1.57 (cn) — the cn arm becomes honestly looser.

## 1. Protocol and parameter cross-check (every value cited to repo records)

| Parameter | Value used | Repo record |
|---|---|---|
| potential | single-species LJ 12-6, eps = sigma = m = 1 | `bench/reference/lj_liquid/provenance.json` (ase.calculators.lj.LennardJones rc 2.5, smooth=False) |
| cutoff | 2.5 sigma, energy-shifted, forces truncated | provenance `potential.parameters.truncation` == LAMMPS `pair_style lj/cut 2.5` + `pair_modify shift yes` (validated ka_convergence mapping, V1 energy check 3.5e-5/atom) |
| N | 500 (fcc 5x5x5) / 2048 (fcc 8x8x8) | provenance protocols; `generate_reference.py:lj_fcc` |
| rho* | 0.85 (L = 8.3788 / 13.4061 sigma, exact from a0 = (4/rho*)^(1/3)) | stored committed frames carry exactly these L (validated V0 bit-exact; the large-case provenance prose "L=13.51" is a typo — the frames say 13.4061) |
| T* | 0.72 | provenance `units.temperatures_K` 8355.256 K |
| ensemble / thermostat | NVT Langevin gamma* = 0.5/tau (LAMMPS damp 2.0, `zero yes`) | provenance `friction_per_tau` 0.5; ka_convergence input template |
| dt | 0.005 tau/step | provenance `units.dt_reduced` |
| equilibration | 2000 steps (10 tau) / 4000 steps (20 tau) | per-case committed `protocol.steps.equilibration` |
| sampling | 45 frames per trajectory at 1000-step (5 tau) stride — the committed stride, extended from 10/5 frames | per-case committed `protocol.steps.sampling_stride` |
| seeds | NEW: base 20261101 + 10·case + traj, sub-seeds ×10+{1 vel, 2 langevin} (velocity 202611011..202611132) | ka_params.py convention; collision-checked vs committed bench seeds 20260928/29 and all ka seeds 202610251..202610546 |
| start | identical fcc construction, Maxwell velocities + COM zeroed (`velocity create ... mom yes` == ASE thermalize) | `generate_reference.py:thermalize` |
| LAMMPS build | cluster module lammps/3Mar20-icc.2022 (the only working build), serial per job | stream brief |

Conventions validated BEFORE generation: `validate_conventions.py`
reproduces every committed floor statistic bit-exactly from the committed
frames (8/8 checks: fcc box, all 45 lj_liquid pairs, lag≥4 21-pair floor,
rotated 5v5 averaged floor, large 21-pair pooled floor, large 11-pair
averaged floor, P90 = np.percentile(·, 90), D15 = max(mean, P90) == the A5
effective floors quoted in acceptance_details.md).

Decorrelation lag: LAG_MIN = 4 frame indices = 20 tau, the committed
lj_liquid frame-pair convention, applied uniformly to both cases
(conservative vs the large case's committed lag ≥ 2). Checked, not assumed:
the measured pair-distance decay is flat from lag 1 on (lj_liquid gr_rms
0.088–0.092 over lags 1–8; large 0.043–0.044), and lag ≥ 2/3/4 give
identical D15 floors (gr 0.0543/0.0544/0.0545, cn 0.0405/0.0405/0.0400 for
large).

## 2. Runs

6 trajectories = 2 cases × 3 independent runs, each 2 000/4 000 equil +
45 × 1 000 steps (47k/49k steps). Submitted as one sbatch job per
trajectory, round-robin on the idle nodes 05–09 only (Z3/fills own 03–04):

| run | case | node | wall | seeds (vel/langevin) | equil T* | end PE/atom |
|---|---|---|---|---|---|---|
| lj500_t00 | lj_liquid | node05 | 10 s | 202611011/202611012 | 0.7092 | −5.2230 |
| lj500_t01 | lj_liquid | node06 | 10 s | 202611021/202611022 | 0.6972 | −5.2180 |
| lj500_t02 | lj_liquid | node07 | 10 s | 202611031/202611032 | 0.7356 | −5.1754 |
| lj2048_t00 | lj_liquid_large | node08 | 44 s | 202611111/202611112 | 0.7162 | −5.2329 |
| lj2048_t01 | lj_liquid_large | node09 | 44 s | 202611121/202611122 | 0.6925 | −5.2300 |
| lj2048_t02 | lj_liquid_large | node05 | 44 s | 202611131/202611132 | 0.7102 | −5.2201 |

Cluster usage: 6 jobs × 1 core, ≈ 3 core-minutes total (the runs are
trivially cheap; a local smoke run measured the same protocol at
47 000 steps in 8 s on the anaconda build). Reference-data sanity (AGENTS
rules, committed `ref_common.py` semantics — averaged-g(r) peak, not
per-frame): density 0.850 exactly; averaged first peak 1.0739–1.0966 sigma
(window 1.05–1.12); zero pairs below the 0.8-sigma hard core
(`sanity_new_frames.json`). Equil T* 0.693–0.736 vs bath 0.72.

Engine equivalence (the new-vs-committed cross distances sit BETWEEN the
two same-set ranges — one equilibrium distribution, not two):

| case | key | committed–committed (45) | new–new cross (5 043) | new–committed cross (1 230) |
|---|---|---|---|---|
| lj_liquid | gr_rms mean/q90 | 0.0928 / 0.1147 | 0.0877 / 0.1088 | 0.0900 / 0.1112 |
| lj_liquid | cn_tv mean/q90 | 0.0593 / 0.0880 | 0.0576 / 0.0860 | 0.0569 / 0.0820 |
| large | gr_rms mean/q90 | 0.0420 / 0.0520 | 0.0435 / 0.0545 | 0.0430 / 0.0542 |
| large | cn_tv mean/q90 | 0.0225 / 0.0301 | 0.0278 / 0.0400 | 0.0253 / 0.0361 |

Mean CN: committed 12.019, new 12.004 (large case).

## 3. The re-grounded floors (D15 = max(mean, P90); bootstrap 95% CI)

Pair sets (`liq_params.py` docstring, committed conventions resampled):
frame-pair floor = cross-trajectory pairs with both indices ≥ 4 (5 043)
POOLED with within-trajectory pairs at index lag ≥ 4 (2 583); averaged
floor = the committed group-size conventions resampled over independent
blocks — lj_liquid: rotated 5v5 half-splits of 12 disjoint 10-frame blocks
plus 12 position-matched cross-trajectory block pairs (132); large: per
5-frame window the committed 5 cyclic 2v3 rotations over 9 windows per
trajectory plus 27 position-matched 5v5 window pairs (162). Bootstrap:
resample pairs with replacement, 1 000 draws, seed 20261101 (the stream
brief's convention; the ka_convergence `bootstrap_ci` scheme at NBOOT 1000).

### lj_liquid

| floor | key | pairs | mean | P90 | D15 floor | bootstrap 95% CI | committed D15 (pairs) | ratio |
|---|---|---|---|---|---|---|---|---|
| frame-pair | gr_rms | 7 626 | 0.0884 | 0.1102 | **0.1102** | [0.1094, 0.1114] | 0.1204 (21) | x0.92 tighten |
| frame-pair | cn_tv | 7 626 | 0.0586 | 0.0860 | **0.0860** | [0.0860, 0.0880] | 0.0920 (21) | x0.93 tighten |
| averaged | gr_rms | 132 | 0.0392 | 0.0530 | **0.0530** | [0.0475, 0.0535] | 0.0472 (10) | x1.12 widen |
| averaged | cn_tv | 132 | 0.0283 | 0.0419 | **0.0419** | [0.0404, 0.0440] | 0.0412 (10) | x1.02 hold |

### lj_liquid_large

| floor | key | pairs | mean | P90 | D15 floor | bootstrap 95% CI | committed D15 (pairs) | ratio |
|---|---|---|---|---|---|---|---|---|
| frame-pair | gr_rms | 7 626 | 0.0435 | 0.0545 | **0.0545** | [0.0542, 0.0549] | 0.0528 (21) | x1.03 hold |
| frame-pair | cn_tv | 7 626 | 0.0279 | 0.0400 | **0.0400** | [0.0396, 0.0405] | 0.0293 (21) | x1.37 widen |
| averaged | gr_rms | 162 | 0.0263 | 0.0326 | **0.0326** | [0.0318, 0.0343] | 0.0335 (11) | x0.97 hold |
| averaged | cn_tv | 162 | 0.0179 | 0.0262 | **0.0262** | [0.0241, 0.0283] | 0.0167 (11) | x1.57 widen |

Diagnostics (`liquid_floors.json`): within-only vs cross-only D15 agree for
every floor (lj_liquid gr 0.1130/0.1088, cn 0.0920/0.0860; large gr
0.0545/0.0545, cn 0.0405/0.0400) — an equilibrated liquid carries no
preparation memory, confirming the committed large-case note (+1%/+10%)
with 200x the pairs. Trajectory-pair D15 spread (3 trajectory pairs, the
honest uncertainty the pair bootstrap under-represents because pairs share
frames): lj_liquid gr 0.1079–0.1094, cn 0.0840–0.0860; large gr
0.0540–0.0554, cn 0.0386–0.0410 — ±1–3%, far smaller than every committed
floor change reported above except the two x1.0 holds.

## 4. What changed and what it means

1. The X7 item's premise is confirmed quantitatively: with 10–21 pairs the
   P90 of a floor's pair distances IS roughly its maximum, in both
   directions — lj_liquid's frame-pair floor was ~9% too high (its P90 was
   the 3rd-largest of 21 pairs), while the large case's cn_tv floors were
   ~40–60% too LOW (21 pairs from 10 frames never saw the histogram-tail
   fluctuations that 135 frames sample: the committed 21-pair max 0.0347 is
   BELOW the new q90 0.0400).
2. `gr_rms` is a well-behaved statistic (all four floors within ±12%);
   `cn_tv` is the fragile one — its pair-distance distribution has a heavy
   relative tail at N = 2048 where the histogram is sharply peaked.
3. Nothing here requires changing the 1.5x gate factor; the floors it
   reads simply move to the values in the tables. The A5 effective floors
   (the averaged ones) become: lj_liquid gr 0.0530 / cn 0.0419; large
   gr 0.0326 / cn 0.0262.
4. NOT done here (out of scope by the stream brief): the molecular
   water/NaCl cases keep their own floors; the CNT electrolyte e0 floors
   are the Stage B / Z3 concern; cu_solid_liquid's 3-pair frame-pair floor
   is a solid-interface case, not a liquid case.

## 5. Artifacts and provenance

- Code: `research/x7_liquid/{liq_params,liq_obs,validate_conventions,
  gen_inputs,parse_runs,analyze_floors,thin_series}.py` — chaord-free (D23);
  conventions validated bit-exact (8/8) before generation.
- Thinned series back in the repo (只算不存; raw dumps deleted after
  thinning, locally and on the cluster):
  `research/x7_liquid/runs/<id>/` — data/in/log/summary + every 3rd frame
  as bench-convention npz (15 per trajectory, 15 tau apart);
  `research/x7_liquid/observables/<case>_obs.npz` — per-frame observables
  of ALL 270 frames, from which every pair distance and both floors
  reproduce bit-exactly (verified round-trip);
  `research/x7_liquid/thinning_manifest.json` — sha256 of all 122 kept
  artifacts; per-run `sha256.txt` additionally records the (now deleted)
  dump hashes.
- Cluster workspace: `/vol02/hdd1/ime001/chaord_x7/liquid_floors/`
  (inputs, logs, sbatch outputs; 0.9 MB after cleanup).
- Numbers: `reports/convergence/liquid_floors.json` (this report's twin).
