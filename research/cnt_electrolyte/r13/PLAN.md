# R13 reservoir-filling test — PRE-REGISTRATION (Review 13, T1–T3, D55)

Written 2026-11-05 by stream S3, BEFORE any R13 job was built, run or
submitted. Committed before any build (D59: the commit message quotes this
file). This file is the analysis contract: when the data exist, T3 runs
exactly the script in section 7 and reports the result either way, adding
no other test after seeing the data.

Scope: ONE pre-registered re-run of the Z3 design at the TRUE RESERVOIR
filling (D55 — the only filling for physics claims; Q3 is void per D54
because it used the uncorrected Q2 draft composition caf8b78). (16,16) only;
fields 0, 0.01, 0.02 V/A; 6 new builds; 98.4 A tubes; one LAMMPS build
(cluster 3 Mar 2020); the ratified 1 ns settle + 2 x 6 ns block protocol.

## 1. Composition — the corrected open-tube uptake, scaled to 98.4 A

Q2's final report (fc8bec2) measured the (16,16) open-tube uptake per A of
tube: EC 1.2701, EMC 0.5917, Li+ 0.2673, PF6- 0.2127. At Lz = 98.4 A
(40 periods, ncell = 40):

| species | per-A (Q2) | x 98.4 | build count |
| --- | --- | --- | --- |
| EC  | 1.2701 | 124.98 | **125** |
| EMC | 0.5917 | 58.22  | **58** |
| Li+ | 0.2673 | 26.30  | **24** (neutralized, see below) |
| PF6-| 0.2127 | 20.93  | **24** (neutralized, see below) |

The salt is neutralized at the ROUNDED MEAN of the measured in-tube ion
counts: (26.3 + 20.9)/2 = 23.6 -> 24 LiPF6 pairs (24 Li+ / 24 PF6-).

**Stated as a FINDING, not as the build (T1 order, verbatim)**: Q2 measured
an in-tube Li+/PF6- imbalance — 26.3 Li+ per 20.9 PF6- per 98.4 A (~+26%
Li+ excess over electroneutrality). The build CANNOT realize a net-charge
tube, so it uses the neutralized 24/24 pair count. The imbalance is a
physical result of the filling dynamics (ion-size-selective entry), reported
in the filling results (analysis/filling_results.json, fc8bec2), and the
neutralization is a stated modelling choice of this pre-registration — its
effect is not separable within this design and is listed as a limit.

Expected counts per build: tube 4 x 16 x 40 = 2560 C; EC 125 x 10 = 1250;
EMC 58 x 15 = 870; PF6- 24 x 7 = 168; Li+ 24. **Total 4872 atoms**;
bonds 125x10 + 58x14 + 24x6 = 2206.

## 2. Run list (18 jobs, fixed)

Job names `r13_e<fieldmV>_b<buildK>`:

| field (V/A) | jobs |
| --- | --- |
| 0 (null)    | r13_e0_b1  ... r13_e0_b6  |
| 0.01 (working field) | r13_e10_b1 ... r13_e10_b6 |
| 0.02        | r13_e20_b1 ... r13_e20_b6 |

- 6 NEW builds `r13_b1..b6` (pack-into-pore at the reservoir composition,
  98.4 A, seeds in section 5), each serving all three fields — the same
  build-paired structure as the committed Z3 set (each field's 6 runs come
  from the 6 builds; the three field runs of a build start from the
  IDENTICAL thermalized state).
- Welch n = 6 per group.

## 3. The D39 analysis call + D48 interval (verbatim contract)

D39 (open_items_v5.md), as run in Z3/Z4: detection of a field = a test of
the mean shift with independent runs as the unit — Welch's t, two-sided,
alpha = 0.05, zero-field run means (blocks A+B averaged) against field-on
block-B values. The D15 3x-floor rule is NOT part of this analysis.

Per-job analysis inputs: the 18 thinned production series, lifted per frame
(blind statement/direct P1 machinery, dialect naming dipole species ->
dipole_p1 measured unconditionally, D29) into `r13_results.json` with one
entry per job carrying `direct_p1_blocks` = {"A": ..., "B": ...} (block A =
[0, 6000) ps, block B = [6000, 12000] ps).

**Decision rule for 0.01 V/A, stated before any R13 data exist: 0.01 V/A
is CONFIRMED AT THE RESERVOIR FILLING iff its Welch p < 0.05 on this set
alone (two-sided, alpha = 0.05, n = 6 per group). No other test is added
after seeing the data.** 0.02 V/A is computed by the same call as a
consistency point. Per D48, EVERY reported comparison carries its 95%
confidence interval on the shift (Welch–Satterthwaite df). The build-paired
contrast is reported as secondary, never as the gate.

## 4. Protocol per job (identical to the committed Z3 protocol)

Stage 1: 100 ps ramp 10 -> 300 K zero field (Langevin). Stage 2: 900 ps
equilibration 300 K zero field -> ladder t = 0 state, identical for the
build's three field jobs. Stage 3: field ON (0 for e0), 1 ns settle, dump
every 10 ps (equil.dump). Stage 4: 2 x 6 ns production blocks, dump every
1 ps (prod.dump, reset_timestep 0), write_data final.data nocoeff. 14 ns
simulated per job; the D39 analysis uses ONLY the 12 ns of blocks.

Thermostat: ONE Langevin scheme, damping 1 ps — `fix rigid/nve molecule
langevin` (NON-distributed variant, per f9cae3f) + `fix nve` + `fix
langevin` (Li+); no Nose-Hoover. dt = 1 fs; lj/cut/coul/long 12.0, pppm
1e-4, arithmetic mixing, exclude molecule/intra. No fix ave/time anywhere
(the 3 Mar 2020 ave/time + reset_timestep defect, e1e4807). Cluster
runs: sbatch, 8 ranks per job, owner cap 28 ranks per task respected by
batching.

Cross-build chaining check (post-run): the three field jobs of a build must
show IDENTICAL thermo at end of stage 2 (step 1000000); a mismatch is
reported, not absorbed.

## 5. Packing at reservoir density, and seeds

The reservoir composition is ~1.7x the pilot packing density per pore
volume; the pilot packer jams near ~55-60% of bulk density by pure
rejection. Per the T1 order: **if the packer jams, pack in STAGES — insert
at reduced LJ radius, then relax to the floor — and record every attempt;
the counts are NEVER lowered.** Every jam event and attempt count goes in
the build manifest.

Seeds (new ranges, disjoint from every committed family: 20261001-04,
20262100-08/3100-08, 20264101-5101702, 21267011-22067032, 20273001-20276062,
20281101-20281606):

| purpose | range |
| --- | --- |
| build (packer) seeds, r13_b1..b6 | 20291101 - 20291106 |
| velocity-create seeds (per build K) | 20291151 - 20291156 |
| thermalization Langevin pairs (per build K: ramp 20291200+20K+1/+2, equil +3/+4) | 20291221 - 20291324 |
| ladder Langevin pairs (per job: 20291400 + field_mv*100 + 10*K, +1 bodies / +2 li) | 20291411 - 20292062 |

All < 2^31. Velocity-create and thermalization seeds are per-BUILD (shared
by the build's three field jobs); the ladder pair is per-JOB (18 distinct).

## 6. Done-when (T2)

Every job's final.data + log.lammps + a thinned-series manifest are in git;
the thinned series (1/12 -> 12 ps cadence, as Q3/Z3) are in the local
analysis archive with sha256 pinning; raw dumps wiped on the cluster after
thinning (owner policy: compute on cluster, retrieve essential evidence
only).

## 7. The one analysis script (T3 runs exactly this, nothing added)

```python
import json, numpy as np
from scipy import stats
J = json.load(open(
    "research/cnt_electrolyte/r13/r13_results.json"))["jobs"]
off = np.array([np.mean([j["direct_p1_blocks"]["A"], j["direct_p1_blocks"]["B"]])
                for j in J.values() if j["field"] == 0])          # n = 6
for E in (0.01, 0.02):
    on = np.array([j["direct_p1_blocks"]["B"] for j in J.values()
                   if j["field"] == E])                            # n = 6
    r = stats.ttest_ind(on, off, equal_var=False)
    # D48: 95% CI on the shift, Welch-Satterthwaite
    se = np.sqrt(on.var(ddof=1)/len(on) + off.var(ddof=1)/len(off))
    df = r.df
    tc = stats.t.ppf(0.975, df)
    d = on.mean() - off.mean()
    print(f"(16,16) reservoir {E}: shift {d:+.4f}  t {r.statistic:4.2f}  "
          f"p {r.pvalue:.4f}  CI95 [{d-tc*se:+.4f}, {d+tc*se:+.4f}]")
```

T3 writes `r13_results.json` from this output and adds go_no_go.md section
15 beside Z4's pilot-filling result, whatever the outcome.
