# Z3 confirmatory set — PRE-REGISTRATION (Review 11, D41)

Written 2026-10-31 by stream S3, BEFORE any confirmatory job was built, run or
submitted. This file is the committed analysis contract: per D41 and the Z3 work
order, the run list and the D39 analysis call below are fixed now; when the data
exist, Z4 runs exactly this analysis and reports the result either way, adding
no other test after seeing the data.

Scope: ONE confirmatory set for the 0.01 V/A decision (D40 leaves it open until
Z3+Z4). (16,16) only; fields 0, 0.01, 0.02 V/A; 6 new builds; 98.4 A tubes; one
LAMMPS build; the ratified 1 ns settle + 2 x 6 ns block protocol.

## 1. Run list (18 jobs, fixed)

Job names `sb16L_e<fieldmV>_b<buildK>` (sb = stage-B-ladder convention,
16L = (16,16) long tube, bK = build K, K = 1..6):

| field (V/A) | jobs |
| --- | --- |
| 0 (null)    | sb16L_e0_b1  ... sb16L_e0_b6  |
| 0.01 (working field) | sb16L_e10_b1 ... sb16L_e10_b6 |
| 0.02        | sb16L_e20_b1 ... sb16L_e20_b6 |

- 6 NEW builds `cf16_b1..b6` (pack-into-pore, 98.4 A, seeds in section 5),
  each serving all three fields (run bK at e0/e10/e20) — the same
  build-paired structure as the ratified Stage B ladder (rK <- build bK for
  every field). D41's "6 new builds per field" is implemented as this one
  6-build set: each field's 6 runs come from the 6 new builds. The three
  field runs of a build start from the IDENTICAL thermalized state (section
  3), so the build-paired contrast is available as a secondary view; the
  D39 primary test (section 2) needs no pairing.
- Welch n = 6 per group (e0 run means vs each field's block-B values).

## 2. The D39 analysis call (verbatim, restricted to this dataset)

D39 (open_items_v5.md): detection of a field = a test of the mean shift with
independent runs as the unit — Welch's t, two-sided, alpha = 0.05, zero-field
run means (blocks A and B averaged) against field-on block-B values. The D15
3x-floor rule is NOT part of this analysis (it stays only for per-block
classification claims, which this set does not make).

Per-job analysis inputs: the 18 thinned production series (section 7), lifted
per frame (same blind statement/direct P1 machinery as the committed Y8
pipeline) into `confirmatory_results.json` with one entry per job carrying
`direct_p1_blocks` = {"A": block-A mean P1, "B": block-B mean P1} (block A =
production t in [0, 6000) ps, block B = [6000, 12000] ps; identical split to
Y8's full-ladder pass). The analysis script to run (Z4, exactly this, nothing
added):

```python
import json, numpy as np
from scipy import stats
J = json.load(open(
    "research/cnt_electrolyte/confirmatory/confirmatory_results.json"))["jobs"]
off = np.array([np.mean([j["direct_p1_blocks"]["A"], j["direct_p1_blocks"]["B"]])
                for j in J.values() if j["field"] == 0])          # n = 6
for E in (0.01, 0.02):
    on = np.array([j["direct_p1_blocks"]["B"] for j in J.values()
                   if j["field"] == E])                            # n = 6
    r = stats.ttest_ind(on, off, equal_var=False)
    print(f"(16,16) {E}: shift {on.mean() - off.mean():+.3f}  "
          f"t {r.statistic:4.1f}  p {r.pvalue:.4f}")
```

**Decision rule for 0.01 V/A, stated before any confirmatory data exist:
0.01 V/A is CONFIRMED iff its Welch p < 0.05 on the confirmatory set alone
(two-sided, alpha = 0.05). No other test is added after seeing the data**
(Z4 work order; D39). 0.02 V/A is computed by the same call as a consistency
point (Stage B detected it at p = 0.0109 with n = 3; D40's E_min for (16,16)
is 0.02 V/A). The build-paired contrast (per build: field-on block B minus
that build's e0 all-block mean) is reported as secondary, never as the gate.

## 3. Protocol per job (13 ns ladder protocol; 14 ns simulated)

Start states are FRESH packed builds (section 4), not Stage A zf finals: the
one structural difference from Stage B is recorded here, in advance. Stage B's
starts carried Stage A's 100 ps ramp + 900 ps zero-field equilibration
upstream; the confirmatory jobs carry the IDENTICAL 1 ns zero-field
thermalization in-job (stages 1-2 below, per-build seeds shared by the build's
three field jobs), then the Stage B ladder protocol runs verbatim:

```
stage 1   100 ps ramp 10 -> 300 K, zero field, Langevin   (per-build seeds)
stage 2   900 ps equilibration at 300 K, zero field       (per-build seeds)
          -> ladder t = 0 state: identical for the build's e0/e10/e20 jobs
stage 3   field ON at t = 0 (stays 0 for e0), 1 ns settle, dump every 10 ps
          (equil.dump)                                     (per-job seed pair)
stage 4   2 x 6 ns production blocks, dump every 1 ps %.10g + image flags
          (prod.dump, one continuous trajectory, reset_timestep 0;
          block A = [0, 6000) ps, block B = [6000, 12000] ps)
          write_data final.data nocoeff                    (per-job seed pair)
```

13 ns of ladder protocol (1 ns settle + 2 x 6 ns blocks) exactly as ratified
(D41, stage_b_plan.md section 2); +1 ns upstream thermalization = 14 ns
simulated per job. The D39 analysis uses ONLY the 12 ns of blocks (the settle
and thermalization are discarded, as in Stage B). Block length 6 ns = 2 x
tau_M(cnt16 @0.01 V/A) = 2 x 3.01 ns, measured in Stage A.

Thermostat: ONE Langevin scheme for every mobile atom, damping 1 ps —
`fix rigid/nve molecule langevin` (multi-atom bodies, NON-distributed variant
per f9cae3f) + `fix nve` + `fix langevin` (Li+); no Nose-Hoover anywhere.
dt = 1 fs; `thermo 1000`, `thermo_modify flush yes`; Tmob =
2*(KE_bodies+KE_Li)/(DOF*kB) with DOF = 837 (section 6). Neighbor/coulomb
setup identical to every Stage A/B job (lj/cut/coul/long 12.0, pppm 1e-4,
arithmetic mixing, exclude molecule/intra).

Cross-build chaining check (post-run, honest): on one LAMMPS build the
in-job thermalization is deterministic, so the three field jobs of a build
must show IDENTICAL thermo (Tmob, PotEng, step count) at the end of stage 2
(step 1000000) — verified from the logs after download; a mismatch means a
non-deterministic path and is reported, not silently absorbed.

## 4. Filling source (recorded decision)

**The PILOT FILLING is used: pack-into-pore via build_systems.py's Packer**
(the same construction as every Stage A/B build), with the composition
recomputed for the 98.4 A tube (section 6). Two independent reasons, both
verified 2026-10-31:

1. **Geometry**: the Y6 reservoir fills are PILOT-LENGTH systems —
   `stageA/jobs/fill_cnt16/system.data` (the v2 staged-decompression build)
   is a 49.2 A tube spanning z in [30, 79.2] of a 109.2 A box. A fill final
   is a different box, tube length and pore content from D41's 98.4 A
   confirmatory tubes; its solvent state cannot be transferred to the longer
   tube. Reservoir-filling the 98.4 A geometry would be a NEW Y6-scale
   development (4116-atom pore + 2 x 30 A reservoirs + lateral shell,
   ~40k+ atoms), not a reuse of Y6 — out of scope for Z3 as ordered.
2. **Availability**: fill_cnt16 is STILL RUNNING (squeue: job 233236 `fill_cnt`
   R 22:13:36 on node04; no final.data yet, prod.dump 5.07 GB partial,
   checked 2026-10-31). fill_cnt10 IS done
   (/vol02/hdd1/ime001/chaord_fills_v2/fill_cnt10/: final.data + prod.dump +
   prod_thinned.dump + THIN_DONE) but it is a (10,10) tube — the confirmatory
   set is (16,16) only, so it cannot serve either.

This is the fallback D41 itself names ("else the pilot filling stated in the
plan"); it is stated here before any job starts. Consequence, recorded: the
confirmatory pore content is the packed 1.10 M pore (no external reservoirs),
same physical system class as every Stage A/B run the D39 ladder was ratified
on; only the tube is twice as long.

## 5. Seeds (new, documented, disjoint from everything used)

All seeds < 2^31. One rng stream per build (numpy default_rng(build_seed)).

| purpose | range |
| --- | --- |
| build (packer) seeds, cf16_b1..b6 | 20273001 - 20273006 |
| velocity-create seeds (stage 1, per build K) | 20273501 - 20273506 |
| thermalization Langevin pairs (per build K: ramp 20273600+20K+1/+2, equil +3/+4) | 20273621 - 20273724 |
| ladder Langevin pairs (per job: 20274000 + field_mv*100 + 10*K, +1 bodies / +2 li) | 20274011 - 20276062 |

Disjointness (every seed family already committed in this study, grepped from
all stageA/stageB in.lammps files): pilot build 20261001-04 / velocity
730101-04; Stage A builds 20262100-05, velocities 20263100-08, thermostats
3100101-3108402; fill block seeds 20264101-02, velocity create 20265100-01,
thermostats 5100101-5101702; Stage B thermostats 21267011-22067032
(base 20267000 + tube*100000 + field_mv*1000 + run*10). The confirmatory
ranges (20273001-20276062) intersect none of these. The velocity-create and
thermalization seeds are per-BUILD (shared by the build's three field jobs —
this is what makes their ladder-t=0 states identical, as in Stage B); the
ladder seed pair is per-JOB (18 distinct pairs — this is what makes the three
field realizations of a build diverge, as in Stage B).

## 6. Systems: 98.4 A tubes, composition recomputed for the same molarity

Builder: `build_systems.py`'s builder, reused unmodified with ncell = 40
(the builder takes (n, ncell) and builds Lz = ncell * a = 40 x 2.46 = 98.4 A;
arbitrary length is supported by construction). Tube radius R = 10.8502 A
(armchair (16,16), a = 2.46 A [Saito98]); vdW pore radius R - 3.2 = 7.6502 A.

Composition: the pilot (16,16) pore (49.2 A, V_vdw = 9.046 nm3, committed
build_manifest.json) holds 37 EC / 24 EMC / 6 LiPF6 = 1.101 M, 1.224 g/cm3.
Doubling the tube doubles the vdW pore volume (18.092 nm3), so the SAME
molarity and density need exactly twice the counts:

| | pilot ncell=20 | confirmatory ncell=40 |
| --- | --- | --- |
| EC / EMC / LiPF6 | 37 / 24 / 6 | **74 / 48 / 12** |
| V_pore_vdw | 9.046 nm3 | 18.092 nm3 |
| molarity (vdW volume) | 1.101 M | **1.101 M** |
| density | 1.224 g/cm3 | **1.224 g/cm3** |
| atoms (tube/liquid) | 2058 (1280/778) | **4116 (2560/1556)** |
| mobile DOF (Tmob) | 417 | **837** (6*(74+48+12) + 3*12 - 3) |

Same EC:EMC volume ratio (37:24 = 74:48), same packing rules and strict
geometric guard (`validate_data_file`, f9cae3f hardening) on every build.

## 7. LAMMPS build identity + cluster plan (只算不存)

- **Production build (all 18 jobs, D41's one-LAMMPS-build rule)**: the
  ime001 cluster's module `lammps/3Mar20-icc.2022` — binary reports
  `LAMMPS (3 Mar 2020)`, launched `mpirun lmp_mpi` (Intel MPI 2021.6.0),
  8 ranks per job (`#SBATCH -n 8 -N 1`). This is the same module the 33
  cluster legs of the ratified Stage B ladder ran on.
- **Smoke builds (validation only, no confirmatory data)**: local
  `C:/ProgramData/anaconda3/Scripts/lmp.exe` (22 Jul 2025 Update 4, serial);
  every input passes `lmp -var smoke 1 -in in.lammps` (500-step stages)
  before upload. Smoke results are recorded per job in
  `smoke/SMOKE_RESULTS.md`; smokes produce no kept output.
- Jobs live on the cluster at `/vol02/hdd1/ime001/chaord_confirmatory/jobs/<name>/`
  ($HOME = /vol02/hdd1/ime001 — on shared storage; NOT /dev/shm: compute
  nodes cannot see shm, and sbatch -o paths must be on /vol02).
- sbatch (per job, from the fills/Stage B template):
  `-p cpu -N 1 -n 8 -t 48:00:00 -o <jobdir>/%j.out`, module load line,
  `cd <jobdir>`, `mpirun lmp_mpi -in in.lammps`, then on success the thin
  step (below) and raw-dump deletion. Cost estimate from Stage B's measured
  cluster throughput (cnt16: 13 ns x 2058 atoms in ~3h08m on 8 ranks):
  ~6.8 h per job (14 ns x 4116 atoms) — far inside 48 h.
- **Submission in 2 waves of 9** (disk-bound, recorded in advance): a raw
  1-ps prod.dump is ~2.4 GB per job (4116 atoms x ~48.4 B/atom/frame x
  12,001 frames; measured from Stage B's cnt16 thinned files), and /vol02
  has 53 GB free (2026-10-31). 18 concurrent raw dumps = ~43 GB — too close
  to the ceiling to be safe on shared storage. 2 waves of 9 cap the raw
  footprint at ~22 GB; each job thins and deletes its raw dump at
  completion, so wave 2 overlaps only thinned residue (~0.2 GB/job).
  CPU is not the constraint (112 cores/node, 7 idle nodes on 2026-10-31);
  the owner's "cores used" intent is honored at 72 concurrent ranks per
  wave. If >= 50 GB is free at submission time, both waves may go at once —
  the decision is recorded in the queue log, not improvised.
- squeue verification after each wave (all 18 in R state before leaving).

## 8. Extraction (只算不存) and manifests

Per job, after a clean exit (final.data exists):

1. `thin_prod.py prod.dump` (cluster-side, stdlib-only, deterministic):
   keep EVERY 12th production frame (timestep // 1000) % 12 == 0 → 12 ps
   spacing, 1001 frames (0 .. 12,000,000), `%.10g` rows preserved verbatim,
   identical cadence to the ratified Stage B record
   (`prod_thinned.dump`, ~200 MB); writes `prod_thinned.sha256`
   (bytes + sha256, same schema as stageB's prod_series_manifest.json) and a
   `THIN_DONE` marker; then prod.dump is DELETED on the cluster.
2. What comes back per job (`confirmatory/cluster_results/<name>/`, mirroring
   stageB/cluster_results): `final.data`, `log.lammps`, `equil.dump` (the
   1-ns settle, 100 frames — the switch-on curve), `prod_thinned.dump`,
   `prod_thinned.sha256`, `sbatch.sh`, `%j.out`, `in.lammps`, `system.data`.
   Raw 1-ps dumps NEVER leave the cluster and die there (只算不存, owner
   policy — same as the 33 Stage B cluster legs).
3. sha256 manifests: `confirmatory/manifests/<name>.json` written at
   preparation time (in.lammps, system.data, sbatch.sh, seeds, protocol
   fields) and completed at download time (final.data, log.lammps,
   prod_thinned.dump, %j.out, wall time, LAMMPS identity string from the
   log header).
4. **Completion criterion (monitoring)**: a job is DONE iff its cluster job
   dir contains `final.data` AND `THIN_DONE` (checked via ssh/squeue; see
   MONITORING.md). All 18 DONE + manifests = Z3's "Done when".

## 9. Honest-record notes (stated before data)

- The build-paired secondary contrast is possible (builds shared across
  fields) and will be reported as secondary, per D39; the ordered gate is the
  Welch call of section 2, unconditionally.
- The in-job thermalization (section 3) is the one structural addition vs
  Stage B, forced by fresh builds; it is zero-field, seed-fixed, excluded
  from the analysis window, and identical across a build's three field jobs.
- n = 6 per group; the Welch test treats the 6 runs of a field as
  independent realizations (distinct chaotic trajectories under distinct
  Langevin seeds), the same unit D39 ratified on Stage B's n = 3.
- 0.005 V/A is NOT in this set (D41 fixes the fields); no interpolation or
  extra field is added later.
- The confirmatory set shares the pilot model (owner_inputs.md D37 note);
  the owner-model question (D46) is separate and does not gate Z3.
