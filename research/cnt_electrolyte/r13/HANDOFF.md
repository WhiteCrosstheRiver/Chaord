# R13 handoff note (2026-11-06, written for the next agent/owner)

## Where everything lives

- **Local repo (all work happens here, nowhere else):**
  `C:\Users\ZemengFeng\Desktop\codes\Chaord` (git, branch master, pushed to
  https://github.com/WhiteCrosstheRiver/Chaord.git via proxy 127.0.0.1:7897
  — pushes fail intermittently with 502/schannel, retry loop works).
- **Cluster (compute only, 只算不存):** `ssh ime001@172.28.191.241` (alias in
  `~/.ssh/config`), slurm, 9 nodes x 112 cores, cap 28 ranks/task, 8 ranks
  per LAMMPS job. Working dirs on shared storage
  `/vol02/hdd1/ime001/`:
  - `chaord_q3/` — Q3 evidence (already retrieved + committed; raw dumps wiped)
  - `chaord_fills_v2/` — the two Y6 fills (already retrieved + committed)
  - `chaord_confirmatory/` — Z3 evidence (already committed)
  - `chaord_r13/` — **TO BE CREATED** by `upload_r13.sh` (T2 step 1)
- Module: `module load lammps/3Mar20-icc.2022` (the default lmp_mpi lacks
  RIGID/KSPACE/MOLECULE). Compute nodes cannot see /dev/shm — everything on
  /vol02.

## Governing documents (read in this order)

1. `AGENTS.md` rule 0 → 2. `docs/reviews/open_items_v7.md` (D54-D59, T1-T7)
3. `research/cnt_electrolyte/r13/PLAN.md` (the T1 pre-registration, 9e305de,
   committed BEFORE any build; never edit it)
4. `docs/records/corrections.md` (Q3 void entry, f7926c4)

## State of T1-T3 at handoff

- T1 DONE: PLAN committed (9e305de). T5a outline committed (01d4f3b).
- T2 BLOCKED AT THE FIRST BUILD — honest record:
  `r13_build.py r13_b1` ran the full staged-floor ladder
  (1.0/0.9/0.85/0.8/0.75/0.7): scales 1.0-0.85 jam on an EC insertion;
  scales 0.8-0.7 insert everything but the geometric push-apart
  `relax()` cannot clear the TRUE floors (translation-only rigid-body
  relaxation is not enough at ~1.5x the pilot packing density).
  Counts were NOT lowered (T1 policy). Nothing fabricated; the attempt log
  is in the r13_b1 run output and should go into the build manifest.
- Next-step options for T2 (pick ONE, record the choice in the build
  manifest; do NOT change PLAN.md's composition or decision rule):
  a. Pack at a reduced floor (e.g. 0.8) WITHOUT strict verify, and let the
     in-job 1 ns NVT thermalization do the physical relaxation (requires a
     smoke test that LAMMPS tolerates the initial overlaps — start with a
     500-step smoke on one build);
  b. Slower/two-phase geometric relaxation (smaller RELAX_RATE, more
     sweeps, or insert-then-relax per molecule subset);
  c. Multi-start: more build seeds at scale 0.85 (some seeds may thread).
- After builds succeed: `python r13_inputs.py` stages 18 job dirs, then
  `make_sbatch.py` + `bash upload_r13.sh`, then `./submit.sh wave1`, later
  `wave2` (each job ~15-16 h on 8 ranks; est. wall ~32 h for both waves).
- T3: run the PLAN §7 script on `r13_results.json`; add go_no_go.md §15.

## Rules that bite (do not re-learn them)

- One agent per work order (D59); every reported number comes from one
  committed results file and the commit message quotes it.
- `tools/check_commit_claims.py` runs in CI: a commit message may only name
  files the commit contains.
- Batch compute goes on the CLUSTER, not the local box (owner directive).
- Raw dumps are thinned 1/12 on the cluster and deleted there; only
  final.data + log.lammps + manifest are retrieved.
- Local LAMMPS for smoke tests: conda 22Jul2025 build; use
  `fix rigid/nve molecule` (the /small variant corrupts stacks there too —
  see f9cae3f), no fix ave/time after reset_timestep.

## Update, 2026-11-06: T2 unblocked, production running

- Route: r13_staged.py (insert at a reduced floor, rigid-body soft relaxation in
  LAMMPS, unmodified strict gate). All six builds pass; see data/build_manifest.json.
  Builds 1 and 6 used floor scale 0.80; builds 2-5 jammed at 0.80 and used 0.75.
- Cluster working directory is now /vol02/hdd1/ime001/zmfeng/chaord_r13/code/r13
  (group rule: jobs under zmfeng, compute only, no persistent storage). The older
  path above is superseded. Clear this directory after retrieval.
- 18 jobs submitted from jobs/: wave 1 234324-234332, wave 2 234333-234341
  (wave 2 held by afterany on wave 1). About 10 h per job at 8 ranks.
- Seeds. r13_inputs.py had velocity seeds off by one against PLAN.md section 5
  and was fixed before submission. Ladder seeds follow PLAN's formula and reach
  20293462, but PLAN's range table ends at 20292062: PLAN contradicts itself; the
  formula was kept. Relaxation seeds 20293101-20293146 are not in PLAN.
- After the runs: check the chaining rule (identical thermo at step 1000000 per
  build), retrieve final.data, log.lammps, thinned series and manifests, run T3.
- Trap: module load lammps puts Intel Python 3.9 first on PATH and sbatch exports
  the environment. Use the absolute anaconda python path.

## Update 2026-11-07 (later): T2 done, T3 done

- All 18 production jobs COMPLETED (exit 0); series retrieved and verified
  18/18 against the cluster's sha256 records (`runs_manifest.json` pins all
  198 retrieved files), then `zmfeng/chaord_r13` was deleted on ime001.
- Chaining check: b1-b4 identical; b5, b6 diverge at e20 -- exactly the two
  jobs that ran on node09 (all others node07). Reported, not absorbed.
- T3 (`r13_analysis.py`, Z4 machinery unchanged, PLAN section 7 run
  verbatim): **0.01 V/A NOT CONFIRMED at the reservoir filling (p 0.5935)**;
  0.02 V/A p 0.0540. Written up in `reports/research/go_no_go.md` section 15.
- Next: T5c (field section of the paper) from section 15.
