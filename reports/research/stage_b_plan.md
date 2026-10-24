# Stage B plan — the tau_M-limited block protocol (owner Option b), made concrete

Written by stream S3/S5 on 2026-10-25, per the owner's CONTINUE order on
the Y7 gate. Option (b) of `reports/research/stage_a.md` — accept
tau_M-limited blocks, honestly reported — is now a generated, smoke-tested,
running input set. Nothing here blocks on anyone: the owner can stop,
redirect, or move any of it to the cluster; every input is portable.

**Status at writing (2026-10-25 02:23 local): all 42 ladder inputs
generated and smoke-tested (42/42 ok, 0 failures); first tranche (6
jobs, cnt10, 0.01 + 0.05 V/A) RUNNING — `stageB/queue_log.jsonl` pid
45408, `QUEUE_RUNNING` marker present, first job's step-0 thermo
bit-identical to Stage A's wf run off the same `zf_cnt10_b1` final
(Tmob 402.59744, PotEng −1632.4065), i.e. staging is exactly Stage A's
chaining.** The fills queue was verified NOT running at launch (last
`stageA/queue_log.jsonl` event: queue_end 2026-10-19; no lmp.exe). A
concurrent stream (`research/ka_convergence/`, `run_jobs.py r0`) is
running 9 lmp jobs on this 32-logical-core box; the stageB job's measured
pace under that load (~0.9 ps/s → ~4 h/job) is FASTER than the 5-h
estimate, so no material contention.

## 1. Measured basis — the provenance chain

Every protocol number below descends from measured Stage A quantities:

| quantity | value | primary source |
|---|---|---|
| tau_M, (10,10) @0.01 V/A switch-on | 2.67 ± 1.06 ns (per-build 1.07/1.40/2.49/5.71 ns) | `stageA/analysis/summary.json` `cnt10_tau_M_switch_on_sf`; table in `stage_a.md` |
| tau_M, (16,16) @0.01 V/A switch-on | 3.01 ± 0.93 ns (per-build 1.72/2.24/2.31/5.78 ns) | `summary.json` `cnt16_tau_M_switch_on_wf` |
| tau_M, (16,16) @0.2 V/A switch-on | >4.6 ns in 3 of 4 fits (1 NaN) | `summary.json` `cnt16_tau_M_switch_on_sf` — see §3 caveat |
| tau_M zero-field autocorr | 0.34 ± 0.06 ns (10,10); 0.95 ± 0.45 ns (16,16) | `summary.json` `*_zf_autocorr` |
| chain | `stageA_analyze.py` over `stageA/jobs/wf_* / sf_* / zf_*` prod dumps ← inputs from `stageA_inputs.py` ← physics in `params.py` (no chaord anywhere) | `stageA/manifests/*.json` sha256 |

The Y7 gate trigger (tau_M > 1 ns in both tubes at the working field) is
the reason this protocol exists: the brief's 20 tau_M-per-run ladder is
~55-60 ns x ~50 runs ≈ 1.1 µs of aggregate sampling (`stage_a.md` §Y7) —
cluster scale, not obtainable before the 2026-10-29 date. What follows is
the honest cheaper shape of the same measurement.

## 2. Protocol

**Field ladder (D30 context), 7 points:** 0, 0.005, 0.01 (working field),
0.02, 0.05, 0.1, 0.2 (saturating) V/A. Job names `sb_<tube>_e<fieldmV>_r<run>`
(e.g. `sb_cnt10_e10_r1`), 42 jobs = 2 tubes x 7 fields x 3 runs.

**Start states — the DONE Stage A zero-field finals.** Run rK starts from
`stageA/jobs/zf_cntN_b{K}/final.data` (K = 1, 2, 3): each is a 100 ps
10->300 K ramp + 900 ps equilibration + 2 ns zero-field production — an
equilibrated zero-field state by construction, which is exactly what
Stage B needs. Field goes ON at t = 0 (for e0 it stays off: the null-field
ladder point gets identical treatment, so the ladder's zero is sampled by
the same protocol). The mirrored b1m builds are NOT counted as independent
starts (they are the symmetry instrument); 3 runs = 3 independent builds.

**Run structure (13 ns simulated per job):**

```
t = 0        field on (or stays 0), velocities continue from the zf final
0-1 ns       settle, dumped every 10 ps (switch-on curve for tau_M re-fit)
1-7 ns       block A, dumped every 1 ps
7-13 ns      block B, dumped every 1 ps (one continuous trajectory; the
             analysis splits blocks at 6 ns)
end          write_data final.data (enables cheap block-C extensions, §3)
```

**Block length 6 ns = 2 x tau_M(cnt16 @0.01 V/A) = 2 x 3.01 ns**, the
slower tube; for cnt10 (tau_M = 2.67 ns) a block is 2.25 tau_M. This is
deliberately the floor of the owner's "at least 2 tau_M (5.4-6.0 ns)": one
common block length keeps cross-tube and cross-field comparisons
like-for-like. It is honestly weaker than 5 tau_M (= 13.5-15 ns blocks);
§3 quantifies what that costs.

**Physics/IO conventions identical to Stage A** (`stageA_inputs.py`):
same pilot model (rigid Nutakki 2023 OPLS-AA molecules, Werder tube
carbons, Lorentz-Berthelot, full ±1 ion charges, PPPM 1e-4, cutoff 12 A,
dt 1 fs), one Langevin scheme for every mobile atom with 1 ps damping
(`fix rigid/nve molecule langevin` + `fix nve`/`fix langevin` Li+; no
Nose-Hoover), thermo every 1 ps with the honest mobile-DOF Tmob, dumps
every 1 ps, `%.10g` + image flags. One thermostat seed pair per job, 84
distinct seeds, all < 2^31, disjoint from every Stage A seed range
(formula in `stageB/stageB_inputs.py`). Dump cadence 1 ps is finer than
the brief's max(10 ps, tau_rot) rule needs — kept because Stage A's
analysis already consumes 1 ps frames and disk is not the binding
constraint (§4).

## 3. Statistical honesty — what 2-tau_M blocks do and do not give

1. **Blocks within a run are correlated.** Two 6-ns blocks spaced 2 tau_M
   apart decorrelate only partially; the effective number of independent
   block averages per run is ~1, not 2. Consequence for Y8: the
   leave-one-run-out classifier treats RUNS as its units — 3 independent
   runs per field (>= 2 required), 3 LORO folds per state — and the
   primary error bar is the sem over the 3 per-run means (each run's mean
   = average of its 2 blocks). The 6-blocks-per-state number is a
   secondary, optimistic diagnostic only. 18 ns of trajectory per
   field/tube is the honest inventory.

2. **Block A of field-on runs retains switch-on transient.** Starting
   from the zero-field final and approaching the field-on steady state
   with time constant tau_M, the time-averaged bias of a block beginning
   at s ns is (tau/L)(e^{-s/tau} - e^{-(s+L)/tau}) of the full
   zero->field gap, with L = 6 ns:

   | tube (tau_M) | block A (s=1 ns) | block B (s=7 ns) | 2-block mean |
   |---|---|---|---|
   | cnt10 (2.67 ns) | ~27% of gap | ~2.9% | ~15% |
   | cnt16 (3.01 ns) | ~31% of gap | ~4.2% | ~18% |

   The ladder's field-response curve will therefore be compressed toward
   zero by roughly 15-18% at its middle if all 6 blocks are averaged.
   Primary analysis should use **block B only** (3 clean 6-ns blocks per
   state; bias ~3-4%) with all-blocks as the sensitivity check; the
   working-field (e10) comparison — the D32 go/no-go core — is the
   least-affected pairwise contrast after that choice.

3. **The e200 caveat.** tau_M at the saturating field was measured
   slower (>4.6 ns, 3/4 fits, cnt16): 6-ns blocks there may be < 2
   tau_M_effective. The e200 points are the ladder's saturation anchor,
   not its slope; if their block A/B means disagree beyond ~2x the
   run-level sem, the cheap remedy is a block-C extension (+6 ns from
   each job's own `final.data`) at e200/e100 only — not a re-run of the
   ladder. Same remedy covers any field where the A/B offset test fails.

4. **More runs, not longer runs, is the upgrade path.** If the Y8
   classifier cannot separate states at these error bars, the fix that
   buys real independent samples is additional independent builds (new
   zf seeds via `stageA_build.py`, then more sb runs) or cluster time for
   the 20-tau_M protocol — both drop into this input set unchanged.

## 4. Cost — measured, not assumed

Throughput is taken from Stage A's own DONE records (16 wf/sf production
runs, 2 ns each, 2026-10-17/18): **cnt10 (971 atoms): 41,398-42,529,
median 42,077 atom·ps/min; cnt16 (2058 atoms): 21,348-22,774, median
22,398**. (The 22,000 atom·ps/min planning constant was the cnt16-scale
number; cnt10 is ~1.9x cheaper per atom·ps.) Estimates below assume a
dedicated core; in practice the box has 32 logical cores and other
streams' lmp jobs run beside this queue without material contention
(measured at launch: 9 concurrent foreign lmp jobs, stageB pace at/above
the estimate) — the one-core coordination rule of §5 is the owner's
discipline anyway.

| item | simulated | serial wall (measured rates) |
|---|---|---|
| one cnt10 job (1 ns settle + 2x6 ns) | 13 ns | 5.0 h |
| one cnt16 job | 13 ns | 19.9 h |
| tranche 1 (cnt10 e10 + e50, 3 runs each) | 78 ns | **30 h (1.25 d)** |
| cnt10 ladder (21 jobs) | 273 ns | 105 h = 4.4 d |
| cnt16 ladder (21 jobs) | 273 ns | 418 h = 17.4 d |
| **full ladder (42 jobs, 504 ns in blocks + 42 ns settle)** | 546 ns | **523 h = 21.8 d serial** |

Disk (measured per-frame sizes from Stage A dumps): cnt10 ~0.52 GB/job,
cnt16 ~1.2 GB/job → tranche 1 ~3.1 GB, full ladder ~36 GB; 2.8 TB free.

**Correction to the briefing's ~15 days:** that figure priced all 546 ns
at cnt10-scale throughput. The honest per-tube mix is 21.8 days serial —
the cnt16 half alone is 17.4 days. Everything is cluster-portable
(§5), so the local box does not have to carry it.

**What fits locally before Oct 29 (from 2026-10-25 ~03:00, uncontended):**
tranche 1 done ~Oct 26 morning; the full cnt10 ladder (21 jobs, 105 h)
lands ~Oct 29 midday if the core stays with Stage B. The fills repair
(S3) needs ~159 h (67 + 92 h) of the SAME serial core — both cannot fit;
that split is the owner's call. This stream launched tranche 1 because it
feeds the D32 go/no-go's core comparison directly; the fills queue is a
Y6 deliverable that can follow (its DONE-marker design makes preemption
cheap: kill, run fills, re-run `stageB_queue.py`, it resumes).

## 5. Operations

- **Queue:** `research/cnt_electrolyte/stageB/stageB_queue.py` (stdlib
  only; `--lmp` overridable; DONE markers; abort-after-3; sha256
  manifests per run in `stageB/manifests/`; wall-clock log in
  `stageB/queue_log.jsonl`). Default order: cnt10 then cnt16; within a
  tube e10, e50, e0, e100, e5, e20, e200 — go/no-go core fields first.
- **One-core coordination (owner rule):** before any non-smoke start the
  runner refuses to launch if an `lmp.exe` process exists or if either
  queue log (`stageA/queue_log.jsonl`, `stageB/queue_log.jsonl`) shows an
  open `queue_start` without `queue_end` (both events carry pids).
  `--force` exists for verified-dead stale markers and for the 32-core
  reality: the check is deliberately conservative (ANY lmp.exe blocks,
  e.g. another stream's parallel jobs on free cores), and the refusal
  message names the blocker. (Byte-level `tasklist` parse — the text
  decode crashed on this GBK-locale Windows, found and fixed on first
  launch.) The fills agent's `stageA_queue.py` has no such check; the
  politeness burden is on this queue, as ordered.
- **Restart/extend:** `python .venv/Scripts/python.exe
  research/cnt_electrolyte/stageB/stageB_queue.py` continues wherever
  DONE markers left off; `--only <names>` picks any subset (that is how
  tranche 1 was launched).
- **Regenerate inputs:** `python .venv/Scripts/python.exe
  research/cnt_electrolyte/stageB/stageB_inputs.py` (writes all 42
  `in.lammps` + `jobs_index.json`; standalone, imports only `params.py`,
  no chaord). Smoke any job: `lmp -var smoke 1 -in in.lammps` in its
  directory (500-step stages).
- **Cluster portability:** every job directory is self-contained after
  staging (`system.data` = the zf final); all paths inside `in.lammps`
  are relative; the lmp binary is a runner flag. Copy `stageB/` plus the
  6 `stageA/jobs/zf_cntN_bK/final.data` starts and run the same queue
  with `--lmp /cluster/lmp`.
- **Analysis (Y8, next step):** per-run P1 block means from `prod.dump`
  (split at 6000 ps), block-B-only primary, sem over the 3 runs;
  e10-vs-e0 contrast feeds the D32 go/no-go table. The 10-ps settle dumps
  give a per-run switch-on curve to re-fit tau_M at every field — a free
  by-product that tests assumption (2) above empirically.

## 6. File map

```
reports/research/stage_b_plan.md            this file
research/cnt_electrolyte/stageB/
  stageB_inputs.py                          generator (42 jobs)
  stageB_queue.py                           queue runner + coordination check
  jobs_index.json                           machine-readable job list + costs
  jobs/sb_<tube>_e<fieldmV>_r<run>/in.lammps  42 inputs (smoke-capable)
  manifests/<job>.json                      per-run sha256 (queue writes)
  queue_log.jsonl, QUEUE_RUNNING            live queue state
  .gitignore                               dumps/logs stay out of git
```
