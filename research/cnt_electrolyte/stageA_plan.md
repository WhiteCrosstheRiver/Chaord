# Stage A plan: relaxation times and filling (Y6, Review 10)

Written by stream S3 on 2026-10-14/15. Inputs, queue runner and analysis
live next to this file (`stageA_*.py`); run outputs live under
`stageA/jobs/` (gitignored except `in.lammps`); sha256 manifests under
`stageA/manifests/` (tracked).

## What was built

Model: the **pilot model** per `docs/reviews/owner_inputs.md` (D37 note:
the owner's Moltemplate model is not yet provided), i.e. rigid
OPLS-AA-based EC/EMC/LiPF6 from Nutakki 2023 Tables II-III, Werder sp2-C
tube carbons, Lorentz-Berthelot mixing (recorded deviation), full +-1 ion
charges, PPPM 1e-4, cutoff 12 A, dt 1 fs.

Jobs (34 total; `stageA/jobs_index.json` is the machine-readable list):

| group | jobs | what |
|---|---|---|
| `zf_cntN_bX` | 8 | zero field; 3 fresh builds per tube (new seeds/random orientations) + mirrored copy of build 1 (z -> Lz - z); 100 ps ramp (10->300 K) + 900 ps equil + 2 ns production |
| `wf_cntN_bX` | 8 | switch-on from each zf final state, field 0.01 V/A (D30 default working field), 2 ns |
| `sf_cntN_bX` | 8 | switch-on, field 0.2 V/A (saturating), 2 ns |
| `off_cntN_bX` | 8 | switch-off from each sf final state, field 0, 2 ns |
| `fill_cntN` | 2 | open tube in a bulk bath (1 M LiPF6, 1:1 EC:EMC, 30 A reservoir each side, 15 A lateral liquid shell), 500 ps T+P anneal then NPT 1 atm 300 K, 5 ns; tube starts EMPTY (imbibition) |

Settings per Y6: one Langevin scheme for every mobile atom, damping 1 ps
(`fix rigid/nve/small molecule langevin` for multi-atom bodies, `fix nve`
+ `fix langevin` for Li+; no Nose-Hoover anywhere); dumps every 1 ps with
`dump_modify ... format float %.10g` and image flags (`ix iy iz`); thermo
every 1 ps.

NPT without breaking the one-Langevin rule: the filling runs barostat
with `fix nph iso ... dilate mobile` - pure pressure control, no
thermostat (this is the fix-rigid docs' rigid-bodies-plus-point-
particles hybrid). `dilate mobile` never rescales the frozen tube. Li+ is
time-integrated by the `fix nph` itself in those two jobs (no `fix nve`
on Li+ there, to avoid double integration). During the 500 ps anneal the
barostat target ramps -2000 atm -> 1 atm (Pdamp 2 ps); the 5 ns
production is constant 1 atm.

Filling baths are built by packing one small tri-periodic liquid block
toward the target bulk density with skip-on-fail insertion (bulk-density
sequential packing jams in a small periodic box - found on execution;
skips are recorded per species and salt is kept neutral), tiling it
2x2x4, and deleting every molecule that clashes with the tube atoms
(Li/PF6 deletions equalized in pairs so the bath stays neutral). The
tube starts EMPTY, so the 5 ns run is a genuine imbibition experiment.
The packed bath sits at ~87% of the estimated bulk number density, and
the run opens with a 500 ps temperature-and-pressure anneal (10->300 K,
barostat target -2000 atm -> 1 atm, Pdamp 2 ps): the smoke test showed
the model's own 1-atm equilibrium density is ~25% BELOW the CRC-based
estimate (initial bath pressure +20,000 atm), so the anneal mostly lets
the bath EXPAND to the model's equation of state before the 5 ns
production. The bath composition (1:1 EC:EMC by volume at build, 1 M
salt) is exact by construction; the equilibrated density is the model's
own, recorded honestly from the run thermo.

## Honest local-cost statement

Measured serial throughput on the pilot set (`lmp.exe` 22 Jul 2025 u4,
1 rank): cnt10 = 971 atoms x 550 ps in ~12.5 min, cnt16 = 2058 atoms x
550 ps in ~51.7 min -> ~22,000 atom*ps/min at the 2k-atom scale. Cost
estimates below use that number. **The filling cells are 10-14x larger
than anything the pilot measured, and per-atom cost grows with system
size (neighbor lists, PPPM grid), so the filling estimates are
optimistic lower bounds; the true serial cost is plausibly 20-50%
higher.**

| job group | atoms | steps | est. wall time (serial) |
|---|---|---|---|
| zf_cnt10 x4 | 971 | 3.0 ns | 4 x 2.2 h = 8.8 h |
| wf/sf/off_cnt10 x12 | 971 | 2.0 ns | 12 x 1.5 h = 17.7 h |
| zf_cnt16 x4 | 2058 | 3.0 ns | 4 x 4.7 h = 18.7 h |
| wf/sf/off_cnt16 x12 | 2058 | 2.0 ns | 12 x 3.1 h = 37.5 h |
| fill_cnt10 | 16,040 | 5.5 ns | **~67 h (~2.8 days)** |
| fill_cnt16 | 22,009 | 5.5 ns | **~92 h (~3.8 days)** |
| **total** | | ~97 ns | **~241 h ~= 10 days** continuous |

(Atom counts are the built values from `data/stageA_build_manifest.json`;
the two fill rows are the honest headline: the reservoir requirement -
3 nm each side of a 4.92 nm tube plus a bulk-like lateral shell - fixes
a 16k/22k atom cell at 5.5 ns, which is **not** a "fits in 1-2 days"
proposition on the local serial box. The 32 tube jobs (~3.5 days) do fit
that budget; the owner_inputs.md 1-2-day estimate priced Stage A's tube
runs, not the filling cells. Smoke-test timings: the fill smokes ran
1,000 steps of the 16k/22k systems in 60/84 s, consistent with the
22k atom*ps/min planning number.)

Everything is delivered runnable regardless: the queue is running now
locally (sequential, tube jobs first, fills last), and the two fill jobs
are the natural candidates to move to the owner's cluster when it exists
- the inputs are self-contained (relative paths, `system.data` staged
per job) and the `lmp` binary is a queue-runner flag
(`python stageA_queue.py --lmp /path/to/lmp`), never baked in.

## Reproducibility / operations

- regenerate everything: `python stageA_build.py && python
  stageA_inputs.py` then `python stageA_queue.py` (all in
  `research/cnt_electrolyte/`, standalone, no chaord import).
- the queue is restartable: finished jobs carry a `DONE` marker and are
  skipped; killing the process at any time is safe.
- smoke self-test of any input: `lmp -var smoke 1 -in in.lammps` in the
  job directory (500-step versions of every stage; used to validate the
  fix set before the queue started).
- per-run manifests: `stageA/manifests/<job>.json` (sha256 of the input,
  staged data, logs, final.data, and every raw dump; dumps stay out of
  git via `stageA/.gitignore`).
- analysis: `python stageA_analyze.py` (numpy only) -> tau_rot, tau_M,
  mirrored-pair verdict, reversal counts, filling counts with block
  SEMs; JSON in `stageA/analysis/`, small P1(t) series in
  `stageA/p1_series/`. It runs after data exists; partial runs are
  analyzed with whatever jobs have finished.

## Addendum 2026-10-16: the 22:33 Oct 15 queue abort, diagnosis, restart

**What happened.** After zf_cnt16_b1 finished (22:33), zf_cnt16_b1m /
b2 / b3 each died in ~0.3 s with `ERROR on proc 0: Out of range atoms -
cannot compute PPPM` immediately after the step-0 thermo line; the
abort-after-3 guard stopped the queue, leaving 24 chained jobs and 2
fills unstarted.

**Diagnosis (the suspected bad builds were NOT bad).** All three data
files pass every packing floor: min inter-molecular distance 2.004 /
2.004 / 2.041 A (floor 2.0 A), min molecule-tube distance 2.493 / 2.510
/ 2.481 A (H-C_tube floor 2.2 A; the packer's wall check is per-atom
against the actual cylinder radius R, so the suspected
"spherical-exclusion clips the wider tube" bug is not present), every
molecule whole in raw coordinates, every bond 1.09-1.606 A. zf_cnt16_b1
(the run that worked) has the same statistics (2.004 / 2.493 A), and
b1m is a byte-faithful mirror: identical step-0 PotEng -5373.4878.
The crash is not overlap physics: step-0 forces are sane (max 76
kcal/mol/A) and the run dies during step 1 EVEN AT dt = 0.1 fs, where
no physical force can displace an atom out of the PPPM grid brick. The
outcome is instead heap-layout dependent in this local LAMMPS build
(conda 22 Jul 2025 u4): adding one inert `variable` line to the input
flips b1m from immediate crash to a clean 12000+-step run; so do other
input perturbations. b1 itself rode out the same packed-start
Coulomb-relaxation transient (Tmob spike to 642 K at step 1000, then a
normal ramp). Root cause: `fix rigid/nve/small molecule` (the
distributed-body variant) corrupts body data at the first integration
step for some heap layouts in this build - 3 of 4 cnt16 fresh builds
lost that lottery; the pilot four and the cnt10 builds won it.

**Fix (run inputs).** All Stage A inputs now use
`fix rigid/nve molecule langevin ...` - the non-distributed rigid
variant: same Langevin keyword, same 1 ps damping, same per-atom
application (the one-Langevin rule is untouched), deterministic body
bookkeeping. Verified: all three failing jobs then integrate cleanly
and reproduce bit-identically across reruns (Tmob tracks the 10->300 K
ramp, e.g. 56 K at 16 ps where the target is 56.4 K). The 5 already
DONE jobs keep their history (their manifests pin the sha256 of the
inputs that actually ran).

**Builder guard (permanent hardening).**
`build_systems.validate_data_file(path)` now re-parses every data file
after writing and REFUSES the build unless (1) every inter-molecular
and molecule-tube minimum-image pair distance is >= 0.95x the packing
floor for its type pair, (2) every liquid molecule is whole in raw
(as-written) coordinates (raw diameter == min-image diameter; the
box-spanning frozen tube is exempt), (3) every bond's raw length is
< 3 A. `stageA_build.py` calls it for every zf build, mirror and fill
and records the stats in `data/stageA_build_manifest.json` under
"guard". All 10 Stage A data files pass: inter-mol >= 2.004 A,
mol-tube >= 2.209 A, zero molecule splits, max raw bond 1.606 A. The
three "failing" systems were rebuilt through the guarded builder and
are byte-identical to the originals (sha256 in the queue/manifests) -
confirming the builds were never the problem.

**Smoke.** All three jobs pass `lmp -var smoke 1 -in in.lammps` with
the new inputs (rc = 0, all 500-step stages complete).

**Restart.** Queue relaunched detached 2026-10-16 16:25 with the same
command as before (`python stageA_queue.py` from
research/cnt_electrolyte, appending to `stageA/queue.out` /
`stageA/queue.err`, events in `stageA/queue_log.jsonl`); the 5 DONE
jobs were skipped and zf_cnt16_b1m started immediately: step 16000
within ~2 min, Tmob 46-64 K (ramp target 56 K), PotEng relaxing toward
the equilibrated ~-6715 kcal/mol.
