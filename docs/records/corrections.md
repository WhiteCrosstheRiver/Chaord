# Corrections — commit messages that named files they did not contain

Per Review 12 D50: over-claiming commit messages are listed here; history is
not rewritten. The D50 check (`tools/check_commit_claims.py`, wired into CI as
the `commit-claims` job of `.github/workflows/ci.yml`) fails any future commit
whose message names a file path absent from that commit's tree — slash paths,
plus bare evidence basenames (`final.data`, `log.lammps`, `equil.dump`,
`prod.dump`, `prod_thinned.dump`, `system.data`, `in.lammps`, the manifests,
`jobs_index.json`).

Audit method (Review 12 Q1, 2026-11-01): for each of the nine commits the work
order names (`c9df674`, `6f889bd`, `1113f15`, `cfe367a`, `e1925b8`,
`8fd427e`, `4de4d14`, `b994426`, `1957a08`) and for the six Q1 commits
themselves, the message was diffed against the commit's tree
(`git show --stat`) and `tools/check_commit_claims.py --range <c>~1..<c>` was
run per commit; the Review 10-12 window was then swept as one range
(`--last 25`, clean except the two rows below). Offenders:

| commit | the message said | what the commit actually contained | status |
| --- | --- | --- | --- |
| `b994426` | "Per job: final.data + log.lammps committed (small evidence)" | only `fills_cluster/fill_series_manifest.json` (30 lines). The four `fills_cluster/fill_{cnt10,cnt16}/{final.data,log.lammps}` files were absent: the fills_cluster `.gitignore` whitelisted `!fill_cnt10/final.data` under a `*` rule while the parent directory itself was excluded — git cannot re-include a file under an excluded directory, so the whitelist was dead until fixed. | corrected by `7e5fe1d`. NOTE: the automated check passes b994426 retroactively (the bare basenames resolve against Stage A evidence already tracked in the same tree) — this is the known limitation recorded in the check's docstring; the exact record is this row. |
| `73f205a` | "even the thinned series (2.8 G total) stay out of git … they live in the local analysis archive research/cnt_electrolyte/stageB/cluster_results/<job>/prod_thinned.dump, retrievable on demand" | the per-job small evidence (final.data + equil.dump + log.lammps for 33 Stage B jobs) + `prod_series_manifest.json`; no prod_thinned.dump. The mention is explicit-exclusion prose, not a presence claim, but the D50 letter ("a commit message may only name files the commit contains") flags it, and the check does. | recorded; the series remain local-only and are now fenced (`9dd0590`); their sha256 pinning moves to the Q4 release archive. |
| `1957a08` | "every job's final.data + equil.dump + prod_thinned.dump (1/12 series) + log retrieved to the local working tree (1.03 GB)" and "the repo carries the sha256 series_manifest.json (54 artifacts pinned)" | only `confirmatory/.gitignore` + `confirmatory/series_manifest.json` (2 files). No run evidence was committed. The check flags it (`prod_thinned.dump` named, absent). Two further defects the diff-stat view missed: (a) the retrieved evidence had landed at `confirmatory/sb16L_*/` (no `jobs/` prefix), not where later records placed it; (b) the series_manifest.json it added pins, 54/54 exactly (name + bytes + sha256), the SMOKE-era `jobs/` artifacts — its `sb16L_e0_b1/equil.dump` record is the 0-byte empty-file hash (e3b0c442…) and its final.data record (398,208 bytes, 07e4013b…) is the smoke worktree file — NOT the retrieved run artifacts, which were sha256-pinned nowhere in git. | corrected by `c2daf75` (run log.lammps) + `9dd0590` (run final.data); sha256 pinning of the oversized local-only series moves to the Q4 release archive. |
| `4cc4dab` | "all 18 jobs' final.data + log.lammps + equil.dump tracked (1957a08's message claimed the evidence was retrieved but only the series manifest + .gitignore landed — recorded in docs/records/corrections.md)" | only `confirmatory/.gitignore` + `confirmatory/jobs/.gitkeep` (2 files): it committed the FENCE, no evidence. At its tree confirmatory/ carried no log.lammps at all, and the final.data/equil.dump tracked under `jobs/` were the 6f889bd smoke-run artifacts (the smoke equil.dump is 0 bytes). The automated check cannot see this substantive over-claim — the bare basenames resolve against smoke/Stage A/Stage B/fills copies elsewhere in the tree; it flags only the forward reference to this file, resolved two commits later. | corrected by `c2daf75` + `9dd0590`. |
| `9dd0590` | "…(776,783 bytes each as retrieved, LF, blob identical to the worktree bytes)" | the 18 `confirmatory/sb16L_*/final.data` blobs are 776,259–777,134 bytes; 776,783 is sb16L_e0_b1's, so "each" is wrong. The file list, the LF statement and the substance of the claim are exact. | corrected here (no history rewrite). |

Forward references inside the Q1 series (a named file that did not exist yet in
the naming commit but landed within the same six-commit series):

| commit | the message said | the file landed in |
| --- | --- | --- |
| `7e5fe1d` | "(b994426's message claimed these were committed but only the manifest was — recorded in docs/records/corrections.md)" | `e818bbe` (two commits later) |
| `4cc4dab` | "— recorded in docs/records/corrections.md)" | `e818bbe` (two commits later) |
| `e818bbe` | "RED TEST VERIFIED on a scratch branch: an over-claiming commit ('adds …sb16L_e0_b1/log.lammps' with the file absent) failed the check with exit 1, then the branch was deleted" | the quoted example path did not exist on master until `c2daf75`, the next commit; the red test itself ran on a scratch branch (re-done and recorded verbatim below) |

Record notes (messages the audit passes; records that mislead without them):

- `6f889bd` (audit clean) committed the Z3 job inputs AND the smoke-run
  outputs under `confirmatory/jobs/sb16L_*/`: in.lammps (5,646 B),
  system.data (205,744 B), smoke prod.dump (392,912 B blob each — ~7 MB for
  the 18, not "10 MB each"), smoke final.data (388,459 B blob) and a 0-byte
  smoke equil.dump. All remain tracked. They evidence the smoke tests, NOT
  the Z3 runs.
- `c2daf75` (audit clean): "git ls-files now shows 18 final.data + 18
  log.lammps + 18 equil.dump for the Z3 jobs" was count-true, but the
  final.data/equil.dump it counted are the smoke artifacts above; the run
  final.data landed only in `9dd0590`. The evidence map below fixes the
  provenance.

## Evidence map (Z3/Y6 provenance after Q1)

- Z3 RUN evidence: `research/cnt_electrolyte/confirmatory/jobs/sb16L_*/log.lammps`
  (1,374,919 B each — the cluster-run log, `c2daf75`) and
  `research/cnt_electrolyte/confirmatory/sb16L_*/final.data` (776,259–777,134 B
  each — the cluster-run endpoint, `9dd0590`). The per-job run equil.dump
  (~20.3 MB) and prod_thinned.dump (~200.8 MB) stay local-only at
  `confirmatory/sb16L_*/`, fenced, to be sha256-pinned by the Q4 archive
  manifest.
- Z3 SMOKE evidence (6f889bd, tracked): `confirmatory/jobs/sb16L_*/` also
  holds the smoke-run in.lammps/system.data/prod.dump/final.data/equil.dump.
  A `jobs/` final.data or equil.dump is NEVER the run endpoint.
- Y6 fill evidence: `research/cnt_electrolyte/fills_cluster/fill_cnt10/`{final.data
  3,259,061 B, log.lammps 646,260 B}, `fill_cnt16/`{final.data 4,491,548 B,
  log.lammps}, plus the sha256 manifest `fill_series_manifest.json`
  (`b994426` added the manifest; `7e5fe1d` completed the set). The ~960 MB
  per-case thinned series stay local-only (Q4 archive).
- Stage A fill-repair evidence: the post-repair fill logs (tracked, `43193a1`)
  and the pre-repair failure logs
  `stageA/jobs/fill_{cnt10,cnt16}/{log.lammps,screen.log}.failedNPT-20261019`
  (`8ea8df1`) — both sides of the documented repair (4b8f2be, a15bc5b).
- Stage B (33 ladder jobs): final.data + equil.dump + log.lammps were already
  tracked (`73f205a`). The thinned prod series were untracked AND unfenced
  (the `!cluster_results/**` whitelist un-ignored them — a bare
  `git add -A` would have staged ~4 GB) until `9dd0590` fenced them.

## D50 check — red tests (2026-11-01)

Both tests ran on scratch branches deleted afterwards (`git branch -D`); no
pushed history carries them. Verbatim:

Test A — the b994426 over-claim class (bare evidence basenames): an orphan
commit `2cfbc71` on branch `scratch-red-a` with the EMPTY tree
(4b825dc642cb6eb9a060e54bf8d69288fbee4904) and message "Per job: final.data +
log.lammps committed (small evidence); the thinned series stay local-only with
the sha256 manifest."

```
$ python tools/check_commit_claims.py --range c2daf75..scratch-red-a
OVER-CLAIM in 2cfbc71: message names paths absent from its tree: ['final.data', 'log.lammps']
1 over-claiming commit(s) -- see docs/records/corrections.md for the known pre-check ones
(the ellipsis token '...' in a message is an abbreviation, not a path: it is ignored, but messages that truncate a real path with it are treated as naming the FULL path when that full path exists in the tree -- see the elision handling above)
EXIT CODE: 1
```

Test B — the absent-path class: an empty commit `12c77c8` on branch
`scratch-red-b` off master, message "adds the per-job manifest
confirmatory/sb16L_e0_b1/manifest.json plus the run final.data and log.lammps
for the job" (no per-job manifest exists anywhere in the tree):

```
$ python tools/check_commit_claims.py --last 1
OVER-CLAIM in 12c77c8: message names paths absent from its tree: ['manifest.json']
1 over-claiming commit(s) -- see docs/records/corrections.md for the known pre-check ones
(the ellipsis token '...' in a message is an abbreviation, not a path: it is ignored, but messages that truncate a real path with it are treated as naming the FULL path when that full path exists in the tree -- see the elision handling above)
EXIT CODE: 1
```

(The slash path itself is exempted by the check's first-segment heuristic —
"confirmatory/…" is not rooted at a tracked top-level directory — and the
bare-basename rule catches the claim.) After both branches were deleted, the
Q1 chain verifies green: `python tools/check_commit_claims.py --last 6` →
"commit-claims check: clean (6 commits)", exit 0.

## Q3 void entry (D54, T4, 2026-11-05)

Q3 was built from the uncorrected Q2 draft composition (caf8b78: EC 0.8984/Å,
EMC 0.5578/Å, Li+ 0.1541/Å), not the corrected reservoir filling (fc8bec2:
EC 1.2701/Å, EMC 0.5917/Å, Li+ 0.2673/Å, PF6- 0.2127/Å). It therefore tested
neither the pilot filling nor the reservoir filling, and is VOID as a filling
test (open_items_v7.md D54). Two results were reported for the same data:

| commit | shift (block B vs zero-field) | p (Welch) | script | data |
| --- | --- | --- | --- | --- |
| `36daad6` / go_no_go.md §14 | +0.008 | 0.78 | `p1_fixed.py` | 12-ps thinned dumps |
| `0f63a06` | +0.035 | 0.1285 | `p1_extract.py` | dense 1-ps dumps |

Both numbers are preserved (with script attribution) in
`research/cnt_electrolyte/q3_evidence/q3_results.json`; the per-job final.data +
log.lammps + the sha256 `q3_series_manifest.json` (54 artifacts, 1/12-thinned
series, raw dumps wiped on the cluster 22 G → 1.3 G) are tracked as of
`841d5bc`. No post-hoc test was added; the pre-registered D39 call
(q3/PLAN.md, dc2607a) was run exactly as committed on a composition the
corrected filling analysis had already superseded — the planning defect is the
record here, not the analysis.

## Large hcp tilings silently lost atoms (2026-11-09)

Found by the second pilot of `research/macrostate_selection`, not by the
acceptance suite. `build_conventional` bounded its lattice-translation search by
box extent over the shortest lattice vector. On a skewed basis the needed integer
offsets are larger, so the far corner of a large orthohexagonal hcp tiling was
dropped: a program stating 2,016 atoms built 1,908, with no refusal, and the
hole melted the crystal under MD. The 32-atom bench case fits inside the old
safety margin, so A3 passed. No published number used an hcp cell this large.

Fix: bounds now come from the cell corners in conventional coordinates, and the
tiling refuses if its atom count differs from the cell volume's. Regression test:
`tests/test_large_hcp_tiling.py`.

## Lift printed a liquid over a thin crystal slab (2026-11-09)

Also found by the macrostate pilots. A frame holding a crystal slab too thin to
have interior atoms is correctly refused by the slab arm, but the `auto` cascade
then fell through to `lift_fluid` and printed a homogeneous liquid with
`assert solid_clusters 0`, although `is_single_phase` had already said the frame
was not one fluid. For atomic frames it now refuses; bonded (molecular) frames
keep the fluid lift, since they fail that check on component sizes and it says
nothing about crystal (`src/chaord/lift/legacy.py`, test
`tests/test_lift_refuses_unresolved_slab.py`). A slab whose crystal is a small
enough fraction to pass `is_single_phase` still lifts as a liquid; this is a
stated limit.

Open, not fixed: the slab arm classifies orientation from a histogram of bond
directions, and near melting it can print `z <111>` for a crystal grown along
[001], with a spurious `constrain strain`. `build_slab` does not read `orient` but
does use the strain, so a misidentified orientation can alter a rebuilt slab.

## Program-length verdict corrected to NOT SUPPORTED (2026-11-09)

`research/program_length/RESULTS.md` and the draft first called the verdict
undetermined. PLAN.md's "iff ... otherwise NOT SUPPORTED" makes NOT SUPPORTED the
registered outcome when a clause cannot be met. `curve.py` was committed only with
its results (7612053), not before the run.
