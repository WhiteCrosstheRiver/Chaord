# Independent verification report — Gate A′ work order W3

* **Run date:** 2026-10-08 20:58 → 2026-10-09 03:53 (+0800; the report file is named
  `verification_2026-10-08.md` per the work order).
* **Verified commit:** `f90e988e03b6ca9006e1a3b0a0790c13420f805d` (HEAD, clean working
  tree at run start; `git status --porcelain` empty).
* **Runner:** local Windows 10, Python 3.13.9 (`.venv`), Git Bash. Every command below
  was run by me from the repository root.
* **Who I am:** a fresh verifier agent that wrote none of this code (work order W3 of
  `docs/reviews/open_items_v2.md`). I modified no source file, made no commit and
  pushed nothing. Scratch artifacts were `scratch_verifier_*` in the repo root (since
  removed) plus a temp directory for per-chunk intermediates.

## How the runs were executed (honest disclosure of a harness constraint)

My execution harness kills any background task at ~60 minutes. Two attempts to run
`tools/acceptance.py` as one command were killed at ~63 and ~60 minutes with no
criterion having reported (only stderr RuntimeWarnings were in the logs; nothing was
lost — the runner prints all verdicts at the end). I therefore ran the **unmodified
runner with `--only` in six chunks**, and the fast pytest suite in seven chunks (same
file set, same alphabetical order), each chunk inside the kill window:

| Acceptance chunk | Criteria | Wall time | Result |
| --- | --- | --- | --- |
| 1 | A1, A6, A9, A10, A12, A14 | 23:02:45 → 23:22:01 (~19 min) | 6/6 pass |
| 2 | A2, A3 | 23:22:17 → 23:22:32 (~15 s) | 2/2 pass |
| 3 | A4 | 23:22:46 → 23:22:51 (~5 s) | 1/1 pass |
| 4 | A5 | 23:23:16 → 00:10:06 (~47 min) | 1/1 pass |
| 5 | A7, A8 | 00:13:58 → 00:13:59 (~1 s) | 2/2 pass |
| 6 | A11, A13 | 00:14:04 → 00:21:53 (~8 min) | 2/2 pass |

Each chunk ran on the same clean tree at `f90e988` and wrote its own
`acceptance.json`-schema report; I merged them into `scratch_verifier_acceptance.json`
(criteria sorted A1..A14, `passed 14/14`) and concatenated the six details files.
Chunk provenance blocks (identical in all six): `commit f90e988e03b6ca9006e1a3b0a0790c13420f805d`,
`working_tree_dirty: false`, `runner "Windows 10"`,
`run_kind "local (bookkeeping per AGENTS.md; citable evidence is the clean-runner artifact)"`,
`run_url: null`. One further deviation from the mandated command line: I added
`--details <scratch>` so my run would not overwrite the tracked
`reports/acceptance_details.md`; the criteria code path is untouched.

## Part 1 — Acceptance A1–A14

**14/14 criteria pass.** Evidence strings, verbatim headline numbers:

| ID | Name | Verdict | Headline numbers in the evidence |
| --- | --- | --- | --- |
| A1 | parse and format | **PASS** | 10/10 spec examples parse; parse==IR and fmt-idempotent property laws hold on 10,000 generated programs |
| A2 | canonical invariance | **PASS** | 9/9 crystal + random-solution cases; 18/18 rigid transforms (rotation + translation + re-imaging + re-ordering, 2 per case) byte-identical |
| A3 | exact round trip (ordered) | **PASS** | 9/9 lift→build→lift byte-identical; 9/9 structure gate (translation-aligned species-aware assignment, displaced mass beyond 0.25 d_NN ≤ 5%) per approved deviation D6; pymatgen StructureMatcher 0/9 (primitive reduction) and 8/9 (primitive_cell=False, misses l12_ni3al) recorded as the reason |
| A4 | defect recovery | **PASS** | 26 host × defect-type × temperature cells (3 hosts, all four K-V kinds + 1 mixed-kind); precision ≥ 0.95 in 26/26 and recall ≥ 0.95 in 26/26; worst precision 1.00, worst recall 1.00 (fcc-Cu/vacancy/room) |
| A5 | statistical round trip | **PASS** | 6/6 cases-with-floor pass; fluid 4/4 (target ≥ 90%), interface 1/1 (≥ 90%), glass 1/1 (≥ 80%, the Kob–Andersen case); 6 no-floor cases honestly skipped (synthetic frames) |
| A6 | conservation | **PASS** | 126 lifts: atom count == conserve line on 126/126; verifier's own region arithmetic pins/cross-checks 81 programs; charge consistent 126/126 |
| A7 | phase segmentation | **PASS** | 10/10 interface frames ≥ 95% correct; worst 0.982 (interface/lj_solid_liquid frame 3); judged fraction disclosed per frame (cu_water 0.854–0.874; lj_solid_liquid 0.355–0.430) |
| A8 | reactive census | **PASS** | census exact on both plans ({H2O 40, HO 9, H 9} and {H2O 25, HO 5, H 7}); adsorption sites 100% (8/8 on Pt(111); gate ≥ 90%) |
| A9 | compression | **PASS** | 5 systems ≥ 1,000 atoms; worst ratio 0.67% (reference/nacl_aq, 1640 atoms); worst crystal 0.49% (l12_ni3al, 1372); gate evidenced on heterogeneous systems (2 of 5) |
| A10 | determinism | **PASS** | repeated lift byte-identical: True; same-seed builds identical coordinates: True |
| A11 | speed | **PASS** | lift of 100,000 atoms (mode=fluid) in 2.9 s (target ≤ 120 s); program states the exact count: yes |
| A12 | static checks | **PASS** | 4/4 seeded errors caught: lattice_mismatch, impossible_density, charge_imbalance, overlap_0.1σ |
| A13 | no crashes | **PASS** | 125/125 bench frames lift with zero unhandled exceptions; 0 atoms in residual blocks; 60 frames carry routing diagnostics |
| A14 | documentation | **PASS** | 35/35 dialect keys with structured reference entries; example citation gated for 26/35 (9 keys have no example anywhere: atom, chirality, dialects, lift_version, nanotube, note, reconstruction, source, stacking — reported, not hidden); sketch_check pass |

## Part 2 — Every snippet in `docs/reviews/open_items_v2.md`

The file contains six verification snippets (plus the shared common header, which I
prepended verbatim to each). Each was run as written, from the repository root, in the
project environment. Snippet location = work-order §, first code line of the block.

### W1 (§2 W1, line 101) — crystal/gas detection, unit-free

```
0 of 9 bench crystals called amorphous under core + glass
argon gas accepted by mode='fluid'
```
(1.8 s). Plan: today 9 of 9 amorphous / argon refused; **done: 0 and accepted.**
**Matches the done state.**

### W4 (§2 W4, line 213) — Cu solid–liquid slab lift

```
['state T 300.00 K', 'conserve atoms Cu 832', 'backend eam', 'lattice fcc']
0 frenkel_pair lines
```
(5.6 s). Plan today: `units lj`, `state T 300.00`, `conserve atoms X 832`,
`backend lj`, `lattice hcp`, 11 frenkel_pair lines. Done (step 2): Cu 832, metal
units, fcc, ≤ 1 defect line. **Matches the done state** on every clause the snippet
prints. Addendum checks I ran beyond the snippet: the lifted program builds
(832 atoms, physics=False); `a 3.615 A` (the FBD-1986 Cu_u3 calibration constant; the
committed test `tests/test_metal_slab_lift.py` gates |a−a₀|/a₀ ≤ 2% and
|a−a(1335 K)|/a(1335 K) ≤ 2% — I did not re-derive a(1335 K) by NPT MD myself; that
test passed in my fast-suite run); the provenance carries
`note "T assumed (dialect default 300); not measured from the frame"` (the W11
assumed-flagging on this path).

### W5 (§2 W5, line 246) — strained boxes build

```
0 of 9 strained crystals do not build: []
```
(2.5 s). Plan: today 6 of 9 refuse; **done: 0. Matches the done state.**

### W6 (§2 W6, line 277) — lj_solid_liquid minimum pair distance

```
0 0.955
1 0.925
2 0.946
3 0.96
4 0.957
```
(0.6 s). Plan: today 0.047–0.182 σ; **done: ≥ 0.80 in every frame.**
**Matches the done state** (min observed 0.925 σ, frame 1).

### W8 (§2 W8, line 334) — build 60 CO2 at 1.00 g/cm³

```
0 builds
1 builds
2 builds
```
(5.6 s). Plan: today 'cannot place 60 CO2 at this density' × 3; done: builds on 20 of
20 seeds. The snippet's own loop is 3 seeds — **all 3 build, matching the done
state**; the 20-of-20 clause is covered by the committed fast-suite test
`tests/test_molecular_contacts.py::test_dense_co2_builds_on_20_seeds_and_clears_the_contact_floor`,
which passed in my run.

### W9 (§2 W9, line 374) — Pt(100) 4-fold hollow classification

```
['adsorb O count 4 site hollow']
```
(0.9 s). Plan: today `site bridge`; **done: `site hollow`. Matches the done state.**

**Snippet tally: 6 of 6 print the plan's done state; 0 differ.** None needed a
reduced loop — the longest ran 5.6 s.

## Extra runs

### Fast suite — `.venv/Scripts/python.exe -m pytest tests -m "not slow" -q --no-header`

Run in seven chunks (harness 1-hour kill; same files, same order; the chunk
collections together match the single-directory collection exactly — verified by a
node-id diff whose only difference was `tests/test_thermal_recovery.py`, whose 3
tests are all `@pytest.mark.slow` and thus deselected either way: 1138 selected,
252 deselected):

| Chunk | Files | Result | Time |
| --- | --- | --- | --- |
| F1 | acceptance/test_ka_glass_a5, acceptance/test_mutations | 24 passed, 3 deselected | 30:49 |
| F2 | acceptance/test_noise_floors, test_o2_o3, adversarial/ | 51 passed, 6 deselected | 29:29 |
| F3 | test_a5_rebuild_meta … test_conservation | 198 passed, 4 deselected | 31:29 |
| F4 | test_crystal … test_hcp_routing | 200 passed, 1 skipped, 5 deselected | 4:09 |
| F5 | test_integrations … test_sro_invariance | 299 passed, 3 deselected | 22:42 |
| F6 | test_strained_boxes | 235 passed, 216 deselected | 1:09 |
| F7 | test_string_escapes … test_xfail_hygiene | **129 passed, 1 FAILED**, 12 deselected | 25:48 |

**Total: 1136 passed, 1 failed, 1 skipped, 252 deselected (slow).** The one skip is
`tests/test_defects.py:71: unary host has no antisites` (expected parametrization
skip). The failure, verbatim:

```
FAILED tests/test_xfail_hygiene.py::test_strict_xfail_only_in_red_team_registry
AssertionError: strict xfail outside the red-team registry: ['tests\\test_strained_boxes.py']
-- register the finding in tests/adversarial/conftest.py FOILED or fix the test)
```

The strict-xfail marks on the 94 known-gap strained-box identity configs (added in
HEAD `f90e988` itself) are not registered in the red-team registry, so the repo's own
xfail-hygiene gate fails: **the fast suite is red at HEAD.**

### `tools/check_magic_numbers.py`

```
magic-number check: clean
```
exit 0.

### `tools/sketch_check.py spec/examples/*.chaord`

10/10 examples `OK` (01_solid_liquid_lifted … 10_brine_spce; per-file block/statement
counts printed), exit 0.

### Slow suite — cited, not run (per the work order)

The slow suite (252 tests: nightly statistical suite incl. the full 216-config
strained identity matrix and the ka_glass A5 statistical test) is the clean CI's job:
`.github/workflows/ci.yml` `nightly` job ("Acceptance A1–A14 (clean machine)" +
"Slow statistical suite", artifact `clean-acceptance-report`). The latest clean-runner
acceptance evidence named in the repo is GitHub Actions run #39 (2026-09-30,
`ed85108`, fully green per `reports/gate_a_prime.md`); I could not verify any newer
clean-runner run from inside this machine.

## Known-context items I verified rather than trusted

* **Strained-box identity matrix (W5):** `tests/test_strained_boxes.py`'s `_GAP` set
  contains **exactly 94** `(case, strain)` entries (counted by me from source); the
  matrix is 9 cases × 24 strain configs = 216, so **122 hard-asserted + 94
  strict-xfail = 216** — the claim's arithmetic checks out structurally. Behaviorally
  I ran the file's fast half (chunk F6): the 9-config reproducer identity test
  (`test_strained_crystal_builds_and_round_trips_at_reproducer_strain`, byte-identical
  lift→build→lift at +0.2% x) and the 216-config build matrix
  (`test_strain_matrix_crystal_builds`) **all 235 passed**, and exactly 216 identity
  tests were deselected as slow. The slow identity matrix itself (122 green + 94
  xfail) is the nightly's job — cited above, not run here.
* **ka_glass A5 uses Kob–Andersen:** `tests/acceptance/test_ka_glass_a5.py` exists
  and frames the case as "Kob–Andersen 80:20 at ρ*=1.2" (D9); `bench/reference/ka_glass/`
  holds 15 frames + provenance with dialect `core + glass + lj_mixtures`;
  `bench/reference/lj_glass/provenance.json` carries
  `a5_excluded: "… RETIRED from A5 by the Kob–Andersen binary case (ka_glass) …"`.
  My A5 run passed with glass 1/1 (the KA case; gr_rms ×0.86, cn_tv ×0.29 of the
  cross-quench floor per that provenance note).
* **W4 step 2 (Cu slab):** verified by the snippet above plus
  `tests/test_metal_slab_lift.py` (6 tests) passing in chunk F5.

## VERDICT

Acceptance is 14/14 on my independent local run and all six open-items snippets print
their planned done state — but the tree is **not** gate-ready:

1. **FAIL — the fast suite is red at HEAD `f90e988`.**
   `tests/test_xfail_hygiene.py::test_strict_xfail_only_in_red_team_registry` fails
   because HEAD added strict xfails for the 94 known-gap strained-box configs in
   `tests/test_strained_boxes.py` without registering them in the red-team registry
   (`tests/adversarial/conftest.py` FOILED). This violates the repo's own hygiene
   gate and D5 (merge only on green CI) — and contradicts HEAD's commit message,
   which claims the strict-xfail move "keeps the enforcement contract exactly" while
   the fast suite it ships is red. Gate A′ (D16) cannot be declared on a red clean
   run.
2. **OPEN — no clean-runner acceptance artifact at the final commit.** The committed
   `reports/acceptance.json` is a **local bookkeeping run** (`run_kind` local,
   working tree dirty) from commit `b49c51b`, 2026-10-05 — not the clean-runner run
   on `f90e988` that W3/D16 require. The newest clean-runner evidence named anywhere
   in the repo is Actions run #39 of 2026-09-30 (`ed85108`).
3. **RECORDED GAP (not new, verified to exist):** the W5 lift-side snap is not
   landed — 94 of 216 strained-identity configs differ on the single a/c line by
   0.001–0.012 Å (defect-arm scan-fit vs spglib idealisation), kept as strict xfails
   pending reviewer sign-off. My independent W5 snippet and the 235 fast
   strained-box tests confirm the build half of W5's done-when holds (0 of 9 refuse;
   216/216 build to the stated cell).
4. **Minor observations (no verdict impact):** lifts of some frames emit numpy
   `RuntimeWarning: Mean of empty slice` / `invalid value encountered in divide`
   (`src/chaord/lift/slab.py:78`, seen on `solutions/lipf6_ec` and the Cu slab);
   A14 itself reports 9 of 35 statement keys have no `spec/examples` example; the
   runner's A11 evidence names only the lift measurement (2.9 s), not the
   build-without-physics number.
5. **Verifier-side deviations (disclosed, forced by the ~1-hour harness kill):**
   acceptance run as 6 `--only` chunks merged by me; fast suite as 7 chunks
   (collection equivalence verified by node-id diff); `--details` redirected to
   scratch to protect the tracked `reports/acceptance_details.md`; two full-run
   attempts were killed at ~60 min with zero criteria reported. No number in this
   report is estimated or copied from another report: everything above is what my
   own runs printed.
