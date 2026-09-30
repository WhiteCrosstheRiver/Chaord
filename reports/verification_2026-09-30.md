# Independent verification report — Review 3 follow-up — 2026-09-30

**Verifier role.** Fresh agent with no authorship of any `src/`, `tests/`,
`tools/` or `bench/` code in this repository (external review 3's final
Done-when item). Scope: rerun the full acceptance suite, the mutation canaries
and the adversarial suite on this machine, and independently spot-check two
criteria plus one hot-frame round trip, without trusting the acceptance
runner's own arithmetic. No repository file was modified by the verifier
except this report; nothing was committed or pushed.

**Environment.** Windows 10 (Git Bash), interpreter `.venv/Scripts/python.exe`
(32-core machine). All acceptance output was written to a temp directory
outside the repo; the checked-in `reports/acceptance*.json/md` and
`docs/reference_generated.md` were checksum-verified untouched after every
run (git status clean throughout — `check_a14` regenerates the generated doc
and reproduced it byte-identically).

## 1. Commands executed and top-line results

| # | Command | Result |
| --- | --- | --- |
| 1 | `.venv/Scripts/python.exe tools/acceptance.py -o <tmp>/acc_verify.json --details <tmp>/acc_verify_details.md` | **14/14 criteria PASS**, exit code 0, ≈ 30 min wall |
| 2 | `.venv/Scripts/python.exe -m pytest tests/acceptance -q` | **24 passed** (22 mutation canaries + 2 noise-floor tests, incl. the slow cross-quench test) in **52 min 28 s** |
| 3 | `.venv/Scripts/python.exe -m pytest tests/adversarial -q -rX` | **41 passed, 2 xfailed, 0 XPASS** in 14 min 47 s |
| 4 | Spot check A5 temperature power (three scoped `check_a5` runs, §3.1) | correct T **PASS**; temp_lo and temp_hi **FAIL** — power confirmed |
| 5 | Spot check A4 mixed cells, independently re-planted (§3.2) | planted/detected multisets **exact**, P = R = 1.000, two seed regimes |
| 6 | Hot-frame round trip `fcc_cu/frame_0.npz` (§3.3) | lift → build → lift **byte-identical** |

## 2. Per-criterion verdicts (verifier's own run)

| ID | Verdict | Key numbers (verifier's run) |
| --- | --- | --- |
| A1 parse & format | **PASS** | 10/10 spec examples; parse==IR and fmt-idempotent on 10,000 generated programs |
| A2 canonical invariance | **PASS** | 9/9 cases; 18/18 rigid transforms byte-identical |
| A3 exact round trip | **PASS** | 9/9 lift-build-lift byte-identical; 9/9 StructureMatcher fit (species-blind + Warren-Cowley for random solutions) |
| A4 defect recovery | **PASS** | **26/26** cells P ≥ 0.95 and R ≥ 0.95 (all 26 at P = R = 1.000), incl. the two mixed-kind cells (fcc-Cu, L12-NiAl) × 2 temperatures |
| A5 statistical round trip | **PASS** | 6/6 with-floor cases ≤ 1.5× (fluid 4/4, interface 1/1, glass 1/1); 6 synthetic no-floor cases skipped honestly; ratios: lj_glass ×0.3/×0.4, lj_liquid ×0.5/×0.6, lj_liquid_large ×0.6/×0.7, lj_solid_liquid ×0.5/×0.9, nacl_aq ×1.1/×0.9, water_tip4p ×0.7/×1.1 |
| A6 conservation | **PASS** | 126/126 lifts three-way exact; verifier region arithmetic pins/cross-checks 81 programs; charge consistent 126/126 |
| A7 phase segmentation | **PASS** | 10/10 frames ≥ 95 % (worst 0.999, cu_water frame 3); judged fraction disclosed (0.355–0.395 lj, 0.854–0.874 cu_water), core-scope gate held |
| A8 reactive census | **PASS** | 2 independently constructed planted cases exact: {H2O 40, HO 9, H 9}, {H2O 25, HO 5, H 7} |
| A9 compression | **PASS** | 4 gated ≥ 1,000-atom systems; worst 0.55 % (reference/lj_solid_liquid, 2,304 atoms, heterogeneous); crystal-class worst 0.49 % |
| A10 determinism | **PASS** | repeated lift byte-identical; same-seed builds identical |
| A11 speed | **PASS** | 100,000-atom fluid lift in 6.6 s (≤ 120 s target) |
| A12 static checks | **PASS** | all four seeded error types caught |
| A13 no crashes | **PASS** | 125/125 bench frames, zero exceptions, zero residual atoms, 61 routing diagnostics recorded |
| A14 documentation | **PASS** | 35/35 keys structurally covered; example citation gated 26/35 (9 keys have no example anywhere in spec/examples/) |

**Total: 14/14 PASS.** Every evidence string matches the checked-in
`reports/acceptance.json` on every criterion (only difference anywhere: A11
6.6 s here vs 6.5 s recorded — timing jitter); the A5 per-case ratios are
identical to the checked-in details (seeded rebuilds are deterministic).

## 3. Independent spot checks

### 3.1 A5 temperature power (lj_liquid_large, 2,048 atoms, 10 reference frames)

Three scoped runs of the acceptance entry point; verdicts recomputed by the
verifier from the distances/floors parsed out of the row notes (not taken from
the runner's status bit):

| run | program T | cn_tv | gr_rms | my gate (≤ 1.5× floor) | verdict |
| --- | --- | --- | --- | --- | --- |
| clean | 0.72 (provenance T*) | 0.010 vs 0.017 (×0.6) | 0.022 vs 0.033 (×0.7) | both INSIDE | **PASS** |
| temp_lo | 0.576 = 0.8×T* | 0.032 (×1.9) | 0.080 (×2.4) | both OVER (gates 0.0255 / 0.0495) | **FAIL** |
| temp_hi | 0.9 = 1.25×T* | 0.044 (×2.6) | 0.064 (×1.9) | both OVER | **FAIL** |

All three runs: backend lj, md_steps 7400, seeds [7, 13, 29], reference =
mean over frames 0–9, rebuild ≈ 290–300 s per 3-draw run. The T values really
moved (0.576 / 0.9, not the untouched 0.72), the mutated notes are tagged
`MUTATED(T x0.8)` / `MUTATED(T x1.25)`, and the numbers reproduce the
red-team F3 status (×2.4/×1.9 and ×1.9/×2.6) — the criterion carries real
temperature evidence on this case.

### 3.2 A4 mixed-lattice cells (fcc-Cu and L12-NiAl)

Hand-derivation from the planting logic: fcc-Cu mixed = perfect fcc 3×3×3
(108 Cu) minus 3 mutually separated vacancies plus 3 Cu interstitials at
0.60 d_NN off parents ≥ 2 d_NN from every vacancy site → multiset
{V_Cu: 3, Cu_i: 3} on 108 atoms; L12-NiAl mixed = L1_2 4×4×4 (256 atoms:
Ni 192 / Al 64) with 3 V_Ni + 4 Al_Ni (on surviving Ni sites, away from
vacancies) + 3 Ni_i → {V_Ni: 3, Al_Ni: 4, Ni_i: 3}. The verifier re-wrote
the planter from scratch (own numpy, own greedy separation, own regex parse
of `defect X count N`, own P/R arithmetic) and ran it twice:

| regime | fcc-Cu mixed (room / 0.8 Tm) | L12-NiAl mixed (room / 0.8 Tm) |
| --- | --- | --- |
| acceptance's exact seeds | detected == planted, P = R = 1.000, tp/fp/fn 6/0/0 (both temps) | detected == planted, P = R = 1.000, tp/fp/fn 10/0/0 (both temps) |
| verifier-chosen fresh seeds | exact match, both temps | exact match, both temps |

This reproduces the A4 table rows exactly and shows the result is not tuned
to the acceptance seeds.

### 3.3 Hot-frame round trip (red team F1 fix evidence)

`bench/data/crystals/fcc_cu/frame_0.npz` (32 Cu, stored thermal MD frame) →
lift → build (seed 5) → lift: **byte-identical** (335 bytes both, sha256
prefix `a851e7f21fba8af1`); the lift of the raw frame is itself
deterministic. The lifted program prints `cell 7.230` with `a 3.615 A`
(2 × 3.615 = 7.230 exactly) — precisely the cell/lattice consistency F1
found broken (a 3.608 vs cell 7.23) before the fix.

## 4. FOILED registry and adversarial suite

`tests/adversarial/conftest.py` carries exactly **one** FOILED id:
`test_a12_lj_overlap_tolerance_covers_the_sanity_hard_core` (F6, the LJ
`overlap_tolerance = 0.7 σ` vs the 0.8 σ sanity hard core — dialect-side,
pending that stream's approval), registered `xfail(strict=True)`. The suite:
41 passed + 2 xfailed (the strict F6 entry plus the non-strict W1-waiting
SRO marker), **0 XPASS** — with the strict marker an XPASS would have failed
the run, so a fix cannot silently bypass its own adversarial test. The
formerly-RED acceptance-side tests (F3/A5, F4, F5-O, F8, F9, F11, F12,
F2/lift_mode) all pass today as green regression assets, matching
`reports/redteam_findings.md`.

## 5. Honest observations (discrepancies and caveats found by this verification)

1. **`gate_a_prime.md` A4 line is stale.** Its table still says "22/22 cells";
   the runner and grid today are **26/26** (the two mixed cells were added
   post-Review-3, as `redteam_findings.md` F12 correctly records). Direction
   of the staleness is conservative (more cells pass than claimed), but the
   gate report should be updated when the Review-3 scorecard lands.
2. **The W1-waiting SRO xfail is vacuous in its current form.**
   `test_power_mutations.py::test_sro_hot_frame_power_mutation` dies with
   `NameError: name 'np' is not defined` (line 107; no numpy import in the
   module) *before* reaching its assertion — verified by calling the test
   body directly. It xfails for the wrong reason: the recorded number
   (max |Δα| = 0.393 vs relabel floor ~0.1) cannot currently be re-derived by
   running the test. Not a false PASS (it is a waiting marker, not a gate),
   but the marker has no measurement content until the import is restored.
3. **A2/A3 acceptance evidence still runs on perfect `_rebuild_crystal`
   frames**, not the bench's stored thermal frames; thermal-frame invariance
   and round-trip are guarded by the adversarial suite
   (`test_a02_a03_thermal_frames.py`, green in §4) rather than by the
   criteria themselves. My §3.3 round trip independently confirms the fix on
   fcc_cu frame_0.
4. **PLAN A8's second half ("adsorption sites ≥ 90 % correct") is not
   exercised** by `check_a8` (census half only). Unchanged since the
   2026-09-28 verification; still an open coverage item.
5. **Canary count bookkeeping:** the gate report says "22/22" mutation
   canaries; `tests/acceptance` collects 24 items today (22 in
   `test_mutations.py` + 2 noise-floor tests, one slow). All 24 pass.
6. **A5 temperature evidence rests on the ≥ 2,000-atom case** — the 500-atom
   lj_liquid does not separate at ×0.8 T (documented honestly in the runner
   and red-team F3; confirmed by the recorded numbers). Fine as disclosed,
   worth keeping on record.
7. Machine variance: A11 6.6 s here vs 3.7 s in the gate table (both far
   inside the 120 s target); everything else reproduced exactly.

## 6. Verifier statement

I wrote no code in this repository and modified no repository file except
this report. All verification scripts (A4 re-planter, A5 temperature driver,
round-trip checker) were written to the system temp directory outside the
repo and are quoted in substance in §3. Everything in §1–§4 comes from my own
runs on this machine today; the repo tree was verified clean (git status,
checksums) after each run. Nothing was committed or pushed.

---

Independently verified by a fresh agent (no code authorship); 2026-09-30
