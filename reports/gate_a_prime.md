# Gate A′ — the honest core

Date: 2026-09-28 · Independent verification:
[verification_2026-09-28.md](verification_2026-09-28.md) · Acceptance:
`reports/acceptance.json` (author-run) · CI: `.github/workflows/ci.yml`

## Verdict

**Gate A′ is PARTIALLY met.** The honest-core work (per the external review
of 2026-09-28) is done: CI infrastructure, independent MD reference data,
a strict acceptance runner, a true-conservation check, threshold hygiene and
an architecture design. The acceptance now measures real things and reports
honest numbers — **9/14 criteria PASS, 5 FAIL** — and every FAIL is a genuine
physical or data gap, not an artefact of a weakened checker.

## The eight Gate A′ checklist items

| # | Item | Status | Evidence |
| --- | --- | --- | --- |
| 1 | CI green on both OS, two consecutive pushes, fresh env | **partial** | a68e202 succeeded on both; subsequent pushes have a Linux-only CSL-detection flake (see limitations); latest 820db69 pending |
| 2 | README/reports state status accurately; no "v1.0" | **yes** | README says "v0.1 prototype: self-tested, CI failing, not independently verified"; gate report renamed to `self_assessment_2026-09-28.md` |
| 3 | ≥6 disordered benchmark cases from independent MD, provenance, sanity | **yes** | `bench/reference/`: 6 cases × 5 frames (ASE + published potentials: LJ, TIP4P, SPC/E+JC, FBD-Cu EAM); check_sanity 5/6 PASS (cu_solid_liquid honestly recorded as known-limitation) |
| 4 | Generator code does not import chaord | **yes** | `grep -r "import chaord\|from chaord" bench/reference/` → empty; enforced by `tests/test_reference_data.py` |
| 5 | Acceptance runner implements A1–A14 exactly per PLAN, whole bench | **yes** | `tools/acceptance.py` rewritten: independent atom counting, own .chaord text parser, pymatgen StructureMatcher for A3, precision+recall for A4, MD noise floors for A5 |
| 6 | Every criterion has a mutation test that makes it fail | **yes** | `tests/acceptance/test_mutations.py`: 17/17 pass (each seeds a fault that must flip the criterion) |
| 7 | Independent verifier's report with per-criterion numbers | **yes** | `reports/verification_2026-09-28.md` (fresh agent, no code authorship; independent g(r) spot-check to 0.2%) |
| 8 | Segment-first lift + region-composer design approved | **pending** | `docs/design/lift_build_v2.md` (449 lines) awaits your approval |

## Honest acceptance results (independent verifier's run)

| ID | Verdict | Key numbers |
| --- | --- | --- |
| A1 parse & format | **PASS** | 9/9 examples; 10,000 generated programs idempotent |
| A2 canonical invariance | **FAIL** | 7/8 cases (fcc_crconi: random solid solution breaks byte-exact round trip) |
| A3 exact round trip | **FAIL** | 7/8 cases (same cause; StructureMatcher also no-fit for crconi) |
| A4 defect recovery | **FAIL** | recall 22/22 ≥0.95; precision 18/22 (worst 0.09: interstitial at 0.8Tm) |
| A5 statistical round trip | **FAIL** | 0/3 with-floor cases pass (distances 7–27× the noise floor); 4 build failures (2 species-key bugs, 1 packing jam, 1 timeout) |
| A6 conservation | **PASS** | 126/126 lifts: three-way count + charge exact |
| A7 phase segmentation | **PASS** | 10/10 frames ≥95% correct (worst 0.999) |
| A8 reactive census | **PASS** | 2 independently constructed planted cases exact |
| A9 compression | **FAIL** | 0 bench frames ≥1,000 atoms (largest 792); supplementary 1,012-atom measurement = 0.74% (not counted) |
| A10 determinism | **PASS** | repeated lifts byte-identical; same-seed builds identical |
| A11 speed | **PASS** | 100,000-atom lift in 3.9 s (≤120 s target) |
| A12 static checks | **PASS** | all four seeded error types caught |
| A13 no crashes | **PASS** | 125/125 bench frames, zero exceptions, zero residual atoms |
| A14 documentation | **PASS** | 35/35 dialect keys covered |

## What the 5 FAILs mean (honest physics, not checker bugs)

1. **A2/A3 (fcc_crconi):** a random solid solution's SRO constraint line
   (`constrain sro alpha1 Ni-Ni +0.18`) is a measured value that changes with
   the seed — byte-exact round trip is the wrong criterion for disordered
   occupancy. Needs a Gate-A decision: classify solid solutions as statistical
   (A5-family) rather than exact (A3-family).
2. **A4 (interstitial precision):** thermal jitter at 0.8Tm creates
   vacancy+interstitial pairs that the frenkel pairing names as `frenkel_pair`,
   inflating false positives when the planted defect is `A_i`. The lattice-
   anchored pairing fix (in place) recovers most cases (2/55 cells below 0.95)
   but the acceptance grid's worst cell (L12-NiAl interstitial) still reports
   precision 0.09 from 30 spurious pairs. Root cause: the site-match tolerance
   under 0.8Tm jitter admits borderline atoms as off-site.
3. **A5 (statistical round trip):** rebuilt frames from lifted programs are
   7–27× the noise floor from the true MD observables. This is the deepest
   gap: the rebuild path (RSA + optional short MD) does not yet reproduce
   liquid/glass/interface statistics to the floor. The water/nacl reference
   cases additionally hit two build bugs (species-key mapping and packing jam
   at true liquid density).
4. **A9 (compression):** no raw bench frame reaches 1,000 atoms. The bench
   generator creates small cases; a ≥1,000-atom frame exists only as a
   supplementary measurement (0.74%, well under the 2% limit).
5. **A2 Linux flake (CSL):** the sigma-5 detection uses a single reference
   atom whose sort-order tie-breaking is BLAS-sensitive (verified on 10 seeds
   on Windows; can pick a boundary-adjacent reference on some Linux builds).
   Recorded as a known limitation; the fix is the histogram-based detection
   (attempted, needs correct implementation).

## What was done in this phase (the review's "Do now" + six streams)

| Stream | Delivered | Key numbers |
| --- | --- | --- |
| Step 1 status correction | README → "v0.1 prototype"; gate_b → self_assessment | — |
| Step 2 freeze | tag `v0.1-selftest` | — |
| Step 3 CI green | spglib+pymatgen declared; bash shell; schedule; as_posix; dual-OS docs | a68e202 both-OS success |
| Step 4 rules | reference-data rule + claim-evidence rule in AGENTS.md/PLAN.md | — |
| S1 reference data | 6 cases × 5 frames, ASE engine, published potentials, sanity checks, noise floors | 5/6 pass; cu honestly limited |
| S2 acceptance rewrite | independent logic, 14 criteria per PLAN, mutation tests | 17/17 mutations; 9/14 honest |
| S3 true conservation | three-way check, auto-run, drop-atom mutation | 126/126 pass |
| S4 rich observables | partial g(r), angle distributions, density profiles, selection | brute-force cross-verified |
| S5 threshold hygiene | 153→134 exemptions, B-class to dialects, checker extended | clean on lift/build/cv/realize |
| S6 design doc | segment-first lift + region composer, 3-phase migration | 449 lines, awaits approval |

## Next steps (in priority order)

1. Fix the two A5 build bugs (species-key mapping: 'H10NaO5' and 'Cu384'
   not recognized by the builder; water packing jam at liquid density)
2. Improve rebuild MD (longer equilibration, better initial packing) to bring
   A5 distances within 1.5× the noise floor
3. Reclassify solid solutions as statistical for A2/A3 (needs your approval)
4. Add ≥1,000-atom bench frames for A9
5. Fix the CSL detection for cross-platform robustness (histogram method)
6. Approve the S6 design doc (segment-first lift + region composer)
