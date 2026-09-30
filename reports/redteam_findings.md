# Red-team findings (T0 adversarial stream, external review 3 follow-up)

Date: 2026-09-30 · Suite: `tests/adversarial/` (this report + tests only; no
`src/` file was touched) · Runner: `.venv/Scripts/python.exe -m pytest
tests/adversarial -q` on Windows Git Bash.

Mission: find inputs where **a criterion passes but the result is wrong**, the
class of hole muller-kushner opened last round (A2 atom ordering, A5 wrong
temperature, Σ detector noise). Fourteen criteria interrogated; every finding
below carries a minimal reproduction and measured numbers. Tests marked RED
fail today because the defence is fooled — they must turn green when the gap
is closed. Tests marked GREEN record defences that held (regression assets).
Nothing here weakens, skips or deletes an existing test.

**Totals: 43 tests — 32 RED (fooled), 10 GREEN (defended), 1 XFAIL
(W1-waiting).**

---

## S1 — the evidence class itself is wrong (criterion "passes" on inputs it was never run on)

### F1. A2/A3 substitute perfect rebuilds for the bench's own thermal frames — and on the real frames the round trip fails 10/18 · RED

- **Criterion**: A2 (invariance), A3 (exact round trip) — reported 9/9 PASS.
- **Repro**: `tests/adversarial/test_a02_a03_thermal_frames.py`. `check_a2`/`check_a3`
  both call `_rebuild_crystal(case)` (a perfect zero-temperature frame from ground
  truth) instead of `case["frames"]` — the 5 stored thermal MD frames per case are
  never invariance- or round-trip-tested.
- **Numbers** (stored frames, `lift → build`, seeds/modes as in acceptance):
  - build REFUSED on **10/18** frames: *"system cell [7.23…] is not an integer
    multiple of the fcc lattice vectors [3.608…]"* — fcc_cu (2/2 frames tested),
    bcc_fe 1, rocksalt 1, diamond 1, perovskite 1, crconi 1, cuau 1, hcp 2 (different
    failure, F2).
  - where the build succeeds, lift→build→lift text is byte-identical on **1/18**
    (l12_ni3al frame_0) against the reported "9/9 text byte-identical".
  - rotation invariance on stored frames: fcc_cu frame_0 lifts `a 3.608 A`; the
    same frame after one rigid transform (rotation+translation+re-imaging+
    re-ordering, rule 3) lifts `a 3.452 A` (**−4.3 %**); l12_ni3al 3.572 → 3.518
    (−1.5 %). The lattice fit of a thermal frame is rotation-dependent.
- **Root cause direction**: the printed `a` comes from spglib idealisation of the
  jittered frame while `cell` is copied from the frame; 2×3.608 = 7.216 ≠ 7.23, so
  the lift emits a program its own builder rejects at its 0.01 integer-multiple
  tolerance. Either fit `a` from the cell (or co-optimise), or align the builder's
  tolerance with the lift's own rounding/symprec, and run A2/A3 on the stored
  frames (or a thermal sweep), not only on perfect rebuilds.

### F2. hcp_mg: all 5 stored frames lift to semantic garbage that A6, A9 and A13 all accept · RED

- **Criterion**: A6 (conservation 126/126), A13 (125/125 clean lifts), A9.
- **Repro**: `tests/adversarial/test_a06_a13_hcp_misroute.py`.
- **The garbage** (32-atom thermal hcp Mg supercell, amplitude 0.06 d_NN — the
  same thermal level as the crystal cases that lift fine, orthohexagonal cell
  6.42 × 11.12 × 10.42):

      liquid fluid : all {
        molecules Mg32 1
        state density 1.736 g/cm3
        assert cn 0.0 +- 0.0 cutoff 3.10 A
      }

- **Why every gate stays green**: the census invents the molecule `Mg32`, whose
  formula parse yields exactly 32 Mg — so all three count sides agree exactly
  (A6 three-way match), nothing is unexplained (A13 zero residual, zero
  exceptions), and A9 measures the case's compression. `derive_counts`
  degrades to the two-way check ("unknown molecule Mg32").
- **Root-cause chain** (each link measured): (1) `lift_crystal` raises
  *no prototype matches the standardised structure* — the case's own ground-truth
  `lift_mode: "crystal"` propagates this as a raw **ValueError** (non-ChaordError,
  an A13-class defect by itself); (2) the auto cascade **silently swallows** the
  exception and reroutes; (3) `is_single_phase` sees zero bonds because the metal
  dialect's `cn_cutoff_fluid = 3.10 Å` (Cu-scale) is **below d_NN(Mg) = 3.21 Å**;
  (4) the fluid census names the whole frame one molecule.
- **Fix directions**: prototype recognition for the orthohexagonal hcp setting;
  a dialect invariant "fluid cutoff > largest d_NN of the covered systems";
  honour the ground truth's `lift_mode` in `lift_all_bench_frames` (it is
  currently ignored — everything lifts auto); convert the cascade's swallowed
  exceptions into recorded routing diagnostics.

### F3. A5 wrong-temperature rebuilds PASS the gate — the known open item, now quantified · RED (partly known)

- **Criterion**: A5 statistical round trip, 6/6 PASS.
- **Repro**: `tests/adversarial/a5_temperature_experiment.py` (experiment module
  + cached numbers), `test_a05_temperature_and_averaging.py` (tests, one live
  canary marked slow). Protocol identical to `check_a5`: lift
  `bench/reference/lj_liquid/frame_0.npz`, rewrite `state T`, rebuild with
  `build_program(physics=True)`, judge the median of draws 7/13 against the
  current floor `max(mean, P90)` = gr 0.107 / cn 0.076, gate 1.5×.
- **Numbers**:

  | T | A5 median-of-2-draws | ratio vs floor | gate verdict |
  | --- | --- | --- | --- |
  | 0.52 (×0.8) | gr 0.095 / cn 0.053 | ×0.89 / ×0.70 | **PASS** (wrong) |
  | 0.65 (×1.0) | gr 0.077 / cn 0.045 | ×0.72 / ×0.59 | PASS |
  | 0.8125 (×1.25) | gr 0.131 / cn 0.052 | ×1.22 / ×0.68 | **PASS** (wrong) |

  - systematic (5-frame-averaged) distances: ×0.8 → gr **0.097**, ×1.0 →
    **0.031**, ×1.25 → **0.078** — the wrong-T systematic shift is **2.5–3.1×**
    the correct-T one, i.e. as large as the noise floor itself, and hides inside
    the single-frame microstate noise the gate is calibrated against.
  - **seed luck**: per-seed gr distance at correct T spans 0.058–0.103 (factor
    1.8); no seed currently flips either wrong-T case (max 0.148 < gate 0.161,
    an 8 % margin on this machine — the documented ISA/BLAS draw divergence
    consumes margins of this size).
  - **reference-frame luck** (the harness always judges against `frames[0]`):
    the seed-mean gr distance moves by up to **0.046 = 43 % of the floor** at
    ×0.8 T (0.097–0.143 across the 5 frames); cn by 0.033 (43 % of its floor).
  - **averaging ablation** (`a5_averaging_experiment.py`): the split-half floor
    of the 5-frame-averaged statistic is gr ≈ 0.071, so even an averaged gate at
    1.5× (gr 0.106) **still passes** the ×1.25 systematic (0.078). Five
    decorrelated frames are not enough — noise shrinks only as 1/√k, so ~20+
    frames or a temperature-sensitive observable is needed. Structural
    contributor: the lifted `state T` is dialect metadata, never measured — the
    language cannot currently represent a wrong frame temperature at all.
- **Fix directions**: measure and assert the rebuild's kinetic temperature
  against the reference provenance (cheapest, catches ×1.25 outright); or move
  the observable set to 5+-frame averages on both sides with more decorrelated
  reference frames; keep the P90 floor.

---

## S2 — checker blind spots (a crafted wrong input passes)

### F4. A6: region arithmetic is unaudited for slab geometries · RED

- **Repro**: `test_a06_conservation_blindspots.py::test_a6_slab_region_arithmetic_is_audited`.
- Take the lifted `interfaces/cu_water` program and rewrite its region statement
  `molecules H2O 136` → `molecules H2O 106` (30 waters = **18 % of the water**
  removed from the region's own accounting; conserve line untouched, as a census
  bug in the lift would leave it). `derive_counts` returns *None* ("non-trivial
  region geometry") and the A6 row passes `counts_ok=derivation_ok=True`.
- **Consequence**: a multi-species-interface census bug is invisible to A6's
  three-way claim; the gate report's "126/126 three-way exact" quietly becomes
  two-way for exactly the class (interfaces) the criterion's name claims to cover.
- **Fix direction**: for slab programs, cross-check the molecules line against
  the frame census the way `derive_counts` does for `all`-geometry regions, or
  record the degraded check in the evidence string as a two-way check.

### F5. A6/A12: multivalent-ion charges are outside every charge table · RED

- **Repro**: `test_a06_conservation_blindspots.py` (parametrised + full demo).
- **Numbers**: `_ion_charge("Ca2+") = +1` (also Mg2+, Zn2+, Fe3+ → +1; O2-,
  SO42- → −1): the digits are parsed as stoichiometry (`_ion_composition` → the
  molecule "Ca2") and only the trailing sign run is counted. The ionic dialect's
  `formal_charges` table is monovalent-only; the acceptance's `ION_CHARGES`
  ({Na, Cl, Li, K, F, Cs}) has no Ca/Mg/Fe/O at all — any frame containing them
  passes the acceptance charge check unaudited (both sides read 0).
- **Demonstrated false pass**: program `molecules Ca2+ 5 Cl- 5`, `conserve charge
  0`, frame = 5 bonded Ca–Ca pairs (2.5 Å) + 5 lone Cl →
  `charge_check`: **PASS** narrating *"total charge +0 (Ca2+ 5 x +1, Cl- 5 x -1)
  equals conserve charge 0"*; `conservation_check`: **PASS** (implied
  {Ca:10, Cl:5} on all three sides); acceptance row **PASS** (−5 == −5). The
  physical charge is **+5**. (With the Ca unbonded the same program produces a
  false *fail* of −5 — both directions are wrong.)
- **Fix direction**: read the pre-sign digits as magnitude for the `ion` species
  grammar (`Ca2+` → +2), add the multivalent entries to `ionic.yaml` and to the
  verifier's table, and make `_ion_composition` refuse names whose
  charge-stripped form contains digits (charge-stoichiography ambiguity should
  be an error, not a molecule guess).

### F6. A12: overlap tolerance contradicts the repo's own sanity floor · RED

- **Repro**: `test_a12_a10_a14_statics.py`.
- **Numbers**: lj `overlap_tolerance = 0.7` (σ) **< 0.8 σ** — the hard core
  AGENTS.md/PLAN.md enforce for reference data; a pair at 0.75 σ passes the
  static check while failing the sanity rule (measured: 0.70 σ passes, detail
  "min pair distance 0.750 vs tolerance 0.7"). metal `overlap_tolerance = 0.5 Å`
  absolute = **0.20 × d_NN(Cu)** — a Cu pair 1.78 Å apart (0.70 d_NN,
  catastrophically overlapping) passes.
- **Fix direction**: express the tolerance relative to the system's d_NN (or per
  dialect ≥ 0.8 σ for LJ), matching the sanity rule.

### F7. A1: formatting swallows string semantics · RED

- **Repro**: `test_a01_string_escapes.py`.
- **Numbers/facts**: source `atom X = file "C:\data\x.xyz"` parses to StrVal
  `'C:datax.xyz'` (unrecognized escapes silently drop the backslash);
  `"C:\temp\n.txt"` parses to a string containing a **TAB and a LF**. The A1
  laws (parse==IR, fmt idempotent) hold on the already-corrupted value, so the
  criterion passes while the file reference changed. Windows paths are the
  natural victim (`physics { potential "…" }`, `= file "…"`).
- **Fix direction**: reject non-`\\ \" \n \t` escapes with a line-numbered
  `ChaordSyntaxError`, or preserve unrecognized escapes literally.

### F8. A14: coverage by prose · RED

- **Repro**: `test_a14_prose_coverage.py`.
- **Numbers**: a "reference" made of unrelated sentences ("We cell things
  carefully. …") passes A14 with **35/35 keys covered**. Real drift scenario: an
  entry deleted/renamed while the word survives in prose keeps the gate green.
  The 9 keys with "(no example in spec/examples/)" also show the example half of
  the criterion is reported, not gated.
- **Fix direction**: require an entry-shaped pattern (heading, table row or
  `` `key` `` code span) and gate on the example column too.

### F9. A7: 64.5 % of atoms are outside the judged core · RED

- **Repro**: `test_a07_judged_core_scope.py`.
- **Numbers** (lj_solid_liquid frame_0, 512 atoms, boundary 6.35, d_NN 1.007,
  band 2×d_NN + periodic z-edges): judged core = **182 atoms (35.5 %)**. A
  labeler that flips **every** label outside the core (330 atoms) scores
  accuracy **1.000** on the metric while being 64.5 %-wrong overall. The seeded
  canary (20 % global flips) is caught only because its flips land in the core.
  The exclusion band is physically motivated; the hole is that the metric never
  reports how much of the system it judged — "≥ 95 % correct" silently means
  "≥ 95 % of 35.5 %".
- **Fix direction**: report the judged fraction in the evidence and cap the
  excluded fraction (or judge the band against a fuzzier truth), so the number
  on the gate report states its coverage.

### F10. A13: NaN input segfaults the interpreter — no exception can record it · RED

- **Repro**: `test_a13_nan_crash.py` (runs the lift in a child process).
- **Numbers**: NaN positions or a NaN cell diagonal → native **SIGSEGV, exit
  code 139** (measured both), inside scipy's cKDTree. Not a `ChaordError`
  refusal, not any exception: A13's "zero unhandled exceptions" evidence
  structurally cannot see it, and one NaN frame would take the whole acceptance
  run down. (Zero-length cell axis, by contrast, raises a `ValueError` — visible
  to the harness but not a `ChaordError` refusal.)
- **Fix direction**: validate finiteness of positions/cell at the `Frame`/
  `lift_frame` boundary and refuse with `ChaordError`.

### F11. A9: the ≥1,000-atom gate is evidenced by one perfect crystal · RED (scope) + GREEN (ratio)

- **Repro**: `test_a09_evidence_class.py`.
- **Numbers**: the only gated bench system is `crystals/l12_ni3al` (1,372 atoms,
  a **perfect crystal**, 0.49 %) plus a tiled supplementary of the same kind;
  measured ratios elsewhere: homogeneous liquid lj_liquid_large (2,048) **0.33 %**,
  ionic solution nacl_aq (1,640) 0.51 %, heterogeneous interface tiled 2×2
  (2,048) **0.68 %**, stored lj_solid_liquid (832) 1.50 %. Sub-1,000-atom systems
  are ungated up to **34.29 %** (bcc_fe, 16 atoms).
- **Verdict**: the known "trivial compression" open item is confirmed and
  quantified — heterogeneous and homogeneous classes both sit far below 2 % at
  ≥1,000 atoms (the tiled-interface regression is kept green), but the gate's
  evidence set contains no heterogeneous case at all, so the 2 % number has
  never been demonstrated where compression is hard.
- **Fix direction**: add an ≥1,000-atom heterogeneous bench case (interface or
  surface with adsorbates) to the gated set.

### F12. A4: the metric cannot score mixed-defect cells — though the lift handles them perfectly · RED (metric) + GREEN (lift)

- **Repro**: `test_power_mutations.py::test_a4_mixed_cell_mutation`,
  `test_defended_greens.py`.
- **Numbers**: one L1_2 cell planted with V_Ni 3 + Al_Ni 4 + Ni_i 3 at the
  0.8-Tm amplitude lifts to **exactly** `{V_Ni: 3, Al_Ni: 4, Ni_i: 3}` (a
  perfect lift), but scoring that cell with the acceptance's per-cell
  `_pr_for_cell` yields **P = 0.30 / 0.40 / 0.30, R = 1.00** — each type's
  evaluation counts the other types' true detections as false positives. The
  criterion is therefore only ever demonstrated on single-type plantings not
  because the lift cannot do mixed cells but because the checker cannot express
  them.
- **Fix direction**: score mixed cells against the planted multiset
  (token-exact P/R over the union), and add one mixed cell to the A4 plan.

---

## S3 — defences that held (green regression assets, `test_defended_greens.py`)

- **A4 hot sweep**: a defect-free fcc-Cu frame swept to the 1.0-Tm-equivalent
  amplitude (promise: 0.9 Tm) reports **0 false defects** at every step
  (0.80/0.85/0.90/0.95/1.00 Tm measured).
- **A4 divacancy**: two *adjacent* vacancies (the suite always plants separated
  ones) lift correctly as `V_Cu 2`.
- **A4 mixed cell**: perfect detection (F12's green half).
- **A10 hash-seed determinism**: `PYTHONHASHSEED` 0/1/42 in separate processes →
  **identical sha256** over the lifted text of 5 diverse cases.
- **A12 lattice-mismatch boundary**: exact fit (7.23 = 2×3.615) correctly
  allowed; +1.0 % and larger mismatches caught.
- **A3/A9 survivors**: the one thermal frame that builds (l12_ni3al frame_0)
  round-trips byte-identically; the tiled 2,048-atom interface compresses to
  0.68 %.

## W1-waiting (not counted red; parallel streams own the fix)

- **SRO hot frame** (`test_power_mutations.py::test_sro_hot_frame_power_mutation`,
  `xfail(strict=False)`): A3's species check for random solutions runs on
  perfect rebuilds; on the *stored thermal* fcc_crconi frame the rebuild's
  species arrangement measures **max |Δα| = 0.393** (seed 5) vs a perfect-frame
  relabel floor of order 0.1. Waiting W1-A/W1-B (SRO/defect lift rework) — the
  number is recorded as their target, not as a red test against in-flight code.

## Regression command and expected shape

    python -m pytest tests/adversarial -q            # ~50 s (fast set ~8 s
                                                     #  with -m "not slow")

Today: **32 RED, 10 GREEN, 1 XFAIL** — every RED test is a measured hole; each
turns green exactly when its criterion stops being fooled.
