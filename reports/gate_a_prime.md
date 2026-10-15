# Gate A′ — declared 2026-10-14

**Gate A′ is closed.** Decision D16 (Review 8, `docs/reviews/open_items_v2.md`)
declared it on 2026-10-14: the twelve work orders W1–W12 merged, decisions
D1–D15 implemented, a clean-runner acceptance **14/14** on the final code
state, and a fresh verifier's report committed. Nothing on the gate is
blocked. The two items handed to the reviewer *with* the gate were settled
in Review 9: the W5 strained-box gap is **closed** (X2, `f7aaaa3`: the 94
registered strict-xfail configurations now pass, 216/216 identity configs,
the FOILED registry is empty) and the Kob–Andersen sanity bounds are
**approved** (D18: P* > −0.5 and per-quench energy flatness ≤ 0.015; full
measured pressure range −0.23..−0.01 recorded in
`bench/reference/ka_glass/provenance.json`).

Evidence of record:

- `reports/acceptance.json` — the clean-runner artifact (run
  [37868982335](https://github.com/WhiteCrosstheRiver/Chaord/actions/runs/37868982335),
  **14/14 on `ba6b594`**, 2026-10-09). The acceptance table below is
  generated from it.
- [verification_2026-10-08.md](verification_2026-10-08.md) — the **latest**
  independent verifier (fresh agent, no code authorship: A1–A14
  independently reproduced 14/14; its one red finding, unregistered strict
  xfails, was fixed and registered in `ba6b594`). The 2026-09-28 and
  2026-09-30 verifier reports are earlier dated records.
- [reports/history/](history/) — superseded sections, per D22. The full
  Review-3/4-7/8 scorecards and the pre-gate phase history live in
  [history/gate_a_prime_v1_2026-10-09.md](history/gate_a_prime_v1_2026-10-09.md);
  decision texts D1–D38 live in the dated briefs under `docs/reviews/`.

## The eight checklist items (current evidence)

| # | Item | Status | Current evidence |
| --- | --- | --- | --- |
| 1 | CI green on both OS, fresh env | **yes** | push runs green on both OS at `ba6b594` (run 37868981153, 2026-10-09); the `workflow_dispatch` clean-machine run on `ba6b594` (37868982335) is fully green — acceptance 14/14 plus the slow statistical suite — and the scheduled nightlies on `ba6b594` repeated it (38129284799 on 10-11, 38235272198 on 10-12) |
| 2 | README/reports state status accurately; no "v1.0" | **yes** | README status rewritten current (X3, ≤ 10 lines, D22); the pre-verification self-assessment stays separate (`self_assessment_2026-09-28.md`) |
| 3 | ≥ 6 disordered benchmark cases from independent MD, provenance, sanity | **yes** | `bench/reference/`: 8 stored cases × 5–15 frames from independent MD (ASE engines, published potentials: Kob–Andersen 1995 80:20 glass N = 2000 × 3 quenches, TIP4P, SPC/E+JC, FBD-Cu EAM); the monatomic `lj_glass` is retired as the A5 glass case (D9) and kept as recorded provenance; sanity green on every case except `cu_solid_liquid` (recorded known-limitation) |
| 4 | Generator code does not import chaord | **yes** | `grep -r "import chaord\|from chaord" bench/reference/` → empty; enforced by `tests/test_reference_data.py` |
| 5 | Acceptance runner implements A1–A14 per PLAN, whole bench | **yes** | `tools/acceptance.py`: independent atom counting, its own `.chaord` parser; A3's gate is the translation-aligned assignment check (decision D6); A5 floors from decorrelated frame pairs, estimator max(mean, P90) (D15). Open (X4): 20 draws for A2/A8, per-arm A11 timings |
| 6 | Every criterion has a mutation test that makes it fail | **yes** | `tests/acceptance/`: 41 tests — 23 mutation canaries plus noise-floor, Kob–Andersen-A5 and O2/O3 pins |
| 7 | Independent verifier's report with per-criterion numbers | **yes** | latest: [verification_2026-10-08.md](verification_2026-10-08.md) (A1–A14 independently reproduced 14/14); earlier dated records: 2026-09-28, 2026-09-30 |
| 8 | Segment-first lift + region-composer design approved | **yes** | approved 2026-09-29 with 3 binding changes; stage 1 landed as `lift/pipeline.py` behind `mode="pipeline"` |

## Acceptance results

<!-- BEGIN GENERATED acceptance results: tools/gen_gate_report.py, do not edit by hand -->

Artifact: clean-runner run [37868982335](https://github.com/WhiteCrosstheRiver/Chaord/actions/runs/37868982335) on `ba6b594` (2026-10-09T02:05:20+0000) -- **14/14 criteria PASS**; the runner's OS was `Linux 6.17.0-1022-azure`, working tree clean. The table below is generated from this artifact (`tools/gen_gate_report.py`); the full per-case tables are in [acceptance_details.md](acceptance_details.md).

### Criteria

| criterion | verdict | key evidence (first clauses; full evidence in acceptance_details.md) |
| --- | --- | --- |
| A1 parse and format | **PASS** | 10/10 spec examples parse; property laws (parse==IR and fmt idempotent) hold on 10000 generated programs |
| A2 canonical invariance | **PASS** | 9/9 crystal + random-solution cases; 18/18 rigid transforms (rotation + translation + re-imaging + re-ordering, 2 per case) lift to byte-identical text |
| A3 exact round trip (ordered) | **PASS** | 9/9 lift-build-lift texts byte-identical; 9/9 structure fits original vs rebuilt ON THE STORED THERMAL FRAMES; ... |
| A4 defect recovery | **PASS** | 26 host x defect-type x temperature cells (3 hosts, all four K-V kinds plus one mixed-kind cell on fcc-Cu and L12-NiAl); ... |
| A5 statistical round trip | **PASS** | pass rate 6/6-with-floor; fluid: 4/4 (target >= 90%); interface: 1/1 (target >= 90%); glass: 1/1 (target >= 80%); 6 cases without a floor on record: no-floor: skipped (synthetic frame); ... |
| A6 conservation | **PASS** | 126 lifts checked: frame count == conserve line on 126/126; verifier's own region arithmetic pins or cross-checks 81 programs; charge consistent on 126/126; ... |
| A7 phase segmentation | **PASS** | 10/10 interface frames (2 cases x 5 frames; interface band of 2 x d_NN excluded around every interface plane, d_NN = verifier's median nearest-neighbour distance) labelled >= 95% correct; ... |
| A8 reactive census | **PASS** | plan 0: census {'H2O': 40, 'HO': 9, 'H': 9} == planted {'H2O': 40, 'HO': 9, 'H': 9}; plan 1: census {'H2O': 25, 'HO': 5, 'H': 7} == planted {'H2O': 25, 'HO': 5, 'H': 7}; ... |
| A9 compression | **PASS** | 5 measured systems with >= 1,000 atoms (bench frames + independent-MD reference frames); worst ratio 0.69% (reference/nacl_aq, 1640 atoms); worst crystal-class 0.50% (crystals/l12_ni3al, 1372 atoms); ... |
| A10 determinism | **PASS** | repeated lift of bench defect frame byte-identical: True; same-seed builds identical coordinates: True |
| A11 speed | **PASS** | lift_frame(mode=fluid) on 100,000 atoms took 3.2 s (target <= 120 s); input = fcc lattice at rho 0.85 with 0.10 x a jitter (thermally disordered, not equilibrated MD); ... |
| A12 static checks | **PASS** | lattice_mismatch: caught; impossible_density: caught; charge_imbalance: caught; overlap_0.1sigma: caught |
| A13 no crashes | **PASS** | 125/125 bench frames lift without any exception; 0 atoms placed in residual blocks across successful lifts; 60 frames carry routing diagnostics (recorded lift_mode refused or names a bench category; ... |
| A14 documentation | **PASS** | 35 distinct keys across dialect YAMLs; reference.md covers 35/35 with structured entries (code span / heading / table row + an actual spec/examples line cited; prose is not coverage); ... |

### A5 statistical round trip, per case (x floor; gate 1.5x)

| case | category | cn_tv xfloor | gr_rms xfloor | backend |
| --- | --- | --- | --- | --- |
| reference/ka_glass | glass | x0.3 | x0.9 | lj |
| reference/lj_liquid | fluid | x0.6 | x1.2 | lj |
| reference/lj_liquid_large | fluid | x1.1 | x0.7 | lj |
| reference/lj_solid_liquid | interface | x0.4 | x1.1 | lj |
| reference/nacl_aq | fluid | x1.1 | x1.3 | classical |
| reference/water_tip4p | fluid | x0.7 | x1.2 | classical |
| fluid/water_box15 | fluid | no floor on record (skipped, honestly) | - | classical |
| fluid/ar_gas_box25 | fluid | no floor on record (skipped, honestly) | - | classical |
| fluid/n2_box22 | fluid | no floor on record (skipped, honestly) | - | classical |
| glass/lj_glass_rho085 | glass | no floor on record (skipped, honestly) | - | lj |
| interface/lj_solid_liquid | interface | no floor on record (skipped, honestly) | - | lj |
| interfaces/cu_water | interfaces | no floor on record (skipped, honestly) | - | eam |

<!-- END GENERATED acceptance results -->

How to read A5: a physics rebuild is judged against the reference's own
frame-pair variability — per-observable median over rebuild draws, floors
from decorrelated frame pairs (cross-quench pairs for the glass), gate
1.5×. Full pair lists: `reports/noise_floors.json`.

## Post-gate record

The declaration artifact stands for `ba6b594`. Merged since (each with its
own red-first suites; the clean runner re-runs only on a new dispatch):

| change | commit | effect |
| --- | --- | --- |
| X1 clean thermal slabs | `02ecce7` | the W10 fake-adatom regression fixed: 0/20 clean Pt(111)/Pt(100) slabs print an adsorbate at every tested amplitude |
| X2 lift-side snap | `f7aaaa3` | W5 closed (D17): one `box_lattice_constant` shared by every lift arm and the builder; 216/216 strained-box identity configs, FOILED registry empty |
| Y1 research record | `1996272` | the pilot retitled "Pilot at a saturating field", with the measured non-equilibration and withdrawn readings on record |
| Y2, Y4 research lane | `1e4726a`, `2fae03a` | `dipole_p1` measured blind of `field=`; the LAMMPS reader wraps real dumps and raises on ambiguous masses |

The research pilot (`research/cnt_electrolyte/RESULTS.md`): at a saturating
field (0.2 V/Å) the pipeline runs end to end and the P1 statement separates
field on from off beyond its noise floor (×3.3–4.8 mean); it does **not**
show weak-field sensitivity, an equilibrated zero-field orientation, or
independent runs. The field ladder and the D32 go/no-go are the active work
(Review 10, `docs/reviews/open_items_v4.md`).

## Known limitations (current)

1. `cu_solid_liquid` is excluded from A5 scoring and recorded "not liftable"
   in A9: the FBD-Cu coexistence MD's solid density drifts as phase fractions
   exchange volume (asserted in the sanity-CLI test).
2. A14: 9 dialect keys have no `spec/examples` example (atom, chirality,
   dialects, lift_version, nanotube, note, reconstruction, source, stacking)
   — reported, nothing to cite (X9; `chirality`/`nanotube` arrive with X8).
3. A2 judges 2 rigid transforms per case and A8's site half one planted
   frame; D21 (X4, open) raises these to 20 draws with a clean-slab
   false-positive check.
4. A11 times only the fluid arm (X4, open: per-arm table at 10⁴/10⁵ atoms).
5. X5 (histories fail closed, D19) and X6 (dense-glass statistics, D27) are
   the S1 stream's open work orders.
6. cu_water cross-relaxation: no published Cu–water cross potential ships;
   each region relaxes under its own backend (stated in the program's
   provenance).

## History

`reports/history/` holds the superseded sections (D22). This file is a
current-state document: every acceptance number above is generated from
`reports/acceptance.json` by `tools/gen_gate_report.py` (`--check` verifies
freshness), and every other claim cites its dated source.
