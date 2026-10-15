# Review 10 · Chaord repo

Oct 15, 2026 · @muller kushner

## Verdict

Your agents built the whole research pipeline overnight and its numbers are real, but the first result does not yet test what your paper needs. Nine commits (Oct 14, 19:42 to Oct 15, 01:38) added a LAMMPS reader, cylinder, nanotube and field statements, dipole and radial CVs, PC and EMC templates, four LAMMPS runs and a classifier. The runs put LiPF6 in EC:EMC inside (10,10) and (16,16) tubes, with the field off and at 0.2 V/Å; the classifier tells field-on frames from field-off frames at 100%.

I recomputed the dipole order P1 from your raw dumps with my own code. All four means and spreads match theirs to the third decimal.

Four things keep it from being the result:

- **The field saturates the solvent.** At 0.2 V/Å an EC dipole (6.06 D in this model) gains about 10 kT. Free dipoles would reach P1 = 0.90 for EC and 0.38 for EMC; the runs give 0.74–0.94 and 0.38–0.40. Any orientation measure separates that, and your MLP is about weak fields.
- **Orientation never equilibrated.** With no field, P1 must average to zero by symmetry, yet RESULTS.md reads the offsets (−0.12 and +0.15) as confinement aligning the solvent. In the 500 ps after heating, 7 of 13 molecules (narrow tube) and 38 of 61 (wide tube) never reverse their axial dipole. The noise floors come from 21 frames of one 100 ps window per state, so they are too small.
- **The lift is told the answer.** It prints the P1 statement only when the caller passes the field. None of the 42 field-off programs carries it, and the classifier reads the absence as P1 = 0. With each frame's measured P1 instead, accuracy is still 100% at this field: the separation is real here, but at weak fields this encoding would compare noise with a constant.
- **The library cannot read real tube frames yet.** The script assembled every program by hand from the frozen tube in `final.data`. Chirality inference succeeds 0 of 20 times with 0.05 Å of thermal noise, with 6-digit dump text, in a 4-period DFT-sized cell, and for the (16,16) tube in its own cell. The repo reader refuses the project's own production dump.

The agents met the bar I set in Review 8, as written. That bar was too weak: it fixed no field strength, did not require a blind lift and counted frames instead of runs. Brief v3 tightened part of it, but v3 never reached the agents. It is not in `docs/reviews/`, AGENTS.md still points to the v1 and v2 briefs, and the commits work from Review 8.

So Review 9's fixes have not started. On `2990d08` a clean Pt(111) slab still prints a fake adatom in 9 of 20 draws, and a quench that asks for 190,000 steps still runs 20,000.

The direction correction: from now on the research test runs on your model at your field strengths, lifts every frame blind, and judges ensembles from independent runs. I made 11 more decisions (D28–D38). The attached brief v4 turns them into 9 new work orders and keeps v3's.

## Progress

Chaord is a little past halfway to a publishable tool. The core is verified and the research pipeline now runs end to end, but the measurement a paper would rest on has not been made.

| Milestone | Status | What is left |
| --- | --- | --- |
| Core: language, compiler, decompiler, criteria A1–A14 | Done; Gate A′ on Oct 14 | Review 9's regression and gaps (X1, X2, X5) |
| Read your simulations | Started overnight | Real dumps, thermal and short tubes, tube plus liquid in the library (Y3, Y4) |
| First research result | Pilot at a saturating field | A blind, weak-field test with independent runs (Y6–Y8) |
| Converged reference numbers | Not started | Glass history, liquid floors, relaxed defects (X7) |
| Laya loop: DFT labels saved | Waits on the go/no-go | R4 |
| Records current | Not started | README, gate report, RESULTS.md (X3, Y1) |

The agents' speed is not the bottleneck: they built the pipeline in about six hours. The bottlenecks are decisions reaching the repo and runs reaching your cluster.

## Scorecard

None of brief v3's work orders has started, because the agents never had it. The research items exist, with gaps, and the first result is a pilot.

| Item | Status | My check on `2990d08` |
| --- | --- | --- |
| v3 X1 fake adatom on clean slabs | Not started | 9 of 20 clean Pt(111) slabs print an adsorbate |
| v3 X2 strained-box snap | Not started | The 94 configurations are still registered as expected failures |
| v3 X3 records | Not started | README and gate report unchanged since `43d683f` |
| v3 X4 acceptance draws | Not started | A2 still judges 2 transforms per case |
| v3 X5 histories | Not started | A quench that asks for 190,000 steps runs 20,000 |
| v3 X6 glass statistics | Not started | `cn` cutoff 1.19 σ and `ring_mode 3` unchanged |
| v3 X7 convergence | Not started | — |
| RA LAMMPS reader | Done, with gaps | Reads data files and triclinic or NPT dumps; refuses the project's own `prod.dump`; no guard for ambiguous masses |
| RB cylinder, nanotube and field statements | Done for ideal tubes | No lift arm calls the nanotube inference; real frames are refused (problem 5) |
| RC dipole and radial CVs | Half done | P1 matches my code; the radial profile crashes on tube cells |
| RD PC and EMC templates | Done | Standard bond lengths; the older EC template is not EC (problem 11) |
| RE four LAMMPS runs | Done, as a pilot | One run per state, 100 ps of production, field 0.2 V/Å |
| RF first result | Pilot | 100% at a saturating field; the lift is not blind; floors from one run |
| CI budget fix (`4c36697`) | Done | The NaCl kernel test now uses its size-aware budget, as D8 intended |

## What is good

The research record is honest, reproducible and well engineered. These habits are worth keeping.

| Strength | Evidence |
| --- | --- |
| Numbers you can trust | Every number in RESULTS.md regenerates from committed data. My own P1 code on the raw dumps matches all four means and spreads to 0.001 |
| House rules kept | The generator never imports chaord; each run has a provenance file with the dump's sha256; sanity and static checks ran on every frame |
| Bugs caught on first execution | The EMC ethoxy O sat 0.68 Å from the carbonyl O; velocities were made at 300 K before a ramp from 10 K; LAMMPS rejects single-atom rigid bodies; frozen tube atoms diluted the temperature. All four were fixed and recorded |
| A real control | The same classifier on the field-independent statements scores 64%, and chance across diameters, so the separation is P1's |
| Honest negatives | RESULTS.md lists the radial-CV crash, the degenerate `cn` and the chirality limits, and states the presence caveat itself |
| The field is stated, not inferred | `state field` is caller-supplied and printed as assumed, following the W11 rule |
| Exact, fail-closed inference | On ideal long tubes, armchair, zigzag and chiral (n, m) come back exactly, behind consistency gates; otherwise it refuses rather than guess |
| A careful CI fix | A Windows timing failure was traced to machine speed and fixed with the case's real budget, not by loosening the limit |

## What is wrong

Eleven problems, most severe first. Four are P0: each one alone stops the first result from counting.

| # | Severity | Problem | Evidence | Fix |
| --- | --- | --- | --- | --- |
| 1 | P0 | Brief v3 never reached the agents | Not in `docs/reviews/`; AGENTS.md points to the v1 and v2 briefs; the commits cite Review 8; X1–X7 untouched | Do now |
| 2 | P0 | The field saturates the solvent | 0.2 V/Å is 2×10⁹ V/m: about 9.8 kT per EC dipole (6.06 D). Free dipoles: P1 0.90 (EC) and 0.38 (EMC); runs: 0.74–0.94 and 0.38–0.40. `params.py` chose it so PF6⁻ drift shows within 100 ps | D30, Y7 |
| 3 | P0 | Orientation never equilibrated, and a zero-field offset was read as physics | At zero field P1 must average to 0. Over 500 ps, 7 of 13 and 38 of 61 molecules never reverse; each molecule's 500 ps mean of cos θ spreads by 0.61 and 0.57, where free rotation gives about 0.1. Field-on runs started at 10 K with the field on; the wide tube rose from 0.46 at the end of the ramp to about 0.6 | D28, Y1, Y6 |
| 4 | P0 | The lift is told the answer | The same frame prints no P1 with `field=None`, and `assert field_dipole_p1 -0.213` with a stated field of 0.0. None of the 42 field-off programs carries P1 | D29, Y2 |
| 5 | P1 | Real tube frames do not lift | 0 of 20 with 0.05 Å noise, with 6-digit text, for (10,10) in 4 periods, and for (16,16) in its 41.71 Å cell; 9 of 20 for an (8,8) cell placed anywhere. The radius gate is exact to 1e-6; the axis comes from the covariance of wrapped coordinates | D33, Y3 |
| 6 | P1 | Tube plus liquid is composed by a script | No lift arm calls the nanotube inference; in my session, `lift_frame` refuses an ideal (10,10) tube under three carbon-dialect stacks. The script finds the tube by molecule ID and triples the cell | D35, Y3 |
| 7 | P1 | The reader refuses real dumps | `read_lammps_dump` raises on `cnt10_field0/prod.dump`, whose atoms sit up to 0.28 Å outside the box, as LAMMPS writes them between re-neighbourings. Unwrapped columns are refused; united-atom CH2 (14.027) would silently become N (14.007) | D34, Y4 |
| 8 | P1 | One run per state, judged per frame | Leave-one-out over 21 correlated frames of one run per state; only the cross-diameter test holds out a run | D31, Y7, Y8 |
| 9 | P2 | Electrolyte observables missing or broken | The radial profile raises ZeroDivisionError; `cn 0.0 ± 0.0` in 55 of 84 frames; `state density 0.357 g/cm3` printed for a liquid at 1.28 g/cm³ in the pore; no Li⁺ solvation shell or ion pairing | D36, Y5 |
| 10 | P2 | The pilot model is not paper-grade | Thesis parameters; EMC charges copied from EC; Lorentz–Berthelot instead of OPLS geometric mixing; full ±1 ion charges; a guessed DOI ("…-ish"); a separate Nosé–Hoover on 2 or 6 Li⁺; a chosen filling read as "confinement-compressed" | D37, Y6 |
| 11 | P2 | The EC template is not EC | A planar ring with C–C 1.43 Å (EC: about 1.52); hydrogens in the ring plane at H–C–H 70° and H–H 1.25 Å (about 109° and 1.78 Å in EC) | Y9 |

## The zero-field runs drift longer than the window

&#91;embedded content: recomputed from research/cnt\_electrolyte/runs/\*/equil.dump and prod.dump · 4 runs, 50–550 ps\]

In the narrow tube the zero-field P1 swings from +0.25 to −0.24; in the wide tube it stays between +0.03 and +0.20 for all 500 ps. The result used only the shaded window, so its noise floors miss most of this drift.

## Which results have converged

Only the sign of the 0.2 V/Å effect has converged. Nothing that would support a weak-field claim has, and all of it is within reach of your cluster.

| Result | Now | Converged? | What to run | Cost |
| --- | --- | --- | --- | --- |
| P1 separation at 0.2 V/Å | Ranges never overlap in all 37 post-ramp frames of each run (my check) | The sign, yes. The size, no: the wide tube drifted from 0.46 to about 0.6 after a cold start | Switch the field on from equilibrated 300 K states, 3 runs per tube | Stage A: about a day of cluster time |
| Zero-field P1 | −0.12 and +0.15 | No: it must be 0, and most molecules stayed frozen for 500 ps | A mirrored start, then 5 independent runs, each at least 20× the measured relaxation time | Stages A and B |
| Research noise floors | 21 frames from one 100 ps window per state | No: about one independent sample of the slow orientation per run | Floors from block differences between independent zero-field runs | Stage B |
| Classifier accuracy | 100% per frame; one run per state; label-dependent text | Only at a saturating field | Leave-one-run-out on blind block programs, field by field | Stage B, then Y8 |
| Weak-field detection | Not measured | — | A field ladder from 0 to 0.2 V/Å, 2 tubes, 3–5 runs per field | About 50 runs of 10 ns at 2,000–4,000 atoms |
| Review 9's open numbers: glass history, liquid floors, temperature resolution, relaxed defects | Unchanged | No | The v3 X7 recipe | As in Review 9 |

A rough estimate (ideal dipoles, linear response, this model's charges) shows why the test must judge ensembles. At 0.01 V/Å, P1 should move by about 0.1, while one frame's zero-field P1 scatters by about 0.16 with 13 molecules and 0.07 with 61. A per-frame classifier would score roughly 60–80%; block averages can separate the states, over a block length that Stage A's relaxation time sets.

## Decisions made under your delegation

Eleven decisions, binding for the agents. Section 1 of brief v4 spells out how each is implemented; D17–D27 from Review 9 stand.

| ID | Decision | Why |
| --- | --- | --- |
| D28 | The overnight result is a pilot: the pipeline works end to end at a saturating field. RESULTS.md withdraws "confinement aligns the solvent" and records the measured non-equilibration | Problems 2 and 3 |
| D29 | Lift blind. The orientation statement is measured for every frame whose dialect names dipolar species, with or without `field=`, which adds only the `state field` line. `field_dipole_p1` becomes `dipole_p1` | Problem 4: a classifier that reads label-dependent text is circular |
| D30 | A field ladder replaces the single comparison: 0, 0.005, 0.01, 0.02, 0.05, 0.1 and 0.2 V/Å, plus your working value. If you name none, the working field is 0.01 V/Å | Your MLP targets weak fields; for this model's EC, linear response ends near 0.02 V/Å, where μE ≈ kT |
| D31 | Judge ensembles, not frames: block programs from independent runs, floors from zero-field runs only, accuracy by leaving one run out | Chaord's rule 1 (a program is a macrostate); problem 8 |
| D32 | Go/no-go on Oct 29, replacing D25's test. Go if Chaord's blind block programs detect the field at the same smallest field as a direct P1 analysis (within one ladder step), and that field is at or below your working field. Report which other statements move. No-go: Chaord stays the builder and recorder for your MLP paper, off its critical path | It tests what Chaord adds, not what the physics allows |
| D33 | Real-frame tube lifts become R2's bar: noise up to 0.1 Å, 6-digit text, cells of 3 or more periods, any position, tube diameter up to 0.8 of the cell, each 20 of 20 with the radius within 0.05 Å. Tolerances live in the dialect | Problem 5. v3 asked only for ideal tubes, which was my gap |
| D34 | The reader wraps periodic coordinates and records the largest excursion, accepts unwrapped columns, and raises on an ambiguous mass unless given a type map | Problem 7; v3 R1 already asked for the mass guard |
| D35 | The library composes tube plus liquid: `lift_frame(mode="auto")` returns the nanotube region and the confined fluid. Research scripts call only the public API | Problem 6 |
| D36 | Electrolyte statements: the Li⁺ solvation shell by species, ion-pair classes, species radial profiles inside the tube radius, and density on the accessible volume. Cutoffs come from reference runs, never from the frames under test | Problem 9; these are what an electrolyte reader looks for |
| D37 | Paper-grade runs use your Moltemplate model with its native mixing rule and a stated charge scaling; the filling comes from a reservoir run; one thermostat scheme covers all mobile atoms; dumps keep 10 significant digits; raw trajectories stay out of git, with sha256 manifests | Problem 10; one model then serves both papers |
| D38 | Brief v4 supersedes v3 where they conflict; v3's X1–X7 and X9 stand. AGENTS.md's first rule points to v4 | Problem 1 |

## Do now

You, today, about 20 minutes. Step 1 is the one that failed last round.

1. Commit the attached `open_items_v3.md` and `open_items_v4.md` to `docs/reviews/`. Make AGENTS.md's first rule: "Read docs/reviews/open\_items\_v4.md before any work; it supersedes older briefs where they conflict."
2. Send the agents one line: "Read docs/reviews/open\_items\_v4.md. Decisions D17–D38 are made. Today S1 starts X1, S2 starts Y1, S4 starts Y2 and Y4, and S3 writes the Stage A inputs. Merge in the order of section 4."
3. Commit `docs/reviews/owner_inputs.md` with two facts: your working field strength(s) in V/Å, and your Moltemplate model of the tube and electrolyte (the files, or where they live). If you have no working value yet, write that; the default is 0.01 V/Å.
4. Give the S3 agent your cluster's job script and the LAMMPS build to use, with the RIGID, KSPACE and MOLECULE packages.

Agents, today: X1 merges first, one line plus its tests. Y2 and Y4 are small, and they unblock the rest of the research lane.

## In what order

&#91;embedded content: working plan · 5 streams, Oct 15 to Oct 29\]

The research lane is the critical path: Y2 and Y4 first, Stage A's relaxation times before Stage B, and Y3 and Y5 merged before the analysis. The fixes and records lanes wait on nothing. The glass-rate runs (v3 X7) share the cluster queue and block nothing.

## In parallel

Five streams, one agent each; no two own the same file.

| Stream | Work orders | Owns | Starts | Takes about |
| --- | --- | --- | --- | --- |
| S1 Fixes | X1, then X2, X5, X6 (from v3) | `lift/surface.py`, the snap, `realize/`, `build/amorphous.py`, `glass.yaml` | Now | 3–4 days |
| S2 Records | Y1, then X3, X4, X9 | `research/cnt_electrolyte/RESULTS.md` and `README.md`, the top-level `README.md`, `reports/`, `tools/acceptance.py`, `spec/examples/` | Now | 2–3 days |
| S3 Cluster runs | Y6 (Stage A), Y7 (Stage B), then v3 X7 | Run generators and inputs under `research/cnt_electrolyte/` (no chaord import), including `params.py`; manifests; `bench/reference/` | Inputs now; runs as the queue allows | About 7 working days, mostly compute |
| S4 Research library | Y2 and Y4, then Y3, Y5, Y9 | `lift/__init__.py`, `lift/nanotube.py`, `io/lammps.py`, `cv/field.py`, the dialects, `build/molecules.py` | Now | 6–7 days |
| S5 Analysis | Y8 | A new `research/cnt_electrolyte/analysis/` and the go/no-go report | When Stage B's data and Y2, Y3, Y5 have merged | 2 days |

Your tasks alongside: submit the jobs S3 writes, read the Stage A report (relaxation times) before Stage B starts, and record the go/no-go on Oct 29.

## Done when

This round is done when every box below is ticked, with the go/no-go recorded on Oct 29, 2026.

- [ ] Y1: RESULTS.md calls the result a pilot at a saturating field, drops "confinement aligns the solvent" and shows the non-equilibration numbers.
- [ ] Y2: one frame lifted with and without `field=` prints the same `dipole_p1` line; only `state field` differs.
- [ ] Y3: the tube probe in brief v4 prints 20/20 on every line, and `lift_frame` on a raw `prod.dump` frame returns (10,10) plus the confined fluid.
- [ ] Y4: all four `prod.dump` files read, and an ambiguous mass raises.
- [ ] Y5: every frame carries the Li⁺ shell and radial-profile statements, with no ZeroDivisionError.
- [ ] Y6: one report gives the relaxation times per tube, the mirrored-start verdict and the reservoir filling.
- [ ] Y7: the field ladder has run, with sha256 manifests.
- [ ] Y8: the smallest detected field per tube, for Chaord and for direct P1, plus the linear-response check and the go/no-go.
- [ ] From v3: X1, X2 and X5 merged; X3 and X4 merged; X7's glass-rate report written.

## If this becomes a paper

The research claim is still the one that makes Chaord a paper, and overnight it moved from nothing to a pilot.

| Claim a paper would make | Evidence now | Status |
| --- | --- | --- |
| Crystals round-trip exactly and canonically | A2 and A3, plus my 900 transforms and 70 shifts (Review 9) | Ready once strained boxes (X2) are in |
| Point defects are recovered | 1.000 on unrelaxed planted defects | Needs MD-relaxed defects (X7) |
| Disordered matter round-trips within MD noise | A5: 6 of 6 cases | Needs converged floors and resolution curves (X7) |
| It reads real simulations | Ideal frozen tubes only; the reader refuses real dumps | Y3, Y4 |
| It sees what a weak field does to a confined electrolyte | A saturating field, one run per state, not blind | Y6–Y8: the claim that makes it a paper |
| It saves DFT labels when training an MLP | Nothing yet | R4, after a go |

Your nanotube MLP paper is the one with a deadline, so keep its critical path independent of Chaord. If the go/no-go passes, the same runs serve both papers: the field-response analysis and the choice of DFT frames. If it fails, Chaord still builds and records the MLP paper's systems, and its own paper becomes a methods paper on the round trip.

The repo still holds no manuscript draft. If the manuscript you mean is the MLP paper, send me its repo or draft. I will check the same things there: system size, run length with block averages, independent runs, field strengths inside linear response, and the MLP's error on held-out fields.

## Sources

- WhiteCrosstheRiver/Chaord, commits `4c36697` to `2990d08` (9 commits, Oct 14–15): `2e0ea1d` the LAMMPS reader, `95ad763` the field observables, `3c6743c` PC and EMC, `471935f` the cylinder, nanotube and field statements, `02a4f4f` and `ca14119` the nanotube wrap fix, `639f468` the four LAMMPS runs, `2990d08` the first result.
- Read: `research/cnt_electrolyte/RESULTS.md`, `README.md`, `params.py`, `build_systems.py`, `analyze_first_result.py`, the four `in.*` inputs and `runs/first_result_summary.json`; `src/chaord/lift/nanotube.py`, `lift/__init__.py`, `io/lammps.py`, `build/nanotube.py`, `build/molecules.py`; AGENTS.md.
- My checks on your code and data: P1 of all four runs from the raw data files and dumps with my own numpy code (the build plus 17 equilibration and 21 production frames each); per-molecule reversal counts; the free-dipole limits for this model's dipoles; chirality inference on 9 chiralities with 20 translations each, 8 noise levels, dump rounding, short cells and wide tubes; the reader on the production dumps; the general lift on an ideal (10,10) tube under three dialect stacks; one frame lifted with and without a field; the EC, PC and EMC templates; brief v3's four reproducers on `2990d08`; and every code block in brief v4, run as written.
- Limits: PyPI is blocked here, so ASE, lark, pint and spglib were stand-ins; none of the findings above depends on them. I could not open your GitHub Actions runs, because no GitHub account is linked to this session. LAMMPS is not available here either, so I re-analysed your committed trajectories rather than running new ones.
