# Chaord: a compiled language for molecular simulation — round-trip fidelity and pre-registered blind field tests in a confined electrolyte

*Working draft. Status: methods and all field results drafted (T5b, T5c); R13
reported from `reports/research/go_no_go.md` section 15; the program-length
curve from `research/program_length/RESULTS.md` (section 7). Markers: **[VERIFY]**
is not yet traced to source; **[CITE]** needs a full reference. Numbers come
from `assets/paper_numbers.json` unless a file is named.*

## Abstract

Atomistic simulation has no shared language for what a configuration is. Inputs
are built by one-off scripts, and existing snapshots can be analysed but not
written back as a description. We introduce Chaord, a language in which a program
is a macrostate: exact where a system is ordered, statistical where it is not,
across crystals, glasses, liquids, gases and interfaces. A compiler samples
coordinates from a program and a decompiler lifts a snapshot back to one. We make
the round trip the correctness test: decompile, rebuild, and compare observables
the program never stated against the frame-to-frame noise floor. Chaord passes
14 acceptance criteria, each guarded by an injected error that must make it fail,
and over 78,078 blind lifts its printed dipole-order statement matches an
independent calculation to the precision of the print. Used as an instrument, its text alone
detects field-induced dipole order in a LiPF6 electrolyte confined in carbon
nanotubes without being told a field is present: a pre-registered test confirmed
0.01 V/Å (p = 0.025, n = 6 runs per group) in tubes packed below their open-tube
uptake. At the measured uptake, a second pre-registered test did not confirm
0.01 V/Å (p = 0.59), and 0.02 V/Å narrowly missed (p = 0.054): the statements
stayed exact, but the response was smaller and the spread between builds as
large as the response. A pre-registered test of whether a longer program can
stand in for physics found that, with the compiler's two fluid restraints, it
cannot: on two liquids every rebuild that passed did so through physics, and the
restraint that met its g(r)-peak target moved the rest of the structure away from
the reference. We state the limits: a pilot force field, packed fillings
and six runs per field.

## 1. Introduction

Simulation studies are usually reported as prose plus scripts, with no
machine-readable statement of what a configuration is. Chaord is a language for
atomic systems in which a program describes a macrostate, a family of
configurations, and a coordinate file is one microstate of it. A compiler
(`chaord build`) realizes coordinates from a program, and a decompiler
(`chaord lift`) produces a program from coordinates.

The project began with a practical question. Fast decision models read text, and
training a machine-learned potential needs repeated decisions about which
configurations deserve an expensive DFT label. Raw coordinates are the assembly
language of an atomic system: complete, but unreadable to such a model. That
turned the problem into a language problem, and the language turned out to be the
harder and more general contribution. It must be exact where a system is ordered
and statistical where it is not, across crystals, glasses, liquids, gases and
reactive interfaces.

This paper does not test that decision loop, and it makes no claim about DFT
labels (D53). It tests the prerequisite: whether the text faithfully carries a
physical signal, without being told it is there. We make the instrument claim
first, because a physics claim made through a language is only as good as that
language's measured fidelity. We then use the instrument on a blind problem:
detecting field-aligned dipole order in a LiPF6 electrolyte confined in carbon
nanotubes, where the lift is never told the field. If the text could not see what
a field does to the liquid, it could not be used to choose training frames; that
it can is necessary for that use but does not establish it. Finally, we test a
claim from the original design, that a program stating more about a fluid should
need less simulation to reproduce it.

**Related work.** Builders are one-directional. Packmol packs molecules into
regions and moltemplate generates topologies [5,6]; neither reads a snapshot back
into a description. ATLAS composes structures from natural language through a
JSON component graph and analyses existing structures into components, but
reports minimal support for amorphous and statistically disordered phases [7].
Text encodings of crystals have been benchmarked for language models, which were
found to rely on local rather than geometric information [8]. Chaord differs in
three ways: a program denotes an ensemble rather than one structure; build and
lift share one definition of every statistical quantity; and correctness is
defined by the round trip against a measured noise floor. **[VERIFY: read [7] and
[8] in full before submission; this paragraph rests on abstracts.]**

## 2. The language and the compiler

Statements have one of six kinds: build, state, constrain, assert, history and
conserve. Thresholds and cut-offs live only in versioned dialect files, and
every program names its physics backend. Counts per species and total charge are
conserved: what a lift cannot explain goes to a `residual`, and a lift that
cannot describe a frame refuses rather than printing a program it cannot stand
behind. Each collective variable is measured, restrained and checked by the same
code, so a printed statement and its check cannot diverge.

The blind rule (D29) is what makes the field test meaningful: when a dialect
names a dipole species, `dipole_p1` is measured unconditionally, and passing a
`field=` argument adds only a `state field` line. No lift in this work received
that argument.

**Figure 4.** A program lifted blind from a (16,16) frame of the early 49.2 Å pilot
at zero field (`figures/fig4_example_program.chaord`, taken verbatim from
`research/cnt_electrolyte/runs/first_result_summary.json`, key `cnt16_field0#0`).
**[Annotate. It predates the final lift pipeline; re-lift a Z4 frame before
submission if an environment with Chaord's dependencies is available.]**

## 3. Instrument validation

**Acceptance.** All 14 of 14 acceptance criteria (A1–A14) pass
(`assets/table1_acceptance.md`; Table 1). The criteria carry mutation tests: the
most likely real error for a criterion is injected and must make it fail. An
independent verifier with no code authorship reproduced the set
(`reports/verification_2026-10-08.md`), and clean CI on both operating systems
passed at commit `ba6b594` (run 37868981153; `reports/gate_a_prime.md`). The
benchmark covers crystals, defects, liquids, glasses, solutions, surfaces,
interfaces and gases: 25 cases in `bench/data` (eight crystals, three defect
sets, three solutions, and fluids, gases, glasses, surfaces, interfaces and a
reactive system), five frames each, plus MD references from an independent
engine. The criteria are the authors' own, and several (A10 determinism, A13 no
crashes, A14 documentation) are software quality rather than science, so we do
not read 14/14 as evidence of physical fidelity.

**Statistical round trip.** Six MD reference cases have a measured noise floor,
and all six pass the A5 gate, both distances within 1.5 times the floor.

**Table 3.** A5 distance as a multiple of the noise floor (`reports/acceptance_details.md`).

| Case | Family | g(r) | Coordination |
| --- | --- | --- | --- |
| Kob–Andersen glass | glass | 0.9 | 0.3 |
| LJ liquid, 500 atoms | fluid | 1.2 | 0.6 |
| LJ liquid, 2,048 atoms | fluid | 0.7 | 1.1 |
| LJ solid–liquid slab | interface | 1.1 | 0.4 |
| NaCl(aq) | fluid | 1.3 | 1.1 |
| TIP4P water | fluid | 1.2 | 0.7 |

Six further cases are synthetic frames with no measured floor, and A5 does not
score them. Two qualifications matter. The ordered gate (A3) as applied is an
optimal-assignment fit on the stored thermal frames, because the planned
StructureMatcher, with default primitive reduction, matched 0 of 9 thermal
frames against their rebuilds. And for fluids a pass largely verifies the
physics stage rather than the text, as section 7 shows.

**Fidelity.** Across all 78,078 blind lifts (42,042 on the full ladder, 18,018 in
the pilot-filling confirmatory set and 18,018 at the reservoir filling), the
largest difference between the statement `dipole_p1` and an independent direct
calculation was 5.00 × 10⁻⁴, which is the half-quantum of the statement's
three-decimal print (Figure 3). The residual is flat across that interval, as
rounding predicts, so the statement and the direct quantity agree to print
precision. There were no failed static checks on any frame of either
confirmatory set; the denser reservoir filling, at 4,872 atoms, did not degrade
the print.

## 4. The physical system

**Model.** LiPF6 in EC:EMC in armchair carbon nanotubes, with rigid OPLS-AA-based
molecules [1,2], Werder sp²-carbon parameters [3] and Lorentz–Berthelot mixing, which is a recorded deviation. Tubes
are frozen and uncharged. We use PPPM (relative accuracy 10⁻⁴), a 12 Å cut-off, a
1 fs timestep and a single Langevin scheme with 1 ps damping for all mobile
atoms.

**Protocol.** Each job runs a 100 ps ramp from 10 to 300 K and 900 ps at 300 K at
zero field, then applies the field, settles for 1 ns, and records two 6 ns
production blocks. All jobs in a set use one LAMMPS build (3 Mar 2020).

**Packing at the pilot filling.** A seeded rejection packer places molecules
against contact floors of max(2.0 Å, 2.2 Å with H, 0.85 σᵢⱼ), and molecules that
cannot be placed strictly are inserted and pushed apart by translation.

**Packing at the reservoir filling.** The reservoir composition jammed this
packer: at floor scales 1.0–0.85 insertion failed on EC, and at 0.8–0.7 insertion
completed but translation alone could not recover the true floors. Following the
pre-registered staged policy, and without changing counts, we insert at 0.8 × the
floor and relax with rigid-body LAMMPS dynamics, so molecules can rotate as well
as translate. The 90 ps relaxation uses only a purely repulsive soft pair
potential ranged on the true floors and a radial wall, with no charges or LJ
terms. A build is accepted only if it clears the unmodified floors. The first
attempt on `r13_b1` missed the wall floor (0.9985 against 0.999), so the wall was
stiffened and the threshold left alone; the result has minimum pair/floor 1.098
and wall/floor 1.028 (`r13/data/build_manifest_r13_b1.json`). Starting production
from this state releases residual strain: the mobile temperature spikes to about
880 K within the first picosecond and is gone within 5 ps, well before any
analysed data (`r13/evidence/smoke/`). That transient was measured on `r13_b1`
only. All six builds pass the same gate (minimum pair/floor 1.097–1.100,
wall/floor 1.024–1.031; `r13/data/build_manifest.json`). Builds 1 and 6 were
inserted at 0.80 × the floor. Builds 2–5 jammed there on an EC insertion and were
inserted at 0.75 ×; every jam is recorded and no count was changed.

## 5. Filling is a state variable

Open-tube filling runs measure the uptake a packed pore should match. For (16,16)
the uptake per Å is EC 1.2701, EMC 0.5917, Li⁺ 0.2673 and PF6⁻ 0.2127, above the
pilot packing by 68.9%, 21.3%, 119.2% and 74.4% (Figure 5). For (10,10), the fill
holds EC and EMC well above the pilot, and no Li⁺ at all in the analysed window.
The uncertainties are lower bounds: the 1.25 ns blocks are shorter than the
collective relaxation time of about 3 ns.

Inside (16,16) the ions are not neutral: 26.3 Li⁺ per 20.9 PF6⁻ per 98.4 Å. We
report this as a finding. A periodic build cannot carry that net charge, so R13
uses 24 LiPF6 at the rounded mean, a stated modelling choice whose effect this
design cannot separate.

An earlier filling-sensitivity test (Q3) is void, because it used a composition
from an uncorrected draft and so tested neither filling. Both of its reported
results are preserved with their scripts (`q3_evidence/q3_results.json`,
`docs/records/corrections.md`). Stage B and Z4 below are labelled "packed below
open-tube uptake" throughout.

## 6. Blind field detection

**Stage B ladder.** Seven fields from 0 to 0.2 V/Å, two tubes, three builds each
(Figure 2a). The smallest detected field depends on the criterion. For (16,16) it
is 0.1 V/Å under the 3× floor rule, 0.02 V/Å under the build-paired 3σ rule and
0.01 V/Å under paired 2σ. For (10,10) it is 0.1 V/Å on both paired rules, and the
3× floor rule detects nothing.

**Confirmatory test (Z4), packed below open-tube uptake.** Pre-registered before
any job was built: 18 jobs, (16,16), 98.4 Å, fields 0, 0.01 and 0.02 V/Å, six
builds, with independent runs as the unit (Welch two-sided, α = 0.05, zero-field
run means against field-on block B). The registered gate passed.

| Contrast | Shift | 95% CI | t | df | p | Runs positive |
| --- | --- | --- | --- | --- | --- | --- |
| 0.01 vs 0 V/Å (gate) | +0.107 | [+0.018, +0.196] | 2.85 | 7.0 | 0.0246 | 5 of 6 |
| 0.02 vs 0 V/Å | +0.187 | [+0.135, +0.240] | 7.97 | 9.9 | 1.3 × 10⁻⁵ | 6 of 6 |

The 0.01 V/Å result is a modest detection: its interval nearly reaches zero and
one run sits on the zero-field side. It should not be read as more than the
registered rule says. The registered statistic is the direct calculation. The
same contrast run on the printed statement instead, which is secondary and was
not registered, gives the same result (0.01 V/Å: p = 0.0246; R13 below: p = 0.594),
so the text carries the detection. The build-paired contrast is secondary and weakened: a
pre-registered chaining check found that 6 of the 18 jobs diverged from their
siblings during the shared zero-field ramp, a floating-point non-reproducibility
on the cluster, so exact pairing does not hold for them. The primary test is
unpaired and is not affected.

**Reservoir filling (R13).** Same design at the corrected uptake, 125 EC, 58 EMC
and 24 LiPF6 per 98.4 Å (4,872 atoms), pre-registered in `r13/PLAN.md` before any
build. The decision rule was that 0.01 V/Å is confirmed at this filling only if
Welch p < 0.05 on this set alone. The analysis imports the Z4 code unchanged and
runs the plan's own test code verbatim. **0.01 V/Å is not confirmed at the
reservoir filling.**

| Contrast | Shift | 95% CI | t | df | p | Runs positive |
| --- | --- | --- | --- | --- | --- | --- |
| 0.01 vs 0 V/Å (gate) | +0.020 | [−0.064, +0.105] | 0.56 | 8.0 | 0.59 | 4 of 6 |
| 0.02 vs 0 V/Å | +0.051 | [−0.001, +0.103] | 2.21 | 9.3 | 0.054 | 5 of 6 |

The measurement is not what failed: the statements matched the direct
calculation to print precision on all 18,018 frames. Two things changed against
the pilot filling (Figure 2b). The 0.02 V/Å response is about a quarter as large
(block-B mean +0.041 against +0.169 for Z4), and the zero-field runs spread from
−0.080 to +0.028 between builds, the same order as that response. With six runs
per group, an unpaired test cannot separate the two. The build-paired contrast,
registered as secondary, removes the between-build spread and shows a consistent
0.02 V/Å shift (+0.051 ± 0.013, positive in 5 of 6 builds), but it does not
change the verdict, and we do not promote it. Whether the smaller response
reflects a denser, more constrained pore liquid or the pilot force field, this
design cannot say; we make no physical claim either way.

The chaining check found 2 of 18 jobs diverging from their siblings during the
shared zero-field ramp, and these are exactly the two jobs the scheduler placed
on a second compute node; the other 16 ran on one node with the same binary,
rank layout and seeds. This turns the floating-point non-reproducibility seen in
Z4 from an inference into an observation for R13. The primary test is unpaired
and unaffected; the paired view loses exact pairing for those two builds.

Taken together, the field claim that stands is narrow: detection at the pilot
filling (0.02 V/Å decisively, 0.01 V/Å modestly) and no confirmed detection at
the measured uptake with this design.

## 7. Program length against physics

The compiler restrains two fluid statements, mean coordination
(`constrain cn`) and the height of the first g(r) peak (`constrain gr_peak`), by
reverse Monte Carlo after packing and again after any physics stage. If programs
can trade length for physics, a richer program should pass with less simulation.
We pre-registered the test before any grid data existed
(`research/program_length/PLAN.md`, commit `5b434b8`). It uses two
Lennard-Jones liquids (`lj_liquid`, `lj_liquid_large`) and three physics levels:
P0 packing only, P1 700 MD steps (10% of the dialect default) and P2 the default
7,000. Each liquid's lifted program, with its assertions removed, gives three
programs: S0 as lifted, S1 adding the coordination restraint copied from the
lifted assertion, and S2 adding the g(r)-peak restraint as well (201, 240 and 277
characters for `lj_liquid`). Every cell was rebuilt with three seeds and scored
by the A5 gate: averaged g(r) and coordination distances each at most 1.5 times
the measured floor. A bond-angle distance, which no level restrains, was the
held-out check.

**Table 2.** Program-length grid. Pass or fail on the A5 gate, and the g(r)
distance as a multiple of its floor, as the range over the two liquids
(`research/program_length/results.json`).

| Program | P0, packing only | P1, 700 steps | P2, 7,000 steps |
| --- | --- | --- | --- |
| S0, lifted | fail, 13.4–18.7 | pass, 1.21–1.27 | pass, 0.59–0.94 |
| S1, + coordination | fail, 13.4–18.7 | pass, 1.21–1.27 | pass, 0.59–0.94 |
| S2, + g(r) peak | fail, 13.4–19.5 | fail, 4.7–7.0 | fail, 9.2–19.3 |

Every pass came from physics. With MD on, the shortest program passed on both
liquids; with packing alone, nothing in the ladder passed. No restraint turned a
failing cell into a passing one. S1 changed nothing, by construction: the copied
tolerance (12.0 ± 1.1 and 12.0 ± 1.2) is the per-atom spread of coordination, not a tolerance
on its mean, so a random packing already satisfies it and the restraint never
moves. S2 made things worse, most of all after full MD. It met its g(r)-peak
target but pulled the rest of the structure away from the reference: the g(r)
distance rose from 0.59–0.94 to 9.2–19.3 times its floor, and the held-out angle
distance rose 13- to 32-fold against S0. The held-out check caught what the
targeted statement could not.

The registered rule does not settle a label. It asks that the shortest passing
program not grow with physics, and that at least one level need a strictly
longer program than P2. It does not say how "no passing program" at P0 compares
with a length. Counted as the longest, the rule returns SUPPORTED; counted as no
length, NOT SUPPORTED. The analysis code chose the first ordering after
registration, though before any data, so we report the verdict as undetermined
and rest on the substance. These data do not show that program length can
replace physics: the compiler's two restraints either do nothing as stated or
damage the structure. One by-product is that 700 MD steps already pass the gate,
at 1.17–1.27 times the floor. Restraints that act on whole distributions, or a
tolerance on the mean taken from the measured floor, would be the next test, and
it would have to be pre-registered as new work.

## 8. What the language adds

The field was detected through lifted programs, never from coordinates with field
knowledge. Conservation and residual accounting are falsifiable statements rather
than conventions: dropping one atom must fail.

The measured noise floors behind these comparisons were themselves tested. Thirty
independent Kob–Andersen quenches [4] at three rates a decade apart show that
inherent-structure energy separates adjacent rates decisively, by 0.023–0.040
ε per atom, or 6–10 times the cross-quench spread. Structure does not: at one
decade, g(r) and coordination differences stay within 0.5–1.0 times their
cross-quench floor (`reports/convergence/ka_rates.md`). So a Chaord statement of
structure alone would not distinguish these glasses, and that is a limit of the
observable, not of the print. Re-grounding the liquid floors on at least 132
averaged pairs changed most of them little. The largest move was the
`lj_liquid_large` averaged coordination floor, which widened by a factor of 1.57,
a sampling effect in the original 21 pairs (`reports/convergence/liquid_floors.md`).

**The decision loop.** Chaord ships a line-diff state codec for a text decision
model (`src/chaord/integrations.py`, 320-token budget), tested for exact round
trips. That codec is plumbing. No decision model has been trained or run on
Chaord text in this work.

## 9. Limitations

1. Pilot force field (rigid OPLS-AA-based molecules, Werder carbon, LB mixing),
   with no DFT anchor yet.
2. Filling: packed-pore constructions. R13 packs the measured uptake, but it is
   not a grand-canonical ensemble, and it neutralizes a measured ion imbalance.
3. Narrow-tube non-ergodicity: (10,10) is glassy-slow, and the block protocol may
   under-sample it.
4. Cluster floating-point non-reproducibility: reruns are not bit-reproducible
   across nodes. It affected 6 of 18 jobs in Z4 and 2 of 18 in R13, where the two
   are exactly the jobs run on a second node.
5. Six runs per field: power against small shifts is limited, and at the
   reservoir filling the between-build spread is as large as the 0.02 V/Å
   response. A larger set, or a design that removes build-to-build variance from
   the primary test, would be needed to resolve it; neither was registered here.
6. Program length against physics was measured for two atomic liquids and two
   scalar restraints only, and its registered verdict is undetermined (section
   7). The negative finding bears on these restraints as specified, not on
   richer ones. Slab, glass and molecular systems have no restraint path.

## Figures

- **Table 1 / Figure 1.** Acceptance A1–A14 (`assets/table1_acceptance.md`).
- **Table 2.** Program-length grid, in section 7
  (`research/program_length/RESULTS.md`, `assets/paper_numbers.json` key
  `program_length`).
- **Figure 2.** `figures/fig2_blind_p1_vs_field.pdf`. Blind P1 against field. (a)
  Stage B, mean ± s.e.m. over three builds. (b) (16,16): Stage B and Z4, both
  packed below open-tube uptake, and R13 at the measured uptake; markers are
  means ± s.e.m., dots individual Z4 and R13 runs, series offset horizontally
  for legibility.
- **Figure 3.** `figures/fig3_fidelity.pdf`. Chaord statement against direct
  calculation on 18,018 confirmatory frames, and the residual, bounded by the
  print half-quantum.
- **Figure 4.** `figures/fig4_example_program.chaord`. Lifted program for a pilot frame.
- **Figure 5.** `figures/fig5_filling.pdf`. Pilot packing against open-tube uptake
  per species and tube.

## References

1. J. S. C. Nutakki, *Molecular Modelling of Lithium Salts in the Binary Mixtures
   of Ethylene and Ethyl Methyl Carbonates*, M.S. thesis, University of Illinois
   Chicago (2023). https://indigo.uic.edu/articles/thesis/25392199
2. W. L. Jorgensen, D. S. Maxwell, J. Tirado-Rives, J. Am. Chem. Soc. 118, 11225
   (1996).
3. T. Werder, J. H. Walther, R. L. Jaffe, T. Halicioglu, P. Koumoutsakos, J. Phys.
   Chem. B 107, 1345 (2003).
4. W. Kob, H. C. Andersen, Phys. Rev. E 51, 4626 (1995).
5. L. Martínez, R. Andrade, E. G. Birgin, J. M. Martínez, J. Comput. Chem. 30,
   2157 (2009). (Packmol) **[VERIFY]**
6. A. I. Jewett et al., J. Mol. Biol. 433, 166841 (2021). (moltemplate) **[VERIFY]**
7. ATLAS: Atomic Translation & Language for Automated Structures, arXiv:2609.29595
   (2026). **[VERIFY authors]**
8. N. Alampara, S. Miret, K. M. Jablonka, MatText: Do Language Models Need More
   than Text & Scale for Materials Modeling?, arXiv:2406.17295 (2024).

**[CITE: LAMMPS, PPPM and any related-work references still to add.]**
