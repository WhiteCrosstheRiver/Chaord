# Chaord: a compiled language for atomic macrostates, tested by round trip

*Working draft. Numbers come from `assets/paper_numbers.json` unless a file is
named. Author list, acknowledgements and the data DOI are open.*

## Abstract

Atomistic simulation has no shared language for what a configuration is. We
introduce Chaord, in which a program is a macrostate, exact where a system is
ordered and statistical where it is not. A compiler samples coordinates from a
program, a decompiler lifts a snapshot back to one, and we test the round trip
against measured noise floors. For two simple liquids the physics did the work
and two fluid restraints added nothing. For Lennard-Jones states sharing potential,
temperature and density at one state point, rebuilt liquid, fcc and hcp matched
only their own references in a pre-registered test, while a control without
structural text always gave the liquid. A crystal–liquid slab missed the gate, so
the four-state claim was not supported. In a LiPF6 electrolyte in carbon nanotubes,
a registered statistic that the printed text reproduces detected field-induced
dipole order at a pilot packing (0.01 V/Å, p = 0.025) but not at a packing of the
measured uptake (p = 0.59).

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
labels. It tests the prerequisite: whether the text faithfully carries a
physical signal, without being told it is there. We make the instrument claim
first, because a physics claim made through a language is only as good as that
language's measured fidelity. We then use the instrument on a blind problem:
detecting field-aligned dipole order in a LiPF6 electrolyte confined in carbon
nanotubes, where the lift is never told the field. If the text could not see what
a field does to the liquid, it could not be used to choose training frames. Where
it can, that is necessary for such use but does not establish it. Finally, we test a
claim from the original design, that a program stating more about a fluid should
need less simulation to reproduce it, and then ask where the text carries
something the physics cannot supply.

**Related work.** Builders are one-directional. Packmol packs molecules into
regions and moltemplate generates topologies [5,6]; neither reads a snapshot back
into a description. ATLAS turns natural-language requests into structures through a
JSON component graph; it can also analyse an existing structure into components,
but that analysis is descriptive and is not converted back into a build
specification, a specification denotes one structure, and no structure-to-text-to-
structure test is reported [7]. Its disordered examples are a bundled
melt-quenched glass and water packed by random insertion, checked for contacts
rather than against a noise floor. Text encodings of materials have been
benchmarked for language models, which were reported to learn poorly from
coordinate information [8], and string descriptions of crystals such as
Robocrystallographer [9] and invertible representations such as SLICES [10] are
aimed at ordered solids. Chaord differs in three ways: a program denotes an
ensemble rather than one structure; build and lift share one definition of every
statistical quantity; and correctness is defined by the round trip against a
measured noise floor.

## 2. The language and the compiler

A Chaord program describes a macrostate: the family of configurations that
satisfy what it says. A coordinate file is one microstate of that family. A
program has one `system` block (cell, periodicity, conserved counts), a
`physics` block naming the backend, optional `species`, one block per region
(`crystal`, `amorphous`, `liquid`, `gas`, `fluid`, `cluster` or `vacuum`), each
bound to a geometry, `interface` blocks between regions, and a `residual` that
accounts for anything not explained. The language is exact where a system is
ordered (a lattice, a Kröger–Vink defect count) and statistical where it is not
(a density, a coordination with its tolerance).

Every statement has one of six kinds (Table 1). What the compiler does with a
statement depends on its kind and on the region's phase; the table states what is
implemented, which is narrower than the language's intent. A dialect is a versioned
file of vocabulary and thresholds for a family of materials. The grammar and every
statement key are specified in `docs/reference.md` and
`src/chaord/lang/chaord.lark`.

**Table 1.** Statement kinds, as implemented.

| Kind | Example | Compiler | Decompiler |
| --- | --- | --- | --- |
| build | `lattice fcc`, `defect V_Ni count 1`, `molecules H2O 620` | constructs it | infers it from the frame |
| state | `state density 0.854`, `state T 300 K` | density is targeted in fluid packing; `T` is used by the physics stage and otherwise metadata | measures it, or records it as supplied |
| constrain | `constrain cn 12.0 +- 0.3 cutoff 1.50` | `cn` and `gr_peak` are restrained by reverse Monte Carlo in atomic fluid regions and an unmet target refuses; `sro` is attempted in solid solutions; `strain` sets the slab lattice | prints it where a held quantity is found |
| assert | `assert gr_peak 1.06 height 3.07` | ignored by the build; verified by `chaord check` | measures it |
| history | `history melt 3000 K for 20 ps -> quench to 300 K at 1 K/ps` | run only by the amorphous builder | prints it only when it is known |
| conserve | `conserve atoms Ni 646 Al 217` | crystal tilings refuse a wrong atom count; otherwise verified by `chaord check` | exact, and verified against the frame |

Analysis thresholds live in dialect files, although a statement may print the
cut-off it was measured with. Lifted programs name a physics backend; for crystals
this is a default, which a physics-off build does not use. **Compilation** dispatches
on the region phases: crystals are tiled from a prototype table, with defects
placed by the stated counts; fluids are packed to the stated density and then, if
physics is on, relaxed under the named backend; slabs combine the two across their
interfaces. **Decompilation** tries each phase in turn, and a frame that no phase can
describe is refused. There are two exceptions: bonded (molecular) frames fall back
to a fluid description, and a crystal that is too small a fraction of the box to
fail the single-phase check is lifted as a liquid (section 8). What a lift cannot
explain goes to a `residual`. For fluid coordination and g(r) peaks, the same code
measures, restrains and checks the quantity, so the statement and its check cannot
diverge.

The blind rule is what makes the field test meaningful. When a dialect names a
dipole species, the statement `dipole_p1`, the order parameter P1 = ⟨cos θ⟩ between
each molecular dipole and an analysis axis, is measured unconditionally, and
passing a `field=` argument adds only a `state field` line. No lift in this work
received that argument. The blindness covers the field, not the observable: the
analyst names the species and the axis is the tube axis.

**Figure 1.** A program lifted from a Lennard-Jones crystal–liquid slab, without
any hint about the state (`figures/fig4_example_program.chaord`, taken verbatim
from `research/macrostate_selection/results.json`, key `coex|0`). Two regions, two
interfaces, 689 characters for 4,032 atoms.

## 3. Instrument validation

**Acceptance.** All 14 acceptance criteria (A1–A14) passed on a clean machine at
commit `ba6b594` (`assets/table1_acceptance.md`; Table 2). The criteria carry mutation tests: the
most likely real error for a criterion is injected and must make it fail. An
independent verifier with no code authorship reproduced the set at `f90e988`
(`reports/verification_2026-10-08.md`), and clean CI on both operating systems
passed at commit `ba6b594` (run 37868981153; `reports/gate_a_prime.md`). That is
the last clean-machine pass. **[Resolve before submission: the scheduled
clean-machine acceptance job failed on every run from 16 October to 9 November 2026,
on later commits, and the failing criterion has not been established.]** At commit
`d06374a`, which includes the corrections described below, a self-run on our cluster
gave 14 of 14 (`reports/bookkeeping/`); under the project's own rule that is
bookkeeping and not clean-machine evidence. The
benchmark covers crystals, defects, liquids, glasses, solutions, surfaces,
interfaces and gases: 25 cases in `bench/data` (eight crystals, three defect
sets, three solutions, and fluids, gases, glasses, surfaces, interfaces and a
reactive system), five frames each, plus MD references. The criteria are the authors' own, and several (A10 determinism, A13 no
crashes, A14 documentation) are software quality rather than science, so we do
not read 14/14 as evidence of physical fidelity.

**Statistical round trip.** Eight MD reference cases have a measured noise floor,
and six are scored. All six pass the A5 gate, both distances within 1.5 times the
floor. `lj_glass` was retired from A5 in favour of `ka_glass`, and
`cu_solid_liquid` is recorded as a known limitation.

**Table 3.** A5 distance as a multiple of the noise floor (`reports/acceptance_details.md`;
clean-machine artifact at `ba6b594`, which predates the corrections in section 8).

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
physics stage rather than the text, as section 7 shows. The gate is also calibrated
by us. The floor definition, max(mean, P90), was adopted after a clean run rejected
`lj_liquid_large` at 1.8 times the earlier floor. The molecular relaxation length
(950 steps) was chosen where the coordination distance to the reference is lowest;
1,400 steps gives 1.54 times the floor, which would fail. The molecular references
and the compiler's classical backend both use ASE, with the same friction and
truncation, so those references are not independent of the compiler's engine.

Nor is 14/14 a guarantee of correctness: the pilots for section 8 exposed a compiler
defect and a decompiler defect that the acceptance set had passed, both corrected
and recorded in `docs/records/corrections.md`.

**Fidelity.** Across all 78,078 blind lifts (42,042 on the full ladder, 18,018 in
the pilot-filling confirmatory set and 18,018 at the reservoir filling), the
largest difference between the statement `dipole_p1` and an independent direct
calculation was 5.00 × 10⁻⁴, which is the half-quantum of the statement's
three-decimal print (Figure 2 shows the 18,018 frames of one set). The residual is flat across that interval, as
rounding predicts, so the statement and the direct quantity agree to print
precision. This is a check that the statement reports the quantity correctly, and
it says nothing about whether the quantity is informative. There were no failed static checks on any frame of either
confirmatory set; the denser reservoir filling, at 4,872 atoms, did not degrade
the print.

## 4. The physical system

**Model.** LiPF6 in EC:EMC in armchair carbon nanotubes, with rigid OPLS-AA-based
molecules [1,2], Werder sp²-carbon parameters [3] and Lorentz–Berthelot mixing, which is a recorded deviation. Tubes
are frozen and uncharged. We use LAMMPS [11] with PPPM [12] (relative accuracy 10⁻⁴), a 12 Å cut-off, a
1 fs timestep and a single Langevin scheme with 1 ps damping for all mobile
atoms.

**Protocol.** Each job runs a 100 ps ramp from 10 to 300 K and 900 ps at 300 K at
zero field, then applies the field, settles for 1 ns, and records two 6 ns
production blocks, A and B. Each confirmatory set uses one LAMMPS build (3 Mar 2020).
In the field ladder, 12 (10,10) jobs ran on the 22 Jul 2025 build and the rest on
3 Mar 2020.

Tubes are filled by packing molecules to a stated count, at the pilot filling
or at the measured reservoir uptake. Every build clears the same contact floors;
the staged packing procedure is given in Methods.

## 5. Filling differs from the pilot packing

Open-tube filling runs measure the uptake a packed pore should match. For (16,16)
the uptake per Å is EC 1.2701, EMC 0.5917, Li⁺ 0.2673 and PF6⁻ 0.2127, above the
pilot packing by 68.9%, 21.3%, 119.2% and 74.4% (Figure 3). For (10,10), the fill
holds EC and EMC well above the pilot, and no Li⁺ at all in the analysed window.
The uncertainties are lower bounds: the 1.25 ns blocks are shorter than the
collective relaxation time of about 3 ns.

Inside (16,16) the ions are not neutral: 26.3 Li⁺ per 20.9 PF6⁻ per 98.4 Å. We
report this as a finding. A periodic build cannot carry that net charge, so the reservoir builds
use 24 LiPF6 at the rounded mean, a stated modelling choice whose effect this
design cannot separate.

An earlier filling-sensitivity test is void, because it used a composition
from an uncorrected draft and so tested neither filling. Both of its reported
results are preserved with their scripts (`q3_evidence/q3_results.json`,
`docs/records/corrections.md`). The field ladder and first confirmatory test below are labelled "packed below
open-tube uptake" throughout.

## 6. Blind field detection

**Field ladder.** Seven fields from 0 to 0.2 V/Å, two tubes, three builds each
(Figure 4a). The smallest detected field depends on the criterion: a shift above
three times the noise floor, or a build-paired shift above 3σ or 2σ. For (16,16) it
is 0.1 V/Å under the 3× floor rule, 0.02 V/Å under the build-paired 3σ rule and
0.01 V/Å under paired 2σ. For (10,10) it is 0.1 V/Å on both paired rules, and the
3× floor rule detects nothing.

**First confirmatory test, packed below open-tube uptake.** Pre-registered before
any job was built: 18 jobs, (16,16), 98.4 Å, fields 0, 0.01 and 0.02 V/Å, six
builds, with independent runs as the unit (Welch two-sided, α = 0.05, zero-field
run means against field-on block B). The registered gate passed. "Runs positive" counts field-on runs with P1 above
zero.

**Table 4.** First confirmatory test.

| Contrast | Shift | 95% CI | t | df | p | Runs positive |
| --- | --- | --- | --- | --- | --- | --- |
| 0.01 vs 0 V/Å (gate) | +0.107 | [+0.018, +0.196] | 2.85 | 7.0 | 0.0246 | 5 of 6 |
| 0.02 vs 0 V/Å | +0.187 | [+0.135, +0.240] | 7.97 | 9.9 | 1.3 × 10⁻⁵ | 6 of 6 |

The 0.01 V/Å result is a modest detection: its interval nearly reaches zero, one run has
negative P1, and two lie inside the range of the zero-field runs. It should not be read as more than the
registered rule says. The registered statistic is the direct calculation. The
same contrast run on the printed statement instead, which is secondary and was
not registered, gives the same result (0.01 V/Å: p = 0.0246; the second confirmatory test below: p = 0.594),
so the printed statement reproduces the registered statistic. The build-paired contrast is secondary and weakened: a
pre-registered chaining check, which tests that jobs sharing a build stay identical through their shared zero-field ramp, found that 6 of the 18 jobs diverged from their
siblings during the shared zero-field ramp, a floating-point non-reproducibility
on the cluster, so exact pairing does not hold for them. The primary test is
unpaired and is not affected.

**Second confirmatory test, at the reservoir filling.** Same design at the corrected uptake, 125 EC, 58 EMC
and 24 LiPF6 per 98.4 Å (4,872 atoms), pre-registered in `research/cnt_electrolyte/r13/PLAN.md` before any
build. The decision rule was that 0.01 V/Å is confirmed at this filling only if
Welch p < 0.05 on this set alone. The analysis imports the code of the first confirmatory test unchanged and
runs the plan's own test code verbatim. **0.01 V/Å is not confirmed at the
reservoir filling.**

**Table 5.** Second confirmatory test.

| Contrast | Shift | 95% CI | t | df | p | Runs positive |
| --- | --- | --- | --- | --- | --- | --- |
| 0.01 vs 0 V/Å (gate) | +0.020 | [−0.064, +0.105] | 0.56 | 8.0 | 0.59 | 4 of 6 |
| 0.02 vs 0 V/Å | +0.051 | [−0.001, +0.103] | 2.21 | 9.3 | 0.054 | 5 of 6 |

The statements matched the direct calculation to print precision on all 18,018
frames, as they must, so this result concerns the response and not the print. Two
things changed against
the pilot filling (Figure 4b). The 0.02 V/Å response is about a quarter as large
(block-B mean +0.041 against +0.169 for the first test), and the zero-field runs spread from
−0.080 to +0.028 between builds, the same order as that response. With six runs
per group, an unpaired test cannot separate the two. The build-paired contrast,
registered as secondary, removes the between-build spread and shows a consistent
0.02 V/Å shift (+0.051 ± 0.013, positive in 5 of 6 builds), but it does not
change the verdict, and we do not promote it. Whether the smaller response
reflects a denser, more constrained pore liquid or the pilot force field, this
design cannot say; we make no physical claim either way.

The chaining check found 2 of 18 jobs diverging from their siblings during the
shared zero-field ramp, and these are exactly the two jobs the scheduler placed
on a second compute node (`research/cnt_electrolyte/r13/HANDOFF.md`); the other 16 ran on one node with the same binary,
rank layout and seeds. This turns the floating-point non-reproducibility seen in
the first test from an inference into an observation for the second test. The primary test is unpaired
and unaffected; the paired view loses exact pairing for those two builds.

Taken together, the field claim that stands is narrow: detection at the pilot
filling (0.02 V/Å decisively, 0.01 V/Å modestly) and no confirmed detection at
the measured uptake with this design.

## 7. Program length against physics

The compiler restrains two fluid statements, mean coordination
(`constrain cn`) and the height of the first g(r) peak (`constrain gr_peak`), by
reverse Monte Carlo [13] after packing and again after any physics stage. If programs
can trade length for physics, a richer program should pass with less simulation.
We pre-registered the test before any grid data existed
(`research/program_length/PLAN.md`, commit `5b434b8`). It uses two
Lennard-Jones liquids (`lj_liquid`, `lj_liquid_large`) and three physics levels:
packing only, short MD (700 slow-stage steps, 10% of the dialect default) and full
MD (the default 7,000). A fixed 400-step fast stage precedes the slow stage whenever
MD runs. Each liquid's lifted program, with its assertions removed, gives three
programs: S0 as lifted, S1 adding the coordination restraint copied from the
lifted assertion, and S2 adding the g(r)-peak restraint as well (201, 240 and 277
characters for `lj_liquid`). Every cell was rebuilt with three seeds and scored
by the A5 gate: averaged g(r) and coordination distances each at most 1.5 times
the measured floor. A bond-angle distance, which no level restrains, was the
held-out check.

**Table 6.** Program-length grid. Pass or fail on the A5 gate, and the g(r)
distance as a multiple of its floor, as the range over the two liquids
(`research/program_length/results.json`).

| Program | Packing only | Short MD | Full MD |
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

**The registered verdict is NOT SUPPORTED.** The rule calls the claim supported if
and only if, for both liquids, the shortest passing program does not grow with
physics and at least one level needs a strictly longer program than full MD, and
not supported otherwise. With packing alone no program passes, so no level needs a
longer program and the second clause is not met. The analysis script instead
treated "no passing program" as longer than any program, and its stored output
reads SUPPORTED. That ordering is not in the plan, and the script was committed only
with the results, so the record cannot show that it preceded the data. We report
the rule as written. These data do not show that program length can replace
physics: the two restraints either do nothing as stated or damage the structure.
One by-product is that short MD already passes the gate, at 1.17–1.27 times the
floor. Restraints that act on whole distributions, or a
tolerance on the mean taken from the measured floor, would be the next test, and
it would have to be pre-registered as new work.

## 8. Testing whether the text selects the macrostate

Section 7 shows that a liquid needs little text. The opposite case is a set of
states that the potential, temperature and density do not choose among within the
run length. We pre-registered that test (`research/macrostate_selection/PLAN.md`,
commit `8deeff2`). Four macrostates of one Lennard-Jones system at T = 0.75 and
density 0.90 are a compressed liquid, an fcc crystal, an hcp crystal and a
crystal–liquid slab, with six independent LAMMPS reference trajectories each.
Each was lifted to a program with no hint about the state. Every program was then
compiled with physics off and given identical MD, so a difference between rebuilds
can only come from the text. A control program kept only cell, temperature, atom
count, density and backend. Scoring is independent of Chaord: g(r), coordination
and common-neighbour classes [14], each within 1.5 times a floor measured between
reference trajectories.

**The registered verdict is NOT SUPPORTED.** Five of six registered conditions
held. The failed condition required every full program to match its own reference, and the
slab reached 2.5 times its floor (Figure 5a). Of its three programs, two were
lifted as a slab and rebuilt one; the third came from the frame with the smallest
crystal (fcc fraction 0.047), which passed the single-phase check and was lifted
as a homogeneous liquid.

The rest of the matrix is what a selective language should produce. Liquid, fcc
and hcp programs matched their own references at 0.4, 0.7 and 1.4 times the floor
and rejected every other reference, the smallest off-diagonal entry being 7.4.
The control gave the liquid from all four sources (0.4–0.9 times the floor) and
matched none of the others (Figure 5b). The hcp match was narrow (coordination 1.45
times the floor). The programs are 346–375 characters against 203–205 for the
control, so about 140–170 characters decide which of these states MD stays in.

This came after five pilots, which the plan records: the state point moved three
times, the slab box was doubled, and two software defects were fixed. The fifth
pilot passed all six conditions on its own seeds; the registered run, on fresh seeds,
did not. That difference shows how close the slab is to the lift's detection limit.

## 9. What the language adds

The lifts never received the field, and the printed statement reproduces the
registered field contrasts. Conservation and residual accounting are falsifiable statements rather
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
a sampling effect in the original 11 averaged pairs (`reports/convergence/liquid_floors.md`).

**The decision loop.** Chaord ships a line-diff state codec for a text decision
model (`src/chaord/integrations.py`, 320-token budget), tested for exact round
trips. That codec is plumbing. No decision model has been trained or run on
Chaord text in this work.

## 10. Limitations

1. Pilot force field (rigid OPLS-AA-based molecules, Werder carbon, LB mixing),
   with no DFT anchor yet.
2. Filling: packed-pore constructions. The second confirmatory test packs the measured uptake, but it is
   not a grand-canonical ensemble, and it neutralizes a measured ion imbalance.
3. Narrow-tube non-ergodicity: (10,10) is glassy-slow, and the block protocol may
   under-sample it.
4. Cluster floating-point non-reproducibility: reruns are not bit-reproducible
   across nodes. It affected 6 of 18 jobs in the first confirmatory test and 2 of 18 in the second confirmatory test, where the two
   are exactly the jobs run on a second node.
5. Six runs per field: power against small shifts is limited, and at the
   reservoir filling the between-build spread is as large as the 0.02 V/Å
   response. A larger set, or a design that removes build-to-build variance from
   the primary test, would be needed to resolve it; neither was registered here.
6. Program length against physics was measured for two atomic liquids and two
   scalar restraints only, and its registered verdict is NOT SUPPORTED (section
   7). The negative finding bears on these restraints as specified, not on
   richer ones. Slab, glass and molecular systems have no restraint path.
7. Macrostate selection was tested at one state point, chosen in pilots, for one
   Lennard-Jones component, and its registered verdict is NOT SUPPORTED. A crystal
   that is a small fraction of the box can be lifted as a liquid.
8. Near melting the slab lift can misstate crystal orientation (`z <111>` for a
   crystal grown along [001]) and print a spurious strain. The slab builder does
   not read `orient`, but it does use the strain, so a misidentified orientation
   can alter a rebuilt slab's lattice spacing.
9. The criteria and the A5 gate are ours and were calibrated on these cases (section
   3). For fluids a pass mainly shows that MD works.

## Discussion

Two results bound what a macrostate language contributes. For the simple liquids
tested it contributed almost nothing beyond state variables: MD found the
structure, and a pass on the round-trip gate mainly verified the physics stage. For
homogeneous crystals and a liquid at one state point, the lifted program determined
the outcome and a control without structure did not. This falls short of a language
advantage. A builder given the word `fcc` could also start a crystal; what the test
establishes is the loop, from a snapshot with no hint, through text, to a matching
state. We have no baseline against another representation and no state where the
text does something a builder cannot. Kob–Andersen glasses show the other limit:
they keep a memory of their preparation that structural statements do not resolve.

As an instrument, the printed dipole statement agreed with a direct calculation
and reproduced the registered field contrasts. The physical conclusion is narrow: a
response at a pilot packing, none confirmed at a packing of the measured uptake. We
do not attribute the difference to filling, because composition, neutralization and
packing protocol changed together.

Testing against pre-registered rules cost us two claims: neither the program-length
curve nor macrostate selection was supported. The pilots for the latter also found
a compiler defect and a decompiler defect that had passed the acceptance set. No
decision model has been trained on Chaord text here, and whether the language helps
select training configurations remains to be tested.

## Methods

**Electrolyte packing.** The pilot filling is 74 EC, 48 EMC and 12 LiPF6 per 98.4 Å.
A seeded rejection packer places molecules
against contact floors of max(2.0 Å, 2.2 Å with H, 0.85 σᵢⱼ), and molecules that
cannot be placed strictly are inserted and pushed apart by translation.

At the reservoir filling, the reservoir composition jammed this
packer: at floor scales 1.0–0.85 insertion failed on EC, and at 0.8–0.7 insertion
completed but translation alone could not recover the true floors. Following the
pre-registered staged policy, and without changing counts, we insert at 0.8 × the
floor and relax with rigid-body LAMMPS dynamics, so molecules can rotate as well
as translate. The 90 ps relaxation uses only a purely repulsive soft pair
potential ranged on the true floors and a radial wall, with no charges or LJ
terms. A build is accepted only if it clears the unmodified floors. The first
attempt on `r13_b1` missed the wall floor (0.9985 against 0.999), so the wall was
stiffened and the threshold left alone; the result has minimum pair/floor 1.098
and wall/floor 1.028 (`research/cnt_electrolyte/r13/data/build_manifest_r13_b1.json`). Starting production
from this state releases residual strain: the mobile temperature spikes to about
880 K within the first picosecond and is gone within 5 ps, well before any
analysed data (`research/cnt_electrolyte/r13/evidence/smoke/`). That transient was measured on `r13_b1`
only. All six builds pass the same gate (minimum pair/floor 1.097–1.100,
wall/floor 1.024–1.031; `research/cnt_electrolyte/r13/data/build_manifest.json`). Builds 1 and 6 were
inserted at 0.80 × the floor. Builds 2–5 jammed there on an EC insertion and were
inserted at 0.75 ×; every jam is recorded and no count was changed.

**Noise floors and the A5 gate.** The reference for each case is the mean of its
observables over decorrelated frames, and a rebuild is the mean over seeds 7, 13
and 29. The floor for each observable is the larger of the mean and the 90th
percentile of the distances between pairs of averaged reference observables. A rebuild passes if
both the g(r) and coordination distances are at most 1.5 times their floors.

**Statistics.** Field contrasts are Welch two-sided tests at α = 0.05 with the
independent run as the unit. Zero-field values are run means; field-on values are
production block B. Each decision rule was registered in a committed plan before its data
existed. The field tests ran from committed scripts; the program-length script was
committed only with its results.

**Macrostate runs.** Liquid and slab references may hold one more frame than the 10
planned, because their dump starts on a multiple of the dump interval; frames were
not retained, so this is inferred. The common-neighbour cut-off is set in
`research/macrostate_selection/matrix.py`.

**Software and reproducibility.** The statistics in this paper are printed by
`docs/paper/scripts/make_paper_assets.py` from committed evidence, and the script
stops if a recomputed statistic disagrees with the stored one. Other values name
their source file. Corrections,
including defects found by the tests reported here, are listed in
`docs/records/corrections.md`.

## Figures and tables

- **Table 1.** Statement kinds (section 2).
- **Table 2.** Acceptance A1–A14 (`assets/table1_acceptance.md`).
- **Table 3.** Per-case A5 distances (section 3).
- **Tables 4 and 5.** Confirmatory field tests (section 6).
- **Table 6.** Program-length grid (section 7; `research/program_length/RESULTS.md`).
- **Figure 1.** `figures/fig4_example_program.chaord`. Lifted program for a
  crystal–liquid slab.
- **Figure 2.** `figures/fig3_fidelity.pdf`. Chaord statement against direct
  calculation on the 18,018 frames of the first confirmatory set, and the residual,
  bounded by the print half-quantum.
- **Figure 3.** `figures/fig5_filling.pdf`. Pilot packing against open-tube uptake
  per species and tube.
- **Figure 4.** `figures/fig2_blind_p1_vs_field.pdf`. P1 against field. (a) Field
  ladder, mean ± s.e.m. over three builds; (10,10) jobs at some fields used a newer
  LAMMPS build. (b) (16,16): the ladder and first confirmatory test, both packed
  below open-tube uptake, and the second confirmatory test at a packing of the
  measured uptake; markers are means ± s.e.m., dots individual runs of the two
  confirmatory tests, series offset horizontally for legibility.
- **Figure 5.** `figures/fig6_macrostate_matrix.pdf`. Largest distance over noise
  floor for each rebuild (rows) against each reference (columns). (a) Lifted
  programs. (b) Control programs. Outlined cells pass the 1.5× gate.

## Data and code availability

Code, programs, evidence and the scripts that print every number are at
https://github.com/WhiteCrosstheRiver/Chaord. An archived release with a DOI will
accompany publication.

## Author contributions and competing interests

**[To be supplied by the authors.]**

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
   2157 (2009). (Packmol)
6. A. I. Jewett et al., J. Mol. Biol. 433, 166841 (2021). (moltemplate)
7. P. Li, Building Atomic Structures from Natural Language, arXiv:2609.29595
   (2026). (ATLAS)
8. N. Alampara, S. Miret, K. M. Jablonka, Less can be more for predicting
   properties with large language models, arXiv:2406.17295 (2024; first posted
   as MatText).
9. A. M. Ganose, A. Jain, MRS Commun. 9, 874 (2019).
10. H. Xiao et al., Nat. Commun. 14, 7027 (2023). (SLICES)
11. A. P. Thompson et al., Comput. Phys. Commun. 271, 108171 (2022). (LAMMPS)
12. R. W. Hockney, J. W. Eastwood, *Computer Simulation Using Particles*
    (McGraw-Hill, 1981). (PPPM)
13. R. L. McGreevy, L. Pusztai, Mol. Simul. 1, 359 (1988). (reverse Monte Carlo)
14. J. D. Honeycutt, H. C. Andersen, J. Phys. Chem. 91, 4950 (1987). (common
    neighbour analysis)
