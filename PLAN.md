# Chaord: Compiler & Decompiler Plan

Plan v0.1 · 2026-09-27 · Markdown copy of the shared plan document, for agents working in the repo.

Name the language **Chaord** (chaos + order, said "KAY-ord"). It writes order exactly where a system has it and statistics where it does not, from crystals to gases. Build it in nine milestones over about 20 weeks; M0–M4 (about 12 weeks) already give a working compiler and decompiler for crystals, defects, fluids and interfaces. A milestone closes only when its acceptance tests pass, run by an agent that did not write the code.

## Name: Chaord

**Chaord** (chaos + order; Chinese 混序) names the axis the whole language rests on: how far exact description reaches before statistics take over. A crystal is nearly all order, a glass is frozen chaos with short-range order, and a liquid is moving chaos with local order. A gas is chaos; a reactive interface is where they meet and convert.

Tagline: **describe the macrostate, sample the microstate.** A program is a macrostate; a coordinate file is one microstate of it.

| Item | Convention |
| --- | --- |
| File extension | `.chaord` |
| Python package and CLI | `chaord` |
| Compile | `chaord build in.chaord -o out.extxyz --seed 1` |
| Decompile | `chaord lift dump.lammpstrj --frame -1 -o out.chaord` |
| Check asserts | `chaord check in.chaord out.extxyz` |
| Canonical format | `chaord fmt in.chaord` |
| Compare two programs | `chaord diff a.chaord b.chaord` |
| Round-trip test | `chaord roundtrip dump.lammpstrj --samples 5` |
| Dialects | `core`, `metal`, `ionic`, `molecular`, `surface`, `glass`, `carbon`, `lj` |

Names considered: Ordo (Latin "order": two small languages and the PyPI name `ordo` already exist), Hyle (Greek "matter": clashes with the HYLE philosophy-of-chemistry journal), Macrostate (exact semantics but a textbook term; kept as the tagline), Tripoint (TriPOINT is an existing bioinformatics tool), Omniphase (an existing company). A quick web search found no software named Chaord. Reserve the PyPI package, GitHub organisation and a domain before announcing it.

## Design rules

These ten rules bind every component; a change that breaks one is rejected in review.

1. **A program is a macrostate.** It describes a family of configurations; a coordinate file is one sample of it. A crystal is the narrow limit, a gas the wide one.
2. **Every statement has a kind** that says how the compiler honours it: build, state, constrain, assert, history or conserve (next section).
3. **One structure, one text.** The canonical form is declarative, one statement per line, printed by `chaord fmt`. Rotated, translated, re-ordered or re-imaged inputs lift to byte-identical text.
4. **Physics is the compile target.** Every program names a physics backend, like a compiler's target platform. A better backend needs fewer statements for the same result.
5. **Dialects fix the rules.** Thresholds and definitions (solid-like cut-off, coordination cut-off, bond rules) live in versioned dialect files and are never re-fitted per frame. Every lifted program records its dialect versions.
6. **Never drop an atom.** Counts per species and total charge are conserved exactly. Whatever the decompiler cannot explain goes into a `residual` block as raw coordinates.
7. **One definition per quantity.** Each collective variable is defined once and used three ways: measured by the decompiler, restrained by the compiler, checked by `assert`.
8. **Reuse existing notation.** Kröger–Vink for point defects, Burgers vectors, CSL Σ, Miller indices, Wood notation, adsorption-site names, (n,m) for nanotubes, SMILES, SSIP/CIP/AGG ion pairing, Voronoi indices.
9. **The round trip is the correctness test.** Lift, build, lift again: held-out observables must match within the noise between two frames of the same simulation.
10. **Deterministic.** The same input, dialect and version give the same text; the same program, seed and backend give the same coordinates.

## The language at a glance

Your assembly analogy maps onto four levels. The `.chaord` text holds only the top two, and every statement carries one of six kinds.

| Level | Holds | Analogy |
| --- | --- | --- |
| L0 coordinates | atoms, cell, species (extxyz, LAMMPS, POSCAR, CIF) | machine code |
| L1 lattice IR | lattice, basis, site occupancy, displacement field | assembly |
| L2 objects | regions, phases, molecules, defects, interfaces, adsorbates | structured code |
| L3 intent | state, history, constraints, assertions | declarative spec |

L0 and L1 appear in the text only inside `residual` blocks; the compiler and decompiler use them internally.

| Kind | Meaning | Compiler does | Decompiler emits | Example |
| --- | --- | --- | --- | --- |
| build (no keyword) | exact, up to symmetry | constructs it | the fitted structure | `defect V_Ni count 1` |
| `state` | equilibrium state; physics fills in the rest | equilibrates with the backend | measured T, P, density, composition | `state T 300 K` |
| `constrain` | a feature physics will not keep alone | restrains its variable while sampling | features the dialect marks as non-equilibrium | `constrain strain zz +0.9 % +- 0.2` |
| `assert` | a test, never enforced | measures, reports pass or fail | fingerprint statistics | `assert cn 4.0 +- 0.1 cutoff 2.85 A` |
| `history` | the process is the shortest description | runs the protocol | only from provenance metadata | `history melt 3000 K for 20 ps -> quench to 300 K at 1 K/ps` |
| `conserve` | exact totals | enforces them exactly | exact counts | `conserve atoms Ni 646 Al 217` |

**Grammar sketch.** Structure only; which keys exist, their arguments and units come from dialect files. Agents turn this into a Lark grammar in M0.

```
(* Chaord v0.1 grammar sketch. One statement per line; ";" may join two.
   Comments start with "#". Keywords are case-sensitive.
   Structural grammar only: which KEYs exist, their arguments and units come from dialect files. *)

program     = header , { block } ;
header      = "chaord" , VERSION , NL , [ "dialect" , NAME , { "+" , NAME } , NL ] ;
block       = system | physics | provenance | species | region | interface | residual ;
system      = "system" , body ;
physics     = "physics" , body ;
provenance  = "provenance" , body ;
species     = "species" , "{" , { specdef | NL } , "}" ;
specdef     = ( "molecule" | "ion" | "atom" ) , NAME , [ "=" , ( "smiles" | "file" ) , STRING ] , NL ;
region      = phase , NAME , ":" , geometry , body ;
phase       = "crystal" | "amorphous" | "liquid" | "gas" | "fluid" | "cluster" | "vacuum" ;
geometry    = shape , { ( "and" | "or" | "minus" ) , shape ;
shape       = "all" | "rest"
            | "slab" , AXIS , range
            | "box" , range , range , range
            | "sphere" , "center" , NUMBER , NUMBER , NUMBER , "radius" , quantity
            | "cylinder" , "axis" , AXIS , "center" , NUMBER , NUMBER , "radius" , quantity ;
interface   = "interface" , NAME , "|" , NAME , body ;
residual    = "residual" , ( "none" | body ) ;
body        = "{" , { statement | NL } , "}" ;
statement   = [ KIND ] , KEY , { value } , [ "+-" , quantity ] , ( NL | ";" ) ;
KIND        = "state" | "constrain" | "assert" | "history" | "conserve" ;   (* no KIND = build *)
value       = quantity | range | NAME | STRING | DIRECTION | FAMILY | PLANE | WOOD | KROGER_VINK
            | "->" | "@" | "+" | "=" ;
quantity    = NUMBER , [ UNIT ] ;
range       = quantity , ".." , quantity ;       (* a slab may wrap: 26.1 .. 12.3 in a periodic cell *)

(* Tokens
   NUMBER       3.572  -0.5  +0.9  1e12  1/3
   UNIT         A nm K bar GPa g/cm3 ML % deg ps fs K/ps (dialect list)
   DIRECTION    [001]  [1-10]            FAMILY  <110>
   PLANE        (110)  (1-11)            WOOD    p(2x1)  c(4x2)  (r3xr3)R30
   KROGER_VINK  V_O^..  Al_Ni  Ni_i  V_Ni^''
   STRING       "C1COC(=O)O1"  "NiAl.eam.alloy"
   NAME / KEY   Ni3Al  Li+  PF6-  Cr-Cr  Ti_5c  bridging_O *)
```

**Examples.** All seven pass the structural checker in the starter kit. The first is real decompiler output from the prototype; numbers in the others are illustrative.

**1 · Solid–liquid snapshot, lifted by the prototype (Lennard-Jones, 2,301 atoms)**

```
chaord 0.1
dialect core + lj

system {
  units lj
  cell 9.65 9.65 27.50
  pbc xyz
  state T 0.65
  conserve atoms X 2301
}

physics {
  backend lj
  epsilon 1
  sigma 1
  cutoff 2.5
}

crystal A : slab z 26.1 .. 12.3 {        # wraps through z = 0
  lattice fcc
  a 1.609
  orient x <100> z <001>
  constrain strain zz +0.9 % +- 0.2
  defect divacancy count 1 depth 5.2 form split          # 4 empty sites, 2 displaced atoms
  defect vacancy count 1 depth 6.1
  assert sites_matched 99.2 %
}

liquid B : slab z 12.3 .. 26.1 {
  state density 0.854
  assert cn 12.0 +- 1.1 cutoff 1.50
  assert gr_peak 1.06 height 3.07
  assert solid_clusters 0
}

interface A | B {
  at z 12.3
  width 1.1
}

interface B | A {
  at z 26.1
  width 1.9
}

residual none
```

**2 · Crystal with point defects (Kröger–Vink names)**

```
chaord 0.1
dialect core + metal

system {
  cell 21.432 21.432 21.432 A
  pbc xyz
  seed 7
  state T 900 K
  conserve atoms Ni 646 Al 217
}

physics {
  backend eam
  potential "NiAl.eam.alloy"
}

crystal matrix : all {
  prototype L1_2
  composition Ni3Al
  a 3.572 A
  orient x [100] y [010] z [001]
  defect V_Ni count 1
  defect Al_Ni count 1          # antisite: Al on a Ni site
}

residual none
```

**3 · Solid solution with short-range order**

```
chaord 0.1
dialect core + metal

system {
  cell 21.36 21.36 21.36 A
  pbc xyz
  seed 3
  state T 300 K
  conserve atoms Cr 288 Co 288 Ni 288
}

physics {
  backend mlp
  model "mace-mp-0"
}

crystal mea : all {
  lattice fcc
  a 3.56 A
  occupancy Cr 1/3 Co 1/3 Ni 1/3
  constrain sro alpha1 Cr-Cr +0.10 +- 0.02     # Warren-Cowley, first shell
}

residual none
```

**4 · Amorphous silicon from a melt–quench history**

```
chaord 0.1
dialect core + glass

system {
  cell auto cubic
  pbc xyz
  seed 5
  conserve atoms Si 512
}

physics {
  backend mlp
  model "mace-mp-0"
}

amorphous aSi : all {
  composition Si
  state density 2.28 g/cm3
  history melt 3000 K for 20 ps -> quench to 300 K at 1 K/ps -> anneal 300 K for 50 ps
  assert cn 4.0 +- 0.1 cutoff 2.85 A
  assert angle_mean 109.0 +- 1.5 deg
}

residual none
```

**5 · Liquid electrolyte (molecular fluid)**

```
chaord 0.1
dialect core + molecular

system {
  cell auto cubic
  pbc xyz
  seed 11
  state T 300 K
  state P 1 bar
  conserve charge 0
}

physics {
  backend mlp
  model "mace-mp-0"
}

species {
  molecule EC = smiles "C1COC(=O)O1"
  ion Li+ = smiles "[Li+]"
  ion PF6- = smiles "F[P-](F)(F)(F)(F)F"
}

liquid electrolyte : all {
  molecules EC 600 Li+ 45 PF6- 45                # about 1 M
  assert cn Li-O 4.0 +- 0.3 cutoff 2.6 A
  assert pairing CIP 0.3 +- 0.1                  # contact-ion-pair fraction
}

residual none
```

**6 · Gas mixture**

```
chaord 0.1
dialect core + molecular

system {
  cell auto cubic
  pbc xyz
  seed 2
  state T 300 K
  state P 1 bar
}

physics {
  backend classical
  forcefield "trappe-ua"
}

gas air : all {
  molecules N2 790 O2 210
  assert compressibility 1.00 +- 0.01
}

residual none
```

**7 · Reactive interface: rutile (110) in water with dissociation**

```
chaord 0.1
dialect core + ionic + surface + molecular

system {
  cell 26.63 25.99 60.0 A
  pbc xyz
  seed 9
  state T 330 K
}

physics {
  backend mlp
  model "mace-mp-0"
}

crystal rutile : slab z 0 .. 18 A {
  prototype rutile
  composition TiO2
  a 4.594 A
  c 2.959 A
  orient x [001] y [1-10] z [110]
  surface (110) top
  termination bridging_O
  defect V_O^.. count 2 layer surface          # bridging-oxygen vacancies
}

liquid water : slab z 18 .. 45 A {
  molecules H2O 620
  state density 1.0 g/cm3
}

vacuum gap : slab z 45 .. 60 A {
}

interface rutile | water {
  dissociate H2O -> OH @ Ti_5c + H @ O_br count 9
  assert coverage OH 0.25 ML +- 0.05
}

residual none
```

## Architecture

Build and lift form one loop through a shared IR:

```
 Program text (.chaord)  <->  Parser + formatter  <->  IR (typed tree: regions, phases,
 one canonical text           Lark, chaord fmt,         kind-tagged statements)
                              JSON schema                 |                  ^
                                                    build |                  | lift
                                                          v                  |
          Compiler (chaord build)                            Decompiler (chaord lift)
          1 check: overlaps, charge, stoichiometry           1 classify atoms; find molecules
          2 construct: lattices, defects, SQS, packing       2 segment phases; fit lattices exactly
          3 realize: equilibrate, run history, restrain      3 defects and lifting; region statistics
          4 assert: measure, report pass or fail             4 interfaces; residual; canonical print
                     |                                                  ^
                     +---- sample (seed) --> Coordinates -- snapshot ---+
                                             (extxyz, LAMMPS, POSCAR)
          round trip: lift(build(p)) must return p

 Shared by build and lift: CV library (measured, restrained, checked) | Dialects (versioned rule files)
                           Physics backends (LJ, EAM, MACE, LAMMPS + PLUMED) | I/O (ASE readers, writers)
```

The IR is the only thing both directions share: `chaord build` walks it down to coordinates, `chaord lift` walks coordinates back up, and the round-trip test closes the loop. The CV library is the other shared piece: one definition per quantity serves measurement, restraints and asserts.

**Reuse, don't rebuild.** Keep the core light (numpy, scipy, ASE, Lark, Pydantic); every heavy tool is an optional extra.

| Job | Reuse | Note |
| --- | --- |
| Parsing | Lark | the grammar sketch becomes the grammar file |
| IR and JSON schema | Pydantic | typed models; schema for LLM tool calls |
| Units | pint | parse and convert every quantity |
| Structures and I/O | ASE, pymatgen | builders, file formats, StructureMatcher |
| Symmetry | spglib | space groups, standard cells |
| Local-structure passes | OVITO Python module | PTM, CNA, Wigner–Seitz, DXA, grain segmentation; check its licence |
| Molecules | RDKit | SMILES to graphs and conformers |
| Packing | Packmol | molecules into regions |
| Solid solutions | icet | SQS and cluster correlations |
| Physics backends | ASE calculators, MACE, LAMMPS | the compile targets |
| Restraints | PLUMED | how `constrain` is enforced |
| Ordered-half design | ATLAS (arXiv 2609.29595) | read its component algebra before designing builders |

## Roadmap

Core compiler and decompiler are done by week 12; v1.0 by week 20 (estimated weeks, agent-driven work with one human reviewer).

| Milestone | Weeks |
| --- | --- |
| M0 Foundations | 1–2 |
| M1 Crystals | 3–4 |
| M2 Defects, solid solutions | 5–6 |
| M3 Fluids | 7–9 |
| M4 Interfaces, surfaces | 10–12 |
| **Gate A: core complete** | **end of week 12** |
| M5 Amorphous, history | 13–15 |
| M6 Reactive, extended defects | 16–18 |
| M8 Integrations (parallel) | 17–20 |
| M7 Benchmark, v1.0 release | 19–20 |
| **Gate B: v1.0** | **week 20** |
| Benchmark data (continuous) | 1–20 |

Gate A (end of week 12): the M0–M4 exit criteria pass on the benchmark, checked by a verification agent, and the statistical tolerances are frozen. Gate B (week 20): every v1.0 acceptance criterion passes.

## Milestones in detail

Each milestone adds one family of matter end to end: builder, lift passes, statistics, tests and examples together.

| Milestone | Weeks | Deliverables | Exit criteria |
| --- | --- | --- | --- |
| M0 Foundations | 1–2 | Repo and CI; Lark grammar; IR models and JSON schema; `chaord fmt`; ASE-based I/O; dialect loader with `core`; CV library skeleton; prototype ported into the package | All spec examples parse; `fmt` is idempotent on 10,000 generated programs; text → IR → text is lossless |
| M1 Crystals | 3–4 | Builders for bulk, supercells, orientations and slabs; lift passes that classify, segment and fit (lattice, orientation, space group); canonicaliser | Exact round trip on every crystal case; identical text under rotation, translation, re-ordering and re-imaging |
| M2 Defects, solid solutions | 5–6 | Vacancies, interstitials, antisites and Frenkel pairs with Kröger–Vink names; the lifting pass for defect complexes; occupancy and Warren–Cowley SRO; SQS via icet | Planted defects recovered with precision and recall ≥ 0.95; net counts exact; SRO within ±0.02 of target |
| M3 Fluids | 7–9 | Molecule templates (RDKit); packing (Packmol); `state` via LJ, EAM and MACE backends; fluid CVs; noise-floor tool; statistical round-trip harness | Held-out observables within 1.5× the noise floor on ≥ 90% of fluid cases; composition exact |
| M4 Interfaces, surfaces | 10–12 | 3-D phase segmentation; interface objects with profiles and widths; Miller surfaces, terminations, Wood notation; adsorbates and coverage | Interface position within 0.5 Å; per-atom phase labels ≥ 95% correct; widths within the noise floor |
| M5 Amorphous, history | 13–15 | Protocols (melt, quench, anneal, deposit); PLUMED restraints for `constrain`; glass statistics (rings, Voronoi); a controller that picks the shortest program passing the round trip | Constrained quantities held within tolerance; amorphous held-out statistics within 1.5× the noise floor on ≥ 80% of cases |
| M6 Reactive, extended defects | 16–18 | Species census from bond graphs; reactions such as `dissociate`; reactive interfaces; dislocations (DXA); grain boundaries (CSL Σ) | Census exact on planted cases; Burgers vectors and Σ values correct on the benchmark |
| M7 Benchmark, v1.0 | 19–20 | Full Chaord-Bench run; performance work; reference manual; three tutorials; release | Every v1.0 acceptance criterion passes |
| M8 Integrations | 17–20 | Scripting front-end that lowers to canonical text; LLM tool schema; Laya state encoder (a program diff under 320 tokens); active-learning hooks | LLM-written programs compile on the first try for ≥ 90% of 50 test prompts |

## Working with AI agents

Give each agent one work package, one branch and its tests. Nothing merges without green CI, and no gate closes without a verification agent's report. Binding rules are in `AGENTS.md`.

```
chaord/
  AGENTS.md          rules every agent reads first
  PLAN.md            this plan
  spec/              grammar, language reference, dialects/*.yaml, examples
  spec/examples/     the seven example programs
  src/chaord/
    lang/            lexer, parser, IR models, formatter, JSON schema
    cv/              collective variables: measure, restrain, check
    build/           lattices, defects, SQS, packing, interfaces
    realize/         physics backends, protocols, restraints
    lift/            decompiler passes
    check/           static checks, asserts, diff
    io/              readers and writers (ASE)
    cli.py
  bench/             Chaord-Bench generators, frames, ground truth
  tests/             unit, property, golden, round-trip, acceptance
  prototype/         the working Lennard-Jones demo
```

| WP | Milestone | Scope | Done when |
| --- | --- | --- | --- |
| 01 | M0 | Lark grammar, parser, line-numbered errors | All examples parse; 30 malformed files report the right line |
| 02 | M0 | IR models (Pydantic), JSON schema | Text → IR → JSON → IR → text is lossless |
| 03 | M0 | Canonical formatter `chaord fmt` | Idempotent on 10,000 generated programs |
| 04 | M0 | I/O and CLI skeleton | Reads and writes extxyz, LAMMPS data and dump, POSCAR, CIF |
| 05 | M0 | Dialect loader, `core` dialect, CV registry | No threshold in pass code; a CI check enforces it |
| 06 | M1 | Crystal builders | 12 prototypes match pymatgen StructureMatcher |
| 07 | M1 | Lift passes: classify, segment, fit | Exact round trip on the crystal set |
| 08 | M1 | Canonicaliser (ordering, naming, rounding) | Invariance suite passes |
| 09 | M2 | Point defects and the lifting pass | Planted-defect precision and recall ≥ 0.95 |
| 10 | M2 | Occupancy, SRO, SQS | SRO targets met within ±0.02 |
| 11 | M3 | Molecule templates and packing | No overlaps at target density; charge conserved |
| 12 | M3 | Physics backends and equilibration | Reproducible from a seed; temperature and energy stable |
| 13 | M3 | Fluid CVs and the noise-floor tool | CVs match brute-force reference code to 1e-6 |
| 14 | M3 | Statistical round-trip harness | Per-system report against the noise floor |
| 15 | M4 | 3-D segmentation, interfaces, surfaces | Phase labels ≥ 95% correct; interface position within 0.5 Å |
| 16 | M4 | Adsorbates, coverage, Wood notation | Site labels ≥ 90% correct |
| 17 | M5 | Protocols and PLUMED restraints | Constrained CVs within tolerance |
| 18 | M5 | Glass statistics and the shortest-program controller | Picks the shortest passing program on the benchmark |
| 19 | M6 | Species census, reactions, reactive interfaces | Census exact on planted cases |
| 20 | M6 | Dislocations and grain boundaries | Burgers vectors and Σ correct |
| 21 | all | Chaord-Bench generators | Each case has a script, ≥ 5 independent frames and ground truth |
| 22 | M8 | LLM schema, scripting layer, Laya encoder | M8 exit criteria pass |

**Review gates.** You approve spec, grammar and dialect-threshold changes, and sign each gate. A verification agent that did not write the code runs the acceptance suite at each gate and writes a one-page report. CI runs unit, property and golden tests on every pull request in under 10 minutes; the statistical round-trip benchmark runs nightly.

## Testing strategy

Ten test layers run at three cadences. The statistical ones are judged against a measured noise floor, never a guessed tolerance.

| Layer | Checks | Example | Cadence |
| --- | --- | --- | --- |
| Unit | each function | the lexer reads `V_O^..` as one Kröger–Vink token | every pull request |
| Property-based (Hypothesis) | parser and formatter laws | `fmt(fmt(p)) == fmt(p)`; `parse(print(ir)) == ir` on 10,000 generated programs | every pull request |
| Golden | exact expected output | a fixed snapshot lifts to a stored `.chaord` text | every pull request |
| Exact round trip | crystals, point defects | build, lift, compare byte for byte | every pull request (small), nightly (full) |
| Statistical round trip | fluids, glasses, interfaces | held-out observables against the noise floor | nightly |
| Known answer | defects, clusters, adsorbates, phases | plant 3 vacancies; lift must report a net 3 | every pull request |
| Invariance | canonical form | rotate, translate, re-order, re-image: same text; replicate: counts scale, intensive lines unchanged | every pull request |
| Robustness | temperature, noise, small regions | temperature sweep to 0.9 of melting; regions of 100 atoms | nightly |
| Cross-check | agreement with trusted tools | spglib space groups, OVITO structure counts, pymatgen StructureMatcher | nightly |
| Performance | speed and memory | lift 100,000 atoms | weekly |

**Noise floor.** For each system and observable, it is the distance between two independent frames of the same reference simulation. A statistical round trip passes when the rebuilt structure is no further from the original than 1.5× this floor. The prototype's Lennard-Jones case measured 0.12 for g(r) (RMS) and 0.015 for the bond-angle distribution.

**Chaord-Bench.** Each case stores its generator script, at least five independent frames and a ground-truth file.

| Category | Cases | Reference generator | Ground truth stored |
| --- | --- | --- | --- |
| Crystals | Cu, Fe, Mg, Si, NaCl, SrTiO3, Ni3Al | thermal MD (EAM, Tersoff, MACE) | prototype, lattice, orientation |
| Point defects | vacancy, divacancy, interstitial, antisite, Frenkel pair in Cu, Ni3Al, NaCl | EAM, MACE | positions and counts |
| Solid solutions | CrCoNi random and with SRO; CuAu | MACE plus Monte Carlo | composition, Warren-Cowley α |
| Liquids | Lennard-Jones, Al, Cu, water | LJ, EAM, MACE, classical water | T, density |
| Solutions | NaCl(aq), 1 M LiPF6 in EC | classical force fields, MACE | composition, counts |
| Gases | Ar, N2/O2, supercritical CO2 | LJ, TraPPE | T, P |
| Amorphous | a-Si, SiO2 glass, CuZr glass | melt–quench with Tersoff, MACE | protocol |
| Interfaces | Al solid–liquid, Si/SiGe, Cu/water, graphene/water, nanotube in water | EAM, Tersoff, MACE | interface positions, phases |
| Surfaces, reactive | Si(001)-(2×1), Pt(111) with O, rutile (110) with water | MACE | reconstruction, sites, coverage, reactions |
| Extended defects | edge dislocation in Cu; Σ5 grain boundary in Cu | EAM | Burgers vector, Σ |

## Acceptance criteria

v1.0 is accepted when all fourteen criteria pass on Chaord-Bench. The targets are starting values: calibrate them on the M1–M3 data, then freeze them at Gate A.

| ID | Criterion | v1. target | Measured by |
| --- | --- | --- | --- |
| A1 | Parse and format | 100% of spec examples parse; `fmt` idempotent on 10,000 generated programs | property tests |
| A2 | Canonical invariance | Byte-identical text under rotation, translation, re-ordering and re-imaging on 100% of crystal cases | invariance suite |
| A3 | Exact round trip, ordered matter | Rebuilt structure matches the original (StructureMatcher: ltol 0.2, stol 0.3, angle 5°) on 100% of crystal cases | round-trip suite |
| A4 | Defect recovery | Precision and recall ≥ 0.95 for planted point defects up to 0.8 of the melting temperature; net counts exact | planted-defect suite |
| A5 | Statistical round trip | Held-out distances ≤ 1.5× the noise floor on ≥ 90% of fluid and interface cases and ≥ 80% of amorphous cases | nightly benchmark |
| A6 | Conservation | Atoms per species and total charge in the program equal the input's on 100% of lifts | checked on every lift |
| A7 | Phase segmentation | Per-atom phase labels ≥ 95% correct against planted ground truth | interface suite |
| A8 | Reactive census | Molecular species counts exact on planted cases; adsorption sites ≥ 90% correct | reactive suite |
| A9 | Compression | Program ≤ 2% of the coordinate file for systems of ≥ 1,000 atoms (prototype: 1.1%) | benchmark report |
| A10 | Determinism | Same input gives byte-identical text; same program, seed and backend give identical coordinates on one platform | repeated runs |
| A11 | Speed | Lift 100,000 atoms in ≤ 2 min on one CPU core; build without physics in ≤ 1 min | performance suite |
| A12 | Static checks | 100% of seeded errors caught: overlaps, charge imbalance, impossible density, lattice mismatch | error suite |
| A13 | No crashes | Zero unhandled exceptions on the whole benchmark; unexplained atoms go to `residual` | benchmark run |
| A14 | Documentation | Every statement key has a reference entry and a passing example | docs build |

## Risks and mitigations

The biggest risk is scope: "all of materials" never finishes, so v1.0 covers only the Chaord-Bench systems.

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Scope creep | Nothing finishes | Gates; v1.0 is limited to the benchmark systems; new domains arrive later as dialects |
| Endless debate over statistical tolerances | No clear "done" | Tolerances are multiples of a measured noise floor, frozen at Gate A |
| Agents weaken tests or hard-code thresholds | False passes | `AGENTS.md` rules; a CI check for numbers in pass code; an independent verification agent |
| Heavy dependencies break installs | Blocked progress | Core runs on numpy, scipy and ASE; OVITO, LAMMPS, PLUMED and MACE are optional extras; CI in containers |
| Licences of third-party tools | Cannot redistribute | Check every licence in M0; own implementations of structure-matching passes as fallback |
| Physics backends too slow for nightly round trips | Slow feedback | Develop on LJ and EAM; run MACE cases nightly on small systems |
| Canonical form ambiguous (names, order, rounding) | Same structure, different text | A written canonicalisation spec with explicit ordering and rounding, enforced by the invariance suite |
| Misclassification near interfaces | Wrong defects and phases | Interfaces are objects of their own; defects are searched only deeper than the interface width plus 1.5 σ, as in the prototype |
| Overlap with ATLAS | Duplicated work; novelty questions | Reuse its component-algebra ideas for ordered matter; the new ground is the statistical half and the round trip |

## Sources

- ATLAS: Atomic Translation & Language for Automated Structures — https://arxiv.org/html/2609.29595
- Name checks: https://github.com/FrankBro/ordo · https://github.com/solo-vey/Ordo · https://pypi.org/project/ordo/ · https://www.hyle.org/ · https://bio.tools/TriPOINT · https://omniphase.io/
- Tools: Lark https://github.com/lark-parser/lark · Pydantic https://docs.pydantic.dev · pint https://pint.readthedocs.io · ASE https://gitlab.com/ase/ase · pymatgen https://pymatgen.org · spglib https://spglib.readthedocs.io · OVITO https://www.ovito.org · RDKit https://www.rdkit.org · Packmol https://m3g.github.io/packmol/ · icet https://icet.materialsmodeling.org · MACE https://github.com/ACEsuit/mace · LAMMPS https://www.lammps.org · PLUMED https://www.plumed.org · Hypothesis https://hypothesis.readthedocs.io
- Round-trip numbers, compression and the lifted example: the Lennard-Jones prototype in `prototype/`
