# Post-gate research decisions: language extensions for the confined-electrolyte target

Decision record for the first research use (open_items_v2.md section 5, item 2:
"Language"). Each decision follows the D9 path: this written decision first,
then a red test, then the code. Owner-written, 2026-10-14. The observables
(dipole orientation, radial density) and the molecular templates (PC, EMC,
EC, PF6-, Li+) are sibling work packages and are deliberately not touched
here; this record covers only the three statements that carry them. RD4
(2026-10-14, same D9 path) extends the record to the first-result round
(section 5, item 3): the statement-level evidence the field leaves in a
lifted program.

## RD1 · A cylindrical region

**Decision.** The geometry gains a *finite* cylinder by extending the existing
`cylinder` shape instead of adding a second keyword:

```
cylinder axis z center 0. 0. 20. radius 10.0 A length 40.0 A
```

The pre-existing form (`cylinder axis z center x y radius r`, an
axis-infinite cylinder) parses and prints unchanged; the finite form adds the
axis coordinate of the centre (a third centre number) and a `length` — the
extent along the axis, centred on the centre coordinate, inclusive at both
ends. `IR.ShCylinder` gains optional `cz` and optional `length`.

**Why this shape.** The reviewer's sketch (`cyl z : radius … length …`)
adapted to the language's actual conventions (rule 8, reuse the notation; and
the closest existing region syntax, `sphere center x y z radius r`) is a
keyword-argument shape on the one keyword a cylinder already has. One concept,
one keyword: a second `cyl` shape next to `cylinder` would be two names for
one geometry. The optional parts keep every existing program and test valid.

**Builder semantics.** New module `src/chaord/build/regions.py`:
`shape_mask(shape, pos, cell_diag)` and `geometry_mask(geometry, pos,
cell_diag)` evaluate membership for **all six** shapes (`and` intersects,
`or` unions, `minus` subtracts, left to right — the printed order). Rules:

- geometry numbers are read in the frame's own unit system (A or sigma),
  exactly as the slab builder reads `slab z lo .. hi`;
- the cylinder's radial distance uses the minimum image in the two
  directions perpendicular to the axis; the axial window is inclusive and
  wraps through the periodic boundary when `lo > hi` (the wrapped-slab
  convention the slab lifter already prints);
- boundaries are inclusive: an atom exactly at `radius` or exactly at an
  end plane is inside.

**Lift.** Not inferred here (the classifier research decides when a region
may be *called* cylindrical); parse, canonical format (idempotent:
parse(format(x)) == x) and membership are exact and tested.
`tools/sketch_check.py` learns the same optional shape parts, staying
vocabulary-free.

## RD2 · A nanotube statement with chirality

**Decision.** A build statement in a crystal region (carbon dialect scope
`region_crystal`, the key already declared in `carbon.yaml`):

```
nanotube (10,10) length 39.36 A
```

`length` is the target; the built tube is **commensurate**: the nearest whole
number `N_T` of translation periods (a tube periodic along its axis needs an
integer count), so the built length is `N_T x |T|` and the statement, the cell
and the coordinates always agree. A new lexer terminal `CHIRAL` (`(n, m)`,
rule 8's "(n,m)" notation) and IR value `Chirality(n, m)` carry the pair;
`fmt` prints it canonically as `(n,m)`.

**Build.** `src/chaord/build/nanotube.py::build_nanotube_region` constructs
the standard ideal geometry (Saito, Dresselhaus & Dresselhaus, *Physical
Properties of Carbon Nanotubes*, ch. 3): graphene lattice `a = 2.46 A`
(exact-geometry constant, dialect-exempt like the prototype tables), `a1 =
a(1,0)`, `a2 = a(1/2, sqrt(3)/2)`, two-atom basis `(0,0)` and `(a1+a2)/3`;
`Ch = n a1 + m a2`, `|Ch| = a sqrt(n^2+nm+m^2)`, `d = |Ch|/pi`; `T = t1 a1 +
t2 a2` with `t1 = (2m+n)/d_R`, `t2 = -(2n+m)/d_R`, `d_R = gcd(2m+n, 2n+m)`;
the sheet is rolled on the tube axis, `R = |Ch|/(2 pi)`. Atom count per
period `4(n^2+nm+m^2)/d_R` — the construction checks it lands exactly. Axis
and centre come from the region's cylinder geometry when the region has one
(the two features compose: the tube sits in its region), else axis `z` and
the cell centre; a stated cell must fit the tube and match the axial period
within `lattice_match_tolerance` (fail closed); the region cylinder's length
must match the built length and its radius may be the tube's or larger (a
wider pore around the tube) but never smaller; with no stated cell the box
is `d + 2 x nanotube_vacuum_margin` (new carbon threshold) square and the
built length tall. Species from `conserve atoms` (default C), count checked.

**Lift.** `src/chaord/lift/nanotube.py::infer_nanotube_chirality(positions,
cell) -> NanotubeChirality(n, m, length)`. The tube axis is the covariance
eigenvector with the LARGEST eigenvalue, snapped to a cell axis (otherwise
refused) [corrected 2026-10-14: the record originally said "smallest",
which only holds for tubes shorter than sqrt(6)*R ~ 2.45 R — axial
variance L^2/12 exceeds the transverse R^2/2 beyond that, and every
acceptance tube is 2.3-6.1x longer than the crossover, so the smallest-
eigenvalue pick lands in the perpendicular plane; caught by the
implementation agent and confirmed on the record's own fixtures];
every atom must sit on one radius (a fcc carbon frame is refused
here — "fail closed", never a wrong chirality). The tube is unrolled to the
sheet (`x = R*phi`, `y = axial`); the graphene lattice is recovered from the
data itself (second-neighbour shell, `a = sqrt(3) x d_NN`), the zigzag
direction is the lattice direction nearest the circumference, and `(n, m)`
are the integer coordinates of the circumferential lattice vector `Ch` in
the (zigzag, +60-degree) basis — canonical `0 <= m <= n` by construction.
`|T|` is the shortest axial lattice period; `length = round(L_axis/|T|) x
|T|`. Consistency gates before returning: `|Ch| = a sqrt(N)`, `|T| =
sqrt(3)|Ch|/d_R`, and the atom count `4 N N_T / d_R` equals the frame's.

## RD3 · An external-field statement

**Decision.** A system-block state statement, next to `state T` / `state P`
(an external field is macrostate metadata of exactly that kind):

```
state field E 0.10 V/A along z
```

kind `state`, key `field`, declared in `carbon.yaml keys.system` (the
confined-carbonate research target is a carbon-dialect system); the unit word
`V/A` joins the carbon dialect's units. It is program metadata: the build
places no charges and moves no atoms (like `model`), and the conservation
check is untouched.

**Lift.** No frame can measure a field, so the value is **always**
caller-supplied: `lift_frame(frame, dialect, T=None, mode="auto", field=None)`
with `field = {"component": "E", "value": 0.10, "unit": "V/A", "axis": "z"}`
(component default `E`, unit default `V/A`). Given a field, every lift path
prints the `state field` statement in the system block and, following the
W11 assumed-value rule exactly as `fluid.AssumedT` does it, a provenance
note recording that the value was assumed, not measured. Given none, nothing
is printed (there is no honest default field).

## Registration (A14 discipline, per AGENTS "a new statement key ships complete")

- grammar/IR/fmt: `cylinder … [cz] … [length q]`, `CHIRAL`, `Chirality`;
  `tools/sketch_check.py` ditto.
- dialects: `carbon.yaml` — `keys.system: [field]`, `units: [V/A]`,
  `thresholds.nanotube_vacuum_margin: 10.0 A` (`nanotube`, `chirality`,
  `stacking` were already declared in `region_crystal`).
- tests: `tests/test_cylindrical_region.py`, `tests/test_nanotube_statement.py`,
  `tests/test_field_statement.py` (red first); example count in
  `tests/test_grammar_examples.py` 10 -> 11.
- docs: `docs/reference.md` geometry line + `field` row (citing the actual
  example lines); one new example `spec/examples/11_nanotube_field.chaord`
  exercising all three statements.

## RD4 · The field's measurable consequence, asserted (first-result round)

Owner-written 2026-10-14, before the code (D9 path; red tests in
`tests/test_field_lift_statements.py`). The trigger is the reviewer's
done-when for the first research result (open_items_v2.md section 5, item 3):
"the programs' **statements**, not the coordinates, separate the states".
RD3's `state field` line is caller-supplied metadata — it names the knob, not
what the knob did. This record adds the statement-level evidence.

**Decision.** When the caller supplies `field=`, the lifted program asserts
the field's measurable consequence in the system block, immediately after
the `state field` statement:

```
state field E 0.2 V/A along z
assert field_dipole_p1 0.821 +- 0.277 along z
```

The value is measured, never assumed: `cv.field.dipole_orientation(frame,
dialect, axis)` on the frame the lift was given — the frame's own per-atom
charges (`frame.info["charges"]`, what the LAMMPS reader recorded) when
present, else the dialect's site-charge table (unchanged RD3-era fallback).
The key is the registered CV's name (`field_dipole_p1`, cv/field.py
registry), so rule 7's one-definition-per-quantity holds: assert, measure
and (later) restrain dispatch to the same code. The statement's provenance
note records that the value was measured and from which charge source — the
exact honesty marker that distinguishes it from the assumed `state field`
line above it.

**The tolerance is the a-priori standard-error bound, not an empirical
floor.** `assert … v +- tol` prints the spread of the measurement
(the fluid arm's convention: `cn 4.2 +- 0.5`). `dipole_orientation` returns
only the mean and the molecule count, and cv/ is frozen this round, so the
printed tolerance is `1/sqrt(n_molecules)`: each cos θ lies in [-1, 1], so
Popoviciu's variance bound (T. Popoviciu, Mathematica (Cluj) 9, 129 (1935):
a variable in [a, b] has variance at most (b-a)^2/4 = 1 here) caps the
standard error of the mean of n cosines at 1/sqrt(n). The note says so
explicitly ("the a-priori standard-error bound … not an empirical floor");
the empirical noise floor of the first-result analysis is measured
independently from frame pairs (the repo's standing estimator) and never
enters the program text.

**No radial assert.** The reviewer's observable list marks the radial
profile's summary scalar as optional here, and on the confined research
cells it is undefined as specified: `inner_outer_ratio` splits the
cell-centred profile at half the shorter transverse box edge, and the outer
shell of a tube-in-vacuum cell is empty — `cv/field.py:280` divides by the
zero outer mean and raises. Recorded as an open finding against cv/field.py
(not fixed there this round; the file is frozen for this work package). The
raw radial profile remains measurable and is reported in the analysis.

**Degrade with a note, never a silent absence, never a broken RD3.** If the
measurement refuses under the dialect in use (no `field_dipole_species`
threshold — e.g. the RD3 neon test lifts under `core + lj` — no named
species in the frame, no charge model), the `state field` statement stands
exactly as RD3 binds it ("every lift path prints it") and the provenance
block gains `field observables not asserted (<reason>)`. Refusing the whole
lift instead would break RD3's binding contract for dialects that simply do
not define the observable; the note keeps the absence visible in the text.
With `field=` absent, nothing field-related is printed (no statement, no
assert) — RD3 unchanged.

**Species view.** The carbonate solvents do NOT join the dialect's default
`field_dipole_species` ([H2O], unchanged): the site-charge fallback table is
keyed by ELEMENT, and EC's carbons carry two different published charges
(Nutakki 2023 Table III: CO +0.99, CT +0.22 e), so an element table cannot
express the model — the yaml's own join condition ("when their site-charge
models are pinned") is not met, and pinning an averaged wrong charge model
would be dishonest. Instead the first-result pipeline passes a
**measurement view** over the loaded dialect — the same Dialect object with
`field_dipole_species = [C3H4O3, C4H8O3]` — exactly the way a caller picks
a centre for the radial profile. The research dumps carry per-atom charges,
which win over the table, so the table is never consulted for these frames;
a chargeless carbonate frame under the view fails closed ("no site-charge
model … and the frame carries no charges"). Nothing is tuned: no numeric
threshold changes, and the CV code is untouched (rule 7 — one definition
per quantity; the view only names WHICH neutral solvent species the average
runs over).

**Vocabulary.** The assert key `field_dipole_p1` is declared in
`research.yaml keys.system` — the one statement key that names the
observable the research dialect's own thresholds define (the yaml's
"no statement vocabulary" note narrows to exactly this exception). The
grammar needs no change (a system-block body already accepts any
`KIND? key values…` statement); `docs/reference.md` gains the row.

**The composed research program (first use, not a lift change).** The four
research systems are mixed frames (a frozen ideal tube + a confined
molecular fluid) that no single cascade arm serves: `mode="auto"` refuses
the full frame ("extended bonded component (800 C atoms)"), and the
pipeline's single-engine chain degrades everything to residual coordinates —
the exact opposite of "statements, not coordinates". RD1 already deferred
region-lift composition to "the classifier research", i.e. to this round.
The composition therefore lives in the analysis pipeline
(`research/cnt_electrolyte/analyze_first_result.py`): the frozen carbon
sublattice lifts through RD2's `infer_nanotube_chirality` into a crystal
region (`nanotube (10,10) length 49.2 A` inside its RD1 cylinder), and the
electrolyte lifts through `lift_frame(..., mode="fluid")` under a
**composition dialect**: the research stack plus two recorded views — the
solvent species view of RD4's P1 measurement, and
`fluid_max_bonded_component = 15` (the registry's largest molecule of this
solvent set, EMC; molecular.yaml itself names the larger SPECIES templates
"routing exceptions, not a raised bound", and `bonded_single_phase` still
performs its real check: every bond-graph component one molecule of at
most 15 atoms). Without the routing view, mode="fluid" refuses on EC's 10
atoms and mode="auto"/"slab" pay a 100–300 s Wigner-Seitz / thin-film scan
before reaching the same fluid lift. Region statements follow
`pipeline._fit_fluid_region`'s recipe: density and g(r) recomputed over
the region volume (a sub-region is not the box), cn as lifted.

**RD2's inference is fed the full-precision frozen sublattice, with one
recorded workaround.** The production dump rounds coordinates to 6
significant digits, which cannot pass RD2's exact single-radius gate
(1e-6 relative), so the chirality comes from `final.data`'s tube
(1e-6-A text precision; the tube is frozen, so it IS every frame's tube —
the pipeline asserts the dump's tube matches it within 1e-3 A every
frame). Second, `infer_nanotube_chirality` re-anchors perpendicular
coordinates by minimum image, which is unique only below half the
perpendicular cell edge: the cnt16 tube's diameter (21.71 A) exceeds its
cell's half-width (20.86 A), so the inference runs with the two
perpendicular cell lengths tripled — no wrap actually occurs in either
system (each tube sits centred, entirely inside its box), the axial
period and every consistency gate are unchanged, and the cnt10 inference
is identical with and without the tripling. Fixing the anchoring inside
`lift/nanotube.py` is next round's item, not this round's.
