# Post-gate research decisions: language extensions for the confined-electrolyte target

Decision record for the first research use (open_items_v2.md section 5, item 2:
"Language"). Each decision follows the D9 path: this written decision first,
then a red test, then the code. Owner-written, 2026-10-14. The observables
(dipole orientation, radial density) and the molecular templates (PC, EMC,
EC, PF6-, Li+) are sibling work packages and are deliberately not touched
here; this record covers only the three statements that carry them.

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
