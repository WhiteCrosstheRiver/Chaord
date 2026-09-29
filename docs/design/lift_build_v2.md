# lift/build v2: segment-first pipeline and region composer

Design doc · 2026-09-28 · status: proposal (no implementation started).
Answers external review P1#6: "no general compiler/decompiler; lift is a
specialized try-cascade, build dispatches on fixed phase sets." Grounded in
PLAN.md (Architecture), AGENTS.md, `src/chaord/lift/__init__.py` (try-cascade),
`src/chaord/build/__init__.py` (phase-set dispatch), and the specialized
lifters under `src/chaord/lift/`. Non-goals: no grammar/IR changes, no new
physics, no OVITO dependency. The IR (`lang/ir.py`: Program ->
system/physics/provenance/species/region/interface/residual blocks) already
supports multi-region programs; v2 only changes how lift produces and build
consumes them.

## 1. What exists, precisely

`lift_frame(frame, dialect, T, mode="auto")` tries, in order, catching every
exception: crystal (spglib exact) -> crystal+defects (Wigner-Seitz) ->
amorphous -> surface (z-gap) -> fluid -> slab (z-profile); `mode` pins one arm.
Consequences: q6 is recomputed up to 4x on cascade misses; each lifter
re-implements segmentation for its own shape assumption (z-slabs, z-layers,
whole box); a frame that is e.g. crystal + amorphous + vacuum fits no arm and
falls through to a wrong or failing path. `build_program` dispatches on the
*set* of region phases: `{crystal}`, `{liquid,gas}`, `{amorphous}`,
`{crystal,vacuum}`, `{crystal,liquid}` — five hard-coded sets, everything else
raises `ChaordError`; there is no place where two regions meet.

Both directions already have the right pieces (3-D segmentation exists in
`lift/segment.py:phase_labels_3d`, interface meshes in `interface_mesh`,
defect-tolerant lattice fitting in `lift/defects.py:fit_crystal`); they are
wired to whole-frame assumptions instead of to regions.

## 2. v2 lift: one segment-first pipeline

### 2.1 Data contract

New module `lift/pipeline.py` defines two plain objects (no IR dependency
until the last stage):

```python
@dataclass
class LiftCtx:                      # computed once, shared by all stages
    frame: Frame                    # io.frames.Frame (pos, cell, symbols, pbc)
    dialect: Dialect
    pos: np.ndarray                 # np.mod(frame.pos, L), cached
    q6, cn, pairs: ...              # passes.qbar(pos, L, rc=q6_cutoff)   [S1]
    edges: list[tuple[int, int]]    # bond graph (molecular/covalent frames)
    labels: np.ndarray              # int region id per atom              [S2]
    regions: list[RegionPlan]

@dataclass
class RegionPlan:
    rid: int
    phase: str                      # crystal | amorphous | liquid | gas | vacuum
    idx: np.ndarray                 # atom indices (vacuum: empty)
    model: str | None               # selected region model, None = residual
    fit: dict | None                # engine output (sites, params, score...)
    seam_band: np.ndarray | bool    # True -> atom within an interface band
```

### 2.2 The seven stages

```
def lift_pipeline(frame, dialect, T=None) -> Program:
    ctx = classify(frame, dialect)          # S1
    ctx = segment(ctx)                      # S2
    for r in ctx.regions:                   # S3  per-region model choice + fit
        r.model, r.fit = fit_region(ctx, r)
    ctx = lift_defects_per_region(ctx)      # S4  lattice-model regions only
    ctx = region_statistics(ctx, T)         # S5  region-level CV asserts
    ctx = interfaces(ctx)                   # S6  region-pair interface blocks
    return assemble_program(ctx)            # S7  residual + one Program
```

**S1 classify.** One `qbar` call (q6, cn, pair list), one KD-tree, one bond
graph (`build.molecules.bond_graph` for molecular frames); every later stage
reuses these arrays, none recomputes them.

**S2 segment.** 3-D per-atom region labels:
- solid mask = `q6 > q6_solid`, flooded by the existing
  `segment.phase_labels_3d` (seeded region growing on the neighbour graph —
  direction-agnostic, already tested in `test_segment3d.py`);
- within the solid label, split connected components of *unlike* bond-angle
  signature (chi0 fcc/hcp/other from `lift/slab.py`) into candidate grains —
  stage 1 may keep grains merged and let the crystal engine fail, see 2.4;
- disordered atoms split into bonded-network component (amorphous candidate)
  vs rest (fluid) using the existing predicates from `lift/amorphous.py` and
  `lift/fluid.py` restricted to the subset;
- vacuum is not atoms: a 3-axis gap scan (generalizes `surface.has_vacuum`,
  which scans z only) marks empty slabs/boxes as vacuum regions;
- interface bands: atoms of two adjacent regions within
  `max(bulk_margin, w + q6_cutoff)` of their common boundary get
  `seam_band = True` (exactly the exclusion rule `lift/slab.py` line ~127
  applies today; PLAN states it as "interface width plus 1.5 sigma"). Band
  atoms are excluded from S3/S4 fits and from defect search.

**S3 per-region fit.** Each region independently selects and runs a model
(2.4). Crystal regions run a fit-engine cascade (2.3); fluid regions reuse the
fluid statistics machinery; amorphous regions reuse the glass CVs. Region atoms
are passed to existing kernels as a *subset frame* — same cell, same pbc:

```python
def subset_frame(ctx, idx):
    return Frame(pos=ctx.frame.pos[idx], cell=ctx.frame.cell,
                 symbols=[ctx.frame.symbols[i] for i in idx],
                 pbc=ctx.frame.pbc)
```

For site generation, ideal sites are tiled over the region's bounding box and
then filtered: a site belongs to the region iff it lies inside the region
geometry or within `site_zone_pad` of a region atom. Without the filter, empty
sites outside the region would count as phantom vacancies.

**S4 defects.** Only in regions whose S3 model is a lattice. Reuse
`defects.defect_diff` + `defects.group_defects` (Wigner-Seitz diff) and
`slab.lift_defects` (complex clustering/naming) on the region subset; the
Kröger-Vink statement assembly from `defect_program.py` moves as-is.
Extended defects (Burgers, CSL — `lift/extended.py`) stay a dialect-gated
per-region extension point; not required for stage 1.

**S5 statistics.** Region-level asserts from the CV library on the region's
atoms (`fluid._rdf_stats` already accepts a `pos=` override; `cv/glass.py`
ring/angle CVs take frames). Emission set and order are fixed by the dialect
(not chosen per region) to keep one-structure-one-text (see risk R3).

**S6 interfaces.** Adjacency from `segment.interface_mesh` on region labels
(which pairs touch); `at`/`width` from `segment.slab_interfaces` (10–90
crossing) for general boundaries; when the pair is exactly two z-slab regions,
use the legacy tanh-profile fit (`lift/slab.py:fit`) so the golden text stays
byte-identical. Reactive statements: `reactive.dissociation_from_census` is
already census-parameterized ("the census may come from any frame region") —
feed it the seam-band census.

**S7 residual + assemble.** Atoms claimed by no model — failed-fit regions,
unlabelled atoms, leftover seam-band atoms — go to the `residual` block as raw
coordinates. Exactly one Program is assembled: system/physics/provenance/
species, regions in canonical order (2.7), interface blocks, residual.
Conservation check runs here (AGENTS rule: after every lift).

### 2.3 Existing lifters mapped onto stages

| Module (lines) | Stage | Disposition |
| --- | --- | --- |
| `passes.py` qbar/pairs_within/otsu (57) | S1 | reuse verbatim, called once |
| `segment.py` phase_labels_3d, interface_mesh, slab_interfaces (258) | S2, S6 | reuse verbatim (public, tested) |
| `surface.py` has_vacuum (709) | S2 | reuse for z; small new 3-axis wrapper |
| `crystal.py` standardize/match_prototype/round_canonical (308) | S3 engine A | reuse verbatim on subset frame (clean bulk: spglib path) |
| `defects.py` fit_crystal/ideal_sites/defect_diff/group_defects (272) | S3 engine B, S4 | reuse; add site-membership filter (new, additive) |
| `slab.py` decompile/orientation/lift_defects (335) | S3 engine C, S4, S6 | wrap, do not rewrite: engine C = run `decompile` on the crystal subset and take its lattice/orientation/defect outputs (its own pass-1 interface outputs are discarded; S6 owns interfaces); `_circ_offset`, chi0 become imported helpers |
| `defect_program.py` statement assembly, SRO bootstrap (120) | S4/S5 | reuse verbatim |
| `fluid.py` _rdf_stats/census/is_single_phase (200) | S3/S5 | reuse; promote `_rdf_stats`, `_centers` to public names |
| `amorphous.py` network_edges/is_amorphous (128) | S2/S3/S5 | reuse verbatim on subset |
| `surface.py` _wood_statement/_identify_hkl/_classify_sites/_zone_census | S3 surface refinement | reuse; promote to public. Surface lifting becomes a refinement of a crystal region adjacent to vacuum (layers restricted to region idx) instead of a whole-frame path |
| `reactive.py` (43) | S6 | reuse verbatim |
| `extended.py` Burgers/CSL (312) | S4 ext-point | untouched in stage 1 (dialect-gated, M6 scope) |
| `__init__.py` try-cascade (72) | compat | moves to `lift/legacy.py` unchanged (stage 1–2), deleted stage 3 |

Rewrites, honestly counted: the *orchestration* is new (~350 LOC pipeline.py +
~250 LOC regions.py); the *numerics* are imported. The one grey zone is
`slab.decompile`, a ~130-line monolith mixing segmentation (replaced by S2)
with lattice fitting, interface fitting, defect clustering and statistics (all
still wanted); stage 1 wraps it as engine C, and extracting its kernels is a
stage-3 cleanup gated by equivalence tests.

### 2.4 Region model selection table (combinatorics control)

The cascade's real cost is trying every lifter on every frame. v2 replaces it
with one table evaluated per region on cheap diagnostics
(`solid_frac` = mean(q6 > q6_solid) over region atoms; `network_frac` = mean
degree >= 3; `bonds` = has molecular bonds; `rho` vs dialect). Rules, in
order; thresholds are existing dialect keys, no new numbers:

| # | condition | model | fit engines (in order) |
| --- | --- | --- | --- |
| 1 | region is single and covers all atoms, `solid_frac` > `1 - fluid_solid_fraction_max` | crystal | A (spglib) -> B (d_NN) |
| 2 | `solid_frac` >= majority, compact | crystal | A -> B -> C (slab wrap) |
| 3 | `network_frac` >= `amorphous_network_min` and disordered | amorphous | glass CVs |
| 4 | `bonds` true | liquid (molecular) | census + packing stats |
| 5 | otherwise disordered | liquid/gas by `rho` | fluid stats |
| 6 | no atoms | vacuum | none |

Worst case per region: 3 crystal engines, each with its own score gate
(`sites_matched`, `prototype_match_tol`). Regions below `region_min_atoms`
(PLAN robustness layer: 100) degrade to residual. A frame with M regions costs
at most 3M bounded fits, never the current unbounded product of lifters x
frames; the region count itself is capped (`pipeline_region_cap`) with
same-phase/same-params merging (risk R2).

### 2.5 Failure policy: degrade, never abort

`fit_region` returns `(None, None)` on any engine failure; the region then
walks one step down the fallback chain (crystal -> amorphous/liquid by
diagnostics -> raw) and finally lands in `residual`. Rules:

- a degraded region's atoms always appear in `residual` (rule 6: never drop an
  atom); `conserve` counts stay exact by construction;
- the degradation is recorded in `provenance` (`note region B: crystal fit
  score 0.31 < gate, residual`) so `chaord check` and the round-trip harness
  can see it; S1/S2 infrastructure errors (I/O, dialect missing) still raise —
  those are caller bugs, not data problems;
- the cascade's whole-lift failure mode (exception -> silently try the next
  lifter -> possibly wrong-but-plausible output) disappears: output is always
  a complete program, possibly with a fatter residual.

### 2.6 `mode=` compatibility layer

Stage 1–2 (old paths alive):

- `mode="pipeline"` (new): always the pipeline.
- `mode` in `{crystal, defects, amorphous, surface, fluid, slab}`: the exact
  legacy arm, moved verbatim into `lift/legacy.py` — today's semantics
  including its raise-on-pinned-mode behaviour.
- `mode="auto"`: legacy until the equivalence suite (4.1) is green, then
  flipped to pipeline by a one-line change in `lift/__init__.py`; CI runs both
  paths on all fixtures during the bake.

Stage 3 (old paths deleted): explicit modes become pipeline constraints —
`mode="fluid"` = S2 forced to a single fluid region; `mode="crystal"` = single
crystal region with no fallback chain (fit failure raises, same contract as
today). The CLI surface never changes.

### 2.7 Canonical output rules

Byte-stability needs pinned rules; these reproduce every existing golden:

- single region keeps the legacy per-phase name: `bulk` (crystal), `fluid`,
  `glass`; multi-region frames use `A, B, C, ...` ordered by (phase priority
  crystal > amorphous > liquid > gas > vacuum, then smallest atom index) —
  this matches the slab output `A` solid / `B` liquid;
- interface blocks sorted by (a, b) name pair; `A|B` before `B|A`, and legacy
  order (by `at`) is reproduced because the canonical sort is stable;
- statement order within a region: build -> state -> constrain -> assert ->
  history (same as current printers);
- rounding via `crystal.round_canonical` with the existing `canonical_*`
  decimals keys.

## 3. v2 build: region composer

### 3.1 Compile model

Replace phase-set dispatch with: **construct each region in local coordinates,
then join geometrically, arbitrate conservation at the join, realize once.**
This maps one-to-one onto PLAN's build steps (check / construct / realize /
assert): composition *is* the construct step for multi-region programs.

```
def build_compose(program, dialect, rng, physics=True, md_steps=None) -> Frame:
    cell = system_cell(program)                      # explicit or auto
    regions = [b for b in program.blocks if b.t == "region"]
    layout = place(cell, regions)                    # geometry -> placement
    local = {r.name: REGION_CONSTRUCT[r.phase](r, layout[r.name], dialect, rng)
             for r in regions}                       # per-region local frames
    seams = [b for b in program.blocks if b.t == "interface"]
    frame = join(cell, local, seams, layout, dialect)# transplant + register
    frame = arbitrate(frame, program, local, dialect)# global conservation
    run_static_checks(frame, program, dialect)       # overlaps, A12 style
    if physics:
        frame = realize(frame, program, md_steps)    # unchanged, global
    return frame
```

`REGION_CONSTRUCT`: crystal -> existing `build.crystal.build_crystal_region`
machinery restricted to the region's sub-box (integer repetition counts still
checked against the *global* cell so lattice commensurability stays a static
error); liquid/gas -> the packing half of `build.fluid.build_fluid` on the
sub-box (density from the region's `state density`); amorphous -> existing
melt-quench protocol on the sub-box; vacuum -> empty placement.

### 3.2 Placement, transplant, interface registration

**Placement.** Geometry expressions map to placements: `all` = identity;
`slab/box` axes give a translation (stage 2 keeps axis-aligned placements —
which is all the grammar emits today in tests and goldens; rotation-valued
placements wait for a real user). Slab ranges may wrap (`26.1 .. 12.3`): the
interval arithmetic mod L moves in from `build/slab.py:_slab_params`.

**Transplant.** Region-local atoms are translated by the placement and
concatenated; a single `wrap(pos, L)` at the end puts everything in the cell.
No stage mixes local and global coordinates in one KD-tree (risk R4).

**Registration (interface blocks).** For each `interface a | b`:
- crystal | crystal: match the two boundary 2-D nets. Reuse the net/HNF
  machinery in `lift/surface.py` (`_net_vectors`, `_hnf2`, promoted to
  public): find the smallest common supercell within
  `lattice_match_tolerance`; the fix is a rigid in-plane shift of region `b`'s
  placement (lattice atoms stay on their lattice). Incommensurate nets: the
  join reports a static error (A12), same as a mismatched cell today.
- crystal | fluid: no lattice matching; the interface plane is the packing
  wall — fluid packing treats the crystal boundary atoms as excluded volume.
- fluid | fluid, anything | vacuum: no-op.
- `dissociate` on an interface builds fragments at the seam after the join
  (existing `build.molecules` templates), consuming counts from arbitration.

### 3.3 Conservation arbitration at the join

Per-region construction produces atom counts; the program's `conserve atoms`
statements state the exact totals. The arbiter reconciles them *after* the
join, in a fixed slack order:

1. fluid regions: add/remove whole molecules (molecule counts are the slack
   variable; `molecules H2O 620` is a target the packer may miss by a few);
2. crystal regions: adjust `defect` counts (one vacancy = one atom);
3. still off -> `ChaordError` naming the region and the deficit (static-check
   style). Bulk crystal atoms are never silently deleted or duplicated —
   that is the compile-side reading of rule 6.

Overlap check (`check/statics`, `overlap_tolerance`) runs on the composed
frame; seam overlaps relax the fluid side (repack with the wall), never the
crystal side.

### 3.4 orient and geometry expressions

`orient` stays what it is today: a *region-internal* statement selecting the
lattice basis (`build.crystal.orient_matrix`). The composer is the first place
`orient` and geometry meet — a slab region whose surface normal disagrees with
its geometry axis is a static error (checked in `place`), not the current
silent z-only assumption. Realize stays global and unchanged (`realize/lj.py`
run_md, EAM, protocols), so equilibration and history need no changes.

## 4. Migration path

Three stages, each independently shippable and reversible; nothing merges
without green CI (AGENTS). Old paths stay importable until stage 3.

### Stage 1 — pipeline lift behind a switch

Scope: S1–S7 for the phase families the legacy cascade covers today (crystal,
defects, amorphous, surface, fluid, slab); extended defects stay legacy-only.
New code is additive; legacy behaviour is bit-for-bit preserved.

Work: new `lift/pipeline.py` (~350 LOC), `lift/regions.py` (~250),
`lift/legacy.py` (cascade moved verbatim, ~80), `tests/test_pipeline_equiv.py`
(~200), golden fixtures for the pipeline path. Edits to existing files:
`lift/__init__.py` (route mode -> legacy/pipeline, ~25 lines),
`dialects/core.yaml` (+4 keys: `region_min_atoms`, `pipeline_region_cap`,
`interface_band_margin`, `region_merge_tol` — one approval, listed in the PR),
and promoting to public names (rename only, no logic): `surface._wood_statement`,
`_identify_hkl`, `_classify_sites`, `_zone_census`, `fluid._rdf_stats`,
`fluid._centers`. Estimate: 9–11 files touched, ~1000–1500 LOC added, no
existing behaviour change.

Acceptance:
- every existing suite green unmodified (`test_crystal`, `test_defects`,
  `test_lift_slab`, `test_segment3d`, `test_surfaces`, `test_fluids`,
  `test_amorphous`, `test_reactive_interface`, `test_thermal_recovery`,
  `test_extended`, `tests/golden/lj_slab.chaord`) — via `mode="auto"` before
  the flip, via explicit modes after it;
- new equivalence test: for every fixture frame used by those suites,
  `format_program(lift_pipeline(f)) == format_program(lift_legacy(f))`
  byte-for-byte. Any unavoidable diff must appear on an enumerated exception
  list in the PR with a reason; an empty list is the target. This is the hard
  gate: it forces the pipeline to call the legacy numeric kernels rather than
  re-implement them;
- degradation tests: corrupt one region of a fixture (shuffle 5% of its atoms)
  -> lift succeeds, corrupted region in `residual`, `conserve` exact (A6),
  no unhandled exception (A13);
- performance budget: pipeline runtime <= 1.3x legacy on the bench frames
  (the single cached `qbar` should make cascade-miss frames *faster*); A11
  (100k atoms <= 2 min) still passes.

### Stage 2 — build region composer

Scope: `build_compose` covers every phase set the legacy dispatch covers, plus
at least one it rejects (crystal | amorphous | vacuum). Legacy builders remain
the default; the composer is selected by `CHAORD_BUILD_COMPOSER=1` (then
flipped) — same bake pattern as stage 1.

Work: new `build/compose.py` (place/join/arbitrate, ~300 LOC), region
construct adapters (~150, mostly the packing half of `build/fluid.py` and the
sub-box clip for `build/crystal.py`), `tests/test_compose_equiv.py` (~250).
Edits: `build/__init__.py` (routing), `build/fluid.py` (expose `pack_region`),
`cli.py` (flag), dialect keys only if registration needs a tolerance beyond
`lattice_match_tolerance`. Estimate: 7–9 files touched, ~900–1300 LOC.

Acceptance:
- reuse the build side of `test_crystal`, `test_fluids`, `test_lift_slab`,
  `test_surfaces`, `test_reactive_interface`, `test_amorphous`,
  `test_protocols` (realize untouched);
- equivalence: `physics=False` builds must produce byte-identical extxyz
  (composer consumes rng in program-block order, which for canonical programs
  — crystal before liquid, as the printer emits — reproduces the legacy
  construction order; a test pins the first 1000 rng draws, the "rng tape");
  `physics=True` builds must produce identical *initial* frames plus the same
  realize call, hence identical trajectories (determinism, A10);
- seam tests: crystal|liquid composition conserves counts exactly, no overlap
  beyond `overlap_tolerance` at the seam; two-grain registration picks the
  smallest common supercell; arbiter raises on an impossible `conserve`;
- a new-case test: crystal | amorphous | vacuum builds (rejected today).

### Stage 3 — delete the old paths

Scope: remove `lift/legacy.py`, the cascade in `lift/__init__.py`, the
phase-set dispatch in `build/__init__.py`, and the program-producing entry
points of `lift/slab.py` / `build/slab.py` — keeping their numeric kernels,
which move (import-path change only) to where the pipeline/composer call them
(`_circ_offset`, tanh interface fit, defect clustering, `_slab_params`).
Explicit `mode=` values are reimplemented as pipeline constraints (2.6).

Work: 6–8 files touched, net −600..−900 LOC; dual goldens collapse to
pipeline-only goldens (regenerated once, reviewed as text diffs);
`docs/reference.md` and the lift/build entry docstrings updated.

Acceptance: full suite green with the deletions; `chaord roundtrip` over the
bench under pipeline only; A1–A14 run by a verification agent that did not
write the code; one release note announces the `mode=` semantics (constraint,
not separate lifter).

## 5. Risk register

| id | risk | impact | mitigation |
| --- | --- | --- | --- |
| R1 | segmentation error cascades into the region fit (mislabelled atoms -> phantom vacancies/antisites -> wrong program) | high | seam bands excluded from fits and defect search (the existing `max(bulk_margin, w + q6_cutoff)` rule, now applied per boundary, not just z); fit score gates before any defect statement is emitted, else degrade; sensitivity test: flip 1% of labels on fixtures, assert text unchanged or region degraded to residual |
| R2 | region-count explosion in polycrystalline / multi-interface frames (tens of grains x per-grain fits; text size vs A9) | medium | `pipeline_region_cap` + same-phase/same-params merge within `region_merge_tol`; grains over the cap go to residual (honest, conservative); per-grain fitting is M6 scope and stays out of v2 |
| R3 | statistical regions have no unique parameterization (density vs g(r) peak vs cn; every frame lifts differently) | medium | emission set and order fixed by dialect keys (one list per phase), rounding by `canonical_*` — one structure, one text within a frame; *across* frames, correctness is the statistical round trip against the noise floor (A5), never byte equality; the shortest-program controller (`check/shortest.py`) arbitrates redundancy |
| R4 | PBC inconsistency at composition (wrapped slab ranges, atoms imaged into the neighbouring region, seam distances computed without minimum image) | high | single `wrap` at the end of `join`; all seam checks on mic displacements (`realize/lj.mic`); wrapped-slab interval arithmetic reused from `_slab_params`; the wrapped golden (`lj_slab.chaord`, `26.1 .. 12.3`) is in the equivalence set |
| R5 | performance: pipeline is multi-pass (classify, segment, fit, stats) vs the cascade's single lucky pass | medium | everything shares the S1 caches (one `qbar`, one KD-tree, one bond graph — the legacy path recomputes q6 up to 4x on a cascade miss); budget test pipeline <= 1.3x legacy; A11 keeps the 100k-atom / 2 min ceiling; the cascade's *worst* case (all arms fail -> slab) is the pipeline's *best* comparison case |
| R6 | equivalence tests flake because the pipeline subtly reimplements a kernel (different binning, different tie-breaking) | high | discipline: stage 1 imports legacy numerics, only orchestration is new; byte-for-byte equivalence gate with an explicit, PR-reviewed exception list; dual goldens |
| R7 | composer rng order differs from legacy, breaking byte-equality of physics builds | medium | block-order construction + rng-tape test (stage 2); non-canonical program order documented as byte-unspecified |
| R8 | dialect additions need human approval (AGENTS) and stall stage 1 | low | exactly 4 new keys, bundled in the first PR with this doc as rationale; every other threshold reuses existing keys |

## 6. Self-check: can a new agent start stage 1 without touching existing behaviour?

Inputs the agent needs and where they are — all verified present:

- routing point: `lift/__init__.py:lift_frame` (the only existing file whose
  behaviour changes, and only by adding branches); S1 kernels:
  `lift/passes.py:qbar, pairs_within`; `build/molecules.py:bond_graph`;
- S2 kernels: `lift/segment.py:phase_labels_3d, interface_mesh, slab_interfaces`;
  `lift/surface.py:has_vacuum`; `lift/amorphous.py:is_amorphous`,
  `lift/fluid.py:is_single_phase`;
- S3 engines: `lift/crystal.py:lift_crystal, match_prototype, round_canonical`;
  `lift/defects.py:fit_crystal, ideal_sites, defect_diff`; `lift/slab.py:decompile`;
  S4: `lift/defects.py:group_defects`, `lift/slab.py:lift_defects`,
  `lift/defect_program.py` assembly;
- S5: `lift/fluid.py:_rdf_stats` (pos-parameterized), `cv/glass.py`,
  `build/defects.py:warren_cowley_alpha1`; S6:
  `lift/reactive.py:dissociation_from_census`;
- fixtures: the frames constructed in the test modules listed in 4.1 plus
  `tests/golden/lj_slab.chaord`; rules: PLAN (architecture, risk table),
  AGENTS (no thresholds in code, tests first, never weaken tests).

Open items the agent must not guess (bring to the PR, per AGENTS): the 4 new
dialect key values (calibrate on bench frames, propose for approval); the
exception list if byte-equality cannot be reached for a fixture (this doc's
position: fix the pipeline, not the test).

Verdict: yes — stage 1 is additive except for the `lift/__init__.py` routing,
four dialect keys, and six public-name promotions (renames, no logic change).
No existing test may be edited; the equivalence suite is written first and
must pass before `mode="auto"` flips.


## Approved changes (2026-09-29, Review 2)

The design is approved with these three binding changes:

1. **Region names must not depend on atom indices.** Section 2.7's
   "smallest atom index" ordering breaks re-ordering invariance; use
   composition + geometry-derived names instead.
2. **Build must realize with the program's physics and never skip silently.**
   An unknown backend or a missing history raises, not returns.
3. **Fluid construction uses grid placement plus minimisation**, not random
   packing at full density.
