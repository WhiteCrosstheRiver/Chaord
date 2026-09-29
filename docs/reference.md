# Chaord reference manual (v0.1)

Chaord describes the macrostate of an atomic system; a coordinate file is one
microstate of it. This entry lists every statement key the core and dialects
define, with one passing example each. The canonical form is printed by
`chaord fmt`; examples live under `spec/examples/`.

## File shape

```
chaord 0.1
dialect core + metal            # core is always implied and comes first

system { ... }                  # exactly one
physics { ... }
species { ... }                 # molecular systems
crystal NAME : GEOMETRY { ... } # regions: crystal | amorphous | liquid | gas |
                                #           fluid | cluster | vacuum
interface A | B { ... }
residual none | { ... }
provenance { ... }              # lift metadata; ignored by `chaord diff`
```

One statement per line; `;` may join two. Comments start with `#`. Every
statement carries a kind: **build** (no keyword), **state**, **constrain**,
**assert**, **history** or **conserve**.

## system

| key | example | meaning |
| --- | --- | --- |
| `units` | `units lj` | reduced units (LJ) or physical (default) |
| `cell` | `cell 21.432 21.432 21.432 A` | box edge lengths; `auto` derives from density |
| `pbc` | `pbc xyz` | periodic directions |
| `seed` | `seed 7` | sampling seed: same seed, same coordinates |

## physics

| key | example | meaning |
| --- | --- | --- |
| `backend` | `backend lj` | the compile target: `lj`, `eam`, `mlp`, `classical` |
| `epsilon` / `sigma` / `cutoff` | `epsilon 1` | LJ parameters |
| `potential` | `potential "NiAl.eam.alloy"` | backend-specific files |
| `model` | `model "mace-mp-0"` | machine-learned potential tag |
| `model` | `model spce` | rigid water model of the `classical` realization (molecular dialect: `tip4p` default, `spce`; see below) |

## species (molecular dialect)

| key | example |
| --- | --- |
| `molecule` / `ion` / `atom` | `molecule EC = smiles "C1COC(=O)O1"` |
| | `ion Li+ = smiles "[Li+]"` |

## region statements

| key | example | meaning |
| --- | --- | --- |
| `lattice` | `lattice fcc` | unary prototype: sc, bcc, fcc, hcp, diamond |
| `prototype` | `prototype L1_2` | multi-species: rocksalt, cscl, zincblende, wurtzite, fluorite, perovskite, L1_2, rutile |
| `composition` | `composition Ni3Al` | slot species by reduced counts |
| `a` / `c` | `a 3.572 A` | lattice parameters |
| `orient` | `orient x [100] y [010] z [001]` | integer directions/families |
| `occupancy` | `occupancy Cr 1/3 Co 1/3 Ni 1/3` | solid solution fractions |
| `defect` | `defect V_Ni count 1` | Kröger-Vink: `V_X` vacancy, `A_B` antisite, `A_i` interstitial, `frenkel_pair` |
| | `defect V_Ni count 2 depth 6.1` | depth targets the M0 slab builder |
| `surface` | `surface (111) top` | Miller cut + side (surface dialect) |
| `termination` | `termination bridging_O` | top-layer element (compound names resolve the cut) |
| `reconstruction` | `reconstruction p(2x1)` | Wood notation: p(nx m), c(nx m), (rkxrk)R30 |
| `adsorb` | `adsorb O count 4 site top coverage 0.25 ML` | top / bridge / hollow |
| `molecules` | `molecules H2O 620` | molecular counts |
| `dislocation` | `dislocation edge` | Volterra construction |
| `grain_boundary` | `grain_boundary sigma 5` | [001] CSL bicrystal |
| `state density` | `state density 0.854` (LJ) / `1.0 g/cm3` | target density |
| `state T` / `state P` | `state T 300 K` | equilibrium state metadata |
| `constrain strain` | `constrain strain zz +0.9 % +- 0.2` | held deformation |
| `constrain sro` | `constrain sro alpha1 Cr-Cr +0.10 +- 0.02` | Warren-Cowley first shell |
| `history` | `history melt 1.2 for 500 -> quench to 0.01 at 0.002 -> anneal 0.01 for 300` | protocol = shortest description |
| | `history ... -> deposit X 20 for 400` | growth: insert atoms during the run |
| `assert cn` | `assert cn 4.0 +- 0.1 cutoff 2.85 A` | measured, never enforced |
| `assert gr_peak` | `assert gr_peak 1.06 height 3.07` | first rdf peak |
| `assert sites_matched` | `assert sites_matched 99.2 %` | site-lattice coverage |
| `assert angle_mean` | `assert angle_mean 109.0 +- 1.5 deg` | bond-angle mean |
| `assert ring_mode` | `assert ring_mode 6` | dominant King ring size |
| `assert coverage` | `assert coverage OH 0.25 ML +- 0.05` | adsorbate coverage |
| `assert solid_clusters` | `assert solid_clusters 0` | fluid purity |
| `chirality` | `chirality (14,3)` | nanotube handedness (carbon dialect) |
| `forcefield` | `forcefield "trappe-ua"` | classical force-field tag |
| `lift_version` | `lift_version "0.1.0"` | provenance: lifting tool version |
| `nanotube` | `nanotube count 1` | carbon dialect construction |
| `note` | `note "any text"` | free-form provenance note |
| `source` | `source "dump.lammpstrj"` | provenance: origin file |
| `stacking` | `stacking AB` | layer stacking order |
| `dialects` | `dialects "core 0.1.0 + lj 0.1.0"` | provenance: dialect versions used |
| `state` (region) | `state T 300 K` inside a region | region-level equilibrium metadata |
| `conserve atoms` | `conserve atoms Ni 646 Al 217` | exact per-species counts |
| `conserve charge` | `conserve charge 0` | total charge (checked against the frame's ionic census) |

## interface / residual

| key | example |
| --- | --- |
| `at` | `at z 12.3` |
| `width` | `width 1.1` |
| `dissociate` | `dissociate H2O -> OH @ surface + H @ surface count 9` |
| `atom` | `atom X 1.02 3.40 5.60` (residual only) |

## geometry

`all`, `rest`, `slab z 0 .. 18 A`, `box .. .. ..`, `sphere center x y z radius r`,
`cylinder axis z center x y radius r`, joined by `and`, `or`, `minus`.

## dialects

`core` (always first) fixes shared algorithm parameters; `metal`, `ionic`,
`molecular`, `surface`, `glass`, `carbon`, `lj` add domain rules. Thresholds
live ONLY in `src/chaord/dialects/*.yaml` — never in pass code (CI-enforced).

### molecular: `classical_pair_potentials`

The `physics backend classical` realization of molecular fluids without
water is driven by a published-parameter table in the molecular dialect
(`classical_pair_potentials`, keyed by element, one entry per LJ site:
`sigma_A`, `epsilon_K` = epsilon/kB in K, `sites`):

| element | sites | sigma (A) | epsilon/kB (K) | source |
| --- | --- | --- | --- | --- |
| `Ar` | 1 | 3.405 | 119.8 | Hansen & Verlet, Phys. Rev. 184, 151 (1969) |
| `N` | 2 | 3.31 | 37.3 | Murthy, Singer, Klein & McDonald, Mol. Phys. 41, 1387 (1980), two-centre LJ nitrogen (point quadrupole dropped: the classical realization is LJ-only) |

Monatomic species carry one LJ site per atom; a `sites: 2` element is the
diatomic molecule of the shipped template (`N2` = contiguous N,N pairs at
the template bond length, held rigid through the relaxation, intramolecular
pair excluded). Cross terms mix Lorentz-Berthelot; the smooth cutoff and MD
protocol are the backend's shared ones. Water (`H2O` → TIP4P) and its
Joung-Cheatham ions are realized by their own kernels and do not mix with
the LJ species — a program naming both is a static error. Extending the
table (O2, CO2, ...) is a dialect change: add the element with its
published parameters and citation, nothing else.

### molecular: `water_models`

Rigid water is not one geometry: the reference data of the fluid benchmark
alone spans two published models (TIP4P and SPC/E differ by 0.043 A in O-H
and 5 deg in H-O-H — under rigid MD those are conserved quantities, so the
difference shows up in every rebuilt frame's intramolecular g(r) peaks).
The molecular dialect's `water_models` table is the single authority both
directions consult:

| model | O-H (A) | H-O-H (deg) | realization | source |
| --- | --- | --- | --- | --- |
| `tip4p` (default) | 0.9572 | 104.52 | O-site LJ + virtual M site, charges via ASE's tip4p `qH`; ASE's smooth truncation on O-O | Jorgensen, Chandrasekhar, Madura, Impey & Klein, J. Chem. Phys. 79, 926 (1983); constants are the ASE module's own |
| `spce` | 1.0000 | 109.47 | three sites: LJ (sigma 3.166 A, epsilon/kB 78.22 K) and charge (q_O -0.8476 e) on O, charge (+0.4238 e) on H; one site-site pair list with Lorentz-Berthelot LJ energy-shifted at rc = 9 A and damped-shifted-force (Wolf) electrostatics, alpha = 0.2 1/A | Berendsen, Grigera & Straatsma, J. Phys. Chem. 91, 6269 (1987); truncation: Fennell & Gezelter, J. Chem. Phys. 124, 234104 (2006), the recorded protocol of the nacl_aq reference |

* **lift** measures the frame's median O-H distance (the conserved
  quantity) and classifies it against each model's window
  (`r_oh_A +- classify_half_width_A`, disjoint by construction). A match
  emits `model <name>` in the physics block and names the water species
  after the model's template: `H2O` for the default, `H2O/spce` for SPC/E
  (only the default keeps the bare name, so pre-existing programs are
  unchanged). Non-rigid or unknown geometry emits no model statement and
  records the measured median in a provenance `note`; the default model
  then applies.
* **build** packs the named template's geometry; the rigid-water constraint
  locks whatever geometry was packed (`H2O` → TIP4P, `H2O/spce` → SPC/E).
  A water template name the registry does not ship is a static error.
* **realize** (the classical ASE backend) resolves the model the same way
  the lift does — from the packed water's own median O-H — and runs the
  model's published potential. TIP4P keeps ASE's own kernel and smooth
  truncation; the three-site models run the reference family's recorded
  truncation (LJ energy-shifted at rc, Wolf/DSF electrostatics, both read
  from the dialect table). The Joung-Cheatham ion parameters are
  SPC/E-calibrated and pair with either water model under Lorentz-Berthelot
  mixing.

The `model` statement is metadata to the fluid builder the way `potential`
is to the `eam` backend: the construct is the species name. An explicit
`model=` passed to the backend must name a parameterized model or the
calculator raises.

### molecular: rebuild equilibration protocol (`ase_md`)

The `physics backend classical` rebuild of a molecular fluid (packed start,
capped FIRE, then seeded Langevin) reads its protocol from the molecular
dialect's `ase_md` table, never from builder code. The relax stage's
friction (`relax_gamma_per_fs: 0.005`) is the shipped MD references' own
recorded protocol (Langevin friction 0.05 per ASE time unit, nacl_aq
provenance), so a rebuilt fluid decorrelates at the references' rate;
`relax_steps: 950` (0.95 ps at `relax_dt_fs: 1.0`) is the acceptance-budget
optimum — the coordination distance to the reference bottoms there and
rises again for longer runs (over-decorrelation: the limit is the
protocol's own fluctuation level, not missing equilibration; measured at
fixed seed: relax 600 → cn_tv ×1.77, 950 → ×1.45, 1400 → ×1.54). Cost:
~125 s of the 150 s per-case acceptance budget.

### lj: rebuild equilibration protocol (`md`)

The `physics backend lj` rebuild of a fluid/slab region packs RSA positions,
then runs a two-stage Langevin equilibration whose every parameter lives in the
lj dialect's `md` table (never in builder code): a short stiff stage
(`relax_steps_fast` steps at `relax_dt_fast`, friction `relax_gamma_fast`,
force capped at `fcap`) that burns off packing contacts, then the equilibration
stage (`relax_steps` steps at `relax_dt`, friction `relax_gamma`, cutoff
`relax_rc`) at the program's `state T`.

`relax_steps` is sized for the largest MD reference fluid (2048 atoms), not
for the smallest. Erasing an RSA-packed start is diffusive — mixing time
~ L^2 while the cubic edge grows as N^(1/3) — so the step need grows as
N^(2/3): the dialect records the basis (`relax_steps_base` steps equilibrate
`relax_steps_ref_n` atoms; `relax_steps_exp` = 2/3) and ships a flat
`relax_steps` + `relax_steps_fast` total that covers it. The count is
deliberately N-independent: the acceptance verifier reads a rebuild's
`md_steps` as the static dialect sum `relax_steps_fast + relax_steps`, so a
step count that scaled with N inside the builder would make that metadata
misdescribe the rebuild. Smaller systems simply over-equilibrate, at a per-step
cost that scales down with N. An explicit `md_steps` argument to the build API
overrides the slow stage's length and is what the metadata then reports.

### Interface slabs: thin-film liquid statistics (`thin_slab_margin_fraction`)

The slab lifter's liquid asserts are measured in a bulk-statistics zone
`|dz - z_liquid| < H_liquid - bulk_margin - g(r) range`, kept away from both
interfaces. A small frame's liquid film can be thinner than
`bulk_margin + gr_rmax` (e.g. the 512-atom bench slab: film half-height
3.17 sigma vs margin + range = 5 sigma) and the zone would be empty. The
degradation rule (lj and metal dialects, key `thin_slab_margin_fraction`,
value 0.8) is honest instead of fatal: both insets shrink proportionally
until they cover at most that fraction of the film half-height; the g(r)
range never drops below `cn_cutoff` (the first shell must stay in range);
if the floor still does not fit the margin drops to zero; a film thinner
than one `cn_cutoff` raises (no honest liquid statements exist). The shrink
is stated, never hidden — the liquid block carries a comment:

```
liquid B : slab z 6.2 .. 12.5 {  # thin-film liquid: bulk margin shrunk 2.50 -> 1.26, g(r) range 2.50 -> 1.50
  state density 0.981
  ...
}
```

### metal 0.2.4: interface slab thresholds

Until 0.2.4 the metal dialect defined none of the slab lifter's thresholds
(a metal-frames interface lift died on the first read). The values are the
lj dialect's semantics converted from sigma to Angstrom at the Cu scale:
the LJ fcc solid has a = 1.609 sigma, so d_nn = a/sqrt(2) = 1.138 sigma
(1 sigma = 0.879 d_nn); Cu fcc a = 3.615 A gives d_nn = 2.556 A, i.e.
1 sigma ~= 2.246 A.

| key | lj (sigma) | in d_nn | metal (A) | meaning |
| --- | --- | --- | --- | --- |
| `bulk_margin` | 2.5 | 2.20 | 5.6 | distance kept from interface centres for bulk statistics |
| `gr_rmax` | 2.5 | 2.20 | 5.6 | g(r) range of the liquid asserts |
| `gr_bins` | 100 | — | 100 | g(r) bins |
| `angle_hist_bin` | 5 | — | 5 | bond-angle histogram bin, degrees |
| `defect_cluster_rc` | 1.3 | 1.14 | 2.9 | linking distance for empty-site/off-lattice complexes |
| `rsa_dmin` | 0.90 | 0.79 | 2.0 | RSA hard core for atomic species (= `fluid_rsa_dmin`) |
| `liquid_margin_shrink` | 0.3 | 0.26 | 0.7 | shrink of the liquid half-slab used for packing |
| `printed_strain_tolerance` | 0.2 | — | 0.2 | tolerance on the printed strain constrain, % |
| `thin_slab_margin_fraction` | 0.8 | — | 0.8 | thin-film degradation rule (section above) |
| `lattice_spacing_scan` | 0.6..1.1 | 0.75..1.37 | 1.2..2.4, step 0.001 | z layer-spacing candidates for the lattice fit (core's sigma range covers LJ fcc(100) a/2 = 0.80 sigma; the metal range covers Cu fcc(100) 1.81, fcc(111) 2.09 and bcc(110) Fe 2.03 A) |
| `md_reference_T` | 0.65 | — | 300 (K) | metadata temperature stated when the caller gives none |

Two builder keys complete the group. `rsa_max_packing_fraction` (0.36, both
dialects) bounds the hard-sphere volume fraction the RSA packer may target:
above it the placement hard core shrinks to the feasible value and the MD
prior (its force cap) relaxes the closer contacts — the placement is a
prior, not the physics. `rsa_max_tries` (2,000,000) is the honest budget:
running out raises instead of spinning. On the metal side,
`slab_packing_margin` (0.05 A) and `slab_packing_max_tries` (16000 per
molecule) drive the molecular interface packer below.

### Multi-species interface programs

A frame with several species (e.g. Cu under water) lifts with the crystal
fitted on the majority solid species' sublattice and the liquid described by
a molecular census; conservation is exact per species and the species of the
unary lattice region is named in `conserve atoms` (the language rule):

```
crystal A : slab z 42.5 .. 21.6 {  # wraps through z = 0
  lattice fcc
  a 3.615 A
  orient <100> z <001>
  constrain strain zz +0.0 % +- 0.2
  assert sites_matched 100.0 %
}

liquid B : slab z 21.6 .. 42.5 {
  molecules H2O 136
  state density 0.930 g/cm3
  assert cn 3.3 +- 1.2 cutoff 3.10 A
  assert gr_peak 3.86 A height 2.16
}
```

The build is exact where the program is: the crystal block constructs its
sites, the molecules RSA-pack into the liquid slab by whole-molecule
insertion with the crystal atoms as excluded obstacles (reject below
`bond_tolerance` x covalent-radii sum + `slab_packing_margin` — the bench
generator's protocol, so the census of the packed frame is exact). Physics
relaxes each region under its own backend (metal region: the dialect's
analytic Finnis-Sinclair EAM parameters, or the shipped tabulated
Cu_u3.eam; molecular region: TIP4P water via `classical`). No published
potential exists in the backend set for the metal-water cross terms, so the
interface is deliberately NOT relaxed across: per-region equilibration in
the shared cell, an honest protocol stated here and in the builder.

## CLI

```
# invoke as `.venv/bin/python -m chaord ...` (Linux/macOS)
# or     `.venv\Scripts\python -m chaord ...` (Windows)
chaord fmt in.chaord [-o out] [--check]
chaord build in.chaord -o out.extxyz [--seed 1] [--no-physics]
chaord lift dump.extxyz [-o out.chaord] [--dialect core+lj] [--mode auto|crystal|defects|surface|amorphous|fluid|slab]
chaord check prog.chaord struct.extxyz
chaord diff a.chaord b.chaord            # provenance ignored
chaord roundtrip dump.extxyz [--samples 5]
```
