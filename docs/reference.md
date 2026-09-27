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

## CLI

```
chaord fmt in.chaord [-o out] [--check]
chaord build in.chaord -o out.extxyz [--seed 1] [--no-physics]
chaord lift dump.extxyz [-o out.chaord] [--dialect core+lj] [--mode auto|crystal|defects|surface|amorphous|fluid|slab]
chaord check prog.chaord struct.extxyz
chaord diff a.chaord b.chaord            # provenance ignored
chaord roundtrip dump.extxyz [--samples 5]
```
