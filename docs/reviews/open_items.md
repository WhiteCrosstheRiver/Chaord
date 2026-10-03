# Open items from external Reviews 4–7 (2026-09-30 to 2026-10-03)

**Read this before starting any work.** These items take priority over every
older plan, including Review 3's stream list. Each item names a reproducer that
fails on commit `14cf5d4` and the test that must pass when it is done. Write the
failing test first (AGENTS.md), then fix. Do not change how a criterion is
measured to make an item pass.

Run the snippets from the repository root, with the project environment.
Common header for all snippets:

```python
import sys, numpy as np
sys.path[:0] = ["src", "tools"]
import acceptance as acc
from chaord.dialects import load_dialect
from chaord.io.frames import Frame
from chaord.lift import lift_frame
from chaord.build import build_program

def shifted(f, t):
    """The same frame, rigidly translated by t (Å) and wrapped into the box."""
    return Frame(pos=np.mod(f.pos + np.asarray(t, float), f.cell_diag),
                 cell=f.cell, symbols=f.symbols, pbc=f.pbc)
```

## Blocked on the owner (agents: do not decide these)

| ID | Decision | Status |
| --- | --- | --- |
| D1 | `lj.yaml` `overlap_tolerance` 0.70 → 0.80 σ | pending since 2026-09-30 |
| D2 | `overlap_sanity_fraction` 0.7 or 0.8 | pending since 2026-09-30 |
| D3 | `lj_solid_liquid` solid-density tolerance 2.5% → 4.0% | **already in effect** without approval |
| D4 | Ratify the new dialect keys (hold `thermal_quench_*` until O3 passes) | pending since 2026-09-30 |
| D5 | Pull requests per stream | pending since 2026-09-30 |
| D6 | A3 assignment check instead of PLAN's StructureMatcher (see O2) | **already in effect** without approval |
| D7 | Glass protocol base values 1500/3000/2850 and `glass_protocol_ref_n` | **already in effect** without approval; tuned to a reference that O9 replaces |
| D8 | Rebuild time budget linear in N | pending |

## Open items, in priority order

### O1 · A2 is not translation-invariant on thermal frames (P0, Review 6)

A2 checks 2 transforms per case. With 40 random shifts per case, `fcc_cu`,
`bcc_fe`, `l12_ni3al`, `fcc_crconi` and `cuau_random` lift to a different
program for 1 to 6 of the 40 shifts. Rotation-only and reordering-only draws
never change the text. All three reproducers below also produce a program that
Chaord's own compiler rejects.

```python
for cid, t in [("crystals/fcc_cu", (-2.17, -2.18, -4.15)),    # a 3.240 A, 2 frenkel_pair
               ("crystals/bcc_fe", (-0.04, -2.54, 3.38)),     # a ~3.10 A, 3 frenkel_pair
               ("crystals/l12_ni3al", (3.86, 1.97, -1.74))]:  # lattice fcc + occupancy
    case = next(c for c in acc._exact_roundtrip_cases(None) if c["id"] == cid)
    dl = load_dialect(case["dialect"])
    f = acc._a2a3_frame(case)                      # the stored thermal frame_0
    g = shifted(f, t)
    same = acc.format_program_text(lift_frame(g, dl)) == \
           acc.format_program_text(lift_frame(f, dl))
    try:
        build_program(lift_frame(g, dl), dl)
        builds = "yes"
    except Exception as e:
        builds = f"{type(e).__name__}: {e}"
    print(cid, "| same text:", same, "| builds:", builds)
# today, all three: "same text: False" and a ChaordError ("not an integer
# multiple of the fcc/bcc lattice vectors" for Cu and Fe, "no neighbours
# within cutoff" for Ni3Al). Done: True and "yes" for all three.
```

Where to look: `lift/defects.py::fit_crystal`. The origin-anchored fit is kept
unless re-anchoring improves the mean distance by 30%
(`lattice_anchor_improvement`), only 4 atoms seed the re-anchoring
(`_anchor_seed`), and the lattice-constant scan accepts constants that do not
tile the periodic box.

Done when: 50 random rigid transforms per case (A2's own `rigid_transform`)
give byte-identical text on all 9 thermal frames, and every lifted program
builds.

### O2 · A3's geometry check depends on the origin (Review 6)

`_geo_fit_thermal` compares the stored frame with its rebuild atom by atom,
with no alignment. The stored frames pass only because `bench/generate.py`
made them with Chaord's own builder, at the rebuild's origin.

```python
bad = []
for case in acc._exact_roundtrip_cases(None):
    if acc._is_random_solution(case):
        continue
    dl = load_dialect(case["dialect"])
    g = shifted(acc._a2a3_frame(case), (0.7, 0.0, 0.0))
    ok, note = acc._geo_fit_thermal(g, build_program(lift_frame(g, dl), dl,
                                                     rng=np.random.default_rng(5)))
    if not ok:
        bad.append((case["id"], note))
print(len(bad), "ordered cases fail:", bad)   # 7 of 7 today; done: 0
```

Also: the stated reason for leaving PLAN's matcher ("no fit even for the frame
against itself") has no committed test, which AGENTS.md requires. A likely real
cause is that the jittered frame cannot be reduced to a primitive cell while
the perfect rebuild can. If a test confirms it, try
`StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=False)`,
which keeps PLAN's tool and searches translations itself. The A3 evidence string
says "p90 matched displacement vs 0.15 d_NN" but the code gates on the fraction
beyond 0.25 d_NN; make them agree.

Done when: shifted copies pass; the four A3 mutations still fail; the evidence
string states the gate applied; D6 is decided.

### O3 · A4 fails on new seeds (Reviews 4, 5, 6)

```python
orig = acc._stable_seed
fails = 0
for k in range(1, 16):
    acc._stable_seed = lambda s, k=k: orig(f"{s}|heldout{k}")
    fails += not acc.check_a4(temps=("0.8Tm",))["passed"]
acc._stable_seed = orig
print(fails, "of 15 seed sets fail")     # 9 of 15 today (k = 2, 3, 5, 6, 7, 9, 10, 12, 15); done: 0
```

Cause: `_plant_defects` places each interstitial 0.6 d_NN from a random parent
in a random direction and does not keep planted defects apart. Failing draws
have an interstitial 0.40 d_NN from a host site, or two interstitials 0.34 d_NN
apart. Real fcc interstitial sites are 0.61 and 0.71 d_NN from their
neighbours.

Done when: interstitials are planted at octahedral or tetrahedral sites (or as
relaxed dumbbells), planted defects are ≥ 2 d_NN apart, every planted frame
passes a 0.8 d_NN minimum-pair check, and all 26 cells pass on 10 new seed
sets. If physical frames still fail, the lifter is next.

### O4 · A8's site classifier ignores periodic boundaries (Review 6)

```python
import re
surf = load_dialect(("core", "metal", "surface"))
f0, truth = acc._plated_adsorbate_frame()        # truth: top 5, bridge 3
L = f0.cell_diag
rng = np.random.default_rng(0)
wrong = 0
for _ in range(20):
    g = shifted(f0, (rng.uniform(0, L[0]), rng.uniform(0, L[1]), 0.0))
    claims = {}
    for m in re.finditer(r"adsorb (\S+) count (\d+) site (\S+)",
                         acc.format_program_text(lift_frame(g, surf))):
        claims[m.group(3)] = claims.get(m.group(3), 0) + int(m.group(2))
    wrong += claims != truth
print(wrong, "of 20 translations change the site census")
# today 3 of 20 (they claim top 6 / bridge 2); done: 0
```

A physical frame fails outright: build the same fcc(111) slab as
`_plated_adsorbate_frame`, put O 1.2 Å above the top layer at a p(2×2) subset
of fcc-hollow sites (the lateral positions of the third layer from the top),
add 0.08 Å Gaussian motion to every atom and keep the slab away from z = 0:
0 of 10 draws call all 8 O "hollow". The same holds for hcp hollows (second
layer from the top). Cause: `_classify_sites` triangulates the top layer
without periodic images; replicating the layer over its 8 neighbouring images
gives 10 of 10 and 20 of 20. A slab that crosses the periodic z boundary lifts
part of its own bottom layer as `adsorb Pt ... site top`. The committed A8
frame itself is unphysical: 6 of its 8 O atoms are 1.35–1.39 Å from another O.

Done when: physical frames with top, bridge, fcc-hollow and hcp-hollow O,
thermal motion and realistic heights are ≥ 90% correct over 20 draws and 20
in-plane translations; a slab across the z boundary lifts unchanged. Adding
`fcc`/`hcp` hollow names to the language is a spec change: ask the owner.

### O5 · Every lifted program must compile, and every test input must be physical (Review 6)

A13 only lifts. Add: every lifted bench program builds (physics off is enough).
Today `gases/co2_dense` fails ("cannot place 60 CO2 at this density"): its
frame is 2.54 g/cm³ (its description says ~1.0; dry ice is about 1.56) with
molecule centres 1.96 Å apart. Apply the reference-data sanity checks (minimum
pair distance, density) to every bench frame and every planted frame (A4, A8).

Done when: every lifted bench program builds; every bench and planted frame
passes the sanity check; `co2_dense` is regenerated at a physical density in
its own reviewed commit.

### O6 · Demo Act 1 rebuilds at an assumed temperature (Review 6)

`tools/tutor_demo.py` lifts `lj_liquid` without the provenance temperature, so
the program it prints says `state T 0.65` (the reference ran at 0.72), and it
passes just before Act 3 says a wrong temperature must fail.

Done when: Act 1 lifts with `T=acc._provenance_tstar(provenance)` as `check_a5`
does; after O1, Act 0 also lifts a shifted Cu copy to the same text.

### O7 · Reports and public claims (Reviews 5–7)

- `acceptance.json` records the commit SHA, runner OS and run URL.
- README: say "14/14 on the committed inputs; open counterexamples O1–O4"
  until they are fixed. Today it says "independently verified ... with power".
- `pass_evidence_dossier.md`: A2 "limitations: none" and A4 "not tuned to the
  acceptance seeds" are contradicted by O1 and O3; correct them.
- Gate report: replace "Numbers below are from the local run" with clean-run
  numbers; its scorecard covers Reviews 4–7, not only Review 3.

### O8 · A5 floors rest on 2–5 pairs (Reviews 4–5)

Store ≥ 10 frames per reference case so each averaged floor has ≥ 10 pairs;
then decide whether `max(mean, P90)` is still needed. State each case's
temperature resolution (today: 20% is detected on the 2,048-atom liquid, 10% is
not).

### O9 · The glass reference is a solid under tension that tears open (Review 7)

At ρ* = 0.85 and T* = 0.01 in a fixed box, the 2,048-atom "glass" is not a
homogeneous glass. Measured on the stored frames (first → last sampled frame of
each quench) and on rebuilds of its lifted program:

| Frames | Pressure (LJ units) | Largest empty sphere (radius) | Volume > 1 σ from any atom | Crystal-like atoms | Energy per atom |
| --- | --- | --- | --- | --- | --- |
| reference, quench 1 (frame 0 → 4) | −1.57 → −1.29 | 2.81 → 2.99 σ | 6.7% → 7.5% | 5.4% → 8.6% | −6.494 → −6.529 |
| reference, quench 2 (frame 5 → 9) | −1.56 → −1.29 | 2.75 → 2.93 σ | 6.5% → 7.5% | 3.3% → 4.2% | −6.474 → −6.521 |
| reference, quench 3 (frame 10 → 14) | −1.44 → −1.22 | 3.16 → 3.25 σ | 7.1% → 8.2% | 1.2% → 1.1% | −6.497 → −6.552 |
| rebuild, anneal cut to 100 steps (seeds 7, 13, 29) | −3.71, −3.80, −3.77 | 1.48, 1.15, 1.43 σ | 0.2%, 0.0%, 0.1% | 1.2%, 1.9%, 0.2% | −6.071, −6.067, −6.069 |
| rebuild, default history (seeds 7, 13) | −1.22, −1.60 | 3.70, 3.25 σ | 8.3%, 6.3% | 7.8%, 10.7% | −6.565, −6.491 |
| liquid, same density, T* = 0.72 | +0.85 | 0.99 σ | 0.0% | 0.0% | — |

Largest empty sphere: the largest distance from any point in the box (a 48³
grid) to the nearest atom centre. Crystal-like: averaged q6 > 0.32 with the lj
cutoff 1.45 σ. Energy: LJ cut and shifted at 2.5 σ. Reproducer, and the core
of the sanity check to add:

```python
from scipy.spatial import cKDTree
from chaord.cv.local import solid_like_fraction

def amorphous_sanity(f, T, rc=2.5):
    """LJ virial pressure, largest empty-sphere radius (48^3 grid), volume
    fraction more than 1 sigma from any atom, and crystal-like fraction."""
    L = f.cell_diag; p = np.mod(f.pos, L); n = len(p); V = np.prod(L)
    tree = cKDTree(p, boxsize=L)
    pr = tree.query_pairs(rc, output_type="ndarray")
    d = p[pr[:, 1]] - p[pr[:, 0]]; d -= L * np.round(d / L)
    inv6 = 1 / np.einsum("ij,ij->i", d, d) ** 3
    P = n * T / V + np.sum(24 * (2 * inv6 ** 2 - inv6)) / (3 * V)
    g = np.stack(np.meshgrid(*[np.linspace(0, L[i], 48, endpoint=False)
                               for i in range(3)], indexing="ij"), -1).reshape(-1, 3)
    dist, _ = tree.query(g)
    return dict(pressure=round(float(P), 2), empty_radius=round(float(dist.max()), 2),
                empty_fraction=round(float((dist > 1.0).mean()), 3),
                crystal_like=round(solid_like_fraction(f, load_dialect(("core", "lj"))), 3))

z = np.load("bench/reference/lj_glass/frame_0.npz", allow_pickle=True)
L = np.asarray(z["L"], float)
f = Frame(pos=np.asarray(z["r"], float), cell=np.diag(L) if L.ndim == 1 else L,
          symbols=[str(s) for s in z["symbols"]], pbc=(True, True, True))
print(amorphous_sanity(f, T=0.01))
# today: pressure -1.57, empty_radius 2.81, empty_fraction 0.067, crystal_like 0.054
```

Right after the quench the box is whole but stretched (pressure about −3.8).
The 7,296-step anneal at T* = 0.01 is where it tears open: the tension partly
relaxes, 6–8% of the volume empties, up to 11% of the atoms turn crystal-like
and the energy per atom drops by about 0.46. The quench leaves no measurable
trace:

- rebuilt with the lifted default history: passes (cn ×0.59, gr ×0.68);
- rebuilt with a 10× faster quench: also passes (cn ×0.59, gr ×0.41), and its
  energy per atom matches the default's (−6.537 against −6.533 over seeds 7,
  13 and 29; the seed-to-seed spread is 0.03);
- rebuilt with the anneal cut to 100 steps: fails by ×7.5 and ×15.5.

So the glass round trip measures how far the box has torn, and passes because
the default anneal (base 2850) was calibrated to the reference's own anneal
depth. Separately, `glass.yaml` sets `q6_cutoff: 3.0` (meant in Å for oxide
networks), which in LJ σ units covers about three neighbour shells: a perfect
LJ fcc crystal (2,048 atoms) lifts under `core + glass` as `amorphous glass`
with a melt–quench history. No sanity check for amorphous references tests
pressure, voids or crystallinity.

Done when: the glass reference is replaced by the standard Kob–Andersen 80:20
binary LJ glass (needs two species in the amorphous builder), or, as the
quicker fix, regenerated at constant zero pressure (monatomic LJ crystallises
easily, so the crystal-like check decides whether that works); amorphous
references pass new sanity checks (pressure stated and near zero, no point
more than about 1.2 σ from an atom, crystal-like fraction below 1%, energy per
atom flat across the sampled frames); a crystal never lifts as amorphous under
any dialect; A5's glass floor and default history are re-derived from the new
reference, and the report states whether A5 can tell a 10× quench-rate change
apart.

### O10 · A floor never includes the system under test (Review 7)

`tests/test_amorphous.py::test_amorphous_round_trip_statistics` now takes
`max(later-frame floor, distance between two rebuilds)`. A noisy rebuild then
widens its own tolerance. Use the reference's cross-quench floor, as A5 does.

### O11 · Metal solid–liquid interfaces cannot be lifted (Review 7)

`reference/cu_solid_liquid` frame 0 fails in every mode: `auto` routes it to
the molecular-fluid check ("extended bonded component (832 Cu atoms)"), and the
slab route fails because `core + metal` defines no threshold `printed_cutoff`.
Add the key (owner approval) and a test; record whether the lift then works.

### O12 · Leftovers

- AGENTS.md: allow strict xfail only for registered red-team findings awaiting
  the owner's decision.
- Flag the water model as assumed unless the input names it (TIP3P, TIP4P and
  TIP4P/2005 all use r(O–H) = 0.9572 Å).

## Rules to add to AGENTS.md (owner approves)

1. Read `docs/reviews/open_items.md` before starting; it overrides older plans.
2. A criterion with a random element is judged over at least 20 draws; one draw
   is not evidence.
3. Every test input, planted or synthetic, passes the reference-data sanity
   checks.
4. A floor or tolerance never includes a quantity measured on the system under
   test.
5. When a clean run fails, report the failure with its power numbers before
   changing how a criterion is measured.
