# Open items and decisions, v2 (external Review 8, 2026-10-04)

**Read this before any work.** It replaces v1 (Reviews 4–7). The owner has
delegated decision authority to the external reviewer for this round, so the
decisions in section 1 are made: implement them, do not reopen them. Then work
the orders in section 2 in the stream order of section 3. Every work order
lists its steps and a done-when test; W1, W4, W5, W6, W8 and W9 also carry a
reproducer that ran on commit `f4fde65`, with its result. Write the done-when
test first, see it fail, then fix (AGENTS.md).

Run the snippets from the repository root, in the project environment.
Common header for all snippets:

```python
import sys, numpy as np
sys.path[:0] = ["src", "tools"]
import acceptance as acc
from chaord.dialects import load_dialect
from chaord.io.frames import Frame, read_frame
from chaord.lift import lift_frame
from chaord.build import build_program

def shifted(f, t):
    """The same frame, rigidly translated by t (A) and wrapped into the box."""
    return Frame(pos=np.mod(f.pos + np.asarray(t, float), f.cell_diag),
                 cell=f.cell, symbols=f.symbols, pbc=f.pbc)

def load_ref(path):
    """A bench/reference frame (npz with r, L, symbols)."""
    z = np.load(path, allow_pickle=True)
    L = np.asarray(z["L"], float)
    return Frame(pos=np.asarray(z["r"], float), cell=np.diag(L) if L.ndim == 1 else L,
                 symbols=[str(s) for s in z["symbols"]], pbc=(True, True, True))
```

## 0. Status of the v1 items (re-checked by the reviewer on `f4fde65`)

| v1 item | Status | Reviewer's check |
| --- | --- | --- |
| O1 A2 translation | **done** | 900 fresh draws of A2's own `rigid_transform` (9 cases × frames 0–4 × 20): 0 text changes, 0 build failures; 1e-9 Å noise, 2×1×1 replicas: unchanged text |
| O2 A3 origin | **done** | 70 random in-box shifts (7 ordered cases × 10): 0 fail; a 10% species swap and 10% of atoms moved 0.4 d_NN are both caught |
| O3 A4 seeds | **done** | 30 fresh seed sets at 0.8 Tm and 30 at room temperature: 0 fail |
| O4 A8 sites | **done on fcc(111)** | Pt(111) with top, bridge, fcc and hcp O in one frame, 20 draws at random translations: 157/157 O correct. (100) surfaces: see W9 |
| O5 compile + physical inputs | **partly** | compile scan and `co2_dense` density fixed; open: W6, W8 |
| O6 demo temperature | **done** | Act 1 lifts at the provenance T. Act 0 shifted-Cu copy not added (W12) |
| O7 reports and claims | **partly** | README and dossier fixed; gate report body and `acceptance.json` stale: W2, W3 |
| O8 floors | **done** | every averaged floor rests on 10–12 pairs |
| O9 glass | **partly** | zero-pressure route honestly reported failed; the crystal fix broke Å crystals (W1); Kob–Andersen decided (D9, W7) |
| O10 floor excludes the tested system | **done** | the slow test's floor comes from two independent builds of the original program |
| O11 metal interfaces | **key added, output wrong** | W4 |
| O12 leftovers | **done** | |

## 1. Decisions (binding, 2026-10-04)

| ID | Decision | Implement |
| --- | --- | --- |
| D1 | `lj.yaml` `overlap_tolerance` 0.70 → 0.80 σ: **approved**. It only aligns the dialect value with the 0.8 σ floor `check/statics.py` already enforces. | W2 |
| D2 | `overlap_sanity_fraction`: **0.70 d_NN for the metal dialect**, LJ stays 0.80 σ. Hot metals have real pairs at 0.75–0.78 d_NN (the Cu reference's own floor is 1.95 Å = 0.76 d_NN); 0.70 still catches every planted overlap seen so far (0.40 d_NN). | W2 |
| D3 | `lj_solid_liquid` solid-density tolerance 2.5% → 4.0%: **approved**. The target stays the construction value and the measured spread is −3.5% to +1.0%. No further widening; if a regeneration needs more, enlarge the solid slab instead. | W2 |
| D4 | Ratify every new dialect key listed in the gate report, **including the four `thermal_quench_*` keys** (A4 passed 60 of 60 fresh seed sets in the reviewer's check): **approved**. | W2 |
| D5 | Process: one branch per stream; merge to `master` only when that branch's CI is green, and name the run URL in the merge commit. A change to `tools/acceptance.py`, `bench/` or a dialect also needs a clean-runner acceptance run before merge. Use GitHub pull requests if you can open them. **Approved.** | all |
| D6 | A3's structure gate is the translation-aligned, species-aware assignment check (at most 5% of matched residuals beyond 0.25 d_NN): **approved** as a recorded deviation from PLAN's StructureMatcher. Write the reason into PLAN.md A3: primitive-cell reduction fits 0/9; `primitive_cell=False` fits 8/9 but never `l12_ni3al` (about 800 s per call); evidence in `tests/acceptance/test_o2_o3.py`. | W2 |
| D7 | Glass protocol base values 1500/3000/2850 at `glass_protocol_ref_n` 500: **provisional**. They stay for the frozen monatomic reference and are never tuned against it again; W7 re-derives them from the Kob–Andersen reference. | W7 |
| D8 | Rebuild time budget max(150 s, 150 s × N/1000): **approved** (a timeout, not physics). | W2 |
| D9 | Replace the glass reference with the **Kob–Andersen 80:20 binary LJ glass at ρ = 1.2**: **approved**, with the spec changes it needs (several species in an amorphous region, a named LJ mixture model, partial g(r) and coordination observables). Details in W7. | W7 |
| D10 | `glass.yaml` `q6_cutoff` 3.0 → 1.3: **rejected as the fix**. It makes all 9 Å bench crystals "amorphous". Replace it with the unit-free rule of W1, then delete `q6_cutoff` from `glass.yaml`. Keep 1.3 only until W1 merges. | W1 |
| D11 | `metal.yaml` `printed_cutoff` 3.25 Å: **approved** as metadata (the F-S cutoff). What the slab lift may print is decided by W4. | W2 |
| D12 | `cu_solid_liquid` sanity restatement (solid target 0.0751 Å⁻³ ± 6%, liquid ± 5%, g(r) window 2.44–2.64 Å): **rejected**. The new solid target is the mean of the same frames it checks, so it cannot fail. Interim: record `known_limitation` = "solid window includes the premelted interface; no independent density target yet" (the case gates nothing). If Cu ever returns to a criterion: target from a bulk NPT run of the same potential at 1335 K (about 0.079 Å⁻³ by your own note), an interior window at least one cell from each interface, tolerance ≤ 3%. Also fix the note that says 8 ps when the window is 4 ps. | W2 |
| D13 | `fcc`/`hcp` hollow names: **deferred** until after the gate (spec change: `site hollow fcc` / `site hollow hcp`, needs second-layer registration). Generic `hollow` stays. | later |
| D14 | `co2_dense` at 0.68 g/cm³: **approved** as the bench state point. The packer ceiling and contact distances become W8. | W8 |
| D15 | Floor rule max(mean, P90): **keep** (P90 > mean in 14/14 records). | — |
| D16 | Gate A′ is declared when W1–W6 are merged, D1–D15 are implemented, a clean-runner acceptance run on the final commit is committed, and a fresh verifier report exists (W3). W7–W12 do not block the gate, but W7 starts now. | W3 |

New AGENTS.md rules (approved; W2 adds them):

1. Move "Read `docs/reviews/open_items.md` before starting work" to the **top** of
   the rules.
2. A threshold change is tested in every unit system its dialect serves (Å and
   σ), on crystals and on disordered frames.
3. Fail closed. A missing threshold or a failed check raises. A lift arm falls
   through to the next arm only on an explicit "not this phase" verdict. A lift
   that cannot describe a frame refuses with an error; it never prints a program
   it cannot stand behind. A value the lift did not measure and was not given
   (temperature, model) is printed as assumed.
4. A sanity target never comes from the frames it checks.

## 2. Work orders

### W1 · Crystal, glass and gas detection must be unit-free and fail closed (P1, gate)

The O9 fix set `glass.yaml` `q6_cutoff` to 1.3, meant in σ. The bench crystals
are in Å, where 1.3 Å holds no neighbour, so `is_amorphous` calls every one of
them amorphous. Under `core + glass` the defects arm cannot run (it needs
`site_match_tol_fraction`, which only `metal.yaml` defines). Unless the spglib
arm happens to accept a frame, the cascade falls through silently and the
crystal lifts as `amorphous glass`. All 9 did in the reviewer's session, where
spglib was a stand-in, so run the reproducer in your environment. The same q6
weakness makes the fluid check call dilute argon "solid-like" (an atom with one
or two neighbours has q6 near 1).

```python
from chaord.lift.amorphous import is_amorphous
gl = load_dialect(("core", "glass"))
bad = [c["id"] for c in acc._exact_roundtrip_cases(None)
       if is_amorphous(acc._a2a3_frame(c), gl)]
print(len(bad), "of 9 bench crystals called amorphous under core + glass")
# today 9 of 9 (0 of 9 with the old 3.0); done: 0
ar = acc.case_by_id("fluid/ar_gas_box25")
try:
    lift_frame(read_frame(ar["frames"][0]), load_dialect(ar["dialect"]), mode="fluid")
    print("argon gas accepted by mode='fluid'")                      # done
except Exception as e:
    print("argon gas refused by mode='fluid':", e)
# today: refused, 'solid-like fraction too high' (A13 hides it: it falls back to auto)
```

No single q6 cutoff works. The reviewer measured the fraction of crystal-like
atoms (q6bar > 0.32) with the cutoff set to f × the frame's own median
nearest-neighbour distance (d_NN). Sc-like crystals need f ≤ 1.2, random packings
need f ≥ 1.25, and rocksalt and perovskite fail at every f:

| Frame | f = 1.15 | 1.2 | 1.25 | 1.3 |
| --- | --- | --- | --- | --- |
| rocksalt NaCl (Å, thermal) | 0.77 | 0.22 | 0.05 | 0.02 |
| perovskite SrTiO₃ (Å, thermal) | 0.75 | 0.72 | 0.57 | 0.57 |
| LJ sc, 0.10 d_NN jitter | 0.92 | 0.75 | 0.44 | 0.15 |
| LJ hcp, 0.10 d_NN jitter | 0.93 | 0.89 | 0.78 | 0.66 |
| bench RSA glass `lj_glass_rho085` | 0.59 | 0.32 | 0.17 | 0.05 |
| glass reference frames | 0.06–0.18 | 0.04–0.16 | 0.03–0.14 | 0.02–0.12 |

With a second rule (neighbours within 1.3 d_NN, at least 4 of them, and
q6bar > 0.32 **or** q4bar > 0.45), LJ fcc, bcc, hcp and sc and every Å crystal
except perovskite read crystal-like (0.65–1.00). All glasses, liquids, gases
and random packings read ≤ 0.12. LJ diamond (0.31) and perovskite (0.20) still
need the lattice fit, which is why step 1 comes first.

Steps:

1. `lift/amorphous.py::is_amorphous`: run the crystal fit first; if any
   prototype fits with site coverage ≥ 0.90, the frame is not amorphous. Make
   the fit runnable under every dialect stack: move the unit-free crystal-fit
   keys the defects arm reads (`site_match_tol_fraction`,
   `lattice_fit_tol_fraction`, `lattice_scan_*`, … check each one's unit) into
   `core.yaml` as defaults; `metal.yaml` may still override them.
2. Secondary local-order test, in `is_amorphous` and in the fluid check
   (`lift/fluid.py::is_single_phase`): neighbours within
   `q_cutoff_factor` × d_NN (new `core.yaml` key, 1.3), only atoms with at least 4
   neighbours count, crystal-like = q6bar > 0.32 or q4bar > 0.45, disordered when
   the crystal-like fraction is below `amorphous_solid_frac_max` (0.30).
3. Delete `except Exception: disordered = True` in `is_amorphous`.
4. `lift/legacy.py::legacy_lift`: introduce a `NotThisPhase` exception; an arm
   falls through only on it. A missing threshold or any other error propagates.
5. Delete `q6_cutoff` from `glass.yaml` once 1–2 land. List every other user of
   an absolute `q6_cutoff` and convert it or note why not.

Done when:

- the reproducer prints 0 of 9 and "argon gas accepted";
- the 9 Å bench crystals and LJ fcc, bcc, hcp, sc and diamond crystals at
  ρ* 0.85 and 1.1 with 0.10 d_NN Gaussian jitter (20 draws each) are not
  amorphous under `core + glass`, and their lift does not print `amorphous`;
- the 15 glass reference frames and the bench RSA glass lift as amorphous, the
  2,048-atom liquid as a fluid;
- `ar_gas_box25`, `n2_box22`, `co2_dense` and `water_box15` pass `mode="fluid"`
  with no routing fallback in A13's diagnostics.

### W2 · Records and decisions (P1, gate)

`reports/acceptance.json` is a local Windows run with a dirty tree on
`9f86f92`, four commits behind `HEAD`. The gate report's body still cites A2
"18/18 transforms", clean run #39 (2026-09-30), the 2026-09-30 verifier report
and a five-item decision list. Its O2 line says the reviewer's "matcher never
fits self" claim was disproven. That claim was your own: the `_geo_fit_thermal`
docstring at `14cf5d4` said "no fit even for the frame against ITSELF". The
reviewer quoted it and named primitive-cell reduction as the likely cause, which
your test then confirmed.

Steps:

1. Implement D1, D2, D3, D4, D6 (PLAN.md text), D8, D11, D12 and D15; remove the
   "PENDING" markers they settle.
2. Replace the gate report's decision list with section 1 of this file, each
   marked implemented or not. Refresh the body: A2 evidence (50 × 9), CI
   evidence (W3), per-case A5 numbers, and the O2 line corrected as above.
3. AGENTS.md: the four rules above (rule 1 = move to the top).
4. README status: name what is open. Suggested text: "v0.1 prototype: 14/14 on the
   committed inputs (clean run <URL>). Known limits: the LJ glass case
   reproduces the reference's cavitation, not its quench history (Kob–Andersen
   replacement in progress); single-species metal solid–liquid interfaces are
   not supported; strained boxes until W5 lands."

Done when: `git grep -n -i "pending owner\|pending human\|pending approval"`
lists only undecided items (none from section 1); README, dossier and gate
report agree with each other.

### W3 · Clean run, fresh verifier, gate (P1, last)

After W1, W2, W4, W5 and W6 are merged: run the clean-runner workflow on the
final commit and commit its `acceptance.json` (run kind clean, run URL, the
commit = `HEAD`), not a local run. Then a fresh agent that wrote none of this
code runs A1–A14 and every snippet in this file and writes
`reports/verification_<date>.md` with each result. Then declare Gate A′ in the
gate report.

### W4 · Single-species metal interfaces: refuse, then lift correctly (P1, gate = step 1)

With the O11 key the Cu solid–liquid frame now lifts and builds, but the program
is wrong. Copper becomes `X` in LJ units at 300 K (the frame is at 1335 K and
nothing marks the 300 K as assumed). The fcc solid is called hcp, most likely
because the antiparallel-pair test (`chi0`) misreads a solid at its melting
point. It also lists 11 `frenkel_pair` lines and a vacancy.

```python
f = load_ref("bench/reference/cu_solid_liquid/frame_0.npz")     # 832 Cu, 1335 K
text = acc.format_program_text(lift_frame(f, load_dialect(("core", "metal"))))
print([l.strip() for l in text.splitlines()
       if l.strip().startswith(("units", "conserve", "state T", "backend", "lattice"))])
print(text.count("frenkel_pair"), "frenkel_pair lines")
# today: units lj, state T 300.00, conserve atoms X 832, backend lj, lattice hcp;
#        11 frenkel_pair lines
```

Steps:

1. For the gate: the slab path refuses a frame whose species are real elements
   when it would print them as `X` or in LJ units (`ChaordError`: "single-species
   metal solid–liquid interfaces are not supported yet"). Red test: the
   reproducer raises. README limitation line (W2).
2. After the gate: carry species and units through the slab path (Cu, Å, eV,
   K); print T from the caller or as assumed (W11); make the fcc/hcp test hold
   at 0.9–1.0 Tm (calibrate `chi0_antiparallel_cos` or average neighbour vectors
   on thermal fcc and hcp frames); print defect lines only above a
   detection floor measured on defect-free thermal frames. Done when the
   reproducer prints `conserve atoms Cu 832`, metal units, `lattice fcc`, a
   within 2% of the potential's lattice constant at 1335 K, and at most one
   defect line, and the program builds.

### W5 · Strained boxes must build (P1, gate)

NPT runs that let the box lengths fluctuate independently (LAMMPS `aniso` or
`tri`) give boxes slightly off an exact tiling. A 0.2% stretch along x makes 6
of the 9 crystal cases lift to a program Chaord cannot build
("system cell … is not an integer multiple of the … lattice vectors"). The text
stays translation-invariant; only the build fails.

```python
bad = []
for case in acc._exact_roundtrip_cases(None):
    dl = load_dialect(case["dialect"])
    f = acc._a2a3_frame(case)
    s = np.array([1.002, 1.0, 1.0])
    g = Frame(pos=f.pos * s, cell=np.diag(f.cell_diag * s), symbols=f.symbols, pbc=f.pbc)
    try:
        build_program(lift_frame(g, dl), dl, rng=np.random.default_rng(1), physics=False)
    except Exception:
        bad.append(case["id"])
print(len(bad), "of 9 strained crystals do not build:", bad)
# today 6 of 9 (rocksalt_nacl, l12_ni3al, hcp_mg, diamond_si, fcc_crconi, cuau_random); done: 0
```

Steps: in the builder, when a crystal region's stated cell is within 2% of an
integer tiling on every axis but outside `lattice_match_tolerance`, tile with
the nearest integer counts and scale each axis to the stated cell. Record
"strained to cell: e_xx e_yy e_zz" in provenance. Above 2%, refuse as today.
The lift is unchanged.

Done when: ±0.2%, ±0.5% and ±1.5% along x, y and z, and isotropic, build for
all 9 cases; lift → build → lift gives identical text for each.

### W6 · The synthetic solid–liquid frames must be physical (P1, gate)

`bench/data/interface/lj_solid_liquid` has atoms 0.047–0.18 σ apart in its
liquid half (pinned in `tests/test_bench_sanity.py`). A7's phase-segmentation
evidence on those frames separates a crystal from overlapping random points,
which is too easy.

```python
from scipy.spatial import cKDTree
for k in range(5):
    fr = read_frame(f"bench/data/interface/lj_solid_liquid/frame_{k}.npz")
    L = fr.cell_diag; p = np.mod(fr.pos, L)
    print(k, round(float(cKDTree(p, boxsize=L).query(p, k=2)[0][:, 1].min()), 3))
# today 0.114, 0.055, 0.047, 0.072, 0.182 sigma; done: >= 0.80 in every frame
```

Steps: regenerate the liquid half by MD (LJ, T* about 0.7) or by RSA at ≥ 0.85 σ
followed by a short MD relaxation; keep the solid half. Do it in its own reviewed
commit with `checksums.json`. Remove the pin. Re-run A6, A7 and A13 and report
A7 before and after.

### W7 · Kob–Andersen glass reference (P1 stream, starts now, not gate-blocking)

Steps, in order:

1. Spec (D9): an amorphous region may hold several species
   (`conserve atoms A 1600 B 400`); the physics block names a model from a new
   dialect table, `lj_mixtures.kob_andersen`: ε_AA 1.0, σ_AA 1.0; ε_AB 1.5,
   σ_AB 0.8; ε_BB 0.5, σ_BB 0.88; equal masses; each pair cut at 2.5 σ_αβ and
   shifted (Kob and Andersen, Phys. Rev. E 51, 4626, 1995).
2. LJ engine (`realize/lj.py`): per-pair ε and σ, with a test against
   hand-computed energies and forces for an AA, AB and BB pair.
3. Reference (`bench/reference/generate_reference.py`, no chaord import):
   N = 2000 (1600 A, 400 B), ρ = 1.2, NVT, dt 0.005 τ. Equilibrate at T = 2.0
   for 20,000 steps, quench linearly to T = 0.1 over 20,000 steps, anneal
   4,000 steps, then sample 5 frames 800 steps apart. Do 3 independent quenches.
4. Sanity, using the checker you committed: pressure > 0 in every frame, largest
   empty sphere radius < 1.0 σ_AA, crystal-like fraction < 1% (W1 rule), energy
   drift across the sampled frames < 0.005 per atom.
5. Lift: composition, density, partial g(r) and coordination asserts, and the
   model printed as assumed (inferred from the composition and σ_AB/σ_AA from
   the first partial g(r) peaks). History default from the dialect.
6. A5 for binary amorphous regions: partial g(r) (AA, AB, BB) rms and partial
   coordination, floors from cross-quench pairs (5 frames × 3 quenches, ≥ 10
   pairs). Re-derive D7's protocol values from this reference.
7. Measure and report default vs 10× faster quench vs a 100-step anneal, for
   the structural observables and for potential energy per atom. If structure
   cannot tell a 10× quench-rate change apart and energy can, add energy per
   atom as a glass observable with its own floor.
8. When the Kob–Andersen case passes A5, retire the monatomic `lj_glass` from
   A5 (keep it only as a documented example of a cavitated solid, or delete it).

Done when: the new reference passes its sanity checks; A5's glass case is
Kob–Andersen; the report states the quench-rate resolution with numbers.

### W8 · Molecular inputs: contact distances and liquid densities (P2)

The regenerated `co2_dense` frames place oxygens of different molecules
1.70–1.81 Å apart, 0.56 of the van der Waals contact (Bondi O–O 3.04 Å), with
24–40 such pairs below 0.75 of contact per frame. The sanity floor for molecular
frames is the covalent window, so they pass. The builder cannot make liquid CO2
at 1.0 g/cm³ (0 of 3 seeds: "cannot place 60 CO2 at this density"). That is why
the case was lowered to the packer's ceiling.

```python
from chaord.lang.ir import Quantity
dl = load_dialect(("core", "molecular"))
f = read_frame("bench/data/gases/co2_dense/frame_0.npz")
p = lift_frame(f, dl, mode="fluid")
L1 = f.cell_diag[0] * (0.68 / 1.00) ** (1 / 3)     # the same 60 CO2 at 1.00 g/cm3
for b in p.blocks:
    for s in getattr(b, "statements", []):
        if s.key == "cell":
            s.values = [Quantity(num=f"{L1:.3f}") if isinstance(v, Quantity) else v
                        for v in s.values]
        if s.key == "density":
            s.values = [Quantity(num="1.00", unit=getattr(v, "unit", None))
                        if isinstance(v, Quantity) else v for v in s.values]
for seed in range(3):
    try:
        build_program(p, dl, rng=np.random.default_rng(seed), physics=False)
        print(seed, "builds")
    except Exception as e:
        print(seed, e)
# today: 'cannot place 60 CO2 at this density' for all 3 seeds; done: builds on 20 of 20
```

Steps:

1. Sanity: intermolecular heavy-atom pairs ≥ 0.75 × the Bondi sum, with an H-bond
   exception (H···O/N ≥ 1.5 Å). Report every bench frame that fails
   (`co2_dense`, `water_box15` and `reactive/water_oh_h_box20` will). Do not
   loosen the rule to pass them.
2. Builder: molecular fluids fall back to the grid + minimise packing the water
   path uses when RSA jams. Red test: 60 CO2 at 1.00 g/cm³ build on 20 of 20
   seeds with every contact above the new floor.
3. Regenerate the failing synthetic molecular frames with pack-then-relax, each
   in its own reviewed commit with checksums.

### W9 · Adsorption sites beyond fcc(111) (P2)

On Pt(100) the 4-fold hollow is the midpoint of a Delaunay triangle's edge, so
the classifier calls it "bridge" (4 of 4 O, 5 of 5 draws).

```python
import re
a = 3.92; ann = a / np.sqrt(2); n = 6; nl = 4
P = np.array([[((i + 0.5 * (k % 2)) * ann) % (n * ann), ((j + 0.5 * (k % 2)) * ann) % (n * ann), k * a / 2]
              for k in range(nl) for i in range(n) for j in range(n)])
L = np.array([n * ann, n * ann, 22.0])
top = P[np.isclose(P[:, 2], (nl - 1) * a / 2)][0, :2]
hol = np.mod(top + [[0.5 * ann, 0.5 * ann], [2.5 * ann, 2.5 * ann], [2.5 * ann, 4.5 * ann], [4.5 * ann, 0.5 * ann]], L[:2])
O = np.column_stack([hol, np.full(4, (nl - 1) * a / 2 + 1.0)])
rng = np.random.default_rng(0)
pos = np.vstack([P, O]) + rng.normal(scale=0.08, size=(len(P) + 4, 3)); pos[:, 2] += 3.0
fr = Frame(pos=np.mod(pos, L), cell=np.diag(L), symbols=["Pt"] * len(P) + ["O"] * 4, pbc=(True,) * 3)
text = acc.format_program_text(lift_frame(fr, load_dialect(("core", "metal", "surface"))))
print(re.findall(r"adsorb \S+ count \d+ site \S+", text))
# today ['adsorb O count 4 site bridge']; done: site hollow
```

Steps: classify by lateral coordination. Count the top-layer atoms whose lateral
distance to the adsorbate is within the smallest one + 0.15 a_NN: 1 = top,
2 = bridge, 3 or 4 = hollow. Keep the periodic images. Done when Pt(100) 4-fold
hollows and bridges are ≥ 90% correct over 20 draws and 20 translations, and
Pt(111) stays at 100%.

### W10 · Determinism (P2)

Your own report: `surfaces/si001_2x1` lifts differently after other lifts in the
same process (`site far`). The reviewer could not trigger it: all 125 bench
frames lifted forward and then in reverse order in one process gave identical
text, so the trigger may depend on your environment. Write the red test (lift
`si001_2x1` in a fresh subprocess, and again after lifting every other bench
frame) and find the hidden state: module caches, unseeded randomness, set or
dict order. Also use quantized keys (round to 1e-6 Å) for every canonical sort
on float coordinates; your `hcp_mg` regeneration found 1e-9 Å noise flipping a
lexsort.

### W11 · Assumed values flagged on every lift path (P2)

The fluid path marks an assumed temperature; the slab path prints
`state T 300.00` with no mark (W4 reproducer). Every value that was neither
measured from the frame nor passed by the caller is printed as assumed, the
same way on every path. Red test: lift a slab frame without T; the program
flags T as assumed.

### W12 · Leftovers (P2)

- Demo Act 0: also lift a shifted Cu copy and show the same text (v1 O6).
- EC and PF6⁻ templates for `solutions/lipf6_ec` (also needed in section 5).
- `_a5_rebuild_meta` crashes on `cu_solid_liquid`: raise a `ChaordError` with
  the reason, or fix.

## 3. Streams, order and merge rules

| Stream | Work orders | Owns | Starts | Blocks the gate |
| --- | --- | --- | --- | --- |
| S1 Classifier | W1 | `lift/amorphous.py`, `lift/fluid.py`, `lift/legacy.py`, `core.yaml`, `glass.yaml` | now | yes |
| S2 Records | W2 | `reports/`, `README.md`, `AGENTS.md`, `PLAN.md`, decided dialect values | now | yes |
| S3 Metal interface | W4 step 1, W11 | `lift/slab.py` | now | yes (W4 step 1) |
| S4 Strain | W5 | the crystal builder | now | yes |
| S5 Inputs | W6, then W8 | `bench/data` (each regeneration its own commit), sanity tests | now | W6 yes, W8 no |
| S6 Glass v3 | W7 | `realize/lj.py`, amorphous lift/build, `bench/reference`; `glass.yaml` and `generate_reference.py` only after S1 and S2 merge | now | no |
| S7 Surfaces and determinism | W9, W10 | `lift/surface.py`, canonical sorts | after S3 merges | no |
| S8 Gate | W3 | `reports/` | after S1–S5 (W6) merge | — |

Merge order: S2 first; then S1; then S3, S4 and S5 in any order, one at a time,
each with a green CI run (D5); then S8. S6 and S7 merge whenever ready, after
the gate or before it, as long as each merge keeps CI green.

## 4. Gate A′ checklist

- [ ] W1–W6 merged, each with its done-when test.
- [ ] D1–D15 implemented; the gate report lists them.
- [ ] README, dossier and gate report agree; limits named (W2).
- [ ] `acceptance.json` from a clean run on the final commit, with its URL (W3).
- [ ] A fresh verifier report covering A1–A14 and every snippet in this file (W3).

## 5. After the gate: first research use

The target is the owner's own system: carbonate solvents (PC, EMC) confined in
a carbon nanotube under an electric field, from the owner's Moltemplate and
LAMMPS runs.

1. **Inputs.** A LAMMPS data/dump reader (type → element from the data file's
   masses), triclinic and NPT boxes (W5 first).
2. **Language.** A cylindrical region; a nanotube description (chirality (n, m),
   length, from the carbon positions); PC, EMC, EC, PF6⁻ and Li⁺ templates; an
   external-field statement (given by the user, printed as assumed if not), plus
   the measurable consequences: a dipole-orientation order parameter and the
   radial density profile.
3. **First result.** Lift 20 frames with the field off and 20 with it on, and two
   tube diameters. Done when the programs' statements (not coordinates)
   separate the states beyond the noise floor, and a classifier that reads only
   the statements tells field on from off with ≥ 95% accuracy.
4. **Laya loop.** Choose frames for DFT labelling from their programs (where the
   MLP is uncertain) and compare the MLP force error per label against random
   selection.
