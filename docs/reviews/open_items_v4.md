# Open items and decisions, v4 (external Review 10, 2026-10-15)

Read this file before any work. It supersedes `open_items_v3.md`, `open_items_v2.md`
and `open_items.md` where they conflict. Brief v3 never reached the repo: commit it
next to this file, because v3's work orders X1–X7 and X9 still stand unchanged
(section 3 below re-checks each one). Decisions D1–D27 stand unless amended here.

- Reviewed HEAD: `2990d08` (9 commits after Gate A′, `4c36697` to `2990d08`).
- Every code block below ran on `2990d08`, from the repository root, and printed
  what the block says. Run it before you change anything, and again when you
  think you are done.
- The owner's inputs live in `docs/reviews/owner_inputs.md`: the working field
  strength(s) in V/Å and the Moltemplate model of the tube and electrolyte. If the
  file is missing or says "unknown", the working field is 0.01 V/Å (D30) and Stage A
  uses the pilot model with a recorded note (D37).

## 0. What the reviewer re-checked on `2990d08`

- The pilot numbers are real. Reviewer code (numpy only, no chaord) recomputed P1
  from `final.data` charges and the raw dumps: cnt10 −0.121 ± 0.076 (off) and
  +0.725 ± 0.078 (on); cnt16 +0.152 ± 0.030 (off) and +0.606 ± 0.053 (on), standard
  deviations over the 21 production frames. All four match `RESULTS.md` to 0.001. With each frame's measured P1, a threshold still
  scores 100% (leave-one-out and both cross-diameter directions) at this field.
- The field saturates the solvent. The model's EC dipole is 6.06 D and EMC's is
  0.78 D; at 0.2 V/Å, μE/kT is 9.8 for EC and 1.25 for EMC. Free dipoles would reach
  P1 = 0.90 (EC) and 0.38 (EMC); the runs give EC 0.74–0.94 and EMC 0.38–0.40.
  Linear response for EC ends near 0.02 V/Å, where μE ≈ kT.
- Orientation did not equilibrate (block 2). With no field, half of the molecules
  never reverse their axial dipole in 500 ps, and each molecule's 500 ps mean of
  cos θ spreads by 0.57–0.61 (free rotation gives about 0.1). The cnt10 zero-field
  P1 swings from +0.25 to −0.24 over the history; cnt16 stays between +0.03 and
  +0.20. The field-on runs started at 10 K with the field on; cnt16 rose from 0.46
  at the end of the ramp to about 0.6 in production.
- The lift is not blind (blocks 1 and 6). The P1 statement exists only when the
  caller passes `field=`.
- Real tube frames do not lift (block 3), the reader refuses the project's own
  dump (block 4), and no lift arm calls the nanotube inference. In the reviewer's
  session (spglib was a stand-in), `lift_frame` refused an ideal (10,10) tube under
  `core + carbon`, `core + molecular + carbon` and
  `core + molecular + carbon + research`.
- The EC template is not EC (block 5).
- v3's reproducers still fire: 9 of 20 clean Pt(111) slabs print an adsorbate
  (X1); a quench asking for 190,000 steps runs 20,000 (X5); the glass program
  still prints `cn` at 1.19 σ and `ring_mode 3` (X6).

## 1. Decisions (binding, 2026-10-15)

| ID | Decision | Implemented by |
| --- | --- | --- |
| D28 | The overnight result is a pilot: the pipeline works end to end at a saturating field. `RESULTS.md` withdraws "confinement aligns the solvent" and "confinement-compressed" and records the measured non-equilibration. | Y1 |
| D29 | Lift blind. The orientation statement is measured for every frame whose dialect names dipolar species, whether or not `field=` is given; `field=` adds only the `state field` line and its assumed note. Rename `field_dipole_p1` to `dipole_p1` (registry, dialects, tests, spec, examples). | Y2 |
| D30 | A field ladder replaces the single comparison: 0, 0.005, 0.01, 0.02, 0.05, 0.1 and 0.2 V/Å, plus the owner's working value(s). Default working field: 0.01 V/Å. | Y6, Y7 |
| D31 | Judge ensembles, not frames. Block statistics from independent runs; floors from zero-field runs only (D15 estimator over block pairs from different runs); accuracy by leaving one run out. Per-frame accuracy may be reported but is never the gate. | Y8 |
| D32 | Go/no-go on 2026-10-29, replacing D25's test. GO if Chaord's blind block statements detect the field at the same smallest field as a direct P1 analysis without chaord (within one ladder step), and that field is at or below the working field. Report every other statement that moves beyond 2× its floor. NO-GO: Chaord stays the builder and recorder for the owner's MLP paper, off its critical path, and R4 does not start. | Y8 |
| D33 | Real-frame tube lifts are R2's bar: per-coordinate noise up to 0.1 Å, 6-significant-digit text, periodic cells of 3 or more axial periods, any position in the cell, tube diameter up to 0.8 of the perpendicular cell length — each 20 of 20 correct, radius within 0.05 Å. Tolerances live in the carbon dialect, with evidence. | Y3 |
| D34 | The reader wraps periodic coordinates and records the largest excursion; accepts unwrapped columns; raises on an ambiguous mass (two candidates inside the window, or a united-atom CHn mass) unless an explicit `type_map` is given. | Y4 |
| D35 | The library composes tube plus liquid: `lift_frame(mode="auto")` returns the nanotube region and the confined fluid. Research scripts call only the public API. | Y3 |
| D36 | Approved spec additions: the Li⁺ solvation-shell statement by species, ion-pair classes (SSIP/CIP/AGG), species radial profiles inside the tube radius, density on the accessible volume. Cutoffs come from reference runs (Stage A zero-field runs), never from the frames under test. | Y5 |
| D37 | Paper-grade runs use the owner's Moltemplate model with its native mixing rule and a stated charge scaling; the filling comes from a reservoir run; one thermostat scheme covers all mobile atoms; dumps keep 10 significant digits; raw trajectories stay out of git, listed in a sha256 manifest. The pilot set stays committed as the pilot record. | Y6, Y7 |
| D38 | This brief supersedes v3 where they conflict; v3's X1–X7 and X9 stand. AGENTS.md's first rule points here. | owner |

## 2. Work orders

### Y1 · Correct the research record (P0, stream S2, half a day)

Reproducers, from the repository root:

```python
import json
S = json.load(open("research/cnt_electrolyte/runs/first_result_summary.json"))
for on in (False, True):
    texts = [t for k, t in S["program_texts"].items() if S["systems"][k.split("#")[0]]["field"] == on]
    print("field on: " if on else "field off:", sum("field_dipole_p1" in t for t in texts), "of", len(texts), "programs carry the P1 line")
# prints:
# field off: 0 of 42 programs carry the P1 line
# field on:  42 of 42 programs carry the P1 line
```

```python
# Chaord-free: does each solvent molecule's axial dipole relax during the 500 ps after the ramp?
import numpy as np
RUN = "research/cnt_electrolyte/runs/"

def atoms(path):                       # LAMMPS data file, Atoms # full: id mol type q x y z
    L = open(path).read().splitlines()
    i = next(k for k, l in enumerate(L) if l.startswith("Atoms"))
    rows = []
    for l in L[i + 2:]:
        if not l.strip():
            break
        rows.append(l.split()[:7])
    a = np.array(rows, float)
    return a[np.argsort(a[:, 0])]

def snapshots(path):                   # LAMMPS dump: id mol type q x y z
    out, f = [], open(path)
    for line in f:
        if line.startswith("ITEM: NUMBER OF ATOMS"):
            n = int(next(f)); next(f)
            box = np.array([next(f).split()[:2] for _ in range(3)], float); next(f)
            a = np.array([next(f).split() for _ in range(n)], float)
            out.append((box[:, 1] - box[:, 0], a[np.argsort(a[:, 0])]))
    return out

for name in ["cnt10_field0", "cnt16_field0", "cnt10_field1", "cnt16_field1"]:
    top = atoms(RUN + name + "/final.data")
    mol, q = top[:, 1].astype(int), top[:, 3]
    solvent = [np.where(mol == m)[0] for m in np.unique(mol) if (mol == m).sum() in (10, 15)]  # EC, EMC
    cos = []
    for L, a in snapshots(RUN + name + "/equil.dump") + snapshots(RUN + name + "/prod.dump")[1:]:
        row = []
        for idx in solvent:
            d = a[idx, 4:7] - a[idx[0], 4:7]
            d -= L * np.round(d / L)
            mu = (q[idx, None] * d).sum(0)
            row.append(mu[2] / np.linalg.norm(mu))
        cos.append(row)
    c = np.array(cos)                  # frames 50..550 ps
    never = int((~((c > 0.5).any(0) & (c < -0.5).any(0))).sum())
    print(f"{name}: P1 range {c.mean(1).min():+.2f}..{c.mean(1).max():+.2f}; "
          f"{never} of {c.shape[1]} molecules never reverse; "
          f"per-molecule 500 ps mean cos, sd {c.mean(0).std():.2f}")
# prints:
# cnt10_field0: P1 range -0.24..+0.25; 7 of 13 molecules never reverse; per-molecule 500 ps mean cos, sd 0.61
# cnt16_field0: P1 range +0.03..+0.20; 38 of 61 molecules never reverse; per-molecule 500 ps mean cos, sd 0.57
# cnt10_field1: P1 range +0.55..+0.84; 10 of 13 molecules never reverse; per-molecule 500 ps mean cos, sd 0.38
# cnt16_field1: P1 range +0.46..+0.71; 27 of 61 molecules never reverse; per-molecule 500 ps mean cos, sd 0.30
```

Steps:

1. Commit the second block as `research/cnt_electrolyte/orientation_check.py`, so
   every number you add has a script behind it.
2. `RESULTS.md`: title and verdict become "Pilot at a saturating field (0.2 V/Å,
   about 10 kT per EC dipole)". Say what it shows (the pipeline runs end to end;
   the field-on and field-off P1 ranges never overlap) and what it does not show
   (weak-field sensitivity; equilibrated zero-field orientation; independent runs).
3. Section 3: delete "confinement aligns the solvent" and the anti-alignment
   reading. Replace them with the symmetry argument (an achiral tube is symmetric
   under z → −z, so the equilibrium zero-field P1 is 0) and the output of
   `orientation_check.py`.
4. Section 5: add the output of the first block, and that a threshold on each
   frame's measured P1 also scores 100% at this field (from `records[].p1_measured`).
5. `research/cnt_electrolyte/README.md`: replace "confinement-compressed" with
   "packed to the target composition; filling not equilibrated with a reservoir".
6. Do not touch `params.py` (S3 owns it; S3 fixes the guessed DOI in Y6).

Done when: the diff touches only `RESULTS.md`, `README.md` and the new
`orientation_check.py` under `research/cnt_electrolyte/`, and every number added
is printed by a committed script.

### Y2 · Blind lift (P0, stream S4, one day)

Reproducer:

```python
import sys
sys.path[:0] = ["src", "research/cnt_electrolyte"]
import analyze_first_result as A
from chaord.lift import lift_frame
from chaord.lang.fmt import format_program

frames, _, _ = A.read_prod_frames("cnt10_field0")
_, electrolyte = A.split_tube(frames[0], A.composition_dialect())
for field in (None, {"value": 0.0, "axis": "z"}):
    text = format_program(lift_frame(electrolyte, A.composition_dialect(), T=300.0, mode="fluid", field=field))
    p1 = [l.strip() for l in text.splitlines() if "field_dipole_p1" in l]
    print("field =", field, "->", p1 or "no P1 statement")
# prints today:
# field = None -> no P1 statement
# field = {'value': 0.0, 'axis': 'z'} -> ['assert field_dipole_p1 -0.213 +- 0.277 along z']
```

Steps:

1. Write the red test first: lift all 84 committed production frames with no
   `field=`; every program must carry `assert dipole_p1 ... along z`. Its power
   mutation (P1 gated on `field` again) must fail it.
2. In `lift/__init__.py`, measure the orientation whenever the dialect names
   dipolar species (`field_dipole_species`) and the frame holds them. Axis: the
   field's axis when given; else the nanotube region's axis when one exists; else
   the dialect's default analysis axis (add `dipole_axis_default: z` to the
   research dialect, with a comment).
3. `field=` keeps only its own job: the `state field` line and the assumed-value
   note.
4. Rename `field_dipole_p1` to `dipole_p1` everywhere (CV registry, dialects,
   tests, reference entry, `spec/examples/11_nanotube_field.chaord`).
5. In `_attach_field_observables`, replace `except Exception` with the documented
   not-measurable errors. Anything else propagates (fail closed).

Done when: the reproducer, edited to search for `dipole_p1`, prints the same line
for both calls, and the only difference between the two texts is the `state field`
line and its note.

### Y3 · Real tube frames, and tube plus liquid in the library (P1, stream S4, 3–4 days)

Reproducer:

```python
import sys, numpy as np
sys.path.insert(0, "src")
from chaord.build.nanotube import nanotube_positions
from chaord.lift.nanotube import infer_nanotube_chirality

rng = np.random.default_rng(2026)

def score(n, m, periods, box=None, noise=0.0, digits=None, shift=False):
    pos, length, d = nanotube_positions(n, m, periods)
    L = np.array([box or d + 20.0, box or d + 20.0, length])
    good = 0
    for _ in range(20):
        p = pos + L / 2 + rng.normal(0.0, noise, pos.shape)
        if shift:
            p[:, :2] += rng.uniform(0, 1, 2) * L[:2]
        p = np.mod(p, L)
        if digits:
            p = np.array([[float(f"{v:.{digits}g}") for v in r] for r in p])
        try:
            r = infer_nanotube_chirality(p, L)
            good += (r.n, r.m) == (n, m)
        except Exception:
            pass
    return f"{good}/20"

print("ideal (10,10), 20 periods, centred      ", score(10, 10, 20))
print("same, 300 K-like noise 0.05 A           ", score(10, 10, 20, noise=0.05))
print("same, written with 6 significant digits ", score(10, 10, 20, digits=6))
print("(10,10), 4 periods (a DFT-sized cell)   ", score(10, 10, 4))
print("(16,16) in its own 41.71 A cell         ", score(16, 16, 20, box=41.71))
print("(8,8), 6 periods, anywhere in the cell  ", score(8, 8, 6, shift=True))
# prints today:
# ideal (10,10), 20 periods, centred       20/20
# same, 300 K-like noise 0.05 A            0/20
# same, written with 6 significant digits  0/20
# (10,10), 4 periods (a DFT-sized cell)    0/20
# (16,16) in its own 41.71 A cell          0/20
# (8,8), 6 periods, anywhere in the cell   9/20
```

The causes, in `lift/nanotube.py`: the radius gate is exact (`_RADIUS_TOL = 1e-6`
relative); the axis is the largest-variance direction of the wrapped coordinates,
which fails for tubes shorter than √6 R and for tubes that straddle a periodic
face; `_unroll` anchors the perpendicular coordinates on atom 0 by minimum image,
which fails once the diameter exceeds half the cell.

Suggested method (the agent may choose another that meets D33):

1. Unwrap the carbon component through its bond graph (breadth-first, adding
   minimum-image bond vectors), never by minimum image to one anchor atom.
2. Axis: the cell vector along which the unwrapped component winds (a periodic
   tube's bond graph closes with a net shift of one cell length along its axis);
   for a finite tube, the longest principal extent of the unwrapped component.
3. Radius: least-squares circle fit in the perpendicular plane; gate on the RMS
   residual against a dialect threshold (propose one with evidence from 300 K and
   600 K frames).
4. (n, m): the lattice constant from the mean C–C bond (a = √3 d_CC); the chiral
   angle from the bond directions in the unrolled sheet (circular mean of
   exp(6iφ) over all bonds); then the integer pair that best fits both |Ch| = 2πR
   and the angle. Refuse when the best and second-best pairs are within a dialect
   margin of each other.
5. A nanotube arm in `lift_frame(mode="auto")`: find all-carbon, three-coordinated
   components that pass the inference; emit the `crystal tube : cylinder ...`
   region; lift the atoms inside the cylinder as the confined fluid region with the
   molecular stack; report atoms outside as their own region.

Done when:

- the reproducer prints 20/20 on every line, and so do σ = 0.1 Å, the (12,0),
  (8,4), (12,8) and (6,5) tubes with 0.05 Å noise, and diameters up to 0.8 of the
  cell;
- `lift_frame` on raw `prod.dump` frames of all four systems (read with the Y4
  reader, no `final.data`, no cell tripling) returns (10,10) or (16,16) and the same
  fluid composition as `first_result_summary.json`;
- `analyze_first_result.py` drops `read_tube_reference`, the tripled
  `inference_cell` and `split_tube`, calls `lift_frame` instead, and reproduces the
  pilot's P1 numbers (port Y2's 84-frame test to `lift_frame` in the same merge).

### Y4 · The reader reads real dumps (P1, stream S4, half a day)

Reproducer:

```python
import sys
sys.path.insert(0, "src")
from chaord.io.lammps import read_lammps_dump
try:
    frames = read_lammps_dump("research/cnt_electrolyte/runs/cnt10_field0/prod.dump")
    print("read", len(frames), "frames")
except Exception as e:
    print(type(e).__name__, e)
# prints today:
# ChaordError prod.dump (timestep 5000): 1 atom position(s) outside the simulation box; LAMMPS data/dump coordinates must already be wrapped
```

Steps:

1. Wrap positions on periodic axes (fractional coordinates modulo 1) and record
   the largest excursion, in box lengths, in `frame.info`. Refuse only beyond a
   bound that signals corrupt data (half a box length).
2. Accept `xu yu zu` and `xsu ysu zsu`: wrap them, and keep the unwrapped copy in
   `frame.info` for transport analysis.
3. Ambiguous masses raise and ask for `type_map` (accepted by both readers): two
   elements inside the tolerance window, or a united-atom mass. With today's
   0.05 amu window, CH2 (14.027) silently becomes N (14.007) and CH4 (16.043)
   becomes O (15.999).

Done when: the reproducer prints `read 21 frames` for all four `prod.dump` files;
a CH2-mass data file raises unless given `type_map`; an `xu yu zu` dump round-trips.

### Y5 · Electrolyte statements (P2, stream S4, 3 days; approved by D36)

1. `cv/field.py`: the radial profile takes r_max from the nanotube region (or a
   caller argument). An empty outer shell raises the documented not-measurable
   error, never `ZeroDivisionError`.
2. Li⁺ solvation shell by species (carbonyl O of EC, carbonyl O of EMC, F of
   PF6⁻) and ion-pair classes (SSIP, CIP, AGG). Cutoffs: the first minimum of the
   Li–O and Li–F g(r) from the Stage A zero-field runs, written into the research
   dialect with their provenance; never measured on the frames under test.
3. Species radial number-density profiles inside the tube radius, bin width from
   the dialect.
4. Density: state it on the accessible volume π(R − r_wall)²L, with r_wall in the
   dialect (3.2 Å is what the generator used), and print the linear density
   (molecules per Å) next to it. Today the program prints 0.357 g/cm³ for a liquid
   at 1.28 g/cm³ in the pore.
5. Replace the molecule-centre `cn` assert for molecular fluids (0.0 ± 0.0 in 55 of
   84 frames) with site-based coordination, or drop it.

Done when: all 84 pilot frames lift with these statements and no
`ZeroDivisionError`, and 20 random rigid transforms of one frame give identical
text.

### Y6 · Stage A: relaxation times and filling (P0, stream S3, inputs today, about 4 days)

The generator rule stands: no `import chaord` in run generators or their analysis.

Model: the owner's Moltemplate model, with its native mixing rule and stated
charge scaling. If it is not in `docs/reviews/owner_inputs.md` by the end of
2026-10-15, use the pilot model and record that. Fix the guessed DOI in
`params.py` (`doi:10.1016/j.c2tec.2023.100157-ish`): give the exact one or drop
the citation.

Runs, both tubes, axial length 49.2 Å (as the pilot, so the numbers compare):

1. Filling: an open-ended tube in a bulk bath (1 M LiPF6 in 1:1 EC:EMC, at least
   3 nm of reservoir on each side), NPT at 1 atm and 300 K, 5 ns. Count molecules
   and ions in the middle of the tube, 1 nm away from each mouth, with block
   standard errors. Use these counts in every closed periodic tube from now on.
2. Zero field: 3 independent builds (new seeds, random orientations), plus one
   mirrored copy of build 1 (z → L − z for every atom); 2 ns of production each
   after equilibration.
3. Switch on: from each zero-field final state, the working field and 0.2 V/Å,
   2 ns each.
4. Switch off: from each 0.2 V/Å final state, field 0, 2 ns.

Settings: 1 fs steps with rigid molecules; one Langevin scheme for every mobile
atom, damping 1 ps (`fix rigid/small ... langevin` for the molecules, `fix nve`
plus `fix langevin` for Li⁺; no separate Nosé–Hoover on a handful of ions); dumps
every 1 ps with `dump_modify <id> format float %.10g` and image flags; thermo
every 1 ps.

Analysis (numpy only): the single-molecule dipole autocorrelation
C1(t) = ⟨μ̂(0)·μ̂(t)⟩ → τ_rot; the P1 autocorrelation and exponential fits to the
switch-on and switch-off curves → τ_M; the mirrored pair (does the zero-field
offset flip sign?); reversal counts as in Y1's second block.

Done when: `reports/research/stage_a.md` gives τ_rot and τ_M per tube and field
with errors, the mirrored-start verdict and the reservoir filling; inputs and
sha256 manifests are committed; raw dumps are not.

### Y7 · Stage B: the field ladder (P0, stream S3, starts after the owner reads Stage A)

- Fields: 0, 0.005, 0.01, 0.02, 0.05, 0.1 and 0.2 V/Å, plus the owner's values.
- Tubes: (10,10) and (16,16), 98.4 Å long (40 periods, twice the ions), filling
  from Y6.
- Independent runs: 5 at 0 and at the working field, 3 at every other field.
- Each run starts from an equilibrated zero-field state, switches the field on,
  equilibrates at least 5 τ_M, then produces at least max(10 ns, 20 τ_M).
- Dumps every max(10 ps, τ_rot) with 10 significant digits; sha256 manifest;
  the per-run P1(t) series (small) committed, raw dumps not.
- One run with a flexible tube (the owner's bonded CNT model) at 0 and at the
  working field, so Y3 meets real thermal tubes.
- Optional: if the owner's field-aware NEP runs, one tube, 3 runs at 0 and at the
  working field. The MLP paper needs that comparison anyway.

About 50 runs of 10 ns at 2,000–4,000 atoms. If Stage A finds τ_M above 1 ns in
either tube, Stage B waits for the owner's decision on run length, because the cost
scales with 20 τ_M. Done when every run has finished and its manifest entry and
P1(t) series are committed.

### Y8 · The go/no-go analysis (P0, stream S5, 2 days, after Y2, Y3, Y5 and Y7)

1. Read every frame with the Y4 reader and lift it with identical options:
   `lift_frame(mode="auto")`, no `field=`.
2. Blocks of at least 5 τ_M per run. For every numeric statement, block mean
   and standard error.
3. Floors: the D15 estimator over |Δ| between zero-field blocks from different
   runs. Never from field-on data (AGENTS rule O10).
4. Per field and tube: the difference of means against 3× the floor, and a
   threshold classifier on blocks scored by leaving one run out.
5. The same analysis on P1 computed without chaord (Y1's second block, adapted):
   E_min(direct).
6. Linear response: P1 at 0.005, 0.01 and 0.02 V/Å proportional to E within errors.
7. Every statement other than `dipole_p1` that moves beyond 2× its floor at or
   below the working field.

Done when `reports/research/go_no_go.md` gives E_min per tube for Chaord and for
direct P1, the linear-response check, the list from step 7, and the D32 verdict
with its numbers, by 2026-10-29.

### Y9 · An EC template that is EC (P2, stream S4, half a day)

Reproducer:

```python
import sys, numpy as np
sys.path.insert(0, "src")
from chaord.build.molecules import TEMPLATES
t = TEMPLATES["EC"]
sym, r = t["symbols"], t["rel"]
d = lambda i, j: float(np.linalg.norm(r[i] - r[j]))
C = [i for i, s in enumerate(sym) if s == "C"]
H = [i for i, s in enumerate(sym) if s == "H"]
cc = min(d(i, j) for i in C for j in C if i < j)
hh = min(d(i, j) for i in H for j in H if i < j)
print(f"EC template: ring C-C {cc:.2f} A, closest H-H {hh:.2f} A, all z = 0: {bool(np.allclose(r[:, 2], 0))}")
# prints today:
# EC template: ring C-C 1.43 A, closest H-H 1.25 A, all z = 0: True
```

Use a cited EC structure: ring C–C about 1.52 Å, ring C–O about 1.44 Å,
C(=O)–O about 1.33 Å, C=O about 1.20 Å, CH2 hydrogens out of the ring plane at
H–C–H about 109°. The frozen `lipf6_ec` bench frames were generated with the old
geometry: keep it as `EC/bench-v1` for those frames so they stay byte-identical,
and let new builds use `EC`.

Done when: the reproducer prints a ring C–C of at least 1.50 Å, H–H of at least
1.7 Å and `False`, and the frozen-bench tests still pass.

## 3. Carried from v3, unchanged (details in `open_items_v3.md` section 2)

| Work order | Today on `2990d08` | Stream |
| --- | --- | --- |
| X1 clean thermal slabs grow a fake adatom | 9 of 20 clean Pt(111) slabs print an adsorbate | S1, first |
| X2 lift-side snap | 94 configurations still registered as expected failures | S1 |
| X3 records | README and gate report unchanged since `43d683f` | S2 |
| X4 acceptance draws | `check_a2` still defaults to 2 transforms | S2 |
| X5 histories mean what they say | a quench asking for 190,000 steps runs 20,000 | S1 |
| X6 dense-glass statistics | `cn` at 1.19 σ and `ring_mode 3` still printed | S1 |
| X7 converged references on the cluster | not started; glass rates first | S3, after Y6 inputs |
| X9 leftovers | unchanged | S2 |

## 4. Streams, order and merge rules

| Stream | Work orders | Owns | Starts | Takes about |
| --- | --- | --- | --- | --- |
| S1 Fixes | X1, then X2, X5, X6 | `lift/surface.py`, the snap, `realize/`, `build/amorphous.py`, `glass.yaml` | now | 3–4 days |
| S2 Records | Y1, then X3, X4, X9 | `research/cnt_electrolyte/RESULTS.md` and `README.md`, the top-level `README.md`, `reports/` (except `reports/research/`), `tools/acceptance.py`, `spec/examples/` | now | 2–3 days |
| S3 Cluster runs | Y6, Y7, then X7 | run generators and inputs under `research/cnt_electrolyte/` including `params.py` (no chaord import), manifests, `reports/research/stage_a.md`, `bench/reference/` | inputs now | about 7 working days, mostly compute |
| S4 Research library | Y2 and Y4, then Y3, Y5, Y9 | `lift/__init__.py`, `lift/nanotube.py`, `io/lammps.py`, `cv/field.py`, the dialects, `build/molecules.py` | now | 6–7 days |
| S5 Analysis | Y8 | `research/cnt_electrolyte/analysis/`, `reports/research/go_no_go.md` | after Y2, Y3, Y5 and Y7 | 2 days |

Merge rules (D5 stands): one branch per stream; merge on green CI with the run
URL; one work order per merge, with its red test. S1 merges X1 first, today. S4
merges Y2 and Y4 before Y3. Stage B starts only after the owner has read
`stage_a.md`. Y8 starts only when Y2, Y3, Y5 and Stage B are in.

## 5. Done when (this round)

- [ ] Y1 merged: the pilot is called a pilot, with the orientation numbers.
- [ ] Y2 merged: blind lift; the reproducer prints the same `dipole_p1` line twice.
- [ ] Y3 merged: 20/20 on every D33 case; `lift_frame` composes tube plus liquid.
- [ ] Y4 merged: all four dumps read; ambiguous masses raise.
- [ ] Y5 merged: electrolyte statements on all 84 pilot frames.
- [ ] Y6 report: τ_rot, τ_M, mirrored start, reservoir filling.
- [ ] Y7 runs finished with manifests.
- [ ] Y8 report and the D32 verdict, by 2026-10-29.
- [ ] Y9 merged.
- [ ] v3: X1, X2, X5 merged; X3, X4 merged; the X7 glass-rate report written.

## 6. `docs/reviews/owner_inputs.md` (the owner writes it; agents read it)

```yaml
working_field_V_per_A: [0.01]      # one or more values; write "unknown" if not decided
model: path/to/moltemplate/        # the tube + electrolyte model the MLP paper uses
charge_scaling: 1.0                # as the model states
cluster_job_template: path/to/job.sh
lammps_build: path/to/lmp          # with RIGID, KSPACE, MOLECULE
```