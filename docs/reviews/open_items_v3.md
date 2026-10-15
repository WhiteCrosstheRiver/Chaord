# Open items and decisions, v3 (external Review 9, 2026-10-14)

**Read this before any work.** It replaces v2 (Review 8), whose work orders
W1–W12 are done and whose Gate A′ is accepted. The owner has again delegated
decision authority to the external reviewer, so section 1 is binding: implement
it, do not reopen it. Then work section 2 in the stream order of section 3.

The gate work is over. Two kinds of work are left: three regressions and
records fixes that take a few days (S1, S2), and the work that decides whether
Chaord matters for the owner's research. That is converged reference data on
the owner's cluster (S3), and the first lift of the owner's own carbon-nanotube
electrolyte frames (S4, the main stream). Every work order lists its steps and a
done-when test. X1, X4, X5, X6 and X8 carry a reproducer that ran on `43d683f`,
with its result. Write the done-when test first, see it fail, then fix
(AGENTS.md).

Run the snippets from the repository root, in the project environment, with the
same common header as v2:

```python
import sys, numpy as np
sys.path[:0] = ["src", "tools"]
import acceptance as acc
from chaord.dialects import load_dialect
from chaord.io.frames import Frame, read_frame
from chaord.lift import lift_frame
from chaord.build import build_program

def load_ref(path):
    """A bench/reference frame (npz with r, L, symbols)."""
    z = np.load(path, allow_pickle=True)
    L = np.asarray(z["L"], float)
    return Frame(pos=np.asarray(z["r"], float), cell=np.diag(L) if L.ndim == 1 else L,
                 symbols=[str(s) for s in z["symbols"]], pbc=(True, True, True))
```

## 0. What the reviewer re-checked on `43d683f`

| Item | Result |
| --- | --- |
| A2 / O1 regression | 900 fresh rigid transforms (9 cases × frames 0–4 × 20): 0 text changes, 0 build failures |
| W1 classification | 200/200 jittered LJ crystals (fcc, bcc, sc, diamond, hcp; ρ* 0.85 and 1.1; 0.10 d_NN jitter; 20 draws each) and the 9 Å bench crystals lift as crystals under `core + glass`; the Kob–Andersen, monatomic and RSA glasses lift as amorphous; argon, N2, CO2 and water pass `mode="fluid"` on 5/5 frames each |
| W4 Cu solid–liquid | all 10 frames: `Cu 832`, `lattice fcc`, a 3.615 Å, 0 defect lines, T marked assumed, the program builds |
| W8 dense CO2 | 60 CO2 at 1.00 g/cm³ build on 20 of 20 fresh seeds; every contact ≥ 0.766 × the Bondi sum |
| W9 sites | mixed Pt(111) top/bridge/fcc/hcp, 20 draws at random translations: 157/157 O correct; Pt(100) 4-fold hollows read hollow |
| Kob–Andersen reference | max empty sphere 0.80–0.89 σ, no volume more than 1 σ from an atom; pressure −0.23 to −0.01 (the provenance says −0.23 to −0.14) |
| Gate A′ | accepted: clean-runner 14/14 on `ba6b594`, only reports changed after it |

## 1. Decisions (binding, 2026-10-14)

| ID | Decision | Implement |
| --- | --- | --- |
| D17 | **W5 lift-side snap: approved.** One function, `box_lattice_constant(cell, reps, prototype)`, returns the lattice constant(s) a stated cell implies (cubic: the mean of L_i / n_i; hcp: a from x and y, c from z). Every lift arm (spglib, defect scan, hcp) snaps its fitted constant to it when the cell is within `strain_cell_tolerance` of an integer tiling, and the builder uses the same function. | X2 |
| D18 | Kob–Andersen sanity bounds: **approved.** Pressure > −0.5 (mild tension at T* = 0.1 is harmless while no point is more than 1 σ from an atom) and per-quench energy flatness ≤ 0.015 (3σ of the thermal fluctuation at N = 2000). Correct the recorded pressure range to the full measured range. | X3 |
| D19 | **History semantics are physical and fail closed.** (a) A quench rate is in temperature per time unit (T*/τ in LJ units, K/ps in metal and molecular units), never per MD step; programs keep their meaning through a one-time conversion of the printed rates. (b) The builder never caps or clamps a stated history (`quench_max_steps`, the `max(rate, 1e-6)` guard): it raises a `ChaordError` that names the requested and the allowed step count. (c) Temperature ramps run with a thermostat that follows the ramp (Langevin γ ≥ 10/τ during melt and quench, or a velocity-rescaling thermostat with τ_T ≤ 0.1 τ), and provenance records the kinetic temperature at the end of every stage. (d) One protocol runner serves every engine; `_run_mixture_history` and `run_protocol` merge. | X5 |
| D20 | For amorphous regions with a named model, A5 adds the **inherent-structure energy per atom** (energy after a local minimisation) as an observable, with floors from cross-quench pairs. | X7 |
| D21 | **Acceptance criteria obey rule 2 (≥ 20 draws).** A2 uses 20 rigid transforms per case. A8's site half runs 20 draws × 20 translations and adds a false-positive check on clean slabs (X1). | X4 |
| D22 | **Records are current-state documents.** README status ≤ 10 lines. `reports/gate_a_prime.md` is rewritten from scratch (≤ 3 pages), with its criteria table generated from `acceptance.json`. Superseded sections move to `reports/history/`. | X3 |
| D23 | Reference data may be generated with LAMMPS or GPUMD on the owner's cluster. The no-chaord-import rule stays, and provenance records the engine version and the full input script. | X7 |
| D24 | An optional LAMMPS physics backend for the builder (Python module or subprocess) for large N and slow protocols. The numpy engine stays the default and the CI engine. | X7 |
| D25 | **Research step approved, time-boxed:** R1–R3 of X8 within 10 working days. Go/no-go then: if programs lifted from the owner's frames do not separate field on from field off beyond their noise floors, Chaord stays off the critical path of the owner's manuscript. | X8 |
| D26 | Spec additions for X8 approved: a `cylinder` region geometry; the carbon dialect's `nanotube` region with `chirality (n,m)`; molecule templates PC, EMC, EC, PF6⁻, Li⁺; a `state field` statement (stated by the user, never inferred); dipole-orientation and radial-profile CVs. | X8 |
| D27 | Dense-glass statistics: per-pair coordination cutoffs at each partial g(r)'s first minimum; `ring_mode` and `angle_mean` are printed only for network glasses (mean coordination ≤ 6). | X6 |

## 2. Work orders

### X1 · Clean thermal slabs grow a fake adatom (P1, regression from W10)

A clean Pt slab with room-temperature motion lifts with an `adsorb Pt` line in
about half of all draws (Pt(111): 7/20 at 0.05 Å, 9/20 at 0.08 Å, 12/20 at
0.10 Å; Pt(100): 9/20, 7/20, 12/20). Cause: W10 made `_aligned_z` take the
rotation point from the **rounded** z of the slab's lowest atom
(`zs = np.sort(quantize_coords(z[idx]))`, then `rot = zs[...]`). When rounding
goes up, that atom ends at z − rot < 0, wraps to the top of the box and becomes
an adatom (site top on ABCA (111), hollow on ABAB (100)). Every A8 frame carries
a foreign adsorbate, so no criterion caught it.

```python
import re
surf = load_dialect(("core", "metal", "surface"))
a = 3.92; ann = a / np.sqrt(2); d111 = a / np.sqrt(3)
nx, nrow, nl = 6, 10, 4
Lx, Ly = nx * ann, (nrow // 2) * ann * np.sqrt(3); step = ann * np.sqrt(3) / 6
P = []
for k in range(nl):
    r = k % 3
    while r * step < Ly - 1e-9:
        off = ann / 2 if r % 2 else 0.0
        P += [[(i * ann + off) % Lx, (r * step) % Ly, k * d111] for i in range(nx)]
        r += 3
P = np.array(P); L = np.array([Lx, Ly, 24.0])
bad = 0
for d in range(20):
    pos = P + np.random.default_rng(3000 + d).normal(scale=0.08, size=P.shape)
    pos[:, 2] += 3.0
    text = acc.format_program_text(lift_frame(
        Frame(pos=np.mod(pos, L), cell=np.diag(L), symbols=["Pt"] * len(P), pbc=(True,) * 3), surf))
    bad += bool(re.search(r"\badsorb\b", text))
print(bad, "of 20 clean Pt(111) slabs print an adsorbate")   # today 9 of 20; done: 0
```

Steps: in `_aligned_z` sort with the quantized keys but read the values from
the unrounded array (`order = np.argsort(quantize_coords(z[idx]), kind="stable")`,
`zs = z[idx][order]`). Audit every other `quantize_coords` call site (`regions.py`,
`extended.py`, `defects.py`, `pipeline.py`): quantized values may only order
things, never be used as coordinates. Add the reproducer, plus the same on
Pt(100) at 0.05, 0.08 and 0.10 Å, as tests.

Done when: 0 of 20 at every amplitude on both surfaces; A8 and the surface
tests unchanged.

### X2 · The lift-side snap (P1, closes W5)

Implement D17. Done when the 94 strict xfails in `tests/test_strained_boxes.py`
are gone from FOILED and pass (216/216), and 9 cases × 20 random anisotropic
strains (each axis uniform in ±1.5%) give identical lift → build → lift text.

### X3 · Records (P1)

The README still says Kob–Andersen is "in progress", strained boxes are open and
"W1–W6 gate Gate A′". It points to the 2026-09-30 verifier report. The gate
report contradicts itself:
- Its Verdict still calls O9 "BLOCKED".
- The Reviews 4–7 scorecard appears twice, once with the old O2, O9 and O11 lines.
- The five-item "Decisions pending" list survives below the D1–D16 table.
- The acceptance table still says A2 "18/18", A7 "worst 0.999", A9 "1 measured
  system", A11 "3.7 s" and A14 "35/35 keys covered".
- The per-case A5 table still shows the retired monatomic glass.
- Checklist items 3, 5 and 7 cite September facts.

Steps: implement D22 and D18. Generate the criteria table from
`reports/acceptance.json` and `acceptance_details.md` with a small script, so it
cannot go stale again.

Done when: one current statement per fact; `git grep -n "BLOCKED\|in progress"` in
README and the gate report finds only live items.

### X4 · Acceptance draws (P1)

`check_a2(n_transforms=2)` judges canonical invariance on 2 draws per case,
against AGENTS rule 2. A8's site half judges one planted frame. A11 times only
the fluid arm.

Steps: implement D21; A11 reports each lift arm (crystal, defects, surface,
slab, fluid, amorphous) at 10,000 and 100,000 atoms.

Done when: the criteria's evidence strings state 20 draws (A2), 20 × 20 draws
plus 0 false adsorbates on clean slabs (A8), and a per-arm timing table (A11).

### X5 · A history must mean what it says (P1)

Three gaps between the stated history and the one realised:

1. **Silent cap.** A quench longer than 20,000 steps is cut to 20,000, with no
   error and no note.

   ```python
   import chaord.realize.protocols as prot
   counted = []
   def counting_run_md(r, v, L, nsteps, *args, **kwargs):
       counted.append(int(nsteps))
       return r, v
   prot.run_md = counting_run_md                     # count steps instead of integrating
   f = Frame(pos=np.random.default_rng(0).uniform(0, 4, (32, 3)), cell=np.diag([4.0] * 3),
             symbols=["X"] * 32, pbc=(True, True, True))
   prot.run_protocol(f, [("melt", 2.0, 10), ("quench", 0.1, 0.1, 1e-5)],
                     load_dialect(("core", "glass")), np.random.default_rng(0))
   print(sum(counted[1:]), "quench steps run; the program asked for", round(1.9 / 1e-5))
   # today: 20000 quench steps run; the program asked for 190000. Done: ChaordError.
   ```

2. **Thermostat lag.** With Langevin γ = 0.5/τ the system cannot follow the
   default ramp. At the end of the default Kob–Andersen quench (bath at 0.1,
   0.095 T*/τ), the reviewer measured a kinetic temperature of 0.280. It was
   0.254 after 100 anneal steps, 0.151 after 500, 0.104 after 2,000 and 0.100
   after 4,000 (N = 500, the builder's own `LJMixture` and `run_md`). Your
   anneal-depth result (×6.9 on a 100-step anneal) measures this lag, not
   history. The "anneal is the binding knob" note in `glass.yaml` describes the
   same lag.
3. **Rate per step.** `at 0.000475` is per MD step, so a program's meaning
   depends on `relax_dt`.

Steps: implement D19 (a)–(d). Re-derive the default history (D7's values) after
the thermostat change, because the anneal no longer has to absorb the lag.

Done when: the cap reproducer raises; the kinetic temperature at the end of
every default-protocol stage is within 10% of the stated one (measured in a
test at N = 500); printed rates are per τ; one protocol runner.

### X6 · Dense-glass statistics (P2)

The lifted Kob–Andersen program prints statistics with weak meaning:
- `assert cn 10.52 +- 0.30 cutoff 1.19` cuts through the A–A first peak (at 1.06;
  its first minimum is near 1.4).
- `assert ring_mode 3` and `assert angle_mean 96.8` are network-glass
  statistics, true for any dense packing.
- `gr_peak 1.69 … pair B-B` is the global maximum of a double-peaked, noisy
  function, not its first peak. Across the 15 reference frames it jumps
  between 1.35 and 1.73 σ. The coordination asserts move by only 0.02–0.16.

```python
f = load_ref("bench/reference/ka_glass/frame_0.npz")
text = acc.format_program_text(lift_frame(f, load_dialect(("core", "glass", "lj_mixtures"))))
print([l.strip() for l in text.splitlines() if "ring_mode" in l or "cutoff" in l or "B-B" in l])
```

Steps: implement D27; first-peak search windows per pair.

Done when: the asserts use per-pair first-minimum cutoffs; dense glasses print
no ring or angle asserts; every printed assert varies across the 15 reference
frames by less than its printed tolerance.

### X7 · Converged references on the owner's cluster (P1 stream, starts now)

All of this follows D23 and D24; the generators stay chaord-free.

1. **Kob–Andersen history.** 10 independent quenches at each of three rates
   (0.019, 0.0019 and 0.00019 T*/τ; N = 2000; ρ = 1.2; same melt and
   anneal), in LAMMPS (`pair_style lj/cut 2.5`, per-pair coefficients,
   `pair_modify shift yes`) or GPUMD. Store 5 frames per quench. Add
   inherent-structure energy (D20). Today's 3 quenches differ by 0.017 per atom
   in inherent-structure energy (−7.615, −7.611, −7.598). That is comparable to
   the energy shift `reports/ka_glass_quench_resolution.json` measured for 5–50×
   faster quenches (+0.015 to +0.039 per atom). Report the energy
   and structure difference per decade of rate with confidence intervals, and
   whether A5 can tell the rates apart.
2. **Liquid floors.** At least 30 decorrelated frame pairs per liquid case (two
   or three independent trajectories), with bootstrap 95% intervals on every
   floor. Today each floor rests on 10–12 pairs, so its P90 is in effect the
   maximum.
3. **Temperature resolution.** The detection rate of a temperature change of
   ΔT = 5, 10 and 20% at N = 2048 and 8192, from ≥ 20 rebuild draws per point.
   This needs the LAMMPS backend (D24). Today: 20% is detected, 10% is not.
4. **Defect realism (A4).** MD-relaxed defects in Cu with the FBD EAM
   potential: a vacancy, a ⟨100⟩ split interstitial and a Frenkel pair at 300 K
   and 0.8 Tm, 20 frames each. Report precision and recall; today's 1.000
   comes from unrelaxed planted defects.

Done when: each item has a report under `reports/convergence/` with the numbers,
intervals and the input scripts.

### X8 · First research use: the owner's nanotube electrolyte (P1 main stream, starts now)

Today a bare (10,10) nanotube does not lift:

```python
d, n, M = 1.42, 10, 10
R = n * 3 * d / (2 * np.pi); L = np.array([30.0, 30.0, M * np.sqrt(3) * d])
pts = []
for k in range(2 * M):
    for j in range(n):
        for b in ([0.0, d] if k % 2 == 0 else [1.5 * d, 2.5 * d]):
            phi = (3 * d * j + b) / R
            pts.append([15 + R * np.cos(phi), 15 + R * np.sin(phi), k * np.sqrt(3) / 2 * d])
tube = Frame(pos=np.array(pts), cell=np.diag(L), symbols=["C"] * len(pts), pbc=(True,) * 3)
try:
    print(acc.format_program_text(lift_frame(tube, load_dialect(("core", "molecular")))))
except Exception as e:
    print(type(e).__name__, e)
# today: ChaordError extended bonded component (400 C atoms): not a molecular fluid; ...
# done: a nanotube region with chirality (10,10), radius 6.78 A and the length
```

The `carbon` dialect declares `nanotube`, `chirality` and `stacking`, but no
code implements them.

- **R1 Read the owner's files (1–2 days).**
  - Add `chaord.io.lammps`: read a LAMMPS data file. Map types to elements from
    `Masses` (the nearest standard mass within 0.1 amu). If a mass is ambiguous
    (for example united-atom CH2, 14.03, against N, 14.01), raise and ask for an
    explicit `type_map`. Read molecule IDs and bonds, so molecules come from the
    topology, not from distances.
  - Read dumps (`id type x y z`, scaled and unwrapped variants, triclinic
    bounds) against that data file, and keep molecules whole across the
    boundary.
  - Done when a toy data file plus a two-frame dump, with a molecule crossing the
    box edge, reads correctly, and an ambiguous mass raises.
- **R2 Describe the system (4–5 days).**
  - The nanotube region (D26): detect the 3-coordinated cylindrical carbon
    sheet, fit the axis, radius and (n, m), and state the length or periodicity;
    the builder generates it. Done when synthetic (10,10), (12,0) and (8,4) tubes
    lift to the right (n, m) and radius ±0.05 Å, and build → lift is identical.
  - The `cylinder` geometry, for regions inside and outside the tube.
  - PC, EMC, EC, PF6⁻ and Li⁺ templates, with geometry taken from the owner's
    Moltemplate files so the census matches his topology.
  - CVs: the radial density profile per species and molecule, and the dipole
    order P1 = ⟨cos θ⟩ and P2 = ⟨(3cos²θ − 1)/2⟩ per radial shell, relative to
    the field direction or the tube axis.
  - The `state field` statement, stated by the user and printed as assumed when
    not given.
  - Done when the tube with 30 PC molecules inside lifts with `molecules PC 30` in
    a cylinder region, and conservation is exact.
- **R3 First result (2 days).**
  - Use the owner's runs with the field off and on (same tube and loading), at
    least 20 decorrelated frames each, and two tube diameters.
  - Lift every frame and compute each statement's noise floor from frame pairs
    within one condition.
  - Done when at least one statement separates field on from off beyond 3× its
    floor, and a classifier that reads only the statements reaches ≥ 95%
    accuracy on held-out trajectories. Report the numbers either way; this is
    the go/no-go of D25.
- **R4 Laya loop (after R3 passes).** Choose frames for DFT labelling by their
  statements (statement-space novelty, or the MLP's uncertainty ranked by
  statements) and compare force error against the number of labels with random
  selection. Use 3 seeds each, on the owner's DFT and MLP pipeline.

### X9 · Leftovers (P2)

- The `RuntimeWarning`s at `lift/slab.py:78` (mean of an empty slice) on
  `solutions/lipf6_ec` and the Cu slab: make it an explicit check.
- A14: add `spec/examples` for the 9 statement keys with none (atom, chirality,
  dialects, lift_version, nanotube, note, reconstruction, source, stacking);
  `chirality` and `nanotube` come with X8.

## 3. Streams, order and merge rules

| Stream | Work orders | Owns | Starts | Takes about |
| --- | --- | --- | --- | --- |
| S1 Fixes | X1, then X2, then X5, then X6 | `lift/surface.py`, the lift arms and builder (snap), `realize/`, `build/amorphous.py`, `glass.yaml` | now | 3–4 days |
| S2 Records and acceptance | X3, X4, X9 | `README.md`, `reports/`, `tools/acceptance.py`, `spec/examples/` | now | 2 days |
| S3 Convergence | X7 | `bench/reference/` generators (LAMMPS/GPUMD inputs), `reports/convergence/`; the LAMMPS backend only after X5 merges (both touch the protocol runner) | now, on the cluster | 1–2 weeks of wall time, mostly compute |
| S4 Research | X8 | `io/lammps.py`, `carbon.yaml`, a nanotube lift and build, `cylinder` geometry, molecule templates, CVs | now | 10 working days to the go/no-go |

Merge rules (D5 stands): one branch per stream, merge only on green CI with the run
URL, one work order per merge. S1 merges X1 first, today. S2's X3 merges before
anyone cites the gate report again. S3 and S4 merge piece by piece. Each S4 merge
adds its spec example (A14) and its red test.

## 4. Done when (this round)

- [ ] X1, X2, X5 merged; clean slabs 0/20; strain identity 216/216; histories fail closed.
- [ ] X3, X4 merged; README and gate report current; A2 and A8 at 20 draws.
- [ ] X7 reports for the Kob–Andersen rates and the liquid floors at least.
- [ ] X8 R1–R3 done and the go/no-go of D25 recorded with numbers.