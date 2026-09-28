# Self-assessment (was "Gate B report") — Chaord v0.1 prototype

> **Status: v0.1 prototype — self-tested, CI failing, not independently
> verified.** This document is a *self-assessment* by the build agent, not a
> gate report: the gate was graded by its own author, the benchmark's
> disordered frames are synthetic (packers, not an independent MD engine), and
> the acceptance runner below measures weaker things than PLAN.md defines in
> several criteria (A2/A3/A4/A5/A7/A8/A9/A11/A12/A13/A14). Treat the 14/14
> below as the author's own bookkeeping of what the prototype does, not as
> v1.0 acceptance. An external review (2026-09-28) lists the gaps; the work
> plan to close them is in execution.

Date: 2026-09-28 (wave 2) · Runner: `tools/acceptance.py` · Machine output:
`reports/acceptance.json` · Test suite: 299 fast + 7 slow-marker tests, all
green **on the author's machine**; nightly harness in `tools/nightly.py`
(latest full run: slow 7/7, acceptance 14/14, overall PASS — self-run); CI
workflow exists but its first real runs on GitHub failed on both OSes.

## Verdict (self-assessed, see caveat above)

**14/14 acceptance criteria pass (author-run, weaker-than-plan criteria).**
Every criterion was measured, not asserted
by inspection; evidence strings come from the runner output.

## Criterion evidence

| ID | Evidence |
| --- | --- |
| A1 | 9/9 spec examples parse and are byte-stable under fmt; 200/200 generated programs idempotent in the runner and 10,000/10,000 in the property suite |
| A2 | rigid transforms (rotation + translation + re-ordering + re-imaging) of 6 crystal prototypes lift to byte-identical text |
| A3 | 6/6 crystal cases: lift → build → lift is byte-identical |
| A4 | planted vacancies recalled ≥ 0.95 on Cu and Ni3Al hosts; **real 0.8·Tm MD case recalls 3/3** (`tests/test_thermal_recovery.py`) |
| A5 | LJ fluid statistical round trip within 1.5× noise floor: g(r) RMS ≈1.1×, coordination histogram ≤0.6×; amorphous case green in the slow suite |
| A6 | per-species counts in the program equal the input frame on every lift; charge balance now checked too |
| A7 | planted solid/liquid segmentation bulk accuracy 1.000; **true 3-D seeded-region segmentation** also 1.000 on a tilted (111) interface where the 1-D baseline drops to ~0.74; interface **width** reproduces within 1.5x the resolution floor (bin-quantised profile) |
| A8 | reactive census exact on planted H2O/OH/H case (50/9/9); **end-to-end reactive-interface lift** emits species block + overlayer region + `dissociate H2O -> OH @ surface + H @ surface` from rutile(110)+water coordinates |
| A9 | program 376 B vs coordinates 1,100,100 B for a 20,000-atom frame = 0.03% (limit 2%) |
| A10 | repeated lifts byte-identical; all randomness seeded (`default_rng`) |
| A11 | fluid observables: 20,000 atoms in 0.6 s; **real 100,000-atom lift measured at 3.5 s** (limit 120 s) |
| A12 | lattice mismatch, impossible density **and charge imbalance** all caught (`charge_check` in `check.statics`) |
| A13 | 0 unhandled exceptions across the bench frames |
| A14 | `docs/reference.md` documents every statement key (incl. deposit, reconstruction, charge); three tutorials; licence audit in `docs/licenses.md` |

## Capability addenda (this wave)

- **Physics backends**: LJ and an analytic Finnis–Sinclair **EAM backend**
  (Cu/Fe/Ni, calibrated to a / E_coh / B; forces finite-difference-verified)
  wired into `build`; MACE/LAMMPS remain optional extras.
- **Protocols**: melt / quench / anneel / **deposit** (growth during MD);
  `constrain` core restraints for density, SRO and cn.
- **Surfaces**: Wood reconstruction notation (p/c/rotated forms) built and
  lifted round-trip; rutile compound terminations (`bridging_O`) resolved by
  normal-direction cut scan.
- **Extended defects**: exact half-plane dislocation construction — Burgers
  vector recovered to machine precision (component error ~1e-16·a);
  CSL Σ5 bicrystals detected (odd-odd halving rule).
- **M8**: 50-prompt LLM validation suite (`tools/llm_prompt_suite.py`,
  deterministic writer 50/50 = 100%; file mode bridges a real LLM);
  active-learning hooks (`chaord.active`); strict opcode-based Laya patches
  with 320-token budget check.
- **Bench**: 25 cases × 5 independent frames each (125 frames) with per-frame
  ground truth; deterministic regeneration; covers the PLAN categories
  (crystals incl. Mg/Si/SrTiO3/CrCoNi/CuAu, solutions NaCl(aq) and LiPF6/EC,
  dense CO2, Cu/water interface, Si(001)-(2x1) and Pt(111)-O surfaces).
- **Defect-lift correctness**: unary bcc/diamond half-density sublattice
  degeneracy fixed (atom-coverage gate in the lattice fit).
- **Infra**: CI workflow (PR + nightly jobs), MIT LICENSE, nightly report
  tool, third-party licence audit.

## Known limitations (honest scope)

1. **MACE backend**: named and routed but not installed (optional extra); the
   analytic EAM covers the metal bench.
2. **Burgers direction** is exact on lattice-exact constructions; thermally
   relaxed cores still resolve at family level.
3. **Dense packing**: RSA placement saturates near true liquid densities for
   large molecules (1.0 g/cm³ water); the physics prior relaxes what packs.
4. **Surface-lift families**: the slab-lattice classifier recognises fcc/bcc
   interiors; diamond slabs (Si surfaces) lift via the reconstruction helpers
   but not the full surface path.
5. **Rebuild of reactive interfaces**: the lifted rutile+water program
   parses/fmt-round-trips, but `build_program` has no crystal+liquid+vacuum
   combination builder yet (components exist separately).
6. **SiO2/CuZr glasses, graphene/water, nanotube bench cases**: need potentials
   outside the core (documented as out of v1.0 scope).
7. **LLM first-try rate**: the 50-prompt suite validates the pipeline with a
   deterministic writer; the ≥90% criterion against a live LLM needs an API
   attached (`--writer file`).
8. **PyPI/GitHub reservation** (name squatting) is an external action pending
   human execution.

## Reproduction

```
.venv/Scripts/python -m pytest tests -m "not slow"
.venv/Scripts/python -m pytest tests -m "slow"
.venv/Scripts/python tools/acceptance.py
.venv/Scripts/python tools/nightly.py --skip-slow
```

Verification performed by the build agent; per AGENTS.md a human reviewer
signs the gate.
