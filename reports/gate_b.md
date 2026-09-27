# Gate B verification report — Chaord v1.0

Date: 2026-09-28 (wave 2) · Runner: `tools/acceptance.py` · Machine output:
`reports/acceptance.json` · Test suite: 283 fast + 10 slow-marker tests, all
green; nightly harness in `tools/nightly.py`; CI workflow in
`.github/workflows/ci.yml`.

## Verdict

**14/14 acceptance criteria pass.** Every criterion was measured, not asserted
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
| A7 | planted solid/liquid segmentation bulk accuracy 1.000; **true 3-D seeded-region segmentation** also 1.000 on a tilted (111) interface where the 1-D baseline drops to ~0.74 |
| A8 | reactive census exact on planted H2O/OH/H case (50/9/9) |
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
- **Bench**: 18 cases × 5 independent frames each (90 frames) with per-frame
  ground truth; deterministic regeneration.
- **Infra**: CI workflow (PR + nightly jobs), MIT LICENSE, nightly report
  tool, third-party licence audit.

## Known limitations (honest scope)

1. **MACE backend**: named and routed but not installed (optional extra); the
   analytic EAM covers the metal bench.
2. **Burgers direction** is exact on lattice-exact constructions; thermally
   relaxed cores still resolve at family level.
3. **Dense packing**: RSA placement saturates near true liquid densities for
   large molecules (1.0 g/cm³ water); the physics prior relaxes what packs.
4. **LLM first-try rate**: the 50-prompt suite validates the pipeline with a
   deterministic writer; the ≥90% criterion against a live LLM needs an API
   attached (`--writer file`).
5. **PyPI/GitHub reservation** (name squatting) is an external action pending
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
