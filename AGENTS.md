# AGENTS.md — rules for every agent working on Chaord

Read this file, `PLAN.md` and `spec/` before writing code. These rules are binding.

## What we are building
Chaord is a language for atomic systems with a compiler (`chaord build`: program → coordinates)
and a decompiler (`chaord lift`: coordinates → program). A program describes a macrostate
(a family of configurations); a coordinate file is one microstate of it.

## The ten design rules (short form; full text in PLAN.md)
1. A program is a macrostate, not one configuration.
2. Every statement has a kind: build (no keyword), state, constrain, assert, history, conserve.
3. One structure, one text: canonical, declarative, one statement per line, printed by `chaord fmt`.
4. Physics is the compile target: every program names its backend.
5. Dialects fix the rules: thresholds and cut-offs live only in versioned dialect YAML files.
6. Never drop an atom: counts per species and charge are conserved; the unexplained goes to `residual`.
7. One definition per quantity: each CV is measured (lift), restrained (build) and checked (assert) by the same code.
8. Reuse existing notation: Kröger–Vink, Burgers vectors, CSL Σ, Miller, Wood, site names, (n,m), SMILES.
9. The round trip is the correctness test: lift → build → lift must match held-out observables within the noise floor.
10. Deterministic: same input + dialect + version → same text; same program + seed + backend → same coordinates.

## How to work
- Take exactly one work package (WP) from PLAN.md. One branch and one pull request per WP.
- Start by writing the failing tests from the WP's "Done when" column. Then implement.
- Never weaken, skip, xfail or delete a test to get green. If a test seems wrong, stop and report.
- No magic numbers in pass or builder code: read thresholds from the dialect by name.
  CI greps `src/chaord/lift` and `src/chaord/build` for bare float literals; exceptions need a comment `# dialect-exempt: <reason>`.
- A new statement key ships complete: dialect entry, grammar/IR support, build, lift, CV (if statistical),
  tests, reference-manual entry and one example under `spec/examples/` that passes `tools/sketch_check.py`.
- Seed all randomness (`numpy.random.default_rng(seed)`); no global RNG, no hidden global state.
- Heavy tools (OVITO, LAMMPS, PLUMED, MACE) are optional extras. Core must run on numpy, scipy, ASE, Lark, Pydantic, pint.
- Conservation check runs after every lift: atoms per species and total charge in the program must equal the input.
- If the spec is unclear, open a question in the PR instead of guessing. Spec, grammar and dialect-threshold
  changes need human approval.

## Definition of done (every PR)
- New tests written first and passing; full unit + property + golden suites green (< 10 min).
- `python tools/sketch_check.py spec/examples/*.chaord` passes.
- Docs updated (reference entry + example) for any user-visible change.
- PR description lists: WP id, what changed, how it was tested, open questions.

## Gates
- Gate A (end of week 12) and Gate B (week 20) are checked by a verification agent that did not write the code.
  It runs the acceptance suite (A1–A14 in PLAN.md) and writes a one-page report. The human reviewer signs off.

## Useful commands (targets for M0; the prototype works today)
    python tools/sketch_check.py spec/examples/*.chaord      # structural check of example programs
    cd prototype && python make_snapshot.py                  # regenerate the LJ solid–liquid snapshot (~1 min)
    cd prototype && python roundtrip.py                      # decompile -> compile -> decompile, compare (~40 s)
