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
- Read `docs/reviews/open_items.md` (and `open_items_v2.md` when it exists)
  before starting ANY work; it overrides every older plan (moved to the top
  by Review 8 D-new-rule-1, 2026-10-04).
- Heavy tools (OVITO, LAMMPS, PLUMED, MACE) are optional extras. Core must run on numpy, scipy, ASE, Lark, Pydantic, pint.
- Conservation check runs after every lift: atoms per species and total charge in the program must equal the input.
- If the spec is unclear, open a question in the PR instead of guessing. Spec, grammar and dialect-threshold
  changes need human approval.
- Reference data for disordered systems (liquids, glasses, interfaces, solutions)
  comes from an independent MD engine with a published potential. Generator code
  for reference data MUST NOT import chaord. Every case records engine, potential
  (name, citation, parameters), protocol and seed, and passes physical sanity
  checks: no pair closer than 0.8 sigma (or the potential's hard core), density
  within 2% of target, first g(r) peak where the literature puts it.
- A claim in README or a report needs a test that could have failed AND a run on
  a clean machine. "Passes on my machine" is not evidence; self-run acceptance
  is bookkeeping, not verification.

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
- A root cause named in a report or PR description must come with a test that
  reproduces the failure and fails before the fix (post Review 2, 2026-09-29).
- One stream, one PR. A stream's changes land behind their own pull request
  (or, while the repository has a single developer, an equivalent per-stream
  commit series) with the acceptance job run on that stream's ref before
  merge; mixed-stream mega-commits are not acceptable (post Review 3,
  2026-09-30: commit 615f2c0 mixed four streams and let one stream's number
  reach a report while the integrated run failed).
- Reports quote numbers only from clean-runner artifacts (the GitHub Actions
  acceptance run), never from an author's local run; a local number may be
  shown only as clearly-labelled bookkeeping (post Review 3, 2026-09-30).
- Every criterion keeps a POWER MUTATION: the single most likely real error
  for that criterion, injected on purpose, whose run must FAIL. A criterion
  whose power mutation passes has no evidence value (post Review 3,
  2026-09-30; the registry lives in tests/acceptance/test_mutations.py and
  tests/adversarial/).
- A criterion with a random element is judged over at least 20 draws; one
  draw is not evidence (post Reviews 4-7).
- Every test input, planted or synthetic, passes the reference-data sanity
  checks (post Reviews 4-7).
- A floor or tolerance never includes a quantity measured on the system
  under test (post Reviews 4-7, O10).
- When a clean run fails, report the failure with its power numbers before
  changing how a criterion is measured (post Reviews 4-7).
- A threshold change is tested in every unit system its dialect serves (A and
  sigma), on crystals and on disordered frames (Review 8, 2026-10-04).
- Fail closed. A missing threshold or a failed check raises. A lift arm falls
  through to the next arm only on an explicit "not this phase" verdict. A lift
  that cannot describe a frame refuses with an error; it never prints a program
  it cannot stand behind. A value the lift did not measure and was not given
  (temperature, model) is printed as assumed (Review 8, 2026-10-04).
- A sanity target never comes from the frames it checks (Review 8, 2026-10-04).
