# Chaord starter kit

Everything an agent team needs to start the Chaord project (a language, compiler and decompiler for
atomic systems). The plan itself is `PLAN.md` (same content as the shared plan document).

| Path | What it is |
| --- | --- |
| `PLAN.md` | The implementation plan: name, design rules, language, architecture, milestones, tests, acceptance criteria |
| `AGENTS.md` | Binding rules for AI agents; read first |
| `spec/grammar.ebnf` | Grammar sketch (structure only) for Chaord v0.1 |
| `spec/examples/*.chaord` | Seven example programs: solid–liquid (real decompiler output), crystal with defects, solid solution with SRO, amorphous Si, electrolyte, gas, reactive interface |
| `tools/sketch_check.py` | Structural checker for the grammar sketch (no vocabulary checks) |
| `tools/test_sketch_check.py` | Positive and negative tests for the checker (seed tests for WP01) |
| `prototype/` | Working Lennard-Jones demo: MD snapshot generator, decompiler, compiler, round-trip test |

## Quick start (Python 3.10+, numpy, scipy)

    python tools/sketch_check.py spec/examples/*.chaord
    python tools/test_sketch_check.py
    cd prototype && python roundtrip.py

## What the prototype already shows
On a 2,301-atom Lennard-Jones solid–liquid snapshot with three planted vacancies:
- The decompiled program is 607 characters versus 54,411 for the coordinate file (1.1%).
- It finds the vacancies as one split divacancy plus one vacancy: net 3, exactly as planted.
- Rebuilding from the program with MD reproduces held-out liquid structure close to the frame-to-frame
  noise (g(r) RMS distance 0.15 vs 0.12; bond-angle distance 0.021 vs 0.015); without MD it is 4–6× worse.
- Strain was not preserved without a restraint, which is why `constrain` exists.
