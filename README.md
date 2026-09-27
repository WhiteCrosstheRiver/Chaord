# Chaord — describe the macrostate, sample the microstate

A language for atomic systems with a compiler (`chaord build`: program →
coordinates) and a decompiler (`chaord lift`: coordinates → program). A
program describes a **macrostate** — a family of configurations; a coordinate
file is one **microstate** of it. Order is written exactly where a system has
it, statistics where it does not: from crystals to gases.

Status: **v1.0 candidate** — M0-M8 complete; acceptance A1-A14 executed by
`tools/acceptance.py` (report in `reports/acceptance.json`).

## Install (from this repository)

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e .[test]
.venv/Scripts/python -m pytest tests -m "not slow"     # fast suite
.venv/Scripts/python -m pytest tests -m "slow"         # statistical round trips
```

Core runs on numpy, scipy, ASE, Lark, Pydantic and PyYAML only; heavy tools
(OVITO, LAMMPS, PLUMED, MACE, icet, RDKit) are optional extras with core
fallbacks (own Wigner-Seitz, SQS-lite, ring statistics, geometric bonds).

## Layout

```
src/chaord/
  lang/        grammar (Lark), parser, Pydantic IR, canonical formatter
  dialects/    versioned YAML: every threshold lives here (CI-enforced)
  cv/          collective variables: measure / restrain / check one definition
  build/       lattices, defects, SQS, packing, interfaces, surfaces, extended
  realize/     LJ MD, protocols (melt/quench/anneal), restraints
  lift/        decompiler passes: crystal, defects, fluid, surface, amorphous...
  check/       static checks, shortest-program controller
  io/          Frame I/O (extxyz, LAMMPS, POSCAR, CIF, npz)
  cli.py       fmt / build / lift / check / diff / roundtrip
  integrations.py  LLM JSON schema, scripting layer, Laya state encoder
bench/         Chaord-Bench generators and ground truth
docs/          reference.md, tutorials.md
prototype/     the original Lennard-Jones demo (reference, not extended)
spec/          grammar sketch and the seven canonical example programs
PLAN.md        the full plan; AGENTS.md the binding rules
```

## The round trip is the correctness test

Lift, build, lift again: held-out observables must match within the noise
between two frames of the same simulation (1.5x, measured — never guessed).
Crystals round trip **byte-identically** under rotation, translation,
re-ordering and re-imaging; fluids and glasses are judged against the floor.

## Milestones

| done | milestone |
| --- | --- |
| M0 | grammar, IR, fmt, dialects, CV registry, CLI, prototype port |
| M1 | 13 crystal prototypes, exact round trip + invariance suite |
| M2 | Kröger-Vink defects, SRO/SQS, planted-defect acceptance |
| M3 | molecules, packing, fluids, noise-floor library, statistical round trip |
| M4 | segmentation, interface objects, Miller slabs, adsorbates |
| M5 | history protocols, restraints, ring/Voronoi statistics, shortest program |
| M6 | Burgers family detection, CSL Sigma bicrystals, reactive census |
| M7 | acceptance runner A1-A14, reference manual, tutorials |
| M8 | LLM schema, scripting layer, Laya state encoder |

See `PLAN.md` for the full design and `AGENTS.md` for the binding rules.
