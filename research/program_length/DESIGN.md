# Program length against physics fidelity: design status

Status 2026-11-07: **not pre-registered, not run.** This note records why.

## The claim (original design)

The program length needed for a faithful round trip shrinks as the physics in the
compiler improves: random packing, then a classical potential, then an MLP.

## What the lifted programs show

`lifted/` holds frame 0 of each A5 reference case, lifted exactly as
`tools/acceptance.py check_a5` does (provenance temperature, case dialect). They
were produced on the gerui cluster by `lift_cases.py` through `launch.sh` and
contain no outcome data. For the LJ liquid, the lines the compiler acts on are
cell, atom count, backend, `state T` and `state density`; the `assert` lines are
checks and do not steer the build.

## Why the curve cannot be measured yet

1. **No restraint path for fluids.** `constrain` is implemented only for
   crystal short-range order (`src/chaord/build/crystal.py`). With physics off,
   no program in the current language can steer a liquid towards its g(r) or
   coordination, however long it is, so physics-off has no finite point on the
   curve. A5's `physics_off` mutation already shows that such rebuilds fail.
2. **No MLP tier.** No machine-learned potential backend is available offline on
   either cluster.
3. With physics on, the minimal program is already the five lines above, so a
   curve today would have one trivial point per physics level.

## What would make it a real result

- **Fluid restraints.** Implement `constrain cn` and `constrain gr` for fluid
  regions in the build, for example reverse Monte Carlo on the packing prior,
  sharing the CV definitions lift already uses (design rule 7). This is a new
  statement capability: grammar, dialect entry, tests and an example, so it
  needs owner approval (AGENTS.md).
- **Then the experiment.** For each case and physics level (packing only, packing
  plus restraints, short MD, full MD), find the shortest program in a fixed
  nested ladder whose rebuild passes the A5 gate, using A5's own protocol: seeds
  7, 13 and 29, frame-averaged reference, 1.5 times the measured floor.
  Pre-register the ladder and the gate before any rebuild.
- **Compute.** The gerui cluster can run it: Chaord is installed privately
  there (`launch.sh`, node-local `/tmp`, results streamed back, nothing left on
  the node).

## gerui cluster notes

- Compute nodes cannot see `/share/home`; only `/apps` (read-only) and node-local
  `/tmp`. `launch.sh` handles this with `sbcast`.
- Python: `/apps/softwares/anaconda3` (3.11) plus private wheels in
  `~/chaord_work/pylib` (ase, lark, spglib, pydantic 2, scipy 1.15.3).
- The account runs other people's jobs; leave them alone and keep to a modest
  share of nodes.
