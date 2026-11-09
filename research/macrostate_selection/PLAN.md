# Macrostate selection: PLAN

Binding once committed. No reanalysis under a changed rule; any new idea is new
work with its own plan.

## Question

The program-length curve (`research/program_length/RESULTS.md`) showed that for
a simple liquid the program's structural statements add nothing: temperature,
density and the potential are enough, and MD does the rest. Does the text ever
carry something MD cannot supply? Test: states that share potential, T and
density, so that physics alone cannot choose among them within the run length.

## Systems

One component, Lennard-Jones (rc 2.5, unshifted), reduced units, NVT at
T = 0.75 and density 0.90, about 2,000 atoms. Four macrostates:

- `liquid`, compressed homogeneous liquid, 2,048 atoms
- `fcc`, homogeneous crystal, 2,048 atoms
- `hcp`, homogeneous crystal, 2,016 atoms
- `coex`, crystal/liquid slab, 4,032 atoms

Each has six independent reference trajectories from LAMMPS (20,000 steps of
equilibration, then 10 frames over 80,000 steps; timestep 0.005).

## Programs

From the last frame of reference trajectories 0, 1, 2 of each state,
`lift_frame(frame, core+lj, T=0.75)`, default mode, no hint about the state:

- **full**: the lifted program, unedited.
- **control**: cell, T, atom count, density 0.90 and `backend lj`, as one liquid
  region. The most physics can be told without saying which macrostate this is.

## Rebuild

Every program is compiled with `physics=False`, then every result gets the same
MD: minimize, then the same equilibration and frames as the references. The MD
does not depend on the program, so any difference between cells comes from the
text.

## Scoring

All scoring is independent of Chaord (`matrix.py`, `cna.py`). Three metrics,
each on the mean over a state's trajectories or over the three rebuilds:
g(r) rms (to r = 4.5, 90 bins), coordination histogram total variation
(cut-off 1.5), and common-neighbour class fractions (fcc, hcp, other; total
variation).

Floor for reference state j and metric m: max(mean, P90) of the distances
between the 15 pairs of its six trajectory means.

A rebuild **matches** reference j if all three distances are at most 1.5 times
j's floors.

## Decision rule

**SUPPORTED** only if all six hold:

1. `no_refusals`: no lift refused, no build refused, no MD crashed. Any refusal
   anywhere makes this false. A refusal never counts as "did not match".
2. `references_valid`: at the last frame of every trajectory, liquid has
   fcc and hcp fractions each below 0.01; fcc has fcc fraction above 0.10 and
   hcp below 0.01; hcp has hcp fraction above 0.10 and fcc below 0.01; coex,
   over 10 slices along z, has at least one slice with fcc fraction at least
   0.10 and at least three slices below 0.01.
3. `diagonal`: full program of s matches reference s, for all four states.
4. `off_diagonal`: full program of s matches no other reference (12 cells).
5. `control_liquid`: the control from every source matches the liquid reference.
6. `control_not_source`: the control from fcc, hcp and coex does not match its
   own source reference.

Otherwise **NOT SUPPORTED**. Every cell is reported either way.

## Seeds

Reference velocity seeds from 1001, build seeds from 2001. Pilot seeds (9001,
9101+, 9201+) are excluded.

## Pilot disclosure

Before this plan was committed, a pilot (seed 9001; four (T, density) points)
chose a state point. A second pilot (seeds 9101/9201) at density 0.97 found two
things, recorded in `docs/records/corrections.md`: the slab crystallized over the
longer run, so density moved to 0.95 and the validity check gained its `coex`
clause; and the compiler silently lost atoms when tiling large hcp cells, now
fixed. A third pilot on the same seeds, at T = 1.0 and density 0.95, gave clean results
for liquid, fcc and hcp but lifted the thin, near-melting crystal slab as a
homogeneous liquid in the program inspected, so the coex diagonal failed. The
state point moved to the middle of the coexistence gap at a lower temperature. A
fourth pilot there missed in the same way. Regenerating its frames showed why: the
slab was too thin to have crystal interior (half-height under 4), the slab arm
refused, and the cascade printed a liquid anyway. That fall-through is fixed
(the lift now refuses, `docs/records/corrections.md`), and the slab box is doubled
along z so the crystal has an interior. The paper reports the thin-slab limit. A
fifth pilot, on the pilot seeds, passed all six checks. No pilot is part of the
result.

## Software

LAMMPS: PyPI wheel 2024.8.29.3.0, serial, sha256 verified. Chaord at the commit
that registers this plan. Compute: gerui `dell-cpu`, node /tmp only.
