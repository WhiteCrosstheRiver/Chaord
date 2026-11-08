# Program length against physics fidelity: PRE-REGISTRATION

Status: REGISTERED 2026-11-07. Binding from the commit that adds it, made before
any rebuild listed here runs. After that commit this file is not edited; anything
added later is reported as post hoc.

## Question

Does the shortest Chaord program that rebuilds a liquid faithfully get shorter
as the compiler's physics gets better? (Original design claim; see DESIGN.md.)

## Systems (fixed)

The two atomic-fluid A5 reference cases, frame 0 lifted exactly as
`tools/acceptance.py check_a5` lifts it (`lifted/`):

- `lj_liquid` (500 atoms) and `lj_liquid_large` (2048 atoms), LJ T* = 0.72.

Not included, and stated as a limit: `lj_solid_liquid` (slab builder) and
`ka_glass` (amorphous builder) have no restraint path yet; the molecular cases
are refused by the restraint by design.

## Physics levels (fixed)

| Level | Build | Meaning |
| --- | --- | --- |
| P0 | `physics=False` | random sequential packing only |
| P1 | `physics=True`, `md_steps` = 10% of the dialect's `md.relax_steps` | packing + short LJ MD |
| P2 | `physics=True`, dialect default | packing + the full A5 MD prior |

## Program ladder (fixed, nested)

Built from the lifted program by keeping only these statements (asserts and
provenance are removed at every level; values are copied verbatim from the
lifted text):

| Level | Statements |
| --- | --- |
| S0 | system (cell, pbc, `state T`, `conserve atoms`), physics `backend`, liquid `state density` |
| S1 | S0 + `constrain cn <value> +- <tol> cutoff <rc>` from the lifted `assert cn` line |
| S2 | S1 + `constrain gr_peak <r> height <h>` from the lifted `assert gr_peak` line (height tolerance: dialect default) |

Program length = characters of the canonical text (`chaord fmt`) of that level.

Order inside a build (as `build.fluid.build_fluid` does it): packing, then the
physics stage of the level, then the restraints of the program level.

## Measurement (fixed: the A5 protocol, unchanged)

For each case x physics level x program level: rebuild with seeds 7, 13 and 29;
average the three draws' observables; compare with the reference mean over the
case's `avg.ref_frames`; distance and effective floor exactly as `check_a5`
(`chaord.cv.noise.distance`, `_a5_effective_floor`). Ratio = distance / floor
for `gr_rms` and `cn_tv`. **A cell passes iff both ratios are <= 1.5.**

A restraint that refuses (target not reached) is recorded as a failed cell, not
retried with other settings.

## Outcome and decision rule (stated before any data)

- Shortest passing program per case and physics level = the lowest S level that
  passes; "none" if S2 fails.
- **The claim is SUPPORTED** iff, for both cases, the shortest passing length is
  non-increasing from P0 to P1 to P2, and at least one physics level needs a
  strictly longer program than P2.
- Otherwise it is NOT SUPPORTED, and is reported as such.
- The full 2 x 3 x 3 table of ratios is reported either way.

## Known weakness, stated in advance

`gr_rms` and `cn_tv` partly overlap the constrained quantities (S1 restrains mean
coordination, S2 the g(r) peak). So a pass at S1 or S2 is weaker evidence than a
pass at S0. As a held-out check, the bond-angle distribution distance
(`angle_tv`, which no level constrains) is reported for every cell, relative to
the P2/S0 cell; it does not enter the decision rule.

## Compute

gerui cluster via `launch.sh`, one node, cells in parallel processes; no files
kept on the cluster after retrieval. Results: `results.json` (every cell: length,
ratios, angle_tv, refusals, seconds, git commit).
