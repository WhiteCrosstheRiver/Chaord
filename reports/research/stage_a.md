# Stage A report — relaxation times, mirrored starts, filling

Preliminary (2026-10-19): all 32 TUBE jobs complete (8 zero-field, 8
switch-on 0.01 V/Å, 8 switch-on 0.2 V/Å, 8 switch-off; every job
sha256-manifested under `research/cnt_electrolyte/stageA/manifests/`); the
2 FILLING runs failed with a physical instability (below) and are being
repaired. Every number regenerates from `stageA_analyze.py`
(`stageA/analysis/per_job.json`, `summary.json`).

## τ_M — the collective orientation relaxation (the Y7 gate quantity)

Ensemble means over 4 independent builds per tube (sem over builds):

| tube | switch-on @0.01 V/Å | switch-on @0.2 V/Å | switch-off | zero-field autocorr |
| --- | --- | --- | --- | --- |
| (10,10) | ~2.7 ns (3 builds; 1 fit NaN: 2.4–6.0 ns range) | 2.67 ± 1.06 ns | 1.96 ± 0.62 ns | 0.34 ± 0.06 ns |
| (16,16) | 3.01 ± 0.93 ns | >4.6 ns (3 of 4 fits; 1 NaN) | 1.35 ± 0.27 ns | 0.95 ± 0.45 ns |

**The Y7 gate condition is TRIGGERED: τ_M at the working field exceeds
1 ns in BOTH tubes** (2.7 and 3.0 ns). Per the brief's own rule, Stage B
waits for the owner's decision on run length: its cost scales as 20 τ_M,
i.e. ~55–60 ns per run × ~50 runs — far beyond both the local machine and
the original 10-ns-per-run plan, and the dump cadence rule
(max(10 ps, τ_rot)) lands at ~1.5–2.5 ns per frame. The owner's options:
(a) cluster budget for the full 20 τ_M protocol; (b) accept τ_M-limited
blocks (shorter than 20 τ_M, larger error bars, honestly reported);
(c) slide the go/no-go date. **The 2026-10-29 date is not achievable with
the protocol as written** — the τ_M numbers are the reason, and they are
measured, not assumed.

## τ_rot — single-molecule dipole reorientation (C1 fits)

Pooled per build (per-molecule fits vary by 2–3 orders of magnitude; the
pooled estimator is the robust one):

| tube/build | EC pooled | EMC pooled | never-reversed fraction |
| --- | --- | --- | --- |
| cnt10 b1 / b2 | 6.1 / 36.8 ns | 0.77 / 2.5 ns | 54% / 69% |
| cnt16 b1 / b2 | 1.46 / 1.90 ns | 1.55 / 1.57 ns | 31% / 34% |

Single-molecule reorientation is glassy-slow in confinement (EC worse than
EMC; the narrow tube worse than the wide one at the build level), consistent
with the reversal statistics and with the pilot's non-equilibration finding.
Note the dump-cadence consequence above.

## Mirrored starts (the symmetry check)

- (10,10): b1 +0.0567 vs mirrored b1m +0.0568 — the offset does NOT flip.
  Both are within ~2 sem of zero (block sems 0.031/0.023): consistent with
  a slow-drift remnant around zero rather than a pinned asymmetry, but the
  mirror test itself is inconclusive at this run length.
- (16,16): b1 −0.0471 (block sem 0.006 — significantly nonzero) vs b1m
  +0.0197 (sem 0.021 — consistent with zero): the sign DOES flip, and the
  mirrored copy relaxes toward zero faster. Evidence that the zero-field
  offsets are initial-condition remnants, not equilibrium order.

Zero-field P1 across all 8 builds: −0.248 to +0.134 — every build's own
block statistics still straddle the slow drift the reviewer measured in the
pilot; none is an equilibrium zero.

## Filling runs — FAILED (physical instability, fix in progress)

Both filling cells blew up during the 10→300 K ramp under NPT:
fill_cnt10 expanded ×26 in volume by 7 ps (Tmob 705 K, density 0.158→0.006)
and died on "Domain too large for neighbor bins"; fill_cnt16 reached
"Non-numeric box dimensions". Root context: the packed bath starts at ~87%
of the CRC density (sequential-insertion ceiling) with an estimated +20,000
atm internal pressure; the plan's 500 ps T+P anneal was meant to relax it,
but the ramp reaches instability first. Repair direction (S3, in progress):
staged decompression (NVT settle → gradual barostat with weak damping
and/or stepwise target pressure) before the production NPT. Until the fills
complete, the reservoir filling counts Y6 asks for are PENDING; the tube
τ numbers above do not depend on them.

## What this means for the go/no-go (D32)

The gate compares Chaord's blind block statements against a direct P1
analysis at the working field. With τ_M ≈ 2.7–3.0 ns, a block average of
5 τ_M is ~15 ns per block and the ~50-run ladder at 20 τ_M is ~1.1 µs of
aggregate sampling — a cluster-scale campaign. The honest paths are the
three options above; the owner picks. Everything DOWNSTREAM of that choice
(Y7 inputs at the chosen run length, Y8's block statistics) is already
specified and ready.
