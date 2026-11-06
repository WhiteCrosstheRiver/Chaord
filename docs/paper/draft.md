# Chaord: a compiled language for molecular simulation — round-trip fidelity and blind field detection in a confined electrolyte

*Working draft. Status: methods and filling-independent results drafted (T5b);
the reservoir-filling field result is pending (T3). Markers: **[T3 pending]**
waits for R13; **[VERIFY]** is not yet traced to source; **[CITE]** needs a full
reference. Numbers come from `assets/paper_numbers.json` unless a file is named.*

## Abstract

**[Write last.]** Three points must survive: the fidelity figure, the blind
field detection with its filling label, and the limits stated up front.

## 1. Introduction

Simulation studies are usually reported as prose plus scripts, with no
machine-readable statement of what a configuration is. Chaord is a language for
atomic systems in which a program describes a macrostate, a family of
configurations, and a coordinate file is one microstate of it. A compiler
(`chaord build`) realizes coordinates from a program, and a decompiler
(`chaord lift`) produces a program from coordinates.

We make the instrument claim first, because a physics claim made through a
language is only as good as that language's measured fidelity. We then use the
instrument on a blind problem: detecting field-aligned dipole order in a LiPF6
electrolyte confined in carbon nanotubes, where the lift is never told the field.

## 2. The language and the compiler

Statements have one of six kinds: build, state, constrain, assert, history and
conserve. Thresholds and cut-offs live only in versioned dialect files, and
every program names its physics backend. Counts per species and total charge are
conserved: what a lift cannot explain goes to a `residual`, and a lift that
cannot describe a frame refuses rather than printing a program it cannot stand
behind. Each collective variable is measured, restrained and checked by the same
code, so a printed statement and its check cannot diverge.

The blind rule (D29) is what makes the field test meaningful: when a dialect
names a dipole species, `dipole_p1` is measured unconditionally, and passing a
`field=` argument adds only a `state field` line. No lift in this work received
that argument.

**Figure 4.** Example program for a tube frame. **[Pending: run lift on a cited
frame; `spec/examples/11_nanotube_field.chaord` is the starting template.]**

## 3. Instrument validation

**Acceptance.** All 14 of 14 acceptance criteria (A1–A14) pass
(`assets/table1_acceptance.md`; Table 1). The criteria carry mutation tests: the
most likely real error for a criterion is injected and must make it fail. An
independent verifier with no code authorship reproduced the set
(`reports/verification_2026-10-08.md`). **[VERIFY: quote the CI run id from the
artifact itself before submission.]**

**Fidelity.** Across all 60,060 blind lifts (42,042 on the full ladder and 18,018
confirmatory), the largest difference between the statement `dipole_p1` and an
independent direct calculation was 5.00 × 10⁻⁴, which is the half-quantum of the
statement's three-decimal print (Figure 3). The residual is flat across that
interval, as rounding predicts, so the statement and the direct quantity agree
to print precision. There were no failed static checks on any confirmatory frame.

## 4. The physical system

**Model.** LiPF6 in EC:EMC in armchair carbon nanotubes, with rigid OPLS-AA-based
molecules (Nutakki 2023 **[CITE]**), Werder sp²-carbon parameters (Werder 2003
**[CITE]**) and Lorentz–Berthelot mixing, which is a recorded deviation. Tubes
are frozen and uncharged. We use PPPM (relative accuracy 10⁻⁴), a 12 Å cut-off, a
1 fs timestep and a single Langevin scheme with 1 ps damping for all mobile
atoms.

**Protocol.** Each job runs a 100 ps ramp from 10 to 300 K and 900 ps at 300 K at
zero field, then applies the field, settles for 1 ns, and records two 6 ns
production blocks. All jobs in a set use one LAMMPS build (3 Mar 2020).

**Packing at the pilot filling.** A seeded rejection packer places molecules
against contact floors of max(2.0 Å, 2.2 Å with H, 0.85 σᵢⱼ), and molecules that
cannot be placed strictly are inserted and pushed apart by translation.

**Packing at the reservoir filling.** The reservoir composition jammed this
packer: at floor scales 1.0–0.85 insertion failed on EC, and at 0.8–0.7 insertion
completed but translation alone could not recover the true floors. Following the
pre-registered staged policy, and without changing counts, we insert at 0.8 × the
floor and relax with rigid-body LAMMPS dynamics, so molecules can rotate as well
as translate. The 90 ps relaxation uses only a purely repulsive soft pair
potential ranged on the true floors and a radial wall, with no charges or LJ
terms. A build is accepted only if it clears the unmodified floors. The first
attempt on `r13_b1` missed the wall floor (0.9985 against 0.999), so the wall was
stiffened and the threshold left alone; the result has minimum pair/floor 1.098
and wall/floor 1.028 (`r13/data/build_manifest_r13_b1.json`). Starting production
from this state releases residual strain: the mobile temperature spikes to about
880 K within the first picosecond and is gone within 5 ps, well before any
analysed data (`r13/evidence/smoke/`). **[T3 pending: statistics for builds 2–6.]**

## 5. Filling is a state variable

Open-tube filling runs measure the uptake a packed pore should match. For (16,16)
the uptake per Å is EC 1.2701, EMC 0.5917, Li⁺ 0.2673 and PF6⁻ 0.2127, above the
pilot packing by 68.9%, 21.3%, 119.2% and 74.4% (Figure 5). For (10,10), the fill
holds EC and EMC well above the pilot, and no Li⁺ at all in the analysed window.
The uncertainties are lower bounds: the 1.25 ns blocks are shorter than the
collective relaxation time of about 3 ns.

Inside (16,16) the ions are not neutral: 26.3 Li⁺ per 20.9 PF6⁻ per 98.4 Å. We
report this as a finding. A periodic build cannot carry that net charge, so R13
uses 24 LiPF6 at the rounded mean, a stated modelling choice whose effect this
design cannot separate.

An earlier filling-sensitivity test (Q3) is void, because it used a composition
from an uncorrected draft and so tested neither filling. Both of its reported
results are preserved with their scripts (`q3_evidence/q3_results.json`,
`docs/records/corrections.md`). Stage B and Z4 below are labelled "packed below
open-tube uptake" throughout.

## 6. Blind field detection

**Stage B ladder.** Seven fields from 0 to 0.2 V/Å, two tubes, three builds each
(Figure 2a). The smallest detected field depends on the criterion. For (16,16) it
is 0.1 V/Å under the 3× floor rule, 0.02 V/Å under the build-paired 3σ rule and
0.01 V/Å under paired 2σ. For (10,10) it is 0.1 V/Å on both paired rules, and the
3× floor rule detects nothing.

**Confirmatory test (Z4), packed below open-tube uptake.** Pre-registered before
any job was built: 18 jobs, (16,16), 98.4 Å, fields 0, 0.01 and 0.02 V/Å, six
builds, with independent runs as the unit (Welch two-sided, α = 0.05, zero-field
run means against field-on block B). The registered gate passed.

| Contrast | Shift | 95% CI | t | df | p | Runs positive |
| --- | --- | --- | --- | --- | --- | --- |
| 0.01 vs 0 V/Å (gate) | +0.107 | [+0.018, +0.196] | 2.85 | 7.0 | 0.0246 | 5 of 6 |
| 0.02 vs 0 V/Å | +0.187 | [+0.135, +0.240] | 7.97 | 9.9 | 1.3 × 10⁻⁵ | 6 of 6 |

The 0.01 V/Å result is a modest detection: its interval nearly reaches zero and
one run sits on the zero-field side. It should not be read as more than the
registered rule says. The build-paired contrast is secondary and weakened: a
pre-registered chaining check found that 6 of the 18 jobs diverged from their
siblings during the shared zero-field ramp, a floating-point non-reproducibility
on the cluster, so exact pairing does not hold for them. The primary test is
unpaired and is not affected.

**Reservoir filling (R13).** Same design at the corrected uptake, 125 EC, 58 EMC
and 24 LiPF6 per 98.4 Å, pre-registered in `r13/PLAN.md` before any build. The
decision rule is that 0.01 V/Å is confirmed at this filling only if Welch p < 0.05
on this set alone. **[T3 pending: result, reported either way.]**

## 7. What the language adds

The field was detected through lifted programs, never from coordinates with field
knowledge. Conservation and residual accounting are falsifiable statements rather
than conventions: dropping one atom must fail. **[VERIFY: the KA-glass and
LJ-liquid measurement-error results in the outline, §8, are not yet traced to
`research/ka_convergence/` and `research/x7_liquid/`.]**

## 8. Limitations

1. Pilot force field (rigid OPLS-AA-based molecules, Werder carbon, LB mixing),
   with no DFT anchor yet.
2. Filling: packed-pore constructions. R13 packs the measured uptake, but it is
   not a grand-canonical ensemble, and it neutralizes a measured ion imbalance.
3. Narrow-tube non-ergodicity: (10,10) is glassy-slow, and the block protocol may
   under-sample it.
4. Cluster floating-point non-reproducibility: reruns are not bit-reproducible,
   and it affected 6 of 18 jobs in Z3.
5. Six runs per field: power against small shifts is limited.

## Figures

- **Table 1 / Figure 1.** Acceptance A1–A14 (`assets/table1_acceptance.md`).
- **Figure 2.** `figures/fig2_blind_p1_vs_field.pdf`. Blind P1 against field. (a)
  Stage B, mean ± s.e.m. over three builds. (b) (16,16) Stage B and Z4; dots are
  individual Z4 runs. Both sets are packed below open-tube uptake. **[T3 pending:
  add R13 points.]**
- **Figure 3.** `figures/fig3_fidelity.pdf`. Chaord statement against direct
  calculation on 18,018 confirmatory frames, and the residual, bounded by the
  print half-quantum.
- **Figure 4.** Annotated example program. **[Pending.]**
- **Figure 5.** `figures/fig5_filling.pdf`. Pilot packing against open-tube uptake
  per species and tube.
