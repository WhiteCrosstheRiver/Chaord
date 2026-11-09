# Submission plan: Nature Communications

Written 2026-11-07, updated through 2026-11-09 after R13, the program-length curve and macrostate selection. An honest reading of what a Nature Communications editor and
referees would ask, and what this repository can answer today.

## What already meets the bar

- **A clear, general idea.** A language whose programs are macrostates, with a
  compiler and a decompiler, and the round trip as the correctness test. The
  related-work search found no tool doing both directions with statistical
  meanings (`draft.md`, related work; to be confirmed against full texts).
- **Unusually strong verification culture.** 14/14 acceptance criteria with
  mutation tests, independent verifier, clean CI, pre-registered tests, and a
  public corrections record. Referees rarely see this; it should be visible.
- **A blind physics demonstration, honestly bounded.** Field detection through
  the text, never told the field, under pre-registered rules: confirmed at the
  pilot filling, NOT confirmed at the measured uptake (R13, p = 0.59; 0.02 V/Å
  p = 0.054). Reporting the negative result as registered is itself evidence of
  the verification culture, but it narrows the physics story.

## What a referee will attack

| Risk | Why it matters | Answer available | Action |
| --- | --- | --- | --- |
| One physics application | "General language" is claimed, but one system is shown in depth | Acceptance covers many families, but as tests, not as results | Add a short cross-family results figure from the existing bench (no new simulation) |
| Field effect does not survive the true filling | Z4 confirmed 0.01 V/Å only at a packed filling; R13 at the measured uptake did not (0.01: p = 0.59; 0.02: p = 0.054; between-build spread ≈ the 0.02 response) | Reported as registered (go_no_go section 15, draft section 6) | Frame the field work as a test of the instrument, which held (fidelity exact on all 78,078 lifts), not as an electrolyte result. Leading with 0.02 V/Å no longer works: it also misses at the true filling. Any rescue design (more runs, variance-removing primary) must be pre-registered as new work, never a reanalysis of R13 |
| No central quantitative curve | Fidelity is near-perfect by construction of the print; with the field result narrowed, the paper needs a second quantitative result | The pre-registered curve ran (7612053) and is reported as draft section 7: physics alone explains every pass, S1 is inert, S2 harms the structure; the registered verdict is NOT SUPPORTED (corrected from "undetermined") | The curve cannot be the central figure. It is an honest negative result about the two scalar restraints. A real length-for-physics test needs distribution-level restraints or floor-derived tolerances, pre-registered as new work. Macrostate selection (8deeff2, result 8c81b59; draft section 8, Figure 6) is now the second result: registered verdict NOT SUPPORTED because the slab missed the gate, while liquid, fcc and hcp were selected and all 12 wrong identifications rejected. It is a partial result, not a clean positive |
| Pilot force field | Physics conclusions depend on it | Stated as a limit | Keep claims instrument-first; no electrolyte-physics claims beyond detection |
| Why it matters | Editors want a use beyond self-consistency | The decision-model loop is the real motivation, but untested (D53) | Say so in the discussion; do not imply it works |
| Reproducibility | Code, data, environment | Repo, CI, sha256 manifests, archive (T6) | Finish T6; deposit data with a DOI |

## Order of work

1. ~~T3 and T5c~~: done (R13 NOT CONFIRMED; draft section 6 rewritten).
2. ~~Program-length curve~~: done (7612053; draft section 7, Table 2).
3. ~~Per-case A5 table and macrostate selection~~: done (Table 3, section 8, Figure 6).
4. ~~Read ATLAS and MatText; references and abstract~~: done.
5. T6 freeze and archive; data availability and code availability statements.
6. Internal review by an agent that wrote none of the text, against this table.

## Owner decisions open

- Labelled DFT set: yes or no (D53, due 2026-11-06).
- Framing after R13: instrument-first paper with the field tests as a bounded
  demonstration (recommended), or hold submission for a new pre-registered
  field design.
- Author list and acknowledgements.
- Whether to pre-register a follow-up on slab detection, or on a second restraint
  design, before submission. The draft reports both negatives as they stand.
