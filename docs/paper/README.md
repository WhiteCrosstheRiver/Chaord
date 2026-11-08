# Paper folder

Everything for the Chaord manuscript lives here.

| Path | Contents |
| --- | --- |
| `draft.md` | Manuscript text (the T5 "done when" file) |
| `outline.md` | T5a outline and claims table |
| `figures/` | Figures, PDF for the manuscript and PNG for review |
| `assets/` | `paper_numbers.json` and `table1_acceptance.md`, both generated |
| `scripts/` | `make_paper_assets.py`, which regenerates `figures/` and `assets/` |

Raw evidence stays where the project keeps it (`reports/`,
`research/cnt_electrolyte/analysis/`, `research/cnt_electrolyte/r13/`), and the
scripts read it from there, so nothing is copied or allowed to drift.

Regenerate from the repository root (needs numpy, scipy and matplotlib; the
script stops if a recomputed statistic disagrees with the stored one):

    python docs/paper/scripts/make_paper_assets.py

Rule (D59): every number in `draft.md` comes from `paper_numbers.json` or a
named committed file. Text in **[VERIFY]** has not been traced to source yet.
