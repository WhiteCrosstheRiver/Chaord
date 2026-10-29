"""D39 reference analysis (Review 11): detection of a field = a test of the
MEAN SHIFT with independent runs as the unit -- Welch's t, two-sided,
alpha = 0.05, zero-field run means (blocks A and B averaged) against
field-on block-B values.

The build-paired contrast is reported as secondary. The D15 3x-floor rule
remains only for per-block classification claims (labelled as such in
go_no_go.md; it tests per-block separation, not a mean shift -- the
reviewer's D31/D32 correction, pre-stated here before the Z3 confirmatory
data exist).

Run:  .venv/Scripts/python.exe research/cnt_electrolyte/analysis/d39_welch.py
Prints the table recorded in docs/reviews/open_items_v5.md section 0 and
returns the same numbers as JSON on stdout with --json.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
RESULTS = (ROOT / "research/cnt_electrolyte/analysis"
           / "full_ladder_results.json")
FIELDS = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2)
TUBES = (10, 16)


def run_means_zero_field(jobs, tube):
    """Zero-field run means: blocks A and B averaged, per independent run
    (D39's unit)."""
    return np.array([
        np.mean([j["direct_p1_blocks"]["A"], j["direct_p1_blocks"]["B"]])
        for j in jobs.values()
        if j["tube"] == tube and j["field"] == 0])


def block_b_on(jobs, tube, field):
    """Field-on block-B values, per independent run."""
    return np.array([j["direct_p1_blocks"]["B"] for j in jobs.values()
                     if j["tube"] == tube and j["field"] == field])


def main():
    jobs = json.loads(RESULTS.read_text(encoding="utf-8"))["jobs"]
    out = {}
    for t in TUBES:
        off = run_means_zero_field(jobs, t)
        for E in FIELDS:
            on = block_b_on(jobs, t, E)
            r = stats.ttest_ind(on, off, equal_var=False)
            shift = on.mean() - off.mean()
            line = (f"({t},{t}) {E}: shift {shift:+.3f}  "
                    f"t {r.statistic:4.1f}  p {r.pvalue:.4f}")
            print(line)
            out[f"{t}_{E}"] = dict(tube=t, field=E, shift=shift,
                                   t=float(r.statistic),
                                   p=float(r.pvalue),
                                   detected=bool(r.pvalue < 0.05))
    if "--json" in sys.argv:
        print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()
