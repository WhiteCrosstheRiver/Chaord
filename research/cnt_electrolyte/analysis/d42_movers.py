"""D42 (Review 11): a non-P1 statement counts as MOVED only if

  (a) it shows a monotone dose-response over at least 3 consecutive
      fields of the ladder,
  (b) its shift is significant at Holm-corrected alpha = 0.05 across ALL
      statements x fields tested (the family is the full per-statement
      per-field grid; Holm over the sorted p-values), and
  (c) ion-pair statements additionally involve at least 10 ions of each
      kind (Li+ and anion) -- with 6 Li+ in the pilot cells nothing
      qualifies.

Until a statement clears all three, NO secondary-mover claim is made in
any report ("none established"). This module implements the bar; its test
pins that the current full-ladder data yield no mover.

Run:  .venv/Scripts/python.exe research/cnt_electrolyte/analysis/d42_movers.py
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
RESULTS = (ROOT / "research/cnt_electrolyte/analysis"
           / "full_ladder_results.json")
FIELDS = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2)
TUBES = (10, 16)
ALPHA = 0.05
MIN_MONOTONE_RUN = 3          # consecutive fields (D42a)
MIN_IONS = 10                 # per kind, ion-pair statements (D42c)


def holm(pvals):
    """Holm-Bonferroni: returns adjusted p-values in the original order."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        adj_i = (m - rank) * pvals[idx]
        running = max(running, adj_i)
        adj[idx] = min(running, 1.0)
    return adj


def statement_series(jobs, tube, key, block="B"):
    """Per-field run values of one statement's block-B mean: {field: [vals]}.
    (The per-run unit under D39/D31: one value per run, from its primary
    block; block A is the transient-biased disclosure block.)"""
    series = {}
    for j in jobs.values():
        if j["tube"] != tube:
            continue
        val = (j.get("blocks", {}) or {}).get(key, {}).get(block)
        if val is None:
            continue
        series.setdefault(j["field"], []).append(val)
    return series


def monotone_run_length(means):
    """Longest strictly monotone (up or down) run over consecutive fields,
    fields sorted ascending."""
    ms = [m for _f, m in sorted(means)]
    best, cur_up, cur_down = 1, 1, 1
    for a, b in zip(ms, ms[1:]):
        if b > a:
            cur_up += 1
        else:
            cur_up = 1
        if b < a:
            cur_down += 1
        else:
            cur_down = 1
        best = max(best, cur_up, cur_down)
    return best


def evaluate():
    jobs = json.loads(RESULTS.read_text(encoding="utf-8"))["jobs"]
    # collect every (tube, statement) with a full-zero baseline
    families = {}
    movers = []
    all_p = []
    grid = []
    for t in TUBES:
        keys = set()
        for j in jobs.values():
            if j["tube"] == t:
                keys.update((j.get("blocks", {}) or {}))
        for key in sorted(keys):
            series = statement_series(jobs, t, key)
            if 0.0 not in series or len(series.get(0.0, [])) < 2:
                continue
            off = np.array(series[0.0], float)
            fields = [f for f in FIELDS if len(series.get(f, [])) >= 2]
            for f in fields:
                on = np.array(series[f], float)
                if off.std() == 0 and on.std() == 0 and off.mean() == on.mean():
                    continue
                r = stats.ttest_ind(on, off, equal_var=False)
                grid.append((t, key, f, float(r.pvalue)))
            # D42a: monotone dose-response over consecutive fields
            means = [(f, np.mean(series[f])) for f in sorted(series)
                     if len(series[f]) >= 2]
            mono = monotone_run_length(means) if len(means) >= 3 else 0
            families[(t, key)] = dict(monotone=mono, series={
                str(k): v for k, v in series.items()})
    # D42b: Holm across the whole grid
    pvals = np.array([p for *_x, p in grid])
    adj = holm(pvals)
    holm_sig = {(grid[i][0], grid[i][1], grid[i][2])
                for i in range(len(grid)) if adj[i] < ALPHA}
    li_counts = {t: min(j["census"].get("Li", 0)
                        for j in jobs.values() if j["tube"] == t)
                 for t in TUBES}
    for (t, key), info in families.items():
        # (c) ion-count bar for ion-pair statements: >= MIN_IONS of each kind
        if "ion_pairs" in key and li_counts[t] < MIN_IONS:
            info["ion_bar_fail"] = (f"{li_counts[t]} Li+ < {MIN_IONS}")
            continue
        qualified = (info["monotone"] >= MIN_MONOTONE_RUN
                     and any(hs[0] == t and hs[1] == key for hs in holm_sig))
        if qualified:
            movers.append(dict(tube=t, statement=key,
                               monotone_run=info["monotone"]))
    rejections = {f"{t}|{k}": v.get("ion_bar_fail")
                  for (t, k), v in families.items()
                  if v.get("ion_bar_fail")}
    return dict(movers=movers, n_tests=len(grid),
                holm_significant=sorted(f"{t}|{k}|{f}"
                                        for t, k, f in holm_sig),
                ion_bar_rejections=rejections)


if __name__ == "__main__":
    out = evaluate()
    print(json.dumps(out, indent=1))
