"""Numpy re-implementation of the repo's floor observables (chaord-free).

Convention sources (cited, not imported -- D23):
  src/chaord/cv/noise.py  observables() + distance(): all-species g(r)
    (rmax gr_rmax_fluid, bins gr_bins_fluid, g = 2h/(N rho shell)),
    cn_hist (cutoff cn_cutoff_fluid, histogram np.arange(-0.5,25.5),
    normalised), gr_rms over bin centres > noise_gr_rmin, cn_tv = half L1.
  src/chaord/cv/rich.py  partial_gr(): g_ab = (2 if a==b else 1) h V /
    (N_a N_b shell), same grid.
  thresholds at the ka_glass dialect: glass.yaml gr_rmax_fluid 2.5,
    gr_bins_fluid 60, cn_cutoff_fluid 1.5; core.yaml noise_gr_rmin 0.8.
Validated bit-exact against reports/noise_floors.json ka_glass pairs
(validate_against_bench.py check V3: every floor statistic reproduced with
rel dev 0.0).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

GR_RMAX, GR_BINS, CN_CUT, GR_RMIN = 2.5, 60, 1.5, 0.8


def gr_all(pos, L):
    """All-species g(r) -- cv/noise.py 'gr' key convention."""
    n = len(pos)
    rho = n / float(np.prod(L))
    tree = cKDTree(pos, boxsize=L)
    pairs = tree.query_pairs(GR_RMAX, output_type="ndarray")
    d = pos[pairs[:, 1]] - pos[pairs[:, 0]]
    d -= L * np.round(d / L)
    r = np.linalg.norm(d, axis=1)
    h, e = np.histogram(r, np.linspace(0, GR_RMAX, GR_BINS + 1))
    shell = 4 / 3 * np.pi * (e[1:] ** 3 - e[:-1] ** 3)
    g = 2 * h / (n * rho * shell)
    return g, 0.5 * (e[1:] + e[:-1])


def gr_partial(pos, L, symbols, a, b):
    """Species-pair g(r) -- cv/rich.py partial_gr convention."""
    syms = np.asarray(symbols)
    n_a = int((syms == a).sum())
    n_b = int((syms == b).sum())
    tree = cKDTree(pos, boxsize=L)
    pairs = tree.query_pairs(GR_RMAX, output_type="ndarray")
    sel = ((syms[pairs[:, 0]] == a) & (syms[pairs[:, 1]] == b)) | \
          ((syms[pairs[:, 0]] == b) & (syms[pairs[:, 1]] == a))
    pairs = pairs[sel]
    d = pos[pairs[:, 1]] - pos[pairs[:, 0]]
    d -= L * np.round(d / L)
    r = np.linalg.norm(d, axis=1)
    h, e = np.histogram(r, np.linspace(0, GR_RMAX, GR_BINS + 1))
    shell = 4 / 3 * np.pi * (e[1:] ** 3 - e[:-1] ** 3)
    w = 2.0 if a == b else 1.0
    g = w * h * float(np.prod(L)) / (n_a * n_b * shell)
    return g, 0.5 * (e[1:] + e[:-1])


def cn_hist(pos, L):
    """Coordination histogram -- cv/noise.py 'cn_hist' convention."""
    tree = cKDTree(pos, boxsize=L)
    counts = np.array([len(x) - 1
                       for x in tree.query_ball_point(pos, CN_CUT)])
    hist = np.histogram(counts, np.arange(-0.5, 25.5))[0].astype(float)
    return hist / max(hist.sum(), 1.0), counts


def frame_observables(pos, L, symbols):
    out = {}
    g, rm = gr_all(np.mod(pos, L), L)
    out["gr"], out["rm"] = g, rm
    h, _ = cn_hist(np.mod(pos, L), L)
    out["cn_hist"] = h
    for a, b in [("A", "A"), ("A", "B"), ("B", "B")]:
        gp, _ = gr_partial(np.mod(pos, L), L, symbols, a, b)
        out[f"gr_{a}-{b}"] = gp
    return out


def distance(o1, o2):
    """cv/noise.py distance for the ka_glass observable set."""
    d = {}
    m = o1["rm"] > GR_RMIN
    d["gr_rms"] = float(np.sqrt(np.mean((o1["gr"][m] - o2["gr"][m]) ** 2)))
    for key in ("A-A", "A-B", "B-B"):
        d[f"gr_rms_{key}"] = float(np.sqrt(np.mean(
            (o1[f"gr_{key}"][m] - o2[f"gr_{key}"][m]) ** 2)))
    n = min(len(o1["cn_hist"]), len(o2["cn_hist"]))
    d["cn_tv"] = float(0.5 * np.abs(
        o1["cn_hist"][:n] - o2["cn_hist"][:n]).sum())
    return d


