"""Numpy re-implementation of the repo's floor observables, single-species LJ
frames (chaord-free).

Convention sources (cited, not imported -- D23):
  src/chaord/cv/noise.py  observables() + distance(): all-species g(r)
    (rmax gr_rmax_fluid, bins gr_bins_fluid, g = 2h/(N rho shell)),
    cn_hist (cutoff cn_cutoff_fluid, histogram np.arange(-0.5,25.5),
    normalised), gr_rms over bin centres > noise_gr_rmin, cn_tv = half L1.
  Thresholds at the core+lj dialect: lj.yaml gr_rmax_fluid 2.5,
    gr_bins_fluid 60, cn_cutoff_fluid 1.5; core.yaml noise_gr_rmin 0.8.
  research/ka_convergence/ka_obs.py: the same re-implementation for the
    ka_glass dialect, validated bit-exact against reports/noise_floors.json
    (rel dev 0.0, validation/report.json V3).  This module is the
    single-species (no partial g(r)) cut of the same code; validated here by
    validate_conventions.py against the committed lj_liquid /
    lj_liquid_large floor pairs (must reproduce every committed pair
    distance bit-exactly).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from liq_params import GR_RMAX, GR_BINS, CN_CUT, GR_RMIN


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


def cn_hist(pos, L):
    """Coordination histogram -- cv/noise.py 'cn_hist' convention."""
    tree = cKDTree(pos, boxsize=L)
    counts = np.array([len(x) - 1
                       for x in tree.query_ball_point(pos, CN_CUT)])
    hist = np.histogram(counts, np.arange(-0.5, 25.5))[0].astype(float)
    return hist / max(hist.sum(), 1.0), counts


def frame_observables(pos, L):
    """The legacy observable set for a single-species frame (no partials)."""
    pos = np.mod(np.asarray(pos, float), np.asarray(L, float))
    g, rm = gr_all(pos, L)
    h, counts = cn_hist(pos, L)
    return {"gr": g, "rm": rm, "cn_hist": h,
            "cn": float(counts.mean())}


def distance(o1, o2):
    """cv/noise.py distance for the legacy observable set."""
    m = o1["rm"] > GR_RMIN
    d = {"gr_rms": float(np.sqrt(np.mean((o1["gr"][m] - o2["gr"][m]) ** 2)))}
    n = min(len(o1["cn_hist"]), len(o2["cn_hist"]))
    d["cn_tv"] = float(0.5 * np.abs(
        o1["cn_hist"][:n] - o2["cn_hist"][:n]).sum())
    return d


def pair_distances(obs_list):
    """All pairwise distances of a frame list (i<j), as a list of dicts."""
    out = []
    for i in range(len(obs_list)):
        for j in range(i + 1, len(obs_list)):
            d = distance(obs_list[i], obs_list[j])
            d["frames"] = [i, j]
            out.append(d)
    return out


def d15(values):
    """The D15 floor estimator: max(mean, P90) (gate_a_prime D15 kept)."""
    v = np.asarray(values, float)
    return float(max(np.mean(v), np.percentile(v, 90)))
