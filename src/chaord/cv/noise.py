"""Noise floor: the distance between two independent frames of one simulation.

A statistical round trip passes when the rebuilt structure is no further from
the original than a multiple (1.5x) of this floor — never a guessed tolerance.
"""
from __future__ import annotations

import numpy as np

from ..io.frames import Frame


def observables(frame: Frame, dialect, rmax=None, bins=None) -> dict:
    """The held-out fluid observables used by the round-trip harness."""
    from scipy.spatial import cKDTree
    L = frame.cell_diag
    pos = np.mod(frame.pos, L)
    rmax = float(rmax if rmax is not None else dialect.threshold("gr_rmax_fluid"))
    bins = int(bins if bins is not None else dialect.threshold("gr_bins_fluid"))
    rho = len(pos) / float(np.prod(L))
    tree = cKDTree(pos, boxsize=L)

    pairs = tree.query_pairs(rmax, output_type="ndarray")
    d = pos[pairs[:, 1]] - pos[pairs[:, 0]]
    d -= L * np.round(d / L)
    r = np.linalg.norm(d, axis=1)
    h, e = np.histogram(r, np.linspace(0, rmax, bins + 1))
    shell = 4 / 3 * np.pi * (e[1:] ** 3 - e[:-1] ** 3)
    g = 2 * h / (len(pos) * rho * shell)

    rc = float(dialect.threshold("cn_cutoff_fluid"))
    counts = np.array([len(x) - 1 for x in tree.query_ball_point(pos, rc)])
    hist = np.histogram(counts, np.arange(-0.5, 25.5))[0].astype(float)
    hist = hist / max(hist.sum(), 1.0)  # dialect-exempt: normalisation guard
    return {"gr": g, "rm": 0.5 * (e[1:] + e[:-1]), "cn": float(counts.mean()),
            "cn_hist": hist}


def distance(o1: dict, o2: dict, rmin=0.8) -> dict:
    """Distances between two observable sets (g(r) RMS, cn histogram TV)."""
    m = o1["rm"] > rmin
    gr_rms = float(np.sqrt(np.mean((o1["gr"][m] - o2["gr"][m]) ** 2)))
    n = min(len(o1["cn_hist"]), len(o2["cn_hist"]))
    cn_tv = float(0.5 * np.abs(o1["cn_hist"][:n] - o2["cn_hist"][:n]).sum())
    return {"gr_rms": gr_rms, "cn_tv": cn_tv}


def noise_floor(frame_a: Frame, frame_b: Frame, dialect) -> dict:
    return distance(observables(frame_a, dialect), observables(frame_b, dialect))


def within_floor(frame_rebuilt: Frame, frame_original: Frame,
                 frame_later: Frame, dialect, factor: float = 1.5) -> dict:
    """Statistical round-trip verdict against the measured noise floor.

    The floor is the distance between `frame_original` and `frame_later`, two
    frames of the same reference simulation; the verdict asks whether the
    rebuilt frame is closer to the original than `factor` times that natural
    fluctuation, key by key."""
    oo = observables(frame_original, dialect)
    ol = observables(frame_later, dialect)
    o_re = observables(frame_rebuilt, dialect)
    d_ref = distance(oo, ol)
    d_re = distance(oo, o_re)
    out = {}
    for k in d_ref:
        floor = max(d_ref[k], 1e-6)  # dialect-exempt: degenerate-floor guard
        out[k] = dict(distance=d_re[k], floor=d_ref[k], ratio=d_re[k] / floor,
                      passed=bool(d_re[k] <= factor * floor))
    return out
