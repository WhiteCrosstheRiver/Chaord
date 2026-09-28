"""Noise floor: the distance between two independent frames of one simulation.

A statistical round trip passes when the rebuilt structure is no further from
the original than a multiple (1.5x) of this floor — never a guessed tolerance.

The observable set is configurable (`observables(..., selection=...)`) and
so is the distance (`distance(..., keys=...)`): the rich observables of
chaord.cv.rich (species-pair g(r), bond angles, density profile) can be
gated individually, which is how the acceptance harness (A5) composes a
verdict on any observable subset.
"""
from __future__ import annotations

import numpy as np

from ..io.frames import Frame
from ..lang.errors import ChaordError

SELECTION_GROUPS = ("gr", "partial_gr", "angle", "density")


def _resolve_selection(selection) -> tuple:
    if selection is None:
        return ("gr",)
    groups = tuple(selection)
    if not groups:
        raise ChaordError("selection must name at least one observable group "
                          f"of {', '.join(SELECTION_GROUPS)}")
    unknown = [g for g in groups if g not in SELECTION_GROUPS]
    if unknown:
        raise ChaordError(f"unknown observable group(s) {', '.join(map(repr, unknown))}; "
                          f"known: {', '.join(SELECTION_GROUPS)}")
    return groups


def observables(frame: Frame, dialect, rmax=None, bins=None,
                selection=None) -> dict:
    """The held-out fluid observables used by the round-trip harness.

    `selection=None` (the default, and what every existing caller uses) is
    the legacy set: the all-species g(r), the mean coordination and the
    coordination histogram.  Selection groups add the rich observables of
    chaord.cv.rich on top (they never change the legacy keys):

      "gr"        — g(r), bin centres, mean cn, cn histogram (legacy keys)
      "partial_gr" — one g(r) per unordered species pair present, keyed
                    ``gr_<a>-<b>`` with bin centres ``rm_<a>-<b>``
      "angle"     — bond-angle density ``angle`` with centres ``angle_centers``
      "density"   — per-bin number density along z, key ``density``
    """
    from scipy.spatial import cKDTree
    groups = _resolve_selection(selection)
    out: dict = {}
    if "gr" in groups:
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
        hist = np.histogram(counts, np.arange(-0.5, 25.5))[0].astype(float)  # dialect-exempt: numerical-guard: half-integer bin edges for integer counts
        hist = hist / max(hist.sum(), 1.0)  # dialect-exempt: numerical-guard: normalisation guard
        out.update({"gr": g, "rm": 0.5 * (e[1:] + e[:-1]), "cn": float(counts.mean()),  # dialect-exempt: numerical-guard: bin centres
                    "cn_hist": hist})
    from .rich import angle_distribution, density_profile, partial_gr
    if "partial_gr" in groups:
        species = sorted(set(frame.symbols))
        from itertools import combinations_with_replacement
        for a, b in combinations_with_replacement(species, 2):
            g, r = partial_gr(frame, dialect, a, b)
            out[f"gr_{a}-{b}"] = g
            out[f"rm_{a}-{b}"] = r
    if "angle" in groups:
        centers, density = angle_distribution(frame, dialect)
        out["angle"] = density
        out["angle_centers"] = centers
    if "density" in groups:
        out["density"] = density_profile(frame, dialect)
    return out


def distance(o1: dict, o2: dict, rmin=None, dialect=None, keys=None) -> dict:
    """Distances between two observable sets.

    Every comparable key present in both sets contributes: the g(r) RMS
    (the legacy all-species ``gr`` and every partial ``gr_<a>-<b>``, each
    ignoring r below `rmin`, read from the dialect's `noise_gr_rmin`
    unless the caller passes one), the cn histogram TV, the bond-angle TV
    and the density-profile RMS.  For a legacy observable set this is
    exactly ``{"gr_rms", "cn_tv"}``, as before.

    `keys` optionally filters the returned distances by exact name or
    prefix — e.g. ``keys=("gr_rms_O-O", "angle_tv")`` gates a water round
    trip on the O-O structure and the bond angles; ``keys=("gr_rms",)``
    keeps the whole g(r) family."""
    if rmin is None:
        from ..dialects import load_dialect
        d = dialect if dialect is not None else load_dialect(("core",))
        rmin = float(d.threshold("noise_gr_rmin"))
    gkeys = ["gr"] + sorted(k for k in set(o1) | set(o2)
                            if k.startswith("gr_") and len(k) > 3)
    comparable = gkeys + ["cn_hist", "angle", "density"]
    bad = [k for k in comparable if (k in o1) != (k in o2)]
    if bad:
        raise ChaordError(
            f"observable sets disagree on {', '.join(map(repr, bad))} "
            "(build both with the same observables selection)")
    out = {}
    for gk in gkeys:
        if gk not in o1:                     # equal presence was checked above
            continue
        suffix = "" if gk == "gr" else "_" + gk[3:]
        rk = "rm" if not suffix else "rm_" + gk[3:]
        if rk not in o1:
            raise ChaordError(f"observable sets lack the bin centres {rk!r}")
        m = o1[rk] > rmin
        out["gr_rms" + suffix] = float(
            np.sqrt(np.mean((o1[gk][m] - o2[gk][m]) ** 2)))
    if "cn_hist" in o1:
        n = min(len(o1["cn_hist"]), len(o2["cn_hist"]))
        out["cn_tv"] = float(0.5 * np.abs(o1["cn_hist"][:n] - o2["cn_hist"][:n]).sum())  # dialect-exempt: exact-geometry
    if "angle" in o1:
        if len(o1["angle"]) != len(o2["angle"]):
            raise ChaordError("angle distributions have different bin counts")
        out["angle_tv"] = float(0.5 * np.abs(o1["angle"] - o2["angle"]).sum())  # dialect-exempt: exact-geometry
    if "density" in o1:
        if len(o1["density"]) != len(o2["density"]):
            raise ChaordError("density profiles have different bin counts")
        out["density_rms"] = float(np.sqrt(np.mean(
            (o1["density"] - o2["density"]) ** 2)))
    if keys is not None:
        patterns = tuple(keys)
        out = {k: v for k, v in out.items()
               if any(k == p or k.startswith(p) for p in patterns)}
        if not out:
            raise ChaordError(f"key filter {patterns} matched no distance key")
    return out


def noise_floor(frame_a: Frame, frame_b: Frame, dialect,
                selection=None) -> dict:
    return distance(observables(frame_a, dialect, selection=selection),
                    observables(frame_b, dialect, selection=selection),
                    dialect=dialect)


def within_floor(frame_rebuilt: Frame, frame_original: Frame,
                 frame_later: Frame, dialect, factor: float = None,
                 selection=None) -> dict:
    """Statistical round-trip verdict against the measured noise floor.

    The floor is the distance between `frame_original` and `frame_later`, two
    frames of the same reference simulation; the verdict asks whether the
    rebuilt frame is closer to the original than `factor` times that natural
    fluctuation, key by key. `factor` defaults to the dialect's
    `noise_floor_factor`.  `selection` forwards to `observables` so the
    verdict can be taken on any observable subset (the default is the
    legacy set, unchanged)."""
    if factor is None:
        factor = float(dialect.threshold("noise_floor_factor"))
    oo = observables(frame_original, dialect, selection=selection)
    ol = observables(frame_later, dialect, selection=selection)
    o_re = observables(frame_rebuilt, dialect, selection=selection)
    d_ref = distance(oo, ol, dialect=dialect)
    d_re = distance(oo, o_re, dialect=dialect)
    out = {}
    for k in d_ref:
        floor = max(d_ref[k], 1e-6)  # dialect-exempt: numerical-guard: degenerate-floor guard
        out[k] = dict(distance=d_re[k], floor=d_ref[k], ratio=d_re[k] / floor,
                      passed=bool(d_re[k] <= factor * floor))
    return out
