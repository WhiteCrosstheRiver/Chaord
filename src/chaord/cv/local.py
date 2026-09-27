"""Local-structure CVs implemented on numpy/scipy (no OVITO needed).

Geometry helpers (wrap/mic/pairs_within) live in chaord.realize.lj and are reused
here so that measurement, restraint and assertion share one definition.
"""
from __future__ import annotations

import numpy as np

from ..io.frames import Frame
from ..lift.passes import pairs_within
from ..realize.lj import mic, wrap
from .registry import register, missing_measure


def cn_mean(frame: Frame, dialect, cutoff=None) -> float:
    """Mean coordination number under the dialect's cn_cutoff (or an override)."""
    rc = float(cutoff if cutoff is not None else dialect.threshold("cn_cutoff"))
    p = pairs_within(frame.pos, frame.cell_diag, rc)
    counts = np.bincount(p.ravel(), minlength=len(frame.pos))
    return float(counts.mean())


def cn_of(frame: Frame, dialect, cutoff=None) -> np.ndarray:
    rc = float(cutoff if cutoff is not None else dialect.threshold("cn_cutoff"))
    p = pairs_within(frame.pos, frame.cell_diag, rc)
    return np.bincount(p.ravel(), minlength=len(frame.pos))


def number_density(frame: Frame, dialect, mask=None) -> float:
    pos = frame.pos if mask is None else frame.pos[mask]
    vol = float(np.prod(frame.cell_diag))
    return float(len(pos) / vol)


def angle_mean(frame: Frame, dialect, cutoff=None) -> float:
    """Mean bond angle in degrees over all bonded triples."""
    rc = float(cutoff if cutoff is not None else dialect.threshold("cn_cutoff"))
    L = frame.cell_diag
    p = pairs_within(frame.pos, L, rc)
    nbrs: dict[int, list[int]] = {}
    for a, b in p:
        nbrs.setdefault(int(a), []).append(int(b))
        nbrs.setdefault(int(b), []).append(int(a))
    angles = []
    for i, js in nbrs.items():
        if len(js) < 2:
            continue
        v = mic(frame.pos[js] - frame.pos[i], L)
        v = v / np.linalg.norm(v, axis=1)[:, None]
        c = v @ v.T
        iu = np.triu_indices(len(js), 1)
        angles.append(np.degrees(np.arccos(np.clip(c[iu], -1, 1))))
    if not angles:
        return float("nan")
    return float(np.mean(np.concatenate(angles)))


def q6_mean(frame: Frame, dialect, cutoff=None) -> float:
    from ..lift.passes import qbar
    rc = float(cutoff if cutoff is not None else dialect.threshold("q6_cutoff"))
    q6, _, _ = qbar(frame.pos, frame.cell_diag, rc=rc)
    return float(q6.mean())


def solid_like_fraction(frame: Frame, dialect, cutoff=None) -> float:
    from ..lift.passes import qbar
    rc = float(cutoff if cutoff is not None else dialect.threshold("q6_cutoff"))
    thr = float(dialect.threshold("q6_solid"))
    q6, _, _ = qbar(frame.pos, frame.cell_diag, rc=rc)
    return float((q6 > thr).mean())


register("cn", None, "mean coordination number at the dialect's first-shell cutoff",
         thresholds=("cn_cutoff",), measure=cn_mean)
register("density", None, "number density (1/volume units of the system)", measure=number_density)
register("angle_mean", "deg", "mean bond angle over bonded triples",
         thresholds=("cn_cutoff",), measure=angle_mean)
register("q6", None, "mean Lechner-Dellago averaged q6 order parameter",
         thresholds=("q6_cutoff",), dialect="lj", measure=q6_mean)
register("solid_like", None, "fraction of atoms with q6 above the dialect's solid-like threshold",
         thresholds=("q6_cutoff", "q6_solid"), dialect="lj", measure=solid_like_fraction)
register("gr_peak", None, "first g(r) peak position and height (M3)",
         thresholds=("gr_rmax", "gr_bins", "cn_cutoff"), dialect="lj",
         measure=missing_measure("gr_peak", None))
register("sites_matched", "%", "fraction of crystal sites occupied within tolerance (M1)",
         thresholds=("site_match_tolerance",),
         measure=missing_measure("sites_matched", None))
register("solid_clusters", None, "number of solid-like clusters in a fluid region (M3)",
         thresholds=("q6_cutoff", "q6_solid"), dialect="lj",
         measure=missing_measure("solid_clusters", None))
register("sro_alpha1", None, "Warren-Cowley first-shell SRO parameter (M2)",
         measure=missing_measure("sro_alpha1", None))
register("coverage", "ML", "adsorbate coverage in monolayers (M4)", dialect="surface",
         measure=missing_measure("coverage", None))
register("pairing", None, "ion-pairing fractions CIP/SSIP/AGG (M3)", dialect="molecular",
         measure=missing_measure("pairing", None))
register("compressibility", None, "compressibility factor Z = PV/nRT (M3)", dialect="molecular",
         measure=missing_measure("compressibility", None))
