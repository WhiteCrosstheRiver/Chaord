"""Phase segmentation and interface objects (M4).

Per-atom labels come from local order (the dialect's q6 rule) smoothed by a
majority vote over neighbours, giving spatially coherent phases. Interfaces
are objects with a position and a width, read from the solid-fraction profile
(10-90 crossings), independent of any fit.
"""
from __future__ import annotations

import numpy as np

from ..io.frames import Frame
from .passes import qbar


def solid_mask(frame: Frame, dialect) -> np.ndarray:
    rc = float(dialect.threshold("q6_cutoff"))
    thr = float(dialect.threshold("q6_solid"))
    q6, _, _ = qbar(np.mod(frame.pos, frame.cell_diag), frame.cell_diag, rc=rc)
    return q6 > thr


def smooth_labels(mask: np.ndarray, frame: Frame, dialect,
                  votes: int = None) -> np.ndarray:
    """Majority vote over each atom's neighbourhood (spatial coherence).

    The vote radius scales with the typical nearest-neighbour distance, so the
    rule works in Angstrom systems and reduced units alike."""
    from scipy.spatial import cKDTree
    from ..build.defects import typical_neighbor_distance
    L = frame.cell_diag
    pos = np.mod(frame.pos, L)
    pos = np.minimum(pos, L * (1 - 1e-9))  # dialect-exempt: strict upper edge for KD trees
    factor = float(dialect.threshold("segment_vote_factor"))
    rc = factor * typical_neighbor_distance(frame)
    flip_frac = float(dialect.threshold("segment_flip_fraction"))
    votes = int(votes if votes is not None else 1)
    tree = cKDTree(pos, boxsize=L)
    out = mask.copy()
    for _ in range(votes):
        flipped = 0
        new = out.copy()
        for i, nbrs in enumerate(tree.query_ball_point(pos, rc)):
            if not nbrs:
                continue                      # no evidence: keep the label
            nv = int(out[nbrs].sum()) - (1 if out[i] else 0)
            n_other = len(nbrs) - 1
            if n_other <= 0:
                continue
            frac_solid = nv / n_other
            if frac_solid >= flip_frac and not out[i]:
                new[i] = True
                flipped += 1
            elif frac_solid <= 1 - flip_frac and out[i]:
                new[i] = False
                flipped += 1
        out = new
        if flipped == 0:
            break
    return out


def phase_labels(frame: Frame, dialect) -> np.ndarray:
    return smooth_labels(solid_mask(frame, dialect), frame, dialect)


def solid_fraction_profile(labels: np.ndarray, frame: Frame, axis=2,
                           bin_size=None, dialect=None):
    """Smoothed solid fraction per bin along one axis."""
    L = frame.cell_diag
    bin_size = float(bin_size if bin_size is not None
                     else dialect_profile_bin(dialect))
    n = max(int(L[axis] / bin_size), 8)
    edges = np.linspace(0, L[axis], n + 1)
    idx = np.clip((frame.pos[:, axis] / L[axis] * n).astype(int), 0, n - 1)
    solid = np.bincount(idx, labels.astype(float), minlength=n)
    total = np.bincount(idx, minlength=n).astype(float)
    w = int(2)  # dialect-exempt: light circular smoothing width
    def circ(a, k):
        pad = np.r_[a[-k:], a, a[:k]]
        return np.convolve(pad, np.ones(k), "same")[k:-k]
    frac = circ(solid, w) / np.maximum(circ(total, w), 1.0)  # dialect-exempt: divide-by-zero guard
    centres = 0.5 * (edges[1:] + edges[:-1])  # dialect-exempt: bin centres
    return centres, frac


def dialect_profile_bin(dialect):
    try:
        return float(dialect.threshold("profile_bin_size"))
    except Exception:
        return 0.25  # dialect-exempt: fallback bin width


def _bin_size_for(dialect, frame):
    L = frame.cell_diag
    default = max(L[2] / 80.0, 0.25)  # dialect-exempt: sane default bin width
    try:
        return float(dialect.threshold("profile_bin_size"))
    except Exception:
        return default


def slab_interfaces(labels: np.ndarray, frame: Frame, dialect, axis=2) -> list[dict]:
    """Interface objects along one axis: position (50% crossing) and width.

    The width is the distance between the 10% and 90% crossings, measured on
    the smoothed solid-fraction profile — the same definition the noise floor
    calibrates, no fit involved."""
    L = frame.cell_diag[axis]
    centres, frac = solid_fraction_profile(labels, frame, axis,
                                           bin_size=_bin_size_for(dialect, frame))
    n = len(centres)
    interfaces = []
    for i in range(n):
        j = (i + 1) % n  # circular: slabs wrap through the periodic boundary
        a, b = frac[i], frac[j]
        if (a - 0.5) * (b - 0.5) < 0:  # dialect-exempt: 50% crossing level
            t = (0.5 - a) / (b - a)  # dialect-exempt: crossing interpolation
            z0 = centres[i] + t * (centres[j] - centres[i]) if j > i else \
                centres[i] + t * ((centres[j] + L) - centres[i])
            z0 = float(np.mod(z0, L))
            # width: 10-90 crossing distance around this interface
            lo = hi = z0
            for k in range(n):
                zk = centres[(i - k) % n] - (L if (i - k) % n > i else 0)
                if frac[(i - k) % n] > 0.9:  # dialect-exempt: crossing level
                    lo = zk
                    break
            for k in range(n):
                idx = (j + k) % n
                zk = centres[idx] + (L if idx < j else 0)
                if frac[idx] < 0.1 + 1e-9:  # dialect-exempt: 10% crossing level
                    hi = zk
                    break
            width = float(min(abs(hi - lo), L))  # dialect-exempt: periodic clamp
            interfaces.append(dict(at=z0, width=width))
    return sorted(interfaces, key=lambda d: d["at"])


def label_accuracy(labels: np.ndarray, truth: np.ndarray) -> float:
    """Fraction of atoms whose phase label matches planted ground truth."""
    return float((labels == truth).mean())
