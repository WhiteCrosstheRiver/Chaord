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


# ---------------------------------------------------- 3-D segmentation (M4) ----


def _robust_neighbor_distance(frame: Frame) -> float:
    """Upper-quartile nearest-neighbour distance.

    In a two-phase frame the median that ``typical_neighbor_distance`` uses is
    dragged down by the close contacts of the disordered phase, which starves
    the neighbour graph on the ordered side (a planted solid-liquid frame
    disconnects its lattice this way). The upper quartile tracks the phase
    that carries the spatial order instead, and stays sensible for a single
    phase as well."""
    from scipy.spatial import cKDTree
    L = frame.cell_diag
    pos = np.mod(frame.pos, L)
    pos = np.minimum(pos, L * (1 - 1e-9))  # dialect-exempt: strict upper edge for KD trees
    d, _ = cKDTree(pos, boxsize=L).query(pos, k=2)
    return float(np.percentile(d[:, 1], 75))  # dialect-exempt: robust quartile, not a threshold


def phase_labels_3d(frame: Frame, dialect) -> np.ndarray:
    """True 3-D phase segmentation: seeded region growing on the bond graph.

    (a) seeds: the solid mask (the dialect q6 rule) marks where the solid
    phase is beyond doubt; (b) the solid phase floods across the geometric
    neighbour graph (radius = segment_vote_factor x typical nearest-neighbour
    distance, as in ``smooth_labels``) into atoms whose order parameter lies
    in the intermediate band segment3d_band_lo_factor..segment3d_band_hi_factor
    (fractions of q6_solid); (c) atoms still unlabelled (order parameter below
    the band) are called by a neighbourhood majority over the flooded labels
    (segment3d_majority); (d) the boolean solid labels are returned. Because
    the phase grows along bonds, the interface may run along any direction --
    no box axis is special, unlike the 1-D z-profile phase call of the slab
    lifter, which needs the interface perpendicular to z."""
    from collections import deque
    from scipy.spatial import cKDTree

    rc = float(dialect.threshold("q6_cutoff"))
    thr = float(dialect.threshold("q6_solid"))
    lo = float(dialect.threshold("segment3d_band_lo_factor")) * thr
    hi = float(dialect.threshold("segment3d_band_hi_factor")) * thr
    L = frame.cell_diag
    pos = np.mod(frame.pos, L)
    pos = np.minimum(pos, L * (1 - 1e-9))  # dialect-exempt: strict upper edge for KD trees
    q6, _, _ = qbar(pos, L, rc=rc)
    factor = float(dialect.threshold("segment_vote_factor"))
    rc_vote = factor * _robust_neighbor_distance(frame)
    nbrs = cKDTree(pos, boxsize=L).query_ball_point(pos, rc_vote)

    labels = np.zeros(len(pos), bool)
    seeds = q6 > thr
    labels[seeds] = True
    queue = deque(np.where(seeds)[0].tolist())
    grow = (q6 > lo) & (q6 <= hi)      # the intermediate band the solid may absorb
    while queue:
        for j in nbrs[queue.popleft()]:
            if not labels[j] and grow[j]:
                labels[j] = True
                queue.append(j)
    # (c) residual vote, decided on the flooded labels only: order-independent
    majority = float(dialect.threshold("segment3d_majority"))
    flooded = labels.copy()
    for i in np.where(~flooded)[0]:
        nb = [j for j in nbrs[i] if j != i]
        if nb and flooded[nb].mean() >= majority:
            labels[i] = True
    return labels


def interface_mesh(labels: np.ndarray, frame: Frame, dialect) -> dict:
    """Interface object for M4: boundary atoms and their normals.

    Interface atoms are the union of every unlike adjacent pair on the same
    neighbour graph the segmentation floods across. The normal of an
    interface atom points from its liquid neighbourhood towards its solid
    neighbourhood: centroid of the solid neighbours minus centroid of the
    liquid neighbours, on minimum-image displacements, normalised. Atoms
    whose normal cannot be oriented (one side empty, or a degenerate centroid
    difference) keep a zero normal. Returns a dict with the interface atom
    ``indices``, their ``normals`` (unit vectors, sign towards the solid),
    and the two radii (``radius`` of the contact graph, ``normal_radius`` of
    the centroid neighbourhoods)."""
    from scipy.spatial import cKDTree

    L = frame.cell_diag
    pos = np.mod(frame.pos, L)
    pos = np.minimum(pos, L * (1 - 1e-9))  # dialect-exempt: strict upper edge for KD trees
    d = _robust_neighbor_distance(frame)
    radius = float(dialect.threshold("segment_vote_factor")) * d
    normal_radius = float(dialect.threshold("segment3d_normal_factor")) * d
    tree = cKDTree(pos, boxsize=L)
    nbr_mesh = tree.query_ball_point(pos, radius)
    nbr_norm = tree.query_ball_point(pos, normal_radius)

    indices, normals = [], []
    for i in range(len(pos)):
        contacts = [j for j in nbr_mesh[i] if j != i]
        if not any(labels[j] != labels[i] for j in contacts):
            continue                      # bulk atom: not on the phase boundary
        indices.append(i)
        nb = [j for j in nbr_norm[i] if j != i]
        disp = pos[nb] - pos[i]
        disp -= L * np.round(disp / L)    # minimum image
        solid = labels[nb]
        normal = np.zeros(3)
        if solid.any() and not solid.all():
            v = disp[solid].mean(axis=0) - disp[~solid].mean(axis=0)
            norm = np.linalg.norm(v)
            if norm > 1e-9:               # dialect-exempt: degenerate-normal guard
                normal = v / norm
        normals.append(normal)
    return dict(indices=np.array(indices, int),
                normals=np.array(normals, float).reshape(len(indices), 3),
                radius=radius, normal_radius=normal_radius)
