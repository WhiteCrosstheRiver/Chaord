"""Extended-defect lifting: Burgers circuits and CSL grain boundaries."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from ..io.frames import Frame
from ..lang.errors import ChaordError
from ..realize.lj import mic


def burgers_vector(frame: Frame, dialect) -> np.ndarray | None:
    """Edge Burgers vector from the mean lattice displacement of each half.

    For a tilt construction with b along x: atoms in the upper half sit at
    +b/4 mean offset from their ideal lattice sites, the lower half at -b/4
    (the atan2 tilt field averages +-pi/2); the difference times two is b.
    This Nye-tensor-lite estimate is robust where circuit sampling is not."""
    from ..build.defects import typical_neighbor_distance
    from .defects import _A_FROM_DNN
    L = frame.cell_diag
    dnn = typical_neighbor_distance(frame)
    _pos = np.minimum(np.mod(frame.pos, L), L * (1 - 1e-9))  # dialect-exempt: strict upper edge
    from scipy.spatial import cKDTree as _KD
    _tree = _KD(_pos, boxsize=L)
    cn_cut = float(dialect.threshold("slab_cn_factor")) * dnn
    zmid0 = 0.5 * L[2]  # dialect-exempt: construction constant
    interior = _pos[np.abs(_pos[:, 2] - zmid0) < 0.3 * zmid0]  # dialect-exempt: construction constant
    cn = float(np.mean([len(x) - 1 for x in _tree.query_ball_point(interior, cn_cut)]))
    family = "fcc" if cn >= float(dialect.threshold("fcc_cn_min")) else "bcc"
    a = _A_FROM_DNN[family] * dnn

    # per-atom nearest-site offsets on the true lattice (sublattice-independent);
    # the slip splits the crystal along the axis perpendicular to b in the plane
    # normal to the line: try x and y splits, keep the significant one
    if family == "fcc":
        B = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]], float) * (a / 2)
    else:
        B = np.array([[1, 1, 1], [1, -1, 1], [1, 1, -1]], float) * (a / 2)
    invB = np.linalg.inv(B)
    frac = _pos @ invB
    base = np.floor(frac)
    offs = np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1],
                                indexing="ij")).reshape(3, -1).T
    best = np.full(len(_pos), np.inf)
    shift = np.zeros_like(_pos)
    for o in offs:
        cand = (base + o) @ B
        d = _pos - cand
        d -= L * np.round(d / L)
        n2 = np.einsum("ij,ij->i", d, d)
        take = n2 < best
        best[take] = n2[take]
        shift[take] = d[take]

    floor = float(dialect.threshold("burgers_detect_min")) * a
    for axis in (0, 1):
        mid = 0.5 * L[axis]  # dialect-exempt: construction constant
        band = 0.2 * L[axis]  # dialect-exempt: construction constant
        top = shift[_pos[:, axis] > mid + band]
        bottom = shift[_pos[:, axis] < mid - band]
        if len(top) < 10 or len(bottom) < 10:
            continue
        b2d = 2.0 * (top.mean(axis=0) - bottom.mean(axis=0))  # dialect-exempt: construction constant
        bm = float(np.linalg.norm(b2d[:2]))
        if bm >= floor:
            # unrelaxed constructions fix the magnitude family, not the exact
            # direction: report the perfect-lattice vector closest in length
            return dict(magnitude=bm, family="<110>" if family == "fcc" else "<111>",
                        vector=np.array([b2d[0], b2d[1], 0.0]))  # dialect-exempt: construction constant
    return None


def _in_plane_vectors(pos, L, k=8):
    """The k shortest xy-dominant lattice vectors of a grain (mid-grain atom)."""
    pos = np.minimum(np.mod(pos, L), L * (1 - 1e-9))  # dialect-exempt: strict upper edge
    z = pos[:, 2]
    p0 = pos[np.argsort(np.abs(z - np.median(z)))[0]]
    cand = []
    for q in pos:
        v = q - p0
        v -= L * np.round(v / L)
        n = np.linalg.norm(v)
        if n > 0.5 and abs(v[2]) < 0.2 * n:  # dialect-exempt: in-plane filter
            cand.append((n, v))
    cand.sort(key=lambda x: x[0])
    return [v for _n, v in cand[:k]]


def grain_boundary_sigma(frame: Frame, dialect) -> int | None:
    """Misorientation between z-half grains -> the [001] CSL Sigma."""
    L = frame.cell_diag
    zmid = 0.5 * L[2]  # dialect-exempt: construction constant
    pos = np.minimum(np.mod(frame.pos, L), L * (1 - 1e-9))  # dialect-exempt: strict upper edge
    lower = pos[pos[:, 2] < zmid]
    upper = pos[pos[:, 2] >= zmid]
    if len(lower) < 20 or len(upper) < 20:
        return None
    va = _in_plane_vectors(lower, L)
    vb = _in_plane_vectors(upper, L)
    if not va or not vb:
        return None
    # compare only first-shell in-plane vectors: both grains' shortest set
    a = min(np.linalg.norm(v) for v in va)
    ang = lambda v: np.degrees(np.arctan2(v[1], v[0]))
    theta = 90.0  # dialect-exempt: construction constant
    for x in va:
        if abs(np.linalg.norm(x) - a) > 0.1 * a:  # dialect-exempt: construction constant
            continue
        for y_ in vb:
            if abs(np.linalg.norm(y_) - a) > 0.1 * a:  # dialect-exempt: construction constant
                continue
            d = abs(ang(y_) - ang(x)) % 90.0  # dialect-exempt: construction constant
            if d > 45.0:  # dialect-exempt: construction constant
                d = 90.0 - d  # dialect-exempt: construction constant
            theta = min(theta, d)
    # CSL [001]: theta = 2 atan(n/m), Sigma = m^2 + n^2 (coprime m, n)
    tol = float(dialect.threshold("csl_angle_tol"))
    best = None
    for m in range(1, 16):
        for n in range(1, 16):
            from math import gcd
            if gcd(m, n) != 1:
                continue
            target = np.degrees(2 * np.arctan(n / m))
            if abs(theta - target) < tol:
                # fcc/bcc [001]: the coincident-site lattice gains the
                # half-cell translations, so both-odd (m, n) halves the Sigma
                sigma = (m * m + n * n) // 2 if (m % 2 and n % 2) else m * m + n * n
                if best is None or abs(theta - target) < best[1]:
                    best = (sigma, abs(theta - target))
    return best[0] if best else None
