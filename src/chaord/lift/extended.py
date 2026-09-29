"""Extended-defect lifting: Burgers circuits and CSL grain boundaries."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from ..io.frames import Frame
from ..lang.errors import ChaordError
from ..realize.lj import mic


def _wrap(pos, L):
    """Positions strictly inside [0, L): cKDTree boxsize-safe."""
    return np.minimum(np.mod(pos, L), L * (1 - 1e-9))  # dialect-exempt: numerical-guard: strict upper edge for KD trees


def _family_and_a(frame: Frame, pos, L, tree, dnn, dialect):
    """fcc/bcc family from interior coordination, a from d_NN, |b| of the family."""
    from .defects import _A_FROM_DNN
    cn_cut = float(dialect.threshold("slab_cn_factor")) * dnn
    zmid0 = 0.5 * L[2]  # dialect-exempt: numerical-guard: box mid-plane
    interior = pos[np.abs(pos[:, 2] - zmid0) < 0.3 * zmid0]  # dialect-exempt: numerical-guard: interior sampling band (middle 30% of the box)
    cn = float(np.mean([len(x) - 1 for x in tree.query_ball_point(interior, cn_cut)]))
    family = "fcc" if cn >= float(dialect.threshold("fcc_cn_min")) else "bcc"
    a = _A_FROM_DNN[family] * dnn
    b_fam = a / np.sqrt(2) if family == "fcc" else a * np.sqrt(3) / 2  # dialect-exempt: exact-geometry
    return family, a, b_fam


def _derived_basis(pos, L, tree, dnn, family, a, dialect):
    """Primitive basis from the frame's own neighbour vectors (orientation-free).

    The classic cubic-axes table only generates the lattice when the frame axes
    are cubic; a real frame may be oriented arbitrarily.  Any three linearly
    independent nearest-neighbour lattice vectors of fcc/bcc span a primitive
    cell, so the first such triple (in a deterministic order) is the basis."""
    cn_cut = float(dialect.threshold("slab_cn_factor")) * dnn
    counts = tree.query_ball_point(pos, cn_cut, return_length=True)
    i0 = int(np.argmax(counts))
    vecs = np.array([mic(pos[j] - pos[i0], L)
                     for j in tree.query_ball_point(pos[i0], cn_cut) if j != i0])
    if len(vecs) < 3:
        return None
    order = np.lexsort((vecs[:, 2], vecs[:, 1], vecs[:, 0],
                        np.linalg.norm(vecs, axis=1)))
    vecs = vecs[order]
    v1 = vecs[0]
    v2 = next((v for v in vecs[1:] if np.linalg.norm(np.cross(v, v1)) > 1e-8), None)  # dialect-exempt: numerical-guard: degenerate-vector guard
    if v2 is None:
        return None
    v3 = next((v for v in vecs[2:]
               if abs(np.dot(np.cross(v1, v2), v)) > 1e-8), None)  # dialect-exempt: numerical-guard: coplanarity guard
    if v3 is None:
        return None
    B = np.array([v1, v2, v3], float)
    # primitive-volume check against the family (fcc a^3/4, bcc a^3/2)
    want = (a ** 3) / 4 if family == "fcc" else (a ** 3) / 2  # dialect-exempt: exact-geometry
    if abs(abs(np.linalg.det(B)) - want) > 1e-6 * want:  # dialect-exempt: numerical-guard: fp tolerance on the primitive volume
        return None
    return B


def _mean_offset_vector(pos, L, tree, family, a, B, dialect):
    """2 x (mean site offset of the top band - bottom band): Volterra-grade fields.

    The smooth atan2 tilt puts the upper half at +b/4 mean offset from its
    sites and the lower half at -b/4; a perfect translation (exact half-plane
    construction) leaves every atom on a site and this estimator sees ~0, which
    is why the row-step estimator below exists."""
    invB = np.linalg.inv(B)
    frac = pos @ invB
    base = np.floor(frac)
    offs = np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1],
                                indexing="ij")).reshape(3, -1).T
    best = np.full(len(pos), np.inf)
    shift = np.zeros_like(pos)
    for o in offs:
        cand = (base + o) @ B
        d = pos - cand
        d -= L * np.round(d / L)
        n2 = np.einsum("ij,ij->i", d, d)
        take = n2 < best
        best[take] = n2[take]
        shift[take] = d[take]

    floor = float(dialect.threshold("burgers_detect_min")) * a
    for axis in (0, 1):
        mid = 0.5 * L[axis]  # dialect-exempt: numerical-guard: box mid-plane
        band = 0.2 * L[axis]  # dialect-exempt: numerical-guard: top/bottom sampling band width
        top = shift[pos[:, axis] > mid + band]
        bottom = shift[pos[:, axis] < mid - band]
        if len(top) < 10 or len(bottom) < 10:
            continue
        b2d = 2.0 * (top.mean(axis=0) - bottom.mean(axis=0))  # dialect-exempt: exact-geometry
        if float(np.linalg.norm(b2d[:2])) >= floor:
            return np.array([b2d[0], b2d[1], 0.0])  # dialect-exempt: numerical-guard: zero z of a line along z
    return None


def _axis_lattice_repeat(nn, axis, dialect):
    """Shortest lattice vector parallel to `axis` from NN vectors and pair sums."""
    cand = list(nn) + [nn[i] + nn[j] for i in range(len(nn)) for j in range(i + 1, len(nn))]
    tol = float(dialect.threshold("burgers_row_angle_tol"))
    keep = []
    for v in cand:
        n = np.linalg.norm(v)
        if n < 1e-8 or v[axis] <= 0:  # dialect-exempt: numerical-guard: degenerate / wrong sense
            continue
        if abs(v[axis]) / n > np.cos(np.radians(tol)):
            keep.append(v)
    return min(keep, key=np.linalg.norm) if keep else None


def _rows_along(pos, L, tree, t, link_tol):
    """Atom rows parallel to the repeat t: union-find over links ~ n*t."""
    n_at = len(pos)
    parent = np.arange(n_at)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for k in (1, 2, 3):  # bridge gaps up to three repeats wide
        kt = k * t
        groups = tree.query_ball_point(pos, np.linalg.norm(kt) + link_tol)
        for i, js in enumerate(groups):
            for j in js:
                if j <= i:
                    continue
                dv = mic(pos[j] - pos[i], L) - kt
                if np.linalg.norm(dv) < link_tol:
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[ri] = rj
    roots = np.array([find(i) for i in range(n_at)])
    rows = {}
    for i in range(n_at):
        rows.setdefault(int(roots[i]), []).append(i)
    return roots, rows


def _two_means(v):
    """Means of the two value clusters of a step profile (1-D 2-means)."""
    thr = 0.5 * (v.min() + v.max())  # dialect-exempt: numerical-guard: midpoint of the value range
    lo, hi = v[v <= thr], v[v > thr]
    m1 = float(lo.mean()) if len(lo) else float(v.mean())
    m2 = float(hi.mean()) if len(hi) else float(v.mean())
    return m1, m2


def _row_step_vector(pos, L, tree, dnn, b_fam, dialect):
    """Burgers vector from matched-row offset profiles between adjacent rows.

    A perfect dislocation displaces the far field by lattice translations, so
    per-atom site offsets vanish; the signal is topological and only a
    matched-order comparison of the atomic rows above and below the slip plane
    sees it: the row above the slip plane is short by b worth of planes, and
    the m-th atom of one row sits on the (m + n)-th slot of the other.  The
    profile of matched offsets is then a step (or a constant) of height b along
    the slip direction; staggered (triangular-stack) neighbours only ever give
    half-integer multiples of b and are rejected by the family band."""
    link_tol = float(dialect.threshold("burgers_row_link_fraction")) * dnn
    band = float(dialect.threshold("burgers_step_band"))
    cn_cut = float(dialect.threshold("slab_cn_factor")) * dnn
    counts = tree.query_ball_point(pos, cn_cut, return_length=True)
    i0 = int(np.argmax(counts))
    nn = np.array([mic(pos[j] - pos[i0], L)
                   for j in tree.query_ball_point(pos[i0], cn_cut) if j != i0])
    if len(nn) < 3:
        return None

    best = None
    for axis in (0, 1):
        t = _axis_lattice_repeat(nn, axis, dialect)
        if t is None:
            continue
        roots, rows = _rows_along(pos, L, tree, t, link_tol)
        scan = 1 - axis
        row_pairs = set()
        for i, j in tree.query_pairs(cn_cut, output_type="ndarray"):
            if roots[i] != roots[j]:
                row_pairs.add(tuple(sorted((int(roots[i]), int(roots[j])))))
        accepted = []
        for ri, rj in row_pairs:
            mem_i, mem_j = np.array(rows[ri]), np.array(rows[rj])
            dy = float(np.mean(pos[mem_j][:, scan]) - np.mean(pos[mem_i][:, scan]))
            dy -= L[scan] * round(dy / L[scan])
            below, above = (mem_i, mem_j) if dy > 0 else (mem_j, mem_i)
            # with the line along +z, b = +x(slip) holds when the half-plane
            # deficit is in the upper half; interfaces where the upper row has
            # MORE atoms than the lower measure the dipole partner
            if len(above) > len(below):
                continue
            A = np.sort(pos[below][:, axis])
            Bv = np.sort(pos[above][:, axis])
            m = min(len(A), len(Bv))
            for sh in (0, len(A) - m):
                dq = Bv[:m] - A[sh:sh + m]
                dq -= L[axis] * np.round(dq / L[axis])
                for mv in _two_means(dq):
                    if abs(abs(mv) - b_fam) <= band * b_fam:
                        accepted.append(float(mv))
        if accepted:
            med = float(np.median(accepted))
            if best is None or abs(med) > best[0]:
                v = np.zeros(3)
                v[axis] = med
                best = (abs(med), v)
    return best[1] if best else None


def burgers_vector(frame: Frame, dialect) -> np.ndarray | None:
    """Edge Burgers vector of an unrelaxed construction (line along z).

    Two estimators run on the same frame: the mean site-offset difference of
    the two halves (sensitive to smooth Volterra fields) and the matched-row
    offset step between rows adjacent across the slip plane (sensitive to exact
    half-plane constructions, where every atom sits on a lattice site).  The
    larger above the detection floor is reported with its direction."""
    from ..build.defects import typical_neighbor_distance
    L = frame.cell_diag
    dnn = typical_neighbor_distance(frame)
    pos = _wrap(frame.pos, L)
    tree = cKDTree(pos, boxsize=L)
    family, a, b_fam = _family_and_a(frame, pos, L, tree, dnn, dialect)

    B = _derived_basis(pos, L, tree, dnn, family, a, dialect)
    if B is None:  # orientation assumption of the classic table (cubic axes)
        if family == "fcc":
            B = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]], float) * (a / 2)
        else:
            B = np.array([[1, 1, 1], [1, -1, 1], [1, 1, -1]], float) * (a / 2)

    candidates = []
    mean_vec = _mean_offset_vector(pos, L, tree, family, a, B, dialect)
    if mean_vec is not None:
        candidates.append(("mean-offset", mean_vec))
    step_vec = _row_step_vector(pos, L, tree, dnn, b_fam, dialect)
    if step_vec is not None:
        candidates.append(("row-step", step_vec))
    floor = float(dialect.threshold("burgers_detect_min")) * a
    candidates = [(k, v) for k, v in candidates if float(np.linalg.norm(v)) >= floor]
    if not candidates:
        return None
    method, vec = max(candidates, key=lambda kv: float(np.linalg.norm(kv[1])))
    # unrelaxed smooth fields fix the magnitude family, not the exact direction
    return dict(magnitude=float(np.linalg.norm(vec)),
                family="<110>" if family == "fcc" else "<111>",
                vector=vec, method=method)


def _folded_shell_angles(half, L, dnn, dialect):
    """First-shell in-plane bond angles of one grain, folded into [0, 45] deg.

    Every atom of the half contributes, not one reference atom: the pair set
    of a KD-tree radius query is a function of the geometry alone, so no sort
    order or tie-breaking enters anywhere (the single-reference enumeration
    was BLAS-sensitive through exactly that tie-break).  In-plane vectors
    connect atoms of one z layer and each layer belongs wholly to one grain,
    so even boundary-adjacent atoms cannot mix the two lattices into one
    angle; the fold is mod 90 (the [001] four-fold period) into half a
    quadrant, its fundamental domain."""
    cn_cut = float(dialect.threshold("slab_cn_factor")) * dnn
    tree = cKDTree(_wrap(half, L), boxsize=L)
    pairs = tree.query_pairs(cn_cut, output_type="ndarray")
    d = half[pairs[:, 1]] - half[pairs[:, 0]]
    d -= L * np.round(d / L)
    in_plane = np.abs(d[:, 2]) < 0.2 * np.linalg.norm(d, axis=1)  # dialect-exempt: numerical-guard: in-plane filter (out-of-plane shell rejection)
    a = np.degrees(np.arctan2(d[in_plane, 1], d[in_plane, 0])) % 90.0  # dialect-exempt: numerical-guard: fold into one [001] quadrant
    return np.where(a > 45.0, 90.0 - a, a)  # dialect-exempt: numerical-guard: fold into half a quadrant


def _dominant_angle(angs):
    """Peak position of the folded-angle histogram, refined inside the bin.

    fcc/bcc [001] first shells are discrete directions, so a grain's angles
    pile into one sharp mode; the histogram peak (not a correlation search)
    locates it, and the mean inside the peak bin recovers sub-bin resolution."""
    if len(angs) < 20:
        return None
    edges = np.arange(0, 46)  # integer grid: exact bin edges over [0, 45] deg
    hist, _ = np.histogram(angs, bins=edges)
    k = int(np.argmax(hist))
    centre = (edges[k] + edges[k + 1]) / 2  # dialect-exempt: numerical-guard: peak-bin centre
    sel = angs[np.abs(angs - centre) <= 1]  # dialect-exempt: numerical-guard: one-bin refinement window, deg
    return float(sel.mean()) if len(sel) else None


def grain_boundary_sigma(frame: Frame, dialect) -> int | None:
    """Misorientation between z-half grains -> the [001] CSL Sigma.

    The first-shell in-plane bond angles of each grain form a four-fold
    distribution (fcc/bcc [001]: a single peak at 45 deg); a CSL rotation
    rigidly shifts one grain's whole distribution, so the offset between the
    two histogram peaks is the misorientation.  All atoms of each half
    contribute, which makes the estimate a property of the geometry rather
    than of one reference atom's neighbour sort order (the cause of the old
    Linux/BLAS flake)."""
    L = frame.cell_diag
    zmid = 0.5 * L[2]  # dialect-exempt: numerical-guard: box mid-plane
    pos = _wrap(frame.pos, L)
    lower = pos[pos[:, 2] < zmid]
    upper = pos[pos[:, 2] >= zmid]
    if len(lower) < 20 or len(upper) < 20:
        return None
    from ..build.defects import typical_neighbor_distance
    dnn = typical_neighbor_distance(frame)
    pa = _dominant_angle(_folded_shell_angles(lower, L, dnn, dialect))
    pb = _dominant_angle(_folded_shell_angles(upper, L, dnn, dialect))
    if pa is None or pb is None:
        return None
    theta = abs(pa - pb)
    # CSL [001]: theta = 2 atan(n/m), Sigma = m^2 + n^2 (coprime m, n)
    from math import gcd
    tol = float(dialect.threshold("csl_angle_tol"))
    best = None
    for m in range(1, 16):
        for n in range(1, 16):
            if gcd(m, n) != 1:
                continue
            if (m, n) == (1, 1):
                continue  # the identity rotation (Sigma 1): true of any crystal
            target = np.degrees(2 * np.arctan(n / m)) % 90.0  # dialect-exempt: numerical-guard: fold into one [001] quadrant
            if target > 45.0:  # dialect-exempt: numerical-guard: fold into half a quadrant
                target = 90.0 - target  # dialect-exempt: numerical-guard: fold into half a quadrant
            if abs(theta - target) < tol:
                # fcc/bcc [001]: the coincident-site lattice gains the
                # half-cell translations, so both-odd (m, n) halves the Sigma
                sigma = (m * m + n * n) // 2 if (m % 2 and n % 2) else m * m + n * n
                if best is None or abs(theta - target) < best[1]:
                    best = (sigma, abs(theta - target))
    return best[0] if best else None
