"""Extended defects: dislocation (Volterra field) and CSL grain boundaries.

The dislocation builder applies the isotropic edge-displacement field to a
perfect lattice; the Burgers circuit stays exact away from the core. The
grain-boundary builder joins two fcc grains misoriented by the [001] CSL
angle theta = 2 atan(n/m) on a box both rotated lattices tile exactly
(Sigma = m^2 + n^2, halved by the fcc half-cell translations when m, n are
both odd)."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from ..io.frames import Frame
from ..lang.errors import ChaordError
from ..lang.ir import RegionBlock, Statement
from .crystal import _num, _stmt_map
from .prototypes import PROTOTYPES, basis, cell_matrix


def build_dislocation(region: RegionBlock, system: dict, dialect, rng) -> Frame:
    """Edge dislocation via the Volterra field: b along x, line along z."""
    stmts = _stmt_map(region)
    name = stmts.get("lattice", stmts.get("prototype")).values[0].text if (
        "lattice" in stmts or "prototype" in stmts) else "fcc"
    a = _num(stmts["a"].values[0])
    reps = (6, 6, 4)

    from .crystal import build_conventional
    frame = build_conventional(name, {"a": a}, ("Cu",), reps)
    L = frame.cell_diag
    b = a / np.sqrt(2)                      # fcc a/2<110>
    x0, y0 = L[0] / 2, L[1] / 2
    x, y = frame.pos[:, 0] - x0, frame.pos[:, 1] - y0
    r2 = x**2 + y**2
    core = float(dialect.threshold("dislocation_core_radius"))
    safe = r2 > core**2
    # isotropic edge field: u_x = (b/2pi) atan2(...), u_y log term; inside the
    # core radius the singular field is left unset (atoms keep lattice positions)
    ux = np.zeros_like(x)
    uy = np.zeros_like(y)
    # pure tilt term: exact far-field closure of b per circuit
    ux[safe] = (b / (2 * np.pi)) * np.arctan2(y[safe], x[safe])  # dialect-exempt: exact-geometry
    uy[safe] = 0.0  # dialect-exempt: numerical-guard: edge dislocation has no y tilt
    pos = frame.pos.copy()
    pos[:, 0] += ux
    pos[:, 1] += uy
    pos = np.mod(pos, L)
    return Frame(pos=pos, cell=frame.cell, symbols=frame.symbols, pbc=frame.pbc)


def build_dislocation_exact(region: RegionBlock, system: dict, dialect, rng) -> Frame:
    """Edge dislocation by exact half-plane removal (no displacement field).

    fcc cell oriented x=[110], y=[1-10], z=[001]: slip plane (1-10), line [001],
    Burgers vector b = a/2[110] = (a/sqrt(2)) x̂.  Along x the fcc (110) planes
    stack with spacing d = a/(2*sqrt(2)) in two alternating sublattices, so a
    full stacking period 2d is exactly one lattice translation b along x.  The
    construction removes the two half-planes of one period (slots just below
    the core) from the upper half y > y_cut and relabels the remaining
    upper-half planes left of the core onto the vacated slots: every retained
    atom lands on an ideal lattice site, so the Burgers circuit closes to
    exactly b everywhere away from the core and the boundary gap (a periodic
    cell necessarily holds the dipole partner at the y boundary).  With the
    line along +z and the half-plane deficit in the upper half, the Burgers
    vector is +a/2[110], i.e. +b x̂ in this frame."""
    stmts = _stmt_map(region)
    name = stmts.get("lattice", stmts.get("prototype")).values[0].text if (
        "lattice" in stmts or "prototype" in stmts) else "fcc"
    if name != "fcc":
        raise ChaordError("the exact half-plane construction is written for fcc")
    a = _num(stmts["a"].values[0])
    reps = (6, 4, 6)

    from .crystal import build_conventional
    T = np.array([[1, 1, 0], [1, -1, 0], [0, 0, 1]], int)   # x [110], y [1-10], z [001]
    frame = build_conventional(name, {"a": a}, ("Cu",), reps, T)
    R = T / np.sqrt(np.sum(T * T, axis=1))[:, None]          # orthonormal frame axes
    L = np.linalg.norm(frame.cell, axis=1)
    pos = np.mod(frame.pos @ R.T, L)

    d = a / (2 * np.sqrt(2))                                 # (110) interplane spacing along x
    b = a / np.sqrt(2)                                       # a/2[110] along x
    n_planes = int(round(L[0] / d))                          # (110) half-plane slots per row
    n_levels = int(round(L[1] / d))                          # atomic y levels
    slot = np.mod(np.round(pos[:, 0] / d).astype(int), n_planes)
    level = np.mod(np.round(pos[:, 1] / d).astype(int), n_levels)
    core_slot = n_planes // 2                                # dislocation core at box centre
    top = level >= n_levels // 2                             # half above the slip-plane cut

    # remove one stacking period (two adjacent half-planes) just left of the
    # core in the upper half; shift the upper half further left by exactly b
    delete = top & (slot >= core_slot - 2) & (slot <= core_slot - 1)
    shift = top & (slot <= core_slot - 3)
    pos[shift, 0] += b
    pos = np.mod(pos, L)
    keep = ~delete
    symbols = [s for s, k in zip(frame.symbols, keep) if k]
    return Frame(pos=pos[keep], cell=np.diag(L), symbols=symbols,
                 pbc=frame.pbc)


# integer [001] misorientations theta = 2 atan(n/m) between the grains;
# Sigma = m^2 + n^2, halved by the fcc half-cell translations when m and n
# are both odd (the classic [001] CSL set for fcc)
CSL_MN = {
    5: (3, 1),
    13: (5, 1),
    17: (4, 1),
}


def _csl_axis_period(m: int, n: int) -> int:
    """Smallest half-lattice period p (in a/2 units) along a cube axis that
    the +theta rotation maps back onto the fcc lattice, so the box edge
    p*a/2 is an exact period of BOTH grains and wrapping the rotated grain
    is a pure lattice translation.

    In the (a/2) integer frame the -theta rotation is
    [[m^2-n^2, 2mn], [-2mn, m^2-n^2]] / (m^2+n^2); a cube-axis vector (p, 0)
    survives the rotation as a lattice point (even coordinate sum) only for
    the periods found here.  p = 2*(m^2+n^2) always qualifies, bounding the
    search."""
    D = m * m + n * n
    M, N = m * m - n * n, 2 * m * n
    for p in range(1, 2 * D + 1):
        if (M * p) % D or (N * p) % D:
            continue
        if (M * p // D - N * p // D) % 2 == 0:
            return p
    raise ChaordError(               # unreachable: p = 2D passes both tests
        f"no cube-axis period for CSL (m, n) = ({m}, {n})")


def build_grain_boundary(region: RegionBlock, system: dict, dialect, rng) -> Frame:
    """Bicrystal: two rigid fcc grains misoriented by the [001] CSL angle.

    The lower half sits on the cube axes; the upper half is the same fcc
    lattice rigidly rotated by +theta = 2 atan(n/m) about z, so the relative
    misorientation is exactly the CSL angle (Sigma per the table).  The box
    edges along x and y are the smallest cube-axis periods common to both
    grains (p*a/2 from _csl_axis_period): the rotated grain tiles the box
    exactly and wrapping can neither squeeze nor duplicate a site -- the
    Review-2 bug was wrapping the rotated grain into a box (8a) it does not
    tile, overlapping 37% of the atoms.  The halves are joined at
    z = box height / 2; upper-grain atoms that land within the dialect's
    overlap fraction of d_NN of the lower grain are removed (the standard
    unrelaxed-GB overlap cleanup), so no pair sits below that floor.  The
    far-field orientations are exact; the boundary region itself is left
    for the physics prior to relax, as in the bench protocol."""
    stmts = _stmt_map(region)
    name = stmts.get("lattice", stmts.get("prototype")).values[0].text if (
        "lattice" in stmts or "prototype" in stmts) else "fcc"
    if name != "fcc":
        raise ChaordError("the [001] CSL construction is written for fcc")
    a = _num(stmts["a"].values[0])
    sigma = int(_num(next(v for s in region.statements if s.key == "grain_boundary"
                          for v in s.values if v.t == "q")))
    if sigma not in CSL_MN:
        raise ChaordError(f"no [001] CSL cell for Sigma {sigma}; known: {sorted(CSL_MN)}")
    m, n = CSL_MN[sigma]
    theta = 2 * np.arctan2(n, m)
    p = _csl_axis_period(m, n)
    n_cells = 6                                     # box height in conventional cells (12 z-layers)
    L = np.array([p * a / 2, p * a / 2, n_cells * a])

    reach = 2 * p + 2                               # index window covering one box translate after rotation
    idx = np.arange(-reach, reach + 1)

    def _slab(kmin: int, kmax: int) -> np.ndarray:
        """fcc points (a/2)(i, j, k) with i+j+k even and k in [kmin, kmax)."""
        I, J, K = np.meshgrid(idx, idx, np.arange(kmin, kmax), indexing="ij")
        I, J, K = I.ravel(), J.ravel(), K.ravel()
        keep = (I + J + K) % 2 == 0
        return (a / 2) * np.stack([I[keep], J[keep], K[keep]], axis=1).astype(float)

    def _wrap_unique(pts: np.ndarray) -> np.ndarray:
        """Wrap into the commensurate box and drop translation duplicates.

        The box edges are exact lattice periods, so the over-generated
        window holds each box site more than once; duplicates differ by fp
        noise only (distinct sites of one grain are d_NN apart), and the
        wrap edge 0 ~ L is clamped onto 0 before the dedupe grid."""
        q = np.mod(pts, L)
        q[q > L - 1e-6] = 0.0  # dialect-exempt: numerical-guard: wrap-edge aliasing onto the origin
        key = np.round(q * 1e5).astype(np.int64)  # dialect-exempt: numerical-guard: 1e-5 A dedupe grid
        _, sel = np.unique(key, axis=0, return_index=True)
        q = q[np.sort(sel)]
        return q[np.lexsort((q[:, 2], q[:, 1], q[:, 0]))]

    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    lower = _wrap_unique(_slab(0, n_cells))
    upper = _wrap_unique(_slab(n_cells, 2 * n_cells) @ rot.T)

    dnn = a / np.sqrt(2)  # dialect-exempt: exact-geometry: fcc a/2<110>
    floor = float(dialect.threshold("gb_overlap_fraction")) * dnn
    tree_lo = cKDTree(lower, boxsize=L)
    overlapped = np.array(
        [len(js) > 0 for js in tree_lo.query_ball_point(upper, floor)])
    upper = upper[~overlapped]

    pos = np.vstack([lower, upper])
    pos = pos[np.lexsort((pos[:, 2], pos[:, 1], pos[:, 0]))]
    # defensive: commensuration plus the overlap cleanup guarantees the
    # floor; a violation means the CSL table or the period search broke
    dmin = float(cKDTree(pos, boxsize=L).query(pos, k=2)[0][:, 1].min())
    if dmin < floor - 1e-9:  # dialect-exempt: numerical-guard: fp slack on the overlap floor
        raise ChaordError(
            f"Sigma {sigma} construction left a pair at {dmin:.3f} A, below "
            f"the dialect overlap floor {floor:.3f} A")
    return Frame(pos=pos, cell=np.diag(L), symbols=["Cu"] * len(pos),
                 pbc=(True, True, True))
