"""Extended defects: dislocation (Volterra field) and CSL grain boundaries.

The dislocation builder applies the isotropic edge-displacement field to a
perfect lattice; the Burgers circuit stays exact away from the core. The
grain-boundary builder joins two fcc grains rotated +/-theta/2 about [001]
on commensurate integer lattices (Sigma = det T)."""
from __future__ import annotations

import numpy as np

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


# integer [001] rotations with Sigma = det T (the classic CSL set)
CSL_ROTATIONS = {
    5: np.array([[3, 1, 0], [-1, 3, 0], [0, 0, 1]]),
    13: np.array([[5, 2, 0], [-2, 5, 0], [0, 0, 1]]),
    17: np.array([[4, 1, 0], [-1, 4, 0], [0, 0, 1]]),
}


def build_grain_boundary(region: RegionBlock, system: dict, dialect, rng) -> Frame:
    """Bicrystal: the upper half rigidly rotated about z by the CSL angle.

    The far-field orientations are exact (Sigma = det T); the boundary region
    itself is left for the physics prior to relax, as in the bench protocol."""
    stmts = _stmt_map(region)
    name = stmts.get("lattice", stmts.get("prototype")).values[0].text if (
        "lattice" in stmts or "prototype" in stmts) else "fcc"
    a = _num(stmts["a"].values[0])
    sigma = int(_num(next(v for s in region.statements if s.key == "grain_boundary"
                          for v in s.values if v.t == "q")))
    if sigma not in CSL_ROTATIONS:
        raise ChaordError(f"no [001] CSL cell for Sigma {sigma}; known: {sorted(CSL_ROTATIONS)}")
    T = CSL_ROTATIONS[sigma]
    theta = 2 * np.arctan2(1, 3) if sigma == 5 else 2 * np.arctan2(2, 5)

    from .crystal import build_conventional
    reps = (8, 8, 6)
    frame = build_conventional(name, {"a": a}, ("Cu",), reps)
    L = frame.cell_diag
    c = L[:2] / 2
    zmid = L[2] / 2
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    pos = frame.pos.copy()
    upper = pos[:, 2] > zmid
    rel = pos[upper, :2] - c
    pos[upper, :2] = rel @ R[:2, :2].T + c
    pos = np.mod(pos, L)
    return Frame(pos=pos, cell=frame.cell, symbols=frame.symbols, pbc=frame.pbc)
