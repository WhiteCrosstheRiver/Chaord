"""Nanotube chirality inference: a carbon frame -> (n, m, length).

RD2 (docs/reviews/research_decisions.md), lift half. This inverts the ideal
construction of build/nanotube.py (Saito, Dresselhaus & Dresselhaus,
"Physical Properties of Carbon Nanotubes", Imperial College Press 1998,
ch. 3):

1. the tube axis is the cell axis the frame extends along -- the covariance
   eigenvector of the LARGEST eigenvalue, snapped to a cell axis, refused
   when no cell axis dominates (a tube whose axis is not a cell axis is
   honestly refused, never given a wrong chirality);
2. every atom must sit on one radius about that axis (the exact circle fit);
   an fcc carbon frame or a flat graphene sheet is refused here -- fail
   closed;
3. the tube is unrolled to the sheet (x = R*phi, y = axial, each wrapped by
   its period) and the graphene lattice is recovered from the data itself:
   the nearest-neighbour distance d_NN gives a = sqrt(3) d_NN, and the
   second-neighbour shell (distance a) holds the six primitive lattice
   vectors +-a1, +-a2, +-(a1-a2);
4. the zigzag direction is the lattice direction nearest the circumference;
   (n, m) are the integer coordinates of the circumferential lattice vector
   Ch in the (zigzag, +60-degree) basis -- canonical 0 <= m <= n by
   construction;
5. |T| is the shortest axial lattice period -- the primitive vector
   T = p*u + q*w of the recovered lattice whose circumferential component
   cancels (T.Ch = 0 identically for the coprime (p, q) = ((2m+n),
   (2n+m))/d_R, Saito/Dresselhaus ch. 3); the built length is
   round(L_axis/|T|) x |T|.

Consistency gates before returning: |Ch| = a sqrt(n^2+nm+m^2) with the a
recovered in step 3, |T| = sqrt(3)|Ch|/d_R with d_R = gcd(2m+n, 2n+m), and
the atom count 4 N N_T / d_R equals the frame's.

Deviation from the record, recorded here rather than silently edited into
it: RD2 sketches the axis as "the covariance eigenvector with the smallest
eigenvalue". That names the tube axis only for a tube shorter than
sqrt(6) R ~ 2.45 R; for any longer tube the axial direction carries the
LARGEST variance (L^2/12 against R^2/2), and a smallest-eigenvalue pick
lands in the perpendicular plane. Every tube this package builds and every
acceptance frame is far longer than 2.45 R, so the largest eigenvector is
used; the snap and radius gates refuse either way when the choice is wrong.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import gcd

import numpy as np

from ..lang.errors import ChaordError

# Numerical guards for ideal-geometry frames. Every tolerance scales with the
# graphene lattice constant recovered from the frame itself, so none depends
# on the system under test (Review 8 rule). Thermally distorted tubes are
# deliberately not inferred: the classifier research decides when a region
# may be *called* a tube (RD1's lift note); until then this is exact
# geometry, and anything the gates cannot confirm exactly is refused.
_RADIUS_TOL = 1e-6  # dialect-exempt: numerical-guard: relative single-radius residual
_SHELL_TOL = 5e-2   # dialect-exempt: numerical-guard: shell isolation window (a against 1.15a)
_INT_TOL = 1e-3     # dialect-exempt: numerical-guard: integer-recovery slack, in units of a
_COUNT_TOL = 1e-3   # dialect-exempt: numerical-guard: dimensionless integer-coordinate slack


@dataclass(frozen=True)
class NanotubeChirality:
    """The inferred chirality (n, m) and the commensurate built length."""
    n: int
    m: int
    length: float


def infer_nanotube_chirality(positions, cell) -> NanotubeChirality:
    """(n, m, length) of an ideal nanotube frame, or a refusal.

    `positions` is the (N, 3) carbon frame in its own unit system, `cell`
    the diagonal lengths or the full (orthogonal) 3x3 cell."""
    L = _cell_diag(cell)
    pos = np.mod(np.asarray(positions, float), L)
    # np.mod(-1e-15, L) returns exactly L (verified for this repo's box
    # lengths); every downstream periodic container is half-open [0, box),
    # so clamp the wrapped coordinates just under the upper face -- the
    # repo's standing wrap idiom (bench sanity, cv helpers)
    pos = np.minimum(pos, L * (1 - 1e-9))
    if pos.ndim != 2 or pos.shape[1] != 3 or len(pos) < 2:
        raise ChaordError("a nanotube frame needs an (N, 3) position array")
    axis = _tube_axis(pos)
    perp = [j for j in range(3) if j != axis]
    x, y, R = _unroll(pos, L, axis, perp)
    ch = 2 * np.pi * R
    a, shell = _lattice_from_data(x, y, ch, L[axis])
    n, m, u, w = _chirality_indices(a, shell, ch)
    t_len = _axial_period(u, w, ch, a)
    n_t = int(round(L[axis] / t_len))
    _gates(n, m, a, ch, t_len, n_t, len(pos))
    return NanotubeChirality(n=n, m=m, length=n_t * t_len)


def _cell_diag(cell) -> np.ndarray:
    """The three box lengths of an orthogonal cell (3-vector or 3x3)."""
    c = np.asarray(cell, float)
    if c.ndim == 1 and c.shape == (3,):
        return c
    if c.ndim == 2 and c.shape == (3, 3) and not np.any(c - np.diag(np.diag(c))):
        return np.diag(c)
    raise ChaordError("infer_nanotube_chirality needs a 3-vector or an "
                      "orthogonal 3x3 cell")


def _tube_axis(pos: np.ndarray) -> int:
    """The cell axis the frame extends along (module docstring, step 1)."""
    centred = pos - pos.mean(axis=0)
    cov = (centred.T @ centred) / len(pos)
    v = np.linalg.eigh(cov)[1][:, -1]       # largest eigenvalue's eigenvector
    axis = int(np.argmax(np.abs(v)))
    if 2 * v[axis] ** 2 <= 1:
        # the extension direction is not dominated by one cell axis: a tube
        # lying diagonally in its box has no honest chirality in this cell
        raise ChaordError("the frame's extension direction does not align "
                          "with a cell axis: not a nanotube in this cell")
    return axis


def _unroll(pos: np.ndarray, L: np.ndarray, axis: int, perp) -> tuple:
    """(x, y, R): the tube unrolled to the sheet and the tube radius.

    The exact circle fit (linear least squares on |q - c|^2 = r^2) recovers
    the axis point in the perpendicular plane; a frame whose atoms do not
    all sit on one radius is refused. The perpendicular coordinates are
    re-anchored on one atom with the minimum image so a tube that straddles
    a periodic wall is still measured coherently."""
    q = pos[:, perp]
    dq = q - q[0]
    dq -= L[perp] * np.round(dq / L[perp])
    q = q[0] + dq
    A = np.column_stack([2 * q, np.ones(len(q))])
    centre, *_ = np.linalg.lstsq(A, np.sum(q * q, axis=1), rcond=None)
    d = q - centre[:2]
    for j in range(2):
        d[:, j] -= L[perp[j]] * np.round(d[:, j] / L[perp[j]])
    r = np.linalg.norm(d, axis=1)
    R = float(r.mean())
    if R <= 0 or r.max() - r.min() > _RADIUS_TOL * R:
        raise ChaordError("the atoms do not sit on one radius about the "
                          "axis: not a nanotube frame")
    x = np.mod(R * np.arctan2(d[:, 1], d[:, 0]), 2 * np.pi * R)
    x = np.minimum(x, 2 * np.pi * R * (1 - 1e-9))   # np.mod can return the box
    y = np.mod(pos[:, axis] - pos[0, axis], L[axis])
    y = np.minimum(y, L[axis] * (1 - 1e-9))         # exactly L (see above)
    return x, y, R


def _lattice_from_data(x, y, ch, L_axis) -> tuple:
    """(a, shell): the graphene lattice constant and the six primitive
    lattice vectors of the second-neighbour shell, in unrolled coordinates
    recovered from the frame itself (module docstring, step 3)."""
    from scipy.spatial import cKDTree
    pts = np.column_stack([x, y])
    box = np.array([ch, L_axis])
    tree = cKDTree(pts, boxsize=box)
    d_nn = tree.query(pts, k=2)[0][:, 1].min()
    if d_nn <= 0:
        raise ChaordError("coincident positions: not a nanotube frame")
    a = np.sqrt(3) * d_nn
    shell = []
    for j in tree.query_ball_point(pts[0], a * (1 + _SHELL_TOL)):
        if j == 0:
            continue
        s = pts[j] - pts[0]
        s -= box * np.round(s / box)
        if abs(np.linalg.norm(s) - a) < _SHELL_TOL * a:
            shell.append(s)
    if len(shell) != 6:
        raise ChaordError(f"the second-neighbour shell holds {len(shell)} "
                          "vectors, not the 6 of a graphene sheet: not a "
                          "nanotube frame")
    return a, shell


def _chirality_indices(a, shell, ch) -> tuple:
    """(n, m, u, w): the integer coordinates of the circumferential lattice
    vector Ch in the (zigzag, +60-degree) basis (module docstring, step 4).

    The zigzag direction u is the shell vector nearest the circumference;
    the +60-degree partner w is chosen among the two candidates as the side
    that makes 0 <= m <= n, which is the canonical form by construction."""
    u = max(shell, key=lambda s: s[0])
    target = np.array([ch, 0])
    cos60 = np.cos(np.pi / 3)
    un = np.linalg.norm(u)
    for s in shell:
        if abs(float(u @ s) / (un * np.linalg.norm(s)) - cos60) > _SHELL_TOL:
            continue
        nf, mf = np.linalg.lstsq(np.column_stack([u, s]), target, rcond=None)[0]
        if (abs(nf - round(nf)) > _COUNT_TOL
                or abs(mf - round(mf)) > _COUNT_TOL):
            continue
        n, m = int(round(nf)), int(round(mf))
        if not (n >= m >= 0 and n >= 1):
            continue
        if np.linalg.norm(n * u + m * s - target) < _INT_TOL * a:
            return n, m, u, s
    raise ChaordError("the circumferential lattice vector is no integer "
                      "combination of the recovered basis: not a nanotube "
                      "frame")


def _axial_period(u, w, ch, a) -> float:
    """|T|: the shortest axial lattice period (module docstring, step 5).

    The primitive axial vector is T = p*u - q*w with the smallest coprime
    (p, q) whose circumferential components cancel (the exact solutions of
    p*u_x = q*w_x are the integer multiples of ((2m+n), (2n+m))/d_R, so the
    first integer hit is the primitive T). Measured through the recovered
    lattice rather than raw column offsets: opposite-sublattice atoms can
    share a circumferential coordinate (they do for every zigzag tube) and
    would shorten a naive column measurement."""
    ratio = float(u[0] / w[0])
    bound = int(3 * ch / a) + 3
    for p in range(1, bound):
        q = round(p * ratio)
        if q >= 1 and abs(p * ratio - q) < _COUNT_TOL:
            return float(np.linalg.norm(p * u - q * w))
    raise ChaordError("no axial lattice period found: not a nanotube frame")


def _gates(n: int, m: int, a: float, ch: float, t_len: float, n_t: int,
           n_atoms: int):
    """The record's consistency gates: fail closed, never a wrong chirality."""
    N = n * n + n * m + m * m
    if abs(ch - a * np.sqrt(N)) > _INT_TOL * a:
        raise ChaordError(f"|Ch| = {ch:.6f} does not match the frame's own "
                          f"a*sqrt(N) with ({n},{m}): not a nanotube frame")
    dr = gcd(2 * m + n, 2 * n + m)
    if abs(t_len - np.sqrt(3) * ch / dr) > _INT_TOL * a:
        raise ChaordError(f"|T| = {t_len:.6f} does not match "
                          f"sqrt(3)|Ch|/d_R with ({n},{m}): not a nanotube "
                          "frame")
    if 4 * N * n_t % dr or n_atoms != 4 * N * n_t // dr:
        raise ChaordError(f"the frame holds {n_atoms} atoms but ({n},{m}) "
                          f"over {n_t} periods builds {4 * N * n_t // dr}: "
                          "not a nanotube frame")
