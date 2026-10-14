"""Region geometry membership: which atoms a region's geometry contains.

The single membership definition for every region shape (RD1,
docs/reviews/research_decisions.md). Build-side consumers (composition,
conserve, the nanotube placement) and any checker that needs "is this atom
in this region" read it here, so a region behaves the same wherever it is
evaluated. Geometry numbers are in the frame's own unit system (A or sigma),
exactly as the slab builder reads `slab z lo .. hi`.

Rules:
- boundaries are inclusive: an atom exactly at a radius, sphere radius, box
  edge or axial end plane is inside;
- distances to a cylinder axis use the minimum image in the two directions
  perpendicular to the axis, so a tube hugging a periodic boundary is
  measured through it;
- a slab (and a finite cylinder's axial window) wraps through the periodic
  boundary when its low bound exceeds its high bound -- the wrapped-slab
  convention the slab lifter prints (e.g. `slab z 42.5 .. 21.6`);
- `rest` has no standalone membership: it is the complement of the union of
  the program's OTHER regions, which only the caller knows. Pass that mask
  via `rest=`; without it a chain containing `rest` fails closed.
"""
from __future__ import annotations

import numpy as np

from ..lang.errors import ChaordError
from ..lang.ir import GeoChain

_AXES = ("x", "y", "z")

# the finite cylinder's axial window is centred on the stated axis
# coordinate; when the third centre number is omitted the window is centred
# on the coordinate origin (the axis-infinite form's convention)
_AXIS_ORIGIN = 0.0  # dialect-exempt: exact-geometry: the coordinate origin


def _num(q) -> float:
    """Numeric value of a Quantity token (fractions such as 1/3 supported)."""
    s = q.num if hasattr(q, "num") else str(q)
    if "/" in s:
        a, b = s.split("/")
        return float(a) / float(b)
    return float(s)


def _axis_index(axis: str) -> int:
    try:
        return _AXES.index(axis)
    except ValueError:
        raise ChaordError(f"unknown geometry axis {axis!r}") from None


def _wrapped_member(values: np.ndarray, lo: float, hi: float,
                    L: float) -> np.ndarray:
    """Interval membership on one periodic axis; lo > hi wraps (slab rule)."""
    span = np.mod(hi - lo, L)
    d = np.mod(values - lo, L)
    return d <= span


def shape_mask(shape, pos: np.ndarray, cell_diag: np.ndarray,
               rest=None) -> np.ndarray:
    """Boolean mask: which of `pos` (N, 3) lie inside one shape.

    `rest` is the mask of atoms already claimed by the program's other
    regions; a `rest` shape evaluates to its complement and fails closed
    when it was not supplied."""
    pos = np.asarray(pos, float)
    L = np.asarray(cell_diag, float)
    t = shape.t
    if t == "all":
        return np.ones(len(pos), bool)
    if t == "rest":
        if rest is None:
            raise ChaordError(
                "the 'rest' shape has no standalone membership: pass the "
                "mask of the other regions' atoms (rest=...) to evaluate it")
        return ~np.asarray(rest, bool)
    if t == "slab":
        i = _axis_index(shape.axis)
        lo, hi = _num(shape.rng.lo), _num(shape.rng.hi)
        return _wrapped_member(pos[:, i], lo, hi, L[i])
    if t == "box":
        out = np.ones(len(pos), bool)
        for rng, vals, Li in zip((shape.xs, shape.ys, shape.zs),
                                 (pos[:, 0], pos[:, 1], pos[:, 2]), L):
            out &= _wrapped_member(vals, _num(rng.lo), _num(rng.hi), Li)
        return out
    if t == "sphere":
        c = np.array([_num(shape.cx), _num(shape.cy), _num(shape.cz)])
        r2 = _num(shape.radius) ** 2
        return np.sum((pos - c) ** 2, axis=1) <= r2
    if t == "cylinder":
        i = _axis_index(shape.axis)
        perp = [j for j in range(3) if j != i]
        # the two centre numbers are the perpendicular coordinates in axis
        # order (x, y for axis z; y, z for axis x; x, z for axis y); the
        # optional third number is the axis coordinate of the centre
        centre = {perp[0]: _num(shape.cx), perp[1]: _num(shape.cy)}
        d2 = np.zeros(len(pos), float)
        for j in perp:
            dj = pos[:, j] - centre[j]
            dj -= L[j] * np.round(dj / L[j])   # minimum image, through the wall
            d2 += dj * dj
        inside = d2 <= _num(shape.radius) ** 2
        if shape.length is not None:
            cz = (_num(shape.cz) if shape.cz is not None else _AXIS_ORIGIN)
            ln = _num(shape.length)
            inside &= _wrapped_member(pos[:, i], cz - ln / 2, cz + ln / 2,
                                      L[i])
        return inside
    raise ChaordError(f"unknown geometry shape {t!r}")  # pragma: no cover


def geometry_mask(geometry: GeoChain, pos: np.ndarray, cell_diag: np.ndarray,
                  rest=None) -> np.ndarray:
    """Boolean mask of a full geometry expression: shapes joined by
    `and` (intersect), `or` (union), `minus` (subtract), evaluated left to
    right -- the order they are printed in."""
    if not geometry.parts:
        raise ChaordError("empty geometry has no membership")
    out = shape_mask(geometry.parts[0], pos, cell_diag, rest=rest)
    for op, part in zip(geometry.ops, geometry.parts[1:]):
        m = shape_mask(part, pos, cell_diag, rest=rest)
        if op == "and":
            out = out & m
        elif op == "or":
            out = out | m
        elif op == "minus":
            out = out & ~m
        else:  # pragma: no cover - the grammar only allows and/or/minus
            raise ChaordError(f"unknown geometry operator {op!r}")
    return out
