"""Nanotube builder: chirality (n, m) + length -> exact carbon positions.

RD2 (docs/reviews/research_decisions.md). The construction is the standard
ideal one (Saito, Dresselhaus & Dresselhaus, "Physical Properties of Carbon
Nanotubes", Imperial College Press 1998, ch. 3): the graphene sheet
(a = 2.46 A, a1 = a(1,0), a2 = a(1/2, sqrt(3)/2), two-atom basis (0,0) and
(a1+a2)/3) is cut by the cell spanned by the chiral vector
Ch = n a1 + m a2 and N_T copies of the translation vector T = t1 a1 + t2 a2
(t1 = (2m+n)/d_R, t2 = -(2n+m)/d_R, d_R = gcd(2m+n, 2n+m)) and rolled onto a
cylinder of radius R = |Ch|/(2 pi). `length L` is a target: the tube is
commensurate, so the built length is the nearest whole number N_T of
translation periods and the statement, the cell and the coordinates always
agree. Atom count = 4 (n^2 + nm + m^2) N_T / d_R, checked exactly.

The axis and the centre come from the region's cylinder geometry when it has
one (the tube sits inside its region); otherwise axis z at the cell centre.
A stated cell must fit the tube and match the axial period within
`lattice_match_tolerance` (fail closed); with no stated cell the box is
(d + 2 x nanotube_vacuum_margin) square and the built length tall.
"""
from __future__ import annotations

from math import gcd

import numpy as np

from ..io.frames import Frame
from ..lang.errors import ChaordError
from ..lang.ir import RegionBlock, Statement

GRAPHENE_A = 2.46  # dialect-exempt: exact-geometry: graphene lattice constant, A
# (Saito/Dresselhaus ch. 3; same class of constant as the prototype tables)

# dialect-exempt: exact-geometry: the graphene sheet basis, fractional-free
_BASIS = (np.zeros(2), np.array([1, 1]) / 3)  # in (a1, a2) units


def graphene_vectors():
    """Graphene primitive vectors a1, a2 (rows) in A."""
    a1 = GRAPHENE_A * np.array([1, 0])
    a2 = GRAPHENE_A * np.array([1, np.sqrt(3)]) / 2
    return a1, a2


def chirality_geometry(n: int, m: int):
    """(Ch, T, d_R) of the (n, m) tube in graphene lattice coordinates:
    the chiral and translation vectors (2-D, A) and their gcd d_R."""
    a1, a2 = graphene_vectors()
    dr = gcd(2 * m + n, 2 * n + m)
    ch = n * a1 + m * a2
    t = ((2 * m + n) * a1 - (2 * n + m) * a2) / dr
    return ch, t, dr


def nanotube_positions(n: int, m: int, n_periods: int, axis: str = "z",
                       centre=(None, None, None)):
    """Ideal carbon positions of the (n, m) tube, `n_periods` translation
    periods long, rolled along `axis` about `centre` (None = 0).

    Returns (pos, built_length, diameter): pos is (4(n^2+nm+m^2) N_T/d_R, 3);
    the axial extent `built_length` is centred on the centre's axial
    coordinate, exactly as the region cylinder's `length` is (RD1)."""
    if not (n >= m >= 0 and n >= 1):
        raise ChaordError(
            f"nanotube chirality must be canonical (n >= m >= 0, n >= 1), "
            f"got ({n},{m})")
    ch, t, dr = chirality_geometry(n, m)
    ch_len = float(np.linalg.norm(ch))
    t_len = float(np.linalg.norm(t))
    R = ch_len / (2 * np.pi)
    ch_hat = ch / ch_len
    t_hat = t / t_len
    t1, t2 = (2 * m + n) // dr, -((2 * n + m) // dr)
    built = n_periods * t_len
    atoms_per_period = 4 * (n * n + n * m + m * m) // dr
    a1, a2 = graphene_vectors()

    # lattice points of one (Ch, N_T x T) cell: u a1 + v a2 with the
    # (Ch, T) projections inside the half-open cell. The preimage of that
    # cell in (u, v) space is the parallelogram spanned by (n, m) and
    # N_T (t1, t2) -- cover its four corners with a 2-cell margin. A
    # fractional pad keeps boundary points deterministically on one side
    # (exact arithmetic has them at exactly 0 or the full length)
    pad = ch_len * 1e-9  # dialect-exempt: exact-geometry: boundary tiebreak
    t1, t2 = (2 * m + n) // dr, -((2 * n + m) // dr)
    corners = [(0, 0), (n, m), (n_periods * t1, n_periods * t2),
               (n + n_periods * t1, m + n_periods * t2)]
    us = [c[0] for c in corners]
    vs = [c[1] for c in corners]
    sheet = []
    for u in range(min(us) - 2, max(us) + 3):
        for v in range(min(vs) - 2, max(vs) + 3):
            for b in _BASIS:
                p = u * a1 + v * a2 + b[0] * a1 + b[1] * a2
                px, py = float(p @ ch_hat), float(p @ t_hat)
                if -pad <= px < ch_len - pad and -pad <= py < built - pad:
                    sheet.append((px, py))
    if len(sheet) != atoms_per_period * n_periods:
        raise ChaordError(  # pragma: no cover - the enumeration is exact
            f"nanotube ({n},{m}) enumeration found {len(sheet)} atoms, "
            f"expected {atoms_per_period * n_periods}")

    i = "xyz".index(axis)
    perp = [j for j in range(3) if j != i]
    c = [0 if k is None else float(k) for k in centre]
    u1 = np.zeros(3)
    u1[perp[0]] = 1
    u2 = np.zeros(3)
    u2[perp[1]] = 1
    ax = np.zeros(3)
    ax[i] = 1
    pos = np.empty((len(sheet), 3))
    for k, (px, py) in enumerate(sheet):
        phi = 2 * np.pi * px / ch_len
        pos[k] = (c + R * np.cos(phi) * u1 + R * np.sin(phi) * u2
                  + (py - built / 2) * ax)
    return pos, built, 2 * R


def _num(v) -> float:
    s = v.num if hasattr(v, "num") else str(v)
    if "/" in s:
        a, b = s.split("/")
        return float(a) / float(b)
    return float(s)


def _nanotube_statement(region: RegionBlock) -> Statement:
    stmts = [s for s in region.statements
             if s.key == "nanotube" and s.kind in ("build", "state")]
    if len(stmts) != 1:
        raise ChaordError(
            f"region {region.name!r}: needs exactly one nanotube statement")
    return stmts[0]


def _parse_statement(stmt: Statement):
    """((n, m), target length) from `nanotube (n, m) length L [unit]`."""
    chir = next((v for v in stmt.values if v.t == "chir"), None)
    if chir is None:
        raise ChaordError(
            "nanotube statement needs a chirality pair: nanotube (n, m) ...")
    li = next((k for k, v in enumerate(stmt.values)
               if v.t == "n" and v.text == "length"), None)
    if li is None or li + 1 >= len(stmt.values) or stmt.values[li + 1].t != "q":
        raise ChaordError(
            "nanotube statement needs a length: nanotube (n, m) length L A")
    return (chir.n, chir.m), _num(stmt.values[li + 1])


def _region_placement(region: RegionBlock, diameter: float, built: float,
                      cell_vals, dialect):
    """(axis, centre, cell_diag) for the tube: the region's own cylinder
    geometry when it has one, else axis z at the cell centre."""
    shp = None
    geo = region.geometry
    if len(geo.parts) == 1 and geo.parts[0].t == "cylinder":
        shp = geo.parts[0]
    tol = float(dialect.threshold("lattice_match_tolerance"))
    axis = shp.axis if shp is not None else "z"
    i = "xyz".index(axis)

    if shp is not None:
        # the region's length is its description of the tube and must match
        # what the chirality builds; its radius may be the tube's or larger
        # (a wider pore around the tube), never smaller (fail closed)
        if shp.length is not None and abs(_num(shp.length) - built) > tol:
            raise ChaordError(
                f"region {region.name!r}: cylinder length {_num(shp.length)} "
                f"does not match the nanotube's built length {built:.6f}")
        if _num(shp.radius) < diameter / 2 - tol:
            raise ChaordError(
                f"region {region.name!r}: cylinder radius {_num(shp.radius)} "
                f"is smaller than the nanotube radius {diameter / 2:.6f}")
        centre = np.zeros(3)
        perp = [j for j in range(3) if j != i]
        centre[perp[0]] = _num(shp.cx)
        centre[perp[1]] = _num(shp.cy)
        centre[i] = _num(shp.cz) if shp.cz is not None else built / 2
    else:
        if cell_vals is None:
            raise ChaordError(
                f"region {region.name!r}: a nanotube needs its cell from a "
                "system cell statement or a cylinder geometry")
        centre = np.asarray(cell_vals, float) / 2

    if cell_vals is None:
        margin = float(dialect.threshold("nanotube_vacuum_margin"))
        L = np.full(3, diameter + 2 * margin)
        L[i] = built
    else:
        L = np.asarray(cell_vals, float)
        if abs(L[i] - built) > tol:
            raise ChaordError(
                f"the stated cell is {L[i]:.6f} along {axis} but the "
                f"nanotube builds {built:.6f} (nearest whole number of "
                "translation periods)")
        for j in range(3):
            if j != i and L[j] < diameter:
                raise ChaordError(
                    f"the stated cell is {L[j]:.6f} along {'xyz'[j]}, narrower "
                    f"than the nanotube diameter {diameter:.6f}")
    return axis, tuple(centre), L


def build_nanotube_region(region: RegionBlock, system: dict, dialect,
                          rng=None) -> Frame:
    """The crystal region carrying a `nanotube` statement -> Frame."""
    del rng  # the construction is exact; there is nothing to sample
    (n, m), target = _parse_statement(_nanotube_statement(region))
    if not (n >= m >= 0 and n >= 1):
        raise ChaordError(
            f"nanotube chirality must be canonical (n >= m >= 0, n >= 1), "
            f"got ({n},{m})")
    ch, t, _dr = chirality_geometry(n, m)
    ch_len = float(np.linalg.norm(ch))
    t_len = float(np.linalg.norm(t))
    diameter = ch_len / np.pi
    n_periods = max(1, round(target / t_len))
    built = n_periods * t_len

    cell_vals = None
    if "cell" in system:
        vals = [_num(v) for v in system["cell"].values if v.t == "q"]
        auto = (system["cell"].values
                and system["cell"].values[0].t == "n"
                and system["cell"].values[0].text == "auto")
        if len(vals) == 3:
            cell_vals = vals
        elif not auto:
            raise ChaordError(
                "a nanotube region needs a numeric or `cell auto` system cell")

    axis, centre, L = _region_placement(region, diameter, built, cell_vals,
                                        dialect)
    pos, built, diameter = nanotube_positions(n, m, n_periods, axis=axis,
                                              centre=centre)

    species = "C"
    n_atoms = len(pos)
    if "atoms" in system:
        named = [v.text for v in system["atoms"].values if v.t == "n"]
        if named:
            species = named[0]
            stated = int(_num(next(v for v in system["atoms"].values
                                   if v.t == "q")))
            if stated != n_atoms:
                raise ChaordError(
                    f"conserve atoms {species} {stated} but the nanotube "
                    f"builds {n_atoms}")
    return Frame(pos=np.mod(pos, L), cell=np.diag(L),
                 symbols=[species] * n_atoms, pbc=(True, True, True))
