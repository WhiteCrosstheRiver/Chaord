"""Crystal prototypes: one registry, exact construction, no fitting.

Each prototype carries its conventional-cell basis (fractional positions with
species slots), free lattice parameters, and space group. Bases are given as
FULL conventional cells (fcc = 4 sites, fluorite = 12, ...). The rational
coordinates below are exact geometry, not thresholds, so the lines carrying
them have dialect-exempt tags per AGENTS.md. Builders construct from these
tables; the lifter identifies structures by matching against them in
Niggli-reduced fractional space.

The one fitting entry point is the orthohexagonal hcp setting at the bottom
(`fit_orthohexagonal_hcp`): the hexagonal registry entry above cannot be
recognised by symmetry on a thermally jittered frame shipped in an
orthorhombic box (spglib sees P1 at lift_symprec), so that setting is fitted
from the box lengths under the sqrt(3)-axis constraint and verified by site
matching -- same dialect gates as the Wigner-Seitz cubic fit.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Species slots: integers index into the composition's species list.


@dataclass
class Prototype:
    name: str                    # canonical name used in `prototype` statements
    family: str                  # lattice family keyword for `lattice` statements
    params: tuple                # free parameters, e.g. ("a",) or ("a", "c")
    spacegroup: int              # international number of the conventional cell
    default_slots: tuple = ()    # slot species when no composition is given
    n_atoms: int = 0             # atoms per conventional cell


PROTOTYPES: dict[str, Prototype] = {}


def _register(name, family, params, sg, slots, n):
    PROTOTYPES[name] = Prototype(name, family, params, sg, slots, n)
    return PROTOTYPES[name]


_register("sc", "sc", ("a",), 221, ("X",), 1)
_register("bcc", "bcc", ("a",), 229, ("X",), 2)
_register("fcc", "fcc", ("a",), 225, ("X",), 4)
_register("hcp", "hcp", ("a", "c"), 194, ("X",), 2)
_register("diamond", "diamond", ("a",), 227, ("X",), 8)
_register("rocksalt", "rocksalt", ("a",), 225, ("Na", "Cl"), 8)
_register("cscl", "cscl", ("a",), 221, ("Cs", "Cl"), 2)
_register("zincblende", "zincblende", ("a",), 216, ("Zn", "S"), 8)
_register("wurtzite", "wurtzite", ("a", "c"), 186, ("Zn", "S"), 4)
_register("fluorite", "fluorite", ("a",), 225, ("Ca", "F"), 12)
_register("perovskite", "perovskite", ("a",), 221, ("Sr", "Ti", "O"), 5)
_register("L1_2", "fcc", ("a",), 221, ("Ni", "Al"), 4)
_register("rutile", "rutile", ("a", "c"), 136, ("Ti", "O"), 6)

# dialect-exempt-begin: exact-geometry
_FCC = ((0.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5))
# dialect-exempt-end


def cell_matrix(name: str, params: dict) -> np.ndarray:
    if name in ("hcp", "wurtzite"):
        a, c = params["a"], params["c"]
        return np.array([[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, c]])
    if name == "rutile":
        return np.diag([params["a"], params["a"], params["c"]])
    return np.diag([params["a"]] * 3)


def basis(name: str, params: dict) -> tuple[np.ndarray, tuple]:
    pos, slots = _basis_raw(name, params)
    return np.mod(pos, 1.0), slots  # dialect-exempt: numerical-guard: fractional wrap to [0,1)


# dialect-exempt-begin: exact-geometry
def _basis_raw(name: str, params: dict) -> tuple[np.ndarray, tuple]:
    """(fractional positions, species slots) of one conventional cell."""
    if name == "sc":
        return np.zeros((1, 3)), (0,)
    if name == "bcc":
        return np.array([[0, 0, 0], [0.5, 0.5, 0.5]]), (0, 0)
    if name == "fcc":
        return np.array(_FCC), (0,) * 4
    if name == "hcp":
        return np.array([[0, 0, 0], [1 / 3, 2 / 3, 0.5]]), (0, 0)
    if name == "diamond":
        pos = [tuple(np.array(f) + np.array([0.25, 0.25, 0.25])) for f in _FCC]
        return np.array(list(_FCC) + pos), (0,) * 8
    if name == "rocksalt":
        return np.array(list(_FCC) + [tuple(np.array(f) + np.array([0.5, 0.5, 0.5])) for f in _FCC]), \
            (0,) * 4 + (1,) * 4
    if name == "cscl":
        return np.array([[0, 0, 0], [0.5, 0.5, 0.5]]), (0, 1)
    if name == "zincblende":
        pos = [tuple(np.array(f) + np.array([0.25, 0.25, 0.25])) for f in _FCC]
        return np.array(list(_FCC) + pos), (0,) * 4 + (1,) * 4
    if name == "wurtzite":
        u = float(params.get("u", 0.375))  # ideal wurtzite u (published crystallographic parameter)
        return (np.array([[1 / 3, 2 / 3, 0], [2 / 3, 1 / 3, 0.5],
                          [1 / 3, 2 / 3, u], [2 / 3, 1 / 3, 0.5 + u]]),
                (0, 0, 1, 1))
    if name == "fluorite":
        f1 = [tuple(np.array(f) + np.array([0.25, 0.25, 0.25])) for f in _FCC]
        f2 = [tuple(np.array(f) + np.array([0.75, 0.75, 0.75])) for f in _FCC]
        return np.array(list(_FCC) + f1 + f2), (0,) * 4 + (1,) * 8
    if name == "perovskite":
        return (np.array([[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]]),
                (0, 1, 2, 2, 2))
    if name == "L1_2":
        # corner = slot 1 (minority), faces = slot 0 (majority): Ni3Al has Al at corners
        return np.array(list(_FCC)), (1, 0, 0, 0)
    if name == "rutile":
        u = float(params.get("u", 0.305))  # TiO2 rutile u (published crystallographic parameter)
        return (np.array([[0, 0, 0], [0.5, 0.5, 0.5],
                          [u, u, 0], [1 - u, 1 - u, 0],
                          [0.5 + u, 0.5 - u, 0.5], [0.5 - u, 0.5 + u, 0.5]]),
                (0, 0, 1, 1, 1, 1))
    raise KeyError(f"unknown prototype {name!r}")
# dialect-exempt-end


# --------------------------------------------- the orthohexagonal hcp setting --
#
# The registry entry above lives in the hexagonal setting (gamma = 120 deg).
# The same lattice in the common simulation setting -- an ORTHORHOMBIC box on
# the axes x [100], y [1 2 0], z [001] -- has the rectangular conventional
# cell a x sqrt(3) a x c with 4 atoms (the orient rows [[1,0,0],[1,2,0],[0,0,1]]
# applied to the hexagonal cell; the bench's own hcp case is built that way).
# spglib cannot recover the hexagonal symmetry of a thermally jittered frame
# (the bench amplitude 0.06 d_NN leaves P1 at lift_symprec), so the lift fits
# this setting directly: a and c come from the box lengths under the sqrt(3)
# ratio constraint, and the identification is VERIFIED by matching every atom
# and every ideal site (the same lattice_fit_gate_min / site_match_tol_fraction
# gates the cubic Wigner-Seitz fit reads).

_SQRT3 = np.sqrt(3)


@dataclass
class OrthohcpFit:
    """A verified hcp-on-orthorhombic-box fit (all lengths in Angstrom)."""
    a: float
    c: float
    cell: tuple             # canonical box lengths (a-axis, sqrt(3)a-axis, c-axis)
    reps: tuple             # integer repeats of the orthohexagonal cell per axis
    n_sites: int            # ideal sites in the box (== frame atom count)
    sites_matched: float    # fraction of ideal sites with an atom within tol
    atom_coverage: float    # fraction of atoms sitting on a site within tol
    median_residual: float  # median atom-to-site distance after alignment, A


# dialect-exempt-begin: exact-geometry
# The orthohexagonal basis: the hexagonal basis {(0,0,0), (1/3,2/3,1/2)}
# plus the hexagonal-lattice translations inside the doubled orthohexagonal
# cell (a2 = (B - A)/2 contributes the (1/2,1/2,0) coset):
#   (0,0,0), (1/2,1/2,0), (0,1/3,1/2), (1/2,5/6,1/2)
_ORTHOHCP_BASIS = np.array([[0.0, 0.0, 0.0],
                            [0.5, 0.5, 0.0],
                            [0.0, 1 / 3, 0.5],
                            [0.5, 5 / 6, 0.5]])
# dialect-exempt-end


def orthohexagonal_hcp_basis() -> np.ndarray:
    """Fractional basis (4 sites) of the hcp orthohexagonal cell."""
    return _ORTHOHCP_BASIS.copy()


def _hcp_fit_thresholds(dialect):
    """The fit gates by name; None when the dialect defines no lattice-fit
    semantics (the arm then honestly refuses instead of guessing)."""
    from ..lang.errors import ChaordError
    try:
        return {key: float(dialect.threshold(key)) for key in (
            "lattice_scan_factor_lo", "lattice_scan_factor_hi",
            "lattice_fit_tol_fraction", "lattice_fit_gate_min",
            "site_match_tol_fraction")}
    except ChaordError:
        return None


def _ortho_canonical(frame):
    """Axis-aligned copy of an orthogonal cell (rotation invariance, rule 3);
    None for a non-orthogonal box (the hexagonal setting has no orthogonal
    description -- the spglib arm owns it)."""
    cell = np.asarray(frame.cell, float)
    L = np.linalg.norm(cell, axis=1)
    U = cell / L[:, None]
    if not np.allclose(U @ U.T, np.eye(3), atol=1e-8):  # dialect-exempt: numerical-guard: orthogonality residue of a rotated box
        return None
    pos = np.mod(frame.pos @ U.T, L)
    pos = np.minimum(pos, L * (1 - 1e-9))     # dialect-exempt: numerical-guard: strict upper edge for KD trees
    return pos, L


def _hcp_box_candidates(La, Lb, Lc, dnn, lo, hi, fit_tol):
    """Integer (n_a, n_q, n_c) assignments of the three box axes to the hcp
    vectors a, sqrt(3) a, c with consistent a estimates (permutation of the
    axes happens in the caller; here axis 0 carries a, axis 1 sqrt(3) a)."""
    out = []
    for n_a in range(max(1, int(La // (hi * dnn))), int(La // (lo * dnn)) + 1):
        a = La / n_a
        if not (lo * dnn <= a <= hi * dnn):
            continue
        for n_q in range(max(1, int(Lb // (hi * _SQRT3 * dnn))),
                         int(Lb // (lo * _SQRT3 * dnn)) + 1):
            a_cross = Lb / (_SQRT3 * n_q)
            if abs(a - a_cross) > fit_tol:
                continue        # the sqrt(3) axis does not agree on a
            for n_c in range(max(1, int(Lc // (hi * dnn))), int(Lc // (lo * dnn)) + 1):
                c = Lc / n_c
                if lo * dnn <= c <= hi * dnn:
                    out.append((n_a, n_q, n_c, a, c))
    return out


def _hcp_site_score(pos, L, a, c, reps, tol_frac, gate):
    """Anchor the ideal site lattice on the atoms and measure the match.

    The anchor shift starts at a spread atom's position and is then refined
    to its fixed point (each pass adds the mean minimum-image residual of
    covered atoms), so the alignment is exact rather than seed-luck: the
    correct lattice then reads the pure thermal jitter while a wrong
    candidate keeps its systematic offset. Returns (sites_matched,
    atom_coverage, median residual) when BOTH directions clear the gate
    (every site near an atom AND every atom near a site -- the coverage
    direction kills half-density sublattice degeneracies), else None.
    Deterministic: anchor seeds are the four lexicographically spread
    atoms, best score wins."""
    from scipy.spatial import cKDTree
    ortho = np.diag([a, _SQRT3 * a, c])
    grid = np.array(np.meshgrid(*[range(r) for r in reps],
                                indexing="ij")).reshape(3, -1).T
    cart = _ORTHOHCP_BASIS @ ortho
    sites = (cart[None, :, :] + grid[:, None, :] @ ortho).reshape(-1, 3)
    # tolerance anchored to the FITTED lattice's own nearest-neighbour
    # distance, never to the jittered atom cloud (review-2 lesson)
    dnn_lat = min(a, float(np.sqrt(a * a / 3 + c * c / 4)))  # dialect-exempt: exact-geometry: hcp nearest-neighbour factor
    tol = tol_frac * dnn_lat
    atoms = cKDTree(pos, boxsize=L)
    order = np.lexsort((pos[:, 2], pos[:, 1], pos[:, 0]))
    picks = pos[order[np.linspace(0, len(pos) - 1, 4).astype(int)]]

    def _wrapped(p):
        w = np.mod(p, L)
        return np.minimum(w, L * (1 - 1e-9))  # dialect-exempt: numerical-guard: strict upper edge

    def _score(shift):
        for _ in range(8):
            shifted = _wrapped(sites + shift)
            stree = cKDTree(shifted, boxsize=L)
            d_atom, i_site = stree.query(pos)
            covered = d_atom < tol
            if not covered.any():
                return None
            resid = pos[covered] - shifted[i_site[covered]]
            resid -= L * np.round(resid / L)
            step = resid.mean(axis=0)
            shift = shift + step
            if float(np.linalg.norm(step)) < 1e-9:  # dialect-exempt: numerical-guard: fixed-point convergence
                break
        shifted = _wrapped(sites + shift)
        stree = cKDTree(shifted, boxsize=L)
        d_site, _ = atoms.query(shifted)            # per site: nearest atom
        d_atom, _ = stree.query(pos)                # per atom: nearest site
        near_site = d_site < tol
        covered = d_atom < tol
        if min(float(near_site.mean()), float(covered.mean())) < gate:
            return None
        # median residual: robust against the one large thermal excursion
        return float(near_site.mean()), float(covered.mean()), float(np.median(d_atom))

    best = None
    for s0 in picks:
        got = _score(s0.copy())
        if got is not None and (best is None or (-min(got[0], got[1]), got[2])
                                < (-min(best[0], best[1]), best[2])):
            best = got
    return best


def fit_orthohexagonal_hcp(frame, dialect):
    """Fit an hcp lattice to a single-species frame in an orthogonal box.

    Measures a and c from the box (the box IS the orthohexagonal supercell,
    so its axis lengths are exact up to the box's own precision), tries every
    assignment of the three axes to a / sqrt(3) a / c, and keeps the
    assignment whose ideal sites match the atoms under the dialect's fit
    gates. Returns an OrthohcpFit with the CANONICAL axis order
    (a, sqrt(3) a, c) -- the emitted program states that order, so the text
    is invariant under permutations of the input box axes -- or None when
    nothing fits (the cascade then tries the next arm)."""
    from .defects import typical_neighbor_distance
    keys = _hcp_fit_thresholds(dialect)
    if keys is None:
        return None
    if len(set(frame.symbols)) != 1:
        return None    # unary hosts only: ordered multi-species hexagonal
                       # structures (wurtzite) need sublattice fitting
    canon = _ortho_canonical(frame)
    if canon is None:
        return None
    pos, L = canon
    # measure d_NN on the wrapped copy: builder tilings can leave atoms at
    # -1e-16 (the keep window of build_conventional), which KD trees with
    # `boxsize` reject outright
    from ..io.frames import Frame
    dnn = typical_neighbor_distance(Frame(
        pos=pos, cell=np.diag(L), symbols=frame.symbols, pbc=frame.pbc))
    n_atoms = len(pos)
    best = None
    for c_axis in range(3):
        for a_axis in range(3):
            if a_axis == c_axis:
                continue
            q_axis = 3 - a_axis - c_axis
            for n_a, n_q, n_c, a, c in _hcp_box_candidates(
                    L[a_axis], L[q_axis], L[c_axis], dnn,
                    keys["lattice_scan_factor_lo"],
                    keys["lattice_scan_factor_hi"],
                    keys["lattice_fit_tol_fraction"] * dnn):
                # defect-free crystal: the site count is the atom count
                # (a vacancy/interstitial host must refuse here)
                if 4 * n_a * n_q * n_c != n_atoms:
                    continue
                got = _hcp_site_score(pos, L, a, c, (n_a, n_q, n_c),
                                      keys["site_match_tol_fraction"],
                                      keys["lattice_fit_gate_min"])
                if got is None:
                    continue
                key = (-min(got[0], got[1]), got[2])
                if best is None or key < best[0]:
                    best = (key, OrthohcpFit(
                        a=a, c=c,
                        cell=(float(L[a_axis]), float(L[q_axis]),
                              float(L[c_axis])),
                        reps=(n_a, n_q, n_c), n_sites=4 * n_a * n_q * n_c,
                        sites_matched=got[0], atom_coverage=got[1],
                        median_residual=got[2]))
    return None if best is None else best[1]
