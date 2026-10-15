"""Nanotube chirality inference: a real carbon frame -> (n, m, length).

RD2 (docs/reviews/research_decisions.md), lift half, rewritten for D33
(docs/reviews/open_items_v4.md, work order Y3): the inference reads REAL
tube frames -- per-coordinate noise up to 0.1 A, 6-significant-digit text,
periodic cells of three or more axial periods, a tube anywhere in its cell,
a diameter up to 0.8 of the perpendicular cell -- each 20 of 20 correct,
the radius within 0.05 A. Ideal frames keep lifting exactly. The
construction inverts build/nanotube.py (Saito, Dresselhaus & Dresselhaus,
"Physical Properties of Carbon Nanotubes", Imperial College Press 1998,
ch. 3) through the bond graph:

1. bonds are atom pairs within the carbon dialect's `cn_cutoff` (the C-C
   first shell). The frame must be ONE connected component -- an fcc
   carbon crystal (nearest neighbour 2.475 A) has no bonds at that cutoff,
   a frame holding two tubes is two components; both are refused here,
   fail closed;
2. the component is unwrapped breadth-first through the bond graph, adding
   MINIMUM-IMAGE bond vectors -- never by minimum image to one anchor. A
   tube whose diameter exceeds half the perpendicular cell (the (16,16) of
   the first-result pilot) has no unique minimum-image anchor, and a tube
   that straddles a periodic face unwraps coherently here. Every bond is
   far shorter than half the cell, so each unwrapped position is a
   periodic image of the wrapped one: no noise accumulates along the
   traversal, whatever its length;
3. the axis is the cell axis along which the unwrapped component WINDS. A
   tube periodic in its cell closes its bond graph with a net shift of one
   cell length along its axis, so its unwrapped extent along that axis
   covers at least the cell length (the largest cyclic gap between
   consecutive atoms is one bond projection, hence the bond-cutoff
   margin); the perpendicular extents are the diameter, at most 0.8 of the
   cell by D33. Two winding axes is a sheet; zero is a finite tube whose
   axis is the longest extent of the unwrapped component (best effort: a
   short finite tube whose axial extent does not clearly exceed its
   diameter is honestly refused by the radius and integer gates);
4. the radius is the mean radial distance of a least-squares circle fit
   (the linear Kasa fit) in the perpendicular plane of the unwrapped
   coordinates. The RMS residual is gated against the dialect's
   `nanotube_radius_rms_max`, measured on 300 K and 600 K thermal tube
   frames (see carbon.yaml's evidence note): loose enough for the 0.1 A
   D33 noise, an order of magnitude below the several-A residual of any
   non-tubular frame (a flat ribbon, a fullerene wall);
5. (n, m): the lattice constant from the unrolled sheet's areal density --
   graphene holds one atom per sqrt(3)/4 a^2, the unrolled tube covers
   2 pi R x L_axis with exactly the frame's N atoms, so
   a = sqrt(8 pi R L_axis / (sqrt(3) N)). Every ingredient is an exact or
   least-squares-averaged quantity, so a is UNBIASED under symmetric noise
   (the mean bond length, the obvious alternative, is inflated by the
   transverse noise -- E|r| > r0 -- and the bias grows like sqrt(N) in
   |Ch|, smearing the integer fit); the chiral angle comes from the bond
   directions in the unrolled sheet -- the circular mean of exp(6i phi)
   over all bonds, the sixth harmonic being invariant under every bond's
   sign and 60-degree relabelling (the three graphene bond directions are
   exactly 60 degrees apart, and bond direction phi and phi + 60 degrees
   contribute identically); then the integer pair whose chiral vector
   n a1 + m a2 best fits BOTH |Ch| = 2 pi R and the measured angle, solved
   as one 2x2 least-squares system in the unrolled frame. The best pair
   must sit within `nanotube_chirality_mismatch_max` of the measured
   vector and the SECOND-best pair at least `nanotube_chirality_margin`
   further away -- an ambiguous frame is refused, never assigned the
   closer of two chiralities;
6. |T| = sqrt(3) |Ch| / d_R (Saito/Dresselhaus ch. 3) with the measured a;
   the built length is round(L_axis / |T|) periods, gated by
   `nanotube_axial_fit_tolerance`; the atom count 4 N n_T / d_R must equal
   the frame's exactly.

Every tolerance lives in the carbon dialect with its evidence; nothing
here scales with the system under test.

The finder (`nanotube_components`) applies the same inference to the
all-carbon three-coordinated components of a mixed frame (D35: the tube
plus its confined liquid), using the D35 wording of the brief: components
of pure carbon whose mean coordination passes
`nanotube_min_mean_coordination` (solvent carbons and benzene rings sit at
2 or below; sp2 lattices at 3) and whose geometry passes every gate above.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import gcd

import numpy as np

from ..lang.errors import ChaordError


@dataclass(frozen=True)
class NanotubeChirality:
    """The inferred chirality (n, m) plus the measured tube geometry.

    `length` is the commensurate built length (n_T periods); `radius` the
    circle-fit mean radial distance with its RMS residual `radius_rms`;
    `axis` the cell axis the tube winds along; `centre` the axis point in
    wrapped cell coordinates (the cylinder region's centre)."""
    n: int
    m: int
    length: float
    radius: float
    axis: str
    centre: tuple
    radius_rms: float


@dataclass(frozen=True, eq=False)
class TubeComponent:
    """One all-carbon component of a frame, identified as a nanotube."""
    indices: np.ndarray
    chirality: NanotubeChirality


def _carbon_stack(dialect):
    """The caller's dialect, else the carbon stack (the reproducer's
    no-dialect call reads the same thresholds the lift arm uses)."""
    if dialect is not None:
        return dialect
    from ..dialects import load_dialect
    return load_dialect(("core", "carbon"))


def _has_carbon_thresholds(dialect) -> bool:
    """Does this dialect carry the nanotube inference's vocabulary?

    A stack without the carbon keys (core + lj, the crystal stacks; note
    core + lj DOES define its own `cn_cutoff` 1.5 for Ar-class first
    shells, so the check must name a carbon-only key) has no tube
    vocabulary: the finder declines instead of raising -- the same
    carve-out the legacy ladder makes for the one ChaordError that is a
    configuration fact, not a phase verdict."""
    try:
        dialect.threshold("nanotube_radius_rms_max")
    except ChaordError as e:
        if "defines no threshold" in str(e):
            return False
        raise
    return True


def infer_nanotube_chirality(positions, cell,
                             dialect=None) -> NanotubeChirality:
    """(n, m, length, geometry) of a nanotube frame, or a refusal.

    `positions` is the (N, 3) carbon frame in its own unit system, `cell`
    the diagonal lengths or the full (orthogonal) 3x3 cell, `dialect` the
    stack carrying the tolerances (default: core + carbon)."""
    dl = _carbon_stack(dialect)
    L = _cell_diag(cell)
    pos = _wrapped(positions, L)
    cut = float(dl.threshold("cn_cutoff"))
    if np.any(L <= 2 * cut):
        raise ChaordError("the cell is smaller than twice the C-C bond "
                          "cutoff: minimum-image bonds are not unique")
    i_arr, j_arr, _bond_vecs = _bonds(pos, L, cut)
    if len(i_arr) == 0:
        raise ChaordError("no C-C bonds at the dialect cutoff: not a "
                          "nanotube frame")
    adjacency = _adjacency(i_arr, j_arr, len(pos))
    u = _unwrap(pos, L, adjacency)
    axis, periodic = _tube_axis(u, L, cut)
    perp = [k for k in range(3) if k != axis]
    R, centre, rms = _circle_fit(u[:, perp])
    if R <= 0 or rms > float(dl.threshold("nanotube_radius_rms_max")):
        raise ChaordError(
            f"the atoms do not sit on one radius about the axis (circle-fit "
            f"RMS residual {rms:.3f} A over radius {R:.3f} A): not a "
            "nanotube frame")
    L_axis = float(L[axis]) if periodic else float(u[:, axis].ptp())
    a = float(np.sqrt(8 * np.pi * R * L_axis / (np.sqrt(3) * len(pos))))
    theta = _chiral_angle(u, i_arr, j_arr, centre, axis, perp)
    n, m, _mis, _gap = _fit_chirality(a, theta, 2 * np.pi * R, dl)
    N = n * n + n * m + m * m
    dr = gcd(2 * m + n, 2 * n + m)
    t_len = float(np.sqrt(3) * a * np.sqrt(N) / dr)
    n_t = int(round(L_axis / t_len))
    length = n_t * t_len
    if abs(L_axis - length) > float(dl.threshold("nanotube_axial_fit_tolerance")):
        raise ChaordError(
            f"the axial extent {L_axis:.4f} A is not {n_t} x |T| = "
            f"{length:.4f} A of the ({n},{m}) lattice: not a nanotube frame")
    if 4 * N * n_t % dr or len(pos) != 4 * N * n_t // dr:
        raise ChaordError(
            f"the frame holds {len(pos)} atoms but ({n},{m}) over {n_t} "
            f"periods builds {4 * N * n_t // dr}: not a nanotube frame")
    c_perp = np.mod(centre, L[perp])
    c_ax = float(L[axis]) / 2 if periodic else float(
        np.mod((u[:, axis].max() + u[:, axis].min()) / 2, L[axis]))
    centre3 = [0.0, 0.0, 0.0]  # dialect-exempt: exact-geometry: axis-point fill
    centre3[perp[0]], centre3[perp[1]], centre3[axis] = c_perp[0], c_perp[1], c_ax
    return NanotubeChirality(n=n, m=m, length=length, radius=float(R),
                             axis="xyz"[axis], centre=tuple(centre3),
                             radius_rms=float(rms))


def nanotube_components(frame, dialect=None) -> list:
    """The all-carbon three-coordinated components of `frame` that pass the
    chirality inference (D35), largest first; [] when there is none (or the
    dialect carries no carbon vocabulary)."""
    dl = _carbon_stack(dialect)
    if not _has_carbon_thresholds(dl):
        return []
    carbon = np.array([k for k, s in enumerate(frame.symbols) if s == "C"],
                      int)
    if len(carbon) == 0:
        return []
    L = np.asarray(frame.cell_diag, float)
    pos = _wrapped(frame.pos[carbon], L)
    cut = float(dl.threshold("cn_cutoff"))
    i_arr, j_arr, _d = _bonds(pos, L, cut)
    if len(i_arr) == 0:
        return []
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(carbon)
    rows = np.concatenate([i_arr, j_arr])
    cols = np.concatenate([j_arr, i_arr])
    _, labels = connected_components(
        coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)),
        directed=False)
    min_coord = float(dl.threshold("nanotube_min_mean_coordination"))
    out = []
    for g in range(labels.max() + 1):
        local = np.where(labels == g)[0]
        if len(local) < 2:
            continue
        deg = np.bincount(rows, minlength=n)[local]
        if deg.mean() < min_coord:
            continue
        try:
            chir = infer_nanotube_chirality(pos[local], L, dialect=dl)
        except ChaordError as e:
            if "defines no threshold" in str(e):
                raise          # a configuration bug, not a not-a-tube verdict
            continue           # not a tube: its atoms are the fluid's problem
        out.append(TubeComponent(indices=carbon[local], chirality=chir))
    out.sort(key=lambda t: -len(t.indices))
    return out


# ------------------------------------------------------------- the pipeline --

def _cell_diag(cell) -> np.ndarray:
    """The three box lengths of an orthogonal cell (3-vector or 3x3)."""
    c = np.asarray(cell, float)
    if c.ndim == 1 and c.shape == (3,):
        return c
    if c.ndim == 2 and c.shape == (3, 3) and not np.any(c - np.diag(np.diag(c))):
        return np.diag(c)
    raise ChaordError("infer_nanotube_chirality needs a 3-vector or an "
                      "orthogonal 3x3 cell")


def _wrapped(positions, L) -> np.ndarray:
    """Positions wrapped into the half-open cell [0, L).

    np.mod(-1e-15, L) returns exactly L (verified for this repo's box
    lengths); every downstream periodic container is half-open [0, box),
    so clamp the wrapped coordinates just under the upper face -- the
    repo's standing wrap idiom (bench sanity, cv helpers)."""
    pos = np.mod(np.asarray(positions, float), L)
    return np.minimum(pos, L * (1 - 1e-9))  # dialect-exempt: numerical-guard: np.mod box clamp


def _bonds(pos, L, cut):
    """(i, j, d): both index arrays and the minimum-image bond vectors."""
    from scipy.spatial import cKDTree
    pairs = cKDTree(pos, boxsize=L).query_pairs(cut, output_type="ndarray")
    i, j = pairs[:, 0], pairs[:, 1]
    d = pos[j] - pos[i]
    d -= L * np.round(d / L)
    return i, j, d


def _adjacency(i_arr, j_arr, n) -> list:
    """Sorted neighbour lists of the bond graph (deterministic traversal)."""
    adjacency = [[] for _ in range(n)]
    for a, b in zip(i_arr.tolist(), j_arr.tolist()):
        adjacency[a].append(b)
        adjacency[b].append(a)
    for lst in adjacency:
        lst.sort()
    return adjacency


def _unwrap(pos, L, adjacency) -> np.ndarray:
    """Breadth-first unwrap through the bond graph (module docstring, 2).

    Each newly visited atom is placed one minimum-image bond vector from
    its parent, so the unwrapped component is the true continuum image of
    the wrapped one (each unwrapped position differs from its wrapped
    position by a whole-cell translation, never by accumulated noise)."""
    from collections import deque
    n = len(pos)
    u = np.empty_like(pos)
    seen = np.zeros(n, bool)
    u[0] = pos[0]
    seen[0] = True
    queue = deque([0])
    while queue:
        a = queue.popleft()
        for b in adjacency[a]:
            if seen[b]:
                continue
            step = pos[b] - pos[a]
            step -= L * np.round(step / L)
            u[b] = u[a] + step
            seen[b] = True
            queue.append(b)
    if not seen.all():
        raise ChaordError("the bond graph is not one connected component "
                          "(no bonds, or more than one tube in the frame): "
                          "not a single nanotube frame")
    return u


def _tube_axis(u, L, cut) -> tuple:
    """(axis index, is_periodic): the winding axis (module docstring, 3)."""
    extents = u.max(axis=0) - u.min(axis=0)
    winding = [k for k in range(3) if extents[k] >= L[k] - cut]
    if len(winding) > 1:
        raise ChaordError("the component winds along more than one cell "
                          "axis (a sheet, not a tube): not a nanotube frame")
    if len(winding) == 1:
        return winding[0], True
    axis = int(np.argmax(extents))
    return axis, False


def _circle_fit(q) -> tuple:
    """(R, centre, RMS residual) of the linear least-squares circle fit on
    the 2-D unwrapped perpendicular coordinates (Kasa's method: minimise
    the algebraic error of |q - c|^2 = R^2)."""
    A = np.column_stack([2 * q, np.ones(len(q))])
    sol = np.linalg.lstsq(A, np.sum(q * q, axis=1), rcond=None)[0]
    centre = sol[:2]
    d = q - centre
    r = np.linalg.norm(d, axis=1)
    R = float(r.mean())
    rms = float(np.sqrt(np.mean((r - R) ** 2)))
    return R, centre, rms


def _chiral_angle(u, i_arr, j_arr, centre, axis, perp) -> float:
    """The chiral angle in radians, canonicalised into [0, 30 degrees].

    Each bond's direction phi in the unrolled sheet (circumferential versus
    axial, in the local frame of the bond's midpoint) contributes exp(6i
    phi); the circular mean's argument over 6 maps the 60-degree lattice
    symmetry onto one turn, so bond sign and relabelling drop out."""
    mid = (u[i_arr] + u[j_arr]) / 2
    rm = mid[:, perp] - centre
    ec = np.column_stack([-rm[:, 1], rm[:, 0]])
    ec /= np.linalg.norm(ec, axis=1)[:, None]
    d = u[j_arr] - u[i_arr]
    dphi = np.arctan2(d[:, axis], np.einsum("ij,ij->i", d[:, perp], ec))
    z = np.exp(6j * dphi).sum()
    sixth = np.angle(z) / 6
    beta = np.mod(sixth, np.pi / 3)
    theta = np.mod(np.pi / 6 - beta, np.pi / 3)
    return float(min(theta, np.pi / 3 - theta))


def _fit_chirality(a, theta, ch_len, dl) -> tuple:
    """(n, m, best mismatch, best-to-second gap): the integer pair whose
    chiral vector best fits the measured |Ch| and angle (module docstring,
    5), with the ambiguity refusal."""
    c, s = np.cos(theta), np.sin(theta)
    c60, s60 = np.cos(np.pi / 3 - theta), np.sin(np.pi / 3 - theta)
    a1 = a * np.array([c, -s])
    a2 = a * np.array([c60, s60])
    target = np.array([ch_len, 0.0])  # dialect-exempt: exact-geometry: the chiral vector is purely circumferential in the sheet frame
    nf, mf = np.linalg.lstsq(np.column_stack([a1, a2]), target, rcond=None)[0]
    scored = []
    for dn in range(-2, 3):
        for dm in range(-2, 3):
            n = int(round(nf)) + dn
            m = int(round(mf)) + dm
            if not (n >= m >= 0 and n >= 1):
                continue
            scored.append((float(np.linalg.norm(n * a1 + m * a2 - target)),
                           n, m))
    if not scored:
        raise ChaordError("no canonical chirality pair fits the measured "
                          "lattice: not a nanotube frame")
    scored.sort()
    mismatch, n, m = scored[0]
    max_mis = float(dl.threshold("nanotube_chirality_mismatch_max"))
    if mismatch > max_mis:
        raise ChaordError(
            f"the best chirality ({n},{m}) misses the measured chiral "
            f"vector by {mismatch:.3f} A (tolerance {max_mis:g}): not a "
            "nanotube frame")
    gap = (scored[1][0] - mismatch) if len(scored) > 1 else np.inf
    margin = float(dl.threshold("nanotube_chirality_margin"))
    if gap < margin:
        other = f"({scored[1][1]},{scored[1][2]})" if len(scored) > 1 else "-"
        raise ChaordError(
            f"the chirality is ambiguous: ({n},{m}) and {other} fit the "
            f"measured chiral vector {gap:.3f} A apart, inside the "
            f"{margin:g} A margin: refusing rather than guessing")
    return n, m, mismatch, gap
