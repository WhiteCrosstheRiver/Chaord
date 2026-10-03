"""Surface lifter: slab + vacuum + adsorbates / molecular overlayer (M4, M6).

Atoms above the surface layer lift as single-atom `adsorb` statements; a
molecular census of the surface zone (the top layer and everything above)
additionally yields a liquid `overlayer` region of `molecules` statements and,
when dissociation fragments (OH + H) are present, a `dissociate` statement on
the slab | overlayer interface."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree, Delaunay

from ..io.frames import Frame
from ..lang.errors import ChaordError
from ..lang.ir import (
    GeoChain, InterfaceBlock, Name, PhysicsBlock, Plane, Program,
    ProvenanceBlock, Quantity, RangeVal, RegionBlock, ResidualBlock, ShSlab,
    SpecDef, SpeciesBlock, Statement, StrVal, SystemBlock, Wood,
)
from ..build.prototypes import PROTOTYPES


def _termination_names(dialect):
    """dialect vocabulary: element -> compound termination name (or {})."""
    try:
        return dict(dialect.threshold("surface_termination_names"))
    except ChaordError:
        return {}


def _net_dialect(dialect):
    """Dialect for the net helpers; defaults to the surface dialect when a
    caller (e.g. a direct test) omits one."""
    if dialect is not None:
        return dialect
    from ..dialects import load_dialect
    return load_dialect(("surface",))


def has_vacuum(frame: Frame, dialect) -> bool:
    """A z-gap in the atom distribution wider than the dialect threshold."""
    try:
        gap_min = float(dialect.threshold("vacuum_gap_min"))
    except Exception:
        return False   # no surface dialect loaded: no vacuum concept
    L = frame.cell_diag
    binw = float(dialect.threshold("profile_bin_size"))
    n = max(int(L[2] / binw), 8)
    # wrap z first: a slab crossing the periodic z boundary must not smear
    # its bottom layer into bin 0
    z = np.mod(frame.pos[:, 2], L[2])
    idx = np.clip((z / L[2] * n).astype(int), 0, n - 1)
    occ = np.bincount(idx, minlength=n) > 0
    # circular: the gap may wrap through z = 0
    best = 0
    run = 0
    for o in np.r_[occ, occ[:1]]:
        if not o:
            run += 1
            best = max(best, run)
        else:
            run = 0
    gap = best * (L[2] / n)
    return gap >= gap_min


def _layer_split(z, dialect):
    """Cluster wrapped z coordinates into layers (gap > layer_tolerance)."""
    tol = float(dialect.threshold("layer_tolerance"))
    order = np.argsort(z, kind="stable")
    layers = []
    current = [order[0]]
    for i in order[1:]:
        if z[i] - z[current[-1]] > tol:
            layers.append(np.array(current))
            current = [i]
        else:
            current.append(i)
    layers.append(np.array(current))
    return layers


def _layers(frame: Frame, dialect, z=None):
    """Cluster atoms into z-layers (gap > layer_tolerance separates).

    `z` are wrapped coordinates (e.g. the gap-aligned array from
    `_aligned_z`); without one the raw wrapped z of the frame is used."""
    if z is None:
        z = np.mod(frame.pos[:, 2], frame.cell_diag[2])
    return _layer_split(z, dialect)


def _aligned_z(frame: Frame, dialect):
    """z coordinates wrapped into the cell and rotated so the condensed body
    (the atom-majority layer group) is contiguous from z = 0 up.

    A slab whose bottom layer crosses the periodic z boundary wraps to the
    top of the box, where raw coordinates would lift it as adsorbates. Any
    fragment cut off at the junction that is made only of species already
    bulk in the body is re-attached first (an overlayer of foreign molecules
    never is); the rotation then places the body's widest circular gap -- the
    vacuum -- at the top of the box, which is invariant under rigid z
    translations of the frame (minimum-image equivalent)."""
    L = float(frame.cell_diag[2])
    z = np.mod(np.asarray(frame.pos[:, 2], float), L)
    if len(z) == 0:
        return z
    gap_min = float(dialect.threshold("vacuum_gap_min"))
    groups: list[list] = []
    for layer in _layer_split(z, dialect):
        if groups and (z[layer].min() - z[groups[-1][-1]].max()) <= gap_min:
            groups[-1].append(layer)
        else:
            groups.append([layer])
    stack = max(groups, key=lambda g: sum(len(l) for l in g))
    if len(groups) > 1:
        syms = np.array(frame.symbols)
        dense = float(dialect.threshold("layer_dense_fraction"))
        biggest = max(len(l) for l in stack)
        bulk = set()
        for layer in stack:
            if len(layer) >= dense * biggest:
                bulk |= set(syms[layer])
        junc = (L - z[np.concatenate(groups[-1])].max()
                + z[np.concatenate(groups[0])].min())
        extra = []
        for end in (groups[0], groups[-1]):
            if end is stack or junc > gap_min:
                continue
            if set(syms[np.concatenate(end)]) <= bulk:
                extra.extend(end)
        stack = stack + extra
    idx = np.concatenate(stack)
    zs = np.sort(z[idx])
    gaps = np.diff(np.r_[zs, zs[0] + L])
    rot = zs[(int(np.argmax(gaps)) + 1) % len(zs)]
    return np.mod(z - rot, L)


def _top_layer(frame: Frame, dialect, z=None):
    """Slab top layer: the highest layer carrying a substantial atom count.

    Adsorbate layers sit above it with only a few atoms of a foreign species;
    the surface layer has at least half the median layer size — or carries
    only species present in the dense (bulk) layers, which keeps sparse
    reconstructions (missing-row p(2x1), (r3xr3)R30) recognisable as the
    surface layer rather than adsorbates. A molecular overlayer sits across a
    vacuum gap from the slab, so layers are first split at gaps wider than the
    dialect's vacuum threshold and only the atom-majority group (the slab
    stack) is searched: a dense all-bulk-species fragment of the overlayer is
    never the surface layer. `z` defaults to the gap-aligned coordinates of
    `_aligned_z`, so a slab crossing the periodic z boundary is judged on its
    rotated (wrapped) coordinates."""
    if z is None:
        z = _aligned_z(frame, dialect)
    layers = _layer_split(z, dialect)
    gap = float(dialect.threshold("vacuum_gap_min"))
    dense_fraction = float(dialect.threshold("layer_dense_fraction"))
    surface_fraction = float(dialect.threshold("layer_surface_fraction"))
    groups: list[list] = []
    for layer in layers:
        if groups and (z[layer].min() - z[groups[-1][-1]].max()) <= gap:
            groups[-1].append(layer)
        else:
            groups.append([layer])
    stack = max(groups, key=lambda g: sum(len(l) for l in g))
    sizes = [len(l) for l in stack if len(l) >= 2] or [len(stack[-1])]
    median = float(np.median(sizes))
    syms = np.array(frame.symbols)
    biggest = max((len(l) for l in stack), default=0)
    bulk_species = set()
    for layer in stack:
        if len(layer) >= dense_fraction * biggest:
            bulk_species |= set(syms[layer])
    for layer in reversed(stack):
        if len(layer) < 2:
            continue
        if len(layer) >= surface_fraction * median or set(syms[layer]) <= bulk_species:
            return layer, float(z[layer].max())
    return stack[-1], float(z[stack[-1]].max())


def _net_vectors(pts2d, cell2d, dialect=None):
    """Two shortest independent primitive vectors of a 2D point set, or None.

    v1 is the shortest inter-atom vector; v2 is the shortest independent
    vector whose cell area matches the point density (rejecting 2x cells).
    Returns None when the points do not form a lattice with that v1."""
    dialect = _net_dialect(dialect)
    length_tol = float(dialect.threshold("net_length_tol"))
    area_tol = float(dialect.threshold("net_cell_area_tol"))
    pts = np.asarray(pts2d, float)
    A2 = np.asarray(cell2d, float)
    pts = np.mod(pts @ np.linalg.inv(A2), 1.0) @ A2  # dialect-exempt: numerical-guard: fractional wrap
    p0 = pts[0]
    cand = []
    for q in pts:
        v = q - p0
        for na in (-1, 0, 1):
            for nb in (-1, 0, 1):
                w = v + na * A2[0] + nb * A2[1]
                n = np.linalg.norm(w)
                if n > 0.5:  # dialect-exempt: numerical-guard: sub-noise duplicate-vector guard, A
                    cand.append((n, w))
    cand.sort(key=lambda x: x[0])
    v1 = cand[0][1]
    area_per_point = abs(A2[0][0] * A2[1][1] - A2[0][1] * A2[1][0]) / len(pts)
    for n, w in cand:
        if abs(abs(np.dot(w, v1) / n) - np.linalg.norm(v1)) < length_tol * np.linalg.norm(v1):
            continue  # collinear with v1 (projected length matches |v1|)
        area = abs(v1[0] * w[1] - v1[1] * w[0])  # 2D cross product
        if abs(area - area_per_point) <= area_tol * area_per_point:
            return np.array(v1, float), np.array(w, float)
    return None


def _surface_net(frame: Frame, top_idx, dialect):
    """(l1, l2, angle) of the primitive in-plane lattice of a layer."""
    net = _net_vectors(frame.pos[top_idx][:, :2], frame.cell[:2, :2], dialect)
    if net is None:
        return 0.0, 0.0, 90.0  # dialect-exempt: numerical-guard: degenerate net sentinel report
    v1, v2 = net
    cosg = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1)
    return (float(np.linalg.norm(v1)), float(np.linalg.norm(v2)),
            float(np.degrees(np.arccos(cosg))))


def _sub_layer(layers, top_idx):
    """Index array of the layer below the surface layer carrying >= 2 atoms.

    This is the substrate the top layer is compared against for Wood
    reconstruction; falls back to the top layer when no such layer exists."""
    top_set = {int(i) for i in top_idx}
    seen = False
    for layer in reversed(layers):
        if seen and len(layer) >= 2:
            return layer
        if {int(i) for i in layer} == top_set:
            seen = True
    return top_idx


# dialect-exempt-begin: exact-geometry
def _xgcd(a: int, b: int):
    """(g, x, y) with g = gcd(a, b) >= 0 and g = x*a + y*b."""
    if a < 0:
        g, x, y = _xgcd(-a, b)
        return g, -x, y
    if b < 0:
        g, x, y = _xgcd(a, -b)
        return g, x, -y
    if b == 0:
        return a, 1, 0
    g, x, y = _xgcd(b, a % b)
    return g, y, x - (a // b) * y


def _hnf2(M):
    """Row Hermite normal form [[h11, h12], [0, h22]] of a 2x2 integer matrix."""
    r1 = np.array(M[0], dtype=object)
    r2 = np.array(M[1], dtype=object)
    g, x, y = _xgcd(int(r1[0]), int(r2[0]))
    n1 = x * r1 + y * r2
    n2 = (int(r2[0]) // g) * r1 - (int(r1[0]) // g) * r2
    r1, r2 = n1, n2
    if r2[1] < 0:
        r2 = -r2
    if r1[0] < 0:
        r1 = -r1
    if r2[1] != 0:
        k = int(r1[1]) // int(r2[1])
        r1 = r1 - k * r2
        if r1[1] < 0:
            r1 = r1 + r2
    return np.array([[int(r1[0]), int(r1[1])], [0, int(r2[1])]])
# dialect-exempt-end


def _wood_ratio_text(r, tol, sqrt_ks=(2, 3, 5, 7)):
    """Format a length ratio as a Wood token component: integer or r<sqrt k>."""
    if abs(r - round(r)) <= tol:
        return str(int(round(r)))
    for k in sqrt_ks:   # crystallographic sqrt ratios (integers, exact)
        if abs(r - np.sqrt(k)) <= tol:
            return f"r{k}"
    return None


def _wood_statement(top_net, sub_net, dialect):
    """Wood notation for the top net relative to the substrate net, or None.

    The integer superstructure matrix M expresses the top-layer primitive
    vectors in substrate-net coordinates; |det M| = the density loss of the
    reconstructed layer. Identity/unimodular (|det| <= 1) nets are not
    reconstructions; incommensurate overlays get no Wood statement."""
    if top_net is None or sub_net is None:
        return None
    v_top = np.array(top_net, float)
    v_sub = np.array(sub_net, float)
    F = v_top @ np.linalg.inv(v_sub)
    if not np.isfinite(F).all():  # pragma: no cover - degenerate net guard
        return None
    M = np.round(F)
    if np.abs(F - M).max() > float(dialect.threshold("wood_commensurate_tol")):
        return None
    M = M.astype(int)
    det = int(round(np.linalg.det(M.astype(float))))
    if abs(det) <= 1:
        return None  # same lattice as the substrate: no reconstruction
    H = _hnf2(M)
    if H[0][1] == 0 and H[1][0] == 0:
        n, m = sorted((int(H[0][0]), int(H[1][1])), reverse=True)
        if n >= 1 and m >= 1 and (n, m) != (1, 1):
            return f"p({n}x{m})"
    # centered overlay: b1 + b2 and b1 - b2 along the substrate axes
    s1, s2 = M[0] + M[1], M[0] - M[1]
    for (u, w) in ((s1, s2), (s2, s1)):
        if u[1] == 0 and w[0] == 0 and abs(u[0]) >= 2 and abs(w[1]) >= 2:
            n, m = sorted((abs(int(u[0])), abs(int(w[1]))), reverse=True)
            return f"c({n}x{m})"
    # rotated overlay: report ratios (integers or sqrt k) and the rotation
    # angle reduced modulo the substrate net's own rotational symmetry
    tol = float(dialect.threshold("wood_ratio_tolerance"))
    a_tol = float(dialect.threshold("wood_angle_tolerance"))
    r1 = np.linalg.norm(v_top[0]) / np.linalg.norm(v_sub[0])
    r2 = np.linalg.norm(v_top[1]) / np.linalg.norm(v_sub[1])
    t1 = _wood_ratio_text(r1, tol)
    t2 = _wood_ratio_text(r2, tol)
    if t1 is None or t2 is None:
        return None  # not expressible with the Wood token vocabulary
    cosg = np.clip(np.dot(v_sub[0], v_top[0])
                   / (np.linalg.norm(v_sub[0]) * np.linalg.norm(v_top[0])), -1, 1)
    theta = int(round(float(np.degrees(np.arccos(cosg)))))
    sym = _net_symmetry_deg(v_sub, dialect)
    theta = min(theta % sym, sym - theta % sym)
    if theta <= a_tol:  # within the rounding tolerance: no measurable rotation
        theta = 0
    if t1 == t2:
        return f"({t1}x{t2})R{theta}" if theta else f"({t1}x{t2})"
    return f"p({t1}x{t2})R{theta}" if theta else f"p({t1}x{t2})"


# dialect-exempt-begin: exact-geometry
def _net_symmetry_deg(v_sub, dialect=None):
    """Smallest rotation mapping the substrate net onto itself (degrees)."""
    dialect = _net_dialect(dialect)
    ratio_tol = float(dialect.threshold("surface_net_ratio_tolerance"))
    angle_tol = float(dialect.threshold("surface_net_angle_tolerance"))
    l1, l2 = np.linalg.norm(v_sub[0]), np.linalg.norm(v_sub[1])
    cosg = np.clip(np.dot(v_sub[0], v_sub[1]) / (l1 * l2), -1, 1)
    gamma = float(np.degrees(np.arccos(cosg)))
    equi = abs(l1 - l2) / max(l1, l2) < ratio_tol
    if equi and abs(min(gamma, 180 - gamma) - 60) < angle_tol:
        return 60   # hexagonal net
    if equi and abs(gamma - 90) < angle_tol:
        return 90   # square net
    return 180      # rectangular / oblique: only +/- counts
# dialect-exempt-end


# dialect-exempt-begin: exact-geometry
def _identify_hkl(l1, l2, gamma, a_bulk, dialect=None):
    dialect = _net_dialect(dialect)
    tol = float(dialect.threshold("surface_net_ratio_tolerance"))
    ang_tol = float(dialect.threshold("surface_net_angle_tolerance"))
    if abs(l1 - l2) / max(l1, l2) < tol and abs(gamma - 90) < ang_tol:
        return "(001)"          # square net
    if abs(gamma - 90) < ang_tol and abs(l2 / l1 - 2**0.5) / 2**0.5 < tol:
        return "(110)"          # rectangular a x a*sqrt(2)
    if abs(l1 - l2) / max(l1, l2) < tol and abs(gamma - 60) < ang_tol:
        return "(111)"          # hexagonal net
    if abs(gamma - 120) < ang_tol and abs(l1 - l2) / max(l1, l2) < tol:
        return "(111)"
    return "(001)"
# dialect-exempt-end


def _prototype_bulk_atoms(name, params, slot_species):
    """ASE Atoms of one conventional prototype cell (build registry geometry)."""
    from ase import Atoms
    from ..build.prototypes import basis as proto_basis, cell_matrix
    frac, slots = proto_basis(name, params)
    return Atoms(symbols=[slot_species[s] for s in slots],
                 scaled_positions=frac, cell=cell_matrix(name, params), pbc=True)


def _frac_wrap(P, cell):
    inv = np.linalg.inv(cell)
    f = P @ inv
    return (f - np.floor(f + 1e-9)) @ cell  # dialect-exempt: numerical-guard: strict wrap below 1


def _inplane_primitive(P, syms, cell, dialect):
    """(v1, v2): the shortest in-plane translations mapping the atom set onto
    itself species-by-species (the primitive surface mesh of the slab)."""
    tol = float(dialect.threshold("site_match_tolerance"))
    collinear_tol = float(dialect.threshold("net_collinear_tol"))
    sym = np.asarray(syms)
    W = _frac_wrap(np.asarray(P, float), cell)
    tree = cKDTree(W)
    cands = [cell[0][:2].copy(), cell[1][:2].copy(),
             (cell[0] + cell[1])[:2].copy(), (cell[0] - cell[1])[:2].copy()]
    z = np.asarray(P, float)[:, 2]
    for i in range(len(P)):
        for j in range(i + 1, len(P)):
            if sym[i] != sym[j] or abs(z[i] - z[j]) > tol:
                continue
            cands.append(np.asarray(P, float)[j, :2] - np.asarray(P, float)[i, :2])

    def is_translation(t2):
        shift = np.array([t2[0], t2[1], float(0)])
        d, k = tree.query(_frac_wrap(np.asarray(P, float) + shift, cell))
        return bool(d.max() < tol and (sym[k] == sym).all())

    span = float(np.linalg.norm(cell[0])) + float(np.linalg.norm(cell[1]))
    valid, seen = [], set()
    for t in cands:
        n = float(np.linalg.norm(t))
        if n < tol or n > span:
            continue
        key = (int(round(t[0] / tol)), int(round(t[1] / tol)))
        if key in seen:
            continue
        seen.add(key)
        if is_translation(t):
            valid.append((n, key, t))
    if not valid:
        return None
    valid.sort()
    v1 = valid[0][2]
    for n, _key, t in valid[1:]:
        area = abs(v1[0] * t[1] - v1[1] * t[0])
        if area > collinear_tol * np.linalg.norm(v1) * n:
            return v1, t
    return None


def _net_shape(v1, v2):
    """Scale-free surface-mesh shape: (long/short length ratio, mesh angle)."""
    l1, l2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
    lo, hi = min(l1, l2), max(l1, l2)
    cosg = np.clip(np.dot(v1, v2) / (l1 * l2), -1, 1)
    ang = float(np.degrees(np.arccos(cosg)))
    return hi / max(lo, 1e-9), min(ang, 180.0 - ang)  # dialect-exempt: numerical-guard: degenerate length guard / angle fold


def _identify_prototype_slab(frame: Frame, dialect, z=None):
    """(name, params, slot_species) of the bulk prototype under a compound slab.

    One stacking period of the slab (cell = surface mesh x stack height) is a
    periodic crystal; standardise + prototype-match identifies it. The stack
    may need an integer number of interplanar steps per period. `z` are the
    gap-aligned coordinates (slab contiguous from 0); without one they are
    computed here, so a slab crossing the periodic z boundary works too."""
    from .crystal import match_prototype, standardize
    if z is None:
        z = _aligned_z(frame, dialect)
    layers = _layer_split(z, dialect)
    sizes = [len(l) for l in layers]
    dense = [i for i, s in enumerate(sizes) if s == max(sizes)]
    if len(dense) < 2:
        raise ChaordError("no stacking period found for the compound slab")
    tol = float(dialect.threshold("layer_tolerance"))
    d = (z[layers[dense[1]]].mean() - z[layers[dense[0]]].mean())
    z0 = z[layers[dense[0]]].mean() - tol / 4
    wrapped = _frac_wrap(frame.pos, frame.cell)
    for k in (1, 2, 3):
        period = k * d
        idx = np.where((z >= z0) & (z < z0 + period))[0]
        if len(idx) == 0:
            continue
        cell_w = frame.cell.copy()
        cell_w[2] = [0, 0, period]
        pos_w = wrapped[idx].copy()
        pos_w[:, 2] = z[idx] - z0
        try:
            sc, sp, sn = standardize(
                Frame(pos=pos_w, cell=cell_w,
                      symbols=[frame.symbols[i] for i in idx]), dialect)
            matched = match_prototype(sc, sp, sn, dialect)
        except Exception:
            continue
        if matched is not None:
            name, params, slot_species = matched[0], matched[1], matched[2]
            return name, dict(params), tuple(slot_species)
    raise ChaordError("cannot identify the bulk prototype of the compound slab")


def _prototype_hkl(P, syms, cell, name, params, slot_species, dialect):
    """Miller index of a prototype slab: its primitive surface mesh shape is
    compared against the meshes of the dialect's candidate cuts."""
    meas = _inplane_primitive(P, syms, cell, dialect)
    if meas is None:
        return None
    r_m, a_m = _net_shape(*meas)
    tol_r = float(dialect.threshold("surface_net_ratio_tolerance"))
    tol_a = float(dialect.threshold("surface_net_angle_tolerance"))
    bulk = _prototype_bulk_atoms(name, params, slot_species)
    from ase.build import surface as ase_surface
    vac = float(dialect.threshold("surface_default_vacuum")) / 2
    best = None
    for hkl in dialect.threshold("surface_miller_candidates"):
        plane = tuple(int(x) for x in hkl)
        try:
            th = ase_surface(bulk, plane, layers=2, vacuum=vac)
        except Exception:
            continue
        net = _inplane_primitive(th.get_positions(), th.get_chemical_symbols(),
                                 np.array(th.cell), dialect)
        if net is None:
            continue
        r_t, a_t = _net_shape(*net)
        if abs(r_m - r_t) > tol_r or abs(a_m - a_t) > tol_a:
            continue
        score = abs(r_m - r_t) / tol_r + abs(a_m - a_t) / tol_a
        if best is None or score < best[0]:
            best = (score, plane)
    return None if best is None else best[1]


def _classify_sites(frame: Frame, ads_idx, top_idx, dialect):
    """top / bridge / hollow by lateral distance to the surface net features.

    The top layer is wrapped into the in-plane cell and replicated over its 8
    neighbouring periodic images before triangulation and KD queries, so an
    adsorbate at (or across) the cell boundary sees the same top / bridge /
    hollow features as one at the cell centre — the minimum-image-equivalent
    site assignment."""
    tol = float(dialect.threshold("adsorbate_site_tol"))
    Lxy = frame.cell_diag[:2]
    pts = np.mod(frame.pos[top_idx][:, :2], Lxy)
    # replicate the layer over the neighbouring images: the triangulation and
    # the feature trees then cover every feature seen across a boundary
    img = np.array([[i * Lxy[0], j * Lxy[1]]   # dialect-exempt: exact-geometry: integer image offsets over {-1,0,1}^2
                    for i in (-1, 0, 1) for j in (-1, 0, 1)])
    pts = (pts[None, :, :] + img[:, None, :]).reshape(-1, 2)
    tri = Delaunay(pts)
    tops = pts
    bridges, seen = [], set()
    for simplex in tri.simplices:
        for x, y in ((0, 1), (1, 2), (0, 2)):
            m = tuple(np.round((pts[simplex[x]] + pts[simplex[y]]) / 2, 3))
            if m not in seen:
                seen.add(m)
                bridges.append(m)
    hollows, seen = [], set()
    for simplex in tri.simplices:
        c = tuple(np.round(pts[simplex].mean(axis=0), 3))
        if c not in seen:
            seen.add(c)
            hollows.append(c)
    out = []
    for i in ads_idx:
        p = np.mod(frame.pos[i][:2], Lxy)
        d_top = cKDTree(tops).query(p)[0]
        d_br = cKDTree(np.array(bridges)).query(p)[0] if bridges else np.inf
        d_ho = cKDTree(np.array(hollows)).query(p)[0] if hollows else np.inf
        best = min(d_top, d_br, d_ho)
        if best > tol * 2:
            out.append(("far", best))
        elif best == d_top:
            out.append(("top", d_top))
        elif best == d_br:
            out.append(("bridge", d_br))
        else:
            out.append(("hollow", d_ho))
    return out


def _zone_census(frame: Frame, dialect, zone):
    """(overlayer census, zone census, atom mask) of the surface zone.

    The bond graph needs the whole periodic frame (a molecule may straddle the
    periodic boundary); only afterwards are the connected components filtered
    to those with every atom under the boolean `zone` mask. The zone census
    counts every fully-in-zone component (single atoms included, they feed the
    dissociation bookkeeping); the overlayer census counts the multi-atom
    molecules only, and the boolean mask marks their atoms."""
    from ..build.molecules import _formula, bond_graph
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    edges = bond_graph(frame, dialect)
    n = len(frame)
    if edges:
        rows = [e[0] for e in edges] + [e[1] for e in edges]
        cols = [e[1] for e in edges] + [e[0] for e in edges]
        _, labels = connected_components(
            coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)), directed=False)
    else:
        labels = np.arange(n)
    zone_census: dict[str, int] = {}
    overlayer: dict[str, int] = {}
    mask = np.zeros(n, bool)
    for g in range(labels.max() + 1):
        idx = np.where(labels == g)[0]
        if not bool(zone[idx].all()):
            continue
        formula = _formula([frame.symbols[i] for i in idx])
        zone_census[formula] = zone_census.get(formula, 0) + 1
        if len(idx) >= 2:
            overlayer[formula] = overlayer.get(formula, 0) + 1
            mask[idx] = True
    return overlayer, zone_census, mask


def lift_surface(frame: Frame, dialect, backend="eam") -> Program:
    """Lift a slab + vacuum (+ adsorbates / molecular overlayer) frame into a
    canonical Program."""
    syms = np.array(frame.symbols)

    # gap-aligned z: wrapped coordinates rotated so the slab stack starts at
    # z = 0 (invariant under rigid z translations, also across z = 0)
    z = _aligned_z(frame, dialect)

    # adsorbates: every atom above the slab's surface layer
    top_layer_idx, top_z = _top_layer(frame, dialect, z=z)
    slab_idx = np.where(z <= top_z + 0.5 * float(  # dialect-exempt: numerical-guard: half of the layer tolerance
        dialect.threshold("layer_tolerance")))[0]
    ads_idx = np.setdiff1d(np.arange(len(frame.pos)), slab_idx)

    # reactive overlayer: molecules of the surface zone (the top layer and
    # everything above) lift as a liquid `overlayer` region; single atoms of
    # the zone keep the adsorb route below. Without a molecular dialect there
    # is no bond rule and the frame lifts exactly as before.
    overlayer: dict[str, int] = {}
    zone_census: dict[str, int] = {}
    mol_mask = np.zeros(len(frame), bool)
    if len(ads_idx):
        zone = z > top_z - float(dialect.threshold("layer_tolerance"))
        try:
            overlayer, zone_census, mol_mask = _zone_census(frame, dialect, zone)
        except ChaordError:
            overlayer, zone_census, mol_mask = {}, {}, np.zeros(len(frame), bool)
        ads_idx = ads_idx[~mol_mask[ads_idx]]

    # bulk identification: a compound slab is matched against the prototype
    # registry through one stacking period; a unary slab keeps the M4 route
    # (interior coordination fixes the cubic family, NN distance fixes a)
    layers = _layers(frame, dialect, z=z)
    sub_idx = _sub_layer(layers, top_layer_idx)
    if len(set(syms[slab_idx])) > 1:
        name, params, slot_species = _identify_prototype_slab(frame, dialect, z=z)
        plane = _prototype_hkl(frame.pos[slab_idx], syms[slab_idx], frame.cell,
                               name, params, slot_species, dialect)
        if plane is None:
            raise ChaordError("cannot identify the Miller plane of the compound slab")
        hkl = f"({plane[0]}{plane[1]}{plane[2]})"
    else:
        from ..build.defects import typical_neighbor_distance
        from .defects import _A_FROM_DNN
        L = frame.cell_diag
        wrapped = frame.pos - L * np.floor(frame.pos / L)
        wrapped = np.minimum(wrapped, L * (1 - 1e-9))  # dialect-exempt: numerical-guard: strict upper edge for KD trees
        slab_pos = wrapped[slab_idx]
        dnn = typical_neighbor_distance(Frame(pos=slab_pos, cell=frame.cell,
                                              symbols=list(syms[slab_idx]), pbc=frame.pbc))
        from scipy.spatial import cKDTree
        tree = cKDTree(slab_pos, boxsize=L)
        cn_cut = float(dialect.threshold("slab_cn_factor")) * dnn
        mid_layer = layers[max(len(layers) // 2 - 1, 0)]
        interior = wrapped[mid_layer]
        cn = float(np.mean([len(x) - 1 for x in tree.query_ball_point(interior, cn_cut)]))
        fcc_min = float(dialect.threshold("fcc_cn_min"))
        bcc_min = float(dialect.threshold("bcc_cn_min"))
        diamond_min = float(dialect.threshold("diamond_cn_min"))
        if cn >= fcc_min:
            name, slot_species = "fcc", None
        elif cn >= bcc_min:
            name, slot_species = "bcc", None
        elif cn >= diamond_min:
            name, slot_species = "diamond", None
        else:
            raise ChaordError(f"slab interior coordination {cn:.1f} is neither "
                              f"fcc-, bcc- nor diamond-like")
        params = {"a": _A_FROM_DNN[name] * dnn}
        # orientation from the substrate net (the layer below the surface:
        # still bulk-terminated when the top layer is reconstructed)
        l1, l2, gamma = _surface_net(frame, sub_idx, dialect)
        if l2 == 0:  # degenerate substrate net: fall back to the surface layer
            l1, l2, gamma = _surface_net(frame, top_layer_idx, dialect)
        hkl = _identify_hkl(l1, l2, gamma, params["a"], dialect)

    # Wood reconstruction: the surface net against the substrate net
    wood = _wood_statement(
        _net_vectors(frame.pos[top_layer_idx][:, :2], frame.cell[:2, :2], dialect),
        _net_vectors(frame.pos[sub_idx][:, :2], frame.cell[:2, :2], dialect),
        dialect)

    # termination: the element of the top layer; compound slabs map it through
    # the dialect vocabulary (an O-terminated compound surface is bridging_O)
    termination = syms[top_layer_idx][0]
    for s in sorted(set(syms[top_layer_idx])):
        if s in _termination_names(dialect):
            termination = _termination_names(dialect)[s]
            break

    # conservation is per element over the whole frame: slab + adsorbates +
    # overlayer molecules all count (never drop an atom)
    counts: dict[str, int] = {}
    for s in frame.symbols:
        counts[s] = counts.get(s, 0) + 1
    conserve_values = []
    for s in sorted(counts):
        conserve_values += [Name(text=s), Quantity(num=str(counts[s]))]

    n_top = len(top_layer_idx)
    region_stmts = [
        Statement(kind="build", key="lattice", values=[_n2(name)]) if slot_species is None
        else Statement(kind="build", key="prototype", values=[_n2(name)]),
    ]
    for pname in PROTOTYPES[name].params:
        region_stmts.append(Statement(
            kind="build", key=pname,
            values=[Quantity(num=f"{params[pname]:.3f}", unit="A")]))
    region_stmts += [
        Statement(kind="build", key="surface", values=[
            Plane(text=hkl), _n2("top")]),
        Statement(kind="build", key="termination", values=[_n2(termination)]),
    ]
    if wood is not None:
        region_stmts.append(Statement(
            kind="build", key="reconstruction", values=[Wood(text=wood)]))
    if slot_species is not None and len(slot_species) > 1:
        from ..build.crystal import SLOT_COUNTS
        from ..lift.crystal import formula_from_slots
        region_stmts.insert(1, Statement(
            kind="build", key="composition",
            values=[_n2(formula_from_slots(name, slot_species))]))

    site_counts: dict[str, int] = {}
    if len(ads_idx):
        classes = _classify_sites(frame, ads_idx, top_layer_idx, dialect)
        ads_syms = syms[ads_idx]
        for (site, _d), s in zip(classes, ads_syms):
            site_counts.setdefault((s, site), 0)
            site_counts[(s, site)] += 1
        for (s, site), n in sorted(site_counts.items()):
            coverage = n / max(n_top, 1)
            region_stmts.append(Statement(
                kind="build", key="adsorb",
                values=[_n2(s), _n2("count"), Quantity(num=str(n)),
                        _n2("site"), _n2(site),
                        _n2("coverage"), Quantity(num=f"{coverage:.2f}", unit="ML")]))

    z_min = float(z[slab_idx].min())
    z_vac = float(frame.cell_diag[2])
    slab_h = top_z - z_min
    Lxy = frame.cell_diag[:2]

    system = SystemBlock(statements=[
        Statement(kind="build", key="cell", values=[
            Quantity(num=f"{Lxy[0]:.3f}"), Quantity(num=f"{Lxy[1]:.3f}"),
            Quantity(num=f"{z_vac:.3f}")]),
        Statement(kind="build", key="pbc", values=[_n2("xyz")]),
        Statement(kind="conserve", key="atoms", values=conserve_values),
    ])
    physics = PhysicsBlock(statements=[
        Statement(kind="build", key="backend", values=[_n2(backend)])])
    rng = RangeVal(lo=Quantity(num=f"{z_min:.1f}"), hi=Quantity(num=f"{top_z:.1f}"))
    region = RegionBlock(phase="crystal", name="slab",
                         geometry=GeoChain(parts=[ShSlab(axis="z", rng=rng)], ops=[]),
                         statements=region_stmts)
    vac = RegionBlock(phase="vacuum", name="gap",
                      geometry=GeoChain(parts=[ShSlab(axis="z", rng=RangeVal(
                          lo=Quantity(num=f"{top_z:.1f}"),
                          hi=Quantity(num=f"{z_vac:.1f}")))], ops=[]),
                      statements=[])
    interface_stmts = [
        Statement(kind="build", key="at", values=[
            _n2("z"), Quantity(num=f"{top_z:.1f}")]),
        Statement(kind="build", key="width", values=[
            Quantity(num=f"{float(dialect.threshold('vacuum_interface_width')):.1f}")]),
    ]
    blocks = [system, physics]
    if overlayer:
        from .reactive import display_name, dissociation_from_census
        # species definitions carry no source: the census identified them, no
        # SMILES is needed to re-build the templates
        blocks.append(SpeciesBlock(defs=[
            SpecDef(k="molecule", name=display_name(f))
            for f in sorted(overlayer, key=display_name)]))
        blocks.append(region)
        blocks.append(RegionBlock(
            phase="liquid", name="overlayer",
            geometry=GeoChain(parts=[ShSlab(axis="z", rng=RangeVal(
                lo=Quantity(num=f"{top_z:.1f}"),
                hi=Quantity(num=f"{z_vac:.1f}")))], ops=[]),
            statements=[Statement(
                kind="build", key="molecules",
                values=[_n2(display_name(f)), Quantity(num=str(overlayer[f]))])
                for f in sorted(overlayer, key=display_name)]))
        dissociation = dissociation_from_census(zone_census)
        if dissociation is not None:
            interface_stmts.append(dissociation)
        iface = InterfaceBlock(a="slab", b="overlayer",
                               statements=interface_stmts)
    else:
        blocks.append(region)
        iface = InterfaceBlock(a="slab", b="gap", statements=interface_stmts)
    blocks += [vac, iface, ResidualBlock(none=True),
               ProvenanceBlock(statements=[
                   Statement(kind="build", key="dialects",
                             values=[StrVal(text=dialect.version_string)]),
                   Statement(kind="build", key="lift_version",
                             values=[StrVal(text="0.1.0")]),
               ])]
    return Program(
        version="0.1", dialects=list(dialect.names),  # dialect-exempt: numerical-guard: language version constant
        blocks=blocks)


def _n2(text):
    return Name(text=text)
