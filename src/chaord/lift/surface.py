"""Surface lifter: slab + vacuum + adsorbates -> canonical program (M4)."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree, Delaunay

from ..io.frames import Frame
from ..lang.errors import ChaordError
from ..lang.ir import (
    GeoChain, InterfaceBlock, Name, PhysicsBlock, Plane, Program,
    ProvenanceBlock, Quantity, RangeVal, RegionBlock, ResidualBlock, ShSlab,
    Statement, StrVal, SystemBlock,
)


def has_vacuum(frame: Frame, dialect) -> bool:
    """A z-gap in the atom distribution wider than the dialect threshold."""
    try:
        gap_min = float(dialect.threshold("vacuum_gap_min"))
    except Exception:
        return False   # no surface dialect loaded: no vacuum concept
    L = frame.cell_diag
    binw = float(dialect.threshold("profile_bin_size"))
    n = max(int(L[2] / binw), 8)
    idx = np.clip((frame.pos[:, 2] / L[2] * n).astype(int), 0, n - 1)
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


def _layers(frame: Frame, dialect):
    """Cluster atoms into z-layers (gap > layer_tolerance separates)."""
    tol = float(dialect.threshold("layer_tolerance"))
    z = frame.pos[:, 2]
    order = np.argsort(z)
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


def _top_layer(frame: Frame, dialect):
    """Slab top layer: the highest layer carrying a substantial atom count.

    Adsorbate layers sit above it with only a few atoms; the surface layer has
    at least half the median layer size."""
    layers = _layers(frame, dialect)
    sizes = [len(l) for l in layers if len(l) >= 2] or [len(layers[-1])]
    median = float(np.median(sizes))
    for layer in reversed(layers):
        if len(layer) >= 0.5 * median and len(layer) >= 2:  # dialect-exempt: layer size floor
            return layer, float(frame.pos[layer, 2].max())
    return layers[-1], float(frame.pos[layers[-1], 2].max())


def _surface_net(frame: Frame, top_idx, dialect):
    """Primitive in-plane lattice vectors of the top layer.

    v1 is the shortest inter-atom vector; v2 is the shortest independent
    vector whose cell area matches the point density (rejecting 2x cells)."""
    pts = frame.pos[top_idx][:, :2]
    A2 = frame.cell[:2, :2]                 # true in-plane lattice (may be hex)
    pts = np.mod(pts @ np.linalg.inv(A2), 1.0) @ A2  # dialect-exempt: fractional wrap
    p0 = pts[0]
    cand = []
    for q in pts:
        v = q - p0
        for na in (-1, 0, 1):
            for nb in (-1, 0, 1):
                w = v + na * A2[0] + nb * A2[1]
                n = np.linalg.norm(w)
                if n > 0.5:  # dialect-exempt: minimal vector length
                    cand.append((n, w))
    cand.sort(key=lambda x: x[0])
    v1 = cand[0][1]
    area_per_point = abs(A2[0][0] * A2[1][1] - A2[0][1] * A2[1][0]) / len(pts)
    for n, w in cand:
        if abs(abs(np.dot(w, v1) / n) - np.linalg.norm(v1)) < 0.1 * np.linalg.norm(v1):  # dialect-exempt: collinearity
            continue  # dialect-exempt: collinearity tolerance
        area = abs(v1[0] * w[1] - v1[1] * w[0])  # 2D cross product
        if abs(area - area_per_point) <= 0.2 * area_per_point:  # dialect-exempt: cell-area tolerance
            v2 = w
            return (float(np.linalg.norm(v1)), float(n),
                    float(np.degrees(np.arccos(np.clip(
                        np.dot(v1, w) / (np.linalg.norm(v1) * n), -1, 1)))))
    return float(np.linalg.norm(v1)), 0.0, 90.0  # dialect-exempt: degenerate net


# dialect-exempt-begin: surface-net classification levels (crystallographic)
def _identify_hkl(l1, l2, gamma, a_bulk):
    tol = 0.12  # dialect-exempt: relative net-matching tolerance
    if abs(l1 - l2) / max(l1, l2) < tol and abs(gamma - 90) < 8:
        return "(001)"          # square net
    if abs(gamma - 90) < 8 and abs(l2 / l1 - 2**0.5) / 2**0.5 < tol:
        return "(110)"          # rectangular a x a*sqrt(2)
    if abs(l1 - l2) / max(l1, l2) < tol and abs(gamma - 60) < 8:
        return "(111)"          # hexagonal net
    if abs(gamma - 120) < 8 and abs(l1 - l2) / max(l1, l2) < tol:
        return "(111)"
    return "(001)"
# dialect-exempt-end


def _classify_sites(frame: Frame, ads_idx, top_idx, dialect):
    """top / bridge / hollow by lateral distance to the surface net features."""
    tol = float(dialect.threshold("adsorbate_site_tol"))
    pts = frame.pos[top_idx][:, :2]
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
        p = frame.pos[i][:2]
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


def lift_surface(frame: Frame, dialect, backend="eam") -> Program:
    """Lift a slab + vacuum (+ adsorbates) frame into a canonical Program."""
    syms = np.array(frame.symbols)

    # adsorbates: every atom above the slab's surface layer
    top_layer_idx, top_z = _top_layer(frame, dialect)
    slab_idx = np.where(frame.pos[:, 2] <= top_z + 0.5 * float(  # dialect-exempt: half layer tolerance
        dialect.threshold("layer_tolerance")))[0]
    ads_idx = np.setdiff1d(np.arange(len(frame.pos)), slab_idx)

    # bulk lattice from orientation-invariant quantities: interior coordination
    # fixes the cubic family, the typical NN distance fixes a (no box alignment
    # is assumed: the slab is rotated so the surface normal is +z)
    from ..build.defects import typical_neighbor_distance
    from .defects import _A_FROM_DNN
    L = frame.cell_diag
    wrapped = frame.pos - L * np.floor(frame.pos / L)
    wrapped = np.minimum(wrapped, L * (1 - 1e-9))  # dialect-exempt: strict upper edge for KD trees
    slab_pos = wrapped[slab_idx]
    dnn = typical_neighbor_distance(Frame(pos=slab_pos, cell=frame.cell,
                                          symbols=list(syms[slab_idx]), pbc=frame.pbc))
    from scipy.spatial import cKDTree
    tree = cKDTree(slab_pos, boxsize=L)
    cn_cut = float(dialect.threshold("slab_cn_factor")) * dnn
    mid_layer = _layers(frame, dialect)[max(len(_layers(frame, dialect)) // 2 - 1, 0)]
    interior = wrapped[mid_layer]
    cn = float(np.mean([len(x) - 1 for x in tree.query_ball_point(interior, cn_cut)]))
    fcc_min = float(dialect.threshold("fcc_cn_min"))
    bcc_min = float(dialect.threshold("bcc_cn_min"))
    if cn >= fcc_min:
        name, slot_species = "fcc", None
    elif cn >= bcc_min:
        name, slot_species = "bcc", None
    else:
        raise ChaordError(f"slab interior coordination {cn:.1f} is neither fcc-like nor bcc-like")
    a = _A_FROM_DNN[name] * dnn

    l1, l2, gamma = _surface_net(frame, top_layer_idx, dialect)
    hkl = _identify_hkl(l1, l2, gamma, a)

    # termination: the element of the top layer
    termination = syms[top_layer_idx][0]

    counts: dict[str, int] = {}
    for s in list(syms[slab_idx]):
        counts[s] = counts.get(s, 0) + 1
    conserve_values = []
    for s in sorted(counts):
        conserve_values += [Name(text=s), Quantity(num=str(counts[s]))]

    n_top = len(top_layer_idx)
    region_stmts = [
        Statement(kind="build", key="lattice", values=[_n2(name)]) if slot_species is None
        else Statement(kind="build", key="prototype", values=[_n2(name)]),
        Statement(kind="build", key="a", values=[
            Quantity(num=f"{a:.3f}", unit="A")]),
        Statement(kind="build", key="surface", values=[
            Plane(text=hkl), _n2("top")]),
        Statement(kind="build", key="termination", values=[_n2(termination)]),
    ]
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

    z_min = float(frame.pos[slab_idx][:, 2].min())
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
    iface = InterfaceBlock(a="slab", b="gap", statements=[
        Statement(kind="build", key="at", values=[
            _n2("z"), Quantity(num=f"{top_z:.1f}")]),
        Statement(kind="build", key="width", values=[
            Quantity(num=f"{float(dialect.threshold('vacuum_interface_width')):.1f}")]),
    ])
    return Program(
        version="0.1", dialects=list(dialect.names),  # dialect-exempt: language version
        blocks=[system, physics, region, vac, iface, ResidualBlock(none=True),
                ProvenanceBlock(statements=[
                    Statement(kind="build", key="dialects",
                              values=[StrVal(text=dialect.version_string)]),
                    Statement(kind="build", key="lift_version",
                              values=[StrVal(text="0.1.0")]),
                ])])


def _n2(text):
    return Name(text=text)
