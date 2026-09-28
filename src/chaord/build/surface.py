"""Surface builder: Miller slabs, terminations, adsorbates (M4).

Cutting: the conventional cell is oriented with the surface normal along +z
using an integer orthogonal basis (cubic (001), (110), (111) families), tiled,
then trimmed to the slab geometry with vacuum added on top. Terminations pick
the cutting height whose top layer carries the termination element. Adsorbates
sit on top / bridge / hollow sites of the top-layer net at the dialect height.

Compound terminations (underscore names such as bridging_O) and prototypes
without a unary ASE builder are cut with ase.build.surface() on the prototype
bulk cell, scanning the cut origin along the plane normal until the top layer
carries the termination element. A `reconstruction <Wood>` statement thins the
top layer to the Wood superstructure by deleting atoms off the superlattice.
"""
from __future__ import annotations

import numpy as np

from ..io.frames import Frame
from ..lang.errors import ChaordError
from ..lang.ir import RegionBlock, Statement
from .crystal import SLOT_COUNTS
from .prototypes import PROTOTYPES, basis, cell_matrix


def _num(v) -> float:
    s = v.num if hasattr(v, "num") else str(v)
    if "/" in s:
        a, b = s.split("/")
        return float(a) / float(b)
    return float(s)


def _stmt_map(region: RegionBlock) -> dict[str, Statement]:
    return {s.key: s for s in region.statements if s.kind in ("build", "state", "constrain")}


def _parse_plane(text: str):
    inner = text.strip()[1:-1]
    import re
    vals = [int(x) for x in re.findall(r"-?\d", inner)]
    if len(vals) != 3:
        raise ChaordError(f"bad Miller plane {text!r}")
    return tuple(vals)


def orientation_for_plane(hkl: tuple) -> np.ndarray:
    """Integer orthogonal rows (u, v, w=hkl) for the cubic families we cut."""
    h, k, l = hkl
    if (h, k, l) == (0, 0, 1):
        return np.eye(3, dtype=int)
    if (h, k, l) == (1, 1, 0):
        return np.array([[1, -1, 0], [0, 0, 1], [1, 1, 0]], int)
    if (h, k, l) == (1, 1, 1):
        return np.array([[1, -1, 0], [1, 1, -2], [1, 1, 1]], int)
    raise ChaordError(
        f"surface ({h}{k}{l}): M4 supports cubic (001), (110) and (111) cuts")


def _element_of(name: str):
    import re
    m = re.search(r"([A-Z][a-z]?)", name)
    return m.group(1) if m else None


def _wood_stmt(region: RegionBlock):
    """The Wood token of the `reconstruction` statement, or None."""
    for s in region.statements:
        if s.key == "reconstruction":
            for v in s.values:
                if v.t == "wood":
                    return v.text
            raise ChaordError(
                f"reconstruction statement needs a Wood token like p(2x1), got "
                + " ".join(str(getattr(v, "text", v.t)) for v in s.values))
    return None


def _parse_wood(text: str):
    """(kind, n, m): the deterministic deletion rule of a Wood token.

    p(n x m) keeps one site per n x m cell corner; c(n x m) (n, m even) also
    keeps the cell centre; (r3 x r3)R30 keeps the sqrt(3) sublattice. Any other
    rotated form has no deterministic deletion rule and is rejected."""
    import re
    from math import lcm
    m = re.fullmatch(r"([pc])?\(\s*(\w+)\s*x\s*(\w+)\s*\)(?:R(\d+))?",
                     text.strip())
    if m is None:
        raise ChaordError(f"bad Wood token {text!r}")
    prefix, t1, t2, rot = m.groups()
    if prefix is None and t1 == t2 == "r3" and rot == "30":
        return ("r3r30", 1, 1), 3
    if prefix not in ("p", "c") or rot is not None or not (t1.isdigit() and t2.isdigit()):
        raise ChaordError(
            f"reconstruction {text!r}: only p(nx m), c(nx m) and (r3xr3)R30 are buildable")
    n, m2 = int(t1), int(t2)
    if n < 1 or m2 < 1 or (n, m2) == (1, 1):
        raise ChaordError(f"reconstruction {text!r} is the unreconstructed surface")
    if prefix == "c" and (n % 2 or m2 % 2):
        raise ChaordError(
            f"reconstruction {text!r}: c(nx m) needs even n and m for site deletion")
    return (prefix or "p", n, m2), lcm(n, m2)


def _top_indices(pos: np.ndarray, dialect):
    """Indices of the top z-layer of a freshly built slab (>= 2 atoms)."""
    tol = float(dialect.threshold("layer_tolerance"))
    z = pos[:, 2]
    order = np.argsort(z)
    layers, current = [], [order[0]]
    for i in order[1:]:
        if z[i] - z[current[-1]] > tol:
            layers.append(current)
            current = [i]
        else:
            current.append(i)
    layers.append(current)
    return np.array(layers[-1])


def _net_basis(pts2d, cell2d, dialect=None):
    """Two shortest independent primitive vectors of the top-layer net."""
    from ..dialects import load_dialect
    dialect = dialect if dialect is not None else load_dialect(("surface",))
    length_tol = float(dialect.threshold("net_length_tol"))
    area_tol = float(dialect.threshold("net_cell_area_tol"))
    A2 = np.asarray(cell2d, float)
    pts = np.mod(np.asarray(pts2d, float) @ np.linalg.inv(A2), 1.0) @ A2  # dialect-exempt: numerical-guard: fractional wrap
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
    area_per_point = abs(np.linalg.det(A2)) / len(pts)
    for n, w in cand:
        if abs(abs(np.dot(w, v1) / n) - np.linalg.norm(v1)) < length_tol * np.linalg.norm(v1):
            continue  # collinear with v1 (projected length matches |v1|)
        area = abs(v1[0] * w[1] - v1[1] * w[0])
        if abs(area - area_per_point) <= area_tol * area_per_point:
            return np.array(v1, float), np.array(w, float)
    return None


def _apply_reconstruction(frame: Frame, wood_text: str, dialect) -> Frame:
    """Delete top-layer atoms so the surviving net carries the Wood cell.

    Every surviving atom must sit on the sublattice generated by the Wood
    matrix in substrate-net coordinates (rows of the matrix are the new
    primitive vectors); p keeps cell corners, c adds the centre, (r3xr3)R30
    keeps the sqrt(3) sublattice."""
    rule, _period = _parse_wood(wood_text)
    kind = rule[0] if isinstance(rule, tuple) else rule
    top = _top_indices(frame.pos, dialect)
    net = _net_basis(frame.pos[top][:, :2], frame.cell[:2, :2], dialect)
    if net is None or len(top) < 2:
        raise ChaordError(
            f"reconstruction {wood_text!r}: the top layer is not a single lattice net")
    v1, v2 = net
    frac = (frame.pos[top][:, :2] - frame.pos[top[0], :2]) @ np.linalg.inv(np.array([v1, v2]))
    k = np.round(frac)
    tol = float(dialect.threshold("wood_commensurate_tol"))
    if np.abs(frac - k).max() > tol:
        raise ChaordError(
            f"reconstruction {wood_text!r}: top layer is not on a single net")
    k = k.astype(int)
    keep = np.zeros(len(top), bool)
    if kind == "p":
        keep = (k[:, 0] % rule[1] == 0) & (k[:, 1] % rule[2] == 0)
    elif kind == "c":
        r = np.mod(k, (rule[1], rule[2]))
        keep = ((r == (0, 0)).all(axis=1)
                | (r == (rule[1] // 2, rule[2] // 2)).all(axis=1))
    else:  # (r3xr3)R30: keep the sqrt(3) sublattice (basis-convention aware)
        cosg = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1)
        gamma = float(np.degrees(np.arccos(cosg)))
        s = k[:, 0] - k[:, 1] if gamma < 90 else k[:, 0] + k[:, 1]  # dialect-exempt: exact-geometry
        keep = np.mod(s, 3) == 0
    if not keep.any():
        raise ChaordError(f"reconstruction {wood_text!r} deletes every top-layer atom")
    drop = {int(i) for i, kp in zip(top, keep) if not kp}
    sel = [i for i in range(len(frame.pos)) if i not in drop]
    return Frame(pos=frame.pos[sel].copy(),
                 cell=frame.cell, symbols=[frame.symbols[i] for i in sel],
                 pbc=frame.pbc)


def _prototype_bulk(name: str, params: dict, slot_species):
    """One conventional cell of the prototype as ASE Atoms."""
    from ase import Atoms
    pos, slots = basis(name, params)
    return Atoms(symbols=[slot_species[s] for s in slots],
                 scaled_positions=pos, cell=cell_matrix(name, params), pbc=True)


def _prototype_surface(name, params, slot_species, hkl, termination, dialect):
    """Compound-termination slab: generic ASE cut with a cut-origin scan.

    ase.build.surface always cuts through the origin, so the bulk is shifted
    along the plane normal over one interplanar spacing; the shift whose top
    layer carries the termination element with the most atoms is kept."""
    from ase.build import surface as ase_surface
    bulk = _prototype_bulk(name, params, slot_species)
    cell = np.array(bulk.cell)
    g = np.array(hkl, float) @ np.linalg.inv(cell).T
    dhkl = 1 / np.linalg.norm(g)
    nhat = (np.array(hkl, float) @ cell)
    nhat = nhat / np.linalg.norm(nhat)
    element = _element_of(termination) if termination else None
    steps = int(dialect.threshold("termination_scan_steps"))
    layers = int(dialect.threshold("surface_default_layers"))
    vac = float(dialect.threshold("surface_default_vacuum")) / 2
    best = None
    for i in range(steps):
        t = dhkl * i / steps
        b = bulk.copy()
        shift = (t * nhat) @ np.linalg.inv(cell)
        b.set_scaled_positions(np.mod(b.get_scaled_positions() + shift, 1.0))  # dialect-exempt: numerical-guard: fractional wrap
        slab = ase_surface(b, tuple(int(x) for x in hkl), layers, vacuum=vac)
        top = _top_indices(slab.get_positions(), dialect)
        top_syms = [slab.symbols[int(j)] for j in top]
        has_el = element is not None and element in top_syms
        key = (1 if has_el else 0, len(top), -t)
        if best is None or key > best[0]:
            best = (key, slab)
    return best[1]


def build_surface(region: RegionBlock, system: dict, dialect, rng) -> Frame:
    """Miller slab via the ASE surface builders (fcc/bcc/diamond unary cuts).

    Termination and adsorbates are applied on the ASE slab: the termination
    statement names the element the top layer must carry, adsorbates are
    placed on top / bridge / hollow sites of the surface net."""
    stmts = _stmt_map(region)
    if "lattice" in stmts:
        name = stmts["lattice"].values[0].text
    elif "prototype" in stmts:
        name = stmts["prototype"].values[0].text
    else:
        raise ChaordError(f"region {region.name!r}: needs lattice or prototype")
    if "surface" not in stmts:
        raise ChaordError("surface region needs a `surface (hkl)` statement")
    hkl_text = stmts["surface"].values[0].text
    hkl = _parse_plane(hkl_text)
    key = { (0, 0, 1): "100", (1, 1, 0): "110", (1, 1, 1): "111" }.get(tuple(hkl))

    params = {k: _num(stmts[k].values[0])
              for k in PROTOTYPES[name].params if k in stmts}
    if set(params) != set(PROTOTYPES[name].params):
        raise ChaordError(f"prototype {name!r} missing parameters")

    wood_text = _wood_stmt(region)
    termination_text = None
    if "termination" in stmts:
        termination_text = stmts["termination"].values[0].text
    family = {"sc": "sc", "fcc": "fcc", "bcc": "bcc", "diamond": "diamond"}.get(name)
    # compound terminations (a name with an underscore, e.g. bridging_O) and
    # prototypes without a unary ASE builder go through the generic cut
    compound = family is None or (termination_text is not None and "_" in termination_text)
    if compound:
        if "composition" in stmts:
            from .crystal import slots_from_formula
            slot_species = slots_from_formula(name, stmts["composition"].values[0].text)
        else:
            slot_species = PROTOTYPES[name].default_slots
        atoms = _prototype_surface(name, params, slot_species, hkl,
                                   termination_text, dialect)
    else:
        if "composition" in stmts or len(SLOT_COUNTS[name]) > 1:
            raise ChaordError("M4 surface builder supports unary slabs")
        if key is None:
            raise ChaordError(f"surface {hkl_text}: M4 supports (001), (110), (111)")
        conserve = system.get("atoms")
        species = "X"
        if conserve is not None:
            named = [v.text for v in conserve.values if v.t == "n"]
            if named:
                species = named[0]

        from ase.build import bcc100, bcc110, bcc111, diamond100, diamond111, fcc100, fcc110, fcc111
        builders = {"fcc": {"100": fcc100, "110": fcc110, "111": fcc111},
                    "bcc": {"100": bcc100, "110": bcc110, "111": bcc111},
                    "diamond": {"100": diamond100, "111": diamond111}}
        if key not in builders[family]:
            raise ChaordError(f"no ASE surface builder for {name}({hkl_text})")
        layers = int(float(dialect.threshold("surface_default_layers")))
        vacuum = float(dialect.threshold("surface_default_vacuum"))
        size = (3, 3, layers)
        if wood_text is not None:
            # the surface box must hold whole reconstruction cells
            _rule, period = _parse_wood(wood_text)
            reps = -(-3 // period) * period  # smallest multiple of period >= 3
            size = (reps, reps, layers)
        kwargs = dict(symbol=species, size=size, a=params["a"], vacuum=vacuum / 2)
        atoms = builders[family][key](**kwargs)

    frame = Frame(pos=atoms.get_positions(), cell=atoms.get_cell().array,
                  symbols=list(atoms.get_chemical_symbols()), pbc=(True, True, True))
    # canonical: shift so the slab starts at z = 0 and the cell is z-positive
    frame.pos[:, 2] -= frame.pos[:, 2].min()

    if wood_text is not None:
        frame = _apply_reconstruction(frame, wood_text, dialect)
    for s in region.statements:
        if s.key == "adsorb":
            frame = _place_adsorbates(frame, s, rng, dialect)
    return frame


def _place_adsorbates(frame: Frame, stmt: Statement, rng, dialect) -> Frame:
    vals = stmt.values
    species = vals[0].text if vals[0].t == "n" else "O"
    site = "top"
    count = None
    coverage = None
    for i, v in enumerate(vals):
        if v.t == "n" and v.text == "site" and i + 1 < len(vals):
            site = vals[i + 1].text
        if v.t == "n" and v.text == "count" and i + 1 < len(vals):
            count = int(_num(vals[i + 1]))
        if v.t == "n" and v.text == "coverage" and i + 1 < len(vals):
            coverage = _num(vals[i + 1])
    height = float(dialect.threshold("adsorbate_height"))
    sites = _surface_sites(frame, dialect, site)
    if count is None and coverage is not None:
        count = int(round(coverage * len(_surface_sites(frame, dialect, "top"))))
    if count is None or count == 0:
        return frame
    if count > len(sites):
        raise ChaordError(f"cannot place {count} adsorbates on {len(sites)} {site} sites")
    chosen = rng.choice(len(sites), size=count, replace=False)
    top_z = frame.pos[:, 2].max()
    new_pos = [np.array([sites[i][0], sites[i][1], top_z + height]) for i in chosen]
    return Frame(pos=np.vstack([frame.pos, new_pos]),
                 cell=frame.cell, symbols=frame.symbols + [species] * count,
                 pbc=frame.pbc)


def _surface_sites(frame: Frame, dialect, site: str):
    """Top / bridge / hollow lateral sites of the top surface net."""
    from scipy.spatial import Delaunay
    tol_layer = float(dialect.threshold("layer_tolerance"))
    top_z = frame.pos[:, 2].max()
    top_idx = np.where(frame.pos[:, 2] > top_z - tol_layer)[0]
    pts = frame.pos[top_idx][:, :2]
    if site == "top":
        return [(p[0], p[1]) for p in pts]
    if site == "bridge":
        tri = Delaunay(pts)
        mids = []
        seen = set()
        for simplex in tri.simplices:
            for a, b in ((0, 1), (1, 2), (0, 2)):
                m = tuple(np.round((pts[simplex[a]] + pts[simplex[b]]) / 2, 3))
                if m not in seen:
                    seen.add(m)
                    mids.append(m)
        return mids
    if site in ("hollow", "fcc"):
        tri = Delaunay(pts)
        cents = []
        seen = set()
        for simplex in tri.simplices:
            c = tuple(np.round(pts[simplex].mean(axis=0), 3))
            if c not in seen:
                seen.add(c)
                cents.append(c)
        return cents
    raise ChaordError(f"unknown adsorption site {site!r}")
