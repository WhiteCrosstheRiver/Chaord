"""Molecule templates and packing (M3).

RDKit is the heavy path for arbitrary SMILES; the core ships exact templates
for the benchmark molecules (water, N2, O2, CO2, Ar, Na+, Cl-, Li+) and packs
them with a seeded random-sequential algorithm (Packmol is the optional extra).
"""
from __future__ import annotations

import numpy as np

from ..io.frames import Frame
from ..lang.errors import ChaordError

# Template geometry is exact published data (dialect-exempt per AGENTS.md).
TEMPLATES: dict[str, dict] = {}
# dialect-exempt-begin: exact-geometry


def _register_mol(name, symbols, rel, charge=0, pack_radius=None):
    from ase.data import covalent_radii, chemical_symbols
    rel = np.asarray(rel, float)
    rel = rel - rel.mean(axis=0)   # place the template at its centroid
    # bonding extent: furthest atom centre plus its covalent radius. Kept for
    # the extent-based packers outside this module (overlayer fixtures, bench
    # generation); pack_molecules itself packs with the tighter census-safe
    # radius computed from the dialect's bond tolerance (see packing_radius)
    extent = max(float(np.linalg.norm(r)) + covalent_radii[chemical_symbols.index(s)]
                 for r, s in zip(rel, symbols))
    TEMPLATES[name] = dict(symbols=list(symbols), rel=rel, charge=charge,
                           radius=extent, pack_radius=pack_radius)


_register_mol("H2O", ["O", "H", "H"],
              [[0.0, 0.0, 0.0],
               [0.586, 0.757, 0.0],    # O-H 0.9572 A, H-O-H angle 104.52 deg
               [0.586, -0.757, 0.0]])  # (TIP4P, the dialect default water model)
_register_mol("H2O/spce", ["O", "H", "H"],
              [[0.0, 0.0, 0.0],
               [0.57735, 0.81650, 0.0],   # O-H 1.0000 A, H-O-H 109.47 deg
               [0.57735, -0.81650, 0.0]])  # (SPC/E: Berendsen, Grigera &
                                          # Straatsma, J. Phys. Chem. 91,
                                          # 6269 (1987); the model-qualified
                                          # template name the fluid lift uses
                                          # for non-default water models -- see
                                          # the dialect's water_models table)
_register_mol("N2", ["N", "N"], [[0.0, 0.0, -0.55], [0.0, 0.0, 0.55]])   # 1.10 A
_register_mol("O2", ["O", "O"], [[0.0, 0.0, -0.605], [0.0, 0.0, 0.605]])  # 1.21 A
_register_mol("CO2", ["O", "C", "O"],
              [[0.0, 0.0, -1.16], [0.0, 0.0, 0.0], [0.0, 0.0, 1.16]])
_register_mol("Ar", ["Ar"], [[0.0, 0.0, 0.0]])
# monatomic ions pack at their Shannon (1976) CN6 ionic radius: the tabulated
# covalent radii of the alkali metals are metallic lengths (see the dialect's
# ion_solvation_elements), far too excluding for solvation-density packing
_register_mol("Na+", ["Na"], [[0.0, 0.0, 0.0]], charge=1, pack_radius=1.02)
_register_mol("Cl-", ["Cl"], [[0.0, 0.0, 0.0]], charge=-1, pack_radius=1.81)
_register_mol("Li+", ["Li"], [[0.0, 0.0, 0.0]], charge=1, pack_radius=0.90)
_register_mol("OH", ["O", "H"], [[0.0, 0.0, 0.0], [0.586, 0.757, 0.0]])  # O-H 0.9572 A
_register_mol("H", ["H"], [[0.0, 0.0, 0.0]])
# dialect-exempt-end


# --------------------------------------------------------------- water models --

def water_model_table(dialect) -> dict:
    """The dialect's ``water_models`` table (published rigid water models:
    geometry, classification windows, realization parameters).

    One canonical source: the molecular dialect's table. Dialect
    combinations that do not ship one (e.g. core+metal realizing a
    water/metal interface, whose liquid region still runs the classical
    backend) resolve the molecular table rather than a per-dialect copy of
    the published data -- two copies could fork, one cannot."""
    if dialect is None:
        raise ChaordError(
            "no dialect: water models resolve through the molecular "
            "dialect's water_models table")
    try:
        table = dialect.threshold("water_models")
    except ChaordError:
        from ..dialects import load_dialect
        table = load_dialect(("core", "molecular")).threshold("water_models")
    if not isinstance(table, dict):
        raise ChaordError(
            "the water_models threshold must be a table of models "
            "(see the molecular dialect)")
    return table


def classify_water_model(dialect, r_oh_median) -> str | None:
    """Which rigid water model does a median O-H distance belong to?

    One definition, used by both directions: the fluid lift classifies a
    frame with it and the classical realization picks its potential with it.
    A model claims the window ``r_oh_A +- classify_half_width_A``; exactly one
    window may match. None means the geometry is no known rigid model
    (non-rigid water, a mixture, or an unknown potential) and the chain falls
    back to the table's default."""
    matches = [name for name, spec in water_model_table(dialect).items()
               if isinstance(spec, dict)
               and abs(float(r_oh_median) - float(spec["r_oh_A"]))
               <= float(spec["classify_half_width_A"])]
    return matches[0] if len(matches) == 1 else None


def default_water_model(dialect) -> str:
    return str(water_model_table(dialect)["default"])


def water_template_name(model: str, dialect) -> str:
    """Program species name of a water model's template: ``H2O`` for the
    table's default model (byte-identical to the pre-model-statement
    programs), ``H2O/<model>`` for any other (the builder's template
    selector: `H2O` packs the default geometry, `H2O/spce` the SPC/E one)."""
    if model == default_water_model(dialect):
        return "H2O"
    return f"H2O/{model}"


def measure_water_oh_median(frame: Frame, dialect) -> float | None:
    """Median minimum-image O-H distance over the frame's water molecules
    (bond-graph H2O components).

    Rigid MD conserves every O-H bond length, so this median is a frame
    invariant that names the water model (``classify_water_model``); flexible
    water or a geometry mixture spreads it and classification honestly
    fails. None when the frame has no H2O component."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    edges = bond_graph(frame, dialect)
    n = len(frame)
    if not edges:
        return None
    rows = [e[0] for e in edges] + [e[1] for e in edges]
    cols = [e[1] for e in edges] + [e[0] for e in edges]
    _, labels = connected_components(
        coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)),
        directed=False)
    L = frame.cell_diag
    pos = np.mod(frame.pos, L)
    d_oh = []
    for g in range(labels.max() + 1):
        idx = np.where(labels == g)[0]
        syms = [frame.symbols[i] for i in idx]
        if sorted(syms) != ["H", "H", "O"]:
            continue
        o = idx[[frame.symbols[i] == "O" for i in idx]][0]
        for h in idx:
            if h == o:
                continue
            d = pos[h] - pos[o]
            d -= L * np.round(d / L)
            d_oh.append(float(np.linalg.norm(d)))
    return float(np.median(d_oh)) if d_oh else None


def random_rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def packing_radius(name: str, dialect) -> float:
    """Census-safe packing radius of a template molecule under `dialect`.

    For every atom: its displacement from the centroid plus `bond_tolerance`
    times its covalent radius. Two molecules whose centres are at least the
    sum of their packing radii apart can never be perceived as bonded,
    whatever their relative orientation: the worst case (both bonding atoms
    aligned with the centre line) leaves that pair exactly at
    bond_tolerance x (r_i + r_j). Putting the tolerance inside the radius (the
    old extent added the bare covalent radii and a 0.6 A gap on top) keeps the
    water contact near the O-O hard core instead of past the random-sequential
    jamming limit, so true liquid densities are packable.

    A template with a published `pack_radius` override (the monatomic ions:
    Shannon ionic radii; the alkali covalent radii are metallic lengths, far
    too excluding at solvation density) packs at that instead -- census safety
    holds because the dialect's ion_solvation_elements rule never bonds those
    elements anyway, and the shipped overrides (Cl- 1.81 > 1.25 x 1.02)
    exceed the covalent rule for the rest."""
    t = TEMPLATES[name]
    if t.get("pack_radius") is not None:
        return float(t["pack_radius"])
    from ase.data import covalent_radii, chemical_symbols
    tol = float(dialect.threshold("bond_tolerance"))
    return max(float(np.linalg.norm(r)) + tol * covalent_radii[chemical_symbols.index(s)]
               for r, s in zip(t["rel"], t["symbols"]))


def pack_molecules(counts: dict[str, int], box, rng, dialect) -> Frame:
    """Random-sequential packing of rigid molecules into a fresh box.

    Placement uses the census-safe contact distance (sum of packing radii plus
    the dialect's numeric margin); the physics prior relaxes what remains."""
    margin = float(dialect.threshold("packing_contact_margin"))
    L = np.asarray(box, float)
    radii_of = {name: packing_radius(name, dialect) for name in counts}
    centers: list[np.ndarray] = []
    radii: list[float] = []
    syms: list[str] = []
    pos_list: list[np.ndarray] = []

    for name, n in counts.items():
        if name not in TEMPLATES:
            raise ChaordError(
                f"no template for molecule {name!r}; known: {', '.join(sorted(TEMPLATES))}")
        t = TEMPLATES[name]
        r_new = radii_of[name]
        placed = 0
        tries = 0
        max_tries = int(dialect.threshold("packing_max_tries"))
        while placed < n:
            tries += 1
            if tries > max_tries * max(n, 1):
                raise ChaordError(
                    f"cannot place {n} {name} at this density "
                    f"(packing failed after {tries} tries)")
            c = rng.uniform(0, L)
            R = random_rotation(rng)
            atoms = t["rel"] @ R.T + c
            ok = True
            for c2, r2 in zip(centers, radii):
                d = c - c2
                d -= L * np.round(d / L)
                if np.linalg.norm(d) < r_new + r2 + margin:
                    ok = False
                    break
            if not ok:
                continue
            centers.append(c)
            radii.append(r_new)
            pos_list.append(atoms)
            syms.extend(t["symbols"])
            placed += 1
    pos = np.vstack(pos_list) if pos_list else np.zeros((0, 3))
    pos = np.mod(pos, L)  # wrap into the box
    return Frame(pos=pos, cell=np.diag(L), symbols=syms, pbc=(True, True, True))


def molecular_mass(name: str) -> float:
    from ase.data import atomic_masses, chemical_symbols
    t = TEMPLATES[name]
    return float(sum(atomic_masses[chemical_symbols.index(s)] for s in t["symbols"]))


def species_mass(name: str) -> float:
    """Mass in u of a program species: a template molecule or a bare element.

    Anything else (a census formula the builder has no species for, e.g. a
    hydration shell 'H10NaO5') is a static error, not a KeyError/ValueError
    from deep inside a mass table. The placeholder 'X' (LJ reduced units,
    one particle kind) carries unit mass, as in the amorphous builder."""
    from ase.data import atomic_masses, chemical_symbols
    if name in TEMPLATES:
        return molecular_mass(name)
    if name == "X":
        return 1.0  # dialect-exempt: numerical-guard: unit-mass placeholder species
    if name in chemical_symbols:
        return float(atomic_masses[chemical_symbols.index(name)])
    raise ChaordError(
        f"unknown species {name!r}; known molecules: "
        f"{', '.join(sorted(TEMPLATES))}; otherwise a bare element symbol")


# --------------------------------------------------------------- detection --

def _ion_solvation_elements(dialect) -> frozenset[str]:
    """Elements the dialect declares as solvation-shell ions (empty without the
    rule, e.g. under the glass or lj dialects)."""
    try:
        return frozenset(dialect.threshold("ion_solvation_elements") or ())
    except ChaordError:
        return frozenset()


def bond_graph(frame: Frame, dialect):
    """Neighbour graph under covalent-radius bonding (molecular dialect rule).

    Elements named by the dialect's `ion_solvation_elements` never bond: in a
    molecular fluid their contacts are ion solvation (their tabulated covalent
    radii are metallic, so Na+-O at ~2.4 A would pass the covalent test and
    merge the ion with its hydration shell in the census)."""
    from ase.data import covalent_radii, chemical_symbols
    from scipy.spatial import cKDTree
    tol = float(dialect.threshold("bond_tolerance"))
    ions = _ion_solvation_elements(dialect)
    L = frame.cell_diag
    syms = frame.symbols
    pos = np.mod(frame.pos, L)
    pos = np.minimum(pos, L * (1 - 1e-9))  # dialect-exempt: strict upper edge for KD trees
    rc_max = max(covalent_radii[chemical_symbols.index(s)] for s in syms) * tol * 2
    tree = cKDTree(pos, boxsize=L)
    pairs = tree.query_pairs(rc_max, output_type="ndarray")
    edges = []
    for i, j in pairs:
        if syms[i] in ions or syms[j] in ions:
            continue  # ion--solvent contact: solvation, not covalence
        d = pos[j] - pos[i]
        d -= L * np.round(d / L)
        dist = float(np.linalg.norm(d))
        ri = covalent_radii[chemical_symbols.index(syms[i])]
        rj = covalent_radii[chemical_symbols.index(syms[j])]
        if dist < tol * (ri + rj):
            edges.append((i, j))
    return edges


def molecule_census(frame: Frame, dialect) -> dict[str, int]:
    """Count molecules by canonical formula (connected components of the bond graph)."""
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
    counts: dict[str, int] = {}
    for g in range(labels.max() + 1):
        idx = np.where(labels == g)[0]
        formula = _formula([frame.symbols[i] for i in idx])
        counts[formula] = counts.get(formula, 0) + 1
    return counts


def _formula(symbols: list[str]) -> str:
    """Canonical molecular formula (C first, H second, then alphabetical)."""
    from collections import Counter
    c = Counter(symbols)
    parts = []
    order = [el for el in ("C", "H") if el in c] + sorted(k for k in c if k not in ("C", "H"))
    for el in order:
        parts.append(f"{el}{c[el] if c[el] > 1 else ''}")
    return "".join(parts)
