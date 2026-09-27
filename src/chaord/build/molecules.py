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
# dialect-exempt-begin: exact molecular geometry (published bond lengths/angles)


def _register_mol(name, symbols, rel, charge=0):
    from ase.data import covalent_radii, chemical_symbols
    rel = np.asarray(rel, float)
    rel = rel - rel.mean(axis=0)   # place the template at its centroid
    # bonding extent: furthest atom centre plus its covalent radius, so packed
    # neighbours stay outside the bond-graph threshold of this molecule
    extent = max(float(np.linalg.norm(r)) + covalent_radii[chemical_symbols.index(s)]
                 for r, s in zip(rel, symbols))
    TEMPLATES[name] = dict(symbols=list(symbols), rel=rel, charge=charge,
                           radius=extent)


_register_mol("H2O", ["O", "H", "H"],
              [[0.0, 0.0, 0.0],
               [0.586, 0.757, 0.0],    # O-H 0.9572 A, H-O-H angle 104.52 deg
               [0.586, -0.757, 0.0]])
_register_mol("N2", ["N", "N"], [[0.0, 0.0, -0.55], [0.0, 0.0, 0.55]])   # 1.10 A
_register_mol("O2", ["O", "O"], [[0.0, 0.0, -0.605], [0.0, 0.0, 0.605]])  # 1.21 A
_register_mol("CO2", ["O", "C", "O"],
              [[0.0, 0.0, -1.16], [0.0, 0.0, 0.0], [0.0, 0.0, 1.16]])
_register_mol("Ar", ["Ar"], [[0.0, 0.0, 0.0]])
_register_mol("Na+", ["Na"], [[0.0, 0.0, 0.0]], charge=1)
_register_mol("Cl-", ["Cl"], [[0.0, 0.0, 0.0]], charge=-1)
_register_mol("Li+", ["Li"], [[0.0, 0.0, 0.0]], charge=1)
_register_mol("OH", ["O", "H"], [[0.0, 0.0, 0.0], [0.586, 0.757, 0.0]])  # O-H 0.9572 A
_register_mol("H", ["H"], [[0.0, 0.0, 0.0]])
# dialect-exempt-end


def random_rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def pack_molecules(counts: dict[str, int], box, rng, dialect) -> Frame:
    """Random-sequential packing of rigid molecules into a fresh box.

    Placement uses the molecular contact diameter (sum of template radii plus
    the dialect's packing gap); the physics prior relaxes what remains."""
    gap = float(dialect.threshold("packing_gap"))
    L = np.asarray(box, float)
    centers: list[np.ndarray] = []
    radii: list[float] = []
    syms: list[str] = []
    pos_list: list[np.ndarray] = []

    for name, n in counts.items():
        if name not in TEMPLATES:
            raise ChaordError(
                f"no template for molecule {name!r}; known: {', '.join(sorted(TEMPLATES))}")
        t = TEMPLATES[name]
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
                if np.linalg.norm(d) < t["radius"] + r2 + gap:
                    ok = False
                    break
            if not ok:
                continue
            centers.append(c)
            radii.append(t["radius"])
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


# --------------------------------------------------------------- detection --

def bond_graph(frame: Frame, dialect):
    """Neighbour graph under covalent-radius bonding (molecular dialect rule)."""
    from ase.data import covalent_radii, chemical_symbols
    from scipy.spatial import cKDTree
    tol = float(dialect.threshold("bond_tolerance"))
    L = frame.cell_diag
    syms = frame.symbols
    pos = np.mod(frame.pos, L)
    rc_max = max(covalent_radii[chemical_symbols.index(s)] for s in syms) * tol * 2
    tree = cKDTree(pos, boxsize=L)
    pairs = tree.query_pairs(rc_max, output_type="ndarray")
    edges = []
    for i, j in pairs:
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
