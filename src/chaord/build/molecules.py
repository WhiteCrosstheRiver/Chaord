"""Molecule templates and packing (M3).

RDKit is the heavy path for arbitrary SMILES; the core ships exact templates
for the benchmark molecules (water, N2, O2, CO2, Ar, Na+, Cl-, Li+) and the
electrolyte species (EC, PF6-; W12 -- general species templates, also the
vocabulary of the later PC/EMC/EC/PF6/Li research case) and packs
them with a seeded random-sequential algorithm (Packmol is the optional extra).
"""
from __future__ import annotations

import numpy as np

from ..io.frames import Frame
from ..lang.errors import ChaordError


class PackingJam(ChaordError):
    """Random-sequential placement ran out of budget (W8): the jam the grid +
    minimise fallback exists to recover from.  Still a ChaordError, so every
    existing caller that catches the packing failure keeps working."""

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
# electrolyte species (W12): general SPECIES templates, not tied to one case
# -- the vocabulary of solutions/lipf6_ec and the later PC/EMC/EC/PF6/Li
# research case. Promoted VERBATIM from bench/generate.py (where they were
# case-local registrations): the frozen lipf6_ec reference frames carry this
# exact geometry, so template, frames and census stay one geometry.


def _ec_rel():
    """Ethylene carbonate (1,3-dioxolan-2-one, C3H4O3), planar: a regular
    five-membered ring of 1.43 A edges (ester C-O single-bond length; the
    C-C ring edge is drawn at the same length as the generator's simplified
    ring), carbonyl C=O 1.20 A, C-H 1.09 A with a 35-deg H-C-H spread.
    Bond lengths from standard organic geometry (F. H. Allen, O. Kennard,
    D. G. Watson, L. Brammer, A. G. Orpen, R. Taylor, J. Chem. Soc. Perkin
    Trans. 2 (1987) S1-S19: ester C-O 1.43 A, ketone C=O 1.21 A, sp3 C-H
    1.09 A); the construction is bench/generate.py's, the geometry the
    frozen lipf6_ec frames were generated with."""
    R = 1.43 / (2.0 * np.sin(np.deg2rad(36.0)))   # pentagon circumradius
    ring = {}
    for name, deg in (("C1", 90), ("O2", 162), ("C3", 234), ("C4", 306),
                      ("O5", 18)):
        ring[name] = R * np.array([np.cos(np.deg2rad(deg)),
                                   np.sin(np.deg2rad(deg)), 0.0])
    ring["O6"] = ring["C1"] + np.array([0.0, 1.20, 0.0])   # carbonyl C=O

    def hydrogens(c):
        r = np.asarray(c[:2], float)
        r = r / np.linalg.norm(r)
        perp = np.array([-r[1], r[0]])
        out = []
        for side in (1.0, -1.0):
            d = (np.cos(np.deg2rad(35.0)) * r
                 + side * np.sin(np.deg2rad(35.0)) * perp)
            out.append(np.array([c[0] + 1.09 * d[0], c[1] + 1.09 * d[1], 0.0]))
        return out

    h3, h4 = hydrogens(ring["C3"]), hydrogens(ring["C4"])
    return [ring["C1"], ring["O2"], ring["C3"], ring["C4"], ring["O5"],
            ring["O6"], h3[0], h3[1], h4[0], h4[1]]


_register_mol("EC", ["C", "O", "C", "C", "O", "O", "H", "H", "H", "H"],
              _ec_rel(), charge=0)
_register_mol("PF6-", ["P"] + ["F"] * 6,                # octahedral P-F
              [[0.0, 0.0, 0.0], [1.58, 0.0, 0.0], [-1.58, 0.0, 0.0],
               [0.0, 1.58, 0.0], [0.0, -1.58, 0.0], [0.0, 0.0, 1.58],
               [0.0, 0.0, -1.58]], charge=-1)            # P-F 1.58 A:
                                                         # crystallographic
                                                         # hexafluorophosphate
                                                         # (LiPF6 structures);
                                                         # the value the frozen
                                                         # lipf6_ec frames and
                                                         # the A12 charge tests
                                                         # use
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


# ------------------------------------------------------------------ W8 floor --

def contact_floor_target(si: str, sj: str, dialect) -> float | None:
    """W8 contact-floor distance for one unordered intermolecular element
    pair, or None when the rule does not constrain the pair.

    One definition, used by the builder's pack-then-relax placement (build),
    the bench sanity check (assert) and the W8 regression tests: heavy-atom
    pairs sit at >= contact_floor_bondi_fraction x the Bondi van der Waals
    sum; an H...O/N hydrogen bond needs only contact_floor_hbond_min; other
    H-involving pairs are outside the rule; ion_solvation_elements pairs are
    exempt exactly as in the bond graph (a hydration contact is ion
    solvation, not a van der Waals overlap).  Radii: the Bondi set shipped
    in ase.data.vdw_radii (A. Bondi, J. Phys. Chem. 68, 441 (1964))."""
    ions = _ion_solvation_elements(dialect)
    if si in ions or sj in ions:
        return None
    from ase.data import chemical_symbols, vdw_radii
    if si != "H" and sj != "H":
        frac = float(dialect.threshold("contact_floor_bondi_fraction"))
        return frac * (float(vdw_radii[chemical_symbols.index(si)])
                       + float(vdw_radii[chemical_symbols.index(sj)]))
    if (si == "H" and sj in ("O", "N")) or (sj == "H" and si in ("O", "N")):
        return float(dialect.threshold("contact_floor_hbond_min"))
    return None


def _relax_contact_target(si: str, sj: str, dialect) -> float | None:
    """Relaxation target of one intermolecular pair: the W8 floor, never
    below the census-safe covalent window, plus contact_floor_margin.

    The window bound keeps the placement census-safe by construction (the
    packing_radius argument of pack_molecules, applied pairwise): two
    molecules whose every atom pair sits beyond this can never be perceived
    as bonded, whatever their relative orientation."""
    ions = _ion_solvation_elements(dialect)
    if si in ions or sj in ions:
        return None
    floor = contact_floor_target(si, sj, dialect)
    from ase.data import chemical_symbols, covalent_radii
    tol = float(dialect.threshold("bond_tolerance"))
    window = tol * (float(covalent_radii[chemical_symbols.index(si)])
                    + float(covalent_radii[chemical_symbols.index(sj)]))
    margin = float(dialect.threshold("contact_floor_margin"))
    return max(floor if floor is not None else 0.0, window) + margin  # dialect-exempt: numerical-guard: max() identity for an unconstrained floor


class _RigidBlock:
    """One rigid body of the packing: element symbols + atom offsets from
    the (unwrapped) body centre."""

    __slots__ = ("symbols", "offsets")

    def __init__(self, symbols, offsets):
        self.symbols = list(symbols)
        self.offsets = np.asarray(offsets, float)


def _relax_rigid_blocks(centres: np.ndarray, blocks, L: np.ndarray,
                        dialect) -> np.ndarray:
    """Rigid-body contact-opening relaxation (the 'minimise' half of the W8
    grid + minimise packing): every intermolecular pair below its
    _relax_contact_target pushes the two rigid bodies apart along the pair
    axis by a grid_pack_relax_rate fraction of the deficit, until no
    violating pair is left.  Deterministic (no randomness inside the loop).
    Returns wrapped atom positions in block order, or raises ChaordError
    after grid_pack_relax_steps iterations (fail closed -- never hand out a
    silently sub-floor frame)."""
    from scipy.spatial import cKDTree
    steps = int(dialect.threshold("grid_pack_relax_steps"))
    rate = float(dialect.threshold("grid_pack_relax_rate"))
    symbols = [s for blk in blocks for s in blk.symbols]
    owner = np.repeat(np.arange(len(blocks)),
                      [len(b.symbols) for b in blocks])
    offsets = np.vstack([b.offsets for b in blocks])
    targets: dict[tuple[str, str], float | None] = {}
    present = sorted(set(symbols))
    for a in present:
        for b in present:
            targets[(a, b)] = _relax_contact_target(a, b, dialect)
    finite = [t for t in targets.values() if t is not None]
    if not finite:
        return np.mod(centres[owner] + offsets, L)
    rcut = max(finite)  # dialect-exempt: numerical-guard: scan radius = the largest relaxation target
    sym_idx = {s: i for i, s in enumerate(present)}
    tmat = np.full((len(present), len(present)), np.nan)
    for (a, b), t in targets.items():
        if t is not None:
            tmat[sym_idx[a], sym_idx[b]] = t
    sidx = np.array([sym_idx[s] for s in symbols])
    for _ in range(steps):
        pos = np.mod(centres[owner] + offsets, L)
        pos = np.minimum(pos, L * (1 - 1e-9))  # dialect-exempt: numerical-guard: strict upper edge for KD trees
        tree = cKDTree(pos, boxsize=L)
        pairs = tree.query_pairs(rcut, output_type="ndarray")
        if not len(pairs):
            return pos
        tgt = tmat[sidx[pairs[:, 0]], sidx[pairs[:, 1]]]
        dv = pos[pairs[:, 1]] - pos[pairs[:, 0]]
        dv -= L * np.round(dv / L)
        d = np.linalg.norm(dv, axis=1)
        inter = owner[pairs[:, 0]] != owner[pairs[:, 1]]
        deficit = np.where(np.isnan(tgt), -np.inf, tgt - d)  # dialect-exempt: numerical-guard: NaN target = unconstrained pair
        bad = inter & (deficit > 1e-9)  # dialect-exempt: numerical-guard: convergence tolerance
        if not bad.any():
            return pos
        u = dv[bad] / d[bad][:, None]
        push = rate * deficit[bad][:, None] * u / 2.0  # dialect-exempt: numerical-guard: split each pair's correction between its two bodies
        disp = np.zeros_like(centres)
        np.add.at(disp, owner[pairs[bad, 0]], -push)
        np.add.at(disp, owner[pairs[bad, 1]], push)
        centres = np.mod(centres + disp, L)
    raise ChaordError(
        f"grid + minimise packing did not clear the intermolecular contact "
        f"floor in {steps} relaxation steps at this density")


def _grid_centres(total: int, L: np.ndarray, rng) -> np.ndarray:
    """Cell-centred simple-cubic grid prior (the W8 'grid' half): the
    smallest k x k x k grid with k^3 >= total sites, total of them kept by
    seeded choice -- the same placement the water reference uses for its O
    sites (bench/reference/generate_reference.py make_water_molecules,
    grid=7)."""
    k = max(1, int(np.ceil(total ** (1 / 3))))
    while k ** 3 < total:
        k += 1
    g = np.array(np.meshgrid(*[np.arange(k)] * 3, indexing="ij")
                 ).reshape(3, -1).T
    sites = (g + 0.5) * (L / k)  # dialect-exempt: exact-geometry: cell-centred grid sites
    return sites[rng.choice(len(sites), total, replace=False)]


def grid_pack_molecules(counts: dict[str, int], box, rng, dialect) -> Frame:
    """Grid + minimise packing of rigid molecules (W8 builder fallback).

    The placement the molecular reference cases use, carried into the
    builder for molecular fluids whose whole-molecule random-sequential
    packing jams (CO2 at liquid density: its census-safe sphere sits above
    the RSA saturation fraction while the rod-shaped molecule itself does
    not): molecule centres on a cell-centred cubic-grid subset with seeded
    random orientations (the 'grid'), then the deterministic rigid-body
    contact-opening relaxation of _relax_rigid_blocks (the 'minimise').

    Every intermolecular pair ends beyond max(W8 contact floor, census-safe
    covalent window) + contact_floor_margin, so the census of the packed
    frame is exact by construction (the packing_radius argument, applied
    pairwise) and the frame clears the sanity floor of review W8."""
    L = np.asarray(box, float)
    names = [name for name, n in counts.items() for _ in range(n)]
    for name in counts:
        if name not in TEMPLATES:
            raise ChaordError(
                f"no template for molecule {name!r}; known: "
                f"{', '.join(sorted(TEMPLATES))}")
    centres = _grid_centres(len(names), L, rng)
    blocks = []
    symbols: list[str] = []
    for name in names:
        t = TEMPLATES[name]
        R = random_rotation(rng)
        blocks.append(_RigidBlock(t["symbols"], t["rel"] @ R.T))
        symbols.extend(t["symbols"])
    pos = _relax_rigid_blocks(centres, blocks, L, dialect)
    return Frame(pos=pos, cell=np.diag(L), symbols=symbols,
                 pbc=(True, True, True))


def relax_intermolecular_contacts(frame: Frame, dialect) -> Frame:
    """Open sub-floor intermolecular contacts by rigid-body relaxation (the
    W8 pack-then-relax regeneration step).

    Every bond-graph component of the frame is one rigid body (a packed
    frame's components are exactly its molecules -- the caller's census
    check says so); _relax_rigid_blocks moves the bodies until every
    intermolecular pair clears max(contact floor, covalent window) +
    contact_floor_margin.  Atom order, symbols, cell and pbc are preserved."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    edges = bond_graph(frame, dialect)
    n = len(frame)
    if edges:
        rows = [e[0] for e in edges] + [e[1] for e in edges]
        cols = [e[1] for e in edges] + [e[0] for e in edges]
        _, labels = connected_components(
            coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)),
            directed=False)
    else:
        labels = np.arange(n)
    L = frame.cell_diag
    pos = np.mod(frame.pos, L)
    blocks = []
    centres = np.zeros((labels.max() + 1, 3))
    members: list[np.ndarray] = []
    for g in range(labels.max() + 1):
        idx = np.where(labels == g)[0]
        c = pos[idx].mean(axis=0)
        centres[g] = c
        blocks.append(_RigidBlock([frame.symbols[i] for i in idx],
                                  pos[idx] - c))
        members.append(idx)
    relaxed = _relax_rigid_blocks(centres, blocks, L, dialect)
    out = np.zeros_like(pos)
    at = 0
    for blk, idx in zip(blocks, members):
        out[idx] = relaxed[at:at + len(blk.symbols)]
        at += len(blk.symbols)
    return Frame(pos=np.mod(out, L), cell=frame.cell,
                 symbols=frame.symbols, pbc=frame.pbc)


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
                raise PackingJam(
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


def _component_percolates(pos, L, edges, member_set) -> bool:
    """Does this bonded component wrap the periodic box?

    Unwraps the component through its bonds (breadth-first, minimum-image
    step vectors). A compact molecule closes every cycle with zero winding;
    a component whose bonds span the box -- a crystal slab, a bonded solid,
    a network -- reaches some atom twice at positions differing by a nonzero
    lattice vector. Deterministic, no thresholds."""
    from collections import defaultdict, deque
    adj = defaultdict(list)
    for i, j in edges:
        if i in member_set and j in member_set:
            adj[i].append(j)
            adj[j].append(i)
    start = next(iter(member_set))
    unwrapped = {start: pos[start]}
    queue = deque([start])
    while queue:
        i = queue.popleft()
        for j in adj[i]:
            step = pos[j] - pos[i]
            step -= L * np.round(step / L)
            candidate = unwrapped[i] + step
            seen = unwrapped.get(j)
            if seen is None:
                unwrapped[j] = candidate
                queue.append(j)
                continue
            diff = candidate - seen
            winding = np.round(diff / L)
            if np.any(winding != 0) and np.allclose(
                    diff - winding * L, 0.0,               # dialect-exempt: numerical-guard: winding residue target is exactly zero
                    atol=1e-6 * float(L.max())):  # dialect-exempt: numerical-guard: winding residue of exact float sums
                return True
    return False


def _refuse_extended_components(frame: Frame, edges, labels, dialect) -> None:
    """The pseudo-molecule guard (red team F2): a bonded component larger
    than the dialect's `fluid_max_bonded_component` that PERCOLATES the
    periodic box or holds at least half the frame's atoms is an extended
    phase -- a bonded crystal (the 'Mg32' of hcp_mg), a metal slab (the
    'Cu384' of the pre-M4 interface lift), an amorphous network -- not one
    molecule of a molecular fluid. Naming it invents a formula whose counts
    then agree exactly on every side (A6 three-way, A13 residual), so the
    garbage passes every gate; the census refuses loudly instead.

    A compact oversized molecule is NOT extended evidence: the shipped EC
    solvent component (C3H4O3, 10 atoms) exceeds the 3-atom fluid bound of
    the molecular dialect while being an ordinary molecule, so size alone
    must not refuse it. Dialects without the bound (the concept is not
    theirs to declare) keep the historical census."""
    try:
        limit = int(dialect.threshold("fluid_max_bonded_component"))
    except ChaordError:
        return
    n_atoms = len(frame)
    L = frame.cell_diag
    pos = np.mod(frame.pos, L)
    for g in range(labels.max() + 1):
        idx = np.where(labels == g)[0]
        n = len(idx)
        if n <= limit:
            continue
        # extended-phase evidence: the component holds at least half the
        # frame's atoms (it is the phase itself, not a molecule of it), or
        # its bonds wrap the periodic box (a slab/solid/network). Wrapping
        # is only testable on a fully periodic frame.
        majority = 2 * n >= n_atoms  # dialect-exempt: numerical-guard: majority means the component is the phase, not a molecule of it
        if not majority and (not all(frame.pbc)
                             or not _component_percolates(pos, L, edges,
                                                          set(idx.tolist()))):
            continue
        syms = {frame.symbols[i] for i in idx}
        formula = _formula([frame.symbols[i] for i in idx])
        desc = (f"{n} {sorted(syms)[0]} atoms" if len(syms) == 1
                else f"{n} atoms ({formula})")
        raise ChaordError(
            f"extended bonded component ({desc}): not a molecular fluid; "
            f"lift as crystal/amorphous")


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
        _refuse_extended_components(frame, edges, labels, dialect)
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
