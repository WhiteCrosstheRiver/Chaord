"""Fluid lifter: a single-phase liquid or gas frame -> canonical program."""
from __future__ import annotations

import numpy as np

from ..build.molecules import molecular_mass, molecule_census
from ..io.frames import Frame
from ..lang.errors import ChaordError
from ..lang.ir import (
    GeoChain, Name, PhysicsBlock, Program, ProvenanceBlock, Quantity,
    RegionBlock, ResidualBlock, ShAll, Statement, StrVal, SystemBlock, Tol,
)


class AssumedT(float):
    """A temperature the caller did NOT give: the dialect's md_reference_T
    default, wrapped so the lifted program's provenance can say the value was
    assumed rather than measured (external review 3, step 3: an assumed
    dialect constant must not be laundered into a fact by the program text).
    Behaves as a plain float everywhere else."""


def _all_elements(frame: Frame) -> bool:
    from ase.data import chemical_symbols
    # chemical_symbols[0] is the placeholder 'X': not a bondable element
    return all(s in chemical_symbols and s != "X" for s in frame.symbols)


def bonded_single_phase(frame: Frame, edges, dialect) -> bool:
    """A bonded frame is one molecular fluid only if every bond-graph
    component is a small molecule (dialect bound, atoms).

    An extended component -- a metal slab under a molecular-containing
    dialect, an amorphous network -- is a second phase and must lift as an
    interface or network program. The M3 shortcut (any bonds -> fluid) predates
    interface segmentation and absorbed whole crystals as pseudo-molecule
    census blobs (e.g. 'Cu384')."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(frame)
    rows = [e[0] for e in edges] + [e[1] for e in edges]
    cols = [e[1] for e in edges] + [e[0] for e in edges]
    _, labels = connected_components(
        coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)),
        directed=False)
    largest = int(np.bincount(labels).max())
    return largest <= int(dialect.threshold("fluid_max_bonded_component"))


def is_single_phase(frame: Frame, dialect) -> bool:
    """Should this frame lift as one homogeneous fluid?

    Molecular frames use the component-size rule above (every bonded
    component a small molecule).  Atomic frames use the unit-free local-order
    rule (W1, Review 8 -- `amorphous.crystal_like_fraction`, one definition
    per quantity): neighbours within `q_cutoff_factor` x the frame's own
    median d_NN, an atom with fewer than `q_min_neighbours` neighbours never
    crystal-like (a gas atom with one or two neighbours has q6bar near 1 --
    the dilute-argon false positive), and the frame is a fluid while the
    crystal-like fraction stays below the shared disordered threshold
    `amorphous_solid_frac_max` -- the SAME bound is_amorphous uses, so no
    frame is ever too solid for a fluid and disordered enough for a glass at
    once.  A missing threshold raises; there is no absolute q6 cutoff left."""
    if not _all_elements(frame):
        edges = []
    else:
        from ..build.molecules import bond_graph
        edges = bond_graph(frame, dialect)
    if edges:
        return bonded_single_phase(frame, edges, dialect)
    from .amorphous import crystal_like_fraction
    return (crystal_like_fraction(frame, dialect)
            < float(dialect.threshold("amorphous_solid_frac_max")))


def _rdf_stats(frame: Frame, dialect, pos=None):
    from scipy.spatial import cKDTree
    L = frame.cell_diag
    rmax = float(dialect.threshold("gr_rmax_fluid"))
    nbin = int(dialect.threshold("gr_bins_fluid"))
    if pos is None:
        pos = frame.pos
    rho = len(pos) / float(np.prod(L))
    pos = np.mod(pos, L)
    tree = cKDTree(pos, boxsize=L)
    pairs = tree.query_pairs(rmax, output_type="ndarray")
    if len(pairs) == 0:
        return np.array([]), np.array([]), rho
    d = pos[pairs[:, 1]] - pos[pairs[:, 0]]
    d -= L * np.round(d / L)
    r = np.linalg.norm(d, axis=1)
    h, e = np.histogram(r, np.linspace(0, rmax, nbin + 1))
    rm = 0.5 * (e[1:] + e[:-1])  # dialect-exempt: numerical-guard: bin centres
    shell = 4 / 3 * np.pi * (e[1:] ** 3 - e[:-1] ** 3)
    g = 2 * h / (len(pos) * rho * shell)
    return rm, g, rho


def _centers(frame: Frame, dialect):
    """Molecule centres (mean position of each bond component)."""
    from ..build.molecules import bond_graph
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    edges = bond_graph(frame, dialect)
    n = len(frame)
    if not edges:
        return None
    rows = [e[0] for e in edges] + [e[1] for e in edges]
    cols = [e[1] for e in edges] + [e[0] for e in edges]
    _, labels = connected_components(
        coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)), directed=False)
    L = frame.cell_diag
    pos = np.mod(frame.pos, L)
    out = []
    for g in range(labels.max() + 1):
        idx = np.where(labels == g)[0]
        rel = pos[idx] - pos[idx[0]]
        rel -= L * np.round(rel / L)
        out.append(pos[idx[0]] + rel.mean(0))
    return np.array(out)


def _cn_stats_centers(frame: Frame, centers, dialect):
    from scipy.spatial import cKDTree
    L = frame.cell_diag
    rc = float(dialect.threshold("cn_cutoff_fluid"))
    pos = np.mod(centers, L)
    tree = cKDTree(pos, boxsize=L)
    counts = np.array([len(x) - 1 for x in tree.query_ball_point(pos, rc)])
    return float(counts.mean()), float(counts.std()), rc


def _cn_stats(frame: Frame, dialect):
    from scipy.spatial import cKDTree
    L = frame.cell_diag
    rc = float(dialect.threshold("cn_cutoff_fluid"))
    pos = np.mod(frame.pos, L)
    tree = cKDTree(pos, boxsize=L)
    counts = np.array([len(x) - 1 for x in tree.query_ball_point(pos, rc)])
    return float(counts.mean()), float(counts.std()), rc


def lift_fluid(frame: Frame, dialect, T=None, backend=None) -> Program:
    """Lift a homogeneous liquid or gas into a canonical Program."""
    if backend is None:
        backend = "lj" if "lj" in dialect.names else "classical"
    census = molecule_census(frame, dialect) if _all_elements(frame) else {}
    molecular = any(len(n) > 2 or not _is_element(n) for n in census)
    L = frame.cell_diag
    centers = _centers(frame, dialect) if molecular else None
    rm, g, rho_n = _rdf_stats(frame, dialect, pos=centers)
    cn, cn_sd, cn_cut = (_cn_stats_centers(frame, centers, dialect)
                         if centers is not None else _cn_stats(frame, dialect))

    # water model: rigid MD conserves the O-H bond length, so the frame's
    # median O-H names the model (dialect water_models table). The physics
    # block states it (`model spce`) and the water species names the model's
    # template (H2O = the default); unclassifiable geometry states nothing
    # (the default applies) and the measured median goes to provenance.
    water_model = None
    water_oh = None
    if molecular and "H2O" in census:
        from ..build.molecules import (
            classify_water_model, measure_water_oh_median,
        )
        water_oh = measure_water_oh_median(frame, dialect)
        if water_oh is not None:
            try:
                water_model = classify_water_model(dialect, water_oh)
            except ChaordError:
                water_model = None

    # conserve per element for molecular systems, per species for atomic ones
    counts: dict[str, int] = {}
    for s in frame.symbols:
        counts[s] = counts.get(s, 0) + 1
    conserve_values = []
    for s in sorted(counts):
        conserve_values += [Name(text=s), Quantity(num=str(counts[s]))]

    region_stmts = []
    phase = "liquid"
    if molecular:
        from .reactive import display_name
        for formula in sorted(census):
            # census keys are Hill formulas (monatomic ions appear as bare
            # elements): programs name species conventionally (Na -> Na+)
            name = display_name(formula, dialect)
            if formula == "H2O" and water_model is not None:
                from ..build.molecules import water_template_name
                name = water_template_name(water_model, dialect)
            region_stmts.append(Statement(
                kind="build", key="molecules",
                values=[Name(text=name),
                        Quantity(num=str(census[formula]))]))
        from ase.data import atomic_masses, chemical_symbols
        total_mass = sum(atomic_masses[chemical_symbols.index(s)] for s in frame.symbols)
        rho_g = total_mass / float(np.prod(L)) / 0.6022140857  # u/A^3 -> g/cm3  # dialect-exempt: exact-geometry
        region_stmts.append(Statement(
            kind="state", key="density",
            values=[Quantity(num=f"{rho_g:.3f}", unit="g/cm3")]))
        region_stmts.append(Statement(
            kind="assert", key="cn",
            values=[Quantity(num=f"{cn:.1f}"), Tol(value=Quantity(num=f"{cn_sd:.1f}")),
                    Name(text="cutoff"), Quantity(num=f"{cn_cut:.2f}", unit="A")]))
        if len(rm):
            ipk = int(np.argmax(g))
            region_stmts.append(Statement(
                kind="assert", key="gr_peak",
                values=[Quantity(num=f"{rm[ipk]:.2f}", unit="A"), Name(text="height"),
                        Quantity(num=f"{g[ipk]:.2f}")]))
        del cn_cut
    else:
        region_stmts.append(Statement(
            kind="state", key="density", values=[Quantity(num=f"{rho_n:.3f}")]))
        region_stmts.append(Statement(
            kind="assert", key="cn",
            values=[Quantity(num=f"{cn:.1f}"), Tol(value=Quantity(num=f"{cn_sd:.1f}")),
                    Name(text="cutoff"), Quantity(num=f"{cn_cut:.2f}")]))
        if len(rm):
            ipk = int(np.argmax(g))
            region_stmts.append(Statement(
                kind="assert", key="gr_peak",
                values=[Quantity(num=f"{rm[ipk]:.2f}"), Name(text="height"),
                        Quantity(num=f"{g[ipk]:.2f}")]))
        region_stmts.append(Statement(
            kind="assert", key="solid_clusters", values=[Quantity(num="0")]))

    system_stmts = [
        Statement(kind="build", key="cell", values=[
            Quantity(num=f"{L[i]:.3f}") for i in range(3)]),
        Statement(kind="build", key="pbc", values=[Name(text="xyz")]),
    ]
    if T is not None:
        system_stmts.append(Statement(kind="state", key="T", values=[Quantity(num=f"{T:.2f}")]))
    system_stmts.append(Statement(kind="conserve", key="atoms", values=conserve_values))

    physics_stmts = [
        Statement(kind="build", key="backend", values=[Name(text=backend)])]
    if water_model is not None:
        physics_stmts.append(Statement(
            kind="build", key="model", values=[Name(text=water_model)]))

    provenance_stmts = [
        Statement(kind="build", key="dialects",
                  values=[StrVal(text=dialect.version_string)]),
        Statement(kind="build", key="lift_version",
                  values=[StrVal(text="0.1.0")]),
    ]
    if molecular and "H2O" in census and water_oh is not None:
        # O12b (Reviews 4-7): the O-H length alone cannot name the model --
        # TIP3P, TIP4P and TIP4P/2005 all use r(O-H) = 0.9572 A. A table
        # match picks the entry, but the choice is ASSUMED from the geometry
        # family unless the input names it (nothing in the frame does);
        # unclassifiable geometry keeps the honest no-match note
        if water_model is None:
            note = (f"water r_OH median {water_oh:.4f} A matches no "
                    f"water_models entry; the default applies")
        else:
            note = (f"water model {water_model} assumed from r_OH median "
                    f"{water_oh:.4f} A (geometry cannot distinguish models "
                    f"sharing this bond length, e.g. TIP3P/TIP4P/TIP4P-2005)")
        provenance_stmts.append(Statement(
            kind="build", key="note", values=[StrVal(text=note)]))
    if isinstance(T, AssumedT):
        # T metadata honesty (review 3, step 3): the `state T` line states a
        # dialect default, not a measurement of this frame -- provenance says so
        provenance_stmts.append(Statement(
            kind="build", key="note",
            values=[StrVal(text=(f"T assumed (dialect default {float(T):g}); "
                                 "not measured from the frame"))]))

    return Program(
        version="0.1", dialects=list(dialect.names),  # dialect-exempt: numerical-guard: language version constant
        blocks=[
            SystemBlock(statements=system_stmts),
            PhysicsBlock(statements=physics_stmts),
            RegionBlock(phase=phase, name="fluid",
                        geometry=GeoChain(parts=[ShAll()], ops=[]),
                        statements=region_stmts),
            ResidualBlock(none=True),
            ProvenanceBlock(statements=provenance_stmts),
        ])


def _is_element(name: str) -> bool:
    from ase.data import chemical_symbols
    return name in chemical_symbols and name != "X"
