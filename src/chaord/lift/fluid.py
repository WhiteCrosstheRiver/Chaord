"""Fluid lifter: a single-phase liquid or gas frame -> canonical program."""
from __future__ import annotations

import numpy as np

from ..build.molecules import molecular_mass, molecule_census
from ..io.frames import Frame
from ..lang.ir import (
    GeoChain, Name, PhysicsBlock, Program, ProvenanceBlock, Quantity,
    RegionBlock, ResidualBlock, ShAll, Statement, StrVal, SystemBlock, Tol,
)
from .passes import pairs_within, qbar


def _all_elements(frame: Frame) -> bool:
    from ase.data import chemical_symbols
    # chemical_symbols[0] is the placeholder 'X': not a bondable element
    return all(s in chemical_symbols and s != "X" for s in frame.symbols)


def is_single_phase(frame: Frame, dialect) -> bool:
    """Should this frame lift as one homogeneous fluid?

    Atomic frames use the dialect's q6 solid-like rule. Molecular frames pass
    for M3: ordered molecular systems are captured earlier by the crystal and
    defect paths, and molecular interface segmentation arrives in M4."""
    try:
        if not _all_elements(frame):
            edges = []
        else:
            from ..build.molecules import bond_graph
            edges = bond_graph(frame, dialect)
        if edges:
            return True
        rc = float(dialect.threshold("q6_cutoff"))
        thr = float(dialect.threshold("q6_solid"))
        q6, _, _ = qbar(np.mod(frame.pos, frame.cell_diag), frame.cell_diag, rc=rc)
        return float((q6 > thr).mean()) < float(dialect.threshold("fluid_solid_fraction_max"))
    except Exception:
        return False


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
    rm = 0.5 * (e[1:] + e[:-1])  # dialect-exempt: bin centres
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
        for formula in sorted(census):
            region_stmts.append(Statement(
                kind="build", key="molecules",
                values=[Name(text=formula), Quantity(num=str(census[formula]))]))
        from ase.data import atomic_masses, chemical_symbols
        total_mass = sum(atomic_masses[chemical_symbols.index(s)] for s in frame.symbols)
        rho_g = total_mass / float(np.prod(L)) / 0.6022140857  # u/A^3 -> g/cm3  # dialect-exempt: unit conversion
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

    return Program(
        version="0.1", dialects=list(dialect.names),  # dialect-exempt: language version
        blocks=[
            SystemBlock(statements=system_stmts),
            PhysicsBlock(statements=[
                Statement(kind="build", key="backend", values=[Name(text=backend)])]),
            RegionBlock(phase=phase, name="fluid",
                        geometry=GeoChain(parts=[ShAll()], ops=[]),
                        statements=region_stmts),
            ResidualBlock(none=True),
            ProvenanceBlock(statements=[
                Statement(kind="build", key="dialects",
                          values=[StrVal(text=dialect.version_string)]),
                Statement(kind="build", key="lift_version",
                          values=[StrVal(text="0.1.0")]),
            ]),
        ])


def _is_element(name: str) -> bool:
    from ase.data import chemical_symbols
    return name in chemical_symbols and name != "X"
