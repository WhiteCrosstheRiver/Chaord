"""Crystal + defect program assembly: the M2 lift path output."""
from __future__ import annotations

import numpy as np

from ..build.defects import nearest_neighbor_distance, warren_cowley_alpha1
from ..io.frames import Frame
from ..lang.ir import (
    GeoChain, KVDefect, Name, PhysicsBlock, Program, ProvenanceBlock, Quantity,
    RegionBlock, ResidualBlock, ShAll, Statement, StrVal, SystemBlock, Tol,
)
from ..lift.crystal import formula_from_slots, round_canonical
from .defects import defect_diff, fit_crystal, group_defects


def _n(text):
    return Name(text=text)


def _kv(text):
    return KVDefect(text=text)


def _neighbor_degrees(pos, cell_diag, cutoff) -> np.ndarray:
    """Degree of every atom in the cutoff neighbour graph (order-invariant:
    the pair SET of a KD-tree query does not depend on the atom ordering,
    and the per-atom degrees are integer counts over that set)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(pos, boxsize=cell_diag)
    pairs = tree.query_pairs(cutoff, output_type="ndarray")
    degrees = np.zeros(len(pos), dtype=np.int64)
    np.add.at(degrees, pairs[:, 0], 1)
    np.add.at(degrees, pairs[:, 1], 1)
    return degrees


def _join_count_null(degrees, n_a: int) -> float:
    """Exact randomisation std of alpha1(a, a) under uniform relabelling.

    Join-count significance (Cliff & Ord, spatial-autocorrelation
    randomisation / nonfree sampling): with the species labels randomly
    permuted over the same sites, the ordered like-neighbour count
    J = sum_ij A_ij * 1[x_i = a] * 1[x_j = a] has the closed form

        E[J]   = 2 M q2
        Var(J) = 4 [ M q2 (1 - q2)
                     + 2 ( H (q3 - q2^2) + D (q4 - q2^2) ) ]

    where M is the number of edges, H the number of edge pairs sharing a
    vertex, D the number of disjoint edge pairs, and q_m the falling-factorial
    ratio N_a(N_a-1)..(N_a-m+1) / N(N-1)..(N-m+1) = P(m given sites all
    carry the species under sampling without replacement).  sd(alpha1) is
    then sqrt(Var(J)) / E[J] (the null alpha is 1 - J/E[J]).  Only integer
    graph invariants and the composition enter, so the band is a symmetric
    function of the labelling: identical for every atom ordering of the same
    frame (rule 3) and free of sampling noise -- no bootstrap draw at all.
    Returns 0.0 for degenerate nulls (too few atoms, sites or like pairs),
    which falls the emission gate back onto the dialect threshold."""
    degrees = np.asarray(degrees, dtype=np.int64)
    n = int(degrees.shape[0])
    m_edges = int(degrees.sum()) // 2
    if n < 4 or n_a < 2 or m_edges < 1:
        return 0
    h_shared = int(sum(d * (d - 1) // 2 for d in degrees.tolist()))
    d_disjoint = m_edges * (m_edges - 1) // 2 - h_shared
    q2 = (n_a * (n_a - 1)) / (n * (n - 1))
    q3 = (n_a * (n_a - 1) * (n_a - 2)) / (n * (n - 1) * (n - 2))
    q4 = (n_a * (n_a - 1) * (n_a - 2) * (n_a - 3)) / (n * (n - 1) * (n - 2) * (n - 3))
    e_j = 2 * m_edges * q2
    if e_j <= 0:
        return 0
    var_j = 4 * (m_edges * q2 * (1 - q2)
                 + 2 * (h_shared * (q3 - q2 * q2)
                        + d_disjoint * (q4 - q2 * q2)))
    return float(np.sqrt(max(var_j, 0)) / e_j)


def _axis_canonical(frame: Frame) -> Frame:
    """Rotate an arbitrarily oriented orthogonal cell onto the coordinate axes.

    The defect pass measures geometry through `cell_diag` boxes (KD trees with
    `boxsize`), which presumes an axis-aligned cell; rule 3 requires identical
    text under rigid rotation, so a rotated input is rotated back first. The
    transform is orthogonal (lengths and angles preserved): every measured
    quantity -- fitted lattice constant, defect tokens, SRO alphas -- is
    unchanged, only the frame orientation. Non-orthogonal cells cannot be
    described by the diagonal-box convention and are rejected honestly."""
    cell = np.asarray(frame.cell, float)
    L = np.linalg.norm(cell, axis=1)
    U = cell / L[:, None]                       # rows: unit cell edges
    if not np.allclose(U @ U.T, np.eye(3), atol=1e-8):  # dialect-exempt: numerical-guard: orthogonality residue
        from ..lang.errors import ChaordError
        raise ChaordError("defect lift needs an orthogonal cell "
                          "(a non-diagonal triclinic box is unsupported)")
    pos = np.mod(frame.pos @ U.T, L)
    pos = np.minimum(pos, L * (1 - 1e-9))       # dialect-exempt: numerical-guard: strict upper edge for KD trees
    return Frame(pos=pos, cell=np.diag(L), symbols=frame.symbols, pbc=frame.pbc)


def lift_crystal_defects(frame, dialect, backend="eam") -> tuple[Program, dict]:
    """Lift a crystal frame with point defects / solid solution into a Program."""
    from ..build.crystal import SLOT_COUNTS
    frame = _axis_canonical(frame)
    name, a, slot_species, score, sites, site_species = fit_crystal(frame, dialect)
    occupancy_mode = slot_species is None
    vacancies, antisites, interstitials = defect_diff(
        frame, sites, site_species, dialect, occupancy=occupancy_mode)
    from .defects import _A_FROM_DNN
    dnn_lattice = a / _A_FROM_DNN[name] if name in _A_FROM_DNN else None
    defect_stmts = group_defects(vacancies, antisites, interstitials, sites,
                                 site_species, frame, dialect,
                                 dnn_lattice=dnn_lattice)

    L = frame.cell_diag
    n_atoms: dict[str, int] = {}
    for s in frame.symbols:
        n_atoms[s] = n_atoms.get(s, 0) + 1
    conserve_values = []
    for s in sorted(n_atoms):
        conserve_values += [_n(s), Quantity(num=str(n_atoms[s]))]

    matched = len(sites) - len(vacancies)
    sites_matched = 100.0 * matched / max(len(sites), 1)  # dialect-exempt: exact-geometry

    region_stmts = []
    if occupancy_mode:
        region_stmts.append(Statement(kind="build", key="lattice", values=[_n(name)]))
        if len(n_atoms) > 1:      # a single species needs no occupancy statement
            from fractions import Fraction
            occ_values = []
            for s in sorted(n_atoms):
                fr = Fraction(n_atoms[s], sum(n_atoms.values())).limit_denominator(1000)
                occ_values += [_n(s), Quantity(num=f"{fr.numerator}/{fr.denominator}")]
            region_stmts.append(Statement(kind="build", key="occupancy", values=occ_values))
        # SRO: measure alpha1 per species pair; emit when it leaves the random band
        d_nn = nearest_neighbor_distance(frame)
        cutoff = float(dialect.threshold("sro_shell1_factor")) * d_nn
        atom_species = sorted(n_atoms)
        # emission band from the EXACT join-count null: the like-neighbour
        # count of a random relabelling of the same composition has a
        # closed-form mean and variance (Cliff-Ord randomisation), so the
        # band depends only on the neighbour-graph invariants and the
        # species counts. Both are symmetric functions of the frame, so the
        # band -- and the emitted text -- is the same for any atom ordering
        # of the same frame (canonical text, rule 3), with no sampling noise
        # and no bootstrap cost.
        degrees = _neighbor_degrees(frame.pos, frame.cell_diag, cutoff)
        noise = {s: _join_count_null(degrees, n_atoms[s]) for s in atom_species}
        # significance gate: a random solution's alpha1 is a finite-sample
        # fluctuation of the alpha = 0 null (rule 1: the program is the
        # macrostate). A constrain line may carry the measured value only when
        # it leaves BOTH the dialect's minimum band and the relabelling null
        # band; otherwise the line bakes one microstate's noise into the text
        # and the round trip is no longer byte-stable across frames/seeds.
        tol_sro = float(dialect.threshold("sro_print_tolerance"))
        for i in range(len(atom_species)):
            alpha = warren_cowley_alpha1(frame, atom_species[i], atom_species[i], cutoff)
            emit_thr = max(float(dialect.threshold("sro_emit_threshold")),
                           float(dialect.threshold("sro_emit_noise_factor"))
                           * noise[atom_species[i]])
            if abs(alpha) > emit_thr:
                region_stmts.append(Statement(
                    kind="constrain", key="sro",
                    values=[_n("alpha1"), _n(f"{atom_species[i]}-{atom_species[i]}"),
                            Quantity(num=f"{alpha:+.2f}"), Tol(value=Quantity(num=f"{tol_sro:.2f}"))]))
    else:
        region_stmts.append(Statement(kind="build", key="prototype", values=[_n(name)]))
        region_stmts.append(Statement(kind="build", key="composition",
                                      values=[_n(formula_from_slots(name, slot_species))]))
    region_stmts.append(Statement(
        kind="build", key="a",
        values=[Quantity(num=round_canonical(a, "canonical_a_decimals", dialect), unit="A")]))
    region_stmts.append(Statement(kind="build", key="orient", values=[
        _n("x"), _n("[100]"), _n("y"), _n("[010]"), _n("z"), _n("[001]")]))
    for _kind, tok, count in defect_stmts:
        region_stmts.append(Statement(
            kind="build", key="defect",
            values=[_kv(tok), _n("count"), Quantity(num=str(count))]))
    region_stmts.append(Statement(
        kind="assert", key="sites_matched",
        values=[Quantity(num=f"{sites_matched:.1f}", unit="%")]))

    system = SystemBlock(statements=[
        Statement(kind="build", key="cell", values=[
            Quantity(num=round_canonical(L[i], "canonical_cell_decimals", dialect))
            for i in range(3)]),
        Statement(kind="build", key="pbc", values=[_n("xyz")]),
        Statement(kind="conserve", key="atoms", values=conserve_values),
    ])
    physics = PhysicsBlock(statements=[
        Statement(kind="build", key="backend", values=[_n(backend)]),
    ])
    region = RegionBlock(phase="crystal", name="bulk",
                         geometry=GeoChain(parts=[ShAll()], ops=[]),
                         statements=region_stmts)
    provenance = ProvenanceBlock(statements=[
        Statement(kind="build", key="dialects", values=[StrVal(text=dialect.version_string)]),
        Statement(kind="build", key="lift_version", values=[StrVal(text="0.1.0")]),
    ])
    diagnostics = dict(n_vacancies=len(vacancies), n_antisites=len(antisites),
                       n_interstitials=len(interstitials), fit_score=score,
                       sites_matched=sites_matched, prototype=name, a=a)
    program = Program(version="0.1", dialects=list(dialect.names),  # dialect-exempt: numerical-guard: language version constant
                      blocks=[system, physics, region, ResidualBlock(none=True), provenance])
    return program, diagnostics
