"""Amorphous lifter: disordered network -> amorphous region program."""
from __future__ import annotations

import numpy as np

from ..cv.glass import ring_distribution
from ..io.frames import Frame
from ..lang.ir import (
    GeoChain, Name, PhysicsBlock, Program, ProvenanceBlock, Quantity,
    RegionBlock, ResidualBlock, ShAll, Statement, StrVal, SystemBlock, Tol,
)


def network_edges(frame: Frame, dialect):
    """Bonds for the amorphous test: covalent when species are elements,
    geometric (a fraction of the typical NN distance) for placeholder species."""
    from ..build.molecules import bond_graph
    from ase.data import chemical_symbols
    if all(s in chemical_symbols and s != "X" for s in frame.symbols):
        return bond_graph(frame, dialect)
    from scipy.spatial import cKDTree
    from ..build.defects import typical_neighbor_distance
    L = frame.cell_diag
    pos = np.minimum(np.mod(frame.pos, L), L * (1 - 1e-9))  # dialect-exempt: numerical-guard: strict upper edge for KD trees
    rc = float(dialect.threshold("amorphous_bond_factor")) * typical_neighbor_distance(frame)
    return [tuple(e) for e in cKDTree(pos, boxsize=L).query_pairs(rc, output_type="ndarray")]


def is_amorphous(frame: Frame, dialect) -> bool:
    """Single phase, everywhere disordered, but with a bonded network."""
    from ..lift.passes import qbar
    try:
        edges = network_edges(frame, dialect)
    except Exception:
        return False
    n = len(frame)
    # a network: most atoms have 3-6 covalent neighbours
    if not edges:
        return False
    deg = np.zeros(n, int)
    for a, b in edges:
        deg[a] += 1
        deg[b] += 1
    network = float((deg >= 3).mean())  # any extended bonded structure
    try:
        rc = float(dialect.threshold("q6_cutoff"))
        thr = float(dialect.threshold("q6_solid"))
        q6, _, _ = qbar(np.mod(frame.pos, frame.cell_diag), frame.cell_diag, rc=rc)
        disordered = float((q6 > thr).mean()) < float(dialect.threshold('amorphous_solid_frac_max'))
    except Exception:
        disordered = True
    return network > float(dialect.threshold('amorphous_network_min')) and disordered


def lift_amorphous(frame: Frame, dialect, backend="lj") -> Program:
    counts: dict[str, int] = {}
    for s in frame.symbols:
        counts[s] = counts.get(s, 0) + 1
    conserve_values = []
    for s in sorted(counts):
        conserve_values += [Name(text=s), Quantity(num=str(counts[s]))]

    L = frame.cell_diag
    rho = len(frame.pos) / float(np.prod(L))
    region_stmts = []
    if len(counts) == 1:
        region_stmts.append(Statement(kind="build", key="composition",
                                      values=[Name(text=next(iter(counts)))]))
    else:
        vals = []
        for s in sorted(counts):
            vals += [Name(text=s), Quantity(num=str(counts[s]))]
        region_stmts.append(Statement(kind="build", key="composition", values=vals))
    region_stmts.append(Statement(kind="state", key="density",
                                  values=[Quantity(num=f"{rho:.4f}")]))
    # the protocol is the shortest description of how a glass is made: the
    # dialect's default melt-quench, in the backend's reduced units (LJ only).
    # These parameters are the dialect defaults, not anything measured from
    # the frame: the provenance note says the protocol was assumed.
    assumed_history = False
    if backend == "lj":
        from ..lang.ir import Arrow
        t_melt = float(dialect.threshold("glass_melt_T"))
        n_melt = int(dialect.threshold("glass_melt_steps"))
        t_q = float(dialect.threshold("glass_quench_T"))
        n_q = int(dialect.threshold("glass_quench_steps"))
        t_a = float(dialect.threshold("glass_anneal_T"))
        n_a = int(dialect.threshold("glass_anneal_steps"))
        rate = round((t_melt - t_q) / n_q, 6)  # dialect-exempt: numerical-guard: printable cooling rate, T per step
        region_stmts.append(Statement(
            kind="history", key="melt",
            values=[Quantity(num=f"{t_melt:g}"), Name(text="for"),
                    Quantity(num=str(n_melt)), Arrow(),
                    Name(text="quench"), Name(text="to"), Quantity(num=f"{t_q:g}"),
                    Name(text="at"), Quantity(num=f"{rate:g}"), Arrow(),
                    Name(text="anneal"), Quantity(num=f"{t_a:g}"), Name(text="for"),
                    Quantity(num=str(n_a))]))
        assumed_history = True
    # held-out network statistics as asserts; placeholder species (X) use a
    # geometric cutoff (fraction of d_NN), real elements the dialect rule
    from ..cv.local import angle_mean, cn_mean
    from ase.data import chemical_symbols
    from ..build.defects import typical_neighbor_distance
    try:
        if all(s in chemical_symbols and s != "X" for s in frame.symbols):
            cut = float(dialect.threshold("cn_cutoff"))
            q_cut = Quantity(num=f"{cut:.2f}", unit="A")
        else:
            cut = float(dialect.threshold("amorphous_bond_factor")) * typical_neighbor_distance(frame)
            q_cut = Quantity(num=f"{cut:.2f}")
        cn = cn_mean(frame, dialect, cutoff=cut)
        ang = angle_mean(frame, dialect, cutoff=cut)
        region_stmts.append(Statement(
            kind="assert", key="cn",
            values=[Quantity(num=f"{cn:.2f}"), Tol(value=Quantity(num="0.30")),  # dialect-exempt: numerical-guard: canonical printed tolerance
                    Name(text="cutoff"), q_cut]))
        region_stmts.append(Statement(
            kind="assert", key="angle_mean",
            values=[Quantity(num=f"{ang:.1f}"), Tol(value=Quantity(num="3.0")),  # dialect-exempt: numerical-guard: canonical printed tolerance
                    Name(text="deg")]))
    except Exception:
        pass
    rings = ring_distribution(frame, dialect, edges=network_edges(frame, dialect))
    if rings:
        dom = max(rings, key=lambda k: rings[k])
        region_stmts.append(Statement(
            kind="assert", key="ring_mode",
            values=[Quantity(num=str(dom))]))

    system = SystemBlock(statements=[
        Statement(kind="build", key="cell", values=[
            Quantity(num=f"{L[i]:.3f}") for i in range(3)]),
        Statement(kind="build", key="pbc", values=[Name(text="xyz")]),
        Statement(kind="conserve", key="atoms", values=conserve_values),
    ])
    region = RegionBlock(phase="amorphous", name="glass",
                         geometry=GeoChain(parts=[ShAll()], ops=[]),
                         statements=region_stmts)
    return Program(
        version="0.1", dialects=list(dialect.names),  # dialect-exempt: numerical-guard: language version constant
        blocks=[system,
                PhysicsBlock(statements=[
                    Statement(kind="build", key="backend",
                              values=[Name(text=backend)])]),
                region, ResidualBlock(none=True),
                ProvenanceBlock(statements=[
                    Statement(kind="build", key="dialects",
                              values=[StrVal(text=dialect.version_string)]),
                    Statement(kind="build", key="lift_version",
                              values=[StrVal(text="0.1.0")]),
                    *([Statement(kind="build", key="note",
                                  values=[StrVal(
                                      text="assumed default protocol from "
                                           "dialect")])]
                      if assumed_history else []),
                ])])
