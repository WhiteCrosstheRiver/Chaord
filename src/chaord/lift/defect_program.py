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


def lift_crystal_defects(frame, dialect, backend="eam") -> tuple[Program, dict]:
    """Lift a crystal frame with point defects / solid solution into a Program."""
    from ..build.crystal import SLOT_COUNTS
    name, a, slot_species, score, sites, site_species = fit_crystal(frame, dialect)
    occupancy_mode = slot_species is None
    vacancies, antisites, interstitials = defect_diff(
        frame, sites, site_species, dialect, occupancy=occupancy_mode)
    defect_stmts = group_defects(vacancies, antisites, interstitials, sites,
                                 site_species, frame, dialect)

    L = frame.cell_diag
    n_atoms: dict[str, int] = {}
    for s in frame.symbols:
        n_atoms[s] = n_atoms.get(s, 0) + 1
    conserve_values = []
    for s in sorted(n_atoms):
        conserve_values += [_n(s), Quantity(num=str(n_atoms[s]))]

    matched = len(sites) - len(vacancies)
    sites_matched = 100.0 * matched / max(len(sites), 1)  # dialect-exempt: percent scaling

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
        # emission threshold from a bootstrap null: alpha1 of the same multiset
        # with labels reshuffled; emit only outside factor x (null std)
        n_boot = int(dialect.threshold("sro_bootstrap_samples"))
        boot_rng = np.random.default_rng(int(dialect.threshold("sro_bootstrap_seed")))
        null = []
        labels = list(frame.symbols)
        for _ in range(n_boot):
            shuffled = labels[:]
            boot_rng.shuffle(shuffled)
            fb = Frame(pos=frame.pos, cell=frame.cell, symbols=shuffled, pbc=frame.pbc)
            null.append(warren_cowley_alpha1(fb, atom_species[0], atom_species[0], cutoff))
        noise = float(np.std(null)) if len(null) > 1 else 0.0  # dialect-exempt: degenerate std
        emit_thr = max(float(dialect.threshold("sro_emit_min")),
                       float(dialect.threshold("sro_emit_noise_factor")) * noise)
        tol_sro = float(dialect.threshold("sro_print_tolerance"))
        for i in range(len(atom_species)):
            alpha = warren_cowley_alpha1(frame, atom_species[i], atom_species[i], cutoff)
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
    program = Program(version="0.1", dialects=list(dialect.names),  # dialect-exempt: language version
                      blocks=[system, physics, region, ResidualBlock(none=True), provenance])
    return program, diagnostics
