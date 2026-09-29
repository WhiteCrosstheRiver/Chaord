"""Segment-first pipeline lift (lift/build v2, stage 1 -- behind mode="pipeline").

Seven stages (docs/design/lift_build_v2.md 2.2):

    S1 classify   one qbar pass, cached on the context
    S2 segment    regions.RegionPlan list from 3-D labels + vacuum gaps
    S3 fit        per-region model choice (2.4 table) + fit, legacy kernels
                  on subset frames ("reuse verbatim", 2.3)
    S4 defects    Wigner-Seitz inside crystal-region fits (defects.defect_diff)
    S5 statistics region-level asserts (fluid/glass CVs via the legacy lifters)
    S6 interfaces region-pair interface blocks
    S7 residual   atoms no model claims go to the residual block

The numerics are imported from the legacy lifters (risk R6: only the
orchestration is new). A frame whose atoms are one phase by the dialect's own
rules (S2.5 fast path = design rule 2.4/1 evaluated at frame level, with the
same predicates the legacy cascade uses) lifts through the same engine as the
legacy arm on the same full frame, so the text is byte-identical; genuinely
mixed frames compose regions the cascade could not express.

Failure policy (2.5): a region whose fit fails degrades -- its atoms land in
the residual block and a note is recorded in provenance; the lift never
aborts on a data problem. S1/S2 infrastructure errors (missing dialect keys,
I/O) still raise: those are caller bugs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..io.frames import Frame
from ..lang.ir import (
    InterfaceBlock, Name, PhysicsBlock, Program, ProvenanceBlock, Quantity,
    RegionBlock, ResidualBlock, Statement, StrVal, SystemBlock,
)
from .crystal import lift_crystal, round_canonical
from .defect_program import lift_crystal_defects
from .defects import _A_FROM_DNN, defect_diff, fit_crystal, group_defects
from .fluid import lift_fluid
from .regions import (
    RegionPlan, all_geometry, region_contacts, segment_regions, subset_frame,
    wrapped_interval,
)
from .segment import phase_labels_3d, slab_interfaces


@dataclass
class _LiftCtx:
    """Shared state of one pipeline run (S1 caches computed once)."""
    frame: Frame
    dialect: object
    T: float | None
    notes: list = field(default_factory=list)
    q6: np.ndarray | None = None
    cn: np.ndarray | None = None
    pairs: np.ndarray | None = None
    labels: np.ndarray | None = None

    def classify(self):
        """S1: one qbar pass; every later stage reuses these arrays."""
        if self.q6 is None:
            from .passes import qbar
            rc = float(self.dialect.threshold("q6_cutoff"))
            L = self.frame.cell_diag
            self.q6, self.cn, self.pairs = qbar(np.mod(self.frame.pos, L), L, rc=rc)
        return self.q6

    def default_T(self):
        """T is metadata: the dialect's md_reference_T when the caller gives none."""
        if self.T is not None:
            return self.T
        try:
            return float(self.dialect.threshold("md_reference_T"))
        except Exception:
            return None


# ------------------------------------------------------------- S2.5 fast path --

def _has_molecular_bonds(ctx: _LiftCtx) -> bool:
    """The bond graph is non-empty (mirrors fluid.is_single_phase's first arm)."""
    from .fluid import _all_elements
    try:
        if not _all_elements(ctx.frame):
            return False
        from ..build.molecules import bond_graph
        return bool(bond_graph(ctx.frame, ctx.dialect))
    except Exception:
        return False


def _frame_is_single(ctx: _LiftCtx) -> bool:
    """Design 2.4 rule 1 at frame level: a frame the dialect's own rules call
    single-phase (is_single_phase's molecular arm; the all-solid and
    all-disordered q6 gates) never needs the 3-D machinery -- it lifts through
    the same engine chain the legacy cascade uses, on the same full frame, so
    the text is byte-identical and the cost stays at parity (risk R5)."""
    from .fluid import is_single_phase
    if _has_molecular_bonds(ctx):
        # one implementation of the molecular rule (component size, not the
        # pre-M4 any-bonds shortcut) shared with the legacy cascade
        return is_single_phase(ctx.frame, ctx.dialect)
    q6 = ctx.classify()
    thr = float(ctx.dialect.threshold("q6_solid"))
    frac = float((q6 > thr).mean())
    fm = float(ctx.dialect.threshold("fluid_solid_fraction_max"))
    return frac > 1 - fm or frac < fm


# ----------------------------------------------------------- single-phase S3 --

def _raw_residual_program(ctx: _LiftCtx, why: str):
    """Last-resort degradation: no engine fits, every atom goes to residual."""
    ctx.notes.append(why)
    regions = [RegionPlan(name="bulk", phase="crystal", geometry=all_geometry(),
                          atom_indices=np.arange(len(ctx.frame)), model="residual")]
    return _compose_program(ctx, regions, set())


def _engine_single(ctx: _LiftCtx):
    """Case 1: one region covers the frame; run the engines in the legacy
    cascade's arm order (2.3: reuse verbatim on the -- full -- subset frame):
    crystal A (spglib) -> crystal B (Wigner-Seitz defects) -> amorphous
    (dialect-gated) -> surface (vacuum-gated) -> fluid. Arms that raise
    degrade to the next arm instead of aborting (2.5); the last resort is the
    raw residual. Matching the arm order keeps single-phase frames
    byte-identical to the cascade, including thermally noisy crystals whose
    raw q6 fraction alone cannot pick the family."""
    frame, dialect = ctx.frame, ctx.dialect
    try:
        return lift_crystal(frame, dialect), 0
    except Exception as e:
        ctx.notes.append(f"crystal engine A failed ({type(e).__name__})")
    try:
        program, _diag = lift_crystal_defects(frame, dialect)
        return program, 0
    except Exception as e:
        ctx.notes.append(f"crystal engine B failed ({type(e).__name__})")
    if "glass" in dialect.names:
        from .amorphous import is_amorphous, lift_amorphous
        try:
            if is_amorphous(frame, dialect):
                return lift_amorphous(frame, dialect), 0
        except Exception as e:
            ctx.notes.append(f"amorphous engine failed ({type(e).__name__})")
    try:
        from .surface import has_vacuum, lift_surface
        if has_vacuum(frame, dialect):
            return lift_surface(frame, dialect), 0
    except Exception as e:
        ctx.notes.append(f"surface engine failed ({type(e).__name__})")
    try:
        return lift_fluid(frame, dialect, T=ctx.default_T()), 0
    except Exception as e:
        ctx.notes.append(f"fluid engine failed ({type(e).__name__})")
    return _raw_residual_program(
        ctx, "no engine fits the frame; all atoms residual")


# ------------------------------------------------------------- mixed frames --

def _select_model(ctx: _LiftCtx, region: RegionPlan) -> str:
    """Design 2.4 selection table, stage-1 reading (existing keys only).

    The 3-D label is the phase call; the table's diagnostics guard it: a
    solid-labelled region is crystal only when compact (rule 2: one connected
    piece -- a scattered archipelago of solid labels is q6 noise, re-called
    fluid by rules 4/5); a disordered region is amorphous when the
    dialect-gated network test passes on the subset (rule 3, same gate as the
    legacy cascade's amorphous arm), fluid otherwise."""
    if len(region) == 0:
        return "vacuum"
    if region.phase == "crystal" and region.n_components == 1:
        return "crystal"
    if "glass" in ctx.dialect.names:
        from .amorphous import is_amorphous
        try:
            if is_amorphous(subset_frame(ctx.frame, region.atom_indices),
                            ctx.dialect):
                return "amorphous"
        except Exception:
            pass
    return "fluid"


def _merge_fluid_regions(ctx: _LiftCtx, regions: list) -> list:
    """Design risk R2, same-params merge: regions that both modelled as fluid
    are one macrostate (a molecular liquid and its q6-noise islands, a gas
    with a noisy order parameter). Merging them reproduces what the legacy
    cascade lifted as one fluid, and keeps region counts bounded."""
    fluids = [r for r in regions if r.model == "fluid"]
    if len(fluids) <= 1:
        return regions
    keep, drop = fluids[0], fluids[1:]
    keep.atom_indices = np.sort(np.concatenate(
        [r.atom_indices for r in fluids]))
    keep.n_components = sum(r.n_components for r in fluids)
    regions = [r for r in regions if r not in drop]
    from .regions import _assign_names_and_geometry
    _assign_names_and_geometry(regions, ctx.frame, ctx.dialect)
    return regions


def _region_min_atoms(dialect, n_atoms: int) -> int:
    """Regions below this floor are not fitted: their atoms degrade to the
    residual (design 2.4). Key proposed for dialect approval (stage-1 PR);
    the fallback is the PLAN robustness-layer value. The floor never exceeds a
    quarter of the frame, so a small frame's genuine majority region is not
    degraded beside its noise."""
    try:
        floor = int(float(dialect.threshold("region_min_atoms")))
    except Exception:
        floor = 100  # dialect-exempt: pending dialect key region_min_atoms (PLAN robustness layer)
    return min(floor, max(n_atoms // 4, 1))


def _is_two_region_slab(ctx: _LiftCtx, regions, contacts) -> bool:
    """The composition the legacy M0 slab lifter owns: exactly one crystal
    region and one fluid region, in contact. Those frames compose through
    decompile/program_from_result so the slab golden text stays byte-identical
    (design 2.3 engine C, 2.2 S6); a decompile that cannot read the frame
    falls back to the single-phase fluid exactly like the legacy slab arm, so
    the pipeline reproduces the cascade's outcome either way. Regions that do
    NOT touch (a vacuum-separated pair) are not a slab: they compose."""
    if len(regions) != 2:
        return False
    if {r.model for r in regions} != {"crystal", "fluid"}:
        return False
    return frozenset((0, 1)) in contacts


def _engine_slab(ctx: _LiftCtx):
    """Case 2 (crystal | liquid z-slabs): engine C wraps decompile (2.3)."""
    from .slab import decompile, program_from_result
    frame, dialect = ctx.frame, ctx.dialect
    T = ctx.default_T()
    try:
        res = decompile(frame.pos, frame.cell_diag, T, dialect,
                        symbols=frame.symbols)
        program = program_from_result(res, dialect)
        explained = sum(c["displaced"] for c in res["defects"])
        return program, max(len(res["off"]) - explained, 0)
    except Exception as e:
        # legacy cascade semantics for a frame decompile cannot read: a
        # single-phase fluid after all (lift/legacy.py slab arm). The
        # pipeline keeps that fallback in stage 1 so the equivalence suite
        # covers it; stage 3 replaces it with honest region composition.
        ctx.notes.append(f"slab engine failed ({type(e).__name__}); "
                         "single-phase fluid fallback")
        return lift_fluid(frame, dialect, T=T), 0


# ----------------------------------------------- S3/S4/S5 region fits (case 3) --

def _region_volume(ctx: _LiftCtx, region: RegionPlan) -> float:
    """Volume of a stage-1 region geometry (z-slab of the full x,y footprint)."""
    L = ctx.frame.cell_diag
    if region.geometry.parts[0].t == "all":
        return float(np.prod(L))
    lo, hi = wrapped_interval(ctx.dialect, ctx.frame.pos[region.atom_indices, 2], L[2])
    thickness = hi - lo if hi >= lo else hi - lo + L[2]
    return float(L[0] * L[1] * thickness)


def _clip_to_region(ctx: _LiftCtx, sub: Frame, region: RegionPlan) -> Frame:
    """Engine B's site-membership filter (design 2.2): tile lattice sites only
    over the region's bounding slab (plus site_zone_pad), not the whole box --
    empty space outside the region must not count as phantom vacancies.

    The clip shifts z so the region starts at 0 (a wrapping slab becomes a
    plain interval); a rigid shift changes no measured quantity."""
    pad = float(ctx.dialect.threshold("site_zone_pad"))
    L = sub.cell_diag
    lo, hi = wrapped_interval(ctx.dialect, sub.pos[:, 2], L[2])
    thickness = hi - lo if hi >= lo else hi - lo + L[2]
    pos = np.mod(sub.pos - np.array([0, 0, lo]), L)  # dialect-exempt: numerical-guard: rigid re-anchor of the clip
    z1 = min(thickness + pad, L[2])
    from .defects import _wrap_strict
    return Frame(pos=_wrap_strict(pos, np.array([L[0], L[1], z1])),
                 cell=np.diag([L[0], L[1], z1]),
                 symbols=sub.symbols, pbc=sub.pbc)


def _fit_crystal_region(ctx: _LiftCtx, region: RegionPlan) -> list:
    """Engine A (exact spglib lift on the subset) then engine B (Wigner-Seitz
    defect fit, S4 included) -- the same kernels the legacy defect path uses,
    restricted to the region (statements in defect_program's order)."""
    frame, dialect = ctx.frame, ctx.dialect
    sub = subset_frame(frame, region.atom_indices)
    try:
        program = lift_crystal(sub, dialect)      # engine A: clean region bulk
        block = next(b for b in program.blocks if b.t == "region")
        return list(block.statements)
    except Exception:
        pass
    clipped = _clip_to_region(ctx, sub, region)   # engine B: defect-tolerant
    name, a, slot_species, _score, sites, site_species = fit_crystal(clipped, dialect)
    occupancy_mode = slot_species is None
    vacancies, antisites, interstitials = defect_diff(
        clipped, sites, site_species, dialect, occupancy=occupancy_mode)
    dnn_lattice = a / _A_FROM_DNN[name] if name in _A_FROM_DNN else None
    defect_stmts = group_defects(vacancies, antisites, interstitials, sites,
                                 site_species, clipped, dialect,
                                 dnn_lattice=dnn_lattice)
    statements = []
    if occupancy_mode:
        statements.append(Statement(kind="build", key="lattice",
                                     values=[Name(text=name)]))
        counts: dict[str, int] = {}
        for s in clipped.symbols:
            counts[s] = counts.get(s, 0) + 1
        if len(counts) > 1:
            from fractions import Fraction
            occ = []
            for s in sorted(counts):
                fr = Fraction(counts[s], sum(counts.values())).limit_denominator(1000)
                occ += [Name(text=s), Quantity(num=f"{fr.numerator}/{fr.denominator}")]
            statements.append(Statement(kind="build", key="occupancy", values=occ))
    else:
        from .crystal import formula_from_slots
        statements.append(Statement(kind="build", key="prototype",
                                    values=[Name(text=name)]))
        statements.append(Statement(kind="build", key="composition",
                                    values=[Name(text=formula_from_slots(name, slot_species))]))
    statements.append(Statement(
        kind="build", key="a",
        values=[Quantity(num=round_canonical(a, "canonical_a_decimals", dialect),
                         unit="A")]))
    statements.append(Statement(kind="build", key="orient", values=[
        Name(text="x"), Name(text="[100]"), Name(text="y"), Name(text="[010]"),
        Name(text="z"), Name(text="[001]")]))
    from .defect_program import _kv
    for _kind, tok, count in defect_stmts:
        statements.append(Statement(kind="build", key="defect",
                                    values=[_kv(tok), Name(text="count"),
                                            Quantity(num=str(count))]))
    matched = len(sites) - len(vacancies)
    sites_matched = 100 * matched / max(len(sites), 1)  # dialect-exempt: exact-geometry
    statements.append(Statement(
        kind="assert", key="sites_matched",
        values=[Quantity(num=f"{sites_matched:.1f}", unit="%")]))
    return statements


def _rdf_region(pos: np.ndarray, L: np.ndarray, rho: float, dialect):
    """g(r) of a region: the binning rule of fluid._rdf_stats with the REGION
    volume as the number-density normalization (a sub-region is not the box)."""
    from scipy.spatial import cKDTree
    rmax = float(dialect.threshold("gr_rmax_fluid"))
    nbin = int(dialect.threshold("gr_bins_fluid"))
    pos = np.mod(pos, L)
    pairs = cKDTree(pos, boxsize=L).query_pairs(rmax, output_type="ndarray")
    if len(pairs) == 0 or rho <= 0:
        return np.array([]), np.array([])
    d = pos[pairs[:, 1]] - pos[pairs[:, 0]]
    d -= L * np.round(d / L)
    r = np.linalg.norm(d, axis=1)
    h, e = np.histogram(r, np.linspace(0, rmax, nbin + 1))
    rm = 0.5 * (e[1:] + e[:-1])  # dialect-exempt: numerical-guard: bin centres
    shell = 4 / 3 * np.pi * (e[1:] ** 3 - e[:-1] ** 3)
    g = 2 * h / (len(pos) * rho * shell)
    return rm, g


def _fit_fluid_region(ctx: _LiftCtx, region: RegionPlan) -> list:
    """S5 for a fluid region: the legacy fluid lifter's statement set, with
    densities referred to the region volume (design 2.2: subset frames keep
    the cell, so the box volume would be wrong by the complement's share)."""
    from .fluid import _all_elements, _centers, _cn_stats, _cn_stats_centers
    frame, dialect = ctx.frame, ctx.dialect
    sub = subset_frame(frame, region.atom_indices)
    L = frame.cell_diag
    volume = _region_volume(ctx, region)
    statements = []
    molecular = False
    if _all_elements(sub):
        from ..build.molecules import bond_graph, molecule_census
        if bond_graph(sub, dialect):
            molecular = True
            from .reactive import display_name
            census = molecule_census(sub, dialect)
            for formula in sorted(census):
                statements.append(Statement(
                    kind="build", key="molecules",
                    values=[Name(text=display_name(formula, dialect)),
                            Quantity(num=str(census[formula]))]))
    if molecular:
        centers = _centers(sub, dialect)
        cn, cn_sd, cn_cut = _cn_stats_centers(sub, centers, dialect)
        from ase.data import atomic_masses, chemical_symbols
        total_mass = sum(atomic_masses[chemical_symbols.index(s)] for s in sub.symbols)
        rho_g = total_mass / volume / 0.6022140857  # dialect-exempt: exact-geometry: u/A^3 -> g/cm3 (fluid.py constant)
        statements.append(Statement(kind="state", key="density",
                                    values=[Quantity(num=f"{rho_g:.3f}", unit="g/cm3")]))
        rm, g = _rdf_region(centers, L, len(centers) / volume, dialect)
    else:
        cn, cn_sd, cn_cut = _cn_stats(sub, dialect)
        rho = len(sub) / volume
        statements.append(Statement(kind="state", key="density",
                                    values=[Quantity(num=f"{rho:.3f}")]))
        rm, g = _rdf_region(sub.pos, L, rho, dialect)
    from ..lang.ir import Tol
    statements.append(Statement(
        kind="assert", key="cn",
        values=[Quantity(num=f"{cn:.1f}"), Tol(value=Quantity(num=f"{cn_sd:.1f}")),
                Name(text="cutoff"), Quantity(num=f"{cn_cut:.2f}")]))
    if len(rm):
        ipk = int(np.argmax(g))
        statements.append(Statement(
            kind="assert", key="gr_peak",
            values=[Quantity(num=f"{rm[ipk]:.2f}"), Name(text="height"),
                    Quantity(num=f"{g[ipk]:.2f}")]))
    if not molecular:
        statements.append(Statement(kind="assert", key="solid_clusters",
                                    values=[Quantity(num="0")]))
    return statements


def _fit_amorphous_region(ctx: _LiftCtx, region: RegionPlan) -> list:
    """S5 for an amorphous region: the legacy amorphous lifter on the subset,
    with the density line recomputed for the region volume."""
    from .amorphous import lift_amorphous
    sub = subset_frame(ctx.frame, region.atom_indices)
    program = lift_amorphous(sub, ctx.dialect)
    block = next(b for b in program.blocks if b.t == "region")
    statements = list(block.statements)
    rho = len(sub) / _region_volume(ctx, region)
    for k, s in enumerate(statements):
        if s.kind == "state" and s.key == "density":
            statements[k] = Statement(
                kind="state", key="density",
                values=[Quantity(num=f"{rho:.4f}")])
            break
    return statements


def _fit_region(ctx: _LiftCtx, region: RegionPlan) -> list:
    if region.model == "crystal":
        return _fit_crystal_region(ctx, region)
    if region.model == "amorphous":
        return _fit_amorphous_region(ctx, region)
    return _fit_fluid_region(ctx, region)


# ------------------------------------------------------ S6/S7 compose (case 3) --

def _interface_blocks(ctx: _LiftCtx, regions, contacts) -> list:
    """S6: one interface block per contacting region pair, at/width read from
    the 10-90 crossings of the pair's own solid-fraction profile (design 2.2
    S6 for general boundaries; the two-z-slab case never reaches here -- it
    goes through the legacy tanh fit for byte-stable goldens)."""
    by_pos = {k: r for k, r in enumerate(regions)}
    blocks = []
    for contact in sorted(contacts, key=lambda c: tuple(sorted(
            by_pos[i].name for i in c))):
        a_idx, b_idx = sorted(contact, key=lambda i: by_pos[i].name)
        ra, rb = by_pos[a_idx], by_pos[b_idx]
        if ra.model in ("residual", "vacuum") or rb.model in ("residual", "vacuum"):
            continue
        mask = np.zeros(len(ctx.frame), bool)
        mask[ra.atom_indices] = True
        found = slab_interfaces(mask, ctx.frame, ctx.dialect)
        if not found:
            continue            # not a z-separated pair: no at/width to state
        at = found[0]
        blocks.append(InterfaceBlock(
            a=ra.name, b=rb.name, statements=[
                Statement(kind="build", key="at",
                          values=[Name(text="z"), Quantity(num=f"{at['at']:.1f}")]),
                Statement(kind="build", key="width",
                          values=[Quantity(num=f"{at['width']:.1f}")]),
            ]))
    return blocks


def _residual_block(ctx: _LiftCtx, idx: list):
    """S7: atoms claimed by no model, as raw coordinates (never drop an atom).
    Sorted by (species, position), not by atom index: re-ordering a frame
    leaves the text unchanged (rule 3)."""
    if not idx:
        return ResidualBlock(none=True)
    frame = ctx.frame
    pos = np.mod(frame.pos[idx], frame.cell_diag)
    syms = np.array(frame.symbols)[idx]
    order = np.lexsort((pos[:, 2], pos[:, 1], pos[:, 0], syms))
    statements = []
    for k in order:
        x, y, z = pos[k]
        statements.append(Statement(
            kind="build", key="atom",
            values=[Name(text=str(syms[k])), Quantity(num=f"{x:.2f}"),
                    Quantity(num=f"{y:.2f}"), Quantity(num=f"{z:.2f}")]))
    return ResidualBlock(none=False, statements=statements)


def _compose_program(ctx: _LiftCtx, regions, contacts):
    """S7 assembly for a composed (multi-region) frame: exactly one Program,
    regions in canonical order, interfaces between them, residual last.
    Returns (program, unexplained atom count)."""
    frame, dialect = ctx.frame, ctx.dialect
    L = frame.cell_diag
    counts: dict[str, int] = {}
    for s in frame.symbols:
        counts[s] = counts.get(s, 0) + 1
    conserve = []
    for s in sorted(counts):
        conserve += [Name(text=s), Quantity(num=str(counts[s]))]
    system_statements = [
        Statement(kind="build", key="cell", values=[
            Quantity(num=round_canonical(L[i], "canonical_cell_decimals", dialect))
            for i in range(3)]),
        Statement(kind="build", key="pbc", values=[Name(text="xyz")]),
    ]
    T = ctx.default_T()
    if T is not None:
        system_statements.append(
            Statement(kind="state", key="T", values=[Quantity(num=f"{T:.2f}")]))
    system_statements.append(
        Statement(kind="conserve", key="atoms", values=conserve))

    min_atoms = _region_min_atoms(dialect, len(frame))
    blocks = [SystemBlock(statements=system_statements)]
    any_crystal = any(r.model == "crystal" for r in regions)
    backend = ("lj" if "lj" in dialect.names
               else "eam" if any_crystal else "classical")
    blocks.append(PhysicsBlock(statements=[
        Statement(kind="build", key="backend", values=[Name(text=backend)])]))

    residual_idx: list = []
    for r in sorted(regions, key=lambda r: r.name):
        if r.model == "vacuum":
            blocks.append(RegionBlock(phase="vacuum", name=r.name,
                                      geometry=r.geometry, statements=[]))
            continue
        if r.model == "residual":
            residual_idx.extend(int(i) for i in r.atom_indices)
            continue
        if len(r) < min_atoms:
            ctx.notes.append(f"region {r.name}: {len(r)} atoms below the "
                             f"region floor ({min_atoms}), residual")
            r.model = "residual"
            residual_idx.extend(int(i) for i in r.atom_indices)
            continue
        try:
            r.statements = _fit_region(ctx, r)
        except Exception as e:
            ctx.notes.append(f"region {r.name}: {r.model} fit failed "
                             f"({type(e).__name__}), residual")
            r.model = "residual"
            residual_idx.extend(int(i) for i in r.atom_indices)
            continue
        blocks.append(RegionBlock(phase=r.phase, name=r.name,
                                  geometry=r.geometry,
                                  statements=r.statements))
    blocks.extend(_interface_blocks(ctx, regions, contacts))
    blocks.append(_residual_block(ctx, residual_idx))

    provenance = [
        Statement(kind="build", key="dialects",
                  values=[StrVal(text=dialect.version_string)]),
        Statement(kind="build", key="lift_version", values=[StrVal(text="0.1.0")]),
    ]
    for note in ctx.notes:
        provenance.append(Statement(kind="build", key="note",
                                    values=[StrVal(text=note)]))
    blocks.append(ProvenanceBlock(statements=provenance))
    return Program(version="0.1", dialects=list(dialect.names),  # dialect-exempt: numerical-guard: language version constant
                   blocks=blocks), len(residual_idx)


# ------------------------------------------------------------------ driver ----

def _run_mixed(ctx: _LiftCtx):
    """S2-S7 for a genuinely mixed frame."""
    if ctx.labels is None:
        ctx.labels = phase_labels_3d(ctx.frame, ctx.dialect)
    regions = segment_regions(ctx.frame, ctx.dialect, labels=ctx.labels)
    for r in regions:
        r.model = _select_model(ctx, r)
    regions = _merge_fluid_regions(ctx, regions)
    if len(regions) == 1 and len(regions[0]) == len(ctx.frame):
        # the mixed gate fired on raw-q6 noise, but the 3-D segmentation and
        # the model table call the frame one phase after all (a dilute gas
        # with a noisy order parameter, a crystal under raw-q6 strain): it
        # lifts through the same engine chain as the fast path
        return _engine_single(ctx)
    contacts = region_contacts(ctx.frame, ctx.dialect, regions)
    if _is_two_region_slab(ctx, regions, contacts):
        return _engine_slab(ctx)
    return _compose_program(ctx, regions, contacts)


def pipeline_lift_detailed(frame, dialect, T=None):
    """(program, unexplained) -- the pair lift_frame records in its diagnostic."""
    ctx = _LiftCtx(frame=frame, dialect=dialect, T=T)
    if _frame_is_single(ctx):            # S1 + the frame-level single-phase gate
        return _engine_single(ctx)
    return _run_mixed(ctx)


def pipeline_lift(frame, dialect, T=None) -> Program:
    """Lift a Frame into a Program through the segment-first pipeline.

    T is metadata (stated in the program, never measured); when omitted the
    dialect's md_reference_T supplies it, exactly as the legacy cascade does."""
    return pipeline_lift_detailed(frame, dialect, T)[0]
