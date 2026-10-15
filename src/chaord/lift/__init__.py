"""Decompiler (chaord lift): coordinates -> program.

lift_frame routes: a fully ordered single-crystal frame lifts through the exact
crystal path; a crystal with point defects lifts through the Wigner-Seitz
defect path; anything with regions of different phase falls back to the M0
solid-liquid slab lifter.

Every program returned by lift_frame carries a lift diagnostic (see
chaord.check.statics.record_lift_diag): the number of atoms the lift could not
explain, plus the failing items of the static checks that run automatically
after every lift. Checks never raise here - they measure; the CLI prints them
again and sets its exit code from them.
"""
from chaord.lift.crystal import lift_crystal  # noqa: F401
from chaord.lift.nanotube import (  # noqa: F401
    NanotubeChirality, TubeComponent, infer_nanotube_chirality,
    nanotube_components,
)
from chaord.lift.passes import otsu, pairs_within, qbar  # noqa: F401
from chaord.lift.slab import decompile, program_from_result  # noqa: F401
from chaord.lift.defect_program import lift_crystal_defects  # noqa: F401
from chaord.lift.fluid import AssumedT, lift_fluid  # noqa: F401

import numpy as np


def lift_frame(frame, dialect, T=None, mode="auto", field=None):
    """Lift a Frame into a Program; mode is auto | pipeline | crystal |
    defects | amorphous | surface | fluid | slab.

    mode="auto" composes a tube-plus-liquid frame first (D35,
    docs/reviews/open_items_v4.md Y3): every all-carbon three-coordinated
    component that passes the nanotube inference becomes a crystal tube
    region in its cylinder, the atoms inside the cylinder lift as the
    confined fluid region with the molecular stack, and any atom outside
    the cylinders becomes its own region. When no component passes the
    inference (the verdict of `nanotube_components`, including dialects
    without the carbon vocabulary) the legacy cascade answers exactly as
    before. Every other mode is the legacy cascade alone.

    mode="pipeline" runs the segment-first pipeline (lift/build v2 stage 1);
    every other mode is the legacy try-cascade (lift/legacy.py). mode="auto"
    stays on the legacy cascade until the pipeline equivalence suite is green.

    T is metadata (stated in the program, never measured); when omitted it is
    read from the dialect's md_reference_T and wrapped as an AssumedT so the
    program's provenance block records "T assumed (dialect default ...)" --
    the stated value is a dialect constant, not a fact about this frame
    (external review 3, step 3).

    field is external-field metadata (RD3, docs/reviews/research_decisions.md):
    no frame can measure a field, so the value is ALWAYS caller-supplied as
    field = {"component": "E", "value": 0.10, "unit": "V/A", "axis": "z"}
    (component defaults to E, unit to V/A; value and axis are required).
    Given one, every lift path prints the `state field` statement in the
    system block next to `state T`, with a provenance note recording that
    the value was assumed, not measured -- the W11 assumed-value rule exactly
    as fluid.AssumedT applies it. Given none, nothing field-related is
    printed (there is no honest default field).

    D29 (docs/reviews/open_items_v4.md) made the orientation statement
    BLIND: `assert dipole_p1 <v> +- <tol> along <axis>` is measured for
    every frame whose dialect names dipolar species (`field_dipole_species`),
    with or without `field=`; the field spec adds ONLY the `state field`
    line and its assumed-value note. The analysis axis is the field's axis
    when one is given, else the nanotube region's axis when the program
    holds one, else the dialect's `dipole_axis_default` (research.yaml).

    Rotation invariance (Y5's 20-transform done-when): every frame whose
    cell is an orthogonal box in a ROTATED orientation is canonicalised to
    the axis-aligned box first (see _canonical_cell_frame) -- a rigid
    rotation of a frame is the same microstate, and the whole cascade
    measures minimum-image invariants that the canonicalisation preserves
    exactly (the fractional coordinates are unchanged)."""
    frame = _canonical_cell_frame(frame)
    field_parts = _field_parts(field) if field is not None else None
    if T is None:
        try:
            T = AssumedT(float(dialect.threshold("md_reference_T")))
        except Exception:
            T = None  # dialects without an MD default state no temperature
    program = unexplained = None
    if mode == "pipeline":
        from chaord.lift.pipeline import pipeline_lift_detailed
        program, unexplained = pipeline_lift_detailed(frame, dialect, T)
    else:
        if mode == "auto":
            from ..lang.errors import ChaordError
            try:
                program, unexplained = _nanotube_composition(frame, dialect, T), 0
            except ChaordError as e:
                if "defines no threshold" in str(e):
                    raise  # a configuration bug, not a not-a-tube verdict
                program = None    # not a tube frame: the legacy cascade
        if program is None:
            from chaord.lift.legacy import legacy_lift
            program, unexplained = legacy_lift(frame, dialect, T, mode)
    field_axis = None
    if field_parts is not None:
        stmt, note = field_parts
        _attach_field(program, stmt, note)
        field_axis = stmt.values[-1].text
    _attach_dipole_p1(program, frame, dialect, field_axis=field_axis)
    return _record_lift_diag(program, frame, dialect, unexplained)


def _canonical_cell_frame(frame):
    """The frame re-expressed in its cell's axis-aligned orientation.

    A rigidly rotated copy of a periodic frame (rotation + translation +
    re-imaging + re-ordering) carries the same microstate, but its cell is
    a rotated box and every `frame.cell_diag` wrap in the molecular stack
    would mis-wrap it.  For an orthogonal box the fractional coordinates
    are rotation-invariant, so the canonical positions are simply
    `frac * L` in the diagonal cell -- exact for the already-diagonal case
    (returned unchanged, the same object), one float-rounding away for a
    rotated one.  A non-orthogonal (triclinic) cell is NOT rotated back:
    no rotation canonicalises it, so it passes through untouched and the
    arms' own orthogonal-cell contracts answer (fail closed)."""
    cell = np.asarray(frame.cell, float)
    diag = np.diag(cell)
    if np.array_equal(cell, np.diag(diag)):
        return frame
    gram = cell @ cell.T
    scale = float(np.max(np.diag(gram)))
    off = gram - np.diag(np.diag(gram))
    if not np.allclose(off, 0.0, atol=1e-9 * scale):   # dialect-exempt: numerical-guard: orthogonality of a rigidly rotated box
        return frame
    L = np.sqrt(np.diag(gram))
    frac = frame.pos @ np.linalg.inv(cell)
    from ..io.frames import Frame
    return Frame(pos=np.mod(frac, 1.0) * L, cell=np.diag(L),  # dialect-exempt: numerical-guard: unit fractional range
                 symbols=frame.symbols, pbc=frame.pbc, info=dict(frame.info))


def _nanotube_composition(frame, dialect, T):
    """D35 (docs/reviews/open_items_v4.md, Y3): the tube-plus-liquid
    composition of mode="auto". Returns the composed Program, or raises
    ChaordError when no all-carbon three-coordinated component passes the
    nanotube inference (`nanotube_components`): that refusal is the arm's
    not-my-phase verdict, and the caller falls through to the legacy
    cascade. Once a tube IS identified, every error below propagates --
    the frame is this arm's to describe or to refuse loudly, never to hand
    a half-identified tube to the slab lifter.

    The composition: one crystal tube region per identified tube (the RD2
    chirality inference inside its RD1 cylinder); the atoms INSIDE each
    cylinder lift as the confined liquid region with the molecular stack
    (census, the accessible-volume density with the linear density, the
    Li+ solvation shell and ion pairs, the species radial profiles, and
    the g(r) peak -- pipeline._fit_fluid_region's rule that a sub-region
    is not the box still governs the g(r) normalisation); the atoms
    outside every cylinder become their own `all minus cylinder`
    region. Every atom is explained; the residual block is none."""
    from ..lang.errors import ChaordError
    from ..lang.ir import (
        Chirality, GeoChain, Name, PhysicsBlock, Program, ProvenanceBlock,
        Quantity, RegionBlock, ResidualBlock, ShAll, ShCylinder, Statement,
        StrVal, SystemBlock)
    from .nanotube import nanotube_components

    tubes = nanotube_components(frame, dialect)
    if not tubes:
        raise ChaordError(
            "no all-carbon three-coordinated component passes the nanotube "
            "inference: not a tube frame")
    L = np.asarray(frame.cell_diag, float)

    counts: dict[str, int] = {}
    for s in frame.symbols:
        counts[s] = counts.get(s, 0) + 1
    conserve = []
    for s in sorted(counts):
        conserve += [Name(text=s), Quantity(num=str(counts[s]))]
    system_statements = [
        Statement(kind="build", key="cell", values=[
            Quantity(num=f"{L[i]:.3f}") for i in range(3)]),
        Statement(kind="build", key="pbc", values=[Name(text="xyz")]),
    ]
    if T is not None:
        system_statements.append(
            Statement(kind="state", key="T",
                      values=[Quantity(num=f"{T:.2f}")]))
    system_statements.append(
        Statement(kind="conserve", key="atoms", values=conserve))

    physics_statements = [Statement(
        kind="build", key="backend",
        values=[Name(text="lj" if "lj" in dialect.names else "classical")])]

    tube_atoms = np.concatenate([t.indices for t in tubes])
    assigned = np.zeros(len(frame), bool)
    assigned[tube_atoms] = True
    regions = []
    # Y5's transform-invariance rule: the PRINTED cylinder centres are
    # canonical, not the fitted cell coordinates -- the largest tube's axis
    # is stated at the cell centre and every other tube at its
    # minimum-image offset from it (a translation of the whole frame is the
    # same macrostate; the fitted perp centre would follow the origin).
    # The partition tests below still use each tube's fitted axis, which
    # differs from the canonical one by a whole-cell shift the minimum
    # image absorbs.
    ref = tubes[0].chirality
    cylinders = []           # the canonical cylinders, reused by `outside`
    for k, tube in enumerate(tubes):
        chir = tube.chirality
        axis = "xyz".index(chir.axis)
        perp = [j for j in range(3) if j != axis]
        d = np.array(chir.centre, float)[perp] - np.array(ref.centre, float)[perp]
        d -= L[perp] * np.round(d / L[perp])
        c_can = [0.0, 0.0, 0.0]  # dialect-exempt: exact-geometry: axis-point fill
        c_can[perp[0]], c_can[perp[1]] = (L[perp] / 2 + d)[0], (L[perp] / 2 + d)[1]
        c_can[axis] = chir.centre[axis]     # L/2 for a tube periodic in its cell
        cylinder = ShCylinder(
            axis=chir.axis,
            cx=Quantity(num=f"{c_can[perp[0]]:.3f}"),
            cy=Quantity(num=f"{c_can[perp[1]]:.3f}"),
            cz=Quantity(num=f"{c_can[axis]:.3f}"),
            radius=Quantity(num=f"{chir.radius:.3f}", unit="A"),
            length=Quantity(num=f"{chir.length:.3f}", unit="A"))
        cylinders.append(cylinder)
        suffix = "" if k == 0 else str(k + 1)
        regions.append(RegionBlock(
            phase="crystal", name=f"tube{suffix}",
            geometry=GeoChain(parts=[cylinder], ops=[]),
            statements=[Statement(
                kind="build", key="nanotube",
                values=[Chirality(n=chir.n, m=chir.m), Name(text="length"),
                        Quantity(num=f"{chir.length:.3f}", unit="A")])]))
        # the confined fluid: non-tube atoms inside this cylinder
        d = frame.pos[:, perp] - np.array(chir.centre)[perp]
        d -= L[perp] * np.round(d / L[perp])
        inside = (~assigned) & (np.hypot(d[:, 0], d[:, 1]) < chir.radius)
        if not inside.any():
            continue
        assigned |= inside
        sub = _subset(frame, np.where(inside)[0])
        volume = np.pi * chir.radius ** 2 * chir.length
        regions.append(RegionBlock(
            phase="liquid", name=f"fluid{suffix}",
            geometry=GeoChain(parts=[_copy_cylinder(cylinder)], ops=[]),
            statements=_confined_statements(sub, volume, dialect, tube=chir)))

    outside = np.where(~assigned)[0]
    if len(outside):
        volume = float(np.prod(L)) - sum(
            np.pi * t.chirality.radius ** 2 * t.chirality.length
            for t in tubes)
        if volume <= 0:
            raise ChaordError(
                "the identified cylinders' volumes exceed the cell (do the "
                "tubes overlap?): the outside region has no volume to state")
        geometry = GeoChain(
            parts=[ShAll()] + [_copy_cylinder(c) for c in cylinders],
            ops=["minus"] * len(tubes))
        regions.append(RegionBlock(
            phase="liquid", name="outside", geometry=geometry,
            statements=_confined_statements(_subset(frame, outside),
                                            volume, dialect)))

    provenance = [
        Statement(kind="build", key="dialects",
                  values=[StrVal(text=dialect.version_string)]),
        Statement(kind="build", key="lift_version",
                  values=[StrVal(text="0.1.0")]),
        Statement(kind="build", key="note", values=[StrVal(text=(
            "composed tube-plus-liquid program (D35, docs/reviews/"
            "open_items_v4.md Y3): all-carbon three-coordinated components "
            "lifted through the chirality inference as crystal tube regions; "
            "atoms inside each cylinder as the confined fluid region; atoms "
            "outside as their own region"))]),
        Statement(kind="build", key="note", values=[StrVal(text=(
            "fluid g(r) referred to the tube-interior region volume; the "
            "density to the accessible volume pi (R - r_wall)^2 L with "
            "r_wall from the dialect (pipeline._fit_fluid_region's rule: "
            "a sub-region is not the box; Y5 item 4, open_items_v4.md)"))]),
    ]
    if isinstance(T, AssumedT):
        provenance.append(Statement(
            kind="build", key="note",
            values=[StrVal(text=(f"T assumed (dialect default {float(T):g}); "
                                 "not measured from the frame"))]))
    return Program(
        version="0.1",  # dialect-exempt: numerical-guard: language version constant
        dialects=list(dialect.names),
        blocks=[SystemBlock(statements=system_statements),
                PhysicsBlock(statements=physics_statements),
                *regions, ResidualBlock(none=True),
                ProvenanceBlock(statements=provenance)])


def _copy_cylinder(cyl):
    """A fresh ShCylinder equal to `cyl` (pydantic models are shared by
    reference; two regions must not share one mutable geometry node)."""
    from ..lang.ir import ShCylinder
    return ShCylinder(axis=cyl.axis, cx=cyl.cx, cy=cyl.cy, cz=cyl.cz,
                      radius=cyl.radius, length=cyl.length)


def _subset(frame, idx):
    """The frame restricted to `idx`, carrying the charges (the LAMMPS
    reader's convention) so the dipole observables of the composed program
    measure from the file's own charges."""
    import numpy as np
    from ..io.frames import Frame
    info = {}
    if frame.info.get("charges") is not None:
        q = np.asarray(frame.info["charges"], float)
        info["charges"] = q[idx]
    return Frame(pos=frame.pos[idx], cell=frame.cell,
                 symbols=[frame.symbols[i] for i in idx],
                 pbc=frame.pbc, info=info)


def _accessible_radius(dialect, tube) -> float | None:
    """R - r_wall of the accessible volume pi (R - r_wall)^2 L (Y5 item 4),
    or None when the dialect names no `pore_wall_exclusion` (stacks without
    the research keys keep the legacy region-volume density) or the tube is
    narrower than the wall exclusion (no accessible volume to state; the
    legacy density stands)."""
    from ..lang.errors import ChaordError
    if tube is None:
        return None
    try:
        r_wall = float(dialect.threshold("pore_wall_exclusion"))
    except ChaordError as e:
        if "defines no threshold" in str(e):
            return None
        raise
    r_acc = float(tube.radius) - r_wall
    return r_acc if r_acc > 0 else None


def _electrolyte_statements(sub, tube, dialect) -> list:
    """The Y5 electrolyte statements of one fluid region (D36,
    docs/reviews/open_items_v4.md): the Li+ solvation shell BY SPECIES on
    the dialect's named sites with its g(r)-derived cutoffs, the
    SSIP/CIP/AGG ion-pair classes, and -- inside a tube -- the species
    radial number-density profiles up to the tube radius.

    Dialect-gated exactly like the blind dipole statement (D29): a stack
    whose dialect names no solvation species (`li_shell_sites` absent)
    declares the observable out of scope and gets none of it; a frame with
    no cation has no shell to average and gets none of it either.  Every
    number comes from chaord.cv.field (one definition per quantity)."""
    from ..cv.field import solvation_shell, species_radial_profiles
    from ..lang.errors import ChaordError
    from ..lang.ir import Name, Quantity, Statement, Tol
    from .reactive import display_name
    try:
        shell = solvation_shell(sub, dialect)
    except ChaordError as e:
        if "defines no threshold" in str(e):
            return []
        raise
    statements = []
    if shell is not None:
        rules = dialect.threshold("li_shell_sites")
        for formula, (mean, sd) in shell["per_species"].items():
            rule = str(rules[formula])
            statements.append(Statement(
                kind="assert", key="li_shell",
                values=[Name(text=display_name(formula, dialect)),
                        Quantity(num=f"{mean:.1f}"),
                        Tol(value=Quantity(num=f"{sd:.1f}")),
                        Name(text=rule),
                        Quantity(num=f"{shell['cutoffs'][formula]:.2f}",
                                 unit="A")]))
        ip = shell["ion_pairs"]
        statements.append(Statement(
            kind="assert", key="ion_pairs",
            values=[v for k in ("SSIP", "CIP", "AGG") for v in
                    (Name(text=k), Quantity(num=str(ip[k])))]))
    if tube is not None:
        axis = "xyz".index(tube.axis)
        perp = [j for j in range(3) if j != axis]
        try:
            width = float(dialect.threshold("species_nr_bin_width"))
            profiles = species_radial_profiles(
                sub, dialect, axis=tube.axis,
                centre=(tube.centre[perp[0]], tube.centre[perp[1]]),
                r_max=float(tube.radius))
        except ChaordError as e:
            if "defines no threshold" in str(e):
                profiles = {}
            else:
                raise
        for formula, prof in profiles.items():
            statements.append(Statement(
                kind="state", key="n_r",
                values=[Name(text=display_name(formula, dialect))]
                + [Quantity(num=f"{v:.4f}") for v in prof["profile"]]
                + [Name(text="bin"), Quantity(num=f"{width:g}", unit="A")]))
    return statements


def _confined_statements(sub, volume, dialect, tube=None) -> list:
    """The legacy fluid lifter's statement set on a sub-region: census,
    density and g(r) referred to the REGION volume (the rule the old
    research pipeline used, one definition -- pipeline._fit_fluid_region).

    Y5 (D36) for a MOLECULAR fluid:
      * inside a tube (`tube`, a NanotubeChirality), the density is stated
        on the ACCESSIBLE volume pi (R - r_wall)^2 L with the linear
        density (molecules per A of tube axis) next to it (item 4);
      * the molecule-centre `cn` assert is REPLACED by the site-based
        solvation-shell asserts (item 5): the centre-centre count at the
        only defensible cutoff -- the O-O first shell `cn_cutoff_fluid` --
        asserts that no two molecule centres come within it, trivially
        true for 10- and 15-atom molecules (0.0 +- 0.0 in 55 of the 84
        pilot frames), while the Li+ shell statements carry this
        electrolyte's actual coordination;
      * `n_r` species radial number-density profiles inside the tube
        radius (item 3).
    Atomic fluids keep their statements unchanged (their atoms ARE the
    sites, so centre-cn is site-based already)."""
    import numpy as np
    from ..build.molecules import molecule_census
    from ..lang.ir import Name, Quantity, Statement, Tol
    from .fluid import _all_elements, _centers, _cn_stats
    from .pipeline import _rdf_region
    from .reactive import display_name

    statements = []
    census = molecule_census(sub, dialect) if _all_elements(sub) else {}
    molecular = any(len(n) > 2 or not _is_element_name(n) for n in census)
    if molecular:
        for formula in sorted(census):
            statements.append(Statement(
                kind="build", key="molecules",
                values=[Name(text=display_name(formula, dialect)),
                        Quantity(num=str(census[formula]))]))
        from ase.data import atomic_masses, chemical_symbols
        total_mass = sum(atomic_masses[chemical_symbols.index(s)]
                         for s in sub.symbols)
        r_acc = _accessible_radius(dialect, tube)
        if r_acc is not None:
            rho_g = (total_mass / (np.pi * r_acc ** 2 * float(tube.length))
                     / 0.6022140857)  # dialect-exempt: exact-geometry: u/A^3 -> g/cm3 (fluid.py constant)
            statements.append(Statement(
                kind="state", key="density",
                values=[Quantity(num=f"{rho_g:.3f}", unit="g/cm3"),
                        Name(text="accessible")]))
            statements.append(Statement(
                kind="state", key="linear_density",
                values=[Quantity(
                    num=f"{sum(census.values()) / float(tube.length):.3f}"),
                    Name(text="A^-1")]))
        else:
            rho_g = total_mass / volume / 0.6022140857  # dialect-exempt: exact-geometry: u/A^3 -> g/cm3  (fluid.py constant)
            statements.append(Statement(
                kind="state", key="density",
                values=[Quantity(num=f"{rho_g:.3f}", unit="g/cm3")]))
        centers = _centers(sub, dialect)
    else:
        rho_n = len(sub) / volume if volume > 0 else 0.0  # dialect-exempt: numerical-guard: empty-region density
        statements.append(Statement(
            kind="state", key="density", values=[Quantity(num=f"{rho_n:.3f}")]))
        centers = None
        cn, cn_sd, cn_cut = _cn_stats(sub, dialect)
        statements.append(Statement(
            kind="assert", key="cn",
            values=[Quantity(num=f"{cn:.1f}"), Tol(value=Quantity(num=f"{cn_sd:.1f}")),
                    Name(text="cutoff"), Quantity(num=f"{cn_cut:.2f}", unit="A")]))
    statements += _electrolyte_statements(sub, tube, dialect)
    if centers is not None:
        rm, g = _rdf_region(centers, sub.cell_diag, len(centers) / volume,
                            dialect)
    else:
        rm, g = _rdf_region(sub.pos, sub.cell_diag, len(sub) / volume, dialect)
    if len(rm):
        ipk = int(np.argmax(g))
        statements.append(Statement(
            kind="assert", key="gr_peak",
            values=[Quantity(num=f"{rm[ipk]:.2f}", unit="A"),
                    Name(text="height"), Quantity(num=f"{g[ipk]:.2f}")]))
    if not molecular:
        statements.append(Statement(
            kind="assert", key="solid_clusters", values=[Quantity(num="0")]))
    return statements


def _is_element_name(name: str) -> bool:
    from ase.data import chemical_symbols
    return name in chemical_symbols and name != "X"


def _field_parts(field):
    """(statement, note text) of a caller-supplied field spec, validated
    fail-closed (unknown axis, missing value/axis refuse the lift)."""
    from ..lang.errors import ChaordError
    from ..lang.ir import Name, Quantity, Statement
    if not isinstance(field, dict) or "value" not in field or "axis" not in field:
        raise ChaordError(
            'field needs at least {"value": v, "axis": "x|y|z"} '
            f'(RD3); got {field!r}')
    axis = str(field["axis"])
    if axis not in ("x", "y", "z"):
        raise ChaordError(f"field axis must be x, y or z, got {axis!r}")
    component = str(field.get("component", "E"))
    unit = field.get("unit", "V/A")
    num = f"{float(field['value']):g}"
    stmt = Statement(kind="state", key="field", values=[
        Name(text=component),
        Quantity(num=num, unit=str(unit) if unit else None),
        Name(text="along"), Name(text=axis)])
    note = (f"field assumed (caller-supplied {component} = {num}"
            + (f" {unit}" if unit else "") + f" along {axis}); not measured")
    return stmt, note


def _attach_field(program, stmt, note):
    """Print the `state field` statement next to `state T` in the system
    block and the assumed-value note in provenance, on every lift path."""
    from ..lang.ir import ProvenanceBlock, Statement, StrVal, SystemBlock
    system = next((b for b in program.blocks if b.t == "system"), None)
    if system is None:
        system = SystemBlock(statements=[])
        program.blocks.insert(0, system)
    t_at = next((k for k, s in enumerate(system.statements)
                 if s.kind == "state" and s.key == "T"), None)
    at = (t_at + 1 if t_at is not None else
          next((k for k, s in enumerate(system.statements)
                if s.kind == "conserve"), len(system.statements)))
    system.statements.insert(at, stmt)
    _append_provenance_note(program, note)


def _append_provenance_note(program, text):
    """Append one `note` statement to the program's provenance block,
    creating the block when the path emitted none."""
    from ..lang.ir import ProvenanceBlock, Statement, StrVal
    prov = next((b for b in program.blocks if b.t == "provenance"), None)
    if prov is None:
        prov = ProvenanceBlock(statements=[])
        program.blocks.append(prov)
    prov.statements.append(Statement(kind="build", key="note",
                                     values=[StrVal(text=text)]))


def _names_dipolar_species(dialect) -> bool:
    """Does this dialect name dipolar species (`field_dipole_species`)?

    The D29 gate: the orientation statement exists only for dialects that
    name the species P1 averages over -- a dialect without the key (core +
    lj, the crystal stacks) declares the observable out of scope, and a
    blind lift of such a frame simply has nothing to assert. The key's
    absence is Dialect.threshold's own "defines no threshold" error; it is
    matched by message, the same carve-out the legacy ladder makes for the
    one ChaordError that is not a phase verdict."""
    from ..lang.errors import ChaordError
    try:
        return bool(dialect.threshold("field_dipole_species"))
    except ChaordError as e:
        if "defines no threshold" in str(e):
            return False
        raise


def _resolve_dipole_axis(program, dialect, field_axis):
    """The `assert dipole_p1 ... along <axis>` analysis axis (Y2 step 2).

    Order: the caller-supplied field's axis; else the nanotube region's
    axis when the program holds one (a confined fluid's natural analysis
    axis is its tube's -- the composition of Y3 emits exactly that region);
    else the dialect's `dipole_axis_default`. Fail closed when nothing
    names an axis: a measured statement must not invent its own geometry.
    Returns (axis, source); the source names the winning rule (kept for
    tests and debugging -- the provenance note stays identical with and
    without a field, so it does not carry it)."""
    if field_axis is not None:
        return str(field_axis), "the field's axis"
    axes = []
    for block in program.blocks:
        if getattr(block, "t", None) != "region":
            continue
        if not any(s.kind == "build" and s.key == "nanotube"
                   for s in block.statements):
            continue
        for part in getattr(block.geometry, "parts", ()):
            axis = getattr(part, "axis", None)
            if axis is not None:
                axes.append(str(axis))
    if len(set(axes)) > 1:
        from ..lang.errors import ChaordError
        raise ChaordError(
            "the program's nanotube regions name different axes "
            f"({sorted(set(axes))}); cannot pick the dipole_p1 analysis axis")
    if axes:
        return axes[0], "the nanotube region's axis"
    return str(dialect.threshold("dipole_axis_default")), \
        "the dialect default (dipole_axis_default)"


def _attach_dipole_p1(program, frame, dialect, field_axis=None):
    """D29 (docs/reviews/open_items_v4.md): assert the dipole-orientation
    order parameter for every frame whose dialect names dipolar species,
    whether or not a field was supplied.

    The value is measured, never assumed: cv.field.dipole_orientation on
    the frame this lift was given (rule 7 -- the same CV the registry names
    `dipole_p1`), with the frame's own per-atom charges when the reader
    carried them, else the dialect's site-charge table. The printed
    tolerance is 1/sqrt(n_molecules): every cos(theta) lies in [-1, 1], so
    Popoviciu's variance bound (T. Popoviciu, Mathematica (Cluj) 9, 129
    (1935): a variable in [a, b] has variance at most (b-a)^2/4 = 1 here)
    caps the standard error of the mean of n cosines -- an a-priori bound,
    stated as such in provenance, NOT the empirical noise floor (which the
    analysis measures independently from frame pairs and never enters the
    text). dipole_orientation returns only the mean and count, so the
    empirical spread is not available without a second definition of the
    quantity (cv/ is frozen this round); the bound needs neither.

    Degrade with a note, never silence: when the dialect names species but
    the measurement cannot run (no such species in the frame, no charge
    model, a zero-dipole species), provenance records the reason and the
    lift stands. A dialect that names no species promises nothing: the
    blind lift stays silent (no assert, no note); with `field=` given the
    RD3 statement stands and its missing consequence is recorded. Every
    other error propagates -- only ChaordError, the documented refusal
    channel, degrades, and never the "defines no threshold" configuration
    bug (the legacy ladder's carve-out, see _names_dipolar_species)."""
    import math
    from ..lang.errors import ChaordError
    from ..lang.ir import Name, Quantity, Statement, Tol

    has_field_line = any(
        s.kind == "state" and s.key == "field"
        for b in program.blocks if b.t == "system" for s in b.statements)
    if not _names_dipolar_species(dialect):
        if has_field_line:
            _append_provenance_note(
                program,
                "field observables not asserted (the dialect names no "
                "field_dipole_species: this stack's fluids have no dipole "
                "to orient)")
        return
    axis, _axis_source = _resolve_dipole_axis(program, dialect, field_axis)
    try:
        from ..cv.field import dipole_orientation
        dip = dipole_orientation(frame, dialect, axis=axis)
    except ChaordError as e:
        if "defines no threshold" in str(e):
            raise   # a configuration bug, not a not-measurable verdict
        prefix = ("field observables" if has_field_line
                  else "dipole observables")
        _append_provenance_note(
            program, f"{prefix} not asserted ({type(e).__name__}: {e})")
        return
    n = int(dip["n_molecules"])
    tol = 1 / math.sqrt(n)
    assert_stmt = Statement(
        kind="assert", key="dipole_p1",
        values=[Quantity(num=f"{float(dip['P1']):.3f}"),
                Tol(value=Quantity(num=f"{tol:.3f}")),
                Name(text="along"), Name(text=axis)])
    system = next(b for b in program.blocks if b.t == "system")
    at = next((k for k, s in enumerate(system.statements)
               if s.kind == "state" and s.key == "field"), None)
    if at is None:
        at = next((k for k, s in enumerate(system.statements)
                   if s.kind == "state" and s.key == "T"), None)
    if at is None:
        at = next((k for k, s in enumerate(system.statements)
                   if s.kind == "conserve"), len(system.statements))
    system.statements.insert(at + 1, assert_stmt)
    source = ("the frame's own per-atom charges"
              if frame.info.get("charges") is not None
              else "the dialect's site-charge model")
    # The note is byte-identical with and without field= (the Y2 done-when:
    # the two texts differ ONLY in the `state field` line and its note), so
    # it records the measurement, never which rule resolved the axis.
    _append_provenance_note(
        program,
        f"dipole P1 measured from {source} (n = {n} molecules, axis "
        f"{axis}); "
        f"tol = 1/sqrt(n) = {tol:.3f} is the a-priori standard-error bound "
        "of a mean of n cosines (Popoviciu, Mathematica (Cluj) 9, 129 "
        "(1935): variance <= 1 for a variable on [-1,1]), not an empirical "
        "floor")


def _record_lift_diag(program, frame, dialect, unexplained):
    """Register the lift diagnostic and run the static checks after the lift.

    The conservation check runs after EVERY lift (AGENTS rule 6); a failure is
    recorded in the diagnostic instead of raised, so lifting a hard frame
    still returns its program - the CLI and `chaord check` report the failures."""
    from ..check.statics import record_lift_diag, run_checks
    diag = {"unexplained": int(unexplained)}
    try:
        failed = [f"{c.name}: {c.detail}"
                  for c in run_checks(program, frame, dialect) if not c.passed]
    except Exception as e:  # checks measure: a broken check must not kill the lift
        diag["checks_error"] = f"{type(e).__name__}: {e}"
    else:
        if failed:
            diag["failed_checks"] = failed
    record_lift_diag(program, diag)
    return program
