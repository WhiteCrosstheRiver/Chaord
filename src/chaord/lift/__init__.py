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
    NanotubeChirality, infer_nanotube_chirality,
)
from chaord.lift.passes import otsu, pairs_within, qbar  # noqa: F401
from chaord.lift.slab import decompile, program_from_result  # noqa: F401
from chaord.lift.defect_program import lift_crystal_defects  # noqa: F401
from chaord.lift.fluid import AssumedT, lift_fluid  # noqa: F401


def lift_frame(frame, dialect, T=None, mode="auto", field=None):
    """Lift a Frame into a Program; mode is auto | pipeline | crystal |
    defects | amorphous | surface | fluid | slab.

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
    holds one, else the dialect's `dipole_axis_default` (research.yaml)."""
    field_parts = _field_parts(field) if field is not None else None
    if T is None:
        try:
            T = AssumedT(float(dialect.threshold("md_reference_T")))
        except Exception:
            T = None  # dialects without an MD default state no temperature
    if mode == "pipeline":
        from chaord.lift.pipeline import pipeline_lift_detailed
        program, unexplained = pipeline_lift_detailed(frame, dialect, T)
    else:
        from chaord.lift.legacy import legacy_lift
        program, unexplained = legacy_lift(frame, dialect, T, mode)
    field_axis = None
    if field_parts is not None:
        stmt, note = field_parts
        _attach_field(program, stmt, note)
        field_axis = stmt.values[-1].text
    _attach_dipole_p1(program, frame, dialect, field_axis=field_axis)
    return _record_lift_diag(program, frame, dialect, unexplained)


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
