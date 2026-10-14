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
    the value was assumed, not measured -- the W11 assumed-value rule exactly as
    fluid.AssumedT applies it. Given none, nothing is printed (there is no
    honest default field).

    RD4 (same record) adds the statement-level evidence: given a field, the
    program also ASSERTS the field's measurable consequence --
    `assert field_dipole_p1 <v> +- <tol> along <axis>` right after the
    state field statement, measured by cv.field.dipole_orientation from the
    frame's own charges -- or, when the dialect cannot measure it (no
    field_dipole_species, no such species, no charge model), records the
    reason in provenance and keeps RD3's statement. Given none, no
    field-related statement exists at all."""
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
    if field_parts is not None:
        stmt, note = field_parts
        _attach_field(program, stmt, note)
        _attach_field_observables(program, frame, dialect,
                                  axis=stmt.values[-1].text)
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


def _attach_field_observables(program, frame, dialect, axis):
    """RD4 (docs/reviews/research_decisions.md): assert the field's
    measurable consequence next to the `state field` statement.

    The value is measured, never assumed: cv.field.dipole_orientation on
    the frame this lift was given (rule 7 -- the same CV the registry names
    `field_dipole_p1`), with the frame's own per-atom charges when the
    reader carried them, else the dialect's site-charge table. The printed
    tolerance is 1/sqrt(n_molecules): every cos(theta) lies in [-1, 1], so
    Popoviciu's variance bound (T. Popoviciu, Mathematica (Cluj) 9, 129
    (1935): a variable in [a, b] has variance at most (b-a)^2/4 = 1 here)
    caps the standard error of the mean of n cosines -- an a-priori bound,
    stated as such in provenance, NOT the empirical noise floor (which the
    analysis measures independently from frame pairs and never enters the
    text). dipole_orientation returns only the mean and count, so the
    empirical spread is not available without a second definition of the
    quantity (cv/ is frozen this round); the bound needs neither.

    Degrade with a note, never silence: when the dialect cannot measure the
    observable (no field_dipole_species threshold -- e.g. core + lj -- no
    named species in the frame, no charge model), RD3's binding statement
    stands and provenance records the reason. With `field=` absent this
    function is never called: no field statement, no field assert."""
    import math
    from ..lang.ir import Name, Quantity, Statement, Tol
    try:
        from ..cv.field import dipole_orientation
        dip = dipole_orientation(frame, dialect, axis=axis)
    except Exception as e:  # the observable is not this dialect's to measure
        _append_provenance_note(
            program, f"field observables not asserted ({type(e).__name__}: {e})")
        return
    n = int(dip["n_molecules"])
    tol = 1 / math.sqrt(n)
    assert_stmt = Statement(
        kind="assert", key="field_dipole_p1",
        values=[Quantity(num=f"{float(dip['P1']):.3f}"),
                Tol(value=Quantity(num=f"{tol:.3f}")),
                Name(text="along"), Name(text=axis)])
    system = next(b for b in program.blocks if b.t == "system")
    field_at = next(k for k, s in enumerate(system.statements)
                    if s.kind == "state" and s.key == "field")
    system.statements.insert(field_at + 1, assert_stmt)
    source = ("the frame's own per-atom charges"
              if frame.info.get("charges") is not None
              else "the dialect's site-charge model")
    _append_provenance_note(
        program,
        f"field P1 measured from {source} (n = {n} molecules); "
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
