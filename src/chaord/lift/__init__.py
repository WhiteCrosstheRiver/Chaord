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
    system block next to `state T`, with a provenance note recording that the
    value was assumed, not measured -- the W11 assumed-value rule exactly as
    fluid.AssumedT applies it. Given none, nothing is printed (there is no
    honest default field)."""
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
        _attach_field(program, *field_parts)
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
    prov = next((b for b in program.blocks if b.t == "provenance"), None)
    if prov is None:
        prov = ProvenanceBlock(statements=[])
        program.blocks.append(prov)
    prov.statements.append(Statement(kind="build", key="note",
                                     values=[StrVal(text=note)]))


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
