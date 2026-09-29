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
from chaord.lift.passes import otsu, pairs_within, qbar  # noqa: F401
from chaord.lift.slab import decompile, program_from_result  # noqa: F401
from chaord.lift.defect_program import lift_crystal_defects  # noqa: F401
from chaord.lift.fluid import lift_fluid  # noqa: F401


def lift_frame(frame, dialect, T=None, mode="auto"):
    """Lift a Frame into a Program; mode is auto | pipeline | crystal |
    defects | amorphous | surface | fluid | slab.

    mode="pipeline" runs the segment-first pipeline (lift/build v2 stage 1);
    every other mode is the legacy try-cascade (lift/legacy.py). mode="auto"
    stays on the legacy cascade until the pipeline equivalence suite is green.

    T is metadata (stated in the program, never measured); when omitted it is
    read from the dialect's md_reference_T."""
    if mode == "pipeline":
        from chaord.lift.pipeline import pipeline_lift_detailed
        program, unexplained = pipeline_lift_detailed(frame, dialect, T)
        return _record_lift_diag(program, frame, dialect, unexplained)
    from chaord.lift.legacy import legacy_lift
    program, unexplained = legacy_lift(frame, dialect, T, mode)
    return _record_lift_diag(program, frame, dialect, unexplained)


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
