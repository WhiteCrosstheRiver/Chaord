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
    """Lift a Frame into a Program; mode is auto | crystal | defects | slab.

    T is metadata (stated in the program, never measured); when omitted it is
    read from the dialect's md_reference_T."""
    program, unexplained = _route(frame, dialect, T, mode)
    return _record_lift_diag(program, frame, dialect, unexplained)


def _route(frame, dialect, T, mode):
    """The routing ladder; returns (program, unexplained atom count).

    Every path explains all of its atoms except the M0 slab lifter, whose
    off-lattice atoms are either accounted as displaced by a defect complex or
    must appear in the program's residual block."""
    from ..lang.errors import ChaordError
    if mode in ("auto", "crystal", "defects"):
        try:
            return lift_crystal(frame, dialect), 0
        except Exception:
            if mode == "crystal":
                raise
        if mode in ("auto", "defects"):
            try:
                program, _diag = lift_crystal_defects(frame, dialect)
                return program, 0
            except Exception:
                if mode == "defects":
                    raise
    if mode in ("auto", "amorphous"):
        from .amorphous import is_amorphous, lift_amorphous
        # a glass and a liquid are both disordered in one frame: the DIALECT
        # decides which macrostate a program describes (glass -> amorphous)
        if ("glass" in dialect.names or mode == "amorphous") and is_amorphous(frame, dialect):
            return lift_amorphous(frame, dialect), 0
        if mode == "amorphous":
            from ..lang.errors import ChaordError
            raise ChaordError("frame is not a bonded disordered network")
    if mode in ("auto", "surface"):
        from .surface import has_vacuum, lift_surface
        if has_vacuum(frame, dialect):
            return lift_surface(frame, dialect), 0
        if mode == "surface":
            raise ChaordError("no vacuum gap found: not a surface frame")
    if mode in ("auto", "fluid"):
        from .fluid import is_single_phase
        if is_single_phase(frame, dialect):
            if T is None:
                try:
                    T = float(dialect.threshold("md_reference_T"))
                except Exception:
                    T = None
            return lift_fluid(frame, dialect, T=T), 0
        if mode == "fluid":
            raise ChaordError("frame is not a single-phase fluid "
                              "(solid-like fraction too high)")
    if mode in ("auto", "slab"):
        if T is None:
            T = float(dialect.threshold("md_reference_T"))
        try:
            res = decompile(frame.pos, frame.cell_diag, T, dialect)
            program = program_from_result(res, dialect)
            # atoms the slab lifter could not explain: off-lattice atoms
            # minus those accounted as displaced by defect complexes
            explained = sum(c["displaced"] for c in res["defects"])
            return program, max(len(res["off"]) - explained, 0)
        except Exception:
            if mode == "slab":
                raise
            # no interfaces found: a single-phase fluid after all
            return lift_fluid(frame, dialect, T=T), 0
    raise ValueError(f"unknown lift mode {mode!r}")


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
