"""Decompiler (chaord lift): coordinates -> program.

lift_frame routes: a fully ordered single-crystal frame lifts through the exact
crystal path; a crystal with point defects lifts through the Wigner-Seitz
defect path; anything with regions of different phase falls back to the M0
solid-liquid slab lifter.
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
    from ..lang.errors import ChaordError
    if mode in ("auto", "crystal", "defects"):
        try:
            return lift_crystal(frame, dialect)
        except Exception:
            if mode == "crystal":
                raise
        if mode in ("auto", "defects"):
            try:
                program, _diag = lift_crystal_defects(frame, dialect)
                return program
            except Exception:
                if mode == "defects":
                    raise
    if mode in ("auto", "amorphous"):
        from .amorphous import is_amorphous, lift_amorphous
        # a glass and a liquid are both disordered in one frame: the DIALECT
        # decides which macrostate a program describes (glass -> amorphous)
        if ("glass" in dialect.names or mode == "amorphous") and is_amorphous(frame, dialect):
            return lift_amorphous(frame, dialect)
        if mode == "amorphous":
            from ..lang.errors import ChaordError
            raise ChaordError("frame is not a bonded disordered network")
    if mode in ("auto", "surface"):
        from .surface import has_vacuum, lift_surface
        if has_vacuum(frame, dialect):
            return lift_surface(frame, dialect)
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
            return lift_fluid(frame, dialect, T=T)
        if mode == "fluid":
            raise ChaordError("frame is not a single-phase fluid "
                              "(solid-like fraction too high)")
    if mode in ("auto", "slab"):
        if T is None:
            T = float(dialect.threshold("md_reference_T"))
        try:
            res = decompile(frame.pos, frame.cell_diag, T, dialect)
            return program_from_result(res, dialect)
        except Exception:
            if mode == "slab":
                raise
            # no interfaces found: a single-phase fluid after all
            return lift_fluid(frame, dialect, T=T)
    raise ValueError(f"unknown lift mode {mode!r}")
