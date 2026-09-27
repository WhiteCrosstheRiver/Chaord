"""Decompiler (chaord lift): coordinates -> program.

lift_frame routes: a fully ordered single-crystal frame lifts through the exact
crystal path; anything with regions of different phase falls back to the M0
solid-liquid slab lifter.
"""
from chaord.lift.crystal import lift_crystal  # noqa: F401
from chaord.lift.passes import otsu, pairs_within, qbar  # noqa: F401
from chaord.lift.slab import decompile, program_from_result  # noqa: F401


def lift_frame(frame, dialect, T=None, mode="auto"):
    """Lift a Frame into a Program; mode is auto | crystal | slab.

    T is metadata (stated in the program, never measured); when omitted it is
    read from the dialect's md_reference_T."""
    if mode in ("auto", "crystal"):
        try:
            return lift_crystal(frame, dialect)
        except Exception:
            if mode == "crystal":
                raise
    if mode in ("auto", "slab"):
        if T is None:
            T = float(dialect.threshold("md_reference_T"))
        res = decompile(frame.pos, frame.cell_diag, T, dialect)
        return program_from_result(res, dialect)
    raise ValueError(f"unknown lift mode {mode!r}")
