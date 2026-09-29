"""Legacy lift: the try-cascade of lift/build v1, moved here verbatim.

Stage 1-2 compatibility path (docs/design/lift_build_v2.md 2.6): the cascade
stays alive behind ``mode=auto`` / the pinned arm modes until the pipeline
equivalence suite is green; it is deleted in stage 3. Behaviour is identical
to the pre-pipeline ``lift/__init__.py:_route`` -- this module is a move, not
a rewrite.
"""
from chaord.lift.crystal import lift_crystal  # noqa: F401
from chaord.lift.defect_program import lift_crystal_defects  # noqa: F401
from chaord.lift.fluid import lift_fluid  # noqa: F401


def legacy_lift(frame, dialect, T=None, mode="auto"):
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
        from .slab import decompile, program_from_result
        if T is None:
            try:
                T = float(dialect.threshold("md_reference_T"))
            except Exception:
                T = None  # T is metadata: dialects without MD defaults omit it
        try:
            res = decompile(frame.pos, frame.cell_diag, T, dialect,
                            symbols=frame.symbols)
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
