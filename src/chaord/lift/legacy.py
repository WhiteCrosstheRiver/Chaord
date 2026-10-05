"""Legacy lift: the try-cascade of lift/build v1, moved here verbatim.

Stage 1-2 compatibility path (docs/design/lift_build_v2.md 2.6): the cascade
stays alive behind ``mode=auto`` / the pinned arm modes until the pipeline
equivalence suite is green; it is deleted in stage 3. Behaviour is identical
to the pre-pipeline ``lift/__init__.py:_route`` -- this module is a move, not
a rewrite -- except the orthohexagonal hcp arm added for the F2 fix
(reports/redteam_findings.md): frames shipped in an orthogonal box on the
hexagonal axes (orient x [100] y [1 2 0] z [001]) cannot be recognised by
spglib once thermally jittered (P1 at lift_symprec) and the cubic
Wigner-Seitz fit supports no hexagonal host, so the cascade swallowed both
refusals and its fluid fallback invented the pseudo-molecule 'Mg32'. The
hcp arm (build/prototypes.py fit) runs before the spglib arm in auto and
crystal modes: where it applies it is exact (the box IS the orthohexagonal
supercell), it claims the perfect rebuild as well as the thermal frame, and
so keeps lift -> build -> lift byte-stable for that setting.
"""
from chaord.lift.crystal import lift_crystal  # noqa: F401
from chaord.lift.defect_program import lift_crystal_defects  # noqa: F401
from chaord.lift.fluid import lift_fluid  # noqa: F401


class NotThisPhase(Exception):
    """An arm's explicit verdict that the frame is not its phase (W1 step 4,
    Review 8: fail closed).

    The cascade falls through to the next arm ONLY on this verdict; a missing
    threshold or any other error propagates -- a lift that cannot describe a
    frame refuses with an error, it never silently walks the ladder into a
    program no arm stands behind.  The arm modules predate the exception and
    signal the same verdict with ChaordError, so the ladder converts (see
    `_not_this_phase`)."""


def _not_this_phase(exc: Exception) -> bool:
    """Is `exc` an arm's explicit 'not my phase' refusal?

    ChaordError is the refusal channel of this codebase (the pinned-mode
    contract: "a refusal is a ChaordError, never a raw engine exception"),
    with ONE exception: the threshold-lookup error raised by
    ``Dialect.threshold`` itself ("dialect ... defines no threshold ...").
    A missing dialect key is a configuration bug, not a phase verdict, and
    must propagate out of the cascade (AGENTS rule 3, W1 step 4)."""
    from ..lang.errors import ChaordError
    return (isinstance(exc, ChaordError)
            and "defines no threshold" not in str(exc))


def _spglib_refusal(exc: Exception) -> bool:
    """The spglib (exact-crystal) arm's refusals, including the mangled one.

    ``lift/crystal.py`` (outside W1's file ownership) predates the ChaordError
    convention and states its not-a-crystal verdicts as bare exceptions: two
    ValueError messages, plus a third that its unpack-before-None-check bug
    mangles into ``TypeError: cannot unpack non-iterable NoneType object``
    (spglib returns None for a frame it cannot standardise, and
    ``cell, pos, numbers = spglib.standardize_cell(...)`` raises before the
    intended "spglib could not standardise the cell" ValueError).  The ladder
    reads all three as the not-a-crystal verdict until the arm states them
    as ChaordError itself.  The None-unpack pattern is the only generic
    message in this set; every other error type still propagates (fail
    closed), so the conversion stays strictly narrower than the blanket
    swallow the ladder had before W1."""
    if isinstance(exc, ValueError) and str(exc).startswith((
            "spglib could not standardise the cell",
            "no prototype matches the standardised structure")):
        return True
    return (isinstance(exc, TypeError)
            and "cannot unpack non-iterable NoneType" in str(exc))


def _refusal(exc: Exception) -> Exception:
    """NotThisPhase for a recognised refusal, the original error otherwise.

    Callers ``raise _refusal(e)`` inside the except block: a recognised
    refusal converts to NotThisPhase (the only exception the ladder falls
    through on); everything else re-raises as itself, original traceback
    intact (the same exception object)."""
    if isinstance(exc, NotThisPhase):
        return exc
    if _not_this_phase(exc) or _spglib_refusal(exc):
        return NotThisPhase(f"{type(exc).__name__}: {exc}")
    return exc


def _default_T(dialect, T):
    """The caller's T, else the dialect's md_reference_T (metadata); dialects
    without an MD default state no temperature."""
    from ..lang.errors import ChaordError
    if T is not None:
        return T
    try:
        return float(dialect.threshold("md_reference_T"))
    except ChaordError:
        return None


def _lift_hcp_ortho(frame, dialect, backend="eam"):
    """Orthohexagonal hcp crystal arm (F2 root fix): fit the hexagonal
    lattice to an orthogonal box, emit the canonical crystal program.

    The fit (build/prototypes.py) measures a and c from the box lengths
    under the sqrt(3)-axis constraint and verifies them by site matching;
    this arm only assembles the program. The stated orientation is the
    orthohexagonal setting -- x [100] y [1 2 0] z [001] on the canonical
    axis order (a, sqrt(3) a, c) -- the only orientation whose lattice
    vectors tile the orthorhombic box (the identity orientation of the
    hexagonal cell does not, which is why the spglib arm cannot serve this
    frame class)."""
    from ..build.prototypes import fit_orthohexagonal_hcp
    from ..lang.errors import ChaordError
    from ..lang.ir import (
        GeoChain, Name, PhysicsBlock, Program, ProvenanceBlock, Quantity,
        RegionBlock, ResidualBlock, ShAll, Statement, StrVal, SystemBlock,
    )
    from .crystal import round_canonical
    from .defect_program import _axis_canonical

    # rule 3 (identical text under rigid rotation): the fit measures the box
    # through axis-aligned lengths, so a rotated orthogonal cell is rotated
    # back first -- exactly the convention the defect pass uses. Without it
    # a rotated hcp frame failed the fit, fell through the cascade and hit
    # the census guard (found when A2 switched to the stored thermal frames)
    frame = _axis_canonical(frame)
    fit = fit_orthohexagonal_hcp(frame, dialect)
    if fit is None:
        raise ChaordError(
            "no orthohexagonal hcp fit: the orthogonal cell does not tile "
            "as n x a, n x sqrt(3) a, n x c with matching site coverage")

    n_atoms: dict[str, int] = {}
    for s in frame.symbols:
        n_atoms[s] = n_atoms.get(s, 0) + 1
    conserve_values = []
    for s in sorted(n_atoms):
        conserve_values += [Name(text=s), Quantity(num=str(n_atoms[s]))]

    unit = dialect.threshold("length_unit")
    region_stmts = [
        Statement(kind="build", key="lattice", values=[Name(text="hcp")]),
        Statement(kind="build", key="a", values=[
            Quantity(num=round_canonical(fit.a, "canonical_a_decimals", dialect),
                     unit=unit if unit != "none" else None)]),
        Statement(kind="build", key="c", values=[
            Quantity(num=round_canonical(fit.c, "canonical_c_decimals", dialect),
                     unit=unit if unit != "none" else None)]),
        Statement(kind="build", key="orient", values=[
            Name(text="x"), Name(text="[100]"),
            Name(text="y"), Name(text="[1 2 0]"),
            Name(text="z"), Name(text="[001]")]),
        Statement(kind="assert", key="sites_matched",
                  values=[Quantity(num=f"{100.0 * fit.sites_matched:.1f}", unit="%")]),  # dialect-exempt: numerical-guard: canonical assert text
    ]
    system = SystemBlock(statements=[
        Statement(kind="build", key="cell", values=[
            Quantity(num=round_canonical(v, "canonical_cell_decimals", dialect))
            for v in fit.cell]),
        Statement(kind="build", key="pbc", values=[Name(text="xyz")]),
        Statement(kind="conserve", key="atoms", values=conserve_values),
    ])
    physics = PhysicsBlock(statements=[
        Statement(kind="build", key="backend", values=[Name(text=backend)]),
    ])
    region = RegionBlock(phase="crystal", name="bulk",
                         geometry=GeoChain(parts=[ShAll()], ops=[]),
                         statements=region_stmts)
    provenance = ProvenanceBlock(statements=[
        Statement(kind="build", key="dialects",
                  values=[StrVal(text=dialect.version_string)]),
        Statement(kind="build", key="lift_version",
                  values=[StrVal(text="0.1.0")]),
    ])
    return Program(version="0.1", dialects=list(dialect.names),  # dialect-exempt: numerical-guard: language version constant
                   blocks=[system, physics, region, ResidualBlock(none=True),
                           provenance])


def legacy_lift(frame, dialect, T=None, mode="auto"):
    """The routing ladder; returns (program, unexplained atom count).

    Every path explains all of its atoms except the M0 slab lifter, whose
    off-lattice atoms are either accounted as displaced by a defect complex or
    must appear in the program's residual block.

    W1 step 4 (Review 8): an arm falls through ONLY on NotThisPhase.  The
    arms predate the exception and signal refusals with ChaordError (plus
    crystal.py's bare-exception set, see _spglib_refusal), so each arm's
    failure is converted with ``raise _refusal(e)`` and only the converted
    NotThisPhase is caught.  A missing threshold (the one ChaordError that
    is a configuration bug, not a verdict) or any other error propagates --
    the cascade refuses instead of laundering the failure into whichever arm
    comes next."""
    from ..lang.errors import ChaordError
    if mode in ("auto", "crystal", "defects"):
        if mode in ("auto", "crystal"):
            try:
                try:
                    return _lift_hcp_ortho(frame, dialect), 0
                except Exception as e:
                    if mode == "crystal":
                        pass        # not hcp: try the exact arm, refuse loudly below
                    else:
                        raise _refusal(e)
            except NotThisPhase:
                pass
        try:
            try:
                return lift_crystal(frame, dialect), 0
            except Exception as e:
                if mode == "crystal":
                    # a pinned-arm refusal is a ChaordError, never a raw
                    # engine exception (F2 root-cause 1: the same escape was
                    # an A13-class defect on hcp_mg)
                    raise ChaordError(
                        f"crystal lift failed: {e}") from e
                raise _refusal(e)
        except NotThisPhase:
            pass
        if mode in ("auto", "defects"):
            try:
                try:
                    program, _diag = lift_crystal_defects(frame, dialect)
                    return program, 0
                except Exception as e:
                    if mode == "defects":
                        raise
                    raise _refusal(e)
            except NotThisPhase:
                pass
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
            return lift_fluid(frame, dialect, T=_default_T(dialect, T)), 0
        if mode == "fluid":
            raise ChaordError("frame is not a single-phase fluid "
                              "(solid-like fraction too high)")
    if mode in ("auto", "slab"):
        from .slab import decompile, program_from_result
        T = _default_T(dialect, T)  # T is metadata: dialects without MD defaults omit it
        try:
            try:
                res = decompile(frame.pos, frame.cell_diag, T, dialect,
                                symbols=frame.symbols)
                program = program_from_result(res, dialect)
                # atoms the slab lifter could not explain: off-lattice atoms
                # minus those accounted as displaced by defect complexes
                explained = sum(c["displaced"] for c in res["defects"])
                return program, max(len(res["off"]) - explained, 0)
            except Exception as e:
                if mode == "slab":
                    raise
                raise _refusal(e)
        except NotThisPhase:
            pass    # no interfaces found: a single-phase fluid after all
        return lift_fluid(frame, dialect, T=T), 0
    raise ValueError(f"unknown lift mode {mode!r}")
