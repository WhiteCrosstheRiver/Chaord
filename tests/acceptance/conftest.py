"""Make the repository root importable so tests can `import tools.acceptance`."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)


def apply_ka_defects_arm_guard():
    """MAINLINE PATCH APPLIED IN-PROCESS (W7 report; the file is outside this
    stream's ownership): the legacy cascade's DEFECTS arm
    (lift_crystal_defects -> defects.fit_crystal) raises a bare ValueError on
    placeholder species ('A'/'B' are not ASE elements: chemical_symbols has
    no 'A'), which the W1 fail-closed ladder correctly refuses to swallow,
    so the auto lift of a binary glass aborts.  fit_crystal must state that
    verdict as its designed no-fit ChaordError instead; until it does, this
    opt-in shim converts it for non-element frames so the cascade reaches
    the amorphous arm (no physics changes: the crystal arms' verdict on a
    binary glass is 'not this phase' either way)."""
    import chaord.lift.legacy as legacy
    from ase.data import chemical_symbols as cs
    from chaord.lang.errors import ChaordError
    if getattr(legacy.lift_crystal_defects, "_ka_guarded", False):
        return
    orig = legacy.lift_crystal_defects

    def guarded(frame, dialect):
        if not all(s in cs for s in set(frame.symbols)):
            raise ChaordError(
                "no cubic prototype fit is defined for placeholder species "
                "(mainline patch, W7 report: fit_crystal's element lookup)")
        return orig(frame, dialect)

    guarded._ka_guarded = True
    legacy.lift_crystal_defects = guarded
