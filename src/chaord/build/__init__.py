"""Compiler (chaord build): program -> coordinates.

build_program dispatches on the region phases: a single crystal region goes to
the exact crystal builder; crystal + liquid slabs go to the M0 slab builder.
"""
from chaord.build.crystal import (  # noqa: F401
    SLOT_COUNTS, build_conventional, build_crystal_region, orient_matrix,
    parse_formula, reduced_counts, slots_from_formula,
)
from chaord.build.fluid import build_fluid  # noqa: F401
from chaord.build.surface import build_surface  # noqa: F401
from chaord.build.slab import build_slab  # noqa: F401
from ..io.frames import Frame
from ..lang.errors import ChaordError
from ..lang.ir import Program


def _system_map(program: Program) -> dict:
    for b in program.blocks:
        if b.t == "system":
            return {s.key: s for s in b.statements if s.kind in ("build", "state", "conserve")}
    return {}


def build_program(program: Program, dialect, rng=None, physics: bool = True,
                  md_steps=None) -> Frame:
    regions = [b for b in program.blocks if b.t == "region"]
    if not regions:
        raise ChaordError("program has no region to build")
    phases = {r.phase for r in regions}
    if phases == {"crystal"}:
        if len(regions) != 1:
            raise ChaordError("the M1 builder supports one crystal region")
        return build_crystal_region(regions[0], _system_map(program), dialect, rng=rng)
    if phases in ({"liquid"}, {"gas"}, {"liquid", "gas"}):
        return build_fluid(program, dialect, rng, physics=physics, md_steps=md_steps)
    if phases == {"amorphous"}:
        from .amorphous import build_amorphous
        return build_amorphous(program, dialect, rng, physics=physics, md_steps=md_steps)
    if phases == {"crystal", "vacuum"} or (
            phases == {"crystal"} and any(
                s.key == "surface" for b in regions for s in b.statements)):
        import numpy as _np
        from .surface import build_surface
        return build_surface(regions[0] if phases == {"crystal"} else
                             next(r for r in regions if r.phase == "crystal"),
                             _system_map(program), dialect,
                             rng or _np.random.default_rng(0))
    if phases == {"crystal", "liquid"}:
        import numpy as np
        rng = rng or np.random.default_rng(0)
        return build_slab(program, dialect, rng, physics=physics, md_steps=md_steps)
    raise ChaordError(f"no builder yet for region phases {sorted(phases)}")
