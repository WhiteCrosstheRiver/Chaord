"""Amorphous builder: density + history protocol -> quenched frame."""
from __future__ import annotations

import numpy as np

from ..io.frames import Frame
from ..lang.errors import ChaordError
from ..lang.ir import Program
from ..realize.protocols import parse_history, run_protocol


def _num(v) -> float:
    s = v.num if hasattr(v, "num") else str(v)
    if "/" in s:
        a, b = s.split("/")
        return float(a) / float(b)
    return float(s)


def build_amorphous(program: Program, dialect, rng, physics=True,
                    md_steps=None) -> Frame:
    regions = [b for b in program.blocks if b.t == "region"]
    if len(regions) != 1 or regions[0].phase != "amorphous":
        raise ChaordError("amorphous builder expects one amorphous region")
    region = regions[0]
    system = {}
    for b in program.blocks:
        if b.t == "system":
            system = {s.key: s for s in b.statements
                      if s.kind in ("build", "state", "conserve")}

    conserve = system.get("atoms")
    if conserve is None:
        raise ChaordError("amorphous program needs conserve atoms")
    species = "X"
    n = None
    vals = conserve.values
    if vals and vals[0].t == "n":
        species = vals[0].text
        if len(vals) > 1 and vals[1].t == "q":
            n = int(_num(vals[1]))

    dens = next((s for s in region.statements
                 if s.kind == "state" and s.key == "density"), None)
    if dens is None or n is None:
        raise ChaordError("amorphous build needs state density and conserve atoms")

    rho = _num(dens.values[0])
    if dens.values[0].unit == "g/cm3":
        from ase.data import atomic_masses, chemical_symbols
        from ..build.defects import typical_neighbor_distance  # noqa: F401
        mass = atomic_masses[chemical_symbols.index(species)] if species in chemical_symbols else 1.0  # dialect-exempt: numerical-guard: unit-mass fallback for placeholder species
        rho_number = rho / mass * 0.6022140857  # atoms/A^3  # dialect-exempt: exact-geometry
    else:
        rho_number = rho
    L = float((n / rho_number) ** (1 / 3))
    # start overlap-free: random sequential addition, then the protocol runs
    from .slab import rsa
    dmin = float(dialect.threshold("fluid_rsa_dmin"))
    rng_start = np.random.default_rng(rng.integers(1 << 31))
    pos0 = rsa(np.zeros((0, 3)), np.array([L] * 3), L / 2, L / 2, n, dmin, rng_start)

    # a history statement's key is its first command word (melt/quench/...),
    # so the statement is recognised by kind, not key
    history = [s for s in region.statements if s.kind == "history"]
    frame = Frame(pos=pos0, cell=np.diag([L] * 3), symbols=[species] * n,
                  pbc=(True, True, True))
    if not physics or not history:
        return frame
    steps = parse_history(history[0], dialect)
    return run_protocol(frame, steps, dialect, rng)
