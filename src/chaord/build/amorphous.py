"""Amorphous builder: density + history protocol -> quenched frame."""
from __future__ import annotations

import sys

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
    if not physics:
        return frame
    assumed = False
    if history:
        steps = parse_history(history[0], dialect)
    else:
        # Review 2: no silent skip to the random packing. A program without a
        # history line builds with the dialect's default melt-quench -- the
        # same glass_* thresholds the amorphous lifter states -- and the
        # assumption is recorded on the frame and printed to stderr.
        steps = _default_protocol(dialect)
        assumed = True
    out = run_protocol(frame, steps, dialect, rng)
    if assumed:
        out.info["assumed_history"] = "assumed default protocol from dialect"
        print("chaord: amorphous program states no history: assumed default "
              "protocol from dialect", file=sys.stderr)
    return out


def _default_protocol(dialect) -> list:
    """The dialect's default melt -> quench -> anneal, as the amorphous lifter
    states it (glass_melt/quench/anneal thresholds, LJ reduced units; the
    quench rate is linear over glass_quench_steps)."""
    t_melt = float(dialect.threshold("glass_melt_T"))
    n_melt = int(dialect.threshold("glass_melt_steps"))
    t_q = float(dialect.threshold("glass_quench_T"))
    n_q = int(dialect.threshold("glass_quench_steps"))
    t_a = float(dialect.threshold("glass_anneal_T"))
    n_a = int(dialect.threshold("glass_anneal_steps"))
    rate = round((t_melt - t_q) / n_q, 6)  # dialect-exempt: numerical-guard: printable cooling rate, T per step
    # `quench to T at rate` parses to (T, T, rate): run_protocol quenches from
    # wherever the protocol currently is (after the melt)
    return [("melt", t_melt, n_melt), ("quench", t_q, t_q, rate),
            ("anneal", t_a, n_a)]
