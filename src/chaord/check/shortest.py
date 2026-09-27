"""Shortest-program controller: the shortest canonical text that round-trips.

Candidates are derived from the lifted program by dropping optional statement
kinds (asserts, provenance) and aggregating defects; each candidate is
re-built, re-lifted and compared against the noise floor. The shortest
passing program wins (M5 deliverable).
"""
from __future__ import annotations

import numpy as np

from ..cv.noise import observables, distance
from ..io.frames import Frame


def _variants(program):
    """Yield (name, program) candidates from most to least informative."""
    yield "full", program

    def filtered(pred):
        p = program.model_copy(deep=True)
        for b in p.blocks:
            if hasattr(b, "statements"):
                b.statements = [s for s in b.statements if pred(s)]
        return p

    yield "no-asserts", filtered(lambda s: s.kind != "assert")
    p = filtered(lambda s: s.kind != "assert")
    p.blocks = [b for b in p.blocks if b.t != "provenance"]
    yield "minimal", p


def shortest_program(frame: Frame, program, dialect, build_fn,
                     rng_seed: int = 0, factor: float = 1.5,
                     reference_later: Frame | None = None):
    """Pick the shortest program whose rebuild reproduces the observables.

    The acceptance test is the statistical round trip: rebuilt observables
    within `factor` x the noise floor (measured against `reference_later`
    when given, otherwise a thermal continuation is used if provided by the
    caller; without a reference the exact statement set is kept)."""
    oo = observables(frame, dialect)
    best = None
    for name, candidate in _variants(program):
        from ..lang.fmt import format_program
        text = format_program(candidate)
        rng = np.random.default_rng(rng_seed)
        try:
            rebuilt = build_fn(candidate, rng)
        except Exception:
            continue
        if len(rebuilt) != len(frame):
            continue  # never drop an atom
        o_re = observables(rebuilt, dialect)
        d = distance(oo, o_re)
        if reference_later is not None:
            ol = observables(reference_later, dialect)
            floor = distance(oo, ol)
            ok = all(d[k] <= factor * max(floor[k], 1e-6) for k in d)  # dialect-exempt: degenerate floor
        else:
            # no reference frames: accept the most detailed program only
            ok = name == "full"
        if ok and (best is None or len(text) < len(best[1])):
            best = (name, text, d)
    if best is None:
        from ..lang.fmt import format_program
        return "full", format_program(program), None
    return best
