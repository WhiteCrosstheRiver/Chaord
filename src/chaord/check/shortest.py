"""Shortest-program controller: the shortest canonical text that round-trips.

A greedy deletion search over removable statement/block categories, in priority
order (most redundant information first):

1. ``assert`` statements        (checks; the build never reads them)
2. ``provenance`` blocks        (lift bookkeeping)
3. ``orient`` statements        (only when the transformation is the identity)
4. ``interface`` blocks         (region coupling hints)
5. ``residual none`` lines      (the explicit empty-residual declaration)

Each candidate deletion is verified the same way: rebuild (``build_fn``), keep
the atom count (conservation is never negotiable) and compare the fluid
observables against the measured noise floor.  A deletion that stays within
`factor` x the floor is kept; anything else is rolled back.  ``conserve``
statements are never deleted, and a residual block that explains atoms is never
deleted either (rule 6: never drop an atom).

Without `reference_later` there is no measured floor, so no deletion can be
certified: the search degenerates to the conservative mode, which only
considers the old categories (asserts, provenance) and accepts none of them —
byte-for-byte the behaviour of the previous three-variant controller
(full / no-asserts / minimal with ``ok = name == "full"``).
"""
from __future__ import annotations

import numpy as np

from ..cv.noise import observables, distance
from ..io.frames import Frame

# deletion priority order (see module docstring); conservative mode is the
# old category set and always rolls back for lack of a noise floor
_SEARCH_ORDER = ("assert", "provenance", "orient", "interface", "residual")
_CONSERVATIVE_ORDER = ("assert", "provenance")

# keep the historic names for the historic deletion combinations
_LEGACY_NAMES = {
    (): "full",
    ("assert",): "no-asserts",
    ("assert", "provenance"): "minimal",
}


def _statements_of(block):
    return getattr(block, "statements", None)


def _has_conserve(block) -> bool:
    return any(s.kind == "conserve" for s in _statements_of(block) or [])


def _drop_statements(program, pred):
    """Deep copy with every statement matching `pred` removed; None if none."""
    q = program.model_copy(deep=True)
    removed = 0
    for b in q.blocks:
        stmts = _statements_of(b)
        if stmts is None:
            continue
        keep = [s for s in stmts if not pred(s)]
        removed += len(stmts) - len(keep)
        b.statements = keep
    return q if removed else None


def _drop_blocks(program, pred):
    """Deep copy with every block matching `pred` removed; None if none.

    A block holding a ``conserve`` statement is never removed."""
    q = program.model_copy(deep=True)
    kept, removed = [], 0
    for b in q.blocks:
        if pred(b) and not _has_conserve(b):
            removed += 1
            continue
        kept.append(b)
    if not removed:
        return None
    q.blocks = kept
    return q


def _is_identity_orient(s) -> bool:
    """True for `orient x [100] y [010] z [001]` (the identity transform)."""
    if s.kind != "build" or s.key != "orient":
        return False
    from ..build.crystal import orient_matrix
    try:
        T = orient_matrix(s)
    except Exception:
        return False  # unparseable or singular: never a deletion candidate
    return T.shape == (3, 3) and bool(np.array_equal(T, np.eye(3, dtype=int)))


def _drop_asserts(p):
    return _drop_statements(p, lambda s: s.kind == "assert")


def _drop_provenance(p):
    return _drop_blocks(p, lambda b: b.t == "provenance")


def _drop_identity_orients(p):
    return _drop_statements(p, _is_identity_orient)


def _drop_interfaces(p):
    return _drop_blocks(p, lambda b: b.t == "interface")


def _drop_residual_none(p):
    # only the explicit `residual none` marker; a residual that explains
    # atoms carries conservation information and survives
    return _drop_blocks(p, lambda b: b.t == "residual" and b.none
                        and not (b.statements or []))


_DELETERS = {
    "assert": _drop_asserts,
    "provenance": _drop_provenance,
    "orient": _drop_identity_orients,
    "interface": _drop_interfaces,
    "residual": _drop_residual_none,
}


def shortest_program(frame: Frame, program, dialect, build_fn,
                     rng_seed: int = 0, factor: float = 1.5,
                     reference_later: Frame | None = None):
    """Pick the shortest program whose rebuild reproduces the observables.

    The acceptance test is the statistical round trip: rebuilt observables
    within `factor` x the noise floor, measured against `reference_later`
    (an independent frame of the same simulation).  Candidate deletions are
    tried one category at a time, cheapest first; each is rebuilt and kept
    only if it still passes, otherwise it is rolled back.

    Without `reference_later` the controller falls back to the conservative
    mode of the previous implementation: only the exact statement set is
    certified, so nothing is deleted.

    Returns ``(name, text, removals)``: the canonical name of the surviving
    variant ("full" when nothing was deleted; the historic "no-asserts" /
    "minimal" names are kept for the historic combinations), its canonical
    text, and the ordered list of deleted categories.  (The third element
    carried the observable-distance dict in earlier revisions; no caller
    read it.)"""
    from ..lang.fmt import format_program

    oo = observables(frame, dialect)
    floor = (distance(oo, observables(reference_later, dialect))
             if reference_later is not None else None)

    def verified(candidate) -> bool:
        """Rebuild, conserve the atom count, compare within the floor."""
        rng = np.random.default_rng(rng_seed)
        try:
            rebuilt = build_fn(candidate, rng)
        except Exception:
            return False
        if len(rebuilt) != len(frame):
            return False  # never drop an atom
        d = distance(oo, observables(rebuilt, dialect))
        return all(d[k] <= factor * max(floor[k], 1e-6) for k in d)  # dialect-exempt: degenerate floor

    order = _SEARCH_ORDER if floor is not None else _CONSERVATIVE_ORDER
    current = program
    applied: list[str] = []
    for category in order:
        candidate = _DELETERS[category](current)
        if candidate is None:
            continue  # nothing of this kind to delete
        if floor is None:
            # conservative mode: no measured floor, no certified deletion —
            # the exact statement set stands (old behaviour)
            continue
        if verified(candidate):
            current = candidate
            applied.append(category)
        # else: rolled back, `current` untouched

    name = _LEGACY_NAMES.get(tuple(applied)) \
        or "+".join(f"no-{c}" for c in applied) or "full"
    return name, format_program(current), applied
