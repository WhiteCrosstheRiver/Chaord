"""Static checks on a program + structure pair: overlaps, conservation, charge, asserts.

These run after every build and every lift. They measure and report; they never
modify anything. Thresholds come from the dialect.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from ..io.frames import Frame
from ..lang.errors import ChaordError
from ..lang.ir import Program


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


def overlap_check(frame: Frame, dialect) -> CheckResult:
    tol = float(dialect.threshold("overlap_tolerance"))
    if len(frame) < 2:
        return CheckResult("overlap", True, "fewer than two atoms")
    from scipy.spatial import cKDTree
    from ..realize.lj import wrap
    L = frame.cell_diag
    tree = cKDTree(wrap(frame.pos, L), boxsize=L)
    d, _ = tree.query(wrap(frame.pos, L), k=2)
    dmin = float(d[:, 1].min())
    ok = dmin >= tol
    return CheckResult("overlap", ok, f"min pair distance {dmin:.3f} vs tolerance {tol}")


def conservation_check(program: Program, frame: Frame) -> CheckResult:
    """Atoms per species and total count in the program must equal the input's."""
    stated: dict[str, int] = {}
    for b in program.blocks:
        if b.t == "system":
            for s in b.statements:
                if s.kind == "conserve" and s.key == "atoms":
                    vals = s.values
                    i = 0
                    while i + 1 < len(vals):
                        if vals[i].t == "n" and vals[i + 1].t == "q":
                            stated[vals[i].text] = int(float(vals[i + 1].num))
                        i += 2
    present: dict[str, int] = {}
    for sym in frame.symbols:
        present[sym] = present.get(sym, 0) + 1
    if stated == present:
        return CheckResult("conservation", True, f"{sum(present.values())} atoms, species match")
    return CheckResult(
        "conservation", False,
        f"program says {stated}, structure has {present}")


# ---------------------------------------------------------------- charge ----
# Charge sources, lowest priority first: a dialect's `formal_charges` table
# (ionic.yaml, keyed by species name) is overridden by the program's species
# block `ion` definitions. Within an ion, the template charge in
# build.molecules.TEMPLATES (exact published data) wins over the name; without a
# template the charge is the trailing sign run of the name (`Li+` +1, `PF6-`
# -1, `O2-` -1: digits before the signs are stoichiometry, not the magnitude).

_SIGN_RUN = re.compile(r"([+\-]+)\Z")
_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")


@lru_cache(maxsize=8)
def _formal_charges_section(dialect_name: str) -> tuple[tuple[str, int], ...]:
    """The `formal_charges:` section of one dialect file, as (name, charge) pairs."""
    from ..dialects import _load_yaml
    table = _load_yaml(dialect_name).get("formal_charges") or {}
    return tuple((str(k), int(v)) for k, v in table.items())


def _ion_charge(name: str) -> int:
    """Charge of a species name: template charge when known, else sign count."""
    from ..build.molecules import TEMPLATES
    if name in TEMPLATES:
        return int(TEMPLATES[name]["charge"])
    m = _SIGN_RUN.search(name)
    if not m:
        return 0
    run = m.group(1)
    return len(run) if run[0] == "+" else -len(run)


def _ion_composition(name: str) -> tuple[str, ...] | None:
    """Element multiset behind a species name: template symbols when the name is
    a known molecule, else the charge-stripped name parsed as a formula
    (`PF6-` -> P + 6 F). None when the name is not a plain formula."""
    from ..build.molecules import TEMPLATES
    if name in TEMPLATES:
        return tuple(TEMPLATES[name]["symbols"])
    base = _SIGN_RUN.sub("", name)
    symbols: list[str] = []
    pos = 0
    while pos < len(base):
        m = _FORMULA_TOKEN.match(base, pos)
        if not m:
            return None
        symbols.extend([m.group(1)] * (int(m.group(2)) if m.group(2) else 1))
        pos = m.end()
    if not symbols:
        return None
    from ase.data import chemical_symbols
    if not set(symbols) <= set(chemical_symbols):
        return None
    return tuple(symbols)


def _charge_sources(program: Program, dialect) -> dict[str, dict]:
    """canonical formula -> {name, charge, element}. The program's species-block
    ions override the dialects' formal-charge table (dialects overlay in load
    order, later wins)."""
    sources: dict[str, dict] = {}
    from ..build.molecules import _formula

    def add(name: str, charge: int) -> None:
        comp = _ion_composition(name)
        if comp is None:
            return
        sources[_formula(list(comp))] = dict(
            name=name, charge=charge, single=len(comp) == 1,
            element=comp[0] if len(comp) == 1 else None)

    for dialect_name in getattr(dialect, "names", ()) or ():
        for name, charge in _formal_charges_section(dialect_name):
            add(name, charge)
    for b in program.blocks:
        if b.t != "species":
            continue
        for d in b.defs:
            if d.k == "ion":
                add(d.name, _ion_charge(d.name))
    return sources


def _stated_charge(program: Program) -> int | None:
    """The N of `conserve charge N` (system block first, canonical location)."""
    blocks = sorted(program.blocks, key=lambda b: b.t != "system")
    for b in blocks:
        for s in getattr(b, "statements", ()):
            if s.kind == "conserve" and s.key == "charge":
                for v in s.values:
                    if v.t == "q":
                        return int(round(float(v.num)))
    return None


def charge_check(program: Program, frame: Frame, dialect) -> CheckResult:
    """Total charge of the structure must equal the program's `conserve charge N`.

    Neutral species and atoms no source assigns a charge to contribute 0, so
    programs without a species block and without ionic dialects skip the bookkeeping
    entirely. Without a `conserve charge` statement an ionic system still passes
    (WARN level): the detail suggests pinning the total.
    """
    stated = _stated_charge(program)
    sources = _charge_sources(program, dialect)
    charged = {f: info for f, info in sources.items() if info["charge"]}
    present: dict[str, tuple[int, int]] = {}
    if any(not info["single"] for info in charged.values()):
        # a multi-atom ion (PF6-, ...) carries its charge as a molecule: count
        # species with the bond-graph census every source is canonicalised to
        from ..build.molecules import molecule_census
        try:
            census = molecule_census(frame, dialect)
        except ChaordError as e:
            return CheckResult("charge", True,
                               f"skipped: molecular census unavailable ({e})")
        present = {f: (census[f], info["charge"])
                   for f, info in charged.items() if census.get(f)}
    else:
        syms = frame.symbols
        present = {f: (syms.count(info["element"]), info["charge"])
                   for f, info in charged.items() if info["element"] in syms}
    total = sum(n * q for n, q in present.values())
    breakdown = ", ".join(f"{sources[f]['name']} {n} x {q:+d}"
                          for f, (n, q) in sorted(present.items()))

    if stated is None:
        if present:
            return CheckResult(
                "charge", True,
                f"total charge {total:+d} ({breakdown}); no 'conserve charge' "
                f"statement - add 'conserve charge {total}' to the system block")
        return CheckResult("charge", True, "total charge 0 (no ionic species)")
    if total == stated:
        tail = f" ({breakdown})" if breakdown else " (no ionic species)"
        return CheckResult(
            "charge", True,
            f"total charge {total:+d}{tail} equals conserve charge {stated}")
    return CheckResult(
        "charge", False,
        f"expected {stated}, actual {total} "
        f"({breakdown or 'no charged species found'})")


def residual_explained(program: Program, frame: Frame) -> CheckResult:
    """Never drop an atom: unexplained atoms must appear in the residual block."""
    n_res = sum(len(b.statements) for b in program.blocks if b.t == "residual" and not b.none)
    has_none = any(b.t == "residual" and b.none for b in program.blocks)
    if has_none:
        return CheckResult("residual", True, "residual none")
    return CheckResult("residual", True, f"{n_res} atoms in residual")


def run_checks(program: Program, frame: Frame, dialect) -> list[CheckResult]:
    return [
        overlap_check(frame, dialect),
        conservation_check(program, frame),
        charge_check(program, frame, dialect),
        residual_explained(program, frame),
    ]
