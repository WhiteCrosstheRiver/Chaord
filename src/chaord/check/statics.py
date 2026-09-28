"""Static checks on a program + structure pair: overlaps, conservation, charge, asserts.

These run after every build and every lift. They measure and report; they never
modify anything. Thresholds come from the dialect.
"""
from __future__ import annotations

import re
import weakref
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


def conservation_check(program: Program, frame: Frame, dialect=None) -> CheckResult:
    """Three-way atom census: the frame, the program text, the conserve line.

    1. frame: count frame.symbols directly (the ground truth microstate);
    2. implied: what the program's own build/state statements construct
       (implied_atom_count: lattice sites x cell, compositions, occupancies,
       defect net effects, molecules lines, adsorb counts, residual atoms);
    3. declared: the system's `conserve atoms` statement.

    All three must agree. Regions whose atom count is only density-derived
    (amorphous composition+density, atomic liquids with `state density`,
    ASE-cut Miller slabs, the M0 strain/depth slab grid) cannot be pinned from
    text: for those the conserve line stands as the declared value, and the
    check only requires the remainder to be explainable - non-negative per
    species, and attributable to a declared region whose stated composition
    names that species. A program with no such region must be implied exactly.
    """
    stated = _stated_atoms(program)
    present: dict[str, int] = {}
    for sym in frame.symbols:
        present[sym] = present.get(sym, 0) + 1
    implied, buckets, anonymous = _implied_inventory(program)
    if anonymous:
        # unary lattice region: the language names its single species only in
        # `conserve atoms` (see build.crystal), so attribute the site count
        single = _single_conserve_species(program)
        if single is not None:
            implied[single] = implied.get(single, 0) + anonymous
            anonymous = 0
    wildcard = anonymous > 0 or any(b is None for b in buckets)
    vocab: set[str] = set()
    for b in buckets:
        if b is not None:
            vocab |= b

    problems = []
    if stated != present:
        problems.append(f"conserve says {_census(stated)}, frame has {_census(present)}")
    for s in sorted(set(stated) | set(implied)):
        left = stated.get(s, 0) - implied.get(s, 0)
        if left < 0:
            problems.append(f"{s}: statements imply {implied[s]}, "
                            f"conserve states {stated.get(s, 0)}")
        elif left > 0 and not wildcard and s not in vocab:
            problems.append(f"{s}: {left} conserve atoms are explained by no "
                            f"region statement")
    declared = {s: stated[s] - implied.get(s, 0)
                for s in stated if stated[s] - implied.get(s, 0) > 0}
    numbers = (f"frame {_census(present)}, implied {_census(implied)}"
               + (f", declared-only {_census(declared)}" if declared else "")
               + (", declared-only (density regions)" if wildcard and not declared else "")
               + f", conserve {_census(stated)}")
    if not problems:
        return CheckResult("conservation", True,
                           f"{sum(present.values())} atoms: {numbers}")
    return CheckResult("conservation", False,
                       f"{'; '.join(problems)} [{numbers}]")


def _census(d: dict[str, int]) -> str:
    return "{" + ", ".join(f"{k}: {d[k]}" for k in sorted(d)) + "}"


def _stated_atoms(program: Program) -> dict[str, int]:
    """Per-species counts of the system's `conserve atoms` statement(s)."""
    stated: dict[str, int] = {}
    for b in program.blocks:
        if b.t == "system":
            for s in b.statements:
                if s.kind == "conserve" and s.key == "atoms":
                    vals = s.values
                    i = 0
                    while i + 1 < len(vals):
                        if vals[i].t == "n" and vals[i + 1].t == "q":
                            stated[vals[i].text] = int(round(float(_num(vals[i + 1]))))
                        i += 2
    return stated


def _single_conserve_species(program: Program) -> str | None:
    stated = _stated_atoms(program)
    return next(iter(stated)) if len(stated) == 1 else None


# ------------------------------------------------------------ implied atoms ----
# Atom counts implied by the program text alone, independently of the frame it
# was lifted from (the fix for the circular conservation check: the conserve
# line used to be compared against the very frame it was written from).
#
# Statements that pin counts exactly:
#   * crystal regions: prototype/lattice parameters + the system cell fix the
#     supercell site count; `composition` distributes it over slots,
#     `occupancy` over species fractions (x site count), and `defect`
#     statements shift it (V_X -1, A_B -1/+1, A_i +1, frenkel_pair 0);
#   * `molecules` lines: species composition (template table, else the name
#     parsed as an element formula) x the stated molecule count;
#   * `adsorb` lines carrying a `count` token;
#   * residual `atom` lines: one atom each.
#
# Regions whose count is only density-derived become "declared" buckets: the
# amorphous builder and the atomic-fluid builder take species and count from
# the conserve line by design, and ASE-cut slabs / the M0 strain grid are not
# recoverable from rounded text. Each bucket records the species its
# statements name (composition, termination); None marks a wildcard bucket
# (unary lattice regions, atomic fluids: the language defers to conserve).


def _num(v) -> float:
    """Numeric value of a Quantity-like token ('27/32' fractions supported)."""
    s = v.num if hasattr(v, "num") else str(v)
    if "/" in s:
        a, b = s.split("/")
        return float(a) / float(b)
    return float(s)


def _region_stmt_map(region) -> dict:
    return {s.key: s for s in region.statements
            if s.kind in ("build", "state", "constrain")}


def _system_cell(program: Program) -> list[float] | None:
    for b in program.blocks:
        if b.t != "system":
            continue
        for s in b.statements:
            if s.key == "cell":
                vals = [_num(v) for v in s.values if v.t == "q"]
                if len(vals) == 3:
                    return vals
    return None


def _molecule_elements(name: str) -> tuple[str, ...] | None:
    """Element multiset behind a molecules-line species name, or None."""
    return _ion_composition(name)


def _has_token(stmt, text: str) -> bool:
    return any(v.t == "n" and v.text == text for v in stmt.values)


def _defect_net_effect(stmt) -> dict[str, int] | None:
    """Per-species atom delta of a Kröger–Vink `defect` statement, or None
    when the statement is M0-style (named clusters with `depth` tokens whose
    site bookkeeping lives in the slab grid)."""
    vals = stmt.values
    if not vals or vals[0].t != "kv" or _has_token(stmt, "depth"):
        return None
    token = vals[0].text
    count = 1
    for i, v in enumerate(vals):
        if v.t == "n" and v.text == "count" and i + 1 < len(vals):
            count = int(_num(vals[i + 1]))
    if token == "frenkel_pair":
        return {}                       # one vacancy + one interstitial: net 0
    from ..build.defects import parse_kv
    site, sub = parse_kv(token)         # raises on a non-KV token
    if site == "V":
        return {sub: -count}
    if sub == "i":
        return {site: +count}
    return {sub: -count, site: +count}  # antisite A on B


def _supercell_reps(name: str, params: dict, stmts: dict, cell_vals):
    """Integer repetitions of the oriented conventional cell along the system
    cell axes (the same arithmetic build.crystal uses before tiling)."""
    if not cell_vals:
        return (1, 1, 1)
    from ..build.prototypes import cell_matrix
    T = np.eye(3, dtype=int)
    if "orient" in stmts:
        from ..build.crystal import orient_matrix
        T = orient_matrix(stmts["orient"])
    conv = T @ cell_matrix(name, params)
    lengths = np.linalg.norm(conv, axis=1)
    return tuple(max(int(round(c / l)), 1) for c, l in zip(cell_vals, lengths))


def _composition_species(stmt) -> set[str]:
    """Elements named by a composition statement (formula or pairs)."""
    from ..build.crystal import parse_formula
    out: set[str] = set()
    for v in stmt.values:
        if v.t == "n":
            try:
                out |= {el for el, _n in parse_formula(v.text)}
            except ChaordError:
                continue
    return out


def _implied_inventory(program: Program):
    """(exact per-species counts, declared buckets, anonymous site count).

    buckets is a list of species sets (None = wildcard); anonymous counts
    unary-lattice sites whose species the region does not name."""
    from ..build.crystal import SLOT_COUNTS, slots_from_formula
    from ..build.prototypes import PROTOTYPES

    exact: dict[str, int] = {}
    buckets: list[set[str] | None] = []
    anonymous = 0
    cell_vals = _system_cell(program)

    def add(delta: dict[str, int]) -> None:
        for s, n in delta.items():
            exact[s] = exact.get(s, 0) + n

    for b in program.blocks:
        if b.t == "residual":
            if not b.none:
                for s in b.statements:
                    if s.key == "atom" and s.values and s.values[0].t == "n":
                        add({s.values[0].text: 1})
            continue
        if b.t != "region" or b.phase == "vacuum":
            continue

        stmts = _region_stmt_map(b)
        region_exact: dict[str, int] = {}
        underivable = False     # some statements pin no exact count
        pinned = False          # this region pinned atoms exactly somehow

        # molecules lines: exact atom counts from the species composition
        for s in b.statements:
            if s.kind == "build" and s.key == "molecules":
                vals = s.values
                i = 0
                while i + 1 < len(vals):
                    if vals[i].t == "n":
                        name = vals[i].text
                        n = int(_num(vals[i + 1])) if vals[i + 1].t == "q" else 1
                        comp = _molecule_elements(name)
                        if comp is None:
                            underivable = True   # unknown species: not countable
                        else:
                            pinned = True
                            for el in comp:
                                region_exact[el] = region_exact.get(el, 0) + n
                    i += 1

        # adsorb lines: exact when a `count` token is present
        for s in b.statements:
            if s.kind == "build" and s.key == "adsorb" and s.values:
                species = s.values[0].text
                count = None
                for i, v in enumerate(s.values):
                    if v.t == "n" and v.text == "count" and i + 1 < len(s.values):
                        count = int(_num(s.values[i + 1]))
                if count is None:
                    underivable = True    # only a coverage: sites unknown
                else:
                    pinned = True
                    region_exact[species] = region_exact.get(species, 0) + count

        # the crystal body: lattice parameters x supercell = site count
        if b.phase == "crystal":
            name = None
            if "prototype" in stmts:
                name = stmts["prototype"].values[0].text
            elif "lattice" in stmts:
                name = stmts["lattice"].values[0].text
            m0_style = ("strain" in stmts
                        or any(s.key == "defect" and _has_token(s, "depth")
                               for s in b.statements))
            whole_cell = (len(b.geometry.parts) == 1
                          and b.geometry.parts[0].t == "all")
            if (name is None or name not in PROTOTYPES or m0_style
                    or "surface" in stmts or not whole_cell):
                # no lattice statement, the M0 strain/depth slab grid, an
                # ASE-cut Miller slab, or a partial-slab geometry (the M0
                # parity grid covers only its z window): the site count is
                # not text-derivable
                underivable = True
            else:
                params = {k: _num(stmts[k].values[0])
                          for k in PROTOTYPES[name].params if k in stmts}
                if set(params) != set(PROTOTYPES[name].params):
                    underivable = True
                else:
                    reps = _supercell_reps(name, params, stmts, cell_vals)
                    n_cells = int(np.prod(reps))
                    slot_counts = SLOT_COUNTS[name]
                    sites = n_cells * sum(slot_counts)
                    if "composition" in stmts:
                        try:
                            slot_species = slots_from_formula(
                                name, stmts["composition"].values[0].text)
                        except ChaordError:
                            underivable = True
                            slot_species = ()
                        for sp, c in zip(slot_species, slot_counts):
                            region_exact[sp] = region_exact.get(sp, 0) + n_cells * c
                        pinned = True
                    elif any(s.key == "occupancy" for s in b.statements):
                        # fraction x site count, replicating the builder's
                        # rounding (round per species, last species fills)
                        pairs: list[tuple[str, float]] = []
                        for s in b.statements:
                            if s.key != "occupancy":
                                continue
                            vals = s.values
                            i = 0
                            while i + 1 < len(vals):
                                if vals[i].t == "n":
                                    pairs.append((vals[i].text, _num(vals[i + 1])))
                                i += 1
                        if not pairs:
                            underivable = True
                        else:
                            total = sum(f for _s, f in pairs) or 1.0
                            counts = [int(round(f / total * sites))
                                      for _s, f in pairs]
                            counts[-1] = sites - sum(counts[:-1])
                            for (sp, _f), c in zip(pairs, counts):
                                region_exact[sp] = region_exact.get(sp, 0) + c
                            pinned = True
                    elif len(slot_counts) == 1:
                        anonymous += sites    # species named only in conserve
                        pinned = True
                    else:
                        underivable = True   # multi-slot prototype, no composition
                    # Kröger–Vink defect net effects (perfect sites -> actual)
                    if not underivable:
                        for s in b.statements:
                            if s.kind != "build" or s.key != "defect":
                                continue
                            try:
                                delta = _defect_net_effect(s)
                            except ChaordError:
                                delta = None
                            if delta is None:
                                underivable = True
                                break
                            for sp, n in delta.items():
                                region_exact[sp] = region_exact.get(sp, 0) + n

        add(region_exact)
        if underivable or not pinned:
            # density-driven / cut / strained region: the conserve line stands
            # for its atoms; record which species its statements name
            vocab: set[str] = set()
            if "composition" in stmts:
                vocab |= _composition_species(stmts["composition"])
            if "termination" in stmts and stmts["termination"].values:
                m = re.search(r"([A-Z][a-z]?)", stmts["termination"].values[0].text)
                if m:
                    vocab.add(m.group(1))
            buckets.append(vocab if vocab else None)

    return exact, buckets, anonymous


def implied_atom_count(program: Program) -> dict[str, int]:
    """Per-species atom counts the program's own statements construct.

    Exactly derivable statements only (crystal sites x cell, compositions,
    occupancies, defect net effects, molecules, adsorb counts, residual
    atoms). Density-only regions (amorphous, atomic liquids, ASE-cut slabs,
    the M0 strain grid) contribute nothing here: by design those builders
    take their counts from the `conserve atoms` line, which
    conservation_check then audits separately. Unary lattice regions name
    their single species only in `conserve atoms` (language rule), so the
    site count is attributed to that species."""
    exact, _buckets, anonymous = _implied_inventory(program)
    if anonymous:
        single = _single_conserve_species(program)
        if single is not None:
            exact[single] = exact.get(single, 0) + anonymous
    return {s: n for s, n in exact.items() if n != 0}


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


# --------------------------------------------------------- lift diagnostics --
# The lifters register per-program diagnostics ("how many atoms could the lift
# not explain") here right before lift_frame returns. It is deliberately NOT
# an attribute of the Pydantic Program: any extra __dict__ key would leak into
# ir_equal's structural comparison and into copies, while fmt/JSON must stay
# clean of it. Programs are unhashable, so the registry is keyed by id() with
# a weakref per entry to detect collection and id reuse.

_LIFT_DIAG: dict[int, tuple[weakref.ref, dict]] = {}


def _prune_lift_diag() -> None:
    dead = [k for k, (ref, _d) in _LIFT_DIAG.items() if ref() is None]
    for k in dead:
        del _LIFT_DIAG[k]


def record_lift_diag(program: Program, diag: dict) -> dict:
    """Attach a lift diagnostic ({'unexplained': n, ...}) to a program."""
    _prune_lift_diag()
    _LIFT_DIAG[id(program)] = (weakref.ref(program), dict(diag))
    return diag


def get_lift_diag(program: Program) -> dict | None:
    """The lift diagnostic of this exact program object, or None."""
    entry = _LIFT_DIAG.get(id(program))
    if entry is None:
        return None
    ref, diag = entry
    return diag if ref() is program else None


def residual_explained(program: Program, frame: Frame) -> CheckResult:
    """Never drop an atom: unexplained atoms must appear in the residual block.

    With a lift diagnostic registered (every program returned by lift_frame
    carries one), the residual block is audited against the count the lift
    itself could not explain: `residual none` requires zero unexplained, and a
    residual block must list exactly the unexplained atoms. Parsed or
    hand-written programs carry no diagnostic: only the structural property
    (a non-none residual block has its atom statements) is checked there."""
    n_res = sum(len(b.statements) for b in program.blocks if b.t == "residual" and not b.none)
    has_none = any(b.t == "residual" and b.none for b in program.blocks)
    diag = get_lift_diag(program)
    if diag is not None:
        unexplained = int(diag.get("unexplained", 0))
        if has_none or n_res == 0:
            if unexplained == 0:
                return CheckResult("residual", True,
                                   "residual none: every atom explained")
            return CheckResult(
                "residual", False,
                f"residual none but the lift left {unexplained} atoms unexplained")
        if n_res == unexplained:
            return CheckResult("residual", True,
                               f"{n_res} residual atoms = {unexplained} unexplained")
        return CheckResult(
            "residual", False,
            f"{n_res} residual atoms but the lift left {unexplained} unexplained")
    if has_none:
        return CheckResult("residual", True, "residual none (no lift diagnostic)")
    return CheckResult("residual", True, f"{n_res} atoms in residual")


def run_checks(program: Program, frame: Frame, dialect) -> list[CheckResult]:
    return [
        overlap_check(frame, dialect),
        conservation_check(program, frame),
        charge_check(program, frame, dialect),
        residual_explained(program, frame),
    ]
