"""Crystal lifter: coordinates -> canonical crystal program.

Invariance design (M1 exit criterion): the text must be byte-identical under
rotation, translation, re-ordering and re-imaging of the input frame.

1. wrap + spglib standardisation removes rotation/translation/origin/permutation
   ambiguity and yields the idealised conventional cell deterministically;
2. prototype identification matches the standardised fractional basis against
   the registry;
3. canonical rounding (dialect keys `canonical_*_decimals`) fixes the text.
"""
from __future__ import annotations

from math import gcd

import numpy as np
import spglib
from ase.data import chemical_symbols

from ..build.crystal import SLOT_COUNTS
from ..build.prototypes import PROTOTYPES, basis
from ..io.frames import Frame
from ..lang.ir import (
    GeoChain, Name, PhysicsBlock, Program, ProvenanceBlock, Quantity,
    RegionBlock, ResidualBlock, ShAll, Statement, StrVal, SystemBlock,
)


def _to_cell_tuple(frame: Frame):
    numbers = [chemical_symbols.index(s) if s in chemical_symbols else 0
               for s in frame.symbols]
    frac = frame.pos @ np.linalg.inv(frame.cell)
    return (frame.cell.tolist(), frac.tolist(), numbers)


def standardize(frame: Frame, dialect):
    symprec = float(dialect.threshold("lift_symprec"))
    cell, pos, numbers = spglib.standardize_cell(
        _to_cell_tuple(frame), to_primitive=False, no_idealize=False, symprec=symprec)
    if cell is None:
        raise ValueError("spglib could not standardise the cell "
                         "(not periodic within symprec)")
    return np.array(cell), np.mod(np.array(pos), 1.0), np.array(numbers)  # dialect-exempt: fractional wrap


def _lengths_angles(cell: np.ndarray):
    lengths = np.linalg.norm(cell, axis=1)
    angles = []
    for i, j in ((1, 2), (0, 2), (0, 1)):
        cosv = np.dot(cell[i], cell[j]) / (lengths[i] * lengths[j])
        angles.append(np.degrees(np.arccos(np.clip(cosv, -1, 1))))
    return lengths, np.array(angles)


def _axis_orders_for(name: str, cell: np.ndarray, tol):
    """All axis permutations consistent with the prototype's lattice family.

    Different conventions (spglib's standard setting vs our tables) may swap
    a1/a2 or put c elsewhere; the matcher tries every geometrically consistent
    permutation and keeps the best fit."""
    from itertools import permutations
    lengths, angles = _lengths_angles(cell)
    cubic = np.ptp(lengths) <= tol and abs(angles - 90).max() <= 2 * tol
    if name in ("hcp", "wurtzite", "rutile"):
        want_gamma = 120.0 if name in ("hcp", "wurtzite") else 90.0  # dialect-exempt: crystallographic constants
        out = []
        for perm in permutations(range(3)):
            i, j, k = perm
            if abs(lengths[i] - lengths[j]) > tol or abs(lengths[k] - lengths[i]) <= tol:
                continue
            ang = _angle_between(cell[i], cell[j])
            others = (_angle_between(cell[i], cell[k]), _angle_between(cell[j], cell[k]))
            if abs(ang - want_gamma) <= 2 * tol and abs(np.array(others) - 90).max() <= 2 * tol:
                out.append(np.array(perm))
        return out
    if cubic:
        return [np.array(perm) for perm in permutations(range(3))]
    return []


def _angle_between(u, v):
    cosv = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return float(np.degrees(np.arccos(np.clip(cosv, -1, 1))))


def _params_for(name: str, cell: np.ndarray, tol):
    """Lattice parameters from the axis-ordered cell, or None."""
    orders = _axis_orders_for(name, cell, tol)
    if not orders:
        return None
    lengths, _ = _lengths_angles(cell[orders[0]])
    if name in ("hcp", "wurtzite", "rutile"):
        return {"a": 0.5 * (lengths[0] + lengths[1]), "c": lengths[2]}  # dialect-exempt: mean of the a-pair
    return {"a": lengths.mean()}


def _basis_rms(cell, pos, numbers, trial, slot_species, tol):
    """Greedy fractional matching of the trial basis against pos/numbers."""
    frac, slots = trial
    rms_acc = []
    used = np.zeros(len(pos), bool)
    for f, slot in zip(frac, slots):
        sym = slot_species[slot]
        num = chemical_symbols.index(sym)
        cand = np.where((numbers == num) & ~used)[0]
        if len(cand) == 0:
            return None
        d = pos[cand] - f
        d -= np.round(d)
        dist = np.linalg.norm(d @ cell, axis=1)
        i = int(np.argmin(dist))
        if dist[i] > tol:
            return None
        used[cand[i]] = True
        rms_acc.append(dist[i])
    return float(np.sqrt(np.mean(np.square(rms_acc))))


def _slot_candidates(numbers, name):
    """All slot->species assignments consistent with per-species atom counts."""
    uniq = sorted(set(int(n) for n in numbers))
    counts = {n: int((numbers == n).sum()) for n in uniq}
    want = SLOT_COUNTS[name]
    total = sum(counts.values())
    want_total = sum(want)
    if total % want_total != 0:
        return []
    reps = total // want_total
    if sorted(counts.values()) != sorted(w * reps for w in want):
        return []
    return [(a, reps) for a in _assign(want, counts, reps)]


def _assign(want, counts, reps):
    """Enumerate assignments: each slot takes an unused species with matching count."""
    avail = list(counts.keys())
    results = []

    def rec(i, used, acc):
        if i == len(want):
            results.append(list(acc))
            return
        w = want[i]
        for n in avail:
            if n not in used and counts[n] == w * reps:
                rec(i + 1, used | {n}, acc + [n])

    rec(0, set(), [])
    return [tuple(chemical_symbols[n] for n in r) for r in results]


def match_prototype(cell, pos, numbers, dialect):
    """Best (name, params, slot_species, reps, rms) matching the standardised basis.

    Searches axis permutations (setting conventions), origin shifts (spglib's
    origin choice) and tied-slot species permutations, keeping the lowest rms."""
    tol = float(dialect.threshold("prototype_match_tol"))
    best = None
    for name, proto in PROTOTYPES.items():
        cands = _slot_candidates(numbers, name)
        if not cands:
            continue
        params = _params_for(name, cell, tol)
        if params is None:
            continue
        for order in _axis_orders_for(name, cell, tol):
            cell_o, pos_o = cell[order], pos[:, order]
            for slot_species, reps in cands:
                rms = _match_with_origin(cell_o, pos_o, numbers, name, params,
                                         slot_species, tol)
                if rms is None:
                    continue
                is_default = slot_species == PROTOTYPES[name].default_slots
                better = (
                    best is None
                    or rms < best[4] - 1e-12                       # dialect-exempt: fp tie margin
                    or (abs(rms - best[4]) <= 1e-12                # dialect-exempt: fp tie margin
                        and is_default
                        and best[2] != PROTOTYPES[best[0]].default_slots)
                    or (abs(rms - best[4]) <= 1e-12                # dialect-exempt: fp tie margin
                        and is_default and best[0] == name))
                if better:
                    best = (name, params, slot_species, reps, rms)
    return best


def _match_with_origin(cell, pos, numbers, name, params, slot_species, tol):
    frac, slots = basis(name, params)
    mirrored = frac.copy()
    mirrored[:, 2] = np.mod(-mirrored[:, 2], 1.0)  # dialect-exempt: polar twins u<->1-u
    first_sym = slot_species[slots[0]]
    first_num = chemical_symbols.index(first_sym)
    best = None
    for trial_frac in (frac, mirrored):
        anchors = pos[numbers == first_num] - trial_frac[0]
        for shift in anchors:
            rms = _basis_rms_shifted(cell, pos, numbers, (trial_frac, slots),
                                     slot_species, tol, shift)
            if rms is not None and (best is None or rms < best):
                best = rms
    return best


def _basis_rms_shifted(cell, pos, numbers, trial, slot_species, tol, shift):
    frac, slots = trial
    frac = np.mod(frac + shift, 1.0)  # dialect-exempt: fractional wrap
    rms_acc = []
    used = np.zeros(len(pos), bool)
    for f, slot in zip(frac, slots):
        sym = slot_species[slot]
        num = chemical_symbols.index(sym)
        cand = np.where((numbers == num) & ~used)[0]
        if len(cand) == 0:
            return None
        d = pos[cand] - f
        d -= np.round(d)
        dist = np.linalg.norm(d @ cell, axis=1)
        i = int(np.argmin(dist))
        if dist[i] > tol:
            return None
        used[cand[i]] = True
        rms_acc.append(dist[i])
    # reconstruction: the shifted basis must tile the whole cell — every atom
    # sits on some basis site, so no prototype can match a subset of the atoms
    for a, num in zip(pos, numbers):
        sym = chemical_symbols[num]
        if sym not in slot_species:
            return None
        d = a - frac[[i for i, s in enumerate(slots) if slot_species[s] == sym]]
        d -= np.round(d)
        if np.linalg.norm(d @ cell, axis=1).min() > tol:
            return None
    return float(np.sqrt(np.mean(np.square(rms_acc))))


def formula_from_slots(name: str, slot_species: tuple) -> str:
    counts = SLOT_COUNTS[name]
    g = 0
    for n in counts:
        g = gcd(g, n)
    return "".join(f"{el}{n // g if n // g > 1 else ''}"
                   for el, n in zip(slot_species, counts))


def round_canonical(v: float, decimals_key: str, dialect) -> str:
    n = int(dialect.threshold(decimals_key))
    return f"{v:.{n}f}"


def lift_crystal(frame: Frame, dialect, backend="eam"):
    """Lift a defect-free crystal frame into a canonical Program."""
    cell, pos, numbers = standardize(frame, dialect)
    matched = match_prototype(cell, pos, numbers, dialect)
    if matched is None:
        raise ValueError("no prototype matches the standardised structure")
    name, params, slot_species, _reps, _rms = matched
    proto = PROTOTYPES[name]

    L = np.linalg.norm(frame.cell, axis=1)
    n_atoms: dict[str, int] = {}
    for s in frame.symbols:
        n_atoms[s] = n_atoms.get(s, 0) + 1
    conserve_values = []
    for s in sorted(n_atoms):
        conserve_values += [Name(text=s), Quantity(num=str(n_atoms[s]))]

    unit = dialect.threshold("length_unit")
    region_stmts = []
    if len(slot_species) > 1:
        region_stmts.append(Statement(kind="build", key="prototype",
                                      values=[Name(text=name)]))
        region_stmts.append(Statement(kind="build", key="composition",
                                      values=[Name(text=formula_from_slots(name, slot_species))]))
    else:
        region_stmts.append(Statement(kind="build", key="lattice",
                                      values=[Name(text=name)]))
    for pname in proto.params:
        key = pname
        decimals = "canonical_a_decimals" if key == "a" else "canonical_c_decimals"
        region_stmts.append(Statement(
            kind="build", key=key,
            values=[Quantity(num=round_canonical(params[pname], decimals, dialect),
                             unit=unit if unit != "none" else None)]))
    region_stmts.append(Statement(kind="build", key="orient", values=[
        Name(text="x"), Name(text="[100]"), Name(text="y"), Name(text="[010]"),
        Name(text="z"), Name(text="[001]")]))
    region_stmts.append(Statement(kind="assert", key="sites_matched",
                                  values=[Quantity(num="100.0", unit="%")]))  # dialect-exempt: canonical text

    system = SystemBlock(statements=[
        Statement(kind="build", key="cell", values=[
            Quantity(num=round_canonical(L[i], "canonical_cell_decimals", dialect))
            for i in range(3)]),
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
        Statement(kind="build", key="dialects", values=[StrVal(text=dialect.version_string)]),
        Statement(kind="build", key="lift_version", values=[StrVal(text="0.1.0")]),
    ])
    return Program(version="0.1", dialects=list(dialect.names),  # dialect-exempt: language version
                   blocks=[system, physics, region, ResidualBlock(none=True), provenance])
