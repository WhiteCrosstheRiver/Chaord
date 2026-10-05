"""Crystal builder: prototype + parameters + orientation + supercell -> Frame.

The construction is exact (no physics): conventional cell from the prototype
table, integer transformation matrix from `orient`, tiling by wrapping. If the
system cell is stated, the repetition counts must be integers within the
dialect's `lattice_match_tolerance` — a mismatch is a static error (A12) —
unless every axis is within `strain_cell_tolerance` (relative) of the integer
tiling, in which case the tiling is strain-scaled onto the stated cell and
the deformation is recorded in the build provenance (W5).
"""
from __future__ import annotations

import re
from math import gcd

import numpy as np

from ..io.frames import Frame
from ..lang.errors import ChaordError
from ..lang.ir import Program, RegionBlock, Statement
from .prototypes import PROTOTYPES, basis, cell_matrix

# per-prototype atom counts per species slot, in slot order
SLOT_COUNTS = {
    "sc": (1,), "bcc": (2,), "fcc": (4,), "hcp": (2,), "diamond": (8,),
    "rocksalt": (4, 4), "cscl": (1, 1), "zincblende": (4, 4), "wurtzite": (2, 2),
    "fluorite": (4, 8), "perovskite": (1, 1, 3), "L1_2": (3, 1), "rutile": (2, 4),
}


def _num(v) -> float:
    s = v.num if hasattr(v, "num") else str(v)
    if "/" in s:
        a, b = s.split("/")
        return float(a) / float(b)
    return float(s)


def _stmt_map(region: RegionBlock) -> dict[str, Statement]:
    return {s.key: s for s in region.statements if s.kind in ("build", "state", "constrain")}


def parse_formula(text: str) -> list[tuple[str, int]]:
    """'Ni3Al' -> [('Ni', 3), ('Al', 1)]; plain element -> [(E, 1)]."""
    pairs = re.findall(r"([A-Z][a-z]?)(\d*)", text)
    pairs = [(el, int(n) if n else 1) for el, n in pairs if el]
    if not pairs or "".join(f"{el}{n if n > 1 else ''}" for el, n in pairs) not in (
            text, text.replace("1", "")):
        raise ChaordError(f"cannot parse composition {text!r}")
    return pairs


def reduced_counts(pairs: list[tuple[str, int]]) -> tuple:
    g = 0
    for _, n in pairs:
        g = gcd(g, n)
    return tuple(n // g for _, n in pairs)


def slots_from_formula(name: str, formula: str) -> tuple:
    """Species per slot, matched by reduced count pattern (ties keep order)."""
    pairs = parse_formula(formula)
    ratio = reduced_counts(pairs)
    want = SLOT_COUNTS[name]
    g = 0
    for n in want:
        g = gcd(g, n)
    want_reduced = tuple(n // g for n in want)
    if want_reduced != ratio and len(want) == len(ratio):
        # allow reversed interpretation for symmetric prototypes
        if tuple(reversed(want_reduced)) == ratio:
            pairs = list(reversed(pairs))
        else:
            raise ChaordError(
                f"composition {formula!r} (ratio {ratio}) does not fit {name} "
                f"(slot ratio {want_reduced})")
    elif want_reduced != ratio:
        raise ChaordError(
            f"composition {formula!r} (ratio {ratio}) does not fit {name} "
            f"(slot ratio {want_reduced})")
    out = [None] * len(want)
    used = [False] * len(pairs)
    for slot, c in enumerate(want_reduced):
        for i, (el, n) in enumerate(pairs):
            if not used[i] and n == c:
                out[slot] = el
                used[i] = True
                break
    for i, (el, _) in enumerate(pairs):  # ties resolved by formula order
        if not used[i]:
            for slot in range(len(out)):
                if out[slot] is None:
                    out[slot] = el
                    used[i] = True
                    break
    if any(s is None for s in out):
        raise ChaordError(f"cannot map composition {formula!r} onto {name} slots")
    return tuple(out)


def _parse_direction(text: str) -> tuple:
    inner = text.strip()[1:-1]
    vals = [int(x) for x in re.findall(r"-?\d", inner)]
    if len(vals) not in (3, 4) or all(v == 0 for v in vals):
        raise ChaordError(f"bad direction {text!r}")
    if len(vals) == 4:  # Miller-Bravais -> Miller
        vals = [vals[0], vals[1] - vals[3], vals[2]]
    return tuple(vals)


def orient_matrix(orient_stmt) -> np.ndarray:
    """Integer transformation rows from an `orient x [u] y [v] z [w]` statement."""
    rows = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}
    vals = orient_stmt.values
    i = 0
    while i + 1 < len(vals):
        axis, tok = vals[i].text, vals[i + 1].text
        if axis in rows and tok.startswith("["):
            rows[axis] = _parse_direction(tok)
        i += 2
    T = np.array([rows["x"], rows["y"], rows["z"]], int)
    if abs(round(np.linalg.det(T))) < 1:  # singular or near-singular basis
        raise ChaordError("orient directions must form an independent lattice basis")
    return T


def build_conventional(name: str, params: dict, slots_species: tuple,
                       reps=(1, 1, 1), orient_rows=None) -> Frame:
    """Tile the conventional cell `reps` times along the oriented axes."""
    conv = cell_matrix(name, params)
    frac, slot_ids = basis(name, params)
    species = [slots_species[s] for s in slot_ids]
    T = np.eye(3, dtype=int) if orient_rows is None else np.asarray(orient_rows, int)
    cell = (T @ conv) * np.array(reps, float)[:, None]
    cart_basis = frac @ conv
    inv_cell = np.linalg.inv(cell)

    # conventional-lattice translations covering the (possibly tilted) new cell
    reach = np.abs(cell).sum(axis=0).max()
    lmin = np.linalg.norm(conv, axis=1).min()
    nmax = int(np.ceil(reach / lmin)) + 2
    grid = np.array(np.meshgrid(*[range(-nmax, nmax + 1)] * 3, indexing="ij")).reshape(3, -1).T
    offsets = grid @ conv

    pos_list, sym_list = [], []
    for p0, s in zip(cart_basis, species):
        cand = p0 + offsets
        f = cand @ inv_cell
        keep = np.all((f > -1e-9) & (f < 1 - 1e-9), axis=1)  # dialect-exempt: numerical-guard: fp tolerance on fractional coordinates
        pos_list.append(cand[keep])
        sym_list.extend([s] * int(keep.sum()))
    pos = np.vstack(pos_list)
    # wrap ONLY the rows outside the half-open cell: the keep-window is open
    # at -1e-9, so a basis point on the negative face of a tilted cell (the
    # orthorhombic hcp basis lands at x = -8.9e-17) survives as a tiny-
    # negative coordinate that cKDTree(boxsize=...) rejects. Re-projecting
    # every row would re-round all coordinates (ULP noise the 3-D mesh
    # normals are sensitive to), so untouched rows keep their exact bits; a
    # wrapped fraction that lands on 1-1e-16 collapses to 0.0 -- the same
    # periodic point on the canonical edge
    f = pos @ inv_cell
    outside = np.any((f < 0.0) | (f >= 1.0), axis=1)  # dialect-exempt: numerical-guard: half-open unit interval
    if outside.any():
        f_fix = np.mod(f[outside], 1.0)  # dialect-exempt: numerical-guard: mod-1 fractional wrap
        f_fix[f_fix > 1.0 - 1e-12] = 0.0  # dialect-exempt: numerical-guard: PBC edge collapse
        pos[outside] = f_fix @ cell

    # canonical atom order: lexicographic in cartesian coordinates
    order = np.lexsort((pos[:, 2], pos[:, 1], pos[:, 0]))
    pos, syms = pos[order], [sym_list[i] for i in order]
    return Frame(pos=pos, cell=cell, symbols=syms, pbc=(True, True, True))


def build_crystal_region(region: RegionBlock, system: dict, dialect, rng=None) -> Frame:
    if rng is None:
        rng = np.random.default_rng(0)
    stmts = _stmt_map(region)
    if "prototype" in stmts:
        name = stmts["prototype"].values[0].text
    elif "lattice" in stmts:
        name = stmts["lattice"].values[0].text
    else:
        raise ChaordError(f"region {region.name!r}: needs a prototype or lattice statement")
    if name not in PROTOTYPES:
        raise ChaordError(f"unknown prototype {name!r}")

    params = {k: _num(stmts[k].values[0])
              for k in PROTOTYPES[name].params if k in stmts}
    missing = [p for p in PROTOTYPES[name].params if p not in params]
    if missing:
        raise ChaordError(f"prototype {name!r} missing parameter(s): {', '.join(missing)}")

    if "composition" in stmts:
        formula = stmts["composition"].values[0].text
        slots_species = slots_from_formula(name, formula)
    else:
        if len(SLOT_COUNTS[name]) > 1:
            raise ChaordError(f"prototype {name!r} needs a composition statement")
        # a unary program names its species only in `conserve atoms <S> N`
        species = "X"
        conserve = system.get("atoms")
        if conserve is not None:
            named = [v.text for v in conserve.values if v.t == "n"]
            if named:
                species = named[0]
        slots_species = (species,)

    T = orient_matrix(stmts["orient"]) if "orient" in stmts else np.eye(3, dtype=int)
    reps = np.array([1, 1, 1], int)
    strain = None  # W5: per-axis (stated - tiled) / tiled once a cell is accepted strained
    if "cell" in system:
        cell_vals = [_num(v) for v in system["cell"].values if v.t == "q"]
        if len(cell_vals) == 3 and "a" in params:
            conv_oriented = T @ cell_matrix(name, params)
            lengths = np.linalg.norm(conv_oriented, axis=1)
            tol = float(dialect.threshold("lattice_match_tolerance"))
            reps = np.array([round(c / l) for c, l in zip(cell_vals, lengths)], int)
            stated = np.array(cell_vals, float)
            tiled = reps * lengths
            if not np.allclose(tiled, stated, atol=tol):
                # W5 (docs/reviews/open_items_v2.md): NPT boxes fluctuate off
                # an exact tiling. `lattice_match_tolerance` is ABSOLUTE (A,
                # over a whole box axis), so a homogeneous deformation of a
                # few tenths of a percent exceeds it on any real box while
                # every axis still reads the same integer repetition count.
                # Within `strain_cell_tolerance` (relative, per axis) the
                # builder tiles with the nearest integer counts and
                # strain-scales each axis onto the stated cell (provenance
                # note below); beyond it the A12 refusal stands. A tilted
                # oriented cell cannot be scaled per box axis, so it keeps
                # the refusal.
                strain_tol = float(dialect.threshold("strain_cell_tolerance"))
                diag_ok = np.allclose(conv_oriented, np.diag(np.diag(conv_oriented)))
                if (np.any(reps < 1) or not diag_ok
                        or float(np.max(np.abs((stated - tiled) / tiled))) > strain_tol):
                    raise ChaordError(
                        f"system cell {cell_vals} is not an integer multiple of the "
                        f"{name} lattice vectors {np.round(lengths, 4).tolist()} "
                        f"(best reps {reps.tolist()}, tolerance {tol})")
                strain = (stated - tiled) / tiled
    frame = build_conventional(name, params, slots_species, tuple(int(r) for r in reps), T)
    if strain is not None:
        # scale each axis of the exact tiling onto the stated cell: the
        # program's stated box IS the build box (F1 cell contract), and the
        # deformation is homogeneous per axis, so the tiling's periodicity
        # and wrapping survive exactly (a scaled half-open box stays
        # half-open)
        frame = Frame(pos=frame.pos * (strain + 1)[None, :],
                      cell=np.diag(stated), symbols=frame.symbols,
                      pbc=frame.pbc)
    if "cell" in system and strain is None and reps.max() > 1:
        # honour the stated cell exactly when it is an integer tiling: the
        # program's stated box IS the build box (F1 cell contract). The
        # comparison is stated lengths vs the LENGTHS of the reps-scaled
        # oriented cell rows -- the previous (3,) vs (3,3) broadcast compared
        # the lengths against off-diagonal zeros and could never fire.
        cell_vals3 = [_num(v) for v in system["cell"].values if v.t == "q"]
        if len(cell_vals3) == 3:
            oriented = T @ cell_matrix(name, params)
            diag_ok = np.allclose(oriented, np.diag(np.diag(oriented)))
            stated = np.array(cell_vals3, float)
            built_lengths = np.linalg.norm(
                np.array([oriented[i] * reps[i] for i in range(3)]), axis=1)
            if diag_ok and np.allclose(
                    stated, built_lengths,
                    atol=float(dialect.threshold("lattice_match_tolerance"))):
                frame = Frame(pos=frame.pos,
                              cell=np.diag(stated) if np.allclose(
                                  frame.cell, np.diag(np.diag(frame.cell))) else frame.cell,
                              symbols=frame.symbols, pbc=frame.pbc)

    # occupancy (solid solution) then point defects then SRO restraint
    from .defects import apply_defects, assign_occupancy, sqs_to_target
    frame = assign_occupancy(frame, region.statements, rng, dialect)
    frame = apply_defects(frame, region.statements, rng, dialect)
    for s in region.statements:
        if s.kind == "constrain" and s.key == "sro":
            vals = s.values
            pair_tok = next(v.text for v in vals if v.t == "n" and "-" in v.text)
            target = next(_num(v) for v in vals if v.t == "q")
            pair = tuple(pair_tok.split("-", 1))
            frame = sqs_to_target(frame, pair, target, rng, dialect)
    if strain is not None:
        # W5 build provenance (frame.info, the amorphous builder's
        # assumed_history channel): the per-axis engineering strain applied to
        # put the exact tiling onto the stated cell. Display precision only
        # (5 decimals of a relative strain), not a threshold.
        e = [float(x) + 0.0 for x in strain]  # dialect-exempt: numerical-guard: +0.0 collapses IEEE -0.0 for display
        frame.info["strained_to_cell"] = (
            f"strained to cell: e_xx {e[0]:+.5f} e_yy {e[1]:+.5f} e_zz {e[2]:+.5f}")
    return frame
