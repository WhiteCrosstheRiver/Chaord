"""Point-defect builder: Kröger-Vink statements -> coordinate operations.

`defect V_Ni count 1`  -> remove one Ni atom (vacancy)
`defect Al_Ni count 2` -> turn two Ni sites into Al (antisite)
`defect Ni_i count 1`  -> add one Ni interstitial near an existing atom
`defect frenkel_pair count 1` -> one vacancy + its atom parked as interstitial

All randomness is seeded through the passed rng. Distances come from the
dialect as fractions of the nearest-neighbour distance (unit-independent).
"""
from __future__ import annotations

import numpy as np

from ..io.frames import Frame
from ..lang.errors import ChaordError
from ..realize.lj import mic


def parse_kv(token: str) -> tuple[str, str]:
    """'V_O^..' -> ('V', 'O');  'Al_Ni' -> ('Al', 'Ni');  'Ni_i' -> ('Ni', 'i')."""
    body = token.split("^")[0]
    if "_" not in body:
        raise ChaordError(f"not a Kröger-Vink token: {token!r}")
    a, b = body.split("_", 1)
    if not a or not b:
        raise ChaordError(f"not a Kröger-Vink token: {token!r}")
    return a, b


def nearest_neighbor_distance(frame: Frame) -> float:
    from scipy.spatial import cKDTree
    L = frame.cell_diag if np.allclose(frame.cell - np.diag(np.diag(frame.cell)), 0) \
        else None
    tree = cKDTree(frame.pos, boxsize=L)
    d, _ = tree.query(frame.pos, k=2)
    return float(d[:, 1].min())


def typical_neighbor_distance(frame: Frame) -> float:
    """Median nearest-neighbour distance: robust to a few interstitial contacts."""
    from scipy.spatial import cKDTree
    L = frame.cell_diag if np.allclose(frame.cell - np.diag(np.diag(frame.cell)), 0)         else None
    tree = cKDTree(frame.pos, boxsize=L)
    d, _ = tree.query(frame.pos, k=2)
    return float(np.median(d[:, 1]))


def _nn_pairs(frame: Frame):
    from scipy.spatial import cKDTree
    L = frame.cell_diag if np.allclose(frame.cell - np.diag(np.diag(frame.cell)), 0) \
        else None
    return cKDTree(frame.pos, boxsize=L)


def apply_defects(frame: Frame, defect_stmts, rng, dialect) -> Frame:
    """Apply `defect` statements (build kind) to a perfect-crystal frame."""
    if not defect_stmts:
        return frame
    d_nn = nearest_neighbor_distance(frame)
    site_frac = float(dialect.threshold("site_match_tol_fraction"))
    int_frac = float(dialect.threshold("interstitial_distance_fraction"))

    pos = list(frame.pos)
    syms = list(frame.symbols)
    L = frame.cell_diag

    for stmt in defect_stmts:
        if stmt.key != "defect":
            continue
        vals = stmt.values
        name_tok = vals[0].text
        count = 1
        for i, v in enumerate(vals):
            if v.t == "n" and v.text == "count" and i + 1 < len(vals):
                count = int(float(vals[i + 1].num))
        rng.shuffle(pos_proxy := np.arange(len(pos)))  # noqa: F841 — keeps rng stream stable

        if name_tok == "frenkel_pair":
            for _ in range(count):
                idx = int(rng.integers(len(pos)))
                p0 = np.array(pos.pop(idx))
                sp = syms.pop(idx)
                direction = rng.normal(size=3)
                direction /= np.linalg.norm(direction)
                new = p0 + direction * int_frac * d_nn
                pos.append(np.mod(new, L))
                syms.append(sp)
            continue

        site, sub = parse_kv(name_tok)
        if site == "V":                     # vacancy: remove `sub` atoms
            for _ in range(count):
                cand = [i for i, s in enumerate(syms) if s == sub]
                if not cand:
                    raise ChaordError(f"no {sub} atoms left to vacate")
                idx = int(rng.choice(cand))
                pos.pop(idx)
                syms.pop(idx)
        elif sub == "i":                    # interstitial of species `site`
            for _ in range(count):
                idx = int(rng.integers(len(pos)))
                direction = rng.normal(size=3)
                direction /= np.linalg.norm(direction)
                new = np.array(pos[idx]) + direction * int_frac * d_nn
                pos.append(np.mod(new, L))
                syms.append(site)
        else:                               # antisite: species `site` on `sub` sites
            for _ in range(count):
                cand = [i for i, s in enumerate(syms) if s == sub]
                if not cand:
                    raise ChaordError(f"no {sub} sites left to replace with {site}")
                idx = int(rng.choice(cand))
                syms[idx] = site
    return Frame(pos=np.array(pos), cell=frame.cell, symbols=syms, pbc=frame.pbc)


# ---------------------------------------------------------------- SRO / SQS --

def assign_occupancy(frame: Frame, occupancy_stmts, rng, dialect) -> Frame:
    """An occupancy statement with unit fractions -> random assignment, alpha = 0."""
    syms = list(frame.symbols)
    n = len(syms)
    species, fracs = [], []
    for stmt in occupancy_stmts:
        if stmt.key != "occupancy":
            continue
        vals = stmt.values
        i = 0
        while i + 1 < len(vals):
            if vals[i].t == "n":
                species.append(vals[i].text)
                fracs.append(_frac(vals[i + 1]))
            i += 2
    if not species:
        return frame
    total = sum(fracs)
    fracs = [f / total for f in fracs]
    counts = [int(round(f * n)) for f in fracs]
    counts[-1] = n - sum(counts[:-1])
    out = []
    for s, c in zip(species, counts):
        out.extend([s] * c)
    rng.shuffle(out)
    return Frame(pos=frame.pos, cell=frame.cell, symbols=out, pbc=frame.pbc)


def _frac(v) -> float:
    s = v.num if hasattr(v, "num") else str(v)
    if "/" in s:
        a, b = s.split("/")
        return float(a) / float(b)
    return float(s)


def warren_cowley_alpha1(frame: Frame, a: str, b: str, cutoff: float) -> float:
    """First-shell Warren-Cowley alpha for pair (a, b).

    alpha_ab = 1 - P(b|a) / x_b, with P(b|a) the fraction of a-atoms' first-shell
    neighbours that are b. The same code serves lift (measure), build (restrain)
    and `assert` (check)."""
    from scipy.spatial import cKDTree
    L = frame.cell_diag
    tree = cKDTree(frame.pos, boxsize=L)
    syms = np.array(frame.symbols)
    ia = np.where(syms == a)[0]
    if len(ia) == 0:
        raise ChaordError(f"species {a!r} not present")
    x_b = float((syms == b).mean())
    if x_b == 0:
        raise ChaordError(f"species {b!r} not present")
    nb_count = 0
    tot = 0
    for i in ia:
        nbrs = tree.query_ball_point(frame.pos[i], cutoff)
        for j in nbrs:
            if j == i:
                continue
            tot += 1
            if syms[j] == b:
                nb_count += 1
    if tot == 0:
        raise ChaordError("no neighbours within cutoff")
    p_ba = nb_count / tot
    return 1.0 - p_ba / x_b  # dialect-exempt: Warren-Cowley definition


def sqs_to_target(frame: Frame, pair: tuple[str, str], target: float,
                  rng, dialect) -> Frame:
    """Monte-Carlo species swaps driving alpha1(pair) to `target` (SQS-lite).

    icet is the heavy-tool path; this is the seeded core fallback."""
    steps = int(dialect.threshold("sro_mc_steps"))
    d_nn = nearest_neighbor_distance(frame)
    cutoff = float(dialect.threshold("sro_shell1_factor")) * d_nn
    syms = list(frame.symbols)
    a, b = pair

    def alpha():
        f = Frame(pos=frame.pos, cell=frame.cell, symbols=syms, pbc=frame.pbc)
        return warren_cowley_alpha1(f, a, b, cutoff)

    current = alpha()
    converge = float(dialect.threshold("sqs_converge_tol"))
    for _ in range(steps):
        if abs(current - target) < converge:
            break
        # candidates must be recomputed each step: swaps move species around.
        # for an auto-correlation pair (a == a) the partner is any non-a site;
        # swapping a with a would be a no-op.
        idx_a = [i for i, s in enumerate(syms) if s == a]
        idx_b = ([i for i, s in enumerate(syms) if s != a] if a == b
                 else [i for i, s in enumerate(syms) if s == b])
        if not idx_a or not idx_b:
            break
        i, j = int(rng.choice(idx_a)), int(rng.choice(idx_b))
        syms[i], syms[j] = syms[j], syms[i]
        trial = alpha()
        if abs(trial - target) < abs(current - target):
            current = trial
        else:
            syms[i], syms[j] = syms[j], syms[i]
    return Frame(pos=frame.pos, cell=frame.cell, symbols=syms, pbc=frame.pbc)
