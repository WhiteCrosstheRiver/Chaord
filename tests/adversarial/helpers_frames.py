"""Shared adversarial frame constructors (imported by the sibling test
modules; pytest's prepend import mode puts this directory on sys.path)."""
from __future__ import annotations

import numpy as np

from chaord.build.crystal import build_conventional
from chaord.io.frames import Frame
from tools import acceptance as acc


def divacancy_fcc_cu(seed: int = 7) -> Frame:
    """fcc-Cu 3x3x3 with one ADJACENT vacancy pair removed (a divacancy)."""
    from scipy.spatial import cKDTree
    perfect = build_conventional("fcc", {"a": 3.615}, ("Cu",), (3, 3, 3))
    d = acc.median_nn_distance(perfect.pos, perfect.cell_diag)
    rng = np.random.default_rng(seed)
    tree = cKDTree(perfect.pos, boxsize=perfect.cell_diag)
    pairs = tree.query_pairs(1.1 * d, output_type="ndarray")
    pair = pairs[rng.integers(len(pairs))]
    keep = np.ones(len(perfect), bool)
    keep[pair] = False
    return Frame(pos=perfect.pos[keep], cell=perfect.cell,
                 symbols=[s for s, k in zip(perfect.symbols, keep) if k],
                 pbc=perfect.pbc)


def mixed_l12_defects(seed: int = 11):
    """L1_2 Ni3Al 4x4x4 with V_Ni x3 (separated), Al_Ni x4 (separated from
    the vacancies) and Ni_i x3 interstitials planted by coordinate surgery."""
    perfect = build_conventional("L1_2", {"a": 3.572}, ("Ni", "Al"), (4, 4, 4))
    d = acc.median_nn_distance(perfect.pos, perfect.cell_diag)
    rng = np.random.default_rng(seed)
    pos, syms = perfect.pos.copy(), list(perfect.symbols)
    idx_v = acc._pick_separated(pos, syms, "Ni", 3, 2.0 * d, rng,
                                return_idx=True)
    cand = [i for i, s in enumerate(syms) if s == "Ni" and i not in idx_v]
    idx_a: list[int] = []
    for i in rng.permutation(cand):
        if idx_a and np.min(np.linalg.norm(pos[idx_a] - pos[i], axis=1)) < 2 * d:
            continue
        if np.min(np.linalg.norm(pos[idx_v] - pos[i], axis=1)) < 2 * d:
            continue
        idx_a.append(int(i))
        if len(idx_a) == 4:
            break
    parents = rng.permutation(len(pos))[:3]
    offs = acc._random_unit(rng, 3) * (0.60 * d)
    keep = np.ones(len(pos), bool)
    keep[idx_v] = False
    mixed_pos = np.vstack([pos[keep],
                           np.mod(pos[parents] + offs, perfect.cell_diag)])
    mixed_syms = [s for s, k in zip(syms, keep) if k] + ["Ni"] * 3
    old2new = {old: new for new, old in enumerate(np.where(keep)[0])}
    for i in idx_a:
        mixed_syms[old2new[i]] = "Al"
    frame = Frame(pos=mixed_pos, cell=perfect.cell, symbols=mixed_syms,
                  pbc=perfect.pbc)
    return frame, float(perfect.cell_diag[0])


def thermal(frame: Frame, amp: float, seed: int) -> Frame:
    """Gaussian displacement wrap into the box (verifier-side)."""
    rng = np.random.default_rng(seed)
    return Frame(pos=np.mod(frame.pos + rng.normal(size=frame.pos.shape) * amp,
                           frame.cell_diag),
                cell=frame.cell, symbols=frame.symbols, pbc=frame.pbc)
