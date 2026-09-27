"""Defect lifting pass: a crystal frame with point defects -> defect statements.

Wigner-Seitz style (own implementation; OVITO is the optional heavy path):
1. fit the lattice directly from the frame (nearest-neighbour distance +
   per-prototype site-match score over the box — no spglib needed, defects do
   not break the fit);
2. generate the ideal sites of that lattice over the box;
3. diff actual against ideal: empty site -> vacancy (V_X), wrong species ->
   antisite (A_X), no site -> interstitial (A_i), vacancy+interstitial of one
   species within the cluster radius -> frenkel_pair.

Every threshold comes from the dialect as a fraction of the nearest-neighbour
distance, so the pass is unit-independent.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from ..build.defects import nearest_neighbor_distance, typical_neighbor_distance
from ..build.prototypes import PROTOTYPES, basis
from ..io.frames import Frame
from ..lang.errors import ChaordError
from ..realize.lj import mic

# factor taking d_NN to the cubic lattice constant for each cubic prototype
# dialect-exempt-begin: exact crystallographic conversion factors (a from d_NN)
_A_FROM_DNN = {
    "sc": 1.0, "fcc": np.sqrt(2), "bcc": 2 / np.sqrt(3), "diamond": 4 / np.sqrt(3),
    "rocksalt": 2.0, "cscl": 2 / np.sqrt(3), "zincblende": 4 / np.sqrt(3),
    "fluorite": 4 / np.sqrt(3), "perovskite": 2.0, "L1_2": np.sqrt(2),
}
# dialect-exempt-end


def _box_is_cubic(frame: Frame, tol_frac=0.02) -> bool:  # dialect-exempt: box sanity heuristic
    L = frame.cell
    off = np.abs(L - np.diag(np.diag(L))).max()
    angles_ok = np.allclose(np.diag(L), np.diag(L).mean() * np.ones(3), rtol=0.05)  # dialect-exempt: box sanity
    return off < 1e-6 and angles_ok  # dialect-exempt: zero tolerance


def ideal_sites(name: str, a: float, box: np.ndarray, slot_species: tuple):
    """All conventional-basis sites of `name` tiling the (cubic) box."""
    reps = np.round(np.diag(box) / a).astype(int)
    conv = np.eye(3) * a
    frac, slots = basis(name, {"a": a})
    cart = frac @ conv
    grid = np.array(np.meshgrid(*[range(r) for r in reps], indexing="ij")).reshape(3, -1).T
    sites = (cart[None, :, :] + grid[:, None, :] @ conv).reshape(-1, 3)
    species = [slot_species[s] for s in slots] * len(grid)
    return sites, species


def fit_crystal(frame: Frame, dialect):
    """Best (name, a, slot_species|None, score, sites, site_species).

    slot_species None marks occupancy mode: a single sublattice with mixed
    species (solid solution); site_species then lists the majority species."""
    d_nn = typical_neighbor_distance(frame)
    tol = float(dialect.threshold("site_match_tol_fraction")) * d_nn
    syms = np.array(frame.symbols)
    uniq = sorted(set(syms))
    L = frame.cell_diag
    tree = cKDTree(frame.pos, boxsize=L)
    pops = sorted(uniq, key=lambda s: (-int((syms == s).sum()), s))

    candidates = []
    unary = ("sc", "bcc", "fcc", "diamond")
    multi = {"rocksalt": 2, "cscl": 2, "zincblende": 2, "fluorite": 2,
             "perovskite": 3, "L1_2": 2}
    want_counts = {"rocksalt": (1, 1), "cscl": (1, 1), "zincblende": (1, 1),
                   "fluorite": (1, 2), "perovskite": (1, 1, 3), "L1_2": (3, 1)}
    slack = float(dialect.threshold("defect_stoichiometry_slack"))
    count_of = {s: int((syms == s).sum()) for s in uniq}
    total = len(syms)
    from itertools import permutations
    for name, n_slots in multi.items():
        if n_slots != len(uniq):
            continue
        wc = want_counts[name]
        w_total = sum(wc)
        reps = max(1, round(total / w_total))
        # counts must sit within the defect slack of the stoichiometric ratio
        if all(any(abs(count_of[s] - w * reps) <= slack * w * reps
                   for w in wc) for s in uniq):
            # tied slots (equal want counts) cannot be told apart by population:
            # try every distinct permutation; the species-aware score decides
            for perm in set(permutations(pops)):
                candidates.append((name, tuple(perm)))

    fit_tol = float(dialect.threshold("lattice_fit_tol_fraction")) * d_nn

    def scan(name, eff_slots, species_aware):
        a0 = _A_FROM_DNN[name] * d_nn
        best_local = None
        # interstitials shrink d_NN, so scan multiplicatively; rank candidates by
        # gate fraction (loose tol) then by mean matched distance (tight ranking:
        # the true lattice constant sits where sites coincide with atoms exactly)
        lo = float(dialect.threshold("lattice_scan_factor_lo"))
        hi = float(dialect.threshold("lattice_scan_factor_hi"))
        n_scan = int(dialect.threshold("lattice_scan_steps"))
        for r in np.geomspace(lo, hi, n_scan):  # dialect-exempt: fit scan window
            aa = a0 * r
            sites, s_sp = ideal_sites(name, aa, frame.cell, eff_slots)
            if len(sites) < total * 0.5:  # dialect-exempt: plausibility floor
                continue
            d_site, i_atom = tree.query(sites)
            near = d_site < tol
            gate = float(near.mean())
            if gate < float(dialect.threshold("lattice_fit_gate_min")):
                continue
            if species_aware:
                sym_arr = np.array(frame.symbols)
                near = near & (sym_arr[i_atom] == np.array(s_sp))
            mean_d = float(d_site[near].mean()) if near.any() else np.inf
            # most species-correct sites first, then smallest mean distance
            key = (-float(near.mean()), mean_d)
            if best_local is None or key < best_local[1]:
                best_local = (aa, key, sites, s_sp)
        if best_local is None:
            return None
        aa, (_neg_ok, mean_d), sites, s_sp = best_local
        sp_ok = -_neg_ok
        # refine: two shrinking fine grids around the coarse best (deterministic)
        def quality(aaa):
            sites2, s_sp2 = ideal_sites(name, aaa, frame.cell, eff_slots)
            if len(sites2) < total * 0.5:  # dialect-exempt: plausibility floor
                return np.inf, None, None
            d2, i2 = tree.query(sites2)
            near2 = d2 < tol
            if float(near2.mean()) < float(dialect.threshold("lattice_fit_gate_min")):
                return np.inf, None, None
            if species_aware:
                sym_arr = np.array(frame.symbols)
                near2 = near2 & (sym_arr[i2] == np.array(s_sp2))
            return (float(d2[near2].mean()) if near2.any() else np.inf), sites2, s_sp2

        refine_half = float(dialect.threshold("lattice_refine_half"))
        half = refine_half
        for _ in range(int(dialect.threshold("lattice_refine_passes"))):  # dialect-exempt: shrink schedule
            for aaa in np.linspace(aa - half, aa + half, 11):  # dialect-exempt: refinement grid
                if aaa <= 0:
                    continue
                qc, sc, spc = quality(aaa)
                if qc < mean_d:
                    aa, mean_d, sites, s_sp = aaa, qc, sc, spc
            half = refine_half / (4 ** (_ + 1))
        return (aa, sp_ok, sites, s_sp)

    best_multi = None
    for name, slot_species in candidates:
        r = scan(name, slot_species, species_aware=True)
        if r and r[1] > 0.5 and (best_multi is None or r[1] > best_multi[3]):  # dialect-exempt: plausibility floor
            best_multi = (name, r[0], slot_species, r[1], r[2], r[3])
    pref_min = float(dialect.threshold("stoichiometric_preference_min"))
    if best_multi is not None and best_multi[3] >= pref_min:
        return best_multi

    best = None
    for name in unary:                        # occupancy mode: any species count
        r = scan(name, (pops[0],), species_aware=False)
        if r and r[1] > 0.5 and (best is None or r[1] > best[3]):  # dialect-exempt: plausibility floor
            best = (name, r[0], None, r[1], r[2], r[3])
    if best is None:
        raise ChaordError("no cubic prototype fits the frame (M2 supports cubic defect hosts)")
    return best


def defect_diff(frame: Frame, sites, site_species, dialect, occupancy=False):
    """Classify every atom and site: vacancies, antisites, interstitials."""
    d_nn = nearest_neighbor_distance(frame)
    tol = float(dialect.threshold("site_match_tol_fraction")) * d_nn
    L = frame.cell_diag
    site_tree = cKDTree(sites, boxsize=L)
    atom_tree = cKDTree(frame.pos, boxsize=L)

    d_site, i_atom = atom_tree.query(sites)       # per site: nearest atom
    occupied = d_site < tol
    d_atom, _i_site = site_tree.query(frame.pos)  # per atom: nearest site

    vacancies = []          # site indices
    antisites = []          # (site_index, atom_index, site_species, atom_species)
    seen = set()
    for s_idx, occ in enumerate(occupied):
        if not occ:
            vacancies.append(s_idx)
            continue
        a_idx = int(i_atom[s_idx])
        if a_idx in seen:
            vacancies.append(s_idx)   # an atom cannot serve two sites
            continue
        seen.add(a_idx)
        if not occupancy and frame.symbols[a_idx] != site_species[s_idx]:
            antisites.append((s_idx, a_idx, site_species[s_idx], frame.symbols[a_idx]))

    interstitials = [i for i in range(len(frame.pos)) if d_atom[i] >= tol]
    return sorted(vacancies), antisites, interstitials


def group_defects(vacancies, antisites, interstitials, sites, site_species,
                  frame: Frame, dialect):
    """Cluster vacancies with nearby interstitials -> frenkel pairs; name the rest."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    d_nn = nearest_neighbor_distance(frame)
    rc = float(dialect.threshold("defect_cluster_rc_fraction")) * d_nn
    L = frame.cell_diag

    vac_by_species: dict[str, list[int]] = {}
    for s_idx in vacancies:
        vac_by_species.setdefault(site_species[s_idx], []).append(s_idx)

    anti_by_pair: dict[tuple[str, str], int] = {}
    for _s_idx, _a_idx, site_sp, atom_sp in antisites:
        anti_by_pair[(atom_sp, site_sp)] = anti_by_pair.get((atom_sp, site_sp), 0) + 1

    inter_by_species: dict[str, list[int]] = {}
    for a_idx in interstitials:
        inter_by_species.setdefault(frame.symbols[a_idx], []).append(a_idx)

    # frenkel pairs: vacancy of species s within rc of an interstitial of species s
    frenkel: dict[str, int] = {}
    used_vac: set[int] = set()
    used_int: set[int] = set()
    for sp, ints in inter_by_species.items():
        for a_idx in ints:
            for s_idx in vac_by_species.get(sp, []):
                if s_idx in used_vac:
                    continue
                d = mic(sites[s_idx] - frame.pos[a_idx], L)
                if np.linalg.norm(d) < rc:
                    frenkel[sp] = frenkel.get(sp, 0) + 1
                    used_vac.add(s_idx)
                    used_int.add(a_idx)
                    break

    statements = []
    for sp in sorted(vac_by_species):
        n = len([s for s in vac_by_species[sp] if s not in used_vac])
        if n:
            statements.append(("defect", f"V_{sp}", n))
    for (atom_sp, site_sp), n in sorted(anti_by_pair.items()):
        if n:
            statements.append(("defect", f"{atom_sp}_{site_sp}", n))
    for sp in sorted(inter_by_species):
        n = len([i for i in inter_by_species[sp] if i not in used_int])
        if n:
            statements.append(("defect", f"{sp}_i", n))
    for sp in sorted(frenkel):
        statements.append(("defect", "frenkel_pair", frenkel[sp]))
    # deterministic order: vacancies, antisites, interstitials, frenkel
    return statements
