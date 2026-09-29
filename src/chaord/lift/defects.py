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
# dialect-exempt-begin: exact-geometry
_A_FROM_DNN = {
    "sc": 1.0, "fcc": np.sqrt(2), "bcc": 2 / np.sqrt(3), "diamond": 4 / np.sqrt(3),
    "rocksalt": 2.0, "cscl": 2 / np.sqrt(3), "zincblende": 4 / np.sqrt(3),
    "fluorite": 4 / np.sqrt(3), "perovskite": 2.0, "L1_2": np.sqrt(2),
}
# dialect-exempt-end


def _box_is_cubic(frame: Frame, tol_frac=0.02) -> bool:  # dialect-exempt: numerical-guard: box-shape sanity heuristic (currently unused)
    L = frame.cell
    off = np.abs(L - np.diag(np.diag(L))).max()
    angles_ok = np.allclose(np.diag(L), np.diag(L).mean() * np.ones(3), rtol=0.05)  # dialect-exempt: numerical-guard: box-shape sanity
    return off < 1e-6 and angles_ok  # dialect-exempt: numerical-guard: zero tolerance for off-diagonal cell entries


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


def _wrap_strict(pos: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Wrap into [0, L) with a strict upper edge (KD trees with `boxsize`)."""
    out = np.mod(pos, L)
    return np.minimum(out, L * (1 - 1e-9))  # dialect-exempt: numerical-guard: strict upper edge


def _anchor_seed(frame: Frame, name: str, eff_slots: tuple,
                 species_aware: bool, tree, tol: float, a0: float):
    """Best lattice-translation seed for re-anchoring a re-imaged frame
    (rule 3: identical text under translation and re-imaging).

    `ideal_sites` generates origin-anchored sites; a rigidly transformed input
    carries an arbitrary offset. Any atom of the anchor species (the species
    of basis site 0, so the seed preserves the ordered species pattern) sits
    on a basis site up to its own displacement, so a few spread over the
    lexsorted anchor atoms seed candidate offsets; interstitials among them
    lose on the gate. Returns the seed scoring best at the base lattice
    estimate, or None when no seed brings sites near atoms."""
    sites0, s_sp = ideal_sites(name, a0, frame.cell, eff_slots)
    sym_arr = np.array(frame.symbols)
    site_species = np.array(s_sp)
    if species_aware:
        pos = frame.pos[sym_arr == s_sp[0]]
        if len(pos) == 0:
            pos = frame.pos
    else:
        pos = frame.pos
    order = np.lexsort((pos[:, 2], pos[:, 1], pos[:, 0]))  # content-fixed seeds
    picks = pos[order[np.linspace(0, len(order) - 1, 4).astype(int)]]
    L = frame.cell_diag
    best = None
    for s0 in picks:
        sites = _wrap_strict(sites0 + s0, L)
        d_site, i_atom = tree.query(sites)
        near = d_site < tol
        if species_aware:
            near = near & (sym_arr[i_atom] == site_species)
        if not near.any():
            continue
        key = (-float(near.mean()), float(d_site[near].mean()))
        if best is None or key < best[0]:
            best = (key, s0)
    return None if best is None else best[1]


def _anchored_sites(name: str, a: float, frame: Frame, eff_slots: tuple,
                    shift: np.ndarray):
    """Ideal sites of the candidate lattice, translated by the anchored shift."""
    sites, s_sp = ideal_sites(name, a, frame.cell, eff_slots)
    return _wrap_strict(sites + shift, frame.cell_diag), s_sp


def fit_crystal(frame: Frame, dialect):
    """Best (name, a, slot_species|None, score, sites, site_species).

    slot_species None marks occupancy mode: a single sublattice with mixed
    species (solid solution); site_species then lists the majority species."""
    d_nn = typical_neighbor_distance(frame)
    tol = float(dialect.threshold("site_match_tol_fraction")) * d_nn
    syms = np.array(frame.symbols)
    pos_wrapped = np.mod(frame.pos, frame.cell_diag)  # atom-coverage queries
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
    plausibility = float(dialect.threshold("site_plausibility_floor"))

    def scan(name, eff_slots, species_aware):
        a0 = _A_FROM_DNN[name] * d_nn
        # interstitials shrink d_NN, so scan multiplicatively; rank candidates by
        # gate fraction (loose tol) then by mean matched distance (tight ranking:
        # the true lattice constant sits where sites coincide with atoms exactly)
        lo = float(dialect.threshold("lattice_scan_factor_lo"))
        hi = float(dialect.threshold("lattice_scan_factor_hi"))
        n_scan = int(dialect.threshold("lattice_scan_steps"))

        def coarse(shift):
            best_local = None
            for r in np.geomspace(lo, hi, n_scan):
                aa = a0 * r
                sites, s_sp = _anchored_sites(name, aa, frame, eff_slots, shift)
                if len(sites) < total * plausibility:
                    continue
                d_site, i_atom = tree.query(sites)
                near = d_site < tol
                if float(near.mean()) < float(dialect.threshold("lattice_fit_gate_min")):
                    continue
                # atom coverage: every atom must sit near some site too, otherwise
                # the candidate is a half-density sublattice of the true lattice
                # (unary bcc/diamond degenerate to sc/fcc without this gate)
                sites_wrapped = np.minimum(np.mod(sites, frame.cell_diag),
                                           frame.cell_diag * (1 - 1e-9))  # dialect-exempt: numerical-guard: strict upper edge
                d_atom, _ = cKDTree(sites_wrapped, boxsize=frame.cell_diag).query(
                    np.minimum(pos_wrapped, frame.cell_diag * (1 - 1e-9)))  # dialect-exempt: numerical-guard: strict upper edge
                if float((d_atom < tol).mean()) < float(
                        dialect.threshold("lattice_fit_gate_min")):
                    continue
                if species_aware:
                    sym_arr = np.array(frame.symbols)
                    near = near & (sym_arr[i_atom] == np.array(s_sp))
                mean_d = float(d_site[near].mean()) if near.any() else np.inf
                # most species-correct sites first, then smallest mean distance
                key = (-float(near.mean()), mean_d)
                if best_local is None or key < best_local[1]:
                    best_local = (aa, key, sites, s_sp)
            return best_local

        def quality(aaa, sh):
            sites2, s_sp2 = _anchored_sites(name, aaa, frame, eff_slots, sh)
            if len(sites2) < total * plausibility:
                return np.inf, None, None
            d2, i2 = tree.query(sites2)
            near2 = d2 < tol
            if float(near2.mean()) < float(dialect.threshold("lattice_fit_gate_min")):
                return np.inf, None, None
            sites2_wrapped = np.minimum(np.mod(sites2, frame.cell_diag),
                                        frame.cell_diag * (1 - 1e-9))  # dialect-exempt: numerical-guard: strict upper edge
            d_atom2, _ = cKDTree(sites2_wrapped, boxsize=frame.cell_diag).query(
                np.minimum(pos_wrapped, frame.cell_diag * (1 - 1e-9)))  # dialect-exempt: numerical-guard: strict upper edge
            if float((d_atom2 < tol).mean()) < float(
                    dialect.threshold("lattice_fit_gate_min")):
                return np.inf, None, None
            if species_aware:
                sym_arr = np.array(frame.symbols)
                near2 = near2 & (sym_arr[i2] == np.array(s_sp2))
            return (float(d2[near2].mean()) if near2.any() else np.inf), sites2, s_sp2

        def chain(shift):
            """Coarse scan + shrinking refine grids for one site offset
            (deterministic)."""
            best_local = coarse(shift)
            if best_local is None:
                return None
            aa, (_neg_gate, mean_d), sites, s_sp = best_local
            refine_half = float(dialect.threshold("lattice_refine_half"))
            half = refine_half
            for _ in range(int(dialect.threshold("lattice_refine_passes"))):
                for aaa in np.linspace(aa - half, aa + half, 11):
                    if aaa <= 0:
                        continue
                    qc, sc, spc = quality(aaa, shift)
                    if qc < mean_d:
                        aa, mean_d, sites, s_sp = aaa, qc, sc, spc
                half = refine_half / (4 ** (_ + 1))
            return (-_neg_gate, mean_d, aa, sites, s_sp)

        # translation re-anchoring (rule 3): the origin-anchored chain is the
        # historical fit and stays in force unless the frame was re-imaged --
        # detected by a re-anchored chain improving the mean matched distance
        # by at least the dialect factor (a marginal improvement is thermal
        # noise, not a translation, and must not move the fitted sites)
        origin = chain(np.zeros(3))
        reanchored = None
        seed = _anchor_seed(frame, name, eff_slots, species_aware, tree,
                            tol, a0)
        if seed is not None and not np.array_equal(seed, np.zeros(3)):
            reanchored = chain(seed)
        improvement = float(dialect.threshold("lattice_anchor_improvement"))
        if origin is None:
            chosen = reanchored
        elif (reanchored is not None
              and reanchored[1] <= improvement * origin[1]):
            chosen = reanchored
        else:
            chosen = origin
        if chosen is None:
            return None
        gate, mean_d, aa, sites, s_sp = chosen
        return (aa, gate, sites, s_sp, mean_d)

    # candidates are ranked by species-aware gate; exact gate ties are broken
    # by the mean matched distance, so a re-anchored tied-slot permutation
    # (species-swapped pattern re-anchored onto the atoms, a looser fit) loses
    # to the correctly assigned origin-anchored candidate
    best_multi = None
    for name, slot_species in candidates:
        r = scan(name, slot_species, species_aware=True)
        if r and r[1] > plausibility and (
                best_multi is None
                or (r[1], -r[4]) > (best_multi[3], -best_multi[6])):
            best_multi = (name, r[0], slot_species, r[1], r[2], r[3], r[4])
    pref_min = float(dialect.threshold("stoichiometric_preference_min"))
    if best_multi is not None and best_multi[3] >= pref_min:
        return best_multi[:6]

    best = None
    for name in unary:                        # occupancy mode: any species count
        r = scan(name, (pops[0],), species_aware=False)
        if r and r[1] > plausibility and (
                best is None or (r[1], -r[4]) > (best[3], -best[6])):
            best = (name, r[0], None, r[1], r[2], r[3], r[4])
    if best is None:
        raise ChaordError("no cubic prototype fits the frame (M2 supports cubic defect hosts)")
    return best[:6]


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
                  frame: Frame, dialect, dnn_lattice=None):
    """Cluster vacancies with nearby interstitials -> frenkel pairs; name the rest.

    dnn_lattice (optional) anchors the pairing radius to the FITTED lattice
    spacing rather than the point cloud's median: planted interstitials
    contaminate the median and would shrink the radius exactly where pairing
    matters."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    d_nn = dnn_lattice if dnn_lattice is not None else typical_neighbor_distance(frame)
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

    # frenkel pairs: a vacancy and an interstitial of the SAME species whose
    # separation is within the pair radius. The pair radius is tighter than the
    # generic cluster radius: a true frenkel interstitial sits in a nearby
    # interstice (~0.6 dnn), while thermal jitter pairs two unrelated atoms at
    # random separation -- pairing those inflates false positives at 0.8Tm.
    frenkel: dict[str, int] = {}
    used_vac: set[int] = set()
    used_int: set[int] = set()
    pair_rc = float(dialect.threshold("frenkel_pair_rc_fraction")) * d_nn
    for sp, ints in inter_by_species.items():
        for a_idx in ints:
            best, best_d = None, np.inf
            for s_idx in vac_by_species.get(sp, []):
                if s_idx in used_vac:
                    continue
                d = np.linalg.norm(mic(sites[s_idx] - frame.pos[a_idx], L))
                if d < best_d:
                    best, best_d = s_idx, d
            if best is not None and best_d < pair_rc:
                frenkel[sp] = frenkel.get(sp, 0) + 1
                used_vac.add(best)
                used_int.add(a_idx)

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
