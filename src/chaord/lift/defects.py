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
from .crystal import snap_a_to_cell

# factor taking d_NN to the cubic lattice constant for each cubic prototype
# dialect-exempt-begin: exact-geometry
_A_FROM_DNN = {
    "sc": 1.0, "fcc": np.sqrt(2), "bcc": 2 / np.sqrt(3), "diamond": 4 / np.sqrt(3),
    "rocksalt": 2.0, "cscl": 2 / np.sqrt(3), "zincblende": 4 / np.sqrt(3),
    "fluorite": 4 / np.sqrt(3), "perovskite": 2.0, "L1_2": np.sqrt(2),
}
# dialect-exempt-end

# ---- thermal quench (review 2 root-cause fix, 2026-09-29) --------------------
#
# A frame at the 0.8 Tm thermal amplitude defeats the Wigner-Seitz diff in two
# coupled ways: the displacement noise pushes on-site atoms past the
# site-match tolerance, and a tolerance derived from the atom cloud (the
# MINIMUM neighbour distance defect_diff used, or the median group_defects
# guards against) is contaminated by the defects themselves -- planted
# interstitials sit 0.4-0.6 d_NN from sites and thermal collisions shrink the
# minimum well below the lattice spacing -- so the diff gets TIGHTER exactly
# where it must not, and thermally displaced atoms read as vacancy +
# interstitial (a spurious frenkel_pair; A4 precision 0.09 on
# L12-NiAl/Ni_i at 0.8 Tm).
#
# The fix is the review-2 thermal quench, run inside defect_diff before the
# comparison: relax a copy of the frame with the backend's Lennard-Jones pair
# potential so the thermal noise (the deviation from the local minimum)
# vanishes while the topological defects survive.  A FREE minimization
# provably cannot do that with one species-blind sigma: the open hosts are
# not even locally stable (an LJ diamond or simple-cubic site net
# reconstructs), and a planted interstitial overlapping a site's own atom at
# ~0.5 sigma carries a ~1e6 repulsion that expels the occupant and plants a
# vacancy that was never there.  The quench therefore minimizes the LJ energy
# RESTRAINED to the fitted lattice: every atom the fit matched to a site
# carries a harmonic spring to that site and is boxed to +/-
# thermal_quench_shift_fraction x d_NN around it (the box constraints of the
# L-BFGS-B solve), while off-site atoms -- the interstitials -- stay frozen
# (a free interstitial would relax into a dumbbell whose atoms sit within one
# site's tolerance and the defect would vanish from the diff).  The
# constraints, not the minimizer's convergence, pin the topology: every free
# atom ends within sqrt(3) x shift_fraction x d_NN of its site, inside the
# site-match tolerance, whatever a planted defect does to its neighbourhood.
# The site-match tolerance itself is anchored to the FITTED site lattice's
# nearest-neighbour distance, mirroring the dnn_lattice argument of
# group_defects one function down.


def _lj_energy_forces(r, L, rc):
    """Lennard-Jones energy and forces (epsilon = sigma = 1, minimum image).

    Self-contained twin of realize.lj.LJ.forces plus the energy the minimizer
    needs; the neighbour list is rebuilt on every evaluation (no skin logic),
    which keeps the quench deterministic under atom reordering.  The cutoff is
    capped below half the smallest box edge: mic distances are only the true
    minimum-image separation below L/2, and a quench of a small cell (2x2x2
    rocksalt with rc = 2.5 sigma > L/2) would otherwise read wrong distances."""
    L = np.asarray(L, float)
    rc = min(float(rc), 0.5 * float(L.min()) - 1e-6)  # dialect-exempt: numerical-guard: minimum-image validity
    pos = _wrap_strict(r, L)
    pairs = cKDTree(pos, boxsize=L).query_pairs(rc, output_type="ndarray")
    d = mic(pos[pairs[:, 1]] - pos[pairs[:, 0]], L)
    r2 = np.einsum("ij,ij->i", d, d)
    inv2 = 1.0 / r2                      # dialect-exempt: numerical-guard: reciprocal one
    inv6 = inv2 ** 3                     # dialect-exempt: exact-geometry: LJ r^-6
    energy = 4.0 * float((inv6 * inv6 - inv6).sum())  # dialect-exempt: exact-geometry: LJ 4eps
    fs = 24.0 * inv2 * inv6 * (2.0 * inv6 - 1.0)     # dialect-exempt: exact-geometry: LJ pair force
    fij = fs[:, None] * d
    forces = np.empty_like(pos)
    n = len(pos)
    for k in range(3):
        forces[:, k] = (np.bincount(pairs[:, 1], fij[:, k], n)
                        - np.bincount(pairs[:, 0], fij[:, k], n))
    return energy, forces


def _quench_thermal(frame: Frame, sites, dnn_lattice: float, dialect) -> Frame:
    """Relax thermal noise out of a copy of the frame before the Wigner-Seitz
    diff: on-site atoms fall back onto their sites, interstitials stay frozen
    in their interstices, vacancies stay empty (see the section comment).

    L-BFGS-B minimization of the LJ energy (reduced units; sigma anchored to
    the pair minimum at the lattice spacing -- the site restraints absorb the
    few-percent lattice-sum pressure difference, so the exact equilibrium
    anchor the free minimization needed is unnecessary here) plus harmonic
    springs from every matched atom to its site, under per-axis box bounds
    around each site.  Returns the input frame unchanged when there is nothing
    to relax, the dialect disables the quench (steps 0), or the minimizer
    fails (best effort: the unquenched diff is the historical path)."""
    tol = float(dialect.threshold("site_match_tol_fraction")) * dnn_lattice
    steps = int(dialect.threshold("thermal_quench_steps"))
    if steps <= 0:
        return frame
    L = frame.cell_diag
    pos = _wrap_strict(frame.pos, L)
    sites_w = _wrap_strict(sites, L)
    d_atom, i_site = cKDTree(sites_w, boxsize=L).query(pos)
    free = d_atom < tol
    if not free.any():
        return frame
    sigma = dnn_lattice / 2 ** (1 / 6)   # dialect-exempt: exact-geometry: LJ pair-minimum anchor
    rc = float(dialect.threshold("thermal_quench_rc"))    # LJ cutoff, sigma units
    spring = float(dialect.threshold("thermal_quench_spring"))
    shift = float(dialect.threshold("thermal_quench_shift_fraction")) * dnn_lattice / sigma
    L_red = L / sigma
    r = pos / sigma
    # spring anchor: the matched site of a free atom, the atom itself if frozen
    anchors = np.where(free[:, None], sites_w[i_site], pos) / sigma
    frozen = r[~free].copy()

    def objective(x):
        rr = np.empty_like(r)
        rr[free] = x.reshape(-1, 3)
        rr[~free] = frozen
        energy, forces = _lj_energy_forces(rr, L_red, rc)
        disp = mic(rr - anchors, L_red)
        energy = energy + 0.5 * spring * float((disp[free] ** 2).sum())  # dialect-exempt: exact-geometry: harmonic energy 1/2 k x^2
        grad = forces - spring * free[:, None] * disp
        return energy, -grad[free].ravel()

    lower = (anchors[free] - shift).ravel()
    upper = (anchors[free] + shift).ravel()
    try:
        from scipy.optimize import minimize
        result = minimize(objective, r[free].ravel(), jac=True, method="L-BFGS-B",
                          bounds=list(zip(lower, upper)),
                          options={"maxiter": steps,
                                   "maxfun": 20 * steps})  # dialect-exempt: numerical-guard: line-search evaluation cap
        if not np.all(np.isfinite(result.x)):
            raise ValueError("quench produced non-finite positions")
    except Exception:
        return frame
    out = np.empty_like(r)
    out[free] = result.x.reshape(-1, 3)
    out[~free] = frozen
    return Frame(pos=_wrap_strict(out * sigma, L), cell=frame.cell,
                 symbols=frame.symbols, pbc=frame.pbc, info=frame.info)


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
    on a basis site up to its own displacement, so EVERY anchor atom is a
    valid seed; a wrong-sublattice seed loses on the species gate. The
    candidates are the anchor atoms' positions reduced modulo the lattice
    constant and deduplicated (one representative per offset cluster), plus
    the origin -- a deterministic, content-fixed set that any rigid
    translation of the frame maps onto itself. Returns the seed scoring best
    at the stated lattice estimate, or None when no seed brings sites near
    atoms."""
    return _best_offset(name, a0, frame, eff_slots, species_aware, tree, tol)


def _offset_pool(pos: np.ndarray, a: float, radius: float) -> list[np.ndarray]:
    """One representative per mod-a cluster of the anchor atoms' offsets.

    Two atoms of the anchor species on the same sublattice differ by a
    lattice vector plus their mutual thermal displacement, so their offsets
    mod a sit within a jitter-sized cluster; atoms on different sublattices
    are at least a/2 apart. Keeping one lexorder-fixed representative per
    cluster (dedup radius = the site tolerance) covers every distinct
    sublattice offset with a handful of candidates, whatever the frame's
    translation or atom ordering -- the seed itself may change within a
    cluster, but every member is a valid seed and the caller's
    mean-displacement correction removes the chosen seed's own displacement
    (see `_canonical_shifts`)."""
    offs = np.mod(np.asarray(pos, float), a)
    order = np.lexsort((offs[:, 2], offs[:, 1], offs[:, 0]))
    kept: list[np.ndarray] = []
    for idx in order:
        o = offs[idx]
        dup = False
        for k in kept:
            d = o - k
            d -= a * np.round(d / a)
            if float(np.linalg.norm(d)) <= radius:
                dup = True
                break
        if not dup:
            kept.append(o)
    return kept


def _best_offset(name: str, a: float, frame: Frame, eff_slots: tuple,
                 species_aware: bool, tree, tol: float):
    """Best site offset at the FIXED lattice constant `a` over the
    deterministic candidate set (the origin plus every mod-a offset cluster
    of the anchor species), scored by species-correct site coverage then by
    mean matched distance -- the same key the scan ranks candidates with
    (O1 root cause b: the previous 4 lexsorted seeds could miss the lattice
    translation entirely; the pool cannot, every cluster IS a translation
    the frame declares)."""
    sites0, s_sp = ideal_sites(name, a, frame.cell, eff_slots)
    sym_arr = np.array(frame.symbols)
    site_species = np.array(s_sp)
    if species_aware:
        mask = sym_arr == s_sp[0]
        pos = frame.pos[mask] if mask.any() else frame.pos
    else:
        pos = frame.pos
    candidates = [np.zeros(3)] + _offset_pool(pos, a, tol)
    L = frame.cell_diag
    best = None
    for s0 in candidates:
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


def _tiling_constants(a_lo: float, a_hi: float, frame: Frame, dialect):
    """Lattice constants that tile the periodic box an integer number of
    times per axis within the scan band, in the builder's own cell-contract
    tolerance (`lattice_match_tolerance`, one definition on both sides,
    rule 7).

    `snap_a_to_cell` snaps one scanned constant to the box; this enumerates
    EVERY plausible tiling inside the band so a drifted scan optimum cannot
    hide the true one (O1 root cause c: the scan accepted constants that
    cannot tile the box, and Chaord's own builder refused the program). The
    box lengths are exact rigid invariants, so the shortlist is itself
    invariant under translation, rotation and re-imaging by construction."""
    from itertools import product
    L = np.asarray(frame.cell_diag, float)
    tol_tile = float(dialect.threshold("lattice_match_tolerance"))
    out = []
    ranges = []
    for i in range(3):
        r_min = max(1, int(np.floor(L[i] / a_hi)))
        r_max = int(np.ceil(L[i] / a_lo))
        ranges.append(range(r_min, r_max + 1))
    for reps in product(*ranges):
        per_axis = L / np.array(reps, float)
        a = float(per_axis.mean())
        # same single-constant test snap_a_to_cell applies (cubic tiling)
        if a_lo <= a <= a_hi and float(np.max(np.abs(per_axis - a))) <= tol_tile:
            out.append(a)
    return sorted(set(out))


def _canonical_shifts(name: str, a: float, frame: Frame, eff_slots: tuple,
                      species_aware: bool, shift: np.ndarray,
                      tree, tol: float) -> list[np.ndarray]:
    """[shift + mean-displacement correction, shift]: canonical form first.

    A seed atom carries its own thermal displacement, which offsets the
    whole site lattice; the minimum-image mean over every matched site-atom
    pair is the same information the quench springs use, averaged over the
    frame. The corrected offset is independent of WHICH valid seed started
    the chain (the seed's own displacement appears in both its site lattice
    and the mean, and cancels), so the final site lattice is a canonical
    function of the frame content -- the property that makes the lift
    byte-identical under arbitrary rigid translation."""
    sites, s_sp = _anchored_sites(name, a, frame, eff_slots, shift)
    d_ref, i_ref = tree.query(sites)
    near_ref = d_ref < tol
    if species_aware:
        near_ref = near_ref & (np.array(frame.symbols)[i_ref]
                               == np.array(s_sp))
    if not near_ref.any():
        return [shift]
    corr = mic(frame.pos[i_ref[near_ref]] - sites[near_ref],
               frame.cell_diag).mean(axis=0)
    return [shift + corr, shift]


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
    from ase.data import chemical_symbols
    if not all(s in chemical_symbols for s in uniq):
        raise ChaordError(
            "no cubic prototype fits the frame: placeholder species "
            f"{sorted(set(uniq) - set(chemical_symbols))} have no atomic "
            "numbers for the prototype fit")
    z_of = {s: chemical_symbols.index(s) for s in uniq}
    for name, n_slots in multi.items():
        if n_slots != len(uniq):
            continue
        wc = want_counts[name]
        w_total = sum(wc)
        reps = max(1, round(total / w_total))
        # each slot takes a species whose count matches THAT slot's ratio
        # (F1: the previous per-species `any(w in wc)` check let a 3:1
        # minority species onto the majority slot and an O onto a
        # single-count slot, so noise could pick antisite-riddled
        # assignments -- Al3Ni with 686 Ni_Al, OTiSr3)
        viable = [p for p in set(permutations(pops))
                  if all(abs(count_of[s] - wc[i] * reps) <= slack * wc[i] * reps
                         for i, s in enumerate(p))]
        if not viable:
            continue
        if len(set(wc)) < len(wc):
            # tied slots: the permutations are the same crystal up to origin
            # (each anchors on its own sublattice), so NO fit score can
            # separate them -- the scan's mean-distance noise decided and the
            # composition line flipped under rotation (NaCl -> ClNa). The
            # text follows the registry's notation instead: the prototype's
            # default slots when the species set is the default one,
            # otherwise ascending atomic number (the exact-crystal engine's
            # own tie convention).
            default = PROTOTYPES[name].default_slots
            if set(default) == set(uniq):
                viable = [default]
            else:
                viable = [min(viable, key=lambda p: tuple(z_of[s] for s in p))]
        candidates.extend((name, p) for p in viable)

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
                return -1.0, np.inf, None, None  # dialect-exempt: numerical-guard: impossible-gate sentinel
            d2, i2 = tree.query(sites2)
            near2 = d2 < tol
            if float(near2.mean()) < float(dialect.threshold("lattice_fit_gate_min")):
                return -1.0, np.inf, None, None  # dialect-exempt: numerical-guard: impossible-gate sentinel
            sites2_wrapped = np.minimum(np.mod(sites2, frame.cell_diag),
                                       frame.cell_diag * (1 - 1e-9))  # dialect-exempt: numerical-guard: strict upper edge
            d_atom2, _ = cKDTree(sites2_wrapped, boxsize=frame.cell_diag).query(
                np.minimum(pos_wrapped, frame.cell_diag * (1 - 1e-9)))  # dialect-exempt: numerical-guard: strict upper edge
            if float((d_atom2 < tol).mean()) < float(
                    dialect.threshold("lattice_fit_gate_min")):
                return -1.0, np.inf, None, None  # dialect-exempt: numerical-guard: impossible-gate sentinel
            if species_aware:
                # the gate a species-aware candidate is ranked by counts
                # SPECIES-CORRECT sites only: the blind fraction saturates at
                # 1.0 for every anchored permutation, so the wrong assignment
                # (Al on the 3:1 majority slot, 686 Ni_Al antisites) tied the
                # correct one and the scan's distance noise picked the text
                # (F1: composition Ni3Al -> Al3Ni under rotation)
                sym_arr = np.array(frame.symbols)
                near2 = near2 & (sym_arr[i2] == np.array(s_sp2))
            return (float(near2.mean()),
                    float(d2[near2].mean()) if near2.any() else np.inf,
                    sites2, s_sp2)

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
                    _g, qc, sc, spc = quality(aaa, shift)
                    if qc < mean_d:
                        aa, mean_d, sites, s_sp = aaa, qc, sc, spc
                half = refine_half / (4 ** (_ + 1))
            # gate/mean/sites re-measured together at the final constant, so
            # the candidate the caller ranks is the one it would rebuild
            g_fin, mean_fin, sites_fin, sp_fin = quality(aa, shift)
            if sites_fin is not None:
                return (g_fin, mean_fin, aa, sites_fin, sp_fin)
            return (-_neg_gate, mean_d, aa, sites, s_sp)

        # ---- O1 tiling-first fit (Review 6) --------------------------------
        # When the box declares its lattice (an integer tiling inside the
        # d_NN-plausible band), the candidate constants ARE those tilings and
        # the anchor comes from the deterministic all-atom offset pool, then
        # canonicalised by the mean matched displacement. Both the constant
        # shortlist and the canonical offset are functions of rigid
        # invariants alone, so the fitted (constant, site lattice) pair --
        # and the lifted text -- cannot follow a translation or re-imaging
        # of the input, whatever the scan optimum would have been (the
        # origin-anchored chain below is kept for boxes no constant tiles,
        # e.g. clipped region slabs, whose frames legitimately keep the
        # scanned value and the builder's honest refusal).
        best_tiling = None
        for aa in _tiling_constants(lo * a0, hi * a0, frame, dialect):
            shift = _best_offset(name, aa, frame, eff_slots, species_aware,
                                 tree, tol)
            if shift is None:
                continue
            for sh in _canonical_shifts(name, aa, frame, eff_slots,
                                        species_aware, shift, tree, tol):
                g_t, mean_t, sites_t, s_sp_t = quality(aa, sh)
                if sites_t is None:
                    continue        # corrected form first, raw seed as backup
                key = (g_t, -mean_t)
                if best_tiling is None or key > best_tiling[0]:
                    best_tiling = (key, aa, g_t, mean_t, sites_t, s_sp_t)
                break
        if best_tiling is not None:
            _key, aa, gate_t, mean_t, sites_t, s_sp_t = best_tiling
            return (aa, gate_t, sites_t, s_sp_t, mean_t)

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
            chosen, chosen_shift = reanchored, seed
        elif (reanchored is not None
              and reanchored[1] <= improvement * origin[1]):
            chosen, chosen_shift = reanchored, seed
        else:
            chosen, chosen_shift = origin, np.zeros(3)
        if chosen is None:
            return None
        gate, mean_d, aa, sites, s_sp = chosen
        # F1 cell contract (reports/redteam_findings.md): the program states
        # its build box, so when the box is an integer tiling of one lattice
        # constant that constant IS the printed `a` -- a function of the box
        # alone, exactly invariant under rigid transforms (the scan optimum
        # is not: it drifted up to -4.3 % under rotation) and never in
        # contradiction with the `cell` line the same program states. Sites
        # are regenerated at the snapped constant so the Wigner-Seitz diff
        # sees the lattice the program declares; frames whose box is not an
        # integer tiling (clipped region slabs) keep the scanned value.
        a_snapped = snap_a_to_cell(aa, frame.cell_diag, dialect)
        if a_snapped is not None and a_snapped != aa:
            gate2, mean_d2, sites2, s_sp2 = quality(a_snapped, chosen_shift)
            if sites2 is not None and mean_d2 < np.inf:
                gate, mean_d, aa, sites, s_sp = gate2, mean_d2, a_snapped, sites2, s_sp2
        if not np.array_equal(chosen_shift, np.zeros(3)):
            # a re-anchored (re-imaged) frame seeds the site offset from ONE
            # atom's jittered position: that atom's own displacement offsets
            # the whole site lattice and marginal atoms flip on/off their
            # sites, so the same crystal read spurious frenkel pairs under
            # rotation (F1, rule 3). Refine the offset by the minimum-image
            # mean displacement of every matched atom-site pair -- the same
            # information the quench springs use, averaged over the frame.
            d_ref, i_ref = tree.query(sites)
            near_ref = d_ref < tol
            if species_aware:
                near_ref = near_ref & (np.array(frame.symbols)[i_ref]
                                       == np.array(s_sp))
            if near_ref.any():
                corr = mic(frame.pos[i_ref[near_ref]] - sites[near_ref],
                           frame.cell_diag).mean(axis=0)
                gate3, mean_d3, sites3, s_sp3 = quality(
                    aa, chosen_shift + corr)
                if sites3 is not None and mean_d3 < np.inf:
                    gate, mean_d, sites, s_sp = gate3, mean_d3, sites3, s_sp3
        return (aa, gate, sites, s_sp, mean_d)

    # candidates are ranked by the species-aware gate (fraction of sites
    # matched with their OWN slot's species), exact ties broken by the mean
    # matched distance. Tied-slot permutations never reach this comparison:
    # they are origin-equivalent descriptions of one crystal (see the
    # candidate construction above), so their scores tie only within noise
    # and the noise -- not the frame -- decided the composition line (F1).
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
    """Classify every atom and site: vacancies, antisites, interstitials.

    Thermal quench first (review 2 root-cause fix, see the section comment):
    the site-match tolerance is anchored to the FITTED site lattice's own
    nearest-neighbour distance -- never to a distance measured on the atom
    cloud, which thermal collisions and planted interstitials shrink exactly
    where the diff must not get tighter (the same contamination argument as
    group_defects' dnn_lattice below)."""
    L = frame.cell_diag
    sites_w = _wrap_strict(sites, L)
    site_tree = cKDTree(sites_w, boxsize=L)
    d_site_nn, _ = site_tree.query(sites_w, k=2)
    d_nn = float(d_site_nn[:, 1].min())
    frame = _quench_thermal(frame, sites_w, d_nn, dialect)
    tol = float(dialect.threshold("site_match_tol_fraction")) * d_nn
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
