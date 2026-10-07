"""Solid-liquid slab lifter: the M0 decompiler path.

Ported from prototype/decompile.py. Every threshold is read from the dialect by
name; there are no fitted constants in this file. The lift produces a result
dictionary (observables, defects, statistics) plus a canonical Program built
through the IR, so all text goes through `chaord fmt`'s printer.

Frames of one species lift exactly as the M0 LJ path always did (byte-identical
text; pass ``symbols=None`` or a single-species list) -- except the W4 step-2
arm (Review 8): a single-species frame whose species is a real element under
the metal dialect carries species and units through (Cu named in conserve
atoms, A / eV / K statements, backend eam; see _program_physical_single). A
multi-species frame (``symbols=`` naming several elements, e.g. a Cu slab
under water) fits the crystal on the majority solid species' sublattice and
describes the liquid by a molecular census (build.molecules bond graph) --
the arms share every threshold and the thin-film degradation rule below.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from ..io.frames import Frame
from ..lang.errors import ChaordError
from ..lang.ir import (
    InterfaceBlock, PhysicsBlock, Program, ProvenanceBlock, Quantity,
    RangeVal, RegionBlock, ResidualBlock, ShSlab, Statement, SystemBlock,
)
from ..realize.lj import mic, wrap
from .passes import pairs_within, qbar

# u/A^3 -> g/cm3 (NA / 1e24): the exact conversion constant used by the fluid
# lifter; physical data, not a tunable threshold (dialect-exempt per AGENTS.md)
_AMU_PER_A3_TO_G_CM3 = 0.6022140857  # dialect-exempt: exact-geometry: unit conversion


def _mz(dz, Lz):
    return dz - Lz * np.round(dz / Lz)


def _circ_offset(x, d):
    z = np.mean(np.exp(2j * np.pi * x / d))
    return abs(z), d * np.angle(z) / (2 * np.pi)


def lift_defects(E, O, L, zcr, Hc, dialect):
    """Group empty sites (E) and off-lattice atoms (O) into defect complexes.

    Net vacancies = empty sites - displaced atoms (atom count is conserved)."""
    rc = float(dialect.threshold("defect_cluster_rc"))
    P = np.r_[E, O]
    n = len(P)
    if n == 0:
        return []
    kind = np.r_[np.zeros(len(E), int), np.ones(len(O), int)]
    Dm = np.linalg.norm(mic(P[:, None, :] - P[None, :, :], L), axis=2)
    _, lab = connected_components(coo_matrix(Dm < rc), directed=False)
    out = []
    for c in np.unique(lab):
        m = lab == c
        ne, no = int((kind[m] == 0).sum()), int((kind[m] == 1).sum())
        depth = float(np.mean(Hc - np.abs(_mz(P[m, 2] - zcr, L[2]))))
        net = ne - no
        name = {1: "vacancy", 2: "divacancy", 3: "trivacancy"}.get(
            net, f"V{net}" if net > 0 else ("frenkel_pair" if net == 0 else "interstitial"))
        out.append(dict(name=name, net=net, sites=ne, displaced=no, depth=depth, xyz=P[m]))
    return sorted(out, key=lambda d: (-d["net"], d["depth"]))


def orientation(bvec, dialect):
    """Compare |cos| of bond directions with an axis to ideal fcc <110> bond sets."""
    ideal = dialect.threshold("orientation_ideals")
    edges = [float(x) for x in dialect.threshold("bond_angle_bins")]
    h = np.histogram(np.abs(bvec), edges)[0]
    h = h / h.sum()
    best = min(ideal, key=lambda k: np.abs(h - np.asarray(ideal[k], float)).sum())
    return "<" + best[1:-1] + ">"


def chi0_mean(r, mask, pairs, L, dialect):
    """Mean antiparallel neighbour-pair count over the masked atoms -- the
    fcc/hcp discriminator (fcc holds 6 antiparallel pairs per atom, hcp 3;
    thermal motion lowers both, so the dialect's chi0_fcc_min /
    chi0_hcp_min must be calibrated on thermal frames -- see metal.yaml).
    Extracted from decompile so the discriminator is testable on its own."""
    nbr = [[] for _ in range(len(r))]
    for a_, b_ in pairs:
        nbr[a_].append(b_)
        nbr[b_].append(a_)
    chi_cos = float(dialect.threshold("chi0_antiparallel_cos"))
    stride = int(dialect.threshold("chi0_sample_stride"))
    chi0 = []
    for i in np.where(mask)[0][::stride]:
        v = mic(r[nbr[i]] - r[i], L)
        v /= np.linalg.norm(v, axis=1)[:, None]
        c = v @ v.T
        chi0.append(np.sum(np.triu(c, 1) < chi_cos))
    return float(np.mean(chi0))


def _empty_angle_hist(dialect):
    """Zeroed bond-angle histogram (molecular liquids describe angles per
    molecule template, not by a histogram; the key stays present for
    downstream consumers that expect the array)."""
    return np.zeros(int(180 / float(dialect.threshold("angle_hist_bin"))))


def _liquid_bulk_windows(Hl, margin, rmax, floor, fraction):
    """Effective (margin, g(r) range, note) for liquid bulk statistics.

    The centre statistics zone |dzl| < Hl - margin - rmax must be non-empty.
    A film thinner than margin + range has no such zone at full settings --
    instead of crashing, both insets shrink proportionally until they cover
    only `fraction` of the film half-height (dialect key
    thin_slab_margin_fraction); the g(r) range never drops below the
    coordination cutoff `floor` (the first shell must stay in range); if the
    floor still does not fit, the margin drops to zero. A film not even one
    coordination cutoff thick has no bulk statistics at all -- that is a
    ChaordError (the caller cannot honestly describe the liquid).

    Returns (margin, rmax, note); note is None when no shrink happened, so the
    single-species text of a film that fits is byte-identical to before.
    """
    if Hl - margin - rmax > 0:
        return margin, rmax, None
    scale = fraction * Hl / (margin + rmax)
    m_eff = scale * margin
    r_eff = max(scale * rmax, floor)
    if Hl - m_eff - r_eff <= 0:
        m_eff = 0
        if Hl - r_eff <= 0:
            raise ChaordError(
                f"liquid film too thin for bulk statistics: half-height "
                f"{Hl:.2f} below the minimum g(r) range {floor:.2f} "
                f"(dialect cn_cutoff); no honest liquid statements exist")
    note = (f"thin-film liquid: bulk margin shrunk {margin:.2f} -> {m_eff:.2f}, "
            f"g(r) range {rmax:.2f} -> {r_eff:.2f}")
    return m_eff, r_eff, note


def decompile(r, L, T, dialect, symbols=None):
    """Lift passes 1-3 for a solid-liquid slab. Returns the result dictionary.

    `symbols` (optional) names the species per atom. A single-species (or
    absent) list runs the M0 path unchanged; several species fit the crystal on
    the majority solid species' sublattice and census the liquid molecularly.
    """
    N = len(r)
    Lx, Ly, Lz = L
    q6_rc = float(dialect.threshold("q6_cutoff"))
    q6_solid = float(dialect.threshold("q6_solid"))
    margin = float(dialect.threshold("bulk_margin"))
    q6, cn, p = qbar(r, L, rc=q6_rc)
    solid = q6 > q6_solid

    # ---- multi-species frames: the crystal species owns the lattice fit ---------------
    syms = list(symbols) if symbols is not None else None
    if syms is not None and len(set(syms)) > 1:
        votes: dict[str, int] = {}
        for i in np.where(solid)[0]:
            votes[syms[i]] = votes.get(syms[i], 0) + 1
        if not votes:                      # no solid-like atom: fall back to the census
            votes = {s: syms.count(s) for s in set(syms)}
        crystal_species = max(votes, key=votes.get)
        cmask = np.array([s == crystal_species for s in syms], bool)
    else:
        crystal_species = None
        cmask = np.ones(N, bool)
    multi = crystal_species is not None

    # ---- pass 1: solid-fraction profile and phase segmentation along z -----------------
    # the profile counts the crystal species only (a molecular phase has no
    # crystal-species atoms, so its bins read liquid); with one species the
    # mask is all-true and the numbers are exactly the M0 ones
    binw = float(dialect.threshold("profile_bin_size"))
    kw = int(dialect.threshold("profile_smooth_wide"))
    kn = int(dialect.threshold("profile_smooth_narrow"))
    frac = float(dialect.threshold("phase_solid_fraction"))
    nb = int(Lz / binw)
    edges = np.linspace(0, Lz, nb + 1)
    zc = 0.5 * (edges[1:] + edges[:-1])  # dialect-exempt: numerical-guard: bin centre arithmetic
    idx = np.minimum((r[:, 2] / Lz * nb).astype(int), nb - 1)
    nsol = np.bincount(idx, (solid & cmask).astype(float), nb)
    ntot = np.bincount(idx, cmask.astype(float), nb)
    sm = lambda a, k: np.convolve(np.r_[a[-k:], a, a[:k]], np.ones(k), "same")[k:-k]
    ph = sm(nsol, kw) / np.maximum(sm(ntot, kw), 1e-9)  # dialect-exempt: numerical-guard: divide-by-zero guard
    if multi:
        # sub-lattice profile: between crystal-species planes (spacing >> bin
        # width, e.g. Cu(100) at 1.8 A vs 0.25 A bins) whole bins hold no
        # crystal-species atom; their 0/0 must not read as "liquid". A bin
        # occupied by other species is genuinely liquid-side (0); a bin empty
        # of every species is a sub-lattice gap and inherits the nearest
        # defined value.
        occ = np.bincount(idx, minlength=nb).astype(float)
        defined = sm(ntot, kw) > 0
        liquid_side = (sm(occ, kw) > 0) & ~defined
        ph = np.where(liquid_side, np.zeros_like(ph), ph)
        missing = ~defined & ~liquid_side
        if defined.any():
            d = np.where(defined)[0]
            nearest = d[np.abs(np.arange(nb)[:, None] - d[None, :]).argmin(1)]
            ph = np.where(missing, ph[nearest], ph)
        elif not liquid_side.any():
            raise ChaordError(
                "no solid-like crystal-species atoms: the frame is not a "
                "crystal-liquid slab")
    tot3 = sm(ntot, kn)
    phi = sm(nsol, kn) / np.maximum(tot3, 1e-9)         # dialect-exempt: numerical-guard: divide-by-zero guard
    on = np.r_[ph > frac, ph > frac]
    best = (0, 0)
    s = None
    for i, b in enumerate(on):
        if b and s is None:
            s = i
        if (not b or i == len(on) - 1) and s is not None:
            n = i - s + (1 if b else 0)
            if n > best[1] and n <= nb:
                best = (s, n)
            s = None
    s, n = best
    guess_lo, guess_up = edges[s % nb], edges[(s + n) % nb]
    fit_win = float(dialect.threshold("interface_fit_window"))
    off_b = [float(x) for x in dialect.threshold("interface_fit_offset_bounds")]
    wid_b = [float(x) for x in dialect.threshold("interface_fit_width_bounds")]
    # tanh-fit data mask: crystal-species bins, plus (multi-species) bins the
    # other species occupies -- their phi is a genuine liquid-side 0, without
    # them the fit at an interface has no liquid-side data to anchor it
    if multi:
        fit_occ = np.maximum(tot3, sm(occ, kn))
    else:
        fit_occ = tot3

    def fit(guess, sign):
        dz = _mz(zc - guess, Lz)
        m = (np.abs(dz) < fit_win) & (fit_occ > 0)
        f = lambda z, z0, w: 0.5 * (1 + sign * np.tanh((z - z0) / w))  # dialect-exempt: exact-geometry
        (z0, w), _ = curve_fit(f, dz[m], phi[m], p0=[0, 1],
                               bounds=([off_b[0], wid_b[0]], [off_b[1], wid_b[1]]))
        return np.mod(guess + z0, Lz), dialect.threshold("interface_width_factor") * w

    z_lo, w_lo = fit(guess_lo, +1)
    z_up, w_up = fit(guess_up, -1)
    Hc = np.mod(z_up - z_lo, Lz) / 2
    zcr = np.mod(z_lo + Hc, Lz)
    Hl = Lz / 2 - Hc
    zliq = np.mod(zcr + Lz / 2, Lz)
    dzc = _mz(r[:, 2] - zcr, Lz)
    dzl = _mz(r[:, 2] - zliq, Lz)
    # ---- pass 2: crystal region -> exact lattice + defects ------------------------------
    site_tol = float(dialect.threshold("site_match_tolerance"))
    zone_pad = float(dialect.threshold("site_zone_pad"))
    # the crystal fit runs on the crystal species' sublattice (all atoms when
    # the frame is single-species)
    ci = (np.abs(dzc) < Hc - max(margin, max(w_lo, w_up) + q6_rc)) & cmask
    hmin = int(dialect.threshold("lattice_harmonics_min"))
    hmax = int(dialect.threshold("lattice_harmonics_max"))
    kx = max(range(hmin, hmax), key=lambda k: _circ_offset(r[ci, 0], Lx / k)[0])
    dx = Lx / kx
    ky = max(range(hmin, hmax), key=lambda k: _circ_offset(r[ci, 1], Ly / k)[0])
    dy = Ly / ky
    scan_lo, scan_hi, scan_step = (float(x) for x in dialect.threshold("lattice_spacing_scan"))
    ds = np.arange(scan_lo, scan_hi, scan_step)
    dz = ds[np.argmax([_circ_offset(dzc[ci], d)[0] for d in ds])]
    ox, oy, oz = _circ_offset(r[ci, 0], dx)[1], _circ_offset(r[ci, 1], dy)[1], _circ_offset(dzc[ci], dz)[1]
    ii = np.round((r[ci, 0] - ox) / dx).astype(int)
    jj = np.round((r[ci, 1] - oy) / dy).astype(int)
    kk = np.round((dzc[ci] - oz) / dz).astype(int)
    par = np.bincount((ii + jj + kk) % 2, minlength=2).argmax()
    kmax = int((Hc - margin) / dz) + 2
    g = np.array(np.meshgrid(range(kx), range(ky), range(-kmax, kmax + 1), indexing="ij")).reshape(3, -1).T
    g = g[(g.sum(1) % 2) == par]
    sites = np.c_[ox + g[:, 0] * dx, oy + g[:, 1] * dy, zcr + oz + g[:, 2] * dz]
    zone = Hc - max(margin, max(w_lo, w_up) + q6_rc)
    sites_ext = sites[np.abs(_mz(sites[:, 2] - zcr, Lz)) < zone + zone_pad]
    sites = sites[np.abs(_mz(sites[:, 2] - zcr, Lz)) < zone]
    tree = cKDTree(wrap(r, L), boxsize=L)
    dist, _ = tree.query(wrap(sites, L))
    empty = dist > site_tol
    vac_depth = np.sort(Hc - np.abs(_mz(sites[empty, 2] - zcr, Lz)))
    E = sites[empty]
    dsite, _ = cKDTree(wrap(sites_ext, L), boxsize=L).query(wrap(r[ci], L))
    off = np.where(ci)[0][dsite > site_tol]
    pi, pj = p[:, 0], p[:, 1]
    both = ci[pi] & ci[pj]
    bv = mic(r[pj[both]] - r[pi[both]], L)
    bv /= np.linalg.norm(bv, axis=1)[:, None]
    orient_z, orient_x = orientation(bv[:, 2], dialect), orientation(bv[:, 0], dialect)
    # Ackland-Jones-style 3-body check: antiparallel neighbour pairs (fcc: 6, hcp: 3)
    chi0 = chi0_mean(r, ci, p, L, dialect)
    fcc_min = float(dialect.threshold("chi0_fcc_min"))
    hcp_min = float(dialect.threshold("chi0_hcp_min"))
    ctype = "fcc" if chi0 > fcc_min else ("hcp" if chi0 > hcp_min else "other")
    # ---- pass 3: liquid region -> statistical statements --------------------------------
    # bulk-statistics windows: honest degradation when the film is too thin for
    # the full margin + g(r) range (see _liquid_bulk_windows)
    rmax = float(dialect.threshold("gr_rmax"))
    rcut = float(dialect.threshold("cn_cutoff"))
    if multi:
        from ..build.molecules import molecule_census
        lmask = ~cmask
        lsyms = list(np.array(syms)[lmask])
        lsub = Frame(pos=r[lmask], cell=np.diag(L), symbols=lsyms,
                     pbc=(True, True, True))
        census = molecule_census(lsub, dialect)
        from .fluid import _is_element
        liquid_molecular = any(len(nm) > 2 or not _is_element(nm) for nm in census)
        if liquid_molecular:
            from .fluid import _centers
            parts = _centers(lsub, dialect)      # molecule centres
            rmax0 = float(dialect.threshold("gr_rmax_fluid"))
            floor = float(dialect.threshold("cn_cutoff_fluid"))
            nbin = int(dialect.threshold("gr_bins_fluid"))
        else:
            parts = lsub.pos                     # atomic liquid of several elements
            rmax0, floor, nbin = rmax, rcut, int(dialect.threshold("gr_bins"))
        m_eff, r_eff, thin_note = _liquid_bulk_windows(
            Hl, margin, rmax0, floor,
            float(dialect.threshold("thin_slab_margin_fraction")))
    else:
        m_eff, r_eff, thin_note = _liquid_bulk_windows(
            Hl, margin, rmax, rcut,
            float(dialect.threshold("thin_slab_margin_fraction")))
        nbin = int(dialect.threshold("gr_bins"))

    if multi:
        # statistics on the liquid particles (molecule centres or atoms)
        dzp = _mz(parts[:, 2] - zliq, Lz)
        liq = np.abs(dzp) < Hl - m_eff
        cen = np.abs(dzp) < Hl - m_eff - r_eff
        if cen.sum() == 0:
            raise ChaordError(
                f"liquid region has no bulk statistics zone: {int(liq.sum())} "
                f"particles in a film of half-height {Hl:.2f}; cannot describe "
                f"the liquid honestly")
        ptree = cKDTree(wrap(parts, L), boxsize=L)
        pp = ptree.query_pairs(r_eff, output_type="ndarray")
        dd = np.linalg.norm(mic(parts[pp[:, 1]] - parts[pp[:, 0]], L), axis=1)
        rr = np.r_[dd[cen[pp[:, 0]]], dd[cen[pp[:, 1]]]]
        h, e = np.histogram(rr, np.linspace(0, r_eff, nbin + 1))
        rm = 0.5 * (e[1:] + e[:-1])  # dialect-exempt: numerical-guard: bin centre arithmetic
        rho_p = liq.sum() / (Lx * Ly * 2 * (Hl - m_eff))
        gr = h / (cen.sum() * rho_p * 4 / 3 * np.pi * (e[1:] ** 3 - e[:-1] ** 3))
        ipk = np.argmax(gr)
        nl = ptree.query_ball_point(wrap(parts[cen], L), floor)
        cnl = np.array([len(x) - 1 for x in nl])
        sl = np.where(liq)[0]
        sub = np.isin(pp[:, 0], sl) & np.isin(pp[:, 1], sl)
        loc = {a: b for b, a in enumerate(sl)}
        if len(sl):
            A = coo_matrix((np.ones(sub.sum()),
                            ([loc[a] for a in pp[sub, 0]],
                             [loc[b] for b in pp[sub, 1]])),
                           shape=(len(sl),) * 2)
            _, lab = connected_components(A, directed=False)
            sizes = np.sort(np.bincount(lab))[::-1]
        else:
            sizes = np.array([], int)
        if liquid_molecular:
            from ase.data import atomic_masses, chemical_symbols
            mass = sum(atomic_masses[chemical_symbols.index(s)] for s in lsyms)
            # density over the stated region geometry (the program's own slab:
            # a checker recomputes it from the region volume); the equivalent
            # half-box number is what the generators plant (0.9 g/cm3 here)
            liquid = dict(
                census=census,
                rho_g=mass / (Lx * Ly * 2 * Hl) / _AMU_PER_A3_TO_G_CM3,
                rho=None, cn=cnl.mean(), cn_sd=cnl.std(), rcut=floor,
                gr=gr, rm=rm, pk_r=rm[ipk], pk_h=gr[ipk],
                bad=_empty_angle_hist(dialect), clusters=sizes, molecular=True)
        else:
            liquid = dict(
                census=census, rho_g=None,
                rho=lmask.sum() / (Lx * Ly * 2 * Hl),
                cn=cnl.mean(), cn_sd=cnl.std(), rcut=rcut,
                gr=gr, rm=rm, pk_r=rm[ipk], pk_h=gr[ipk],
                bad=_empty_angle_hist(dialect), clusters=sizes, molecular=False)
    else:
        li = np.abs(dzl) < Hl - m_eff
        rho = li.sum() / (Lx * Ly * 2 * (Hl - m_eff))
        ang_bin = float(dialect.threshold("angle_hist_bin"))
        cen = np.abs(dzl) < Hl - m_eff - r_eff
        if cen.sum() == 0:
            raise ChaordError(
                f"liquid region has no bulk statistics zone: {int(li.sum())} "
                f"atoms in a film of half-height {Hl:.2f}; cannot describe "
                f"the liquid honestly")
        pp = tree.query_pairs(r_eff, output_type="ndarray")
        dd = np.linalg.norm(mic(r[pp[:, 1]] - r[pp[:, 0]], L), axis=1)
        rr = np.r_[dd[cen[pp[:, 0]]], dd[cen[pp[:, 1]]]]
        h, e = np.histogram(rr, np.linspace(0, r_eff, nbin + 1))
        rm = 0.5 * (e[1:] + e[:-1])  # dialect-exempt: numerical-guard: bin centre arithmetic
        gr = h / (cen.sum() * rho * 4 / 3 * np.pi * (e[1:] ** 3 - e[:-1] ** 3))
        ipk = np.argmax(gr)
        nl = tree.query_ball_point(wrap(r[cen], L), rcut)
        cnl = np.array([len(x) - 1 for x in nl])
        ang = []
        for c_i, lst in zip(np.where(cen)[0], nl):
            lst = [x for x in lst if x != c_i]
            v = mic(r[lst] - r[c_i], L)
            v /= np.linalg.norm(v, axis=1)[:, None]
            cc = (v @ v.T)[np.triu_indices(len(lst), 1)]
            ang.append(np.degrees(np.arccos(np.clip(cc, -1, 1))))
        bad = np.histogram(np.concatenate(ang), np.arange(0, 181, ang_bin))[0].astype(float)
        bad /= bad.sum()
        sl = np.where(li & solid)[0]
        sub = np.isin(pi, sl) & np.isin(pj, sl)
        loc = {a: b for b, a in enumerate(sl)}
        if len(sl):
            A = coo_matrix((np.ones(sub.sum()), ([loc[a] for a in pi[sub]], [loc[b] for b in pj[sub]])),
                           shape=(len(sl),) * 2)
            _, lab = connected_components(A, directed=False)
            sizes = np.sort(np.bincount(lab))[::-1]
        else:
            sizes = np.array([], int)
        liquid = dict(
            census=None, rho_g=None, rho=rho, cn=cnl.mean(), cn_sd=cnl.std(),
            rcut=rcut, gr=gr, rm=rm, pk_r=rm[ipk], pk_h=gr[ipk], bad=bad,
            clusters=sizes, molecular=False)

    species_counts: dict[str, int] = {}
    if syms is not None:
        for s in syms:
            species_counts[s] = species_counts.get(s, 0) + 1
    if multi:
        q6l = q6[lmask & (np.abs(dzl) < Hl - m_eff)].mean()
    else:
        q6l = q6[li].mean()
    defects = lift_defects(E, r[off], L, zcr, Hc, dialect)
    # W4 step 2: defect lines print only above the dialect's measured
    # detection floor. The floor is a metal-arm calibration (measured on
    # defect-free thermal fcc frames at 0.9-1.0 Tm, see metal.yaml
    # slab_defect_print_floor); dialects without the key keep printing
    # every measured complex (the lj contract, pinned by the golden tests).
    if _metal_single_species(syms, dialect) is not None:
        floor = int(dialect.threshold("slab_defect_print_floor"))
        defects = [c for c in defects if c["sites"] + c["displaced"] >= floor]
    return dict(
        N=N, L=L, T=T, z_lo=z_lo, z_up=z_up, w_lo=w_lo, w_up=w_up, Hc=Hc, Hl=Hl,
        zcr=zcr, zliq=zliq, ctype=ctype, chi0=chi0, a_x=2 * dx, a_y=2 * dy, a_z=2 * dz,
        orient_z=orient_z, orient_x=orient_x, nvac=int(empty.sum()), vac_depth=vac_depth,
        empty_xyz=E, defects=defects,
        match=1 - empty.mean(), off=off, off_xyz=r[off], q6l=q6l,
        x_solid_cr=solid[ci].mean(),
        # liquid statistics (single- and multi-species arms share the keys)
        **liquid,
        # thin-film degradation record (None when the full margins fit)
        thin_margin_l=m_eff, thin_rmax_l=r_eff, thin_note=thin_note,
        n_centre=int(cen.sum()),
        # multi-species record (None keys on the single-species path)
        crystal_species=crystal_species,
        species_counts=species_counts or None,
    )


# ---------------------------------------------------------------- program ----

def _q(num, unit=None):
    return Quantity(num=num, unit=unit)


def _s(kind, key, *values, comment=None):
    return Statement(kind=kind, key=key, values=list(values), comment=comment)


def _metal_single_species(syms, dialect):
    """Species name when this is the W4 step-2 arm: a single-species frame
    whose one species is a real element, under a dialect that carries metal
    units (an `eam_potentials` table lives only there). None for everything
    else: several species take the multi arm (it already names each element
    and prints Angstrom units), and `symbols=None` / the pseudo-species `X`
    are the byte-identical M0 contract."""
    from .fluid import _is_element
    if syms is None or len(set(syms)) != 1:
        return None
    only = syms[0]
    return only if (_is_element(only) and "metal" in dialect.names) else None


def _refuse_unsupported_metal_slab(d, dialect):
    """W4 gate (step 1), narrowed by step 2 to what is STILL unsupported.

    Step 2 carries species and units through the slab path (the species
    named in `conserve atoms`, Angstrom cells and lattice constant, Kelvin
    T stated or flagged assumed, `backend eam`), so the Cu solid-liquid
    reference lifts. Still refused, with the same message (fail closed:
    never print a program the lift cannot stand behind):

    1. a real-element frame under a dialect without metal units -- the
       program would state LJ/reduced units for a metal;
    2. a real element the dialect's eam_potentials does not parameterise --
       no metal physics exists to name in the physics block (the build's
       physics gate refuses the same species).

    Not refused: `symbols=None` and the pseudo-species `X` (the M0 contract
    -- the lift does not know the species), and multi-species interfaces
    (the multi arm names each element and prints Angstrom units)."""
    counts = d.get("species_counts")
    if d.get("crystal_species") is not None or not counts or len(counts) != 1:
        return
    only = next(iter(counts))
    from .fluid import _is_element
    if not _is_element(only):
        return
    if "metal" not in dialect.names:
        raise ChaordError(
            "single-species metal solid\u2013liquid interfaces are not "
            f"supported yet: species {only!r} under dialect "
            f"{dialect.version_string} would be printed in LJ units (only "
            "the metal dialect carries A/eV/K; open_items_v2 W4)")
    try:
        pots = dialect.threshold("eam_potentials") or {}
    except ChaordError:
        pots = {}
    if only not in pots:
        raise ChaordError(
            "single-species metal solid\u2013liquid interfaces are not "
            f"supported yet: species {only!r} has no EAM parameters in "
            f"the metal dialect (parameterised: {', '.join(sorted(pots)) or 'none'}); "
            "no metal physics exists to state in the program "
            "(open_items_v2 W4)")


def program_from_result(d, dialect, symbol="X"):
    """Build the canonical Program IR from a decompile() result dictionary."""
    _refuse_unsupported_metal_slab(d, dialect)
    if d.get("crystal_species") is not None:
        return _program_multi(d, dialect)
    metal_species = _metal_single_species(
        [*d["species_counts"]] if d.get("species_counts") else None, dialect)
    if metal_species is not None:
        return _program_physical_single(d, dialect, metal_species)
    return _program_single(d, dialect, symbol)


def _liquid_comment(d, wrap_comment):
    """Comment of the liquid block: the wrap note plus the visible thin-film
    degradation note (a shrunk statistics zone must be stated, not hidden)."""
    note = d.get("thin_note")
    if note is None:
        return wrap_comment
    if wrap_comment is None:
        return note
    return f"{wrap_comment}; {note}"


def _provenance(d, dialect):
    """Provenance statements shared by both program arms: dialects, lift
    version, and -- W11, exactly as the fluid path does it (fluid.AssumedT)
    -- the note that a caller-given-absent T is the dialect's default, not a
    measurement of this frame. A caller-given T carries no note."""
    stmts = [
        _s("build", "dialects", _sv(dialect.version_string)),
        _s("build", "lift_version", _sv("0.1.0")),
    ]
    from .fluid import AssumedT
    if isinstance(d.get("T"), AssumedT):
        stmts.append(_s("build", "note", _sv(
            f"T assumed (dialect default {float(d['T']):g}); "
            "not measured from the frame")))
    return stmts


def _program_single(d, dialect, symbol="X"):
    """Canonical program of a single-species slab (the M0 text, unchanged
    unless the film was too thin and the statistics zone had to shrink)."""
    L = d["L"]
    strain = 100 * (d["a_z"] / d["a_x"] - 1)
    ncl = int((d["clusters"] >= 2).sum())
    wrap_c = d["z_lo"] > d["z_up"]

    system_statements = [
        _s("build", "units", _n("lj")),
        _s("build", "cell", _q(f"{L[0]:.2f}"), _q(f"{L[1]:.2f}"), _q(f"{L[2]:.2f}")),
        _s("build", "pbc", _n("xyz")),
    ]
    if d.get("T") is not None:  # T is metadata; dialects without MD omit it
        system_statements.append(_s("state", "T", _q(f"{d['T']:.2f}")))
    system_statements.append(
        _s("conserve", "atoms", _n(symbol), _q(str(d["N"]))))
    system = SystemBlock(statements=system_statements)
    physics = PhysicsBlock(statements=[
        _s("build", "backend", _n("lj")),
        _s("build", "epsilon", _q("1")),
        _s("build", "sigma", _q("1")),
        _s("build", "cutoff", _q(str(dialect.threshold("printed_cutoff")))),
    ])
    crystal_stmts = [
        _s("build", "lattice", _n(d["ctype"])),
        _s("build", "a", _q(f"{d['a_x']:.3f}")),
        _s("build", "orient", _n(d["orient_x"].replace("<001>", "<100>")),
           _n("z"), _n(d["orient_z"])),
        _s("constrain", "strain", _n("zz"), _q(f"{strain:+.1f}", "%"), _tol(dialect.threshold("printed_strain_tolerance"))),
    ]
    for c in d["defects"]:
        vals = [_n(c["name"]), _n("count"), _q("1"), _n("depth"), _q(f"{c['depth']:.1f}")]
        if c["displaced"]:
            vals += [_n("form"), _n("split")]
        crystal_stmts.append(_s("build", "defect", *vals,
                               comment=f"{c['sites']} empty sites, {c['displaced']} displaced atoms"
                               if c["displaced"] else None))
    crystal_stmts.append(_s("assert", "sites_matched", _q(f"{100 * d['match']:.1f}", "%")))
    crystal = RegionBlock(
        phase="crystal", name="A",
        geometry=_geo_z(d["z_lo"], d["z_up"]),
        statements=crystal_stmts,
        comment="wraps through z = 0" if wrap_c else None)

    liquid = RegionBlock(
        phase="liquid", name="B",
        geometry=_geo_z(d["z_up"], d["z_lo"]),
        comment=_liquid_comment(d, "wraps through z = 0" if not wrap_c else None),
        statements=[
            _s("state", "density", _q(f"{d['rho']:.3f}")),
            _s("assert", "cn", _q(f"{d['cn']:.1f}"), _tol(d["cn_sd"]), _n("cutoff"),
               _q(f"{d['rcut']:.2f}")),
            _s("assert", "gr_peak", _q(f"{d['pk_r']:.2f}"), _n("height"), _q(f"{d['pk_h']:.2f}")),
            _s("assert", "solid_clusters", _q(str(ncl))),
        ])
    ifaces = [
        InterfaceBlock(a="A", b="B", statements=[
            _s("build", "at", _n("z"), _q(f"{d['z_up']:.1f}")),
            _s("build", "width", _q(f"{d['w_up']:.1f}")),
        ]),
        InterfaceBlock(a="B", b="A", statements=[
            _s("build", "at", _n("z"), _q(f"{d['z_lo']:.1f}")),
            _s("build", "width", _q(f"{d['w_lo']:.1f}")),
        ]),
    ]
    explained = sum(c["displaced"] for c in d["defects"])
    if len(d["off"]) - explained == 0:
        residual = ResidualBlock(none=True)
    else:
        stmts = [_s("build", "atom", _n(symbol), _q(f"{x:.2f}"), _q(f"{y:.2f}"), _q(f"{z:.2f}"))
                 for x, y, z in d["off_xyz"]]
        residual = ResidualBlock(none=False, statements=stmts)
    provenance = ProvenanceBlock(statements=_provenance(d, dialect))
    return Program(
        version="0.1",  # dialect-exempt: numerical-guard: language version constant
        dialects=list(dialect.names),
        blocks=[system, physics, crystal, liquid, *ifaces, residual, provenance],
    )


def _program_physical_single(d, dialect, symbol):
    """W4 step 2: the single-species metal slab program -- the M0 statement
    structure with species and units carried through: the species named in
    `conserve atoms`, the cell and lattice constant in Angstrom, T stated in
    Kelvin (from the caller, or the dialect default flagged assumed in
    provenance, exactly as the M0/W11 path does), and `backend eam` for the
    physics. Statement forms follow the metal convention of
    spec/examples/02_crystal_defects.chaord (trailing-A cell, `a ... A`,
    `state T ... K`); no `units` line is stated -- physical-unit programs
    mark each quantity, they do not declare a reduced unit system (the same
    rule as the multi-species arm)."""
    L = d["L"]
    strain = 100 * (d["a_z"] / d["a_x"] - 1)
    ncl = int((d["clusters"] >= 2).sum())
    wrap_c = d["z_lo"] > d["z_up"]
    unit_a = "A"   # physical-unit dialects state lengths in Angstrom

    system_statements = [
        _s("build", "cell", _q(f"{L[0]:.3f}"), _q(f"{L[1]:.3f}"),
           _q(f"{L[2]:.3f}", unit_a)),
        _s("build", "pbc", _n("xyz")),
    ]
    if d.get("T") is not None:  # T is metadata; dialects without MD omit it
        system_statements.append(_s("state", "T", _q(f"{d['T']:.2f}", "K")))
    system_statements.append(
        _s("conserve", "atoms", _n(symbol), _q(str(d["N"]))))
    system = SystemBlock(statements=system_statements)
    physics = PhysicsBlock(statements=[_s("build", "backend", _n("eam"))])

    crystal_stmts = [
        _s("build", "lattice", _n(d["ctype"])),
        _s("build", "a", _q(f"{d['a_x']:.3f}", unit_a)),
        _s("build", "orient", _n(d["orient_x"].replace("<001>", "<100>")),
           _n("z"), _n(d["orient_z"])),
        _s("constrain", "strain", _n("zz"), _q(f"{strain:+.1f}", "%"),
           _tol(dialect.threshold("printed_strain_tolerance"))),
    ]
    for c in d["defects"]:
        vals = [_n(c["name"]), _n("count"), _q("1"), _n("depth"), _q(f"{c['depth']:.1f}")]
        if c["displaced"]:
            vals += [_n("form"), _n("split")]
        crystal_stmts.append(_s("build", "defect", *vals,
                               comment=f"{c['sites']} empty sites, {c['displaced']} displaced atoms"
                               if c["displaced"] else None))
    crystal_stmts.append(_s("assert", "sites_matched", _q(f"{100 * d['match']:.1f}", "%")))
    crystal = RegionBlock(
        phase="crystal", name="A",
        geometry=_geo_z(d["z_lo"], d["z_up"]),
        statements=crystal_stmts,
        comment="wraps through z = 0" if wrap_c else None)

    liquid = RegionBlock(
        phase="liquid", name="B",
        geometry=_geo_z(d["z_up"], d["z_lo"]),
        comment=_liquid_comment(d, "wraps through z = 0" if not wrap_c else None),
        statements=[
            _s("state", "density", _q(f"{d['rho']:.3f}")),
            _s("assert", "cn", _q(f"{d['cn']:.1f}"), _tol(d["cn_sd"]), _n("cutoff"),
               _q(f"{d['rcut']:.2f}", unit_a)),
            _s("assert", "gr_peak", _q(f"{d['pk_r']:.2f}", unit_a), _n("height"),
               _q(f"{d['pk_h']:.2f}")),
            _s("assert", "solid_clusters", _q(str(ncl))),
        ])
    ifaces = [
        InterfaceBlock(a="A", b="B", statements=[
            _s("build", "at", _n("z"), _q(f"{d['z_up']:.1f}")),
            _s("build", "width", _q(f"{d['w_up']:.1f}")),
        ]),
        InterfaceBlock(a="B", b="A", statements=[
            _s("build", "at", _n("z"), _q(f"{d['z_lo']:.1f}")),
            _s("build", "width", _q(f"{d['w_lo']:.1f}")),
        ]),
    ]
    explained = sum(c["displaced"] for c in d["defects"])
    if len(d["off"]) - explained == 0:
        residual = ResidualBlock(none=True)
    else:
        stmts = [_s("build", "atom", _n(symbol), _q(f"{x:.2f}"), _q(f"{y:.2f}"), _q(f"{z:.2f}"))
                 for x, y, z in d["off_xyz"]]
        residual = ResidualBlock(none=False, statements=stmts)
    provenance = ProvenanceBlock(statements=_provenance(d, dialect))
    return Program(
        version="0.1",  # dialect-exempt: numerical-guard: language version constant
        dialects=list(dialect.names),
        blocks=[system, physics, crystal, liquid, *ifaces, residual, provenance],
    )


def _program_multi(d, dialect):
    """Canonical program of a multi-species interface slab: the crystal region
    names its lattice on the majority solid species' sublattice (the unary
    lattice rule: the species is named in `conserve atoms`), the liquid region
    carries the molecular census of the other species, conservation is exact
    per species, and the interface blocks are the same objects as the
    single-species path's."""
    L = d["L"]
    counts = d["species_counts"] or {}
    strain = 100 * (d["a_z"] / d["a_x"] - 1)
    wrap_c = d["z_lo"] > d["z_up"]

    conserve = []
    for s in sorted(counts):
        conserve += [_n(s), _q(str(counts[s]))]
    system_statements = [
        _s("build", "cell", _q(f"{L[0]:.3f}"), _q(f"{L[1]:.3f}"), _q(f"{L[2]:.3f}")),
        _s("build", "pbc", _n("xyz")),
    ]
    if d.get("T") is not None:  # T is metadata; dialects without MD omit it
        system_statements.append(_s("state", "T", _q(f"{d['T']:.2f}")))
    system_statements.append(_s("conserve", "atoms", *conserve))
    system = SystemBlock(statements=system_statements)
    backend = "classical" if d["molecular"] else "lj"
    if "metal" in dialect.names:
        backend = "eam"
    physics = PhysicsBlock(statements=[_s("build", "backend", _n(backend))])

    crystal_stmts = [
        _s("build", "lattice", _n(d["ctype"])),
        _s("build", "a", _q(f"{d['a_x']:.3f}", "A")),
        _s("build", "orient", _n(d["orient_x"].replace("<001>", "<100>")),
           _n("z"), _n(d["orient_z"])),
        _s("constrain", "strain", _n("zz"), _q(f"{strain:+.1f}", "%"),
           _tol(dialect.threshold("printed_strain_tolerance"))),
    ]
    for c in d["defects"]:
        vals = [_n(c["name"]), _n("count"), _q("1"), _n("depth"), _q(f"{c['depth']:.1f}")]
        if c["displaced"]:
            vals += [_n("form"), _n("split")]
        crystal_stmts.append(_s("build", "defect", *vals,
                               comment=f"{c['sites']} empty sites, {c['displaced']} displaced atoms"
                               if c["displaced"] else None))
    crystal_stmts.append(_s("assert", "sites_matched", _q(f"{100 * d['match']:.1f}", "%")))
    crystal = RegionBlock(
        phase="crystal", name="A",
        geometry=_geo_z(d["z_lo"], d["z_up"]),
        statements=crystal_stmts,
        comment="wraps through z = 0" if wrap_c else None)

    liquid_stmts = []
    unit_a = "A"   # physical-unit dialects state lengths in Angstrom
    if d["molecular"]:
        from .reactive import display_name
        for formula in sorted(d["census"]):
            liquid_stmts.append(_s(
                "build", "molecules", _n(display_name(formula, dialect)),
                _q(str(d["census"][formula]))))
        liquid_stmts.append(_s("state", "density", _q(f"{d['rho_g']:.3f}", "g/cm3")))
    else:
        liquid_stmts.append(_s("state", "density", _q(f"{d['rho']:.3f}")))
    liquid_stmts += [
        _s("assert", "cn", _q(f"{d['cn']:.1f}"), _tol(d["cn_sd"]), _n("cutoff"),
           _q(f"{d['rcut']:.2f}", unit_a)),
        _s("assert", "gr_peak", _q(f"{d['pk_r']:.2f}", unit_a), _n("height"),
           _q(f"{d['pk_h']:.2f}")),
    ]
    if not d["molecular"]:
        ncl = int((d["clusters"] >= 2).sum())
        liquid_stmts.append(_s("assert", "solid_clusters", _q(str(ncl))))
    liquid = RegionBlock(
        phase="liquid", name="B",
        geometry=_geo_z(d["z_up"], d["z_lo"]),
        comment=_liquid_comment(d, "wraps through z = 0" if not wrap_c else None),
        statements=liquid_stmts)

    ifaces = [
        InterfaceBlock(a="A", b="B", statements=[
            _s("build", "at", _n("z"), _q(f"{d['z_up']:.1f}")),
            _s("build", "width", _q(f"{d['w_up']:.1f}")),
        ]),
        InterfaceBlock(a="B", b="A", statements=[
            _s("build", "at", _n("z"), _q(f"{d['z_lo']:.1f}")),
            _s("build", "width", _q(f"{d['w_lo']:.1f}")),
        ]),
    ]
    explained = sum(c["displaced"] for c in d["defects"])
    if len(d["off"]) - explained == 0:
        residual = ResidualBlock(none=True)
    else:
        stmts = [_s("build", "atom", _n(d["crystal_species"]),
                    _q(f"{x:.2f}"), _q(f"{y:.2f}"), _q(f"{z:.2f}"))
                 for x, y, z in d["off_xyz"]]
        residual = ResidualBlock(none=False, statements=stmts)
    provenance = ProvenanceBlock(statements=_provenance(d, dialect))
    return Program(
        version="0.1",  # dialect-exempt: numerical-guard: language version constant
        dialects=list(dialect.names),
        blocks=[system, physics, crystal, liquid, *ifaces, residual, provenance],
    )


def _n(text):
    from ..lang.ir import Family, Name
    if text.startswith("<") and text.endswith(">"):
        return Family(text=text)
    return Name(text=text)


def _sv(text):
    from ..lang.ir import StrVal
    return StrVal(text=text)


def _tol(v):
    from ..lang.ir import Tol
    return Tol(value=Quantity(num=f"{v:.1f}"))


def _geo_z(lo, hi, comment=None):
    rng = RangeVal(lo=Quantity(num=f"{lo:.1f}"), hi=Quantity(num=f"{hi:.1f}"))
    from ..lang.ir import GeoChain
    return GeoChain(parts=[ShSlab(axis="z", rng=rng)], ops=[])
