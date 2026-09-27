"""Solid-liquid slab lifter: the M0 decompiler path (Lennard-Jones style systems).

Ported from prototype/decompile.py. Every threshold is read from the dialect by
name; there are no fitted constants in this file. The lift produces a result
dictionary (observables, defects, statistics) plus a canonical Program built
through the IR, so all text goes through `chaord fmt`'s printer.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from ..lang.ir import (
    InterfaceBlock, PhysicsBlock, Program, ProvenanceBlock, Quantity,
    RangeVal, RegionBlock, ResidualBlock, ShSlab, Statement, SystemBlock,
)
from ..realize.lj import mic, wrap
from .passes import pairs_within, qbar


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


def decompile(r, L, T, dialect):
    """Lift passes 1-3 for a solid-liquid slab. Returns the result dictionary."""
    N = len(r)
    Lx, Ly, Lz = L
    q6_rc = float(dialect.threshold("q6_cutoff"))
    q6_solid = float(dialect.threshold("q6_solid"))
    margin = float(dialect.threshold("bulk_margin"))
    q6, cn, p = qbar(r, L, rc=q6_rc)
    solid = q6 > q6_solid
    # ---- pass 1: solid-fraction profile and phase segmentation along z -----------------
    binw = float(dialect.threshold("profile_bin_size"))
    kw = int(dialect.threshold("profile_smooth_wide"))
    kn = int(dialect.threshold("profile_smooth_narrow"))
    frac = float(dialect.threshold("phase_solid_fraction"))
    nb = int(Lz / binw)
    edges = np.linspace(0, Lz, nb + 1)
    zc = 0.5 * (edges[1:] + edges[:-1])  # dialect-exempt: bin centre arithmetic
    idx = np.minimum((r[:, 2] / Lz * nb).astype(int), nb - 1)
    nsol = np.bincount(idx, solid.astype(float), nb)
    ntot = np.bincount(idx, minlength=nb).astype(float)
    sm = lambda a, k: np.convolve(np.r_[a[-k:], a, a[:k]], np.ones(k), "same")[k:-k]
    ph = sm(nsol, kw) / np.maximum(sm(ntot, kw), 1e-9)  # dialect-exempt: divide-by-zero guard
    tot3 = sm(ntot, kn)
    phi = sm(nsol, kn) / np.maximum(tot3, 1e-9)         # dialect-exempt: divide-by-zero guard
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

    def fit(guess, sign):
        dz = _mz(zc - guess, Lz)
        m = (np.abs(dz) < fit_win) & (tot3 > 0)
        f = lambda z, z0, w: 0.5 * (1 + sign * np.tanh((z - z0) / w))  # dialect-exempt: tanh profile model
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
    ci = np.abs(dzc) < Hc - max(margin, max(w_lo, w_up) + q6_rc)
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
    nbr = [[] for _ in range(N)]
    for a_, b_ in p:
        nbr[a_].append(b_)
        nbr[b_].append(a_)
    chi_cos = float(dialect.threshold("chi0_antiparallel_cos"))
    stride = int(dialect.threshold("chi0_sample_stride"))
    chi0 = []
    for i in np.where(ci)[0][::stride]:
        v = mic(r[nbr[i]] - r[i], L)
        v /= np.linalg.norm(v, axis=1)[:, None]
        c = v @ v.T
        chi0.append(np.sum(np.triu(c, 1) < chi_cos))
    chi0 = float(np.mean(chi0))
    fcc_min = float(dialect.threshold("chi0_fcc_min"))
    hcp_min = float(dialect.threshold("chi0_hcp_min"))
    ctype = "fcc" if chi0 > fcc_min else ("hcp" if chi0 > hcp_min else "other")
    # ---- pass 3: liquid region -> statistical statements --------------------------------
    li = np.abs(dzl) < Hl - margin
    rho = li.sum() / (Lx * Ly * 2 * (Hl - margin))
    rmax = float(dialect.threshold("gr_rmax"))
    nbin = int(dialect.threshold("gr_bins"))
    rcut = float(dialect.threshold("cn_cutoff"))
    ang_bin = float(dialect.threshold("angle_hist_bin"))
    cen = np.abs(dzl) < Hl - margin - rmax
    pp = tree.query_pairs(rmax, output_type="ndarray")
    dd = np.linalg.norm(mic(r[pp[:, 1]] - r[pp[:, 0]], L), axis=1)
    rr = np.r_[dd[cen[pp[:, 0]]], dd[cen[pp[:, 1]]]]
    h, e = np.histogram(rr, np.linspace(0, rmax, nbin + 1))
    rm = 0.5 * (e[1:] + e[:-1])  # dialect-exempt: bin centre arithmetic
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
    return dict(
        N=N, L=L, T=T, z_lo=z_lo, z_up=z_up, w_lo=w_lo, w_up=w_up, Hc=Hc, Hl=Hl,
        zcr=zcr, zliq=zliq, ctype=ctype, chi0=chi0, a_x=2 * dx, a_y=2 * dy, a_z=2 * dz,
        orient_z=orient_z, orient_x=orient_x, nvac=int(empty.sum()), vac_depth=vac_depth,
        empty_xyz=E, defects=lift_defects(E, r[off], L, zcr, Hc, dialect),
        match=1 - empty.mean(), off=off, off_xyz=r[off], rho=rho, cn=cnl.mean(),
        cn_sd=cnl.std(), rcut=rcut, gr=gr, rm=rm, pk_r=rm[ipk], pk_h=gr[ipk], bad=bad,
        q6l=q6[li].mean(), clusters=sizes, x_solid_cr=solid[ci].mean(),
    )


# ---------------------------------------------------------------- program ----

def _q(num, unit=None):
    return Quantity(num=num, unit=unit)


def _s(kind, key, *values, comment=None):
    return Statement(kind=kind, key=key, values=list(values), comment=comment)


def program_from_result(d, dialect, symbol="X"):
    """Build the canonical Program IR from a decompile() result dictionary."""
    L = d["L"]
    strain = 100 * (d["a_z"] / d["a_x"] - 1)
    ncl = int((d["clusters"] >= 2).sum())
    wrap_c = d["z_lo"] > d["z_up"]

    system = SystemBlock(statements=[
        _s("build", "units", _n("lj")),
        _s("build", "cell", _q(f"{L[0]:.2f}"), _q(f"{L[1]:.2f}"), _q(f"{L[2]:.2f}")),
        _s("build", "pbc", _n("xyz")),
        _s("state", "T", _q(f"{d['T']:.2f}")),
        _s("conserve", "atoms", _n(symbol), _q(str(d["N"]))),
    ])
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
        comment="wraps through z = 0" if not wrap_c else None,
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
    provenance = ProvenanceBlock(statements=[
        _s("build", "dialects", _sv(dialect.version_string)),
        _s("build", "lift_version", _sv("0.1.0")),
    ])
    return Program(
        version="0.1",  # dialect-exempt: language version constant
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
