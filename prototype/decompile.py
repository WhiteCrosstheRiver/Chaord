"""Decompile a solid-liquid slab snapshot into a short declarative program."""
import numpy as np
from scipy.optimize import curve_fit
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from passes import qbar, pairs_within
from ljmd import wrap, mic

Q6_SOLID = 0.32          # 'solid-like' is part of the dialect definition, not re-fitted per frame
MARGIN = 2.5             # distance kept from interface centres when measuring bulk statistics

def mz(dz, Lz): return dz - Lz*np.round(dz/Lz)

def circ_offset(x, d):
    z = np.mean(np.exp(2j*np.pi*x/d)); return abs(z), d*np.angle(z)/(2*np.pi)

def lift_defects(E, O, L, zcr, Hc, rc=1.3):
    """Lifting pass: group empty sites (E) and off-lattice atoms (O) into defect complexes.
    Net vacancies = empty sites - displaced atoms (atom count is conserved)."""
    P = np.r_[E, O]; n = len(P)
    if n == 0: return []
    kind = np.r_[np.zeros(len(E), int), np.ones(len(O), int)]
    Dm = np.linalg.norm(mic(P[:, None, :] - P[None, :, :], L), axis=2)
    _, lab = connected_components(coo_matrix(Dm < rc), directed=False)
    out = []
    for c in np.unique(lab):
        m = lab == c; ne, no = int((kind[m] == 0).sum()), int((kind[m] == 1).sum())
        depth = float(np.mean(Hc - np.abs(mz(P[m, 2] - zcr, L[2]))))
        net = ne - no
        name = {1: 'vacancy', 2: 'divacancy', 3: 'trivacancy'}.get(net, f'V{net}' if net > 0 else ('frenkel_pair' if net == 0 else 'interstitial'))
        out.append(dict(name=name, net=net, sites=ne, displaced=no, depth=depth, xyz=P[m]))
    return sorted(out, key=lambda d: (-d['net'], d['depth']))

def orientation(bvec):
    """Compare |cos| of bond directions with an axis to ideal fcc <110> bond sets."""
    ideal = {'[001]': [1/3, 0, 2/3, 0], '[110]': [1/6, 2/3, 0, 1/6], '[111]': [1/2, 0, 1/2, 0]}
    h = np.histogram(np.abs(bvec), [0, .25, .6, .9, 1.0001])[0]; h = h/h.sum()
    return '<' + min(ideal, key=lambda k: np.abs(h - ideal[k]).sum())[1:-1] + '>'

def decompile(r, L, T):
    N = len(r); Lx, Ly, Lz = L
    q6, cn, p = qbar(r, L); solid = q6 > Q6_SOLID
    # ---- pass 1: solid-fraction profile and phase segmentation along z -------------
    nb = int(Lz/0.25); edges = np.linspace(0, Lz, nb+1); zc = 0.5*(edges[1:]+edges[:-1])
    idx = np.minimum((r[:, 2]/Lz*nb).astype(int), nb-1)
    nsol = np.bincount(idx, solid.astype(float), nb); ntot = np.bincount(idx, minlength=nb).astype(float)
    sm = lambda a, k: np.convolve(np.r_[a[-k:], a, a[:k]], np.ones(k), 'same')[k:-k]
    ph = sm(nsol, 5)/np.maximum(sm(ntot, 5), 1e-9)            # ratio of smoothed counts: no empty-bin artefacts
    tot3 = sm(ntot, 3); phi = sm(nsol, 3)/np.maximum(tot3, 1e-9)
    on = np.r_[ph > .5, ph > .5]; best = (0, 0); s = None
    for i, b in enumerate(on):
        if b and s is None: s = i
        if (not b or i == len(on)-1) and s is not None:
            n = i - s + (1 if b else 0)
            if n > best[1] and n <= nb: best = (s, n)
            s = None
    s, n = best
    guess_lo, guess_up = edges[s % nb], edges[(s+n) % nb]
    def fit(guess, sign):
        dz = mz(zc - guess, Lz); m = (np.abs(dz) < 4.5) & (tot3 > 0)
        f = lambda z, z0, w: 0.5*(1 + sign*np.tanh((z - z0)/w))
        (z0, w), _ = curve_fit(f, dz[m], phi[m], p0=[0, 1], bounds=([-3, .05], [3, 6]))
        return np.mod(guess + z0, Lz), 2.197*w
    z_lo, w_lo = fit(guess_lo, +1); z_up, w_up = fit(guess_up, -1)
    Hc = np.mod(z_up - z_lo, Lz)/2; zcr = np.mod(z_lo + Hc, Lz)
    Hl = Lz/2 - Hc; zliq = np.mod(zcr + Lz/2, Lz)
    dzc = mz(r[:, 2] - zcr, Lz); dzl = mz(r[:, 2] - zliq, Lz)
    # ---- pass 2: crystal region -> exact lattice + defects ------------------------------
    ci = np.abs(dzc) < Hc - max(MARGIN, max(w_lo, w_up) + 1.5)
    kx = max(range(8, 30), key=lambda k: circ_offset(r[ci, 0], Lx/k)[0]); dx = Lx/kx
    ky = max(range(8, 30), key=lambda k: circ_offset(r[ci, 1], Ly/k)[0]); dy = Ly/ky
    ds = np.arange(0.6, 1.1, 0.0005); dz = ds[np.argmax([circ_offset(dzc[ci], d)[0] for d in ds])]
    ox, oy, oz = circ_offset(r[ci, 0], dx)[1], circ_offset(r[ci, 1], dy)[1], circ_offset(dzc[ci], dz)[1]
    ii = np.round((r[ci, 0]-ox)/dx).astype(int); jj = np.round((r[ci, 1]-oy)/dy).astype(int)
    kk = np.round((dzc[ci]-oz)/dz).astype(int); par = np.bincount((ii+jj+kk) % 2, minlength=2).argmax()
    kmax = int((Hc - MARGIN)/dz) + 2
    g = np.array(np.meshgrid(range(kx), range(ky), range(-kmax, kmax+1), indexing='ij')).reshape(3, -1).T
    g = g[(g.sum(1) % 2) == par]
    sites = np.c_[ox + g[:, 0]*dx, oy + g[:, 1]*dy, zcr + oz + g[:, 2]*dz]
    zone = Hc - max(MARGIN, max(w_lo, w_up) + 1.5)
    sites_ext = sites[np.abs(mz(sites[:, 2]-zcr, Lz)) < zone + 1.0]   # for atom->site assignment
    sites = sites[np.abs(mz(sites[:, 2]-zcr, Lz)) < zone]
    tree = cKDTree(wrap(r, L), boxsize=L)
    dist, _ = tree.query(wrap(sites, L)); empty = dist > 0.45
    vac_depth = np.sort(Hc - np.abs(mz(sites[empty, 2]-zcr, Lz))); E = sites[empty]
    dsite, _ = cKDTree(wrap(sites_ext, L), boxsize=L).query(wrap(r[ci], L))
    off = np.where(ci)[0][dsite > 0.45]
    pi, pj = p[:, 0], p[:, 1]; both = ci[pi] & ci[pj]; bv = mic(r[pj[both]]-r[pi[both]], L)
    bv /= np.linalg.norm(bv, axis=1)[:, None]
    orient_z, orient_x = orientation(bv[:, 2]), orientation(bv[:, 0])
    # Ackland-Jones-style 3-body check: pairs of neighbours at ~180 deg (fcc: 6, hcp: 3)
    nbr = [[] for _ in range(N)]
    for a_, b_ in p: nbr[a_].append(b_); nbr[b_].append(a_)
    chi0 = []
    for i in np.where(ci)[0][::3]:
        v = mic(r[nbr[i]] - r[i], L); v /= np.linalg.norm(v, axis=1)[:, None]
        c = v @ v.T; chi0.append(np.sum(np.triu(c, 1) < -0.945))
    chi0 = float(np.mean(chi0)); ctype = 'fcc' if chi0 > 4.5 else ('hcp' if chi0 > 2 else 'other')
    # ---- pass 3: liquid region -> statistical statements -----------------------------------
    li = np.abs(dzl) < Hl - MARGIN; rho = li.sum()/(Lx*Ly*2*(Hl - MARGIN))
    rmax, nbin = 2.5, 100; cen = np.abs(dzl) < Hl - MARGIN - rmax
    pp = tree.query_pairs(rmax, output_type='ndarray'); dd = np.linalg.norm(mic(r[pp[:, 1]]-r[pp[:, 0]], L), axis=1)
    rr = np.r_[dd[cen[pp[:, 0]]], dd[cen[pp[:, 1]]]]
    h, e = np.histogram(rr, np.linspace(0, rmax, nbin+1)); rm = 0.5*(e[1:]+e[:-1])
    gr = h/(cen.sum()*rho*4/3*np.pi*(e[1:]**3 - e[:-1]**3))
    ipk = np.argmax(gr); rcut = 1.50
    nl = tree.query_ball_point(wrap(r[cen], L), rcut)
    cnl = np.array([len(x)-1 for x in nl])
    ang = []
    for c_i, lst in zip(np.where(cen)[0], nl):
        lst = [x for x in lst if x != c_i]
        v = mic(r[lst]-r[c_i], L); v /= np.linalg.norm(v, axis=1)[:, None]
        cc = (v @ v.T)[np.triu_indices(len(lst), 1)]; ang.append(np.degrees(np.arccos(np.clip(cc, -1, 1))))
    bad = np.histogram(np.concatenate(ang), np.arange(0, 181, 5))[0].astype(float); bad /= bad.sum()
    sl = np.where(li & solid)[0]; sub = np.isin(pi, sl) & np.isin(pj, sl)
    loc = {a: b for b, a in enumerate(sl)}
    if len(sl):
        A = coo_matrix((np.ones(sub.sum()), ([loc[a] for a in pi[sub]], [loc[b] for b in pj[sub]])), shape=(len(sl),)*2)
        _, lab = connected_components(A, directed=False); sizes = np.sort(np.bincount(lab))[::-1]
    else:
        sizes = np.array([], int)
    res = dict(N=N, L=L, T=T, z_lo=z_lo, z_up=z_up, w_lo=w_lo, w_up=w_up, Hc=Hc, Hl=Hl, zcr=zcr, zliq=zliq,
               ctype=ctype, chi0=chi0, a_x=2*dx, a_y=2*dy, a_z=2*dz, orient_z=orient_z, orient_x=orient_x,
               nvac=int(empty.sum()), vac_depth=vac_depth, empty_xyz=E, defects=lift_defects(E, r[off], L, zcr, Hc), match=1-empty.mean(), off=off, off_xyz=r[off],
               rho=rho, cn=cnl.mean(), cn_sd=cnl.std(), rcut=rcut, gr=gr, rm=rm, pk_r=rm[ipk], pk_h=gr[ipk],
               bad=bad, q6l=q6[li].mean(), clusters=sizes, x_solid_cr=solid[ci].mean())
    return res

def to_text(d, dialect="core + lj"):
    """Print a decompiled result in Chaord v0.1 canonical syntax (one statement per line)."""
    L = d['L']; strain = 100*(d['a_z']/d['a_x'] - 1)
    ncl = int((d['clusters'] >= 2).sum())
    fam = lambda f: f
    wrap_c = d['z_lo'] > d['z_up']; wrap_l = d['z_up'] > d['z_lo']
    out = ["chaord 0.1", f"dialect {dialect}", "",
           "system {", "  units lj", f"  cell {L[0]:.2f} {L[1]:.2f} {L[2]:.2f}", "  pbc xyz",
           f"  state T {d['T']:.2f}", f"  conserve atoms X {d['N']}", "}", "",
           "physics {", "  backend lj", "  epsilon 1", "  sigma 1", "  cutoff 2.5", "}", "",
           f"crystal A : slab z {d['z_lo']:.1f} .. {d['z_up']:.1f} {{" + ("        # wraps through z = 0" if wrap_c else ""),
           f"  lattice {d['ctype']}", f"  a {d['a_x']:.3f}",
           f"  orient x {d['orient_x'].replace('<001>', '<100>')} z {d['orient_z']}",
           f"  constrain strain zz {strain:+.1f} % +- 0.2"]
    for c in d['defects']:
        line = f"  defect {c['name']} count 1 depth {c['depth']:.1f}"
        if c['displaced']:
            line += f" form split          # {c['sites']} empty sites, {c['displaced']} displaced atoms"
        out.append(line)
    out += [f"  assert sites_matched {100*d['match']:.1f} %", "}", "",
            f"liquid B : slab z {d['z_up']:.1f} .. {d['z_lo']:.1f} {{" + ("        # wraps through z = 0" if wrap_l else ""),
            f"  state density {d['rho']:.3f}",
            f"  assert cn {d['cn']:.1f} +- {d['cn_sd']:.1f} cutoff {d['rcut']:.2f}",
            f"  assert gr_peak {d['pk_r']:.2f} height {d['pk_h']:.2f}",
            f"  assert solid_clusters {ncl}", "}", "",
            "interface A | B {", f"  at z {d['z_up']:.1f}", f"  width {d['w_up']:.1f}", "}", "",
            "interface B | A {", f"  at z {d['z_lo']:.1f}", f"  width {d['w_lo']:.1f}", "}", ""]
    explained = sum(c['displaced'] for c in d['defects'])
    if len(d['off']) - explained == 0:
        out.append("residual none")
    else:
        out.append("residual {")
        out += [f"  atom X {x:.2f} {y:.2f} {z:.2f}" for x, y, z in d['off_xyz']]
        out.append("}")
    return "\n".join(out) + "\n"
