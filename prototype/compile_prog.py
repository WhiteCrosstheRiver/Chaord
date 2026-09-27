"""Compile a (decompiled) program back into coordinates.
Uses ONLY build/state statements: cell, T, crystal slab + lattice + strain + defects (by depth),
liquid density. 'assert' lines (cn, g(r) peak, clusters) are NOT used - they are checked afterwards."""
import numpy as np
from ljmd import LJ, run_md, wrap, mic

def mz(dz, Lz): return dz - Lz*np.round(dz/Lz)

def rsa(existing, L, zc, H, n, dmin, rng):
    """Random sequential addition in the slab |z - zc| < H (Packmol-like, no physics)."""
    nc = np.maximum((L/dmin).astype(int), 1); cs = L/nc; grid = {}
    key = lambda p: (int(p[0]//cs[0]) % nc[0], int(p[1]//cs[1]) % nc[1], int(p[2]//cs[2]) % nc[2])
    for p in wrap(existing, L): grid.setdefault(key(p), []).append(tuple(p))
    out = []; d2 = dmin*dmin; Lx, Ly, Lz = L
    while len(out) < n:
        p = (rng.random()*Lx, rng.random()*Ly, (zc + (2*rng.random() - 1)*H) % Lz)
        k = key(p); ok = True
        for a in (-1, 0, 1):
            for b in (-1, 0, 1):
                for c in (-1, 0, 1):
                    for q in grid.get(((k[0]+a) % nc[0], (k[1]+b) % nc[1], (k[2]+c) % nc[2]), ()):
                        dx = q[0]-p[0]; dx -= Lx*round(dx/Lx)
                        dy = q[1]-p[1]; dy -= Ly*round(dy/Ly)
                        dz = q[2]-p[2]; dz -= Lz*round(dz/Lz)
                        if dx*dx + dy*dy + dz*dz < d2: ok = False; break
                    if not ok: break
                if not ok: break
            if not ok: break
        if ok: grid.setdefault(k, []).append(p); out.append(p)
    return np.array(out)

def compile_program(P, rng, physics=True, md_steps=2000):
    L = P['L'].copy(); Lx, Ly, Lz = L; T = P['T']
    # crystal: in-plane spacing commensurate with the cell, z spacing from the strain statement
    kx = int(round(Lx/(P['a_x']/2))); dx = Lx/kx; ky = int(round(Ly/(P['a_y']/2))); dy = Ly/ky
    dz = dx*(P['a_z']/P['a_x'])
    nplanes = int(round(2*P['Hc']/dz)); zcr = P['zcr']
    xs = []
    for k in range(nplanes):
        z = zcr + (k - (nplanes - 1)/2)*dz
        for i in range(kx):
            for j in range(ky):
                if (i + j + k) % 2 == 0: xs.append((i*dx, j*dy, z))
    X = np.array(xs)
    # defects: remove sites at the stated depth; in-plane position is free (translational symmetry)
    depth = P['Hc'] - np.abs(mz(X[:, 2] - zcr, Lz)); keep = np.ones(len(X), bool)
    for c in P['defects']:
        for _ in range(c['net']):
            cand = np.where(keep & (np.abs(depth - c['depth']) < dz/2 + 1e-9))[0]
            if len(cand) == 0: cand = np.where(keep)[0][np.argsort(np.abs(depth[keep] - c['depth']))[:20]]
            if _ == 0: first = X[rng.choice(cand)]
            else:     # neighbouring site for divacancy / trivacancy
                dd = np.linalg.norm(mic(X[cand] - first, L), axis=1); cand = cand[np.argsort(dd)[:1]]
            idx = cand[0] if _ else np.where((X == first).all(1))[0][0]
            keep[idx] = False
    X = X[keep]
    # liquid: stated density over the liquid slab, placed with a hard-core rule only
    nliq = P['N'] - len(X)          # exact conservation: total atom count is stated in the program
    Y = rsa(X, L, P['zliq'], P['Hl'] - 0.3, nliq, 0.90, rng)
    r = np.r_[X, Y].astype(float)
    if physics:   # the physics prior (here LJ; in real use, your MLP) fills in everything unstated
        v = np.zeros_like(r); lj = LJ(L)
        r, v = run_md(r, v, L, 400, 0.001, T, 5.0, rng, lj=lj, fcap=30.0)
        r, v = run_md(r, v, L, md_steps, 0.005, T, 0.5, rng, lj=lj)
    return wrap(r, L), L
