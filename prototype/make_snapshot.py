import numpy as np, time
from ljmd import *
rng = np.random.default_rng(11)
t0 = time.time()
a0 = (4/0.96)**(1/3); nx = ny = 6; nz = 16
r = fcc(nx, ny, nz, a0); L = np.array([nx, ny, nz]) * a0; zmid = L[2]/2
cryst = r[:, 2] < zmid - 1e-6
cand = np.where(cryst & (r[:, 2] > 2*a0) & (r[:, 2] < zmid - 2*a0))[0]
vac = rng.choice(cand, 3, replace=False)
keep = np.setdiff1d(np.arange(len(r)), vac); r, cryst = r[keep], cryst[keep]
v = rng.standard_normal(r.shape) * np.sqrt(0.65)
# 1) melt the top half with the bottom half frozen
r, v = run_md(r, v, L, 1500, 0.004, np.where(cryst, 0.0, 2.5), 2.0, rng, frozen=cryst)
# 2) expand the liquid half along z to liquid-like density
s = 0.96/0.845; lo = zmid - 0.5*a0
zl = lo + np.mod(r[~cryst, 2] - lo, L[2])
r[~cryst, 2] = zmid + (zl - zmid)*s
L2 = L.copy(); L2[2] = zmid + (L[2] - zmid)*s
# 3) everything free at T = 0.65
lj = LJ(L2)
r, v = run_md(r, v, L2, 3000, 0.005, 0.65, 0.5, rng, lj=lj)
np.savez('snap.npz', r=wrap(r, L2), L=L2, v=v, vac_z=None)
# a second, later frame of the same run = natural-fluctuation reference
r2, v2 = run_md(r.copy(), v.copy(), L2, 1000, 0.005, 0.65, 0.5, rng, lj=lj)
np.savez('snap_later.npz', r=wrap(r2, L2), L=L2)
print(f"N={len(r)}  box={L2.round(3)}  time {time.time()-t0:.0f}s")
