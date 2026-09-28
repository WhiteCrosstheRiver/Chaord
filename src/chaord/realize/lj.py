"""Minimal Lennard-Jones MD (reduced units, m = kB = 1), numpy + scipy only.

Ported from prototype/ljmd.py unchanged in physics; MD defaults are read from the
dialect (`md:` section) by callers, never hard-coded here.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def wrap(r, L):
    w = np.mod(r, L)
    return np.where(w >= L, w - L, w)


def mic(d, L):
    return d - L * np.round(d / L)


class LJ:
    """Lennard-Jones pair potential with Verlet-style skin neighbour list."""

    def __init__(self, L, rc=None, skin=None):
        self.L = np.asarray(L, float)
        self.rc = float(rc) if rc is not None else None
        self.skin = float(skin) if skin is not None else None
        self.pairs = None
        self.r_last = None

    def build(self, r):
        rc = self.rc
        t = cKDTree(wrap(r, self.L), boxsize=self.L)
        self.pairs = t.query_pairs(rc + self.skin, output_type="ndarray")
        self.r_last = r.copy()

    def forces(self, r):
        if self.pairs is None or np.max(
                np.sum(mic(r - self.r_last, self.L) ** 2, 1)) > (0.5 * self.skin) ** 2:  # dialect-exempt: exact-geometry
            self.build(r)
        i, j = self.pairs[:, 0], self.pairs[:, 1]
        d = mic(r[j] - r[i], self.L)
        r2 = np.einsum("ij,ij->i", d, d)
        m = r2 < self.rc ** 2
        i, j, d, r2 = i[m], j[m], d[m], r2[m]
        inv2 = 1.0 / r2  # dialect-exempt: numerical-guard: reciprocal one
        inv6 = inv2 ** 3
        fs = 24.0 * inv2 * inv6 * (2.0 * inv6 - 1.0)  # dialect-exempt: exact-geometry
        fij = fs[:, None] * d
        N = len(r)
        F = np.empty_like(r)
        for k in range(3):
            F[:, k] = np.bincount(j, fij[:, k], N) - np.bincount(i, fij[:, k], N)
        return F


def run_md(r, v, L, nsteps, dt, T, gamma, rng, frozen=None, lj=None, fcap=None):
    """BAOAB Langevin. T may be a scalar or per-atom array. Frozen atoms never move."""
    lj = lj or LJ(L)
    N = len(r)
    T = np.full(N, float(T)) if np.ndim(T) == 0 else np.asarray(T, float)
    c1 = np.exp(-gamma * dt)
    c2 = np.sqrt((1 - c1 ** 2) * T)[:, None]
    mob = np.ones(N, bool) if frozen is None else ~frozen

    def f(r):
        F = lj.forces(r)
        F[~mob] = 0
        if fcap is not None:
            n = np.linalg.norm(F, axis=1, keepdims=True)
            F = np.where(n > fcap, F * fcap / n, F)
        return F

    F = f(r)
    v[~mob] = 0
    for _ in range(nsteps):
        v += 0.5 * dt * F  # dialect-exempt: exact-geometry
        r += 0.5 * dt * v  # dialect-exempt: exact-geometry
        v = c1 * v + c2 * rng.standard_normal(r.shape)
        v[~mob] = 0
        r += 0.5 * dt * v  # dialect-exempt: exact-geometry
        F = f(r)
        v += 0.5 * dt * F  # dialect-exempt: exact-geometry
    return r, v


def fcc(nx, ny, nz, a):
    b = np.array([[0, 0, 0], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]])  # dialect-exempt: exact-geometry
    g = np.array(np.meshgrid(range(nx), range(ny), range(nz), indexing="ij")).reshape(3, -1).T
    return ((g[:, None, :] + b[None]).reshape(-1, 3)) * a
