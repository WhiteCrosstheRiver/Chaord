"""Analytic Finnis-Sinclair / EAM MD backend (numpy + scipy only), eV and A units.

Potential (Finnis & Sinclair 1984 analytic form, square-root embedding):

    E = sum_i [ 1/2 sum_{j!=i} V_ij(r_ij) - A_i sqrt(rho_i) ]
    rho_i = sum_{j!=i} phi_ij(r_ij)

    V(r)   = (r - c)^2 (c0 + c1 r + c2 r^2)   for r < c   (else 0)
    phi(r) = (r - d)^2                         for r < d   (else 0)

Both functions vanish quadratically at their cutoffs, so the potential is C1
continuous and the neighbour-list skin logic is safe. Per-species parameters
(A, c, d, c0, c1, c2) are read from the dialect (`eam_potentials`), never
hard-coded here. Cross-species pair/density functions use the arithmetic mean
of the two elements' coefficients (pure-element limits are exact).

Units: eV, A; `run_md` (realize/lj.py) integrates with m = kB = 1, so the
implicit time unit is sqrt(u A^2 / eV) ~ 10.18 fs and temperatures are passed
in eV (kB T). The class exposes the same build(r)/forces(r) interface as LJ,
plus energy(r); any object with forces/build plugs into run_md unchanged.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .lj import mic, wrap  # noqa: F401  (wrap re-exported for callers)

_PAIR_KEYS = ("c", "c0", "c1", "c2", "d")


def _mix(a, b):
    """Arithmetic-mean mixing of two elements' pair/density coefficients."""
    return 0.5 * (a + b)


class EAM:
    """Analytic Finnis-Sinclair potential with a Verlet-style skin neighbour list."""

    def __init__(self, L, rc=None, skin=None, species_params=None, symbols=None):
        self.L = np.asarray(L, float)
        self.params = dict(species_params or {})
        if not self.params:
            raise ValueError("EAM needs a non-empty species_params mapping")
        names = sorted(self.params)
        for s in names:
            missing = [k for k in ("A",) + _PAIR_KEYS if k not in self.params[s]]
            if missing:
                raise ValueError(f"EAM species {s!r} lacks parameters {missing}")
        self.names = names
        # per-species vectors
        self.A = np.array([float(self.params[s]["A"]) for s in names])
        # pair-mixing matrices [si, sj] (arithmetic mean across species)
        self._mix = {k: np.array([[float(self.params[a][k]) for b in names]
                                  for a in names]) for k in _PAIR_KEYS}
        self.rc = float(rc) if rc is not None else float(
            max(m.max() for m in self._mix.values()))
        self.skin = float(skin) if skin is not None else 0.3  # fallback; callers read the dialect
        if symbols is None:
            if len(names) != 1:
                raise ValueError("multi-species EAM needs the per-atom symbols list")
            self.sp_idx = np.zeros(1, int)
            self._symbols = None
        else:
            index = {s: k for k, s in enumerate(names)}
            unknown = sorted(set(symbols) - set(index))
            if unknown:
                raise ValueError(f"EAM has no parameters for species {unknown}")
            self.sp_idx = np.array([index[s] for s in symbols], int)
            self._symbols = list(symbols)
        self.pairs = None
        self.r_last = None

    # ------------------------------------------------------------------ setup --
    def _atom_species(self, n):
        if self._symbols is None:
            return np.zeros(n, int)
        if len(self._symbols) != n:
            raise ValueError("atom count changed since construction")
        return self.sp_idx

    def build(self, r):
        t = cKDTree(wrap(r, self.L), boxsize=self.L)
        self.pairs = t.query_pairs(self.rc + self.skin, output_type="ndarray")
        self.r_last = r.copy()

    def _needs_rebuild(self, r):
        if self.pairs is None:
            return True
        disp2 = np.sum(mic(r - self.r_last, self.L) ** 2, 1)
        return disp2.max() > (0.5 * self.skin) ** 2

    # --------------------------------------------------------------- internals --
    def _bond(self, r):
        """Pair list with min-image distances; rebuilds on skin violation."""
        if self._needs_rebuild(r):
            self.build(r)
        i, j = self.pairs[:, 0], self.pairs[:, 1]
        d = mic(r[j] - r[i], self.L)
        r2 = np.einsum("ij,ij->i", d, d)
        return i, j, d, np.sqrt(r2)

    # ------------------------------------------------------------ API: forces --
    def forces(self, r):
        r = np.asarray(r, float)
        n = len(r)
        self._sp = self._atom_species(n)
        i, j, d, dist = self._bond(r)
        sp_i, sp_j = self._sp[i], self._sp[j]
        cc = self._mix["c"][sp_i, sp_j]
        c0 = self._mix["c0"][sp_i, sp_j]
        c1 = self._mix["c1"][sp_i, sp_j]
        c2 = self._mix["c2"][sp_i, sp_j]
        dd = self._mix["d"][sp_i, sp_j]

        # rho_i = sum_j (r - d)^2  [r < d]
        m_d = dist < dd
        w = np.where(m_d, (dist - dd) ** 2, 0.0)
        rho = np.bincount(i, w, n) + np.bincount(j, w, n)

        # embedding derivative coefficient A_i / (2 sqrt(rho_i)); 0 for isolated atoms
        rho_safe = np.where(rho > 0.0, rho, 1.0)
        k = np.where(rho > 0.0, self.A[self._sp] / (2.0 * np.sqrt(rho_safe)), 0.0)

        # dV/dr = 2(r-c)(c0+c1r+c2r^2) + (r-c)^2 (c1+2 c2 r)  [r < c]
        m_c = dist < cc
        x = dist - cc
        poly = c0 + c1 * dist + c2 * dist ** 2
        dv = np.where(m_c, 2.0 * x * poly + x ** 2 * (c1 + 2.0 * c2 * dist), 0.0)
        # dphi/dr = 2(r - d)  [r < d]
        dw = np.where(m_d, 2.0 * (dist - dd), 0.0)

        # F_i = sum_j [ V'(r) - (A_i/(2 sqrt(rho_i)) + A_j/(2 sqrt(rho_j))) phi'(r) ]
        #       * (r_j - r_i) / r   (the embedding attraction carries the minus sign:
        # E_emb = -A sqrt(rho), so its gradient pulls i towards higher density)
        g = dv - (k[i] + k[j]) * dw
        inv_r = np.where(dist > 0.0, 1.0 / np.where(dist > 0.0, dist, 1.0), 0.0)
        fij = (g * inv_r)[:, None] * d  # force on i (pair label i), reaction on j
        F = np.empty_like(r)
        for axis in range(3):
            F[:, axis] = np.bincount(i, fij[:, axis], n) - np.bincount(j, fij[:, axis], n)
        return F

    # ------------------------------------------------------------ API: energy --
    def energy(self, r):
        """Total potential energy (eV); pair sum plus square-root embedding."""
        r = np.asarray(r, float)
        n = len(r)
        self._sp = self._atom_species(n)
        i, j, _, dist = self._bond(r)
        sp_i, sp_j = self._sp[i], self._sp[j]
        cc = self._mix["c"][sp_i, sp_j]
        c0 = self._mix["c0"][sp_i, sp_j]
        c1 = self._mix["c1"][sp_i, sp_j]
        c2 = self._mix["c2"][sp_i, sp_j]
        dd = self._mix["d"][sp_i, sp_j]

        m_c = dist < cc
        x = dist - cc
        pair = np.where(m_c, x ** 2 * (c0 + c1 * dist + c2 * dist ** 2), 0.0)
        e_pair = float(np.sum(pair))  # pairs are unordered (i<j): no 1/2 factor

        m_d = dist < dd
        w = np.where(m_d, (dist - dd) ** 2, 0.0)
        rho = np.bincount(i, w, n) + np.bincount(j, w, n)
        e_emb = -float(np.sum(self.A[self._sp] * np.sqrt(rho)))
        return e_pair + e_emb
