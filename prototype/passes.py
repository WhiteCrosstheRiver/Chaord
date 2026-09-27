"""Decompiler passes: local order (Lechner-Dellago q6), neighbour graph, profiles."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import sph_harm_y
from ljmd import wrap, mic

def pairs_within(r, L, rc):
    return cKDTree(wrap(r, L), boxsize=L).query_pairs(rc, output_type='ndarray')

def qbar(r, L, l=6, rc=1.45):
    p = pairs_within(r, L, rc); i, j = p[:, 0], p[:, 1]
    d = mic(r[j] - r[i], L); N = len(r)
    I = np.concatenate([i, j]); J = np.concatenate([j, i]); D = np.concatenate([d, -d])
    rn = np.linalg.norm(D, axis=1)
    th = np.arccos(np.clip(D[:, 2]/rn, -1, 1)); ph = np.arctan2(D[:, 1], D[:, 0])
    cnt = np.bincount(I, minlength=N)
    q = np.zeros((N, 2*l+1), complex)
    for k, m in enumerate(range(-l, l+1)):
        Y = sph_harm_y(l, m, th, ph)
        q[:, k] = (np.bincount(I, Y.real, N) + 1j*np.bincount(I, Y.imag, N)) / np.maximum(cnt, 1)
    qb = q.copy()
    for k in range(2*l+1):
        qb[:, k] += np.bincount(I, q[J, k].real, N) + 1j*np.bincount(I, q[J, k].imag, N)
    qb /= (cnt + 1)[:, None]
    return np.sqrt(4*np.pi/(2*l+1) * np.sum(np.abs(qb)**2, 1)), cnt, p

def otsu(x, bins=100):
    h, e = np.histogram(x, bins); c = 0.5*(e[1:]+e[:-1]); w = h/h.sum()
    best, thr = -1, None
    for k in range(1, bins):
        w0, w1 = w[:k].sum(), w[k:].sum()
        if w0 == 0 or w1 == 0: continue
        m0 = (w[:k]*c[:k]).sum()/w0; m1 = (w[k:]*c[k:]).sum()/w1
        s = w0*w1*(m0-m1)**2
        if s > best: best, thr = s, c[k]
    return thr
