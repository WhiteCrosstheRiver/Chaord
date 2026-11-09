# -*- coding: utf-8 -*-
"""Common-neighbour analysis, written without importing chaord: the structural
check on rebuilds must not share code with the compiler or the decompiler."""
import numpy as np
from scipy.spatial import cKDTree


def _longest_chain(bonds):
    """Longest continuous bond chain among the common neighbours."""
    best = 0
    remaining = set(bonds)
    while remaining:
        first = remaining.pop()
        nodes, length, grew = set(first), 1, True
        while grew:
            grew = False
            for b in list(remaining):
                if nodes & set(b):
                    remaining.discard(b)
                    nodes |= set(b)
                    length += 1
                    grew = True
        best = max(best, length)
    return best


def classify(pos, lengths, rc, labels=False):
    """Fractions of atoms classed fcc, hcp and other (or, with labels=True, each
    atom's class 0/1/2). Orthogonal periodic box."""
    p = np.mod(np.asarray(pos, float), lengths)
    p[p >= lengths] = 0.0
    tree = cKDTree(p, boxsize=lengths)
    adj = [set() for _ in range(len(p))]
    for i, j in tree.query_pairs(rc, output_type="ndarray"):
        adj[i].add(int(j))
        adj[j].add(int(i))
    kinds = np.full(len(p), 2, int)
    for i, nb in enumerate(adj):
        if len(nb) != 12:
            continue
        n421 = n422 = 0
        for j in nb:
            common = nb & adj[j]
            if len(common) != 4:
                continue
            c = sorted(common)
            bonds = [(a, b) for k, a in enumerate(c) for b in c[k + 1:] if b in adj[a]]
            if len(bonds) != 2:
                continue
            if _longest_chain(bonds) == 1:
                n421 += 1
            else:
                n422 += 1
        if n421 == 12:
            kinds[i] = 0
        elif n421 == 6 and n422 == 6:
            kinds[i] = 1
    return kinds if labels else np.bincount(kinds, minlength=3) / len(p)
