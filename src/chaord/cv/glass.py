"""Glass statistics: ring-size distribution and Voronoi indices."""
from __future__ import annotations

import numpy as np
from collections import Counter

from ..io.frames import Frame
from ..build.molecules import bond_graph


def ring_distribution(frame: Frame, dialect, max_ring=None, edges=None) -> Counter:
    """King-style shortest-path rings, counted per bond of the bond graph.

    Each bond contributes the size of its shortest cycle (3..max_ring); bonds
    in no cycle within the search size are ignored. The distribution is the
    per-bond count, the standard network-glass fingerprint."""
    max_ring = int(max_ring if max_ring is not None
                   else dialect.threshold("ring_search_max"))
    if edges is None:
        edges = bond_graph(frame, dialect)
    adj: dict[int, set[int]] = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    rings = Counter()
    for a, b in edges:
        path = _bfs(adj, b, a, max_len=max_ring)
        if path is not None and 3 <= len(path) <= max_ring:
            rings[len(path)] += 1
    return rings


def _bfs(adj, start, goal, max_len):
    """Shortest cycle-closing path start->goal without the direct bond.

    The direct start-goal edge is excluded at the first level; the goal may be
    reached through any longer path."""
    from collections import deque
    q = deque([start])
    dist = {start: 0}
    prev = {start: None}
    while q:
        x = q.popleft()
        if dist[x] >= max_len:
            continue
        for y in adj.get(x, ()):
            if y == goal and dist[x] == 0:
                continue  # this is the removed bond, not a cycle
            if y in dist and dist[y] <= dist[x] + 1:
                continue
            dist[y] = dist[x] + 1
            prev[y] = x
            if y == goal:
                path = [y]
                while path[-1] != start:
                    path.append(prev[path[-1]])
                return path
            q.append(y)
    return None


def voronoi_index_distribution(frame: Frame, dialect, max_atoms=600) -> Counter:
    """Voronoi face counts of the periodic structure, per atom.

    Computed on a 3x3x3 replica with scipy (periodic Voronoi is not in scipy);
    faces are counted through the ridge-point adjacency, which is robust to
    degenerate vertices. Atoms beyond max_atoms are subsampled (breaking the
    lattice is acceptable: this is a fingerprint, not a symmetry probe)."""
    from scipy.spatial import Voronoi
    L = frame.cell_diag
    pos = np.mod(frame.pos, L)
    n = len(pos)
    if n > max_atoms:
        idx = np.linspace(0, n - 1, max_atoms).astype(int)
        pos = pos[idx]
    offsets = np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1],
                                   indexing="ij")).reshape(3, -1).T * L
    big = (pos[None, :, :] + offsets[:, None, :]).reshape(-1, 3)
    vor = Voronoi(big)
    n_inner = len(pos)
    start = 13 * n_inner
    degrees = np.bincount(vor.ridge_points.ravel(),
                          minlength=len(big))[start:start + n_inner]
    return Counter(int(x) for x in degrees)
