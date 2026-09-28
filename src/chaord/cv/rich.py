"""Rich held-out observables: species-resolved structure for the round trip.

The default set in chaord.cv.noise (all-species g(r) + one coordination
histogram) is blind to species structure: in molecular systems the
intramolecular O-H bond at 0.96 A dominates the whole-box g(r).  The
observables here resolve it:

  * partial_gr         — g(r) of one species pair; same-molecule pairs are
                         excluded under a dialect that sets
                         `partial_gr_exclude_intramolecular` (molecular), so
                         the delta-like intramolecular peaks cannot dominate;
                         atomic dialects use the raw atom pair list
  * angle_distribution — bond-angle density P(theta), compared between
                         frames by total variation
  * density_profile    — per-bin number density along one cell axis,
                         compared by RMS (sharper than a whole-box g(r) for
                         interface systems)

They compose with the configurable observable set:

    o = observables(frame, dialect,
                    selection=("gr", "partial_gr", "angle", "density"))
    d = distance(o_rebuilt, o_original, dialect=dialect,
                 keys=("gr_rms_O-O", "angle_tv"))

which is how the acceptance harness (tools/acceptance.py, A5) gates a
statistical round trip on any observable subset.  Every threshold comes
from the dialect by name (design rule 5).
"""
from __future__ import annotations

import numpy as np

from ..io.frames import Frame
from ..lang.errors import ChaordError
from ..realize.lj import mic, wrap


def _threshold(dialect, name, fallback: str | None = None):
    """Read a dialect threshold, falling back to a second name if absent."""
    try:
        return dialect.threshold(name)
    except ChaordError:
        if fallback is None:
            raise
        return dialect.threshold(fallback)


def _molecule_ids(frame: Frame, dialect):
    """Atom -> molecule index under the covalent-radius bond rule, or None
    when the dialect does not ask for intramolecular exclusion."""
    try:
        exclude = bool(dialect.threshold("partial_gr_exclude_intramolecular"))
    except ChaordError:
        return None
    if not exclude:
        return None
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    from ..build.molecules import bond_graph
    edges = bond_graph(frame, dialect)
    n = len(frame)
    if not edges:
        return np.arange(n)
    rows = [e[0] for e in edges] + [e[1] for e in edges]
    cols = [e[1] for e in edges] + [e[0] for e in edges]
    _, labels = connected_components(
        coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)),
        directed=False)
    return labels


def partial_gr(frame: Frame, dialect, species_a, species_b=None,
               rmax=None, bins=None):
    """Species-pair radial distribution function g_ab(r).

    Sites are the atoms of the two species.  Under a dialect that sets
    `partial_gr_exclude_intramolecular` (molecular), pairs of sites that
    belong to one molecule (a connected component of the bond graph) are
    dropped first: the intramolecular O-H peak at 0.96 A would otherwise
    dominate every molecular g(r).  Atomic dialects use the atom pair list
    directly.

    Returns ``(g, r)`` — the normalised pair distribution and the bin
    centres — with the ideal-gas reference of the two selections,
    g_ab = (2 if a == b else 1) * h_ab * V / (N_a N_b shell), identical to
    the all-species observables g(r) when both species select every atom.
    """
    from scipy.spatial import cKDTree

    b = species_a if species_b is None else species_b
    L = frame.cell_diag
    rmax = float(rmax if rmax is not None
                 else _threshold(dialect, "gr_rmax_fluid"))
    bins = int(bins if bins is not None
               else _threshold(dialect, "gr_bins_fluid"))
    syms = np.asarray(frame.symbols)
    n_a = int((syms == species_a).sum())
    n_b = int((syms == b).sum())
    if n_a == 0 or n_b == 0:
        raise ChaordError(
            f"species pair {species_a!r}-{b!r} is empty in this frame "
            f"(present: {', '.join(sorted(set(frame.symbols)))})")
    pos = wrap(frame.pos, L)
    tree = cKDTree(pos, boxsize=L)
    pairs = tree.query_pairs(rmax, output_type="ndarray")
    sel = ((syms[pairs[:, 0]] == species_a) & (syms[pairs[:, 1]] == b)) | \
          ((syms[pairs[:, 0]] == b) & (syms[pairs[:, 1]] == species_a))
    pairs = pairs[sel]
    mol = _molecule_ids(frame, dialect)
    if mol is not None and len(pairs):
        pairs = pairs[mol[pairs[:, 0]] != mol[pairs[:, 1]]]
    d = pos[pairs[:, 1]] - pos[pairs[:, 0]]
    d -= L * np.round(d / L)
    r = np.linalg.norm(d, axis=1)
    h, e = np.histogram(r, np.linspace(0, rmax, bins + 1))
    shell = 4 / 3 * np.pi * (e[1:] ** 3 - e[:-1] ** 3)
    weight = 2.0 if species_a == b else 1.0   # dialect-exempt: exact-geometry: unordered-pair factor
    g = weight * h * float(np.prod(L)) / (n_a * n_b * shell)
    return g, 0.5 * (e[1:] + e[:-1])          # dialect-exempt: numerical-guard: bin centres


def angle_distribution(frame: Frame, dialect, cutoff=None, bins=None):
    """Bond-angle distribution P(theta) over neighbour pairs at each atom.

    Neighbours of a central atom are the atoms within `cutoff`; every
    unordered neighbour pair at a centre with at least two neighbours
    contributes one angle.  `cutoff` defaults to the dialect's
    `angle_cutoff_fluid` with `cn_cutoff_fluid` as fallback — the molecular
    dialect uses the covalent O-H bond shell (so water shows the 104.5 deg
    H-O-H bend), atomic dialects the first coordination shell.

    Returns ``(theta_centres, density)`` in degrees, the density summing
    to one; the distance between two frames is the total variation of the
    density (see chaord.cv.noise.distance, key ``angle_tv``).
    """
    from ..lift.passes import pairs_within

    cutoff = float(cutoff if cutoff is not None
                   else _threshold(dialect, "angle_cutoff_fluid",
                                   "cn_cutoff_fluid"))
    bins = int(bins if bins is not None
               else _threshold(dialect, "angle_bins_fluid"))
    L = frame.cell_diag
    pos = wrap(frame.pos, L)
    pairs = pairs_within(pos, L, cutoff)
    nbrs: dict[int, list[int]] = {}
    for i, j in pairs:
        nbrs.setdefault(int(i), []).append(int(j))
        nbrs.setdefault(int(j), []).append(int(i))
    angles = []
    for i in sorted(nbrs):
        js = nbrs[i]
        if len(js) < 2:
            continue
        v = mic(pos[js] - pos[i], L)
        v = v / np.linalg.norm(v, axis=1)[:, None]
        c = v @ v.T
        iu = np.triu_indices(len(js), 1)
        angles.append(np.degrees(np.arccos(np.clip(c[iu], -1, 1))))
    edges = np.linspace(0.0, 180.0, bins + 1)  # dialect-exempt: exact-geometry: the angle domain
    if angles:
        h, _ = np.histogram(np.concatenate(angles), edges)
    else:
        h = np.zeros(bins)
    density = h / max(h.sum(), 1.0)  # dialect-exempt: numerical-guard: empty-frame normalisation
    return 0.5 * (edges[1:] + edges[:-1]), density  # dialect-exempt: numerical-guard: bin centres


def density_profile(frame: Frame, dialect, axis: int = 2, nbins=None):
    """Per-bin number density along one cell axis (default z).

    The interface observable: a solid-liquid slab shows the two-phase
    density gap and the solid's layering far more sharply than a whole-box
    g(r).  Returns the ``(nbins,)`` density in atoms per volume unit of
    the system; the distance between two frames is the profile RMS (see
    chaord.cv.noise.distance, key ``density_rms``).
    """
    L = frame.cell_diag
    axis = int(axis)
    if not 0 <= axis < 3:
        raise ChaordError(f"profile axis must be 0, 1 or 2, got {axis!r}")
    nbins = int(nbins if nbins is not None
                else _threshold(dialect, "density_profile_bins"))
    x = np.mod(frame.pos[:, axis], L[axis])
    h, _ = np.histogram(x, np.linspace(0.0, L[axis], nbins + 1))  # dialect-exempt: exact-geometry: profile starts at the cell edge
    others = [i for i in range(3) if i != axis]
    bin_volume = (L[axis] / nbins) * L[others[0]] * L[others[1]]
    return h / bin_volume
