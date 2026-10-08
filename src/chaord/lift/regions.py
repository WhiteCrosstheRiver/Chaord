"""Region plans: stage S2 of the segment-first pipeline (lift/build v2 stage 1).

``segment_regions`` turns a frame into a list of :class:`RegionPlan` objects --
one per phase (same-phase connected components are merged in stage 1, design
2.2 "stage 1 may keep grains merged") plus one per vacuum gap -- using the
existing kernels: 3-D labels from ``segment.phase_labels_3d``, connected
components on the same neighbour graph the segmentation floods across, and the
z-gap scan generalising ``surface.has_vacuum`` (it returns the gap's extent,
not just a bool). Nothing here fits a model; the pipeline
(``lift/pipeline.py``) owns model choice, fitting and assembly.

Region names derive from composition and geometry only, never from atom
indices (design doc, approved change #1): a re-ordered frame produces the same
names in the same order.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..io.frames import Frame
from ..lang.ir import GeoChain, Quantity, RangeVal, ShAll, ShSlab
from .crystal import quantize_coords
from .segment import _robust_neighbor_distance, dialect_profile_bin

# canonical region order: crystal first, vacuum last (design 2.7)
PHASE_PRIORITY = {"crystal": 0, "amorphous": 1, "liquid": 2, "gas": 3, "vacuum": 4}

# a single region that covers the frame keeps the legacy per-phase name, so
# stage-1 single-region output stays byte-identical to the legacy cascade
_SINGLE_NAMES = {"crystal": "bulk", "amorphous": "glass", "liquid": "fluid",
                 "gas": "fluid"}


@dataclass
class RegionPlan:
    """One region of a segmented frame, before model fitting.

    ``phase`` is the segmentation call (crystal = solid-like, liquid =
    disordered, vacuum = empty gap); the pipeline's model selection may refine
    it (a scattered solid-noise island, say, is re-called liquid by the model
    table's compactness rule). ``model`` and ``statements`` are filled by the
    pipeline's S3-S5, not here. ``n_components`` is the number of connected
    pieces merged into this plan (the stage-1 compactness diagnostic).
    """
    name: str
    phase: str
    geometry: GeoChain
    atom_indices: np.ndarray
    model: str = ""
    statements: list = field(default_factory=list)
    n_components: int = 1

    def __len__(self):
        return len(self.atom_indices)


def subset_frame(frame: Frame, idx) -> Frame:
    """A region's atoms as a frame: same cell, same pbc (design 2.2)."""
    idx = np.asarray(idx, int)
    return Frame(pos=frame.pos[idx], cell=frame.cell,
                 symbols=[frame.symbols[i] for i in idx],
                 pbc=frame.pbc)


def slab_geometry(lo: float, hi: float, axis: str = "z") -> GeoChain:
    """slab axis lo .. hi, printed at the legacy M0 rounding."""
    rng = RangeVal(lo=Quantity(num=f"{lo:.1f}"), hi=Quantity(num=f"{hi:.1f}"))
    return GeoChain(parts=[ShSlab(axis=axis, rng=rng)], ops=[])


def all_geometry() -> GeoChain:
    return GeoChain(parts=[ShAll()], ops=[])


def _circular_empty_run(occ: np.ndarray) -> tuple[int, int]:
    """(start, length) of the longest circular run of empty bins.

    Length is capped at n (a fully empty axis is one gap wrapping around);
    ties keep the first start, so the result depends only on the occupancy
    pattern, never on atom order."""
    n = len(occ)
    best_len = best_start = 0
    run = start = 0
    for i in range(2 * n):
        if not occ[i % n]:
            if run == 0:
                start = i % n
            run += 1
            if run > best_len:
                best_len, best_start = run, start
                if best_len >= n:  # everything is empty: one wrapping gap
                    break
        else:
            run = 0
    return best_start, min(best_len, n)


def _bins(values: np.ndarray, L: float, dialect) -> np.ndarray:
    binw = float(dialect_profile_bin(dialect))
    n = max(int(L / binw), 8)
    return np.clip((values / L * n).astype(int), 0, n - 1)


def wrapped_interval(dialect, values: np.ndarray, L: float) -> tuple[float, float]:
    """Occupied interval of one region along an axis, wrapping allowed.

    Returns (lo, hi) with lo > hi when the region wraps through the periodic
    boundary; (0, L) when every bin is occupied."""
    n = max(int(L / float(dialect_profile_bin(dialect))), 8)
    binw = L / n
    occ = np.bincount(_bins(values, L, dialect), minlength=n) > 0
    start, length = _circular_empty_run(occ)
    if length == 0:
        return 0, float(L)
    occupied_from = (start + length) % n      # first occupied bin after the gap
    occupied_bins = n - length
    lo = occupied_from * binw
    hi = (occupied_from + occupied_bins) * binw  # may exceed L: wraps
    if hi > L:
        hi -= L                                   # legacy wrapped-slab convention
    return float(lo), float(hi)


def vacuum_gap(frame: Frame, dialect):
    """The widest empty z-slab wider than the dialect's vacuum minimum, or None.

    Same occupancy rule as ``surface.has_vacuum`` (z bins at the profile bin
    width, circular scan); returns the empty interval (lo, hi) so the vacuum
    region can carry a geometry."""
    from ..lang.errors import ChaordError
    try:
        gap_min = float(dialect.threshold("vacuum_gap_min"))
    except ChaordError:
        return None                   # no surface dialect: no vacuum concept
    L = frame.cell_diag[2]
    n = max(int(L / float(dialect_profile_bin(dialect))), 8)
    occ = np.bincount(_bins(frame.pos[:, 2], L, dialect), minlength=n) > 0
    start, length = _circular_empty_run(occ)
    if length == 0 or length * (L / n) < gap_min:
        return None
    lo = start * (L / n)
    hi = (start + length) * (L / n)
    return float(lo), float(min(hi, L))


def _neighbour_pairs(frame: Frame, radius: float) -> np.ndarray:
    """Pair list of the region-growing graph (the S2 flooding radius)."""
    from scipy.spatial import cKDTree
    from .defects import _wrap_strict
    L = frame.cell_diag
    pos = _wrap_strict(frame.pos, L)
    return cKDTree(pos, boxsize=L).query_pairs(radius, output_type="ndarray")


def _components(frame: Frame, labels: np.ndarray, pairs: np.ndarray) -> dict:
    """Connected components of each label class on the neighbour graph.

    Returns {label_value: [atom_index_array, ...]} sorted by size (largest
    first); ties broken by the component's smallest wrapped coordinate sum, so
    the order depends on geometry, never on atom indexing."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(frame)
    if len(pairs):
        adj = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                         shape=(n, n))
        _, lab = connected_components(adj, directed=False)
    else:
        lab = np.arange(n)
    pos = np.mod(frame.pos, frame.cell_diag)
    out = {}
    for value in (True, False):
        idx = np.where(labels == value)[0]
        if len(idx) == 0:
            out[value] = []
            continue
        comps = []
        for c in np.unique(lab[idx]):
            m = idx[lab[idx] == c]
            # W10: quantized (1e-6 A) keys on the geometry tiebreak
            key = float(np.sort(quantize_coords(pos[m].sum(axis=1)))[0])
            comps.append((m, len(m), key))
        comps.sort(key=lambda t: (-t[1], t[2]))
        out[value] = [m for m, _k, _t in comps]
    return out


def composition(frame: Frame, idx) -> str:
    """Sorted Hill-style composition string of a region's atoms."""
    counts: dict[str, int] = {}
    for i in idx:
        s = frame.symbols[i]
        counts[s] = counts.get(s, 0) + 1
    return "".join(f"{s}{counts[s]}" for s in sorted(counts))


def _assign_names_and_geometry(regions, frame: Frame, dialect):
    """Names and geometries from composition + geometry (approved change #1).

    A single region covering the frame keeps the legacy per-phase name; a
    multi-region frame is named A, B, C ... in canonical order (phase priority,
    then the wrapped z interval, then composition) -- never by atom index."""
    atom_regions = [r for r in regions if len(r)]
    single = (len(atom_regions) == 1 and len(atom_regions[0]) == len(frame))
    for r in regions:
        if r.phase == "vacuum":
            continue                      # geometry set at construction
        if single:
            r.geometry = all_geometry()
            continue
        lo, hi = wrapped_interval(dialect, frame.pos[r.atom_indices, 2],
                                  frame.cell_diag[2])
        r.geometry = slab_geometry(lo, hi)
    if single:
        atom_regions[0].name = _SINGLE_NAMES.get(atom_regions[0].phase, "bulk")
        return
    def key(r):
        if len(r) == 0:
            geo = (r.phase,)              # vacuum: after every atom region
        else:
            lo, hi = wrapped_interval(dialect, frame.pos[r.atom_indices, 2],
                                      frame.cell_diag[2])
            start = lo if lo <= hi else lo - frame.cell_diag[2]
            geo = (start, hi - lo if hi >= lo else hi - lo + frame.cell_diag[2])
        return (PHASE_PRIORITY.get(r.phase, len(PHASE_PRIORITY)),
                *geo, composition(frame, r.atom_indices))
    for k, r in enumerate(sorted(regions, key=key)):
        r.name = chr(ord("A") + k)        # canonical letters, index-free order


def segment_regions(frame: Frame, dialect, labels: np.ndarray | None = None) -> list:
    """Segment a frame into RegionPlans (design 2.2 S2, stage-1 rules).

    3-D solid/disordered labels come from ``segment.phase_labels_3d``; each
    label class becomes ONE region (same-phase components are merged, stage 1)
    carrying its component count as the compactness diagnostic; a vacuum
    z-gap wider than the dialect minimum becomes an atom-less vacuum region.
    Atom indices are sorted ascending: the plan is a deterministic function of
    the atom set, not of the frame's ordering."""
    if labels is None:
        from .segment import phase_labels_3d
        labels = phase_labels_3d(frame, dialect)
    factor = float(dialect.threshold("segment_vote_factor"))
    radius = factor * _robust_neighbor_distance(frame)
    pairs = _neighbour_pairs(frame, radius)
    comps = _components(frame, labels, pairs)

    regions = []
    for value, phase in ((True, "crystal"), (False, "liquid")):
        pieces = comps[value]
        if not pieces:
            continue
        idx = np.sort(np.concatenate(pieces))
        regions.append(RegionPlan(name="", phase=phase, geometry=all_geometry(),
                                  atom_indices=idx, n_components=len(pieces)))
    vac = vacuum_gap(frame, dialect)
    if vac is not None:
        regions.append(RegionPlan(name="", phase="vacuum",
                                  geometry=slab_geometry(vac[0], vac[1]),
                                  atom_indices=np.array([], int)))
    _assign_names_and_geometry(regions, frame, dialect)
    return regions


def region_contacts(frame: Frame, dialect, regions) -> set:
    """Pairs of region positions (into ``regions``) whose atoms bond."""
    where = np.full(len(frame), -1, int)
    for k, r in enumerate(regions):
        where[r.atom_indices] = k
    factor = float(dialect.threshold("segment_vote_factor"))
    radius = factor * _robust_neighbor_distance(frame)
    pairs = _neighbour_pairs(frame, radius)
    contacts = set()
    for i, j in pairs:
        a, b = int(where[i]), int(where[j])
        if a >= 0 and b >= 0 and a != b:
            contacts.add(frozenset((a, b)))
    return contacts
