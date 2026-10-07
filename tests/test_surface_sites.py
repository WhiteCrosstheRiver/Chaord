"""O4 (Review 6): A8's site classifier must respect the periodic boundaries.

Three red reproducers from docs/reviews/open_items.md, written red first:

1. random in-plane translations of the planted frame change the site census
   (3 of 20 today claim top 6 / bridge 2);
2. a physical frame with O at fcc/hcp hollow sites of a Pt(111) slab never
   classifies all 8 O as hollow (0 of 10 draws today);
3. a slab whose bottom layer crosses the periodic z boundary lifts part of
   its own bottom layer as ``adsorb Pt ... site top``.

Plus the W9 reproducer (open_items_v2.md): on Pt(100) the 4-fold hollow is
the midpoint of a Delaunay triangle edge, so the feature classifier calls it
``bridge``; the site class must come from lateral coordination instead.

fcc and hcp hollows are the same ``hollow`` statement in the language (the
classifier only sees the top layer); separating them is a spec change for the
owner, not something these tests assume.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src"), str(ROOT / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from acceptance import _plated_adsorbate_frame, format_program_text  # noqa: E402
from chaord.dialects import load_dialect  # noqa: E402
from chaord.io.frames import Frame  # noqa: E402
from chaord.lift import lift_frame  # noqa: E402

A_PT = 3.92                      # Pt lattice constant, A (verifier geometry)
ANN = A_PT / np.sqrt(2.0)        # (111) in-plane NN distance
D111 = A_PT / np.sqrt(3.0)       # (111) interlayer spacing


@pytest.fixture(scope="module")
def dialect():
    return load_dialect(("core", "metal", "surface"))


def _shifted(f, t):
    """The same frame, rigidly translated by t (A) and wrapped into the box."""
    return Frame(pos=np.mod(f.pos + np.asarray(t, float), f.cell_diag),
                 cell=f.cell, symbols=f.symbols, pbc=f.pbc)


def _site_claims(frame, dialect):
    """(per-site-type adsorb counts, full program text) of the lifted frame."""
    text = format_program_text(lift_frame(frame, dialect))
    claims = {}
    for m in re.finditer(r"adsorb (\S+) count (\d+) site (\S+)", text):
        claims[m.group(3)] = claims.get(m.group(3), 0) + int(m.group(2))
    return claims, text


def _hollow_frame(kind, seed, z_bottom=1.5):
    """A physical Pt(111) slab with 8 O at a p(2x2) subset of hollow sites.

    Same verifier-side fcc(111) parametrisation as ``_plated_adsorbate_frame``
    (absolute rows; layer k owns rows r = k mod 3; odd rows shifted by half
    the in-plane NN distance), on an even cell count (4 x 8 cells) so the
    p(2x2) site set is a true sublattice. ``kind`` = 'fcc' puts O above the
    lateral positions of the third layer from the top (fcc hollows, the ABC
    continuation), 'hcp' above the second layer from the top (hcp hollows).
    Every atom gets 0.08 A Gaussian motion; the slab is kept clear of z = 0.
    """
    nx, n_rows, n_layers = 4, 8, 4
    Lx = nx * ANN
    Ly = (n_rows // 2) * ANN * np.sqrt(3.0)
    Lz = 18.79
    step = ANN * np.sqrt(3.0) / 6.0
    layer_xy = [[] for _ in range(n_layers)]
    pos, syms = [], []
    for k in range(n_layers):
        z = z_bottom + k * D111
        r = k % 3
        while r * step < Ly - 1e-9:
            row_off = (ANN / 2.0) if r % 2 else 0.0
            for i in range(nx):
                xy = np.mod([i * ANN + row_off, r * step], [Lx, Ly])
                pos.append([xy[0], xy[1], z])
                syms.append("Pt")
                layer_xy[k].append(xy)
            r += 3
    # p(2x2) sublattice of the hollow layer: primitive net vectors of one
    # layer are a1 = (ANN, 0) and a2 = (-ANN/2, 3*step); every 2nd along each
    # gives (nx/2) * (n_rows/2) = 8 sites, pairwise >= 2 a_NN apart
    hollow_k = {"fcc": n_layers - 3, "hcp": n_layers - 2}[kind]
    a1 = np.array([ANN, 0.0])
    a2 = np.array([-ANN / 2.0, 3.0 * step])
    base = np.asarray(layer_xy[hollow_k][0], float)
    z_top = z_bottom + (n_layers - 1) * D111
    for j in range(n_rows // 2):
        for i in range(nx // 2):
            xy = np.mod(base + 2.0 * (j * a2 + i * a1), [Lx, Ly])
            pos.append([xy[0], xy[1], z_top + 1.2])
            syms.append("O")
    rng = np.random.default_rng(seed)
    pos = np.asarray(pos, float) + rng.normal(0.0, 0.08, (len(pos), 3))
    return Frame(pos=np.mod(pos, [Lx, Ly, Lz]), cell=np.diag([Lx, Ly, Lz]),
                 symbols=syms, pbc=(True, True, True))


# ---------------------------------------------------------------- 1. census --
def test_site_census_invariant_under_in_plane_translations(dialect):
    """O4 reproducer 1: 3 of 20 random in-plane translations change the site
    census (they claim top 6 / bridge 2). With the top layer replicated over
    its neighbouring periodic images before triangulation, all 20 lifts must
    reproduce the planted census."""
    f0, truth = _plated_adsorbate_frame()
    L = f0.cell_diag
    rng = np.random.default_rng(0)
    wrong = 0
    for _ in range(20):
        g = _shifted(f0, (rng.uniform(0, L[0]), rng.uniform(0, L[1]), 0.0))
        claims, _ = _site_claims(g, dialect)
        wrong += claims != truth
    assert wrong == 0, f"{wrong} of 20 translations change the site census"


# -------------------------------------------------------- 2. physical frame --
@pytest.mark.parametrize("kind", ["fcc", "hcp"])
def test_hollow_adsorbates_classified_hollow(kind, dialect):
    """O4 reproducer 2: O at true fcc/hcp hollow sites, 1.2 A above the top
    layer, with 0.08 A thermal motion, must lift as ``site hollow`` at least
    90% of the time over 20 draws x 20 in-plane translations (today 0 of 10
    draws get all 8 right)."""
    rng = np.random.default_rng(11)
    good = total = 0
    for draw in range(20):
        f = _hollow_frame(kind, seed=100 + draw)
        L = f.cell_diag
        for _ in range(20):
            g = _shifted(f, (rng.uniform(0, L[0]), rng.uniform(0, L[1]), 0.0))
            claims, _ = _site_claims(g, dialect)
            good += claims.get("hollow", 0)
            total += 8
    assert total == 20 * 20 * 8
    assert good / total >= 0.90, f"{kind}: only {good}/{total} O called hollow"


# ------------------------------------------------------- 3. z boundary slab --
def test_slab_crossing_the_z_boundary_lifts_unchanged(dialect):
    """O4 reproducer 3: rigidly moving the plated slab so its bottom layer
    crosses z = 0 used to lift that layer as ``adsorb Pt ... site top``. The
    lift must be byte-identical to the unshifted frame (z judged on wrapped,
    gap-aligned coordinates)."""
    f0, _ = _plated_adsorbate_frame()
    base = _site_claims(f0, dialect)[1]
    for dz in (-0.8, -3.0, 2.5):
        text = _site_claims(_shifted(f0, (0.0, 0.0, dz)), dialect)[1]
        assert text == base, f"a rigid z shift of {dz} A changed the lift"


def test_hollow_frame_crossing_the_z_boundary_lifts_unchanged(dialect):
    """The same invariance for a thermal physical frame (0.08 A motion)."""
    f = _hollow_frame("fcc", seed=5)
    base = _site_claims(f, dialect)[1]
    assert "site hollow" in base
    text = _site_claims(_shifted(f, (0.0, 0.0, -1.8)), dialect)[1]
    assert text == base, "crossing z = 0 changed the lift of the physical frame"


# ------------------------------------------- planted-frame physical sanity --
def test_plated_adsorbates_keep_physical_o_o_separation():
    """O4 3a: 6 of the 8 planted O were 1.35-1.39 A apart (the top-site pick
    took neighbouring sites). Under the minimum image convention every
    planted O pair must now stay >= 2 a_NN (two in-plane lattice spacings)
    apart, so the A8 input itself passes the reference-data sanity rule."""
    from scipy.spatial import cKDTree
    f, truth = _plated_adsorbate_frame()
    assert truth == {"top": 5, "bridge": 3}
    L = f.cell_diag
    o = np.mod(f.pos[[i for i, s in enumerate(f.symbols) if s == "O"]], L)
    assert len(o) == 8
    d = cKDTree(o, boxsize=L).query(o, k=2)[0][:, 1]
    # the verifier places the ideal sites >= 2 a_NN apart and then jitters
    # each by up to 0.03 A per axis; two jitter vectors close a pair by at
    # most 2*sqrt(2)*0.03 A
    floor = 2 * ANN - 2 * np.sqrt(2) * 0.03
    assert d.min() >= floor, f"closest planted O pair {d.min():.3f} A"


# ------------------------------------------------- 4. W9: Pt(100) 4-fold sites --
A_PT100 = 3.92                       # Pt lattice constant, A (reviewer geometry)
ANN100 = A_PT100 / np.sqrt(2.0)      # (100) in-plane NN distance, A

# 4-fold hollows of the (100) square net: cell-centre offsets (in a_NN) from
# one top-layer atom; the reviewer's verbatim reproducer set. The reference
# atom sits at (0.5, 0.5) in net units, so hollow offsets keep both
# components half-odd (abs = integers) and bridge offsets make exactly one
# component an odd integer multiple of 0.5 (edge midpoints).
W9_HOLLOWS = ((0.5, 0.5), (2.5, 2.5), (2.5, 4.5), (4.5, 0.5))
# bridges: midpoints of nearest-neighbour edges of the same net, kept
# >= 2 a_NN apart like every planted adsorbate set in this file
W9_BRIDGES = ((0.5, 0.0), (2.5, 0.0), (0.5, 2.0), (2.5, 2.0))


def _pt100_frame(offsets, seed):
    """The W9 Pt(100) slab (reviewer parametrisation, verbatim): 4 layers of
    a 6x6 square net, alternate layers shifted by half the in-plane NN
    distance (fcc(100) stacking), with 4 O at ``offsets`` (in units of the
    in-plane NN distance) from one top-layer atom, 1.0 A above it, and 0.08 A
    Gaussian noise on every atom; z kept clear of the periodic boundary."""
    a, ann, n, nl = A_PT100, ANN100, 6, 4
    P = np.array([[((i + 0.5 * (k % 2)) * ann) % (n * ann),
                   ((j + 0.5 * (k % 2)) * ann) % (n * ann), k * a / 2]
                  for k in range(nl) for i in range(n) for j in range(n)])
    L = np.array([n * ann, n * ann, 22.0])
    top = P[np.isclose(P[:, 2], (nl - 1) * a / 2)][0, :2]
    hol = np.mod(top + np.asarray(offsets, float) * ann, L[:2])
    O = np.column_stack([hol, np.full(len(offsets), (nl - 1) * a / 2 + 1.0)])
    rng = np.random.default_rng(seed)
    pos = np.vstack([P, O]) + rng.normal(scale=0.08,
                                         size=(len(P) + len(offsets), 3))
    pos[:, 2] += 3.0
    return Frame(pos=np.mod(pos, L), cell=np.diag(L),
                 symbols=["Pt"] * len(P) + ["O"] * len(offsets),
                 pbc=(True,) * 3)


def test_w9_pt100_fourfold_hollow_reproducer(dialect):
    """W9 verbatim reproducer (seed 0): the 4-fold hollow of Pt(100) is the
    midpoint of a Delaunay triangle's hypotenuse, so the feature classifier
    claims ``site bridge`` for all 4 O. Classifying by lateral coordination
    (4 top-layer neighbours equidistant from each O) must claim hollow."""
    claims, text = _site_claims(_pt100_frame(W9_HOLLOWS, seed=0), dialect)
    assert re.findall(r"adsorb \S+ count \d+ site \S+", text) == \
        ["adsorb O count 4 site hollow"], text
    assert claims.get("hollow", 0) == 4, f"claims {claims}"


@pytest.mark.parametrize("offsets,site", [(W9_HOLLOWS, "hollow"),
                                          (W9_BRIDGES, "bridge")],
                         ids=["hollow", "bridge"])
def test_pt100_sites_over_20_draws_20_translations(offsets, site, dialect):
    """W9 done-when: Pt(100) 4-fold hollows and bridges >= 90% correct over
    20 noise draws x 20 random in-plane translations (1600 site calls each).
    Pre-fix the hollow arm is 0/1600 (every 4-fold hollow is the midpoint of
    a Delaunay triangle edge -> bridge); the bridge arm already passed and
    guards the coordination override against over-widening (a window past
    0.30 a_NN would pull a bridge's 3rd shell, 0.366 a_NN above d_min, into
    the 3-count hollow override). Measured post-fix: hollow 1600/1600,
    bridge 1600/1600."""
    rng = np.random.default_rng(23)
    good = total = 0
    for draw in range(20):
        f = _pt100_frame(offsets, seed=400 + draw)
        L = f.cell_diag
        for _ in range(20):
            g = _shifted(f, (rng.uniform(0, L[0]), rng.uniform(0, L[1]), 0.0))
            claims, _ = _site_claims(g, dialect)
            good += claims.get(site, 0)
            total += len(offsets)
    assert total == 20 * 20 * len(offsets)
    assert good / total >= 0.90, \
        f"Pt(100) {site}: only {good}/{total} correct"
