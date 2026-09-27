"""M4 acceptance: true 3-D phase segmentation and interface meshes.

``test_segmentation_labels_planted`` in test_surfaces.py plants a solid/liquid
interface perpendicular to z. Here the planted boundary is the plane with
normal (1,1,1)/sqrt(3): per-atom truth compares (x+y+z)/sqrt(3) against the
mid-sum threshold and liquid atoms are scattered on the liquid side, so the
interface runs diagonally through the box with no relation to any box axis.

Two geometry facts drive the tests below:
* a diagonal cut of a periodic box glues pieces of the same plane onto the box
  faces (its periodic images), so the exclusion zone keeps every atom within
  2 d_NN of the planted plane *or* of a box face, mirroring how the z-case
  test excludes both its mid-plane and its z = 0 wrap;
* the 1-D segmentation the current pipeline actually uses is the z-profile
  phase call of lift/slab.py (pass 1: smoothed per-bin solid fraction against
  phase_solid_fraction). ``phase_labels`` itself (q6 mask + majority vote) is
  orientation-free, so it cannot demonstrate the difference; it is compared
  against phase_labels_3d directly in the z-interface regression test.
"""
import numpy as np
import pytest

from chaord.build.crystal import build_conventional
from chaord.build.defects import typical_neighbor_distance
from chaord.dialects import load_dialect
from chaord.io.frames import Frame
from chaord.lift.passes import qbar
from chaord.lift.segment import (
    interface_mesh,
    label_accuracy,
    phase_labels,
    phase_labels_3d,
)

A_FCC = 3.615  # Cu lattice constant used by the planted frames
N_111 = np.ones(3) / np.sqrt(3.0)  # planted interface normal


@pytest.fixture(scope="module")
def dialect():
    return load_dialect(("core", "metal"))


def _wrap(pos, L):
    w = pos - L * np.floor(pos / L)
    return np.minimum(w, L * (1 - 1e-9))


def _planted_tilted(seed, reps):
    """fcc solid below the (1,1,1) mid-sum plane, random liquid above it."""
    rng = np.random.default_rng(seed)
    solid = build_conventional("fcc", {"a": A_FCC}, ("Cu",), reps)
    L = solid.cell_diag
    f = solid.pos.sum(axis=1)
    f0 = 0.5 * L.sum()
    truth = f < f0                              # solid: (x+y+z) below the mid sum
    liquid = []
    need = int((~truth).sum())
    while need > 0:
        cand = rng.uniform(0, L, (3 * need, 3))
        cand = cand[cand.sum(axis=1) >= f0]     # liquid side of the tilted plane
        liquid.append(cand[: min(need, len(cand))])
        need -= len(liquid[-1])
    pos = solid.pos.copy()
    pos[~truth] = np.vstack(liquid)
    frame = Frame(pos=_wrap(pos, L), cell=solid.cell, symbols=solid.symbols)
    return frame, truth, f0


def _tilted_core(frame, f0):
    """Atoms farther than 2 d_NN from the planted interface and its images."""
    L = frame.cell_diag
    d_nn = typical_neighbor_distance(frame)
    d_face = np.minimum(frame.pos, L - frame.pos).min(axis=1)
    d_plane = np.abs(frame.pos.sum(axis=1) - f0) / np.sqrt(3.0)
    return (d_face > 2.0 * d_nn) & (d_plane > 2.0 * d_nn)


def _profile_labels_1d(frame, dialect):
    """The 1-D z-profile phase call (the pass-1 rule of lift/slab.py)."""
    rc = float(dialect.threshold("q6_cutoff"))
    thr = float(dialect.threshold("q6_solid"))
    L = frame.cell_diag
    pos = np.mod(frame.pos, L)
    q6, _, _ = qbar(pos, L, rc=rc)
    bin_w = float(dialect.threshold("profile_bin_size"))
    k = int(dialect.threshold("profile_smooth_wide"))
    level = float(dialect.threshold("phase_solid_fraction"))
    n = int(L[2] / bin_w)
    idx = np.minimum((pos[:, 2] / L[2] * n).astype(int), n - 1)
    solid = np.bincount(idx, (q6 > thr).astype(float), n)
    total = np.bincount(idx, minlength=n).astype(float)
    sm = lambda a, w: np.convolve(np.r_[a[-w:], a, a[:w]], np.ones(w), "same")[w:-w]
    frac = sm(solid, k) / np.maximum(sm(total, k), 1e-9)
    return frac[idx] > level


def test_tilted_interface_labels_3d(dialect):
    """3-D segmentation labels a (1,1,1) interface; the z-profile call cannot.

    The box is wide in x, y and short in z, so the tilted plane decorrelates
    z from the phase: every z-bin holds both phases and the 1-D profile
    crossing sits at mid-z, mislabelling every atom whose height disagrees
    with its (x+y+z)."""
    frame, truth, f0 = _planted_tilted(seed=31, reps=(7, 7, 4))
    core = _tilted_core(frame, f0)
    acc_3d = label_accuracy(phase_labels_3d(frame, dialect)[core], truth[core])
    acc_1d = label_accuracy(_profile_labels_1d(frame, dialect)[core], truth[core])
    assert core.sum() >= 50, f"exclusion left only {core.sum()} core atoms"
    assert acc_3d >= 0.95, f"3-D label accuracy {acc_3d:.3f}"
    assert acc_3d - acc_1d >= 0.1, f"3d {acc_3d:.3f} vs 1d z-profile {acc_1d:.3f}"


def test_tilted_interface_mesh_normals(dialect):
    """Interface normals point along (1,1,1)/sqrt(3) within 20 degrees (mean).

    Judged on interface atoms deeper than one contact radius plus one normal
    radius from every face: the periodic images of the plane are glued onto
    the faces, where unlike pairs cross the face nearly face-normal, so the
    centroid estimator is only meaningful in the box interior. The box is a
    thick cube so the interior still holds a full patch of the plane."""
    frame, truth, f0 = _planted_tilted(seed=31, reps=(9, 9, 8))
    labels = phase_labels_3d(frame, dialect)
    mesh = interface_mesh(labels, frame, dialect)
    idx, nrm = mesh["indices"], mesh["normals"]
    assert len(idx) > 0
    lens = np.linalg.norm(nrm, axis=1)
    assert np.all(np.isclose(lens, 1.0) | np.isclose(lens, 0.0))
    L = frame.cell_diag
    deep = (np.minimum(frame.pos[idx], L - frame.pos[idx]).min(axis=1)
            > mesh["radius"] + mesh["normal_radius"])
    n = nrm[deep][lens[deep] > 0.0]
    assert len(n) >= 10, f"only {len(n)} interior interface normals"
    ang = np.degrees(np.arccos(np.clip(np.abs(n @ N_111), 0.0, 1.0)))
    assert ang.mean() <= 20.0, f"mean normal angle {ang.mean():.1f} deg"


def test_z_interface_regression_vs_phase_labels(dialect):
    """Backwards compatibility: on a z interface (the test_surfaces.py case)
    the 3-D segmentation matches the orientation-free phase_labels rule."""
    rng = np.random.default_rng(31)
    solid = build_conventional("fcc", {"a": A_FCC}, ("Cu",), (4, 4, 5))
    L = solid.cell_diag
    liquid_planted = solid.pos[:, 2] > L[2] / 2
    pos = solid.pos.copy()
    n_liq = int(liquid_planted.sum())
    upper = rng.uniform(0, L, (n_liq, 3))
    upper[:, 2] = L[2] / 2 + rng.uniform(0, L[2] / 2 - 0.5, n_liq)
    pos[liquid_planted] = upper
    frame = Frame(pos=_wrap(pos, L), cell=solid.cell, symbols=solid.symbols)
    truth = ~liquid_planted                   # solid below mid-height
    d_nn = typical_neighbor_distance(frame)
    z = frame.pos[:, 2]
    core = ((np.abs(z - L[2] / 2) > 2.0 * d_nn)
            & (z > 2.0 * d_nn) & (z < L[2] - 2.0 * d_nn))
    acc_1d = label_accuracy(phase_labels(frame, dialect)[core], truth[core])
    acc_3d = label_accuracy(phase_labels_3d(frame, dialect)[core], truth[core])
    assert acc_3d >= 0.95, f"3-D label accuracy {acc_3d:.3f}"
    assert acc_3d - acc_1d >= -0.05, f"3d {acc_3d:.3f} vs phase_labels {acc_1d:.3f}"
