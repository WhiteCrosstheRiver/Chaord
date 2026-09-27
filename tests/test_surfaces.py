"""M4 acceptance: segmentation, interfaces, surfaces, adsorbates.

Exit criteria (PLAN M4): interface position within 0.5 A; per-atom phase labels
>= 95% correct; adsorption sites >= 90% correct.
"""
import numpy as np
import pytest

from chaord.build import build_program
from chaord.dialects import load_dialect
from chaord.io.frames import Frame, read_frame
from chaord.lang.api import load
from chaord.lang.fmt import format_program
from chaord.lift import lift_frame
from chaord.lift.segment import label_accuracy, phase_labels, slab_interfaces

SURF_PROGRAM = """chaord 0.1
dialect core + metal + surface

system {{
  cell {lx:.3f} {ly:.3f} {lz:.3f}
  pbc xyz
  conserve atoms Ni {n}
}}

physics {{
  backend eam
}}

crystal slab : slab z 0 .. {ztop:.1f} {{
  lattice fcc
  a 3.615 A
  surface ({hkl}) top
  adsorb O count 4 site top
}}

vacuum gap : slab z {ztop:.1f} .. {lz:.1f} {{
}}
"""


@pytest.fixture(scope="module")
def dialect():
    return load_dialect(("core", "metal", "surface"))


@pytest.fixture(scope="module")
def surf111(dialect, tmp_path_factory):
    text = SURF_PROGRAM.format(lx=7.669, ly=6.641, lz=18.26, n=36, hkl="111", ztop=6.3)
    path = tmp_path_factory.mktemp("m4") / "surf111.chaord"
    path.write_text(text)
    frame = build_program(load(path), dialect, rng=np.random.default_rng(3))
    return path, frame


def _wrap(pos, L):
    w = pos - L * np.floor(pos / L)
    return np.minimum(w, L * (1 - 1e-9))


def test_surface_geometry(surf111, dialect):
    from scipy.spatial import cKDTree
    _, f = surf111
    L = f.cell_diag
    slab = np.array([p for s, p in zip(f.symbols, f.pos) if s == "Ni"])
    assert len(slab) == 36  # ASE 3x3x4 fcc(111) slab
    # MIC-aware nearest-neighbour distance (the in-plane cell is hexagonal)
    frac = slab @ np.linalg.inv(f.cell)
    nnd = []
    for i in range(len(frac)):
        d = frac - frac[i]
        d -= np.round(d)
        cart = d @ f.cell
        dist = np.linalg.norm(cart, axis=1)
        dist[dist < 1e-9] = np.inf
        nnd.append(dist.min())
    assert min(nnd) == pytest.approx(3.615 / np.sqrt(2), abs=1e-3)
    assert float(f.cell[2, 2]) > float(slab[:, 2].max()) + 8.0  # vacuum present
    # adsorbates sit on top sites at the dialect height
    top_z = slab[:, 2].max()
    ads = _wrap(np.array([p for s, p in zip(f.symbols, f.pos) if s == "O"]), L)
    assert len(ads) == 4
    assert np.allclose(ads[:, 2] - top_z, 2.0, atol=1e-6)
    dist, _ = cKDTree(slab[np.abs(slab[:, 2] - top_z) < 0.5][:, :2]).query(ads[:, :2])
    assert dist.max() < 0.1  # directly above surface atoms


def test_surface_lift(surf111, dialect):
    path, f = surf111
    text = format_program(lift_frame(f, dialect, mode="surface"))
    assert "lattice fcc" in text
    assert "a 3.615 A" in text
    assert "surface (111) top" in text
    assert "termination Ni" in text
    assert "adsorb O count 4 site top" in text
    assert "conserve atoms Ni 36" in text


@pytest.mark.parametrize("hkl,lx,ly", [("001", 10.845, 10.845), ("110", 10.225, 7.230)])
def test_other_surfaces_round_trip(hkl, lx, ly, dialect, tmp_path):
    text = SURF_PROGRAM.format(lx=lx, ly=ly, lz=41.22, n=300, hkl=hkl, ztop=25.0)
    path = tmp_path / f"surf{hkl}.chaord"
    path.write_text(text)
    f = build_program(load(path), dialect, rng=np.random.default_rng(7))
    lifted = format_program(lift_frame(f, dialect, mode="surface"))
    assert f"surface ({hkl}) top" in lifted


def test_segmentation_labels_planted(dialect):
    """Solid lower half + disordered upper half: labels >= 95% correct."""
    rng = np.random.default_rng(31)
    a = 3.615
    reps = (4, 4, 5)
    from chaord.build.crystal import build_conventional
    solid = build_conventional("fcc", {"a": a}, ("Cu",), reps)
    L = solid.cell_diag
    truth = np.zeros(len(solid), bool)
    truth[solid.pos[:, 2] > L[2] / 2] = True
    # disorder the upper half beyond recognition (kept inside the upper half)
    pos = solid.pos.copy()
    n_liq = int(truth.sum())
    upper = rng.uniform(0, L, (n_liq, 3))
    upper[:, 2] = L[2] / 2 + rng.uniform(0, L[2] / 2 - 0.5, n_liq)
    pos[truth] = upper
    frame = Frame(pos=_wrap(pos, L), cell=solid.cell, symbols=solid.symbols)
    labels = phase_labels(frame, dialect)
    # atoms within one d_NN of the planted boundary belong to the interface
    # object, not to either phase: the criterion is evaluated outside that zone
    from chaord.build.defects import typical_neighbor_distance
    d_nn = typical_neighbor_distance(frame)
    z = frame.pos[:, 2]
    # exclude both interfaces: the planted one at mid-height and the periodic
    # wrap at z = 0 (the box joins solid bottom to liquid top)
    core = ((np.abs(z - L[2] / 2) > 2.0 * d_nn)
            & (z > 2.0 * d_nn) & (z < L[2] - 2.0 * d_nn))
    acc = label_accuracy(labels[core], (~truth)[core])
    assert acc >= 0.95, f"bulk label accuracy {acc:.3f}"


def test_interface_position_matches_tanh_fit(dialect):
    """M4 profile interface vs the M0 tanh fit agree within 0.5 A."""
    from pathlib import Path
    root = Path(__file__).parent.parent
    z = np.load(root / "prototype" / "snap.npz")
    from chaord.lift.slab import decompile
    lj = load_dialect(("core", "lj"))
    res = decompile(z["r"], z["L"], 0.65, lj)
    frame = Frame(pos=z["r"], cell=np.diag(z["L"]), symbols=["X"] * len(z["r"]))
    labels = phase_labels(frame, lj)
    interfaces = slab_interfaces(labels, frame, lj)
    assert len(interfaces) == 2
    for iface, z_ref in zip(interfaces, (res["z_up"], res["z_lo"])):
        d = abs(iface["at"] - z_ref)
        d = min(d, abs(d - float(z["L"][2])))  # periodicity
        assert d <= 0.5, f"interface at {iface['at']:.2f} vs tanh {z_ref:.2f}"


def test_surface_program_round_trip_stable(surf111, dialect):
    """lift -> build -> lift keeps the canonical surface text stable."""
    path, f = surf111
    text1 = format_program(lift_frame(f, dialect, mode="surface"))
    p2 = path.parent / "lifted.chaord"
    p2.write_text(text1)
    rng = np.random.default_rng(11)
    f2 = build_program(load(p2), dialect, rng=rng)
    text2 = format_program(lift_frame(f2, dialect, mode="surface"))
    import difflib
    diff = [l for l in difflib.unified_diff(text1.splitlines(), text2.splitlines(), lineterm="", n=0)
            if l.startswith(("+", "-")) and not l.startswith(("+++", "---"))]
    # counts and the lattice line must survive; site identity is free by symmetry
    for key in ("conserve atoms", "lattice fcc", "a 3.615", "surface (111)", "termination Ni",
                "adsorb O count 4 site top"):
        assert any(key in l for l in text2.splitlines()), f"lost: {key}"
    assert len(diff) <= 6  # only cosmetic/coverage digits may move
