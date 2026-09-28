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


# ---------------------------------------------------------------------------
# Wood reconstruction statements + compound (rutile) terminations
# ---------------------------------------------------------------------------

RECON_PROGRAM = """chaord 0.1
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
  termination Ni
{recon}
}}

vacuum gap : slab z {ztop:.1f} .. {lz:.1f} {{
}}
"""

RUTILE_110_PROGRAM = """chaord 0.1
dialect core + metal + surface

system {
  cell 6.497 2.959 23.727
  pbc xyz
  conserve atoms Ti 8 O 16
}

physics {
  backend eam
}

crystal slab : slab z 0 .. 11.7 {
  prototype rutile
  composition TiO2
  a 4.594 A
  c 2.959 A
  surface (110) top
  termination bridging_O
}

vacuum gap : slab z 11.7 .. 23.7 {
}
"""


def _missing_row_frame(a=3.615):
    """fcc(100) 4x4x4 slab with every other atom row of the top layer deleted.

    An independent construction of the p(2x1) missing-row reconstruction:
    the deletion works on sorted unique x columns, not on any chaord helper."""
    from ase.build import fcc100
    slab = fcc100("Ni", size=(4, 4, 4), a=a, vacuum=6.0)
    pos = slab.get_positions()
    z = pos[:, 2]
    top = np.where(z > z.max() - 0.5)[0]
    xs = np.sort(np.unique(np.round(pos[top][:, 0], 3)))
    step = xs[1] - xs[0]
    drop = {int(i) for i in top
            if int(round((pos[i, 0] - xs[0]) / step)) % 2 == 1}
    keep = [i for i in range(len(pos)) if i not in drop]
    assert len(top) - len(drop) == len(top) // 2  # half the top layer survives
    return Frame(pos=pos[keep], cell=np.array(slab.cell),
                 symbols=[slab.symbols[i] for i in keep])


def _top_of(frame, tol=0.5):
    z = frame.pos[:, 2]
    return np.where(z > z.max() - tol)[0]


def _net_lengths(frame, idx):
    from chaord.lift.surface import _net_vectors
    net = _net_vectors(frame.pos[idx][:, :2], frame.cell[:2, :2])
    assert net is not None, "top layer is not a lattice net"
    return sorted(np.linalg.norm(net, axis=1))


def test_wood_lift_missing_row(dialect):
    """A planted fcc(100) missing-row top layer lifts as reconstruction p(2x1)."""
    f = _missing_row_frame()
    text = format_program(lift_frame(f, dialect, mode="surface"))
    assert "surface (001) top" in text
    assert "reconstruction p(2x1)" in text
    assert "conserve atoms Ni 56" in text


def test_wood_no_false_positive(dialect):
    """The unreconstructed fcc(100) slab lifts without a reconstruction line."""
    from ase.build import fcc100
    slab = fcc100("Ni", size=(4, 4, 4), a=3.615, vacuum=6.0)
    f = Frame(pos=slab.get_positions(), cell=np.array(slab.cell),
              symbols=list(slab.symbols))
    text = format_program(lift_frame(f, dialect, mode="surface"))
    assert "reconstruction" not in text


def test_wood_round_trip_missing_row(dialect, tmp_path):
    """build(lift(planted 2x1 slab)) rebuilds an equivalent top layer."""
    f = _missing_row_frame()
    text = format_program(lift_frame(f, dialect, mode="surface"))
    path = tmp_path / "recon.chaord"
    path.write_text(text)
    f2 = build_program(load(path), dialect, rng=np.random.default_rng(5))
    assert len(f2) == 56
    top1, top2 = _top_of(f), _top_of(f2)
    assert len(top1) == len(top2) == 8
    # equivalent nets: same primitive vector lengths (up to the 90 deg basis choice)
    assert np.allclose(_net_lengths(f, top1), _net_lengths(f2, top2), atol=0.05)
    # and the rebuilt slab re-lifts to the same statement
    text2 = format_program(lift_frame(f2, dialect, mode="surface"))
    assert "reconstruction p(2x1)" in text2


@pytest.mark.parametrize("wood,n_atoms,hkl,lx,ly", [
    ("p(2x1)", 56, "001", 10.225, 10.225),
    ("p(2x2)", 52, "001", 10.225, 10.225),
    ("c(2x2)", 56, "001", 10.225, 10.225),
    ("(r3xr3)R30", 30, "111", 7.669, 6.641),
])
def test_reconstruction_build_lift_round_trip(wood, n_atoms, hkl, lx, ly,
                                              dialect, tmp_path):
    """Every buildable Wood statement survives build -> lift unchanged."""
    text = RECON_PROGRAM.format(lx=lx, ly=ly, lz=17.422, n=n_atoms, hkl=hkl,
                                ztop=6.0, recon=f"  reconstruction {wood}")
    path = tmp_path / f"recon_{wood.replace('/', '-')}.chaord"
    path.write_text(text)
    f = build_program(load(path), dialect, rng=np.random.default_rng(9))
    assert len(f) == n_atoms
    lifted = format_program(lift_frame(f, dialect, mode="surface"))
    assert f"reconstruction {wood}" in lifted


def test_rutile_110_bridging_oxygen(dialect, tmp_path):
    """rutile (110): generic cut picks an O-terminated top layer; lift names it."""
    path = tmp_path / "rutile110.chaord"
    path.write_text(RUTILE_110_PROGRAM)
    f = build_program(load(path), dialect, rng=np.random.default_rng(2))
    assert len(f) == 24  # four Ti2O4 stacking units
    top = _top_of(f)
    assert "O" in {f.symbols[i] for i in top}  # top layer contains O
    text = format_program(lift_frame(f, dialect, mode="surface"))
    assert "prototype rutile" in text
    assert "composition TiO2" in text
    assert "a 4.594 A" in text
    assert "c 2.959 A" in text
    assert "surface (110) top" in text
    assert "termination bridging_O" in text
    assert "conserve atoms O 16 Ti 8" in text  # lift sorts species names


def test_rutile_110_round_trip_stable(dialect, tmp_path):
    """lift -> build reproduces the rutile slab and lifts to the same text."""
    path = tmp_path / "rutile110.chaord"
    path.write_text(RUTILE_110_PROGRAM)
    f1 = build_program(load(path), dialect, rng=np.random.default_rng(3))
    text1 = format_program(lift_frame(f1, dialect, mode="surface"))
    p2 = tmp_path / "rutile110_lifted.chaord"
    p2.write_text(text1)
    f2 = build_program(load(p2), dialect, rng=np.random.default_rng(4))
    assert len(f1) == len(f2) == 24
    text2 = format_program(lift_frame(f2, dialect, mode="surface"))
    assert "surface (110) top" in text2
    assert "termination bridging_O" in text2
    assert "reconstruction" not in text2


def test_interface_width_within_noise_floor():
    """M4 exit criterion: interface widths reproduce within the noise floor.

    The floor between two reference frames is zero here (widths are quantised
    to the profile bin), so the effective floor is the bin width: sub-bin
    width differences are not resolvable. A rebuild with physics must land
    within 1.5x that floor of the original width."""
    import numpy as np
    from chaord.io.frames import Frame as F
    from chaord.lift.segment import phase_labels, slab_interfaces
    from pathlib import Path as P
    root = P(__file__).parent.parent
    lj = load_dialect(("core", "lj"))

    def interfaces(fr):
        Lz = fr.cell_diag[2]
        labels = phase_labels(fr, lj)
        return [i for i in slab_interfaces(labels, fr, lj) if i["width"] < 0.6 * Lz]

    fa = read_frame(root / "prototype" / "snap.npz")
    fb = read_frame(root / "prototype" / "snap_later.npz")
    ia = interfaces(fa)
    ib = interfaces(fb)
    assert len(ia) >= 1 and len(ib) >= 1
    floor = abs(ia[0]["width"] - ib[0]["width"])
    binw = float(lj.threshold("profile_bin_size"))
    eff_floor = max(floor, binw)

    from chaord.lang.api import load, save
    from chaord.build import build_program
    from chaord.lift import lift_frame
    import tempfile
    prog = lift_frame(fa, lj, mode="slab")
    with tempfile.TemporaryDirectory() as td:
        pp = P(td) / "p.chaord"
        save(prog, pp)
        rebuilt = build_program(load(pp), lj, rng=np.random.default_rng(3),
                                physics=True)
    ir = interfaces(rebuilt)
    # crossings cluster near the (slightly shifted) rebuilt interface position:
    # take the median width of the cluster nearest the reference crossing
    near = [i for i in ir if abs(i["at"] - ia[0]["at"]) < 3.0] or ir
    w_rebuilt = float(np.median([i["width"] for i in near]))
    w_ref = ia[0]["width"]
    assert abs(w_rebuilt - w_ref) <= 1.5 * eff_floor, (
        f"width {w_rebuilt:.2f} vs {w_ref:.2f}, floor {eff_floor:.2f}")
