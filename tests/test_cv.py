"""CV measures against brute-force reference code (agreement to 1e-6)."""
import numpy as np
import pytest

from chaord.cv import CVS, measure
from chaord.cv.local import cn_mean, number_density
from chaord.dialects import load_dialect
from chaord.io.frames import Frame
from chaord.lang.errors import ChaordError


@pytest.fixture
def frame():
    rng = np.random.default_rng(5)
    L = np.array([6.0, 6.0, 6.0])
    return Frame(pos=rng.uniform(0, 6, (40, 3)), cell=np.diag(L),
                 symbols=["X"] * 40)


@pytest.fixture
def dialect():
    return load_dialect(("core", "lj"))


def _brute_cn(pos, L, rc):
    n = 0
    for i in range(len(pos)):
        for j in range(len(pos)):
            if i == j:
                continue
            d = pos[j] - pos[i]
            d -= L * np.round(d / L)
            if np.linalg.norm(d) < rc:
                n += 1
    return n / len(pos)


def test_cn_matches_brute_force(frame, dialect):
    rc = dialect.threshold("cn_cutoff")
    assert cn_mean(frame, dialect) == pytest.approx(_brute_cn(frame.pos, frame.cell_diag, rc), abs=1e-6)


def test_density(frame, dialect):
    assert number_density(frame, dialect) == pytest.approx(40 / 216, abs=1e-12)


def test_registry_measure_dispatch(frame, dialect):
    assert measure("cn", frame, dialect) == pytest.approx(
        cn_mean(frame, dialect), abs=1e-12)


def test_registry_unknown_cv():
    with pytest.raises(ChaordError, match="unknown CV"):
        measure("nope", None, None)


def test_registry_unmeasurable_cv():
    with pytest.raises(ChaordError, match="no measure yet"):
        measure("gr_peak", None, None)


def test_one_definition_per_quantity():
    from chaord.cv.registry import register
    with pytest.raises(ChaordError, match="already registered"):
        register("cn", None, "duplicate")


def test_key_cvs_registered():
    for name in ("cn", "density", "angle_mean", "q6", "gr_peak", "sites_matched",
                 "solid_clusters", "sro_alpha1", "coverage", "pairing",
                 "compressibility", "solid_like"):
        assert name in CVS


def test_angle_mean_tetrahedral(dialect):
    # four atoms around a centre at perfect tetrahedral angles
    import itertools
    centre = np.zeros(3)
    v = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
    v = v / np.sqrt(3)
    pos = np.vstack([centre, v])
    frame = Frame(pos=pos, cell=np.diag([10.0] * 3), symbols=["X"] * 5)
    # tetrahedral angle = 109.47 deg; the centre sees 6 pairs at that angle
    assert measure("angle_mean", frame, dialect, cutoff=1.2) == pytest.approx(
        np.degrees(np.arccos(-1 / 3)), abs=1e-6)


def test_gr_drops_intramolecular_pairs_for_molecular_dialects():
    """Coordinator approval 2026-09-29: the legacy fluid g(r) drops
    same-molecule pairs under the molecular dialect's
    partial_gr_exclude_intramolecular (the same switch the partial g(r)
    uses): the intramolecular O-H/H-H peaks are exact geometry constants of
    the `model` statement, not thermal statistics, and SPC/E's r_OH = 1.0 A
    sits on a histogram bin edge where the reference frames split by float
    rounding. Asserted: a packed-water frame's g(r) carries no density in
    the intramolecular bins, and its integral scale is unchanged elsewhere."""
    from chaord.build.molecules import pack_molecules
    from chaord.cv.noise import observables
    from chaord.dialects import load_dialect

    mol = load_dialect(("core", "molecular"))
    f = pack_molecules({"H2O": 40}, [13.0] * 3, np.random.default_rng(5), mol)
    o = observables(f, mol)
    rm, g = o["rm"], o["gr"]
    # intramolecular peaks (O-H 0.9572, H-H 1.515) must be gone: the O-H
    # bin empty, the H-H bin carrying only genuine intermolecular contacts
    # (the un-excluded intramolecular delta there is g ~ 1.3)
    assert g[(rm > 0.9) & (rm < 1.1)].max() == 0.0
    assert g[(rm > 1.45) & (rm < 1.6)].max() < 0.5
    # ... while the O-O first shell is still there
    oo = g[(rm > 2.5) & (rm < 3.2)]
    assert oo.max() > 1.0


def test_gr_unchanged_for_atomic_dialects():
    """The exclusion is dialect-gated: under core/lj (no switch) the legacy
    g(r) keeps the raw atom pair list -- every existing lj floor and
    distance stays byte-identical."""
    from chaord.cv.noise import observables
    from chaord.dialects import load_dialect

    lj = load_dialect(("core", "lj"))
    rng = np.random.default_rng(7)
    f = Frame(pos=rng.uniform(0, 6, (60, 3)), cell=np.diag([6.0] * 3),
              symbols=["X"] * 60, pbc=(True, True, True))
    o = observables(f, lj)
    from scipy.spatial import cKDTree
    L = f.cell_diag
    pos = np.mod(f.pos, L)
    pairs = cKDTree(pos, boxsize=L).query_pairs(
        float(lj.threshold("gr_rmax_fluid")), output_type="ndarray")
    d = pos[pairs[:, 1]] - pos[pairs[:, 0]]
    d -= L * np.round(d / L)
    h, e = np.histogram(np.linalg.norm(d, axis=1),
                        np.linspace(0, float(lj.threshold("gr_rmax_fluid")),
                                    int(lj.threshold("gr_bins_fluid")) + 1))
    shell = 4 / 3 * np.pi * (e[1:] ** 3 - e[:-1] ** 3)
    g_ref = 2 * h / (len(pos) * (len(pos) / float(np.prod(L))) * shell)
    assert np.allclose(o["gr"], g_ref)
