"""M6 acceptance: extended defects and reactive census.

Exit criteria (PLAN M6): census exact on planted cases; Burgers vectors and
Sigma values correct on the benchmark constructions.
"""
import numpy as np
import pytest
from scipy.spatial import cKDTree

from chaord.build.extended import build_dislocation, build_grain_boundary
from chaord.build.molecules import molecule_census, pack_molecules
from chaord.dialects import load_dialect
from chaord.io.frames import Frame
from chaord.lang.fmt import fmt_statement
from chaord.lang.ir import (
    GeoChain, Name, Quantity, RegionBlock, ShAll, Statement,
)
from chaord.lift.extended import burgers_vector, grain_boundary_sigma
from chaord.lift.reactive import dissociation_statement


@pytest.fixture(scope="module")
def metal():
    return load_dialect(("core", "metal"))


def _region(*stmts):
    return RegionBlock(phase="crystal", name="bulk",
                       geometry=GeoChain(parts=[ShAll()], ops=[]),
                       statements=list(stmts))


def test_reactive_census_exact():
    d = load_dialect(("core", "molecular"))
    rng = np.random.default_rng(3)
    f = pack_molecules({"H2O": 50, "OH": 9, "H": 9}, [20.0] * 3, rng, d)
    census = molecule_census(f, d)
    assert census["H2O"] == 50
    assert census.get("OH", census.get("HO")) == 9
    assert census["H"] == 9
    # element conservation
    n_h = sum(1 for s in f.symbols if s == "H")
    n_o = sum(1 for s in f.symbols if s == "O")
    assert (n_h, n_o) == (50 * 2 + 9 + 9, 50 + 9)


def test_dissociation_statement():
    d = load_dialect(("core", "molecular"))
    rng = np.random.default_rng(3)
    f = pack_molecules({"H2O": 50, "OH": 9, "H": 9}, [20.0] * 3, rng, d)
    line = fmt_statement(dissociation_statement(f, d))
    assert line == "dissociate H2O -> OH @ surface + H @ surface count 9"


def test_dissociation_zero_without_fragments():
    d = load_dialect(("core", "molecular"))
    rng = np.random.default_rng(3)
    f = pack_molecules({"H2O": 20}, [18.0] * 3, rng, d)
    assert dissociation_statement(f, d) is None


def test_burgers_family(metal):
    frame = build_dislocation(_region(
        Statement(kind="build", key="lattice", values=[Name(text="fcc")]),
        Statement(kind="build", key="a", values=[Quantity(num="3.615")]),
        Statement(kind="build", key="dislocation", values=[Name(text="edge")])),
        {}, metal, np.random.default_rng(0))
    result = burgers_vector(frame, metal)
    assert result is not None
    assert result["family"] == "<110>"
    # unrelaxed construction: magnitude within the family detection band
    assert abs(result["magnitude"] - 3.615 / np.sqrt(2)) < 0.35 * 3.615 / np.sqrt(2)


def test_no_burgers_in_perfect_crystal(metal):
    from chaord.build.crystal import build_conventional
    frame = build_conventional("fcc", {"a": 3.615}, ("Cu",), (6, 6, 4))
    assert burgers_vector(frame, metal) is None


def _sigma5_region():
    return _region(
        Statement(kind="build", key="lattice", values=[Name(text="fcc")]),
        Statement(kind="build", key="a", values=[Quantity(num="3.615")]),
        Statement(kind="build", key="grain_boundary",
                  values=[Name(text="sigma"), Quantity(num="5")]))


def test_gb_min_pair_distance_above_overlap_floor(metal):
    """Root-cause reproduction of the Review-2 overlap bug (fails pre-fix).

    The old builder rotated the upper grain rigidly and wrapped it into a
    box the rotated grain cannot tile (8a is not a period of the
    36.87-deg-rotated fcc lattice), so wrapped slivers of the upper grain
    overlapped the lower one: 568/1536 atoms sat closer than 2.05 A and the
    closest pair was 1.14 A.  Sanity floor per AGENTS.md: no pair closer
    than 0.8 d_NN, d_NN = a/sqrt(2) for fcc."""
    frame = build_grain_boundary(_sigma5_region(), {}, metal,
                                 np.random.default_rng(0))
    L = frame.cell_diag
    pos = np.mod(frame.pos, L)
    tree = cKDTree(pos, boxsize=L)
    d, _ = tree.query(pos, k=2)
    dnn = 3.615 / np.sqrt(2)
    assert float(d[:, 1].min()) >= 0.8 * dnn - 1e-9


def _sigma_from_misorientation(theta, tol):
    """The detector's (m, n) table: Sigma = m^2 + n^2, halved when the fcc
    half-cell translations apply (both m, n odd)."""
    from math import gcd
    best = None
    for m in range(1, 16):
        for n in range(1, 16):
            if gcd(m, n) != 1 or (m, n) == (1, 1):
                continue
            target = np.degrees(2 * np.arctan(n / m)) % 90.0
            if target > 45.0:
                target = 90.0 - target
            if abs(theta - target) < tol:
                sigma = (m * m + n * n) // 2 if (m % 2 and n % 2) else m * m + n * n
                if best is None or abs(theta - target) < best[1]:
                    best = (sigma, abs(theta - target))
    return best[0] if best else None


def test_gb_tied_reference_atoms_all_sigma5(metal):
    """Every tied reference atom reads its grain as Sigma 5.

    The pre-histogram detector drew its reference atom from the tied top of
    the coordination-count ranking (argmax over equal counts -- the exact
    tie the old BLAS flake lived in).  For every atom of that tied class
    the misorientation between its OWN folded first-shell in-plane angles
    and the other half's dominant angle must map to Sigma 5.  On the broken
    builder the tied class is made of overlap-cluster atoms and every one
    of them reads a wrong orientation."""
    from chaord.lift.extended import _dominant_angle, _folded_shell_angles
    frame = build_grain_boundary(_sigma5_region(), {}, metal,
                                 np.random.default_rng(0))
    L = frame.cell_diag
    pos = np.mod(frame.pos, L)
    dnn = 3.615 / np.sqrt(2)
    cn_cut = float(metal.threshold("slab_cn_factor")) * dnn
    tree = cKDTree(pos, boxsize=L)
    counts = tree.query_ball_point(pos, cn_cut, return_length=True)
    tied = np.flatnonzero(counts == counts.max())
    assert len(tied) >= 20            # a bicrystal has an interior to speak of
    zmid = L[2] / 2
    lower, upper = pos[pos[:, 2] < zmid], pos[pos[:, 2] >= zmid]
    pa = _dominant_angle(_folded_shell_angles(lower, L, dnn, metal))
    pb = _dominant_angle(_folded_shell_angles(upper, L, dnn, metal))
    assert pa is not None and pb is not None
    tol = float(metal.threshold("csl_angle_tol"))
    for r in tied:
        js = [j for j in tree.query_ball_point(pos[r], cn_cut) if j != r]
        dv = pos[js] - pos[r]
        dv -= L * np.round(dv / L)
        in_plane = np.abs(dv[:, 2]) < 0.2 * np.linalg.norm(dv, axis=1)
        assert in_plane.sum() >= 4    # a reference atom sees its 4 in-plane NN
        ang = np.degrees(np.arctan2(dv[in_plane, 1], dv[in_plane, 0])) % 90.0
        ang = np.where(ang > 45.0, 90.0 - ang, ang)
        own = float(np.median(ang))
        other = pb if pos[r, 2] < zmid else pa
        assert _sigma_from_misorientation(abs(own - other), tol) == 5


def test_gb_deterministic_across_ten_seeds(metal):
    """Rule 10: same program + seed + backend -> same coordinates; the
    bicrystal is pure geometry, so all ten seeds must agree bit for bit."""
    ref = None
    for seed in range(10):
        frame = build_grain_boundary(_sigma5_region(), {}, metal,
                                     np.random.default_rng(seed))
        key = (frame.pos.tobytes(), tuple(frame.symbols),
               frame.cell.tobytes(), tuple(frame.pbc))
        if ref is None:
            ref = key
        assert key == ref


def test_sigma5_detected(metal):
    # CSL detection folds every first-shell in-plane bond angle of each grain
    # mod 90 deg and reads the misorientation off the two histogram peaks;
    # all atoms contribute, so no reference-atom sort tie-breaking (the old
    # BLAS-sensitive path, see reports/gate_a_prime.md A2) is involved.
    frame = build_grain_boundary(_sigma5_region(), {}, metal,
                                 np.random.default_rng(0))
    assert grain_boundary_sigma(frame, metal) == 5


def test_no_sigma_in_single_crystal(metal):
    from chaord.build.crystal import build_conventional
    frame = build_conventional("fcc", {"a": 3.615}, ("Cu",), (8, 8, 6))
    assert grain_boundary_sigma(frame, metal) is None
