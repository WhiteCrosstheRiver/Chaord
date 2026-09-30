"""M6 acceptance: extended defects and reactive census.

Exit criteria (PLAN M6): census exact on planted cases; Burgers vectors and
Sigma values correct on the benchmark constructions.

Review 3 (2026-09-30) additions: the Sigma detector must survive 0.10 A of
Gaussian position noise (20 seeds x Sigma 5/13/17), accept unwrapped
coordinates (positions outside the box) at its entry, apply the Brandon
criterion (15 deg / sqrt(Sigma); Brandon, Acta Metall. 14, 1479 (1966))
instead of emitting a spurious large Sigma, and stay silent (None) on frames
with no clear grain orientation.
"""
import numpy as np
import pytest
from scipy.spatial import cKDTree

from chaord.build.extended import build_dislocation, build_grain_boundary, CSL_MN
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


def _sigma_region(sigma: int):
    return _region(
        Statement(kind="build", key="lattice", values=[Name(text="fcc")]),
        Statement(kind="build", key="a", values=[Quantity(num="3.615")]),
        Statement(kind="build", key="grain_boundary",
                  values=[Name(text="sigma"), Quantity(num=str(sigma))]))


@pytest.fixture(scope="module")
def gb_frames(metal):
    """The three shipped [001] CSL bicrystals, built once (deterministic)."""
    return {s: build_grain_boundary(_sigma_region(s), {}, metal,
                                    np.random.default_rng(0))
            for s in (5, 13, 17)}


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
    return _sigma_region(5)


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
    from chaord.lift.extended import _dominant_angle, _shell_angles
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
    pa = _dominant_angle(_shell_angles(lower, L, dnn, metal), metal)
    pb = _dominant_angle(_shell_angles(upper, L, dnn, metal), metal)
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


# --- Review 3: robustness of the CSL detector (2026-09-30) ---

NOISE_AMPLITUDE = 0.10        # A, per-axis Gaussian position noise per atom
NOISE_SEEDS = 20              # independent seeds per Sigma


def _noisy(frame, seed, amp=NOISE_AMPLITUDE):
    """frame + seeded Gaussian noise per atom, wrapped back into the box."""
    rng = np.random.default_rng(seed)
    L = frame.cell_diag
    return Frame(pos=np.mod(frame.pos + rng.normal(scale=amp, size=frame.pos.shape), L),
                 cell=frame.cell, symbols=frame.symbols, pbc=frame.pbc)


def test_gb_sigma_survives_0p10A_noise_20_seeds(metal, gb_frames):
    """Review 3: 20/20 correct Sigma for each built bicrystal at 0.10 A noise.

    Root cause of the pre-fix failure: the folded first-shell angle peak of
    the grain on the cube axes sits exactly at the 45-deg fold edge, where
    the fold maps 45 +/- delta onto 45 - |delta| -- a half-normal whose mean
    is biased low by 0.8 sigma.  At 0.10 A (sigma_angle ~ 3.2 deg) that is a
    ~2.5-deg misreading of the misorientation: Sigma 17 came out as 233 in
    3 of 5 review samples.  The detector must average the peak without the
    fold bias and stay clear of the few-percent cross-grain satellite pairs
    that noise drags through the in-plane filter."""
    for sigma, frame in gb_frames.items():
        for seed in range(NOISE_SEEDS):
            got = grain_boundary_sigma(_noisy(frame, 20260930 + seed), metal)
            assert got == sigma, (
                f"Sigma {sigma} with 0.10 A noise, seed {seed}: read {got}")


def test_gb_unwrapped_positions(metal, gb_frames):
    """Review 3: unwrapped coordinates must not crash the entry points.

    Positions translated by random integer multiples of the box (a wrapped
    equivalent frame) crashed pre-fix: typical_neighbor_distance builds a
    cKDTree with boxsize=L on the raw positions and scipy raises
    'Some input data are greater than the size of the periodic box' before
    any wrapping happens.  Both entry points must mod into the box first."""
    frame = gb_frames[5]
    L = frame.cell_diag
    rng = np.random.default_rng(7)
    shifts = rng.integers(-2, 3, size=frame.pos.shape)
    unwrapped = Frame(pos=frame.pos + shifts * L, cell=frame.cell,
                      symbols=frame.symbols, pbc=frame.pbc)
    assert np.any(np.abs(unwrapped.pos) > L)     # really outside the box
    assert grain_boundary_sigma(unwrapped, metal) == 5


def test_burgers_unwrapped_positions(metal):
    """Same entry contract for the Burgers pass (review 3)."""
    frame = build_dislocation(_region(
        Statement(kind="build", key="lattice", values=[Name(text="fcc")]),
        Statement(kind="build", key="a", values=[Quantity(num="3.615")]),
        Statement(kind="build", key="dislocation", values=[Name(text="edge")])),
        {}, metal, np.random.default_rng(0))
    L = frame.cell_diag
    rng = np.random.default_rng(5)
    shifts = rng.integers(-2, 3, size=frame.pos.shape)
    unwrapped = Frame(pos=frame.pos + shifts * L, cell=frame.cell,
                      symbols=frame.symbols, pbc=frame.pbc)
    result = burgers_vector(unwrapped, metal)
    assert result is not None
    assert result["family"] == "<110>"


def _twist_upper_half(frame, delta_deg):
    """Rigidly rotate the upper z half by delta_deg about z (box centred)."""
    L = frame.cell_diag
    zmid = 0.5 * L[2]
    t = np.radians(delta_deg)
    c, s = np.cos(t), np.sin(t)
    pos = frame.pos.copy()
    up = pos[:, 2] >= zmid
    x, y = pos[up, 0] - 0.5 * L[0], pos[up, 1] - 0.5 * L[1]
    pos[up, 0] = c * x - s * y + 0.5 * L[0]
    pos[up, 1] = s * x + c * y + 0.5 * L[1]
    return Frame(pos=np.mod(pos, L), cell=frame.cell,
                 symbols=frame.symbols, pbc=frame.pbc)


def test_gb_low_angle_twist_returns_none(metal):
    """Review 3: Brandon criterion -- no spurious large Sigma near 0 deg.

    A 2.5-deg [001] twist of a single crystal is a low-angle boundary (a
    dislocation network, not a coincidence lattice).  Pre-fix, the flat
    2-deg match window read it as Sigma 421 (target 3.95 deg).  Brandon's
    criterion (15 deg / sqrt(Sigma); Brandon, Acta Metall. 14, 1479 (1966))
    caps the acceptance window of Sigma 421 at 0.73 deg, so no candidate
    survives and the detector must emit nothing."""
    from chaord.build.crystal import build_conventional
    frame = build_conventional("fcc", {"a": 3.615}, ("Cu",), (8, 8, 6))
    assert grain_boundary_sigma(_twist_upper_half(frame, 2.5), metal) is None


def _rsa_frame(rng, n_target=800, box=36.0, dmin=2.0):
    """Seeded random-close-refusal frame: no orientation anywhere."""
    pts = []
    while len(pts) < n_target:
        p = rng.uniform(0.0, box, size=3)
        if not pts:
            pts.append(p)
            continue
        d = np.abs(np.asarray(pts) - p)
        d = np.minimum(d, box - d)
        if float(np.sqrt((d ** 2).sum(axis=1)).min()) > dmin:
            pts.append(p)
    pts = np.asarray(pts)
    return Frame(pos=pts, cell=np.diag([box, box, box]),
                 symbols=["Cu"] * len(pts), pbc=(True, True, True))


def test_gb_disordered_frame_returns_none(metal):
    """Review 3: a frame with no clear orientation must not emit a Sigma.

    Pre-fix, a random RSA frame read as Sigma 193: the two halves' folded
    angle histograms have no peak (their mass spreads over the whole
    [0, 45] fold), yet the peak-bin means differ by some angle that happened
    to sit inside the flat 2-deg match window of some (m, n) pair."""
    frame = _rsa_frame(np.random.default_rng(11))
    assert grain_boundary_sigma(frame, metal) is None


def test_gb_sigma13_sigma17_zero_overlap(metal, gb_frames):
    """Review 3 (verify): the commensurate-box fix holds for Sigma 13/17.

    Overlap sanity floor per AGENTS.md: no pair closer than 0.8 x d_NN
    (d_NN = a/sqrt(2) = 2.556 A for Cu), counted with periodic images."""
    dnn = 3.615 / np.sqrt(2)
    for sigma in (13, 17):
        frame = gb_frames[sigma]
        L = frame.cell_diag
        pos = np.mod(frame.pos, L)
        tree = cKDTree(pos, boxsize=L)
        n_overlap = len(tree.query_pairs(0.8 * dnn, output_type="ndarray"))
        assert n_overlap == 0, f"Sigma {sigma}: {n_overlap} pairs below 0.8 d_NN"


def test_gb_csl_table_integer_matrices(metal):
    """Review 3 (verify): every table entry is an integer [001] rotation.

    For theta = 2 atan(n/m) the rotation acts on the (a/2) integer lattice
    as M = [[m^2-n^2, 2mn], [-2mn, m^2-n^2]] / (m^2+n^2); det(M_int) =
    (m^2+n^2)^2 must be a perfect square whose root is the Sigma up to the
    fcc half-cell halving (both m, n odd), and the table's Sigma must equal
    exactly that value."""
    for sigma, (m, n) in CSL_MN.items():
        M = np.array([[m * m - n * n, 2 * m * n],
                      [-2 * m * n, m * m - n * n]])
        D = m * m + n * n
        assert round(abs(np.linalg.det(M))) == D * D     # det = D^2 exactly
        assert float(np.sqrt(round(abs(np.linalg.det(M))))) == D
        expected = D // 2 if (m % 2 and n % 2) else D
        assert expected == sigma, (sigma, m, n, expected)
        # the rotation itself maps integer lattice vectors to integer ones
        # in the D-fold frame (the common cell both grains tile)
        assert np.allclose(M / D @ M.T / D, np.eye(2), atol=1e-12)  # orthogonal
