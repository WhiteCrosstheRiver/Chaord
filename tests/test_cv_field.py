"""Field observables: dipole-orientation order parameter P1 and the
cylindrical radial density profile (chaord.cv.field).

The reviewer's binding (docs/reviews/open_items_v2.md, section 5, item 2:
"After the gate: the first research use") asks for "the measurable
consequences" of the external-field statement: "a dipole-orientation order
parameter and the radial density profile".  These are the observables the
field-on / field-off separation of item 3 will be judged on, so each one is
checked here against a test-internal independent implementation (the house
rule for every observable), on

  * water molecules from the template registry placed at KNOWN angles to
    the field axis -- P1 exact to float precision;
  * a random small system (seeded rotations) -- P1 equals the brute-force
    molecular-dipole average;
  * EC molecules from the template registry with reader-carried frame
    charges (frame.info["charges"] wins over the dialect's site-charge
    table);
  * a known arrangement -- radial bin counts exact, plus the periodic
    boundary wrap case about a caller centre;
  * the summary inner/outer shell density ratio at its exact closed-form
    value;
  * fail-closed paths: no named species, no charge model, a non-neutral
    charge table, a zero dipole, malformed frame charges, a missing
    research dialect, a bad axis.

Charge models are INPUTS shared by library and reference (a physical model
parameter read from the dialect by name, like every other threshold), not
part of the measured quantity: the independent implementations recompute
the dipole moment, the orientation average and the radial histogram with
their own arithmetic.

Geometry: TEMPLATES["H2O"] (TIP4P geometry, exact published data in
build/molecules.py) with the research dialect's SPC/E site charges
(Berendsen, Grigera & Straatsma, J. Phys. Chem. 91, 6269 (1987)) -- the
charge set is the model input under test, the template geometry only fixes
the atoms' arrangement.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from chaord.build.molecules import TEMPLATES               # noqa: E402
from chaord.cv import CVS, measure                          # noqa: E402
from chaord.io.frames import Frame                          # noqa: E402
from chaord.lang.errors import ChaordError                  # noqa: E402


# --------------------------------------------------------------- fixtures --

@pytest.fixture(scope="module")
def research():
    from chaord.dialects import load_dialect
    return load_dialect(("core", "research"))


@pytest.fixture(scope="module")
def mol_research():
    """The stack the research case will actually run: molecular templates
    and rules with the research measurement keys on top."""
    from chaord.dialects import load_dialect
    return load_dialect(("core", "molecular", "research"))


def _water_charges(research):
    """Per-element site charges the dialect ships for H2O (model input)."""
    table = research.threshold("field_dipole_site_charges")["H2O"]
    return {el: float(q) for el, q in table.items()}


def _template_dipole(research, name="H2O"):
    """Dipole moment of a template under the dialect's charge table,
    computed in the template's own frame (reference-side model input)."""
    t = TEMPLATES[name]
    rel = np.asarray(t["rel"], float)
    q = _water_charges(research) if name == "H2O" else None
    assert q is not None
    mu = sum(q[s] * rel[i] for i, s in enumerate(t["symbols"]))
    return np.asarray(mu, float)


def _rotation_from_to(u, v):
    """Rotation matrix mapping unit vector u onto unit vector v (Rodrigues)."""
    u = np.asarray(u, float)
    u = u / np.linalg.norm(u)
    v = np.asarray(v, float)
    v = v / np.linalg.norm(v)
    c = float(np.dot(u, v))
    if c > 1.0 - 1e-12:
        return np.eye(3)
    if c < -1.0 + 1e-12:                      # antiparallel: pi about a perpendicular
        perp = np.cross(u, [1.0, 0.0, 0.0])
        if np.linalg.norm(perp) < 1e-12:
            perp = np.cross(u, [0.0, 1.0, 0.0])
        perp = perp / np.linalg.norm(perp)
        K = np.array([[0.0, -perp[2], perp[1]],
                      [perp[2], 0.0, -perp[0]],
                      [-perp[1], perp[0], 0.0]])
        return np.eye(3) + 2.0 * (K @ K)
    axis = np.cross(u, v)
    axis = axis / np.linalg.norm(axis)
    s = float(np.sqrt(1.0 - c * c))
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


def _water_frame(rotations, centres, L):
    """A frame of template water molecules, one per (rotation, centre)."""
    t = TEMPLATES["H2O"]
    rel = np.asarray(t["rel"], float)
    pos = np.vstack([np.asarray(c, float) + rel @ R.T
                     for R, c in zip(rotations, centres)])
    symbols = list(t["symbols"]) * len(rotations)
    return Frame(pos=pos, cell=np.diag(np.asarray(L, float)),
                 symbols=symbols, pbc=(True, True, True))


def _brute_force_p1(pos, symbols, charges_per_atom, axis_vec, per_molecule=3):
    """Independent P1: molecular dipoles from plain (non-MIC) arithmetic --
    the test places every molecule intact inside the box, so the reference
    needs no unwrapping, no bond graph and no histogram machinery."""
    axis_vec = np.asarray(axis_vec, float)
    axis_vec = axis_vec / np.linalg.norm(axis_vec)
    n = len(pos) // per_molecule
    cosines = []
    for m in range(n):
        idx = slice(m * per_molecule, (m + 1) * per_molecule)
        p = np.asarray(pos, float)[idx]
        q = np.asarray(charges_per_atom, float)[idx]
        mu = ((p - p[0]) * q[:, None]).sum(axis=0)
        cosines.append(float(mu @ axis_vec) / float(np.linalg.norm(mu)))
    return float(np.mean(cosines)), n


def _brute_force_radial(pos, L, axis_idx, centre, r_max, bins):
    """Independent cylindrical profile: one atom at a time, explicit
    minimum-image wrap, explicit bin search (no np.histogram)."""
    perp = [i for i in range(3) if i != axis_idx]
    edges = [r_max * b / bins for b in range(bins + 1)]
    counts = [0] * bins
    outside = 0
    for p in np.asarray(pos, float):
        d = [p[perp[0]] - centre[0], p[perp[1]] - centre[1]]
        d[0] -= L[perp[0]] * round(d[0] / L[perp[0]])
        d[1] -= L[perp[1]] * round(d[1] / L[perp[1]])
        r = np.sqrt(d[0] * d[0] + d[1] * d[1])
        if r > r_max:
            outside += 1
            continue
        b = bins - 1
        for k in range(bins):
            if edges[k] <= r < edges[k + 1]:
                b = k
                break
        else:
            b = bins - 1                      # r == r_max closes the last bin
        counts[b] += 1
    shell = [np.pi * (edges[k + 1] ** 2 - edges[k] ** 2) * L[axis_idx]
             for k in range(bins)]
    return np.asarray(counts, float) / np.asarray(shell, float), outside


# --------------------------------------------- dipole-orientation P1 exact --

def test_p1_known_angles_exact(research):
    """Four water molecules at 0, 60, 90 and 180 deg to +z: P1 must be the
    mean of cos(0), cos(60), cos(90), cos(180) = 0.125 exactly."""
    from chaord.cv.field import dipole_orientation
    L = np.array([24.0, 24.0, 24.0])
    mu0 = _template_dipole(research)
    thetas = [0.0, 60.0, 90.0, 180.0]
    ws = [np.array([np.sin(np.deg2rad(t)), 0.0, np.cos(np.deg2rad(t))])
          for t in thetas]
    rots = [_rotation_from_to(mu0, w) for w in ws]
    centres = [(6.0, 6.0, 12.0), (18.0, 6.0, 12.0), (6.0, 18.0, 12.0),
               (18.0, 18.0, 12.0)]
    f = _water_frame(rots, centres, L)
    out = dipole_orientation(f, research, axis="z")
    assert out["n_molecules"] == 4
    assert out["P1"] == pytest.approx(
        float(np.mean([np.cos(np.deg2rad(t)) for t in thetas])), abs=1e-12)
    # the brute-force reference agrees on the very same frame
    q = _water_charges(research)
    charges = [q[s] for s in f.symbols]
    brute, n = _brute_force_p1(f.pos, f.symbols, charges, [0.0, 0.0, 1.0])
    assert n == 4
    assert out["P1"] == pytest.approx(brute, abs=1e-12)


def test_p1_random_rotations_match_brute_force(research):
    """24 seeded random water molecules: library P1 == brute-force average
    of the per-molecular cos(theta), and n_molecules counts the molecules."""
    from chaord.cv.field import dipole_orientation
    from chaord.build.molecules import random_rotation
    rng = np.random.default_rng(20261014)
    L = np.array([26.0, 26.0, 26.0])
    rots = [random_rotation(rng) for _ in range(24)]
    # 4 x 3 x 2 distinct sites, 6 A apart, clear of the box edges
    centres = [(5.0 + 6.0 * (i % 4), 5.0 + 6.0 * ((i // 4) % 3),
                8.0 + 10.0 * ((i // 12) % 2)) for i in range(24)]
    assert len(set(centres)) == 24
    f = _water_frame(rots, centres, L)
    q = _water_charges(research)
    brute, n = _brute_force_p1(f.pos, f.symbols,
                               [q[s] for s in f.symbols], [0.0, 0.0, 1.0])
    assert n == 24
    out = dipole_orientation(f, research, axis="z")
    assert out["n_molecules"] == 24
    assert out["P1"] == pytest.approx(brute, abs=1e-12)
    # every cos(theta) is in [-1, 1] by construction of the average
    assert -1.0 <= out["P1"] <= 1.0


def test_p1_axis_x_matches_brute_force(research):
    """The order parameter follows the named axis: the same frame measured
    against +x equals the brute-force x-axis average."""
    from chaord.cv.field import dipole_orientation
    from chaord.build.molecules import random_rotation
    rng = np.random.default_rng(7)
    L = np.array([26.0, 26.0, 26.0])
    rots = [random_rotation(rng) for _ in range(12)]
    centres = [(5.0 + 6.0 * (i % 3), 5.0 + 6.0 * ((i // 3) % 2),
                8.0 + 10.0 * ((i // 6) % 2)) for i in range(12)]
    assert len(set(centres)) == 12
    f = _water_frame(rots, centres, L)
    q = _water_charges(research)
    brute, _ = _brute_force_p1(f.pos, f.symbols, [q[s] for s in f.symbols],
                               [1.0, 0.0, 0.0])
    out = dipole_orientation(f, research, axis="x")
    assert out["P1"] == pytest.approx(brute, abs=1e-12)


def test_p1_translation_invariant_and_deterministic(research):
    """Shifting the whole frame changes nothing (P1 is internal to each
    molecule); the same frame measured twice is byte-identical."""
    from chaord.cv.field import dipole_orientation
    from chaord.build.molecules import random_rotation
    rng = np.random.default_rng(11)
    L = np.array([26.0, 26.0, 26.0])
    rots = [random_rotation(rng) for _ in range(8)]
    centres = [(5.0 + 6.0 * (i % 4), 7.0 + 9.0 * (i // 4), 12.0)
               for i in range(8)]
    assert len(set(centres)) == 8
    f = _water_frame(rots, centres, L)
    shifted = Frame(pos=f.pos + 7.3, cell=f.cell, symbols=f.symbols,
                    pbc=f.pbc, info=dict(f.info))
    a = dipole_orientation(f, research)
    b = dipole_orientation(f, research)
    c = dipole_orientation(shifted, research)
    assert a["P1"] == b["P1"] and a["n_molecules"] == b["n_molecules"]
    assert a["P1"] == pytest.approx(c["P1"], abs=1e-9)


# ---------------------------------------------- frame charges (reader path) --

def test_p1_frame_charges_win_over_dialect_table(research):
    """EC molecules (template registry) with reader-carried per-atom charges
    in frame.info['charges']: those charges are used (the dialect table has
    no EC entry and is never consulted), and flipping the charge pattern
    flips the measured order -- the frame's model, not the table's.

    Charges are SYNTHETIC (a sum-zero test set, not a published model);
    they exist to pin the precedence rule."""
    from chaord.cv.field import dipole_orientation
    t = TEMPLATES["EC"]
    rel = np.asarray(t["rel"], float)
    # sum-zero synthetic set with a nonzero dipole
    q_set = np.array([0.5, -0.3, 0.1, 0.1, -0.3, -0.4, 0.0, 0.0, 0.3, 0.0])
    assert abs(q_set.sum()) < 1e-12
    rng = np.random.default_rng(3)
    L = np.array([30.0, 30.0, 30.0])
    from chaord.build.molecules import random_rotation
    rots = [random_rotation(rng) for _ in range(9)]
    # 9 A spacing: EC's ~2.3 A bonding extent cannot reach between sites
    centres = [(6.0 + 9.0 * (i % 3), 6.0 + 9.0 * ((i // 3) % 3), 15.0)
               for i in range(9)]
    pos = np.vstack([np.asarray(c, float) + rel @ R.T
                     for R, c in zip(rots, centres)])
    symbols = list(t["symbols"]) * len(rots)
    charges = np.tile(q_set, len(rots))
    f = Frame(pos=pos, cell=np.diag(L), symbols=symbols,
              pbc=(True, True, True), info={"charges": charges})
    brute, n = _brute_force_p1(f.pos, f.symbols, charges,
                               [0.0, 0.0, 1.0], per_molecule=10)
    assert n == 9

    # EC is not the dialect's named species -- re-point it, still WITHOUT a
    # charge-table entry: only the frame charges can resolve the model
    research_ec = load_research_copy()
    research_ec._data["thresholds"]["field_dipole_species"] = ["C3H4O3"]
    out = dipole_orientation(f, research_ec, axis="z")
    assert out["n_molecules"] == 9
    assert out["P1"] == pytest.approx(brute, abs=1e-12)
    # the flipped charge set flips the order parameter
    flipped = np.tile(-q_set, len(rots))
    f2 = Frame(pos=pos, cell=np.diag(L), symbols=symbols,
               pbc=(True, True, True), info={"charges": flipped})
    out2 = dipole_orientation(f2, research_ec, axis="z")
    assert out2["P1"] == pytest.approx(-out["P1"], abs=1e-12)


def load_research_copy():
    """A private mutable copy of the research dialect (fail-closed tests
    re-point single thresholds without touching the shared object)."""
    from chaord.dialects import load_dialect
    return load_dialect(("core", "research"))


# ------------------------------------------------------- radial profile --

def test_radial_profile_exact_bin_counts(research):
    """Known arrangement, exact bin counts: atoms placed by hand at radii
    inside, on and beyond r_max; the library profile equals the independent
    per-atom histogram, and n_outside counts the corner-region atoms."""
    from chaord.cv.field import radial_density_profile
    L = np.array([20.0, 20.0, 12.0])
    bins = 8                                   # r_max = 10, bin width 1.25
    r_max = float(np.min(L[:2])) / 2
    centre = (10.0, 10.0)
    pts = [
        (10.0, 10.0, 1.0), (10.0, 10.0, 5.0), (10.0, 10.0, 11.0),   # r = 0
        (10.6, 10.0, 2.0), (10.0, 10.4, 3.0),                       # bin 0
        (11.3, 10.0, 4.0), (10.0, 8.9, 6.0),                        # r 1.3, 1.1
        (15.1, 10.0, 2.0),                                          # bin 4
        (10.0, 17.4, 5.0), (17.2, 13.1, 7.0),                       # bins 5, 6
        (19.5, 19.5, 2.0), (0.5, 0.6, 8.0),                         # corners
    ]
    f = Frame(pos=np.asarray(pts, float), cell=np.diag(L), symbols=["Ar"] * len(pts),
              pbc=(True, True, True))
    out = radial_density_profile(f, research, axis="z", bins=bins)
    brute, outside = _brute_force_radial(f.pos, L, 2, centre, r_max, bins)
    assert out["n_outside"] == outside == 2
    assert out["r_max"] == pytest.approx(r_max)
    np.testing.assert_allclose(out["profile"], brute, rtol=1e-12)
    np.testing.assert_allclose(
        out["r_centers"],
        [(k + 0.5) * r_max / bins for k in range(bins)], rtol=1e-12)
    # exact counts: the two corner atoms are excluded, the rest binned
    expected_counts = np.zeros(bins)
    for p in pts[:-2]:
        r = np.hypot(p[0] - centre[0], p[1] - centre[1])
        expected_counts[min(int(r // (r_max / bins)), bins - 1)] += 1
    counts = out["profile"] * np.pi * (
        (np.arange(1, bins + 1) * r_max / bins) ** 2
        - (np.arange(bins) * r_max / bins) ** 2) * L[2]
    np.testing.assert_allclose(counts, expected_counts)


def test_radial_profile_periodic_wrap_about_caller_centre(research):
    """Atoms near opposite box edges are both near a caller centre on the
    boundary: minimum-image offsets put each at its nearest-image distance
    from the axis (the wrap case of the reviewer letter)."""
    from chaord.cv.field import radial_density_profile
    L = np.array([16.0, 16.0, 10.0])
    bins = 16
    r_max = float(np.min(L[:2])) / 2
    centre = (0.5, 8.0)                        # axis just inside the x edge
    pts = [
        (0.2, 8.0, 5.0),                       # dx = -0.3
        (15.9, 8.0, 5.0),                      # dx = 15.4 wraps to -0.6
        (0.9, 8.2, 2.0),                       # r ~ 0.45
        (15.4, 8.3, 4.0),                      # dx wraps to -1.1, r ~ 1.14
        (8.0, 8.0, 1.0),                       # r = 7.5, outer
        (0.4, 15.7, 6.0),                      # r ~ 7.7, outer
    ]
    f = Frame(pos=np.asarray(pts, float), cell=np.diag(L), symbols=["Ar"] * len(pts),
              pbc=(True, True, True))
    out = radial_density_profile(f, research, axis="z", centre=centre, bins=bins)
    brute, outside = _brute_force_radial(f.pos, L, 2, centre, r_max, bins)
    assert out["n_outside"] == outside == 0
    np.testing.assert_allclose(out["profile"], brute, rtol=1e-12)
    # the boundary-straddling atoms land in the innermost bins
    assert (out["profile"][:3] > 0).all()


def test_radial_profile_integral_conserves_atoms(research):
    """Shell-volume integral of the profile plus n_outside returns every
    atom (number density normalisation)."""
    from chaord.cv.field import radial_density_profile
    rng = np.random.default_rng(42)
    L = np.array([20.0, 20.0, 14.0])
    pos = rng.uniform(0, 1, (120, 3)) * L
    f = Frame(pos=pos, cell=np.diag(L), symbols=["Ar"] * 120,
              pbc=(True, True, True))
    out = radial_density_profile(f, research, axis="z")
    edges_r = np.linspace(0, out["r_max"],
                          int(research.threshold("radial_profile_bins")) + 1)
    shell = np.pi * (edges_r[1:] ** 2 - edges_r[:-1] ** 2) * L[2]
    assert out["profile"] @ shell + out["n_outside"] == pytest.approx(
        len(f), abs=1e-9)


def test_radial_profile_axis_x_and_reproducible(research):
    """axis='x' bins about the x-cell-centred y-z cylinder and equals the
    independent reference; the same frame twice is byte-identical."""
    from chaord.cv.field import radial_density_profile
    rng = np.random.default_rng(5)
    L = np.array([14.0, 18.0, 18.0])
    pos = rng.uniform(0, 1, (60, 3)) * L
    f = Frame(pos=pos, cell=np.diag(L), symbols=["Ar"] * 60,
              pbc=(True, True, True))
    bins = int(research.threshold("radial_profile_bins"))
    r_max = float(np.min(L[1:])) / 2
    out = radial_density_profile(f, research, axis="x")
    out2 = radial_density_profile(f, research, axis="x")
    brute, _ = _brute_force_radial(f.pos, L, 0, (9.0, 9.0), r_max, bins)
    np.testing.assert_allclose(out["profile"], brute, rtol=1e-12)
    assert out["profile"].tobytes() == out2["profile"].tobytes()


# ---------------------------------------------- summary ratio / composer --

def test_inner_outer_ratio_exact_closed_form(research):
    """6 atoms in bin 1 (shell volume V1) and 2 in bin 7 (V7): the
    inner/outer mean-density ratio is exactly 3 * V7 / V1 = 15 for the
    10-bin unit-width shells of a 20 A box (V_k = pi*((k+1)^2 - k^2)*L_z,
    V7/V1 = 15/3)."""
    from chaord.cv.field import field_observables, inner_outer_ratio, \
        radial_density_profile
    L = np.array([20.0, 20.0, 12.0])
    pts = [(11.2, 10.0, 1.0), (10.0, 11.1, 2.0), (11.3, 10.4, 3.0),
           (10.5, 11.0, 4.0), (11.1, 10.8, 5.0), (10.9, 11.2, 6.0)] + \
          [(17.3, 10.0, 2.0), (10.0, 17.4, 8.0)]
    f = Frame(pos=np.asarray(pts, float), cell=np.diag(L),
              symbols=["Ar"] * len(pts), pbc=(True, True, True))
    out = radial_density_profile(f, research, axis="z", bins=10)
    ratio = inner_outer_ratio(out, research)
    assert ratio == pytest.approx(15.0, rel=1e-12)


def test_field_observables_keys_and_distance(research):
    """The composer returns scalars AND raw arrays; the field distance
    between a frame and itself is zero, and between orientation-differing
    frames it is positive on P1 (the future noise-floor input)."""
    from chaord.cv.field import field_distance, field_observables, \
        inner_outer_ratio, radial_density_profile
    mu0 = _template_dipole(research)
    L = np.array([24.0, 24.0, 24.0])
    thetas = [20.0, 70.0, 110.0, 160.0]
    rots = [_rotation_from_to(mu0, np.array(
        [np.sin(np.deg2rad(t)), 0.0, np.cos(np.deg2rad(t))])) for t in thetas]
    centres = [(6.0, 6.0, 12.0), (18.0, 6.0, 12.0), (6.0, 18.0, 12.0),
               (18.0, 18.0, 12.0)]
    f = _water_frame(rots, centres, L)
    o = field_observables(f, research)
    assert set(o) == {"field_P1", "field_n_molecules", "radial_profile",
                      "radial_r_centers", "radial_r_max", "radial_n_outside",
                      "radial_inner_outer_ratio"}
    assert o["field_n_molecules"] == 4
    # the composed ratio equals the standalone observable on the same frame
    assert o["radial_inner_outer_ratio"] == pytest.approx(
        inner_outer_ratio(radial_density_profile(f, research), research),
        abs=1e-15)
    self_d = field_distance(o, field_observables(f, research))
    assert self_d["field_p1_delta"] == pytest.approx(0.0, abs=1e-15)
    assert self_d["field_radial_rms"] == pytest.approx(0.0, abs=1e-15)
    assert self_d["field_ratio_delta"] == pytest.approx(0.0, abs=1e-15)
    # rotate molecule 0 to 40 deg from z -> P1 changes, atoms move -> profile changes
    thetas2 = [40.0, 70.0, 110.0, 160.0]
    rots2 = [_rotation_from_to(mu0, np.array(
        [np.sin(np.deg2rad(t)), 0.0, np.cos(np.deg2rad(t))])) for t in thetas2]
    g = _water_frame(rots2, centres, L)
    o2 = field_observables(g, research)
    d = field_distance(o, o2)
    assert d["field_p1_delta"] > 0.0
    assert d["field_radial_rms"] >= 0.0


def test_field_distance_bin_mismatch_fails(research):
    from chaord.cv.field import field_distance, field_observables
    mu0 = _template_dipole(research)
    L = np.array([24.0, 24.0, 24.0])
    f = _water_frame([np.eye(3)] * 2, [(8.0, 8.0, 12.0), (16.0, 16.0, 12.0)], L)
    a = field_observables(f, research, bins=10)
    b = field_observables(f, research, bins=20)
    with pytest.raises(ChaordError, match="bin"):
        field_distance(a, b)


# ------------------------------------------------------- registry wiring --

def test_field_cvs_registered_and_measurable(research):
    """Rule 7 (one definition per quantity): both headline scalars are in
    the CV registry and measure() dispatches to the field module."""
    from chaord.cv.field import dipole_orientation, inner_outer_ratio, \
        radial_density_profile
    for name in ("dipole_p1", "radial_inner_outer_ratio"):
        assert name in CVS, f"{name} missing from the CV registry"
    mu0 = _template_dipole(research)
    L = np.array([24.0, 24.0, 24.0])
    thetas = [0.0, 60.0, 120.0, 180.0]
    rots = [_rotation_from_to(mu0, np.array(
        [np.sin(np.deg2rad(t)), 0.0, np.cos(np.deg2rad(t))])) for t in thetas]
    centres = [(6.0, 6.0, 12.0), (18.0, 6.0, 12.0), (6.0, 18.0, 12.0),
               (18.0, 18.0, 12.0)]
    f = _water_frame(rots, centres, L)
    assert measure("dipole_p1", f, research) == pytest.approx(
        dipole_orientation(f, research)["P1"], abs=1e-15)
    rad = radial_density_profile(f, research, axis="z")
    assert measure("radial_inner_outer_ratio", f, research) == pytest.approx(
        inner_outer_ratio(rad, research), abs=1e-15)
    # thresholds are declared on the CV definitions (fail-closed contract)
    assert CVS["dipole_p1"].dialect == "research"
    assert "field_dipole_species" in CVS["dipole_p1"].thresholds


def test_molecular_research_stack_works(mol_research):
    """The stack the research case runs (core+molecular+research) resolves
    every key the observables read (bond rule from molecular, measurement
    keys from research)."""
    from chaord.cv.field import dipole_orientation
    mu0 = _template_dipole(mol_research)
    L = np.array([24.0, 24.0, 24.0])
    rots = [_rotation_from_to(mu0, np.array([0.0, 0.0, 1.0]))]
    f = _water_frame(rots, [(12.0, 12.0, 12.0)], L)
    out = dipole_orientation(f, mol_research)
    assert out["P1"] == pytest.approx(1.0, abs=1e-12)


# ------------------------------------------------------------- fail closed --

def _n2_frame():
    t = TEMPLATES["N2"]
    rel = np.asarray(t["rel"], float)
    pos = np.vstack([np.asarray([12.0, 12.0, 12.0], float) + rel,
                     np.asarray([6.0, 18.0, 12.0], float) + rel])
    return Frame(pos=pos, cell=np.diag([24.0] * 3),
                 symbols=list(t["symbols"]) * 2, pbc=(True, True, True))


def test_p1_no_named_species_refuses(research):
    """A frame whose census has no molecule of the named solvent species
    refuses (fail closed), it does not return a silent 0."""
    from chaord.cv.field import dipole_orientation
    rng = np.random.default_rng(1)
    L = np.array([20.0, 20.0, 20.0])
    f = Frame(pos=rng.uniform(2, 18, (20, 3)), cell=np.diag(L),
              symbols=["Ar"] * 20, pbc=(True, True, True))
    with pytest.raises(ChaordError, match="H2O"):
        dipole_orientation(f, research)


def test_p1_species_without_charge_model_refuses(research):
    from chaord.cv.field import dipole_orientation
    d = load_research_copy()
    d._data["thresholds"]["field_dipole_species"] = ["N2"]
    with pytest.raises(ChaordError, match="site-charge"):
        dipole_orientation(_n2_frame(), d)


def test_p1_nonneutral_charge_table_refuses(research):
    """A charge-table typo that breaks neutrality is refused at read time
    (the order parameter is defined on neutral molecules)."""
    from chaord.cv.field import dipole_orientation
    d = load_research_copy()
    d._data["thresholds"]["field_dipole_site_charges"]["H2O"]["H"] = 0.5
    mu0 = _template_dipole(research)
    f = _water_frame([_rotation_from_to(mu0, np.array([0.0, 0.0, 1.0]))],
                     [(12.0, 12.0, 12.0)], np.array([24.0] * 3))
    with pytest.raises(ChaordError, match="neutral"):
        dipole_orientation(f, d)


def test_p1_zero_dipole_refuses(research):
    """A named species whose charge model gives |mu| = 0 has no orientation;
    the measurement refuses rather than averaging an arbitrary direction."""
    from chaord.cv.field import dipole_orientation
    d = load_research_copy()
    d._data["thresholds"]["field_dipole_species"] = ["N2"]
    d._data["thresholds"]["field_dipole_site_charges"]["N2"] = {"N": 0.0}
    with pytest.raises(ChaordError, match="dipole"):
        dipole_orientation(_n2_frame(), d)


def test_p1_malformed_frame_charges_refuse(research):
    from chaord.cv.field import dipole_orientation
    mu0 = _template_dipole(research)
    f = _water_frame([_rotation_from_to(mu0, np.array([0.0, 0.0, 1.0]))],
                     [(12.0, 12.0, 12.0)], np.array([24.0] * 3))
    bad_len = Frame(pos=f.pos, cell=f.cell, symbols=f.symbols, pbc=f.pbc,
                    info={"charges": np.zeros(len(f) - 1)})
    with pytest.raises(ChaordError, match="charges"):
        dipole_orientation(bad_len, research)
    nan_q = Frame(pos=f.pos, cell=f.cell, symbols=f.symbols, pbc=f.pbc,
                  info={"charges": np.full(len(f), np.nan)})
    with pytest.raises(ChaordError, match="finite"):
        dipole_orientation(nan_q, research)


def test_field_observables_need_the_research_dialect():
    """Under a dialect stack without the research keys the observables fail
    closed on the missing threshold (design rule 5: keys live in dialects)."""
    from chaord.cv.field import dipole_orientation, radial_density_profile
    from chaord.dialects import load_dialect
    core = load_dialect(("core",))
    mu0 = _template_dipole(load_dialect(("core", "research")))
    f = _water_frame([_rotation_from_to(mu0, np.array([0.0, 0.0, 1.0]))],
                     [(12.0, 12.0, 12.0)], np.array([24.0] * 3))
    with pytest.raises(ChaordError, match="no threshold"):
        dipole_orientation(f, core)
    with pytest.raises(ChaordError, match="no threshold"):
        radial_density_profile(f, core)


def test_bad_axis_refuses(research):
    from chaord.cv.field import dipole_orientation, radial_density_profile
    mu0 = _template_dipole(research)
    f = _water_frame([_rotation_from_to(mu0, np.array([0.0, 0.0, 1.0]))],
                     [(12.0, 12.0, 12.0)], np.array([24.0] * 3))
    with pytest.raises(ChaordError, match="axis"):
        dipole_orientation(f, research, axis="w")
    with pytest.raises(ChaordError, match="axis"):
        radial_density_profile(f, research, axis=7)
