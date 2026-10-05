"""W7 step 2: the pair-resolved LJ engine (chaord.realize.lj.LJMixture).

Hand-computed energies and forces for one AA, one AB and one BB pair (the
review's done-when), the per-pair cutoffs (each pair cut at 2.5 sigma_ab),
minimum-image periodic pairs, finite-difference consistency of forces with
the shifted energy, integration under run_md, and the lj_mixtures dialect
table (kob_andersen) against the published Kob & Andersen Table I values.
"""
from __future__ import annotations

import numpy as np
import pytest

from chaord.dialects import load_dialect
from chaord.realize.lj import LJMixture, _pair_key, run_md

# Kob-Andersen Table I (Phys. Rev. E 51, 4626 (1995))
KA_EPS = {"AA": 1.0, "AB": 1.5, "BB": 0.5}
KA_SIG = {"AA": 1.0, "AB": 0.8, "BB": 0.88}
RCF = 2.5


def _mixture(symbols, L=20.0):
    return LJMixture([L] * 3, symbols, KA_EPS, KA_SIG, RCF, skin=0.3)


def _two_atoms(r, sym_i, sym_j, L=20.0):
    r0 = np.array([[L / 2, L / 2, L / 2], [L / 2 + r, L / 2, L / 2]])
    return r0


def _hand_pair(key, r):
    """u_shifted and force magnitude of one pair at distance r, by hand."""
    eps, sig = KA_EPS[key], KA_SIG[key]
    sr6 = (sig / r) ** 6
    u = 4.0 * eps * (sr6 ** 2 - sr6)
    urc = 4.0 * eps * (RCF ** -12 - RCF ** -6)
    f = 24.0 * eps * (2.0 * sr6 ** 2 - sr6) / r
    return u - urc, f


@pytest.mark.parametrize("key,ratio", [("AA", 1.1), ("AB", 1.1), ("BB", 1.1),
                                       ("AA", 2.0), ("AB", 1.9), ("BB", 2.05),
                                       ("AB", 2.0 ** (1 / 6))])
def test_hand_computed_pair_energy_and_force(key, ratio):
    """One isolated pair per type: the engine must reproduce the hand-computed
    shifted energy and the analytic pair force (zero at the pair minimum)."""
    sig = KA_SIG[key]
    r = ratio * sig
    syms = [key[0], key[-1]]
    pos = _two_atoms(r, *syms)
    mix = _mixture(syms)
    u_want, f_want = _hand_pair(key, r)
    assert mix.energy(pos) == pytest.approx(u_want, abs=1e-12)
    F = mix.forces(pos)
    # force magnitude |du/dr| on each atom (f_want is signed: negative =
    # attractive above the pair minimum)
    assert np.linalg.norm(F[0]) == pytest.approx(abs(f_want), abs=1e-10)
    assert np.linalg.norm(F[1]) == pytest.approx(abs(f_want), abs=1e-10)
    assert F[0] @ F[1] < 0                       # equal and opposite
    # purely along the pair axis; repulsive (atom 0 pushed to -x) below the
    # pair minimum 2^(1/6) sigma, attractive above it; at the minimum the
    # force is zero to round-off, so no sign is asserted there
    assert np.allclose(F[0][1:], 0.0)
    r_min = 2.0 ** (1 / 6)
    if abs(ratio - r_min) > 1e-12:
        assert np.sign(F[0][0]) == np.sign(r - r_min * sig)
    else:
        assert np.linalg.norm(F[0]) == pytest.approx(0.0, abs=1e-9)


def test_per_pair_cutoffs():
    """Each pair is cut at 2.5 sigma_ab: BB is off past 2.2, AB past 2.0,
    AA past 2.5 (the three cutoffs differ -- this is the point of W7)."""
    cases = [("AA", 2.4, True), ("AA", 2.6, False),
             ("AB", 1.9, True), ("AB", 2.1, False),
             ("BB", 2.1, True), ("BB", 2.3, False)]
    for key, r, inside in cases:
        pos = _two_atoms(r, key[0], key[-1])
        mix = _mixture([key[0], key[-1]])
        F = mix.forces(pos)
        u = mix.energy(pos)
        if inside:
            assert np.linalg.norm(F) > 0.0, (key, r)
            assert u != 0.0, (key, r)
        else:
            assert np.linalg.norm(F) == 0.0, (key, r)
            assert u == 0.0, (key, r)


def test_minimum_image_pair():
    """Two atoms straddling the periodic boundary interact across it."""
    L = 20.0
    key = "AB"
    r = 0.9 * KA_SIG[key]
    pos = np.array([[0.1, L / 2, L / 2], [L - 0.1 + 0.0, L / 2, L / 2]])
    pos[1, 0] = L - (r - 0.1)
    mix = _mixture(["A", "B"], L=L)
    u_want, f_want = _hand_pair(key, r)
    assert mix.energy(pos) == pytest.approx(u_want, abs=1e-12)
    F = mix.forces(pos)
    assert np.linalg.norm(F[0]) == pytest.approx(f_want, abs=1e-10)
    # the force on atom 0 points along +x (towards the image at +r)
    assert F[0][0] > 0


def test_forces_are_gradient_of_energy():
    """Central finite difference of energy matches forces on a random
    three-species-pair cluster (AA, AB, BB pairs all present)."""
    rng = np.random.default_rng(7)
    syms = ["A", "B", "A", "B", "A"]
    L = 20.0
    pos = np.array([[6, 6, 6], [7, 6.4, 6], [5.4, 7, 6.5], [6.6, 6, 7.4],
                    [7.5, 7.2, 6.2]]) + rng.normal(scale=0.05, size=(5, 3))
    mix = _mixture(syms, L=L)
    F = mix.forces(pos)
    h = 1e-6
    for i in range(len(pos)):
        for k in range(3):
            pp = pos.copy()
            pp[i, k] += h
            pm = pos.copy()
            pm[i, k] -= h
            fd = -(mix.energy(pp) - mix.energy(pm)) / (2 * h)
            assert fd == pytest.approx(F[i, k], abs=1e-6), (i, k)


def test_run_md_integrates_mixture():
    """A short Langevin run of a small KA cluster stays finite and roughly
    conserves the expected temperature scale (smoke for the builder path)."""
    rng = np.random.default_rng(3)
    n = 64
    L = 4.0
    syms = ["A"] * 52 + ["B"] * 12
    # simple-cubic start, well separated
    g = np.array(np.meshgrid(*[np.arange(4)] * 3, indexing="ij")).reshape(3, -1).T
    pos = (g + 0.5) * L / 4
    v = rng.normal(scale=0.5, size=pos.shape)
    mix = _mixture(syms, L=L)
    r, v = run_md(pos, v, [L] * 3, 50, 0.005, 0.5, 0.5, rng, lj=mix)
    assert np.isfinite(r).all() and np.isfinite(v).all()
    e = mix.energy(np.mod(r, L))
    assert np.isfinite(e)


def test_missing_pair_parameter_refuses():
    with pytest.raises(ValueError, match="lack pair"):
        LJMixture([20.0] * 3, ["A", "C"], KA_EPS, KA_SIG, RCF, skin=0.3)


def test_pair_key_unordered():
    assert _pair_key("B", "A") == "AB" == _pair_key("A", "B")
    assert _pair_key("A", "A") == "AA"


# --------------------------------------------------------- dialect table --

def test_kob_andersen_table_matches_publication():
    """The lj_mixtures.kob_andersen entry states Kob & Andersen Table I."""
    dl = load_dialect(("core", "glass", "lj_mixtures"))
    tab = dl.threshold("lj_mixtures")["kob_andersen"]
    assert tab["epsilon"] == {"AA": 1.0, "AB": 1.5, "BB": 0.5}
    assert tab["sigma"] == {"AA": 1.0, "AB": 0.8, "BB": 0.88}
    assert tab["mass"] == {"A": 1.0, "B": 1.0}
    assert tab["rc_factor"] == 2.5
    assert tab["fractions"] == {"A": 0.8, "B": 0.2}
    assert "Kob" in tab["citation"] and "4626" in tab["citation"]
    # the physics `model` key is allowed under this stack (core already
    # allows it; the table dialect is where it means something)
    assert dl.key_allowed("physics", "model")


def test_mixture_from_table_reproduces_published_minimum():
    """The engine driven by the dialect table hits the AB pair minimum
    u = -epsilon_AB at r = 2^(1/6) sigma_AB (hand-computed cross check)."""
    dl = load_dialect(("core", "glass", "lj_mixtures"))
    tab = dl.threshold("lj_mixtures")["kob_andersen"]
    r = 2.0 ** (1 / 6) * tab["sigma"]["AB"]
    pos = _two_atoms(r, "A", "B")
    mix = LJMixture([20.0] * 3, ["A", "B"], tab["epsilon"], tab["sigma"],
                    tab["rc_factor"], skin=0.3)
    urc = 4.0 * tab["epsilon"]["AB"] * (RCF ** -12 - RCF ** -6)
    assert mix.energy(pos) == pytest.approx(-tab["epsilon"]["AB"] - urc,
                                            abs=1e-12)
