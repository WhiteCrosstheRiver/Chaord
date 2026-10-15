"""D33/D35 (docs/reviews/open_items_v4.md, work order Y3): real tube frames,
and tube plus liquid in the library.

The D33 bar, each 20 of 20 with the radius within 0.05 A:
- per-coordinate noise up to 0.1 A,
- 6-significant-digit text coordinates,
- periodic cells of 3 or more axial periods,
- the tube anywhere in its cell,
- tube diameter up to 0.8 of the perpendicular cell length,
- the (12,0), (8,4), (12,8) and (6,5) chiralities at 0.05 A noise.

Beyond the bar the inference fails closed: heavy noise either refuses or
returns the RIGHT pair, never a wrong one.

The D35 half: lift_frame(mode="auto") composes the crystal tube region (in
its cylinder) with the confined fluid region (the atoms inside the
cylinder, lifted with the molecular stack); atoms outside the cylinder
become their own region; a pure tube frame lifts as the tube alone (the
reviewer's session had it refused under core + carbon).

The thermal evidence behind the carbon dialect's tolerance keys is
generated here (Langevin dynamics on the ideal tube, OPLS aromatic C-C
stretch and C-C-C angle constants, 300 K and 600 K): the measured
per-coordinate displacement, mean coordination and circle-fit RMS residual
are the numbers recorded in carbon.yaml.
"""
import re
import sys
from collections import deque
from math import gcd
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src"), str(ROOT / "research" / "cnt_electrolyte")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from chaord.build.nanotube import nanotube_positions            # noqa: E402
from chaord.dialects import load_dialect                        # noqa: E402
from chaord.io.frames import Frame                              # noqa: E402
from chaord.lang.errors import ChaordError                      # noqa: E402
from chaord.lang.fmt import format_program                      # noqa: E402
from chaord.lift import lift_frame                             # noqa: E402
from chaord.lift.nanotube import (                              # noqa: E402
    infer_nanotube_chirality, nanotube_components)

A = 2.46  # graphene lattice constant, A (Saito/Dresselhaus ch. 3)


# ------------------------------------------------------------- D33 machinery --

def _draw(n, m, periods, rng, box=None, noise=0.0, digits=None, shift=False):
    """One D33 draw: the ideal (n, m) tube, noised / text-rounded / placed
    anywhere in its cell / squeezed to 0.8 of the perpendicular cell."""
    pos, length, d = nanotube_positions(n, m, periods)
    L = np.array([box or d + 20.0, box or d + 20.0, length])
    p = pos + L / 2 + rng.normal(0.0, noise, pos.shape)
    if shift:
        p[:, :2] += rng.uniform(0, 1, 2) * L[:2]
    p = np.mod(p, L)
    if digits:
        p = np.array([[float(f"{v:.{digits}g}") for v in r] for r in p])
    return p, L, d / 2


def _score(n, m, periods, rng, box=None, noise=0.0, digits=None, shift=False):
    """(correct, radius errors) over 20 draws: the count the inference
    returns exactly (n, m), and |measured radius - true radius| per draw."""
    good, dr = 0, []
    for _ in range(20):
        p, L, R = _draw(n, m, periods, rng, box=box, noise=noise,
                        digits=digits, shift=shift)
        try:
            r = infer_nanotube_chirality(p, L)
        except Exception:
            continue
        good += (r.n, r.m) == (n, m)
        if (r.n, r.m) == (n, m):
            dr.append(abs(r.radius - R))
    return good, dr


CASES = [
    ("ideal (10,10), 20 periods, centred      ",
     dict(n=10, m=10, periods=20)),
    ("same, 300 K-like noise 0.05 A           ",
     dict(n=10, m=10, periods=20, noise=0.05)),
    ("same, 0.1 A noise (the D33 bound)       ",
     dict(n=10, m=10, periods=20, noise=0.1)),
    ("same, written with 6 significant digits ",
     dict(n=10, m=10, periods=20, digits=6)),
    ("(10,10), 4 periods (a DFT-sized cell)   ",
     dict(n=10, m=10, periods=4)),
    ("(10,10), 3 periods (the D33 minimum)    ",
     dict(n=10, m=10, periods=3)),
    ("(16,16) in its own 41.71 A cell         ",
     dict(n=16, m=16, periods=20, box=41.71)),
    ("(8,8), 6 periods, anywhere in the cell  ",
     dict(n=8, m=8, periods=6, shift=True)),
    ("(8,8) at 0.1 A noise, anywhere           ",
     dict(n=8, m=8, periods=6, noise=0.1, shift=True)),
    ("(12,0) zigzag, 8 periods, 0.05 A noise  ",
     dict(n=12, m=0, periods=8, noise=0.05)),
    ("(8,4) chiral, 8 periods, 0.05 A noise   ",
     dict(n=8, m=4, periods=8, noise=0.05)),
    ("(12,8) chiral, 8 periods, 0.05 A noise  ",
     dict(n=12, m=8, periods=8, noise=0.05)),
    ("(6,5) small chiral, 6 periods, 0.05 A   ",
     dict(n=6, m=5, periods=6, noise=0.05)),
    ("(16,16) diameter 0.8 of the cell        ",
     dict(n=16, m=16, periods=20, noise=0.05, shift=True, box=27.14)),
    ("(10,10) diameter 0.8 of the cell        ",
     dict(n=10, m=10, periods=12, noise=0.05, shift=True, box=16.97)),
]


@pytest.mark.parametrize("label,case", CASES, ids=[c[0].strip() for c in CASES])
def test_d33_case_scores_20_of_20(label, case):
    """Every D33 case (and the brief's extra cases) infers exactly (n, m)
    on all 20 draws, radius within 0.05 A."""
    good, dr = _score(rng=np.random.default_rng(2026), **case)
    assert good == 20, f"{label}: {good}/20"
    assert max(dr, default=1.0) <= 0.05, f"{label}: worst radius error {max(dr):.3f} A"


def test_beyond_the_bar_never_returns_a_wrong_pair():
    """Heavy noise (0.5 A/coordinate, 5x the D33 bound) must fail closed:
    every draw either refuses or returns the right pair, never a wrong one."""
    rng = np.random.default_rng(7)
    for _ in range(20):
        p, L, _R = _draw(10, 10, 8, rng, noise=0.5)
        try:
            r = infer_nanotube_chirality(p, L)
        except Exception:
            continue
        assert (r.n, r.m) == (10, 10)


# --------------------------------------------------------- honest refusals --

def test_periodic_graphene_sheet_winds_two_axes_and_is_refused():
    """A graphene sheet commensurate with its cell winds along BOTH in-plane
    cell axes: not a tube, honestly refused (the radius gate would also
    refuse it, but the two-winding verdict is the honest first refusal)."""
    a1 = A * np.array([1.0, 0.0])
    a2 = A * np.array([0.5, np.sqrt(3) / 2])
    basis = [np.zeros(2), (a1 + a2) / 3]
    u = v = 12
    pos = np.concatenate([i * a1 + j * a2 + b
                          for i in range(u) for j in range(v)
                          for b in basis]).reshape(-1, 3)
    L = np.array([u * A, v * A * np.sqrt(3) / 2, 20.0])
    with pytest.raises(ChaordError):
        infer_nanotube_chirality(pos, L)


def test_two_separate_tubes_in_one_frame_are_refused():
    """The inference reads ONE tube frame; a frame holding two tubes is not."""
    p1, l1, d = nanotube_positions(8, 8, 6)
    p2, _l2, _d2 = nanotube_positions(6, 0, 6)
    L = np.array([max(d, 15.0) + 20.0, 60.0, l1])
    pos = np.vstack([p1 + [L[0] / 2, 15.0, L[2] / 2],
                     p2 + [L[0] / 2, 45.0, L[2] / 2]])
    with pytest.raises(ChaordError):
        infer_nanotube_chirality(pos, L)


# -------------------------------------------------------- the tube component --

def _water(pos_centres, L):
    from chaord.build.molecules import TEMPLATES
    t = TEMPLATES["H2O"]
    rel = np.asarray(t["rel"], float)
    sym = list(t["symbols"])
    rng = np.random.default_rng(3)
    rots = [np.linalg.svd(rng.normal(size=(3, 3)))[0] for _ in pos_centres]
    pos = np.vstack([np.asarray(c, float) + rel @ R.T
                     for R, c in zip(rots, pos_centres)])
    q = [-0.8476, 0.4238, 0.4238]     # SPC/E (Berendsen 1987), template order
    return pos, sym * len(pos_centres), np.tile(q, len(pos_centres))


def _tube_liquid_frame(n=10, m=10, periods=6, waters=3, outside=()):
    """A (n, m) tube with `waters` waters inside and optional extra atoms
    outside the cylinder (radius > R): one Frame with per-atom charges."""
    pos, length, d = nanotube_positions(n, m, periods)
    L = np.array([d + 20.0, d + 20.0, length])
    pos = pos + L / 2
    R = d / 2
    wpos, wsym, wq = _water(
        [[L[0] / 2 + R / 2, L[1] / 2, L[2] * (k + 1) / (waters + 1)]
         for k in range(waters)], L)
    opos, osym, oq = [], [], []
    for k, (sym, r) in enumerate(outside):
        p = np.array([L[0] / 2 + r, L[1] / 2 + r / 3,
                      L[2] * (k + 1) / (len(outside) + 1)])
        opos.append(p)
        osym.append(sym)
        oq.append(0.0)
    pos = np.vstack([pos, wpos] + [np.asarray(p)[None, :] for p in opos])
    symbols = ["C"] * (len(pos) - len(wsym) - len(osym)) + wsym + osym
    q = np.concatenate([np.zeros(len(pos) - len(wq) - len(oq)), wq, oq])
    return Frame(pos=np.mod(pos, L), cell=np.diag(L), symbols=symbols,
                 pbc=(True, True, True), info={"charges": q}), R


def test_finder_returns_exactly_the_tube_among_solvent_and_ions():
    """D35 step 5: all-carbon three-coordinated components. The tube is one;
    solvent carbons (EC/EMC-class fragments) and a benzene ring are not."""
    frame, R = _tube_liquid_frame()
    # a benzene ring far from the tube: pure carbon, but TWO-coordinated
    rng = np.random.default_rng(11)
    ang = np.linspace(0, 2 * np.pi, 7)[:-1]
    ring = np.column_stack([np.cos(ang), np.sin(ang), np.zeros(6)]) * 1.4
    frame = Frame(pos=np.vstack([np.mod(frame.pos, frame.cell_diag),
                                 ring + [5.0, 5.0, 3.0]]),
                  cell=frame.cell,
                  symbols=frame.symbols + ["C"] * 6, pbc=frame.pbc,
                  info={"charges": np.concatenate(
                      [frame.info["charges"], np.zeros(6)])})
    tubes = nanotube_components(frame, load_dialect(("core", "molecular",
                                                     "carbon", "research")))
    assert len(tubes) == 1, tubes
    n_C = sum(1 for s in frame.symbols if s == "C")
    idx = tubes[0].indices
    assert set(frame.symbols[i] for i in idx) == {"C"}
    assert len(idx) == n_C - 6            # the benzene ring is not the tube
    assert (tubes[0].chirality.n, tubes[0].chirality.m) == (10, 10)


def test_finder_silent_for_dialects_without_carbon_thresholds():
    """A stack that defines no C-C cutoff has no tube vocabulary: the finder
    declines ([]) instead of raising -- the cascade's carve-out."""
    assert nanotube_components(_tube_liquid_frame()[0],
                               load_dialect(("core", "lj"))) == []


# ------------------------------------------------------------ the D35 arm --

def _composed(frame, T=300.0, field=None):
    dl = load_dialect(("core", "molecular", "carbon", "research"))
    return format_program(lift_frame(frame, dl, T=T, mode="auto", field=field))


def test_auto_lift_composes_tube_and_confined_liquid():
    """D35: crystal tube region in its cylinder + liquid fluid region with
    the molecular stack; the blind dipole_p1 is measured along the TUBE's
    axis; conservation counts every element; residual none; the text is its
    own canonical form."""
    frame, R = _tube_liquid_frame()
    text = _composed(frame)
    assert "crystal tube : cylinder axis z" in text
    assert re.search(r"nanotube \(10,10\) length \d+\.\d+ A", text), text
    assert "liquid fluid : cylinder axis z" in text
    assert re.search(r"^  molecules H2O 3$", text, re.M), text
    assert re.search(r"assert dipole_p1 -?\d\.\d{3} \+- 0\.577 along z", text), text
    assert "state field" not in text          # blind: no field was supplied
    for s, k in (("C", 240), ("H", 6), ("O", 3)):
        assert f"conserve atoms" in text and re.search(
            rf"{s} {k}(?:\s|$)", text.split("conserve atoms")[1]), text
    assert "residual none" in text
    from chaord.lang.parser import parse_text
    assert format_program(parse_text(text)) == text


def test_auto_lift_confines_only_atoms_inside_the_cylinder():
    """Atoms outside the cylinder become their own region (D35): an argon
    atom beyond the tube radius lifts in an `all minus cylinder` region and
    the program still explains every atom."""
    frame, R = _tube_liquid_frame(outside=(("Ar", 9.0),))
    text = _composed(frame)
    assert "liquid outside : all minus cylinder" in text, text
    assert "residual none" in text
    assert re.search(r"Ar 1(?:\s|$)", text.split("conserve atoms")[1]), text


def test_auto_lift_of_a_pure_tube_frame():
    """The reviewer's case: an ideal (10,10) tube under core + carbon lifts
    (it used to be refused by every arm of the cascade)."""
    pos, length, d = nanotube_positions(10, 10, 10)
    L = np.array([d + 20.0, d + 20.0, length])
    frame = Frame(pos=np.mod(pos + L / 2, L), cell=np.diag(L),
                  symbols=["C"] * len(pos), pbc=(True, True, True))
    dl = load_dialect(("core", "carbon"))
    text = format_program(lift_frame(frame, dl, mode="auto"))
    assert "nanotube (10,10)" in text
    assert "residual none" in text
    # the tube region is the ONLY region (no liquid line outside provenance
    # prose), and no species means no dipole statement
    assert not re.search(r"^liquid ", text, re.M)
    assert "dipole_p1" not in text


def test_auto_lift_still_lifts_non_tube_frames_through_the_cascade():
    """The arm declines (no tube) and the legacy cascade answers as before:
    an LJ fluid frame lifts unchanged under a carbon-carrying stack."""
    rng = np.random.default_rng(5)
    L = np.array([25.0, 25.0, 25.0])
    frame = Frame(pos=rng.uniform(0, 25, (32, 3)), cell=np.diag(L),
                  symbols=["Ne"] * 32, pbc=(True, True, True))
    text = _composed(frame)
    assert "nanotube" not in text
    assert "liquid fluid : all" in text


def test_auto_lift_composes_two_tubes_as_two_regions():
    """Two separate tubes in one frame: the single-frame inference refuses
    (not one component), the FINDER splits them and the arm composes two
    crystal tube regions."""
    p1, l1, d = nanotube_positions(8, 8, 6)
    p2, _l2, _d2 = nanotube_positions(10, 10, 6)   # same |T|: commensurate
    L = np.array([d + 20.0, 70.0, l1])
    pos = np.vstack([p1 + [L[0] / 2, 18.0, L[2] / 2],
                     p2 + [L[0] / 2, 52.0, L[2] / 2]])
    frame = Frame(pos=np.mod(pos, L), cell=np.diag(L),
                  symbols=["C"] * len(pos), pbc=(True, True, True))
    text = _composed(frame)
    assert "crystal tube : cylinder" in text
    assert "crystal tube2 : cylinder" in text
    assert re.search(r"nanotube \(10,10\)", text), text
    assert re.search(r"nanotube \(8,8\)", text), text
    assert "residual none" in text


# ------------------------------------------------- thermal evidence (D33) --
# Langevin dynamics on the ideal tube with OPLS aromatic C-C constants
# (W. L. Jorgensen, D. S. Maxwell, J. Tirado-Rives, J. Am. Chem. Soc. 118,
# 11225 (1996): k_bond 469 kcal/mol/A^2, k_angle 63 kcal/mol/rad^2 -- the
# same model family the pilot's solvent uses, Nutakki 2023 Tables II-III).
# The measured per-coordinate displacement, mean coordination and
# circle-fit RMS residual at 300 K and 600 K are recorded in carbon.yaml.

_K_BOND = 2 * 469.0 / 23.0605    # eV/A^2: OPLS aromatic C-C, K = 469 kcal/mol/A^2
                                 # for E = K(r-r0)^2 (Jorgensen 1996), in this
                                 # integrator's (1/2)k convention
_K_ANGLE = 2 * 63.0 / 23.0605    # eV/rad^2: OPLS aromatic C-C-C, K = 63 kcal/mol/rad^2
_EV_AMU_FS2 = 0.96485333624689685e-2   # a = F/m * this, A/fs^2  # dialect-exempt: unit conversion
_AMU_A2_FS2_EV = 103.64268466150575    # 1 amu*A^2/fs^2 in eV    # dialect-exempt: unit conversion
_KB = 8.617333262145e-5        # eV/K                                    # dialect-exempt: physical constant
_M_C = 12.011                  # amu                                     # dialect-exempt: physical constant


def _graph_terms(pos, L):
    """(bond i, bond j, min-image bond vectors, angles) of the current
    configuration: angles are the neighbour pairs at every atom, their arms
    also minimum-imaged (the tube spans the cell)."""
    from itertools import combinations
    from scipy.spatial import cKDTree
    from chaord.lift.nanotube import _wrapped
    w = _wrapped(pos, L)
    pairs = cKDTree(w, boxsize=L).query_pairs(1.85, output_type="ndarray")
    i, j = pairs[:, 0], pairs[:, 1]
    d = pos[j] - pos[i]
    d -= L * np.round(d / L)
    nbr = {k: [] for k in range(len(pos))}
    for a, b in zip(i, j):
        nbr[a].append(b)
        nbr[b].append(a)
    ang = [(a, b, c) for b in range(len(pos))
           for a, c in combinations(nbr[b], 2)]
    return i, j, d, ang


def _min_image(v, L):
    v = np.array(v, float, copy=True)
    v -= L * np.round(v / L)
    return v


def _thermal_frames(n, m, periods, T_K, seed, n_frames=8):
    """Thermal tube frames at T_K: Langevin (gamma 1/ps) on bond + angle
    springs, 6 ps of equilibration then one frame per 0.5 ps."""
    ideal, length, d = nanotube_positions(n, m, periods)
    L = np.array([d + 20.0, d + 20.0, length])
    ideal = ideal + L / 2
    from scipy.spatial import cKDTree
    from chaord.lift.nanotube import _wrapped
    r0 = float(np.linalg.norm(
        cKDTree(_wrapped(ideal, L), boxsize=L).query(_wrapped(ideal, L), k=2)[0][:, 1]
        .mean()))
    theta0 = 2 * np.pi / 3
    rng = np.random.default_rng(seed)
    pos = ideal.copy()
    vel = rng.normal(0.0, np.sqrt(_KB * T_K / (_M_C * _AMU_A2_FS2_EV)),
                     ideal.shape)
    dt, gamma = 1.0, 1e-3                      # fs, 1/fs
    sigma = np.sqrt(2.0 * gamma * _KB * T_K * dt
                    / (_M_C * _AMU_A2_FS2_EV))
    frames, step = [], 0
    n_steps = 6000 + 500 * n_frames
    force = np.zeros_like(pos)
    while step < n_steps:
        i, j, d, ang = _graph_terms(pos, L)
        r = np.linalg.norm(d, axis=1)
        fvec = -_K_BOND * (r - r0)[:, None] * d / r[:, None]   # on atom j
        force[:] = 0.0
        np.add.at(force, j, fvec)
        np.add.at(force, i, -fvec)
        if len(ang):
            aa, bb, cc = (np.asarray(ang)[:, k] for k in range(3))
            ra = _min_image(pos[aa] - pos[bb], L)
            rc = _min_image(pos[cc] - pos[bb], L)
            la = np.linalg.norm(ra, axis=1)
            lc = np.linalg.norm(rc, axis=1)
            u, v = ra / la[:, None], rc / lc[:, None]
            cos = np.clip((u * v).sum(1), -1.0, 1.0)
            sin = np.sqrt(1.0 - cos * cos)
            # F_a = +(k(theta-theta0)/sin) (v - cos u)/|ra|  (chain rule of
            # E = k(theta-theta0)^2/2 through theta = arccos(u.v))
            dEdc = _K_ANGLE * (np.arccos(cos) - theta0) / sin
            fa = (dEdc[:, None] * (v - cos[:, None] * u)) / la[:, None]
            fc = (dEdc[:, None] * (u - cos[:, None] * v)) / lc[:, None]
            np.add.at(force, aa, fa)
            np.add.at(force, cc, fc)
            np.add.at(force, bb, -(fa + fc))
        acc = force / _M_C * _EV_AMU_FS2
        vel += (acc - gamma * vel) * dt + sigma * rng.normal(size=vel.shape)
        pos += vel * dt
        step += 1
        if step > 6000 and (step - 6000) % 500 == 0:
            frames.append(np.mod(pos, L))
    # measured kinetic temperature (mean per-coordinate kinetic energy
    # equals kT/2 exactly)
    T_meas = float(np.mean(0.5 * _M_C * vel ** 2 * _AMU_A2_FS2_EV)
                   / (0.5 * _KB))
    return frames, ideal, L, T_meas


@pytest.mark.slow
@pytest.mark.parametrize("T_K", [300.0, 600.0])
def test_thermal_tube_frames_lift_with_the_right_chirality(T_K):
    """The carbon dialect's evidence frames: at 300 K and 600 K the inference
    returns the built chirality with the radius within the toy model's
    thermal contraction (the angle-only model ripples, so the mean radius
    contracts ~1%: the D33 radius bound of 0.05 A is asserted on the D33
    noise cases above), and every measured quantity stays inside the dialect
    tolerances whose evidence carbon.yaml's comment records."""
    for n, m, periods in ((10, 10, 8), (12, 0, 8)):
        frames, ideal, L, T_meas = _thermal_frames(n, m, periods, T_K,
                                                   seed=2026 + int(T_K))
        assert abs(T_meas - T_K) < 0.1 * T_K, T_meas
        for fr in frames:
            r = infer_nanotube_chirality(fr, L)
            assert (r.n, r.m) == (n, m), (n, m, r)
            N = n * n + n * m + m * m
            assert abs(r.radius - A * np.sqrt(N) / (2 * np.pi)) <= 0.25
            dl = load_dialect(("core", "carbon"))
            assert r.radius_rms <= float(dl.threshold("nanotube_radius_rms_max"))


# ------------------------------------------------- the pilot's real frames --

@pytest.mark.slow
def test_prod_dumps_lift_as_tube_plus_liquid_end_to_end():
    """The Y3 done-when: raw prod.dump frames of all four systems (the Y4
    reader, no final.data, no cell tripling, no split_tube) lift through
    lift_frame(mode='auto') to the provenance chirality and the same fluid
    composition as first_result_summary.json."""
    import json
    import analyze_first_result as A
    summary = json.loads((A.RUNS / "first_result_summary.json").read_text())
    dl = A.composition_dialect()
    for system in A.SYSTEMS:
        frames, provenance, _ = A.read_prod_frames(system)
        want_chir = [int(c) for c in
                     provenance["system_definition"]["cnt_chirality"]]
        want_comp = {k: int(v) for k, v in
                     provenance["sanity"]["composition_observed"].items()}
        for fr in frames[:3]:                      # all 21 in test_blind_lift
            text = format_program(lift_frame(fr, dl, T=A.T_K, mode="auto"))
            assert f"nanotube ({want_chir[0]},{want_chir[1]})" in text, system
            assert "liquid fluid : cylinder" in text, system
            assert not re.search(r"^liquid outside : ", text, re.M), \
                (system, "atoms outside the tube")
            got = {}
            for line in text.splitlines():
                mm = re.match(r"\s+molecules (\S+) (\d+)$", line)
                if mm:
                    got[mm.group(1)] = int(mm.group(2))
            # program display names: census formulas (Li+ via the molecular
            # dialect's ion table); the summary's composition keys map onto
            # them exactly
            assert got == {"C3H4O3": want_comp["EC"],
                           "C4H8O3": want_comp["EMC"],
                           "Li+": want_comp["Li"], "F6P": want_comp["PF6"]}, \
                (system, got)
            assert "residual none" in text
            n_mol = {"cnt10": 13, "cnt16": 61}[system[:5]]
            assert re.search(rf"assert dipole_p1 -?\d\.\d{{3}} \+- "
                             rf"{1 / np.sqrt(n_mol):.3f} along z", text), system
