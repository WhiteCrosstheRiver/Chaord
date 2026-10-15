"""Y5 (docs/reviews/open_items_v4.md, D36-approved): the electrolyte
statements of the confined LiPF6/EC:EMC research case.

  1. cv/field.py radial profile: r_max from the caller (the nanotube region
     passes its tube radius); an empty outer shell raises the documented
     not-measurable error, never ZeroDivisionError.
  2. The Li+ solvation shell BY SPECIES (carbonyl O of EC and of EMC, F of
     PF6-) and the ion-pair classes SSIP / CIP / AGG, with cutoffs from
     reference runs written into the research dialect with their provenance
     (never measured on the frames under test -- D36).
  3. Species radial number-density profiles inside the tube radius, bin
     width from the dialect.
  4. Density on the ACCESSIBLE volume pi(R - r_wall)^2 L (r_wall from the
     dialect), with the linear density (molecules per A) next to it.
  5. The molecule-centre `cn` assert for molecular fluids (0.0 +- 0.0 in 55
     of 84 pilot frames) is replaced by the site-based solvation-shell
     asserts (2): the centre-centre count at the only defensible cutoff
     (the O-O first shell, cn_cutoff_fluid) asserts that no two molecule
     centres come within 3.10 A of each other -- trivially true for 10- and
     15-atom molecules, hence uninformative; the Li+ shell carries the
     coordination physics of this electrolyte.

Done-when (the brief): all 84 pilot frames lift with these statements and
no ZeroDivisionError, and 20 random rigid transforms of one frame give
identical text.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src"), str(ROOT / "research" / "cnt_electrolyte")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from chaord.build.molecules import TEMPLATES               # noqa: E402
from chaord.dialects import load_dialect                   # noqa: E402
from chaord.io.frames import Frame                         # noqa: E402
from chaord.lang.errors import ChaordError                 # noqa: E402
from chaord.lang.fmt import format_program                 # noqa: E402
from chaord.lift import lift_frame                        # noqa: E402

import analyze_first_result as A                           # noqa: E402

# U/A^3 -> g/cm3 (the repo's exact constant, fluid.py / lift/__init__.py)
_U_PER_A3_TO_G_CM3 = 0.6022140857

# ------------------------------------------------------------- the dialect --

R_STACK = ("core", "molecular", "carbon", "research")


def _research(**overrides):
    """The research stack of the composition, optionally with threshold
    overrides (a private copy: the shared dialect object stays frozen)."""
    from chaord.dialects import Dialect
    dl = load_dialect(R_STACK)
    if not overrides:
        return dl
    thresholds = dict(dl._data["thresholds"])
    thresholds.update(overrides)
    return Dialect(dl.names, {**dl._data, "thresholds": thresholds}, dl.versions)


# --------------------------------------------- statement text of one program --

_LI_SHELL = re.compile(
    r"^  assert li_shell (\S+) (\d+\.\d) \+- (\d+\.\d) (\S+) (\d+\.\d{2}) A$")
_ION_PAIRS = re.compile(
    r"^  assert ion_pairs SSIP (\d+) CIP (\d+) AGG (\d+)$")
_N_R = re.compile(r"^  state n_r (\S+) ((?:\d+\.\d{4} )*\d+\.\d{4}) bin (\d+\.\d) A$")
_DENSITY_ACC = re.compile(
    r"^  state density (\d+\.\d{3}) g/cm3 accessible$", re.M)
_LINEAR = re.compile(
    r"^  state linear_density (\d+\.\d{3}) A\^-1$", re.M)
_MOLECULES = re.compile(r"^  molecules (\S+) (\d+)$")
_TUBE_RADIUS = re.compile(
    r"^liquid fluid : cylinder axis [xyz] center \S+ \S+ \S+ radius (\d+\.\d{3}) A "
    r"length (\d+\.\d{3}) A \{$", re.M)


def _region_block(text: str, header: str) -> str:
    """The body of the region whose header line starts with `header`
    (e.g. 'liquid fluid'), up to the closing brace."""
    lines = text.splitlines()
    start = next(k for k, l in enumerate(lines) if l.startswith(header))
    body = []
    for l in lines[start + 1:]:
        if l == "}":
            break
        body.append(l)
    return "\n".join(body)


def _fluid_block(text: str) -> str:
    return _region_block(text, "liquid fluid")


def assert_y5_statements(text: str, expect: dict) -> None:
    """The full Y5 statement contract of one composed pilot program.

    `expect` = {"n_li": int, "species": set of display names in the fluid
    census, "shell_species": {display: (site word, cutoff str)},
    "density": float (independent recomputation), "linear": float,
    "tube": (radius, length) as parsed from the text}.

    Every missing or malformed statement fails HERE (also the checker the
    power mutations must not pass)."""
    fluid = _fluid_block(text)
    # 2: one li_shell line per shell species named by the dialect, with the
    # dialect's cutoff and site word
    shells = {m.group(1): m for m in map(_LI_SHELL.match, fluid.splitlines())
              if m}
    for name, (site, cut) in expect["shell_species"].items():
        assert name in shells, (f"no `assert li_shell {name}` line:\n" + fluid)
        m = shells[name]
        assert m.group(4) == site and m.group(5) == cut, m.group(0)
    # 2: exactly one ion_pairs line whose classes sum to the Li count
    ip = [_ION_PAIRS.match(l) for l in fluid.splitlines()]
    ip = [m for m in ip if m]
    assert len(ip) == 1, fluid
    s, c, g = (int(x) for x in ip[0].groups())
    assert s + c + g == expect["n_li"], ip[0].group(0)
    # 3: one n_r line per census species, all with the dialect bin width
    nr = {m.group(1): m for m in map(_N_R.match, fluid.splitlines()) if m}
    assert set(nr) == expect["species"], (sorted(nr), sorted(expect["species"]))
    n_bins = int(expect["tube"][0] // float(next(iter(nr.values())).group(3)))
    for name, m in nr.items():
        values = m.group(2).split()
        assert len(values) == n_bins, (name, len(values), n_bins)
        assert m.group(3) == next(iter(nr.values())).group(3)
    # 4: accessible-volume density and the linear density next to it
    dens = _DENSITY_ACC.search(fluid)
    lin = _LINEAR.search(fluid)
    assert dens and lin, fluid
    assert abs(float(dens.group(1)) - expect["density"]) <= 0.002, dens.group(0)
    assert abs(float(lin.group(1)) - expect["linear"]) <= 0.001, lin.group(0)
    # 5: the molecule-centre cn assert is gone from the confined fluid
    assert "assert cn" not in fluid, fluid


# -------------------------------- Y5 item 1: the radial profile r_max fix --

def test_radial_profile_takes_r_max_from_the_caller():
    """r_max is a caller argument (the nanotube region passes its tube
    radius): bins run to it, and atoms beyond are n_outside."""
    from chaord.cv.field import radial_density_profile
    dl = load_dialect(("core", "research"))
    L = np.array([40.0, 40.0, 20.0])
    rng = np.random.default_rng(4)
    pos = rng.uniform(0, 1, (200, 3)) * L
    f = Frame(pos=pos, cell=np.diag(L), symbols=["Ar"] * 200,
              pbc=(True, True, True))
    out = radial_density_profile(f, dl, axis="z", centre=(20.0, 20.0),
                                 bins=10, r_max=6.781)
    assert out["r_max"] == pytest.approx(6.781)
    edges = np.linspace(0, 6.781, 11)
    r = np.hypot(pos[:, 0] - 20.0, pos[:, 1] - 20.0)
    inside = r[r <= 6.781]
    h, _ = np.histogram(r[r <= 6.781], edges)
    shell = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2) * L[2]
    np.testing.assert_allclose(out["profile"], h / shell, rtol=1e-12)
    assert out["n_outside"] == int((r > 6.781).sum())
    assert len(inside) + out["n_outside"] == len(f)


def test_radial_profile_r_max_over_half_the_cell_refuses():
    """A cylinder beyond min(L_perp)/2 overlaps its own periodic image; the
    measurement refuses instead of normalising a double-counting shell."""
    from chaord.cv.field import radial_density_profile
    dl = load_dialect(("core", "research"))
    L = np.array([20.0, 30.0, 10.0])
    rng = np.random.default_rng(5)
    f = Frame(pos=rng.uniform(0, 1, (50, 3)) * L, cell=np.diag(L),
              symbols=["Ar"] * 50, pbc=(True, True, True))
    with pytest.raises(ChaordError, match="r_max"):
        radial_density_profile(f, dl, axis="z", r_max=11.0)


def test_inner_outer_ratio_empty_outer_shell_refuses_not_zero_division():
    """Y5 item 1's bug: a fully depleted outer shell (a narrow pore with
    vacuum at the wall) used to divide by the empty shell's zero mean and
    raise ZeroDivisionError; the documented not-measurable error is a
    ChaordError."""
    from chaord.cv.field import inner_outer_ratio, radial_density_profile
    dl = load_dialect(("core", "research"))
    L = np.array([20.0, 20.0, 10.0])
    # every atom at r <= 1 A from the axis: bins beyond hold vacuum
    pts = [[10.0 + dx, 10.0, z] for dx in (0.0, 0.5, -0.5, 0.9)
           for z in (2.0, 5.0, 8.0)]
    f = Frame(pos=np.asarray(pts, float), cell=np.diag(L),
              symbols=["Ar"] * len(pts), pbc=(True, True, True))
    out = radial_density_profile(f, dl, axis="z", centre=(10.0, 10.0),
                                 bins=10, r_max=10.0)
    assert (out["profile"][5:] == 0).all()          # the outer half is empty
    with pytest.raises(ChaordError, match="outer shell"):
        inner_outer_ratio(out, dl)


# -------------------------- Y5 item 2: the solvation shell and ion pairs --

def _place(template: str, rotation, centre) -> tuple[np.ndarray, list]:
    t = TEMPLATES[template]
    rel = np.asarray(t["rel"], float)
    return np.asarray(centre, float) + rel @ rotation.T, list(t["symbols"])


def _site_offset(template: str, site: int, rotation) -> np.ndarray:
    """The rotated offset of template atom `site` from the molecule's
    (centroid) placement centre."""
    return np.asarray(TEMPLATES[template]["rel"], float)[site] @ rotation.T


def _carbonyl_index(template: str) -> int:
    """The template's carbonyl O, identified INDEPENDENTLY of the code under
    test: the O with the shortest C-O distance (C=O 1.20 A, against the
    1.35/1.43 A single bonds)."""
    rel = np.asarray(TEMPLATES[template]["rel"], float)
    syms = TEMPLATES[template]["symbols"]
    return min((i for i, s in enumerate(syms) if s == "O"),
               key=lambda i: min(np.linalg.norm(rel[i] - rel[j])
                                 for j, s2 in enumerate(syms) if s2 == "C"))


def _place_site(template: str, site: int, direction, distance, origin):
    """Place `template` so its atom `site` sits `distance` A from `origin`
    along `direction` (unit), the MOLECULE on the far side (the site is its
    nearest atom to `origin`)."""
    u = np.asarray(direction, float)
    u /= np.linalg.norm(u)
    off = _site_offset(template, site, np.eye(3))
    rot = _rot_to(off, -u)          # site points back at the origin
    target = np.asarray(origin, float) + distance * u
    centre = target - _site_offset(template, site, rot)
    return _place(template, rot, centre)


def _frame_of(blocks, L):
    pos = np.vstack([p for p, _ in blocks])
    syms = [x for _, s in blocks for x in s]
    return Frame(pos=pos, cell=np.diag(np.asarray(L, float)), symbols=syms,
                 pbc=(True, True, True))


def _rot_to(u, v):
    """Rotation matrix mapping unit vector u onto v (Rodrigues)."""
    u = np.asarray(u, float) / np.linalg.norm(u)
    v = np.asarray(v, float) / np.linalg.norm(v)
    c = float(np.dot(u, v))
    if c > 1.0 - 1e-12:
        return np.eye(3)
    if c < -1.0 + 1e-12:
        perp = np.cross(u, [1.0, 0.0, 0.0])
        if np.linalg.norm(perp) < 1e-12:
            perp = np.cross(u, [0.0, 1.0, 0.0])
        perp /= np.linalg.norm(perp)
        K = np.array([[0.0, -perp[2], perp[1]],
                      [perp[2], 0.0, -perp[0]],
                      [-perp[1], perp[0], 0.0]])
        return np.eye(3) + 2.0 * (K @ K)
    axis = np.cross(u, v)
    axis /= np.linalg.norm(axis)
    s = float(np.sqrt(1.0 - c * c))
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


def test_li_shell_matches_brute_force_on_a_synthetic_frame():
    """One Li+ with hand-placed contacts: the EC carbonyl O at 1.95 A (in
    the shell), an EC ETHER O at 2.0 A (NOT a carbonyl site: excluded), the
    EMC carbonyl O at 2.7 A (out), and one PF6- F at 2.10 A (in).
    Independent expectation: shell EC 1, EMC 0, PF6- 1; one Li whose anion
    is unshared: a CIP."""
    from chaord.build.molecules import molecule_census
    from chaord.cv.field import solvation_shell
    dl = load_dialect(R_STACK)
    cuts = dl.threshold("li_shell_cutoffs")
    cut_O, cut_F = float(cuts["C3H4O3"]), float(cuts["F6P"])
    li_at = np.array([17.0, 17.0, 17.0])
    blocks = [([li_at], ["Li"])]
    # EC_a: carbonyl O (template idx 5) at cut_O - 0.5 A along +x
    blocks.append(_place_site("EC", _carbonyl_index("EC"), (1, 0, 0),
                              cut_O - 0.5, li_at))
    # EC_b: ring ETHER O (an O that is not the carbonyl) at 2.0 A along +y
    ether = next(i for i, s in enumerate(TEMPLATES["EC"]["symbols"])
                 if s == "O" and i != _carbonyl_index("EC"))
    blocks.append(_place_site("EC", ether, (0, 1, 0), 2.0, li_at))
    # EMC: carbonyl O at 2.7 A along +z (outside the cutoff)
    blocks.append(_place_site("EMC", _carbonyl_index("EMC"), (0, 0, 1),
                              2.7, li_at))
    # PF6-: one F (template idx 1, +x in the template) at cut_F - 0.35 A
    # along -y (the one free octant: +x, +y, +z hold the three solvents)
    blocks.append(_place_site("PF6-", 1, (0, -1, 0), cut_F - 0.35, li_at))
    frame = _frame_of(blocks, [36.0, 36.0, 36.0])
    # nothing bonded across molecules (the census sees four species + Li)
    assert molecule_census(frame, dl) == {"C3H4O3": 2, "C4H8O3": 1,
                                          "F6P": 1, "Li": 1}
    out = solvation_shell(frame, dl)
    assert out["n_li"] == 1
    assert out["per_species"]["C3H4O3"] == (pytest.approx(1.0),
                                            pytest.approx(0.0))  # O=C O only
    assert out["per_species"]["C4H8O3"][0] == pytest.approx(0.0)   # 2.7 A away
    assert out["per_species"]["F6P"][0] == pytest.approx(1.0)      # the F
    assert out["ion_pairs"] == {"SSIP": 0, "CIP": 1, "AGG": 0}
    assert out["cutoffs"] == {"C3H4O3": cut_O, "C4H8O3": cut_O,
                              "F6P": cut_F}


def test_ion_pair_classes_ssip_cip_agg_on_a_synthetic_frame():
    """Four Li+ and two PF6-: Li1 and Li2 both contact the SAME anion (AGG,
    AGG), Li3 has no F within the cutoff (SSIP), Li4 contacts the second
    anion alone (CIP)."""
    from chaord.build.molecules import molecule_census
    from chaord.cv.field import solvation_shell
    dl = load_dialect(R_STACK)
    cut_F = float(dl.threshold("li_shell_cutoffs")["F6P"])
    d = cut_F - 0.3                       # a clear contact distance
    a1, s_a = _place("PF6-", np.eye(3), [8.0, 8.0, 8.0])
    a2, _ = _place("PF6-", np.eye(3), [22.0, 8.0, 8.0])
    # PF6- template F at +-1.58 along each axis: two Li flanking a1's first F
    li = [a1[1] - np.array([d, 0, 0]), a1[1] + np.array([d, 0, 0]),
          np.array([8.0, 20.0, 8.0]), a2[1] - np.array([d, 0, 0])]
    frame = _frame_of(
        [(np.array([p]), ["Li"]) for p in li] + [(a1, s_a), (a2, s_a)],
        [32.0, 32.0, 32.0])
    assert molecule_census(frame, dl) == {"F6P": 2, "Li": 4}
    out = solvation_shell(frame, dl)
    assert out["n_li"] == 4
    assert out["ion_pairs"] == {"SSIP": 1, "CIP": 1, "AGG": 2}
    assert sum(out["ion_pairs"].values()) == 4


def test_solvation_shell_without_lithium_returns_none():
    """A frame with no cation promises no shell: None, not an empty
    average over zero ions."""
    from chaord.cv.field import solvation_shell
    dl = load_dialect(R_STACK)
    ec, syms = _place("EC", np.eye(3), [15.0, 15.0, 15.0])
    assert solvation_shell(_frame_of([(ec, syms)], [30.0] * 3), dl) is None


# ------------------- Y5 item 3: species radial number-density profiles --

def test_species_radial_profiles_exact_bins():
    """Hand-placed molecule centres at known radii: the per-species profile
    equals the independent shell count over shell volume, centres beyond the
    tube radius are excluded and counted.  (Water: a 1 A-extent species, so
    hand-placed centres 2 A apart never bond -- the profile is species-
    generic, this test pins the bin arithmetic.)"""
    from chaord.cv.field import species_radial_profiles
    dl = load_dialect(R_STACK)
    w = float(dl.threshold("species_nr_bin_width"))
    L = np.array([30.0, 30.0, 20.0])
    R = 4.0                                   # the "tube radius" caller value
    blocks = []
    # three H2O centres at r = 0.5, 1.5, 3.5 about (15, 15); one H2O at 2.5
    # on the opposite side (>= 2.4 A from the r = 1.5 one: no bond); one Li+
    # atom at r = 4.4 (outside)
    for dx, dy in ((0.5, 0.0), (-1.5, 0.0), (-3.5, 0.0), (0.0, 2.5)):
        blocks.append(_place("H2O", np.eye(3), [15.0 + dx, 15.0 + dy, 10.0]))
    blocks.append(([np.array([15.0 + 4.4, 15.0, 7.0])], ["Li"]))
    out = species_radial_profiles(_frame_of(blocks, L), dl, axis="z",
                                  centre=(15.0, 15.0), r_max=R)
    assert set(out) == {"H2O", "Li"}
    n_bins = int(R // w)
    edges = np.linspace(0, n_bins * w, n_bins + 1)
    shell = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2) * L[2]
    h2o = np.zeros(n_bins)
    for r in (0.5, 1.5, 3.5, 2.5):
        h2o[int(r // w)] += 1
    np.testing.assert_allclose(out["H2O"]["profile"], h2o / shell, rtol=1e-12)
    assert out["H2O"]["count"] == 4
    # the Li+ at r = 4.4: its own profile, one bin, excluded from R = 4.0
    assert out["Li"]["n_outside"] == 1
    assert out["Li"]["count"] == 1


# ---------------------------------- Y5 items 2-5 in the composed lift text --

def test_composed_pilot_frame_carries_the_electrolyte_statements():
    """One real pilot frame end to end: the fluid region holds the li_shell,
    ion_pairs, n_r, accessible-density and linear-density statements; the
    molecule-centre cn is gone; the independent recomputations agree."""
    dl = A.composition_dialect()
    frames, _, _ = A.read_prod_frames("cnt10_field0")
    fr = frames[0]
    text = format_program(lift_frame(fr, dl, T=A.T_K, mode="auto"))
    fluid = _fluid_block(text)
    # the site/cutoff words come from the dialect (2.45 A: the g(r) minimum
    # of the zero-field reference runs -- see the dialect's provenance note)
    cuts = dl.threshold("li_shell_cutoffs")
    assert_y5_statements(text, {
        "n_li": 2,
        "species": {"C3H4O3", "C4H8O3", "F6P", "Li+"},
        "shell_species": {
            "C3H4O3": ("carbonyl_O", f"{float(cuts['C3H4O3']):.2f}"),
            "C4H8O3": ("carbonyl_O", f"{float(cuts['C4H8O3']):.2f}"),
            "F6P": ("F", f"{float(cuts['F6P']):.2f}"),
        },
        "tube": (float(_TUBE_RADIUS.search(text).group(1)),
                 float(_TUBE_RADIUS.search(text).group(2))),
        "density": _accessible_density(fr, text, dl),
        "linear": _linear_density(fr, text),
    })
    # item 5: the old molecule-centre cn assert is gone (0.0 +- 0.0 in 55 of
    # 84 frames); the Li+ shell statements are the site-based coordination
    assert re.search(r"^  assert cn ", fluid, re.M) is None


def _tube_indices(fr, dl):
    from chaord.lift import nanotube_components
    return nanotube_components(fr, dl)[0].indices


def _accessible_density(fr, text, dl) -> float:
    """Independent recomputation: fluid mass / (pi (R - r_wall)^2 L)."""
    from ase.data import atomic_masses, chemical_symbols
    m = _TUBE_RADIUS.search(text)
    R, Lz = float(m.group(1)), float(m.group(2))
    r_wall = float(dl.threshold("pore_wall_exclusion"))
    keep = np.ones(len(fr), bool)
    keep[_tube_indices(fr, dl)] = False
    mass = sum(atomic_masses[chemical_symbols.index(s)]
               for s, k in zip(fr.symbols, keep) if k)
    return mass / (np.pi * (R - r_wall) ** 2 * Lz) / _U_PER_A3_TO_G_CM3


def _linear_density(fr, text) -> float:
    """Independent recomputation: fluid molecules per A of tube axis."""
    Lz = float(_TUBE_RADIUS.search(text).group(2))
    return sum(int(m.group(2)) for m in
               map(_MOLECULES.match, _fluid_block(text).splitlines()) if m) / Lz


# --------------------------------------------------- the 84-frame done-when --

@pytest.mark.slow
def test_all_84_pilot_frames_carry_the_electrolyte_statements():
    """Y5's done-when: every committed pilot production frame lifts (no
    ZeroDivisionError anywhere) with the full electrolyte statement set;
    the accessible density matches the independent recomputation and the
    ion-pair classes sum to the Li+ count on every frame."""
    dl = A.composition_dialect()
    cuts = dl.threshold("li_shell_cutoffs")
    shell_species = {
        "C3H4O3": ("carbonyl_O", f"{float(cuts['C3H4O3']):.2f}"),
        "C4H8O3": ("carbonyl_O", f"{float(cuts['C4H8O3']):.2f}"),
        "F6P": ("F", f"{float(cuts['F6P']):.2f}"),
    }
    n_li_of = {"cnt10": 2, "cnt16": 6}
    n_frames = 0
    for system in A.SYSTEMS:
        frames, _, _ = A.read_prod_frames(system)
        for fr in frames:
            text = format_program(lift_frame(fr, dl, T=A.T_K, mode="auto"))
            assert_y5_statements(text, {
                "n_li": n_li_of[system[:5]],
                "species": {"C3H4O3", "C4H8O3", "F6P", "Li+"},
                "shell_species": shell_species,
                "tube": (float(_TUBE_RADIUS.search(text).group(1)),
                         float(_TUBE_RADIUS.search(text).group(2))),
                "density": _accessible_density(fr, text, dl),
                "linear": _linear_density(fr, text),
            })
            n_frames += 1
    assert n_frames == 84, n_frames


# ------------------------------------------- the 20-transform done-when --

def _rigid_transform_with_charges(frame, rng):
    """Rotation + translation + re-imaging + atom re-ordering, carrying the
    per-atom charges (the pilot frames' own charge model; acc.rigid_transform
    drops frame.info, which would change the dipole measurement basis by
    construction, not by the lift)."""
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    t = rng.uniform(-5, 5, 3)
    cell = frame.cell @ R.T
    pos = frame.pos @ R.T + t
    pos = np.mod(pos @ np.linalg.inv(cell), 1.0) @ cell
    perm = rng.permutation(len(pos))
    info = {}
    if frame.info.get("charges") is not None:
        info["charges"] = np.asarray(frame.info["charges"], float)[perm]
    return Frame(pos=pos[perm], cell=cell,
                 symbols=[frame.symbols[i] for i in perm],
                 pbc=frame.pbc, info=info)


@pytest.mark.slow
def test_20_rigid_transforms_of_one_frame_give_identical_text():
    """Y5's done-when: 20 seeded random rigid transforms (rotation +
    translation + re-imaging + re-ordering, charges carried through the
    permutation) of one pilot frame lift to byte-identical text."""
    dl = A.composition_dialect()
    frames, _, _ = A.read_prod_frames("cnt10_field1")
    fr = frames[7]
    ref = format_program(lift_frame(fr, dl, T=A.T_K, mode="auto"))
    rng = np.random.default_rng(20261015)
    for k in range(20):
        g = _rigid_transform_with_charges(fr, rng)
        text = format_program(lift_frame(g, dl, T=A.T_K, mode="auto"))
        assert text == ref, (
            f"rigid transform {k} lifts to different text "
            f"(rotation\n{np.round(rng.normal(size=4), 3)})")


# ------------------------- cutoff provenance: the reference-run derivation --

def _equil_frames(system: str):
    """The zero-field EQUILIBRIUM frames of one pilot system (17 frames;
    NOT part of the 84 production frames under test), read through the Y4
    reader with the build data file's type map."""
    import json
    base = A.RUNS / system
    provenance = json.loads((base / f"provenance_{system}.json").read_text())
    _, data_meta = A.read_lammps_data(
        A.HERE / provenance["outputs"]["data_file"])
    return A.read_lammps_dump(base / "equil.dump",
                              type_map=data_meta["type_map"])


def _solvation_gr_first_minimum() -> dict:
    """The estimator of the dialect's Li-solvation cutoffs, recomputed from
    the zero-field equilibration trajectories of both pilot tubes (the
    reference runs available today; Stage A's zero-field runs replace them,
    D36/Y6):

    Li-O(carbonyl of EC+EMC) and Li-F g(r), 0.1 A bins, every frame
    normalised by its own site count in the geometric pore volume
    pi R^2 Lz (R from the run's provenance record), averaged over the 34
    frames; the cutoff is the FIRST LOCAL MINIMUM after the first peak.
    """
    from chaord.build.molecules import bond_graph
    from collections import defaultdict

    dl = load_dialect(R_STACK)
    BIN = 0.1                       # test-side derivation constant
    edges = np.arange(1.5, 4.0 + BIN / 2, BIN)
    rm = 0.5 * (edges[1:] + edges[:-1])
    gs = {"LiO": [], "LiF": []}
    for system in ("cnt10_field0", "cnt16_field0"):
        import json
        prov = json.loads((A.RUNS / system / f"provenance_{system}.json")
                          .read_text())
        R = float(prov["sanity"]["cnt_radius_A"])
        Lz = float(prov["system_definition"]["box_A"][5])
        V = np.pi * R ** 2 * Lz
        for fr in _equil_frames(system):
            nb = defaultdict(list)
            for i, j in bond_graph(fr, dl):
                nb[i].append(j)
                nb[j].append(i)
            carbonyl = []
            for i, s in enumerate(fr.symbols):
                if s != "O":
                    continue
                heavy = [j for j in nb[i] if fr.symbols[j] != "H"]
                if len(heavy) != 1 or fr.symbols[heavy[0]] != "C":
                    continue
                if sum(1 for j in nb[heavy[0]] if fr.symbols[j] == "O") >= 3:
                    carbonyl.append(i)
            li = [k for k, s in enumerate(fr.symbols) if s == "Li"]
            f_idx = [k for k, s in enumerate(fr.symbols) if s == "F"]
            L3 = fr.cell_diag
            d = {"LiO": [], "LiF": []}
            for i in li:
                for key, idx in (("LiO", carbonyl), ("LiF", f_idx)):
                    v = fr.pos[idx] - fr.pos[i]
                    v -= L3 * np.round(v / L3)
                    d[key].extend(np.linalg.norm(v, axis=1).tolist())
            shell = 4 / 3 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
            for key, n_sites in (("LiO", len(carbonyl)), ("LiF", len(f_idx))):
                h, _ = np.histogram(d[key], edges)
                rho = n_sites / V
                gs[key].append(h / (len(li) * rho * shell))
    out = {}
    for key in ("LiO", "LiF"):
        g = np.mean(gs[key], axis=0)
        peak = int(np.argmax(g))
        k = peak + 1
        while k + 1 < len(g) and g[k + 1] <= g[k]:
            k += 1
        out[key] = {"peak_r": float(rm[peak]), "peak_g": float(g[peak]),
                    "min_r": float(rm[k]), "min_g": float(g[k])}
    return out


@pytest.mark.slow
def test_solvation_cutoffs_match_the_zero_field_reference_runs():
    """D36: the dialect's Li-shell cutoffs are the first g(r) minima of the
    zero-field reference trajectories (today the pilot's equilibration
    dumps, marked provisional-pending-Stage-A in the dialect; never the 84
    frames under test).  A cutoff that drifted from its derivation fails
    here."""
    dl = load_dialect(R_STACK)
    cuts = dl.threshold("li_shell_cutoffs")
    derived = _solvation_gr_first_minimum()
    assert derived["LiO"]["min_r"] == pytest.approx(
        float(cuts["C3H4O3"]), abs=1e-9), derived["LiO"]
    assert derived["LiO"]["min_r"] == pytest.approx(
        float(cuts["C4H8O3"]), abs=1e-9), derived["LiO"]
    assert derived["LiF"]["min_r"] == pytest.approx(
        float(cuts["F6P"]), abs=1e-9), derived["LiF"]
    # the peaks sit at the contact distances of the literature's Li+ shell
    assert 1.8 <= derived["LiO"]["peak_r"] <= 2.1, derived["LiO"]
    assert 1.8 <= derived["LiF"]["peak_r"] <= 2.1, derived["LiF"]


# ------------------------------------------------------ the power mutations --

def test_power_mutation_wrong_cutoff_fails_the_statement_check():
    """The single most likely real error: a cutoff that no longer matches
    its reference-run derivation (or was measured on the frames under
    test).  Mutating the dialect's Li-O cutoff to 5.0 A must break the
    statement check (the printed cutoff word changes)."""
    dl = A.composition_dialect()
    frames, _, _ = A.read_prod_frames("cnt10_field0")
    fr = frames[0]
    text = format_program(lift_frame(fr, dl, T=A.T_K, mode="auto"))
    cut = float(dl.threshold("li_shell_cutoffs")["C3H4O3"])
    mutated = A.Dialect(
        dl.names,
        {**dl._data, "thresholds": {**dl._data["thresholds"],
                                    "li_shell_cutoffs": {
                                        "C3H4O3": 5.0, "C4H8O3": cut,
                                        "F6P": cut}}},
        dl.versions)
    bad = format_program(lift_frame(fr, mutated, T=A.T_K, mode="auto"))
    assert bad != text, "a mutated cutoff changed nothing"
    with pytest.raises(AssertionError):
        assert_y5_statements(bad, {
            "n_li": 2, "species": {"C3H4O3", "C4H8O3", "F6P", "Li+"},
            "shell_species": {
                "C3H4O3": ("carbonyl", f"{cut:.2f}"),
                "C4H8O3": ("carbonyl", f"{cut:.2f}"),
                "F6P": ("F", f"{cut:.2f}")},
            "tube": (float(_TUBE_RADIUS.search(bad).group(1)),
                     float(_TUBE_RADIUS.search(bad).group(2))),
            "density": _accessible_density(fr, text, dl),
            "linear": _linear_density(fr, text)})


def test_power_mutation_dropped_statement_fails_the_statement_check():
    """Dropping the electrolyte statements entirely (the regression this
    round guards against) must fail the checker."""
    dl = A.composition_dialect()
    frames, _, _ = A.read_prod_frames("cnt10_field0")
    fr = frames[0]
    text = format_program(lift_frame(fr, dl, T=A.T_K, mode="auto"))
    stripped = re.sub(r"^  assert li_shell .*\n|^  assert ion_pairs .*\n",
                      "", text, flags=re.M)
    assert "assert li_shell" not in stripped and "assert ion_pairs" not in stripped
    with pytest.raises(AssertionError):
        assert_y5_statements(stripped, {
            "n_li": 2, "species": {"C3H4O3", "C4H8O3", "F6P", "Li+"},
            "shell_species": {"C3H4O3": ("carbonyl_O", "2.45"),
                              "C4H8O3": ("carbonyl_O", "2.45"),
                              "F6P": ("F", "2.45")},
            "tube": (float(_TUBE_RADIUS.search(text).group(1)),
                     float(_TUBE_RADIUS.search(text).group(2))),
            "density": _accessible_density(fr, text, dl),
            "linear": _linear_density(fr, text)})
