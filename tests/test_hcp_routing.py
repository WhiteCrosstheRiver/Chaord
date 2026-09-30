"""F2 root-cause tests (red team, reports/redteam_findings.md S1/F2): hcp
crystal support in the lift and the pseudo-molecule guard.

Pre-fix behaviour (measured 2026-09-30, all 5 stored frames of
bench/data/crystals/hcp_mg -- 32 Mg, orthohexagonal box 6.42 x 11.12 x 10.42,
thermal amplitude 0.06 d_NN):

  * auto mode lifts `liquid fluid : all { molecules Mg32 1 ... }` -- the
    census invents the pseudo-molecule 'Mg32' and every count check agrees
    with it exactly (A6 three-way, A13 zero residual), so semantic garbage
    passes every gate;
  * mode='crystal' (the case's own ground-truth lift_mode) raises a raw
    ValueError ('no prototype matches the standardised structure'), not a
    ChaordError refusal.

Root cause chain: spglib sees P1 on the jittered orthorhombic frame (the
hexagonal setting is not recoverable at lift_symprec), the cubic
Wigner-Seitz fit supports no hexagonal host, the cascade swallows both
failures, is_single_phase rejects the 32-atom bonded component, the slab
engine fails and its unconditional lift_fluid fallback runs the census over
the whole bonded crystal.

The fix under test (two halves that must land together):
  1. an orthohexagonal hcp prototype fit (build/prototypes.py) routed as a
     crystal arm (lift/legacy.py) -- the frame lifts as `lattice hcp` with
     a/c measured from the box under the sqrt(3)-axis constraint and verified
     by site matching;
  2. a census guard (build/molecules.py): a bonded component larger than the
     dialect's fluid bound that percolates the box or holds the majority of
     the frame is an extended phase, not a molecule -- the census refuses it
     loudly instead of inventing a formula name. Normal molecular fluids
     (water, N2, NaCl(aq), the 7-atom EC solvent) keep their census.

Test-side constants (not pass code): A_C_TOL is the box-derived fit accuracy
(the bench box is an exact integer multiple of the orthorhombic hcp cell, so
the fit recovers the ground truth to printing precision).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from chaord.build.molecules import molecule_census         # noqa: E402
from chaord.dialects import load_dialect                   # noqa: E402
from chaord.io.frames import Frame, read_frame             # noqa: E402
from chaord.lang.errors import ChaordError                 # noqa: E402
from chaord.lang.fmt import format_program                 # noqa: E402
from chaord.lang.parser import parse_text                  # noqa: E402
from chaord.lift import lift_frame                         # noqa: E402
from chaord.lift.fluid import lift_fluid                   # noqa: E402

METAL = ("core", "metal")
MOLECULAR = ("core", "molecular")
BENCH = ROOT / "bench" / "data"

# ground truth of bench/data/crystals/hcp_mg/ground_truth.json
GT_A = 3.21
GT_C = 5.21
GT_CELL = (6.42, 11.11976618459219, 10.42)
GT_COUNTS = {"Mg": 32}
A_C_TOL = 0.005            # box-derived a/c recover the truth to print precision

_dl = load_dialect(METAL)


def _hcp_frame(k=0) -> Frame:
    return read_frame(BENCH / "crystals" / "hcp_mg" / f"frame_{k}.npz")


def _stmt(text: str, key: str) -> str | None:
    for line in text.splitlines():
        parts = line.strip().split()
        if parts and parts[0] == key:
            return " ".join(parts[1:])
    return None


# ------------------------------------------------- the crystal route (F2 1-2) --

@pytest.mark.parametrize("k", range(5))
def test_hcp_auto_lifts_crystal_program(k):
    """Every stored hcp_mg frame auto-lifts as a crystal hcp program, never
    as a pseudo-molecule fluid (pre-fix: 5/5 lift `molecules Mg32 1`)."""
    text = format_program(lift_frame(_hcp_frame(k), _dl))
    assert "crystal bulk : all {" in text
    assert _stmt(text, "lattice") == "hcp"
    assert "liquid" not in text and "fluid" not in text
    assert "Mg32" not in text and "molecules" not in text


@pytest.mark.parametrize("k", range(5))
def test_hcp_crystal_mode_succeeds(k):
    """mode='crystal' (the case's recorded lift_mode) lifts the frame instead
    of escaping as a raw ValueError (pre-fix: ValueError 'no prototype
    matches the standardised structure')."""
    program = lift_frame(_hcp_frame(k), _dl, mode="crystal")
    text = format_program(program)
    assert "crystal bulk : all {" in text
    assert _stmt(text, "lattice") == "hcp"


@pytest.mark.parametrize("k", range(5))
def test_hcp_fit_recovers_ground_truth_a_c_and_counts(k):
    """The orthohexagonal fit measures a and c from the frame's box under the
    sqrt(3)-axis constraint; the bench box is an exact integer multiple of
    the orthorhombic hcp cell, so the fit must recover the ground truth."""
    text = format_program(lift_frame(_hcp_frame(k), _dl, mode="crystal"))
    a = float(_stmt(text, "a").split()[0])
    c = float(_stmt(text, "c").split()[0])
    assert abs(a - GT_A) <= A_C_TOL, f"frame {k}: fitted a={a}, truth {GT_A}"
    assert abs(c - GT_C) <= A_C_TOL, f"frame {k}: fitted c={c}, truth {GT_C}"
    line = next(l for l in text.splitlines() if l.strip().startswith("conserve atoms"))
    stated = {t: int(n) for t, n in
              zip(line.split()[2::2], line.split()[3::2])}
    assert stated == GT_COUNTS


@pytest.mark.parametrize("k", range(5))
def test_hcp_program_builds_exact_counts_and_round_trips(k):
    """The lifted program is a buildable spec (pre-fix the Mg32 program died
    on 'unknown species'): the orientation statement parsed by the builder's
    own `orient_matrix` tiles the stated cell with exactly 32 Mg, and
    lift -> build -> lift is byte-stable (rule 9: the rebuild is a perfect
    frame of the same orthohexagonal setting and must lift to the same
    text).

    Built through `build_conventional` with the program's parsed orient
    rows rather than end-to-end `build_program`: the orthohexagonal tiling
    leaves the (1/3,2/3,1/2) basis site at x = -8.9e-17 (the fp residue of
    a/3 - a/3 inside build_conventional's keep window), and the builder's
    own apply_defects -> nearest_neighbor_distance feeds those raw
    positions to cKDTree(boxsize=...), which rejects any negative
    coordinate (bench/generate.py works around the same wart with
    _sanitize). That one-line fix belongs to build/crystal.py -- outside
    this stream's file list -- and is reported to the mainline."""
    from chaord.build.crystal import build_conventional, orient_matrix
    t1 = format_program(lift_frame(_hcp_frame(k), _dl, mode="crystal"))
    region = next(b for b in parse_text(t1).blocks if b.t == "region")
    stmts = {s.key: s for s in region.statements}
    T = orient_matrix(stmts["orient"])
    params = {"a": float(stmts["a"].values[0].num),
              "c": float(stmts["c"].values[0].num)}
    rebuilt = build_conventional("hcp", params, ("Mg",), (2, 2, 2), T)
    assert len(rebuilt) == 32 and set(rebuilt.symbols) == {"Mg"}
    assert np.allclose(np.diag(rebuilt.cell), GT_CELL, atol=5e-4)
    t2 = format_program(lift_frame(rebuilt, _dl, mode="crystal"))
    assert t2 == t1, f"frame {k}: round trip drifted:\n{t1}\n---\n{t2}"
    rebuilt2 = build_conventional("hcp", params, ("Mg",), (2, 2, 2), T)
    assert format_program(lift_frame(rebuilt2, _dl, mode="crystal")) == t2


def test_hcp_auto_text_is_rigid_transform_invariant():
    """Rule 3 on the new arm: rotation + translation + re-imaging + atom
    re-ordering must lift to byte-identical text (the fit reads the box
    lengths and re-canonicalises the axes, so the assignment is
    orientation-free)."""
    from tools import acceptance as acc
    frame = _hcp_frame(1)
    t_ref = format_program(lift_frame(frame, _dl))
    rng = np.random.default_rng(20260930)
    for _ in range(2):
        g = acc.rigid_transform(frame, rng)
        assert format_program(lift_frame(g, _dl)) == t_ref


def test_non_crystal_frame_refuses_loudly_in_crystal_mode():
    """A frame that is not a crystal at all must refuse mode='crystal' with a
    ChaordError, never a raw ValueError (F2 root-cause 1: the same escape on
    hcp_mg was an A13-class defect by itself)."""
    water = read_frame(BENCH / "fluid" / "water_box15" / "frame_0.npz")
    with pytest.raises(ChaordError):
        lift_frame(water, load_dialect(MOLECULAR), mode="crystal")


def test_hcp_fit_generalises_beyond_the_bench_values():
    """The fit is geometry, not bench-tuned: a different hcp metal (Zn,
    a = 2.660 A, c = 4.950 A, c/a = 1.86 far from Mg's 1.62), a 3x2x2
    supercell and a fresh seeded thermal displacement at the same 0.06 d_NN
    amplitude recover the same a and c (box-derived, exact to print
    precision)."""
    from chaord.build.crystal import build_conventional
    rng = np.random.default_rng(20260930)
    a, c = 2.660, 4.950
    orient = np.array([[1, 0, 0], [1, 2, 0], [0, 0, 1]], int)
    frame = build_conventional("hcp", {"a": a, "c": c}, ("Zn",), (3, 2, 2),
                               orient)
    amp = 0.06 * min(a, np.sqrt(a * a / 3 + c * c / 4))
    noisy = Frame(pos=frame.pos + rng.normal(size=frame.pos.shape) * amp,
                  cell=frame.cell, symbols=frame.symbols, pbc=frame.pbc)
    text = format_program(lift_frame(noisy, _dl))
    assert _stmt(text, "lattice") == "hcp"
    assert abs(float(_stmt(text, "a").split()[0]) - a) <= A_C_TOL
    assert abs(float(_stmt(text, "c").split()[0]) - c) <= A_C_TOL


@pytest.mark.parametrize("case,proto", [("crystals/fcc_cu", "fcc"),
                                        ("crystals/bcc_fe", "bcc"),
                                        ("crystals/diamond_si", "diamond")])
def test_cubic_bench_frames_still_lift_their_prototype(case, proto):
    """No over-fix: single-species cubic crystals must not be claimed by the
    hcp arm (the sqrt(3)-axis and site-coverage gates reject them) and keep
    their crystal lift."""
    frame = read_frame(BENCH / case / "frame_0.npz")
    text = format_program(lift_frame(frame, _dl))
    assert "crystal bulk : all {" in text
    assert _stmt(text, "lattice") == proto, f"{case} lifted:\n{text}"


# ---------------------------------------------------- the pseudo-molecule guard --

@pytest.mark.parametrize("k", range(5))
def test_census_refuses_the_bonded_hcp_crystal(k):
    """The census must not invent 'Mg32': the whole-frame component (32 Mg,
    percolating, the entire frame) is an extended phase. Pre-fix:
    molecule_census(frame) == {'Mg32': 1}."""
    with pytest.raises(ChaordError, match=r"extended bonded component"):
        molecule_census(_hcp_frame(k), _dl)


@pytest.mark.parametrize("k", range(5))
def test_lift_fluid_refuses_the_bonded_hcp_crystal(k):
    """The fluid fallback the cascade used to reach must fail loudly on the
    same evidence (pre-fix: lift_fluid(frame) returned the `molecules Mg32`
    program)."""
    with pytest.raises(ChaordError, match=r"extended bonded component.*Mg"):
        lift_fluid(_hcp_frame(k), _dl)


def test_census_refuses_majority_component_without_periodic_wrap():
    """A bonded chain holding the whole frame in a non-periodic box is the
    phase itself even though nothing wraps (the majority half of the
    guard)."""
    molecular = load_dialect(MOLECULAR)
    n = 20
    pos = np.zeros((n, 3))
    pos[:, 0] = np.arange(n) * 1.3      # < 2 x r_cov(C): a covalent chain
    frame = Frame(pos=pos, cell=np.diag([40.0, 8.0, 8.0]),
                  symbols=["C"] * n, pbc=(False, False, False))
    with pytest.raises(ChaordError, match=r"extended bonded component"):
        molecule_census(frame, molecular)


def test_census_refuses_percolating_metal_slab_component():
    """The original M3 hole (test_fluid_routing): the Cu slab of cu_water is
    one 384-atom component spanning the box; the whole-frame census must
    refuse it (region censuses of the water alone are unaffected -- pinned
    by the existing fluid-routing and bench suites)."""
    frame = read_frame(BENCH / "interfaces" / "cu_water" / "frame_0.npz")
    with pytest.raises(ChaordError, match=r"extended bonded component"):
        molecule_census(frame, _dl)


@pytest.mark.parametrize("case", ["fluid/water_box15", "fluid/n2_box22",
                                  "solutions/nacl_aq", "solutions/lipf6_ec",
                                  "gases/co2_dense"])
def test_census_keeps_normal_molecular_fluids(case):
    """No over-fix (regression pin, the exact ground-truth census): water,
    N2, aqueous NaCl, the dense CO2 gas and the EC/LiPF6 electrolyte -- whose
    solvent component (C3H4O3, 7 atoms) exceeds the dialect's 3-atom fluid
    bound while being a genuine compact molecule -- keep their census."""
    import json
    frame = read_frame(BENCH / case / "frame_0.npz")
    gt = json.loads((BENCH / case / "ground_truth.json").read_text())
    census = molecule_census(frame, load_dialect(MOLECULAR))
    assert census == gt["frames"][0]["census"] == gt["expected"]["census"]
