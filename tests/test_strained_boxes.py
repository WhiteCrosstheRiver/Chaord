"""W5 (docs/reviews/open_items_v2.md): strained boxes must build.

NPT runs that let the box lengths fluctuate independently (LAMMPS `aniso` or
`tri`) give boxes slightly off an exact tiling. Scaling a bench crystal frame
and its box by the same factors is a pure homogeneous deformation: the lift
text stays translation-invariant (measured: identical under rigid shifts),
but pre-fix the builder refused its own program -- "system cell ... is not an
integer multiple of the ... lattice vectors" -- because the absolute box
mismatch (strain x box length) exceeded `lattice_match_tolerance` (0.01 A)
while every axis still read the same integer repetition count of one lattice.

Pre-fix state (reproduced 2026-10-02, before the builder change):

  * +0.2 % along x or y: 6 of 9 cases refuse to build (rocksalt_nacl,
    l12_ni3al, hcp_mg, diamond_si, fcc_crconi, cuau_random); along z 5 of 9
    (hcp_mg still builds: its fitted a absorbs the x/y mismatch and only the
    sqrt(3)-axis tiling drifts out).
  * +-0.5 % and +-1.5 % along x or y: 9 of 9 refuse; along z 8 of 9.
  * isotropic scaling never refuses the build (0 of 9 at every magnitude):
    a uniform scale multiplies the fitted lattice constant and the box alike,
    so the box stays an exact tiling.

The fix (builder only; the lift is unchanged): when a crystal region's stated
cell is within `strain_cell_tolerance` (2 %, core.yaml) of an integer tiling
on EVERY axis but outside `lattice_match_tolerance`, tile with the nearest
integer counts and scale each axis to the stated cell, recording
"strained to cell: e_xx ... e_yy ... e_zz ..." in the build provenance
(frame.info, the amorphous builder's assumed_history channel). Above the
fraction the builder still refuses (A12).

Known open item (measured with the fix, 2026-10-02, this matrix): full
lift -> build -> lift byte identity held for 122 of the 216 matrix configs
(all of +-0.2 % x, and fcc_cu / bcc_fe / rocksalt_nacl everywhere). The
other 94 differed on the single `a` (hcp: `c`) line by 0.001-0.012 A: the
first lift of a jittered strained frame runs the defect arm, whose SCAN-fitted
lattice constant drifts below the per-axis box mean as strain grows, while
the re-lift of the exact rebuild runs the spglib arm, whose idealised (mean)
constant is a function of the stated box alone. The built frame is fully
determined by the program text (box, reps, species), so no builder change
can close that gap; the lift-side remedy needed the reviewer's sign-off
because W5 says "the lift is unchanged".

CLOSED (X2, 2026-10-14; D17 in docs/reviews/open_items_v3.md approved the
lift-side snap): one function, `box_lattice_constant(cell, reps, prototype)`
in build/prototypes.py, returns the constant(s) the STATED cell implies
(cubic: the mean of L_i/n_i; hcp: a from x and y, c from z -- the ortho-
hexagonal axis order every chaord hcp program states) whenever every axis
is within `strain_cell_tolerance` of that integer tiling, None otherwise.
Every lift arm snaps its fitted constant to it (spglib arm: lift/crystal.py;
defect scan: lift/defects.py, sites regenerated at the snapped constant;
hcp: lift/legacy.py), and the builder's strained-cell acceptance
(build/crystal.py) is the same function -- one definition on both sides,
rule 7. Two details make the round trip byte-stable: the snap reads the
STATED (canonical-rounded) cell, not the raw frame box -- the re-lift
reads the stated cell back, and the hcp gap configs differed on the `c`
line by exactly one rounding boundary of the raw vs stated box -- and the
superseded absolute-gate `snap_a_to_cell` is gone. With X2 all 216 matrix
configs are text-identical and the 9-case x 20-draw random anisotropic
sweep below (each axis uniform in +-1.5 %) holds as well; the _GAP set and
the FOILED registry are empty of strained-box ids.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from chaord.build import build_program                     # noqa: E402
from chaord.build.crystal import orient_matrix             # noqa: E402
from chaord.build.prototypes import cell_matrix            # noqa: E402
from chaord.dialects import load_dialect                   # noqa: E402
from chaord.io.frames import Frame                         # noqa: E402
from chaord.lang.errors import ChaordError                 # noqa: E402
from chaord.lang.fmt import format_program                 # noqa: E402
from chaord.lang.parser import parse_text                  # noqa: E402
from chaord.lift import lift_frame                         # noqa: E402
from tools import acceptance as acc                        # noqa: E402

# the W5 reproducer's case set: every crystal case plus the random solid
# solution, wherever the bench files it
CRYSTAL_CASES = [c["id"] for c in acc._exact_roundtrip_cases(None)]

# W5 done-when magnitudes on each single axis and isotropically
STRAIN_MAGS = (0.002, 0.005, 0.015)
AXES = {"x": (0,), "y": (1,), "z": (2,), "iso": (0, 1, 2)}
REFUSE_FRACTION = 0.03          # above the 2 % strain gate: the builder refuses

_dl_cache: dict = {}


def _dialect(case):
    if case["dialect"] not in _dl_cache:
        _dl_cache[case["dialect"]] = load_dialect(case["dialect"])
    return _dl_cache[case["dialect"]]


def _strained_frame(case, fraction, axes):
    """The case's stored thermal frame under a homogeneous box deformation:
    positions and cell scaled by the same per-axis factors (wrapped into the
    new box), exactly the W5 reproducer's construction."""
    f = acc._a2a3_frame(case)
    s = np.array([1.0, 1.0, 1.0])
    for i in axes:
        s[i] = 1.0 + fraction
    return Frame(pos=np.mod(f.pos * s, f.cell_diag * s),
                 cell=np.diag(f.cell_diag * s),
                 symbols=f.symbols, pbc=f.pbc)


def _round_trip(frame, dialect):
    """lift -> build -> lift: (first text, second text, built frame)."""
    t1 = format_program(lift_frame(frame, dialect))
    built = build_program(parse_text(t1), dialect,
                          rng=np.random.default_rng(1), physics=False)
    t2 = format_program(lift_frame(built, dialect))
    return t1, t2, built


def _cell_of(text):
    for line in text.splitlines():
        parts = line.strip().split()
        if parts and parts[0] == "cell":
            return np.array([float(v) for v in parts[1:4]])
    raise AssertionError(f"no cell statement in:\n{text}")


def _tiled_lengths(text, stated):
    """reps x conventional-axis lengths of the program's own lattice, in the
    stated axis order (from the orient statement; hcp uses the orthohexagonal
    setting x [100] y [1 2 0] z [001])."""
    import re

    from chaord.lang.ir import Name

    lines = [l.strip().split() for l in text.splitlines()]
    name = next(p[1] for p in lines if p and p[0] in ("lattice", "prototype"))
    params = {}
    for p in lines:
        if p and p[0] in ("a", "c"):
            params[p[0]] = float(p[1])
    T = np.eye(3, dtype=int)
    orient = next((l for l in text.splitlines()
                   if l.strip().startswith("orient")), None)
    if orient is not None:
        # keep bracketed directions like [1 2 0] as single tokens
        toks = re.findall(r"\S*\[[^\]]*\]|\S+", orient.strip())[1:]

        class _S:              # an orient-statement stand-in for orient_matrix
            values = [Name(text=tok) for tok in toks]

        T = orient_matrix(_S())
    conv = T @ cell_matrix(name, params)
    lengths = np.linalg.norm(conv, axis=1)
    reps = np.rint(stated / lengths).astype(int)
    return reps * lengths


def _assert_text_identical(cid, label, t1, t2):
    assert t2 == t1, (
        f"{cid} {label}: lift -> build -> lift text differs:\n"
        f"--- first lift ---\n{t1}\n--- after round trip ---\n{t2}")


# ---- the W5 reproducer: +0.2 % along x, all 9 cases ==========================

@pytest.mark.parametrize("cid", CRYSTAL_CASES)
def test_strained_crystal_builds_and_round_trips_at_reproducer_strain(cid):
    """The W5 reproducer config: +0.2 % along x. Pre-fix 6 of 9 cases raise
    'system cell ... is not an integer multiple ...'; post-fix every case
    builds and lift -> build -> lift is byte-identical."""
    case = acc.case_by_id(cid)
    t1, t2, _built = _round_trip(_strained_frame(case, 0.002, (0,)),
                                 _dialect(case))
    _assert_text_identical(cid, "+0.2% x", t1, t2)


# ---- X2: the snap function's own contract (D17) ===============================

def test_box_lattice_constant_cubic_is_the_per_axis_mean():
    """D17, cubic: the constant a stated cell implies is the mean of
    L_i / n_i; a cell whose implied per-axis constants disagree with that
    mean by more than strain_cell_tolerance implies no constant (None).
    (The gate is relative TO THE MEAN: a single axis at +2.5 % is only
    ~1.65 % off the three-axis mean, so it still snaps -- the mean is what
    every arm prints; one axis must pass ~3.1 % for the mean itself to
    leave the gate. The builder keeps its own per-axis check against the
    PROGRAM's constant, which is what refuses the A12 3 % case.)"""
    from chaord.build.prototypes import box_lattice_constant
    dialect = load_dialect(("core", "metal"))
    # 2x2x2 tiling of a 3.615 A fcc cell, +0.8 % along x: implied constants
    # 3.6440 / 3.6150 / 3.6150, mean 3.6247 (all within the 2 % gate)
    cell = np.array([2 * 3.615 * (1 + 0.008), 2 * 3.615, 2 * 3.615])
    got = box_lattice_constant(cell, (2, 2, 2), "fcc", dialect)
    assert got is not None and list(got) == ["a"]
    assert abs(got["a"] - cell.mean() / 2) < 1e-12
    # unequal reps are honoured per axis: 2 x 3 x 1 of a bcc cell
    a = 2.866
    cell2 = np.array([2 * a, 3 * a, a])
    got2 = box_lattice_constant(cell2, (2, 3, 1), "bcc", dialect)
    assert got2 is not None and abs(got2["a"] - a) < 1e-12
    # one axis far enough off that even the mean leaves the gate: no
    # constant, the caller keeps its fit and the builder's refusal stands
    assert box_lattice_constant(
        np.array([2 * a * 1.035, 2 * a, 2 * a]), (2, 2, 2),
        "bcc", dialect) is None
    # a rep below 1 is not a tiling of the cell
    assert box_lattice_constant(np.array([a, 2 * a, 2 * a]), (0, 2, 2),
                                "bcc", dialect) is None


def test_box_lattice_constant_hcp_takes_a_from_x_and_y_c_from_z():
    """D17, hcp: in the orthohexagonal setting (x [100] y [1 2 0] z [001])
    the a-pair is x/n_x and y/(sqrt(3) n_y) -- their mean is a -- and
    z/n_z is c; the gate is per axis against those (c is its own base)."""
    from chaord.build.prototypes import box_lattice_constant
    dialect = load_dialect(("core", "metal"))
    a, c = 3.210, 5.210
    # the bench hcp_mg box: n_a = 2, n_q = 2 (sqrt(3)-axis), n_c = 2
    sqrt3 = float(np.sqrt(3.0))
    cell = np.array([2 * a, 2 * sqrt3 * a, 2 * c])
    got = box_lattice_constant(cell, (2, 2, 2), "hcp", dialect)
    assert got is not None and set(got) == {"a", "c"}
    assert abs(got["a"] - a) < 1e-12 and abs(got["c"] - c) < 1e-12
    # x and y strains average into a (0.6 % and -0.4 % -> +0.1 %): still one a
    strained = cell * np.array([1.006, 1.0 - 0.004, 1.0])
    got2 = box_lattice_constant(strained, (2, 2, 2), "hcp", dialect)
    assert got2 is not None
    assert abs(got2["a"] - a * 1.001) < 1e-9 and abs(got2["c"] - c) < 1e-12
    # an x/y disagreement that leaves the a-mean gate implies no single a
    bad = cell * np.array([1.045, 1.0, 1.0])
    assert box_lattice_constant(bad, (2, 2, 2), "hcp", dialect) is None
    # a z strain cannot leave the snap's own gate: c IS L_z/n_z (the tautology
    # of a free second parameter -- any z is an exact tiling of the c it
    # implies). The z bound is the builder's program-relative check, which
    # refuses a stated c contradicted by the cell beyond the strain
    # tolerance (test_cell_beyond_strain_tolerance_is_refused); the snap
    # just reports the implied (strained) c.
    strained_c = cell * np.array([1.0, 1.0, 1.025])
    got_c = box_lattice_constant(strained_c, (2, 2, 2), "hcp", dialect)
    assert got_c is not None and abs(got_c["c"] - c * 1.025) < 1e-9
    assert abs(got_c["a"] - a) < 1e-12


def test_stated_cell_rounding_makes_constants_a_round_trip_fixed_point():
    """The X2 subtlety the hcp gap configs exposed: constants derived from
    the RAW box are not a fixed point of lift -> build -> lift when the raw
    and canonical-rounded boxes sit on opposite sides of a rounding
    boundary (bench hcp_mg at +0.2 % iso: c = 5.22042 raw vs 5.2205 stated,
    printing 5.220 vs 5.221). stated_cell_lengths returns the lengths as
    stated, and the snap reads those."""
    from chaord.build.prototypes import box_lattice_constant
    from chaord.lift.crystal import stated_cell_lengths
    dialect = load_dialect(("core", "metal"))
    sqrt3 = float(np.sqrt(3.0))
    raw = np.array([2 * 3.210 * 1.002, 2 * sqrt3 * 3.210 * 1.002,
                    2 * 5.210 * 1.002])
    stated = stated_cell_lengths(raw, dialect)
    assert not np.allclose(raw, stated), "raw box must differ from stated here"
    # the second lift sees the stated cell verbatim: snapping the stated
    # cell is idempotent (snapping its own output returns the same lengths)
    assert np.allclose(stated_cell_lengths(stated, dialect), stated)
    got = box_lattice_constant(stated, (2, 2, 2), "hcp", dialect)
    assert got is not None
    # and the raw-box constant differs from the stated-box one: the print
    # boundary the gap configs failed on
    got_raw = box_lattice_constant(raw, (2, 2, 2), "hcp", dialect)
    assert f"{got_raw['c']:.3f}" != f"{got['c']:.3f}"


# ---- the full W5 done-when matrix ============================================

def _matrix_configs():
    return [(sign * mag, axname, axes)
            for mag in STRAIN_MAGS for sign in (+1, -1)
            for axname, axes in AXES.items()]


_MATRIX_IDS = [f"{f:+.3f}|{ax}" for f, ax, _ in _matrix_configs()]


@pytest.mark.parametrize("fraction,axname,axes", _matrix_configs(), ids=_MATRIX_IDS)
@pytest.mark.parametrize("cid", CRYSTAL_CASES)
def test_strain_matrix_crystal_builds(cid, fraction, axname, axes):
    """W5 done-when, build half: +-0.2 %, +-0.5 % and +-1.5 % along x, y and
    z, and isotropic, build for all 9 cases, the built box IS the stated cell
    on every axis, and the strain provenance fires exactly when the stated
    cell leaves the integer-multiple tolerance. Pre-fix failures: 6/9 at
    +-0.2 % (x, y), 5/9 (z), 9/9 at +-0.5 % and +-1.5 % (x, y), 8/9 (z);
    isotropic always built."""
    case = acc.case_by_id(cid)
    dialect = _dialect(case)
    program = lift_frame(_strained_frame(case, fraction, axes), dialect)
    text = format_program(program)
    built = build_program(program, dialect,
                          rng=np.random.default_rng(1), physics=False)
    stated = _cell_of(text)
    assert np.allclose(built.cell_diag, stated, atol=5e-4), (
        f"{cid} {fraction:+.1%} {axname}: built box {built.cell_diag.tolist()} "
        f"!= stated cell {stated.tolist()}")
    tiled = _tiled_lengths(text, stated)
    tol = float(dialect.threshold("lattice_match_tolerance"))
    # the builder's own within-tolerance predicate (np.allclose: atol plus a
    # relative term), so the test fires on exactly the builds that strained
    fired = not np.allclose(tiled, stated, atol=tol)
    note = built.info.get("strained_to_cell")
    if fired:
        off = float(np.max(np.abs(stated - tiled)))
        assert note is not None and note.startswith("strained to cell:"), (
            f"{cid} {fraction:+.1%} {axname}: stated cell is {off} A off the "
            f"tiling (> tolerance {tol}) but no strain provenance was "
            f"recorded (frame.info = {built.info})")
        got = [float(tok) for tok in note.split()
               if tok.startswith(("+", "-"))]
        assert np.allclose(got, stated / tiled - 1.0, atol=1e-5, rtol=0), (
            f"{cid}: provenance strains {got} != applied strains "
            f"{(stated / tiled - 1.0).tolist()} ({note!r})")
    else:
        assert note is None, (
            f"{cid} {fraction:+.1%} {axname}: stated cell is an exact tiling "
            f"within {tol} A but the build recorded {note!r}")


def _matrix_params():
    """One param per (case, strain config); the known-gap combos carry
    their strict xfail via the FOILED registry (tests/
    conftest.py), applied at collection -- _GAP below is the documented
    mirror, kept in sync by test_gap_set_matches_registry."""
    out = []
    for cid in CRYSTAL_CASES:
        for fraction, axname, axes in _matrix_configs():
            mid = f"{fraction:+.3f}|{axname}"
            out.append(pytest.param(cid, fraction, axname, axes,
                                    id=f"{cid}-{mid}"))
    return out



# The known-gap set, re-measured 2026-10-08 on HEAD (f657997-era tree) after
# the W10 quantized-key wiring moved some configs across the boundary
# relative to the 2026-10-02 measurement in the module docstring: 94
# (cid, matrix-id) combos whose first lift (defect arm, scan-fitted constant
# of the jittered strained frame) and re-lift (spglib arm, idealised constant
# of the exact rebuild) disagreed on the single a/c line by 0.001-0.012 A,
# registered as strict xfails in the FOILED registry (O12a's sanctioned
# home, applied at collection). CLOSED by X2 (2026-10-14, D17): with the
# lift-side snap all 94 XPASSed and their ids left the registry (the strict
# markers turned the XPASSes red, the designed forcing function). _GAP
# stays as the readable mirror machinery -- EMPTY now, and
# test_gap_set_matches_registry keeps it in sync with the registry should a
# strained-box id ever be registered again.
_GAP: set = set()


@pytest.mark.slow
@pytest.mark.parametrize("cid,fraction,axname,axes", _matrix_params())
def test_strain_matrix_lift_build_lift_text_identical(cid, fraction, axname, axes):
    """W5 done-when, text half: lift -> build -> lift gives identical text
    for every strained config. Open for 94 of 216 configs until X2 (all on
    the single a/c line -- see the module docstring); closed by the D17
    lift-side snap: every arm snaps its fitted constant to
    box_lattice_constant of the STATED cell, the same function the builder's
    strained path accepts by. This test is the done-when itself: 216/216."""
    case = acc.case_by_id(cid)
    t1, t2, _built = _round_trip(_strained_frame(case, fraction, axes),
                                 _dialect(case))
    _assert_text_identical(cid, f"{fraction:+.1%} {axname}", t1, t2)


# ---- X2 done-when, random half: 20 anisotropic draws per case =================

# rule 2 (>= 20 draws): independent per-axis strains, uniform in +-1.5 %
# (the matrix's largest magnitude), seeded per (case, draw) so a failure
# names its draw and reproduces
_RANDOM_DRAWS = 20
_RANDOM_STRAIN = 0.015
_SWEEP_SEED = 71000


def _random_strain_factors(cid: str, draw: int) -> np.ndarray:
    rng = np.random.default_rng(_SWEEP_SEED + 997 * draw + CRYSTAL_CASES.index(cid))
    return 1.0 + rng.uniform(-_RANDOM_STRAIN, _RANDOM_STRAIN, 3)


@pytest.mark.slow
@pytest.mark.parametrize("cid", CRYSTAL_CASES)
def test_random_anisotropic_strain_lift_build_lift_identical(cid):
    """X2 done-when, random half: 20 draws per case, each axis strained
    independently and uniformly in +-1.5 %, give identical lift -> build ->
    lift text. The matrix samples single-axis and isotropic strains; this
    samples the interior of the strain cube (axis combinations the matrix
    never hits, e.g. x +1.2 % with y -0.9 % and z +0.3 %), where the
    per-axis constants disagree the most and the snap's mean is the only
    text both lifts can agree on."""
    case = acc.case_by_id(cid)
    f = acc._a2a3_frame(case)
    for d in range(_RANDOM_DRAWS):
        s = _random_strain_factors(cid, d)
        fr = Frame(pos=np.mod(f.pos * s, f.cell_diag * s),
                   cell=np.diag(f.cell_diag * s),
                   symbols=f.symbols, pbc=f.pbc)
        t1, t2, _built = _round_trip(fr, _dialect(case))
        _assert_text_identical(
            cid, f"draw {d} (strains {(s - 1.0).tolist()})", t1, t2)


# ---- the refusal gate ========================================================

def test_cell_beyond_strain_tolerance_is_refused():
    """Above the strain fraction the builder refuses as before (A12): a
    hand-written program whose stated cell is 3 % off the tiling raises the
    integer-multiple error, not a silent strained build."""
    dialect = load_dialect(("core", "metal"))
    # diamond a 5.430 A in a 2x2x2 box, stretched 3 % along x
    box_x = round((2 * 5.430) * (1.0 + REFUSE_FRACTION), 3)
    text = (
        "chaord 0.1\n"
        "dialect core + metal\n"
        "\n"
        "system {\n"
        f"  cell {box_x:.3f} 10.860 10.860\n"
        "  pbc xyz\n"
        "  conserve atoms Si 64\n"
        "}\n"
        "\n"
        "physics {\n"
        "  backend eam\n"
        "}\n"
        "\n"
        "crystal bulk : all {\n"
        "  lattice diamond\n"
        "  a 5.430 A\n"
        "  orient x [100] y [010] z [001]\n"
        "}\n"
        "\n"
        "residual none\n")
    with pytest.raises(ChaordError, match="not an integer multiple"):
        build_program(parse_text(text), dialect,
                      rng=np.random.default_rng(1), physics=False)


@pytest.mark.parametrize("cid", CRYSTAL_CASES)
def test_unstrained_crystal_build_records_no_strain(cid):
    """The strain provenance appears only when the stated cell is outside the
    integer-multiple tolerance: the unstrained bench cases (exact tilings)
    must build without a strain note -- the W5 path must not paper over the
    F1 cell contract."""
    case = acc.case_by_id(cid)
    dialect = _dialect(case)
    program = lift_frame(acc._a2a3_frame(case), dialect)
    built = build_program(program, dialect,
                          rng=np.random.default_rng(1), physics=False)
    assert "strained_to_cell" not in built.info, (
        f"{cid}: unstrained build carries a strain note "
        f"({built.info.get('strained_to_cell')!r})")


def test_gap_set_matches_registry():
    """_GAP above and the FOILED registry must name exactly the same configs
    (empty since X2 closed the W5 gap): the registry applies the marks, this
    set is the readable record -- if they drift, one of them silently stops
    describing the gap."""
    import ast as _ast
    conf = (ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")
    tree = _ast.parse(conf)
    foiled = next(n.value for n in tree.body
                  if isinstance(n, _ast.Assign)
                  and getattr(n.targets[0], "id", "") == "FOILED")
    registered = {k.value for k in foiled.keys}
    gap_ids = {"tests/test_strained_boxes.py::"
               "test_strain_matrix_lift_build_lift_text_identical"
               f"[{c}-{m}]" for c, m in _GAP}
    assert registered == gap_ids, sorted(registered ^ gap_ids)[:4]
