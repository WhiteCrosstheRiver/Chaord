"""Interface-slab lift and build: thin liquid films and multi-species slabs.

Two root causes from the acceptance A5 run (2026-09-29), both in the M0 slab
lifter (lift/slab.py decompile):

1. interface/lj_solid_liquid (512 atoms): the liquid bulk-statistics centre
   zone |dzl| < Hl - bulk_margin - gr_rmax is EMPTY on a small frame -- the
   film (half-height ~3.17 sigma) is thinner than margin + g(r) range
   (2.5 + 2.5 sigma) -- and decompile died inside np.concatenate(ang)
   (ValueError: need at least one array to concatenate) instead of degrading
   honestly, so the frame fell back to `liquid fluid : all`.

2. interfaces/cu_water (792 atoms, core + metal): the metal dialect defined
   none of the slab thresholds, so decompile died on the first read
   (ChaordError: dialect 'core + metal' defines no threshold 'bulk_margin');
   multi-species interface frames also need the crystal fit on the majority
   species sublattice and a molecular census of the liquid, and the builder
   needs a multi-species path (exact crystal sites + molecular RSA packing
   with the crystal as excluded obstacles, per-region physics).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chaord.build import build_program  # noqa: E402
from chaord.check.statics import conservation_check, overlap_check, run_checks  # noqa: E402
from chaord.dialects import load_dialect  # noqa: E402
from chaord.io.frames import read_frame  # noqa: E402
from chaord.lang.errors import ChaordError  # noqa: E402
from chaord.lang.fmt import format_program  # noqa: E402
from chaord.lang.parser import parse_text  # noqa: E402
from chaord.lift import lift_frame  # noqa: E402
from chaord.lift.slab import decompile, program_from_result  # noqa: E402

BENCH = ROOT / "bench" / "data"
LJ_CASE = BENCH / "interface" / "lj_solid_liquid"
CU_CASE = BENCH / "interfaces" / "cu_water"
BUILD_BUDGET_S = 150          # the A5 per-rebuild time budget (test-side constant)


def _counts(symbols):
    out: dict[str, int] = {}
    for s in symbols:
        out[s] = out.get(s, 0) + 1
    return out


# ------------------------------------------------- thin liquid film (core+lj) --

@pytest.fixture(scope="module")
def thin():
    frame = read_frame(LJ_CASE / "frame_0.npz")
    dialect = load_dialect(("core", "lj"))
    res = decompile(frame.pos, frame.cell_diag, 0.65, dialect)
    program = program_from_result(res, dialect)
    return frame, dialect, res, program


def test_thin_film_decompile_degrades_instead_of_crashing(thin):
    """Before the fix: ValueError from np.concatenate on an empty centre zone.

    After: honest degradation -- the margin and the g(r) range shrink
    proportionally (dialect key thin_slab_margin_fraction), the g(r) range
    never drops below cn_cutoff, and the shrink is visible in the result."""
    _, dialect, res, _ = thin
    assert res["thin_margin_l"] < float(dialect.threshold("bulk_margin"))
    assert res["thin_rmax_l"] >= float(dialect.threshold("cn_cutoff"))
    # the centre statistics zone is genuinely non-empty now
    assert res["n_centre"] > 0
    assert res["thin_note"] and "thin-film" in res["thin_note"]


def test_thin_film_lifts_a_real_interface_program(thin):
    """Before the fix the frame lifted as `liquid fluid : all`; now the lift
    states the crystal, the liquid and both interfaces."""
    _, _, res, program = thin
    text = format_program(program)
    assert "crystal A : slab z" in text
    assert "liquid B : slab z" in text
    assert text.count("interface A | B") == 1
    assert text.count("interface B | A") == 1
    assert "fluid" not in text
    # the degradation is visible in the program: the liquid block says so
    liq = next(b for b in program.blocks if b.t == "region" and b.phase == "liquid")
    assert liq.comment and "thin-film" in liq.comment


def test_thin_film_mode_slab_matches_direct_decompile(thin):
    """The legacy cascade's slab arm (mode='slab') produces the same text as
    calling decompile/program_from_result directly."""
    frame, dialect, _, program = thin
    via_mode = format_program(lift_frame(frame, dialect, mode="slab"))
    assert via_mode == format_program(program)


def test_thin_film_program_conserves_and_reparses(thin):
    frame, dialect, _, program = thin
    assert parse_text(format_program(program)) is not None
    assert conservation_check(program, frame, dialect).passed


def test_thin_film_build_under_the_time_budget(thin):
    """The rebuilt interface frame must come back with every atom, pass the
    static checks, and finish inside the A5 budget (interface builds pack
    per region; they do not spin like a homogeneous-fluid RSA)."""
    frame, dialect, _, program = thin
    text = format_program(program)
    t0 = time.perf_counter()
    built = build_program(parse_text(text), dialect,
                          rng=np.random.default_rng(7), physics=True)
    seconds = time.perf_counter() - t0
    assert seconds < BUILD_BUDGET_S, f"rebuild took {seconds:.1f}s"
    assert len(built) == len(frame) == 512
    for c in run_checks(parse_text(text), built, dialect):
        assert c.passed, f"{c.name}: {c.detail}"


# ------------------------------------------- multi-species interface (metal) --

@pytest.fixture(scope="module")
def metal_keys():
    return load_dialect(("core", "metal"))


@pytest.mark.parametrize("key", [
    "bulk_margin", "gr_rmax", "gr_bins", "angle_hist_bin", "defect_cluster_rc",
    "thin_slab_margin_fraction", "printed_strain_tolerance",
    "rsa_dmin", "liquid_margin_shrink",
])
def test_metal_defines_the_slab_thresholds(metal_keys, key):
    """Before the fix: ChaordError on the first read (bulk_margin)."""
    dialect = metal_keys
    try:
        dialect.threshold(key)
    except ChaordError as e:
        pytest.fail(f"core + metal lacks slab threshold {key!r}: {e}")


@pytest.fixture(scope="module")
def cu():
    frame = read_frame(CU_CASE / "frame_0.npz")
    dialect = load_dialect(("core", "metal"))
    res = decompile(frame.pos, frame.cell_diag, 300.0, dialect,
                    symbols=list(frame.symbols))
    program = program_from_result(res, dialect)
    return frame, dialect, res, program


def test_cu_water_multispecies_lift(cu):
    """Crystal fit on the Cu sublattice, molecular census in the liquid,
    per-species conservation, interface blocks kept."""
    frame, dialect, res, program = cu
    text = format_program(program)
    assert "lattice fcc" in text
    # lattice constant recovered near the planted a = 3.615 A
    crystal = next(b for b in program.blocks if b.t == "region" and b.phase == "crystal")
    a_stmt = next(s for s in crystal.statements if s.key == "a")
    assert float(a_stmt.values[0].num) == pytest.approx(3.615, abs=0.02)
    assert "molecules H2O 136" in text
    assert "conserve atoms Cu 384 H 272 O 136" in text
    assert text.count("interface A | B") == 1
    assert text.count("interface B | A") == 1
    assert conservation_check(program, frame, dialect).passed
    assert parse_text(text) is not None


def test_cu_water_liquid_density_stated(cu):
    """The liquid block states the water density (~0.9 g/cm3 planted)."""
    _, _, _, program = cu
    liquid = next(b for b in program.blocks if b.t == "region" and b.phase == "liquid")
    dens = next(s for s in liquid.statements if s.key == "density")
    assert dens.values[0].unit == "g/cm3"
    assert float(dens.values[0].num) == pytest.approx(0.9, abs=0.05)


def test_cu_water_build_under_the_time_budget(cu):
    """Cu built exactly on lattice sites, water RSA-packed against the Cu
    obstacles, per-region physics; exact per-species counts; inside 150 s."""
    frame, dialect, _, program = cu
    text = format_program(program)
    t0 = time.perf_counter()
    built = build_program(parse_text(text), dialect,
                          rng=np.random.default_rng(7), physics=True)
    seconds = time.perf_counter() - t0
    assert seconds < BUILD_BUDGET_S, f"rebuild took {seconds:.1f}s"
    assert _counts(built.symbols) == _counts(frame.symbols) == {
        "Cu": 384, "H": 272, "O": 136}
    assert overlap_check(built, dialect).passed, overlap_check(built, dialect).detail
    assert conservation_check(parse_text(text), built, dialect).passed


def test_cu_water_build_is_deterministic(cu):
    _, dialect, _, program = cu
    text = format_program(program)
    a = build_program(parse_text(text), dialect, rng=np.random.default_rng(7),
                      physics=False)
    b = build_program(parse_text(text), dialect, rng=np.random.default_rng(7),
                      physics=False)
    assert np.array_equal(a.pos, b.pos)
    assert a.symbols == b.symbols


# ------------------------------------------------------ single-species parity --

def test_single_species_path_is_unchanged_by_symbols_argument():
    """Passing symbols for a one-species frame must not change the text, and
    the shipped LJ snapshot golden must hold through the symbols argument
    (the single-species path is byte-identical, regression gate)."""
    golden = (ROOT / "tests" / "golden" / "lj_slab.chaord").read_text(encoding="utf-8")
    z = np.load(ROOT / "prototype" / "snap.npz")
    dialect = load_dialect(("core", "lj"))
    plain = decompile(z["r"], z["L"], 0.65, dialect)
    named = decompile(z["r"], z["L"], 0.65, dialect, symbols=["X"] * len(z["r"]))
    assert format_program(program_from_result(named, dialect)) == \
        format_program(program_from_result(plain, dialect)) == golden
