"""Golden lift of the shipped LJ snapshot + build round trip + conservation.

The golden text is deterministic for a given snap.npz, dialect versions and
package version; it is regenerated with --regenerate-golden.
"""
import warnings
from pathlib import Path

import numpy as np
import pytest

from chaord.build.slab import build_slab
from chaord.check.statics import run_checks
from chaord.dialects import load_dialect
from chaord.io.frames import Frame, read_frame
from chaord.lang.api import load, save
from chaord.lang.errors import ChaordError
from chaord.lang.fmt import format_program
from chaord.lang.parser import parse_text
from chaord.lift import lift_frame
from chaord.lift.slab import decompile, program_from_result

ROOT = Path(__file__).parent.parent
SNAP = ROOT / "prototype" / "snap.npz"
GOLDEN = Path(__file__).parent / "golden" / "lj_slab.chaord"
LIPF6_EC = ROOT / "bench" / "data" / "solutions" / "lipf6_ec" / "frame_0.npz"
CU_REF = ROOT / "bench" / "reference" / "cu_solid_liquid" / "frame_0.npz"


@pytest.fixture(scope="module")
def lifted():
    dialect = load_dialect(("core", "lj"))
    z = np.load(SNAP)
    res = decompile(z["r"], z["L"], 0.65, dialect)
    return res, program_from_result(res, dialect), dialect


def test_golden_lift(lifted):
    res, program, _ = lifted
    if not GOLDEN.exists():
        GOLDEN.parent.mkdir(exist_ok=True)
        GOLDEN.write_text(format_program(program), encoding="utf-8")
        pytest.fail("golden file was missing; wrote it, rerun")
    assert format_program(program) == GOLDEN.read_text(encoding="utf-8")


def test_defects_found(lifted):
    res, _, _ = lifted
    names = [c["name"] for c in res["defects"]]
    assert "divacancy" in names or "vacancy" in names
    net = sum(c["net"] for c in res["defects"])
    assert net == 3  # three vacancies were planted in make_snapshot.py


def test_lifted_program_reparses(lifted):
    _, program, _ = lifted
    text = format_program(program)
    from chaord.lang.ir import ir_equal
    assert ir_equal(parse_text(text), program)


def test_conserves_every_atom(lifted):
    res, program, dialect = lifted
    z = np.load(SNAP)
    frame_atoms = len(z["r"])
    stated = None
    for b in program.blocks:
        if b.t == "system":
            for s in b.statements:
                if s.kind == "conserve" and s.key == "atoms":
                    stated = int(s.values[-1].num)
    assert stated == frame_atoms == res["N"]


def test_build_round_trip_exact_counts(lifted, tmp_path):
    res, program, dialect = lifted
    path = tmp_path / "p.chaord"
    save(program, path)
    rng = np.random.default_rng(3)
    frame = build_slab(load(path), dialect, rng, physics=False)
    assert len(frame) == res["N"]  # never drop an atom
    checks = run_checks(load(path), frame, dialect)
    for c in checks:
        assert c.passed, f"{c.name}: {c.detail}"


def test_stated_observables_recovered(lifted, tmp_path):
    """After build (no physics) + lift, the stated lines must come back."""
    res, program, dialect = lifted
    path = tmp_path / "p.chaord"
    save(program, path)
    rng = np.random.default_rng(3)
    frame = build_slab(load(path), dialect, rng, physics=False)
    res2 = decompile(frame.pos, frame.cell_diag, 0.65, dialect)
    assert res2["N"] == res["N"]
    assert res2["a_x"] == pytest.approx(res["a_x"], abs=0.01)
    assert res2["rho"] == pytest.approx(res["rho"], abs=0.05)
    assert sum(c["net"] for c in res2["defects"]) >= 1  # defects re-seen (statistical)


@pytest.mark.slow
def test_build_with_md_statistical_round_trip(lifted, tmp_path):
    """g(r) distance within 1.5x the prototype's noise floor (0.12)."""
    res, program, dialect = lifted
    path = tmp_path / "p.chaord"
    save(program, path)
    rng = np.random.default_rng(3)
    frame = build_slab(load(path), dialect, rng, physics=True, md_steps=2000)
    res2 = decompile(frame.pos, frame.cell_diag, 0.65, dialect)
    m = res["rm"] > 0.8
    rms = float(np.sqrt(np.mean((res2["gr"][m] - res["gr"][m]) ** 2)))
    noise_floor = 0.12
    assert rms <= 1.5 * noise_floor, f"g(r) RMS {rms} vs {1.5 * noise_floor}"


# ------------------------------------------- X9 (Review 10): explicit refusal --

def _runtime_warnings(recorded):
    return [f"{w.category.__name__}: {w.message} ({w.filename}:{w.lineno})"
            for w in recorded if issubclass(w.category, RuntimeWarning)]


def test_decompile_refuses_empty_crystal_interior_without_numpy_warnings():
    """X9: a frame whose interface fit leaves no crystal-interior atom
    (bench solutions/lipf6_ec, lifted under core+molecular) must refuse
    with ONE explicit ChaordError -- never a trail of numpy RuntimeWarnings
    (mean of an empty slice at the chi0 / _circ_offset means, the 0/0
    orientation histogram at slab.py:78) on the normal refuse path.

    Red before the empty-interior check: decompile warned twice at
    slab.py:78 (and several empty-slice means inside numpy), then raised
    'liquid film too thin' anyway."""
    frame = read_frame(LIPF6_EC)
    dialect = load_dialect(("core", "molecular"))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", RuntimeWarning)
        with pytest.raises(ChaordError, match="interior"):
            decompile(frame.pos, frame.cell_diag, None, dialect,
                      symbols=frame.symbols)
    assert not _runtime_warnings(w), _runtime_warnings(w)


def test_lift_frame_slab_fallback_is_warning_free():
    """The full lift of the same frame (pipeline slab attempt -> ChaordError
    -> single-phase fluid fallback, X9) stays RuntimeWarning-free end to end
    and keeps its program: the molecular census of the pinned compiler-gap
    case (PINNED_COMPILER_GAPS, test_bench_sanity.py) is unchanged."""
    frame = read_frame(LIPF6_EC)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", RuntimeWarning)
        program = lift_frame(frame, load_dialect(("core", "molecular")))
    assert not _runtime_warnings(w), _runtime_warnings(w)
    text = format_program(program)
    assert "molecules C3H4O3 40" in text
    assert "molecules F6P 4" in text


def test_cu_slab_under_wrong_units_refuses_w4_without_warnings():
    """X9's Cu-slab arm: the Cu reference lifted under core + lj (sigma-scale
    q6 cutoff sees no Cu-Cu bond -> empty crystal-interior bond set) must
    keep refusing with the W4 units message, warning-free -- the empty
    orientation histogram is an explicit branch (None -> program-side
    refusal after the W4 gate), never a numpy 0/0.

    Red before the fix: the W4 refusal was reached only after
    orientation() divided an all-zero histogram by its zero sum
    (RuntimeWarning at slab.py:78)."""
    frame = read_frame(CU_REF)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", RuntimeWarning)
        with pytest.raises(ChaordError,
                           match="single-species metal solid"):
            lift_frame(frame, load_dialect(("core", "lj")), mode="slab")
    assert not _runtime_warnings(w), _runtime_warnings(w)
