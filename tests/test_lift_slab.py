"""Golden lift of the shipped LJ snapshot + build round trip + conservation.

The golden text is deterministic for a given snap.npz, dialect versions and
package version; it is regenerated with --regenerate-golden.
"""
from pathlib import Path

import numpy as np
import pytest

from chaord.build.slab import build_slab
from chaord.check.statics import run_checks
from chaord.dialects import load_dialect
from chaord.lang.api import load, save
from chaord.lang.fmt import format_program
from chaord.lang.parser import parse_text
from chaord.lift.slab import decompile, program_from_result

ROOT = Path(__file__).parent.parent
SNAP = ROOT / "prototype" / "snap.npz"
GOLDEN = Path(__file__).parent / "golden" / "lj_slab.chaord"


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
