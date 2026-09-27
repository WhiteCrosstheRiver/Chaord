"""M3 acceptance: molecules, packing, fluid CVs, noise floor, statistical round trip.

Exit criteria (PLAN M3): held-out observables within 1.5x the noise floor on
>= 90% of fluid cases; composition exact.
"""
import numpy as np
import pytest

from chaord.build.molecules import (
    TEMPLATES, molecule_census, pack_molecules, molecular_mass,
)
from chaord.cv.noise import noise_floor, observables, distance, within_floor
from chaord.dialects import load_dialect
from chaord.io.frames import Frame
from chaord.lang.api import load, save
from chaord.lang.fmt import format_program
from chaord.lift import lift_frame


@pytest.fixture(scope="module")
def mol_dialect():
    return load_dialect(("core", "molecular"))


@pytest.fixture(scope="module")
def lj_dialect():
    return load_dialect(("core", "lj"))


def test_templates_sane(mol_dialect):
    water = TEMPLATES["H2O"]
    assert water["symbols"] == ["O", "H", "H"]
    d = np.linalg.norm(water["rel"][1] - water["rel"][0])
    assert d == pytest.approx(0.9572, abs=1e-3)      # O-H bond
    v1 = water["rel"][1] - water["rel"][0]
    v2 = water["rel"][2] - water["rel"][0]
    angle = np.degrees(np.arccos(np.dot(v1, v2)
                                 / np.linalg.norm(v1) / np.linalg.norm(v2)))
    assert angle == pytest.approx(104.52, abs=0.2)   # H-O-H angle at the oxygen
    n2 = TEMPLATES["N2"]
    assert np.linalg.norm(n2["rel"][1] - n2["rel"][0]) == pytest.approx(1.10, abs=1e-6)
    assert molecular_mass("H2O") == pytest.approx(18.015, abs=0.01)


@pytest.mark.parametrize("spec,box", [
    ({"H2O": 60}, [16.0] * 3),
    ({"N2": 80}, [20.0] * 3),
    ({"CO2": 40}, [20.0] * 3),
])
def test_packing_census_exact(spec, box, mol_dialect):
    rng = np.random.default_rng(7)
    f = pack_molecules(spec, box, rng, mol_dialect)
    census = molecule_census(f, mol_dialect)
    assert census == spec
    n_expected = sum(len(TEMPLATES[k]["symbols"]) * v for k, v in spec.items())
    assert len(f) == n_expected


def test_packing_no_molecular_merges(mol_dialect):
    rng = np.random.default_rng(11)
    f = pack_molecules({"H2O": 80}, [17.0] * 3, rng, mol_dialect)
    assert molecule_census(f, mol_dialect) == {"H2O": 80}


def test_water_lift_round_trip_conserves(mol_dialect, tmp_path):
    rng = np.random.default_rng(13)
    f = pack_molecules({"H2O": 60}, [15.0] * 3, rng, mol_dialect)
    text = format_program(lift_frame(f, mol_dialect, mode="fluid"))
    assert "molecules H2O 60" in text
    assert "conserve atoms H 120 O 60" in text
    # rebuild (no physics: classical backend has no core MD)
    path = tmp_path / "water.chaord"
    path.write_text(text)
    from chaord.build import build_program
    rng2 = np.random.default_rng(17)
    g = build_program(load(path), mol_dialect, rng=rng2, physics=True)
    assert molecule_census(g, mol_dialect) == {"H2O": 60}
    assert len(g) == 180
    text2 = format_program(lift_frame(g, mol_dialect, mode="fluid"))
    assert "molecules H2O 60" in text2
    assert "conserve atoms H 120 O 60" in text2


def test_impossible_density_is_a_static_error(mol_dialect, tmp_path):
    text = ("chaord 0.1\n\nsystem {\n  cell 10 10 10\n  pbc xyz\n}\n\n"
            "physics {\n  backend classical\n}\n\n"
            "liquid water : all {\n  molecules H2O 200\n  state density 1.0 g/cm3\n}\n")
    path = tmp_path / "bad.chaord"
    path.write_text(text)
    from chaord.build import build_program
    from chaord.lang.errors import ChaordError
    with pytest.raises(ChaordError, match="impossible density"):
        build_program(load(path), mol_dialect)


def test_noise_floor_shipped_snapshot(lj_dialect):
    """The prototype's two frames give the documented floor (g(r) ~0.12)."""
    from chaord.io.frames import read_frame
    from pathlib import Path
    root = Path(__file__).parent.parent
    a = read_frame(root / "prototype" / "snap.npz")
    b = read_frame(root / "prototype" / "snap_later.npz")
    floor = noise_floor(a, b, lj_dialect)
    assert 0.01 < floor["gr_rms"] < 0.4    # same order as the prototype's 0.12
    assert floor["cn_tv"] < 0.3


def test_observables_deterministic(lj_dialect):
    rng = np.random.default_rng(19)
    pos = rng.uniform(0, 6, (80, 3))
    f = Frame(pos=pos, cell=np.diag([6.0] * 3), symbols=["X"] * 80)
    o1 = observables(f, lj_dialect)
    o2 = observables(Frame(pos=pos + 6.0, cell=f.cell, symbols=f.symbols), lj_dialect)
    assert np.allclose(o1["gr"], o2["gr"])


@pytest.mark.slow
def test_lj_liquid_statistical_round_trip(lj_dialect, tmp_path):
    """M3 exit criterion: rebuilt liquid within 1.5x the noise floor."""
    text = ("chaord 0.1\n\nsystem {\n  cell 7.4 7.4 7.4\n  pbc xyz\n  state T 0.70\n"
            "  conserve atoms X 320\n}\n\n"
            "physics {\n  backend lj\n  epsilon 1\n  sigma 1\n  cutoff 2.5\n}\n\n"
            "liquid bulk : all {\n  state density 0.79\n}\n")
    path = tmp_path / "liq.chaord"
    path.write_text(text)
    from chaord.build import build_program
    from chaord.realize.lj import LJ, run_md

    rng = np.random.default_rng(23)
    frame_a = build_program(load(path), lj_dialect, rng=rng, physics=True,
                          md_steps=4000)
    # an independent later frame of the same (reference) simulation
    lj = LJ(frame_a.cell_diag, rc=2.5, skin=0.3)
    v = rng.normal(size=frame_a.pos.shape) * np.sqrt(0.70)
    r_later, _ = run_md(frame_a.pos.copy(), v, frame_a.cell_diag, 600, 0.005,
                        0.80, 0.5, rng, lj=lj)
    frame_later = Frame(pos=r_later, cell=frame_a.cell, symbols=frame_a.symbols)

    program = lift_frame(frame_a, lj_dialect)
    ppath = tmp_path / "lifted.chaord"
    save(program, ppath)
    rng2 = np.random.default_rng(29)
    frame_rebuilt = build_program(load(ppath), lj_dialect, rng=rng2, physics=True,
                                 md_steps=4000)

    verdict = within_floor(frame_rebuilt, frame_a, frame_later, lj_dialect)
    for k, v in verdict.items():
        print(f"{k}: distance {v['distance']:.3f} floor {v['floor']:.3f} "
              f"ratio {v['ratio']:.2f} -> {'PASS' if v['passed'] else 'FAIL'}")
        assert v["passed"], f"{k} outside 1.5x noise floor"
    # composition exact
    assert len(frame_rebuilt) == 320
