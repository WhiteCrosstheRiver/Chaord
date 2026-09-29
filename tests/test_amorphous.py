"""M5 acceptance: protocols, restraints, glass statistics, shortest program.

Exit criteria (PLAN M5): constrained quantities held within tolerance;
amorphous held-out statistics within 1.5x the noise floor on >= 80% of cases.
"""
import numpy as np
import pytest

from chaord.build.crystal import build_conventional
from chaord.cv.glass import ring_distribution, voronoi_index_distribution
from chaord.dialects import load_dialect
from chaord.io.frames import Frame
from chaord.lang.api import load
from chaord.lang.fmt import format_program
from chaord.lift import lift_frame
from chaord.realize.protocols import parse_history, restrained_sample, run_protocol
from chaord.lang.ir import Name, Quantity, Statement

AMORPH_PROGRAM = """chaord 0.1
dialect core + glass

system {{
  pbc xyz
  conserve atoms X {n}
}}

physics {{
  backend lj
}}

amorphous glass : all {{
  state density {rho}
  history melt 2 for 3000 -> quench to 0.01 at 0.000332 -> anneal 0.01 for 2000
}}
"""


@pytest.fixture(scope="module")
def dialect():
    return load_dialect(("core", "glass"))


def test_history_parse():
    stmt = Statement(kind="history", key="melt", values=[
        Name(text="melt"), Quantity(num="1.2"), Name(text="for"), Quantity(num="300"),
        Name(text="quench"), Name(text="to"), Quantity(num="0.01"),
        Name(text="at"), Quantity(num="0.01"),
        Name(text="anneal"), Quantity(num="0.01"), Name(text="for"), Quantity(num="200"),
    ])
    # the parser walks the whole statement: build it as one statement with arrow
    from chaord.lang.ir import Arrow
    stmt = Statement(kind="history", key="melt", values=[
        Quantity(num="1.2"), Name(text="for"), Quantity(num="300"), Arrow(),
        Name(text="quench"), Name(text="to"), Quantity(num="0.01"),
        Name(text="at"), Quantity(num="0.01"), Arrow(),
        Name(text="anneal"), Quantity(num="0.01"), Name(text="for"), Quantity(num="200"),
    ])
    steps = parse_history(stmt)
    assert steps[0] == ("melt", 1.2, 300)
    assert steps[1] == ("quench", 0.01, 0.01, 0.01)
    assert steps[2] == ("anneal", 0.01, 200)


def test_density_restraint_holds(dialect):
    rng = np.random.default_rng(3)
    L = np.array([8.0, 8.0, 8.0])
    frame = Frame(pos=rng.uniform(0, L, (300, 3)), cell=np.diag(L),
                  symbols=["X"] * 300)
    target = 0.9
    out = restrained_sample(frame, "density", target, 0.02, dialect, rng)
    rho = len(out.pos) / float(np.prod(out.cell_diag))
    assert abs(rho - target) <= 0.02


def test_ring_statistics_diamond(dialect):
    dia = build_conventional("diamond", {"a": 5.43}, ("Si",), (2, 2, 2))
    rings = ring_distribution(dia, dialect)
    assert rings[6] == 128  # every diamond bond closes exactly 6-rings
    assert set(rings) == {6}


def test_voronoi_index_sc(dialect):
    sc = build_conventional("sc", {"a": 3.0}, ("Ar",), (3, 3, 3))
    vd = voronoi_index_distribution(sc, dialect, max_atoms=81)
    # the simple-cubic Wigner-Seitz cell is a cube: exactly 6 faces
    assert vd.most_common(1)[0][0] == 6


@pytest.mark.slow
def test_amorphous_round_trip_statistics(dialect, tmp_path):
    """M5 exit criterion: rebuilt glass within 1.5x the noise floor.

    The frame is a sample of the macrostate the lifter describes: the
    dialect's canonical melt-quench (glass_melt_T/steps, glass_quench_*,
    glass_anneal_* in glass.yaml), at the N=500 size the criterion is
    calibrated on (A5's reference glass). A shorter toy protocol would put
    the frame in a different (faster-quenched) family than the lifted
    program rebuilds from."""
    from chaord.build import build_program
    from chaord.cv.noise import observables, distance
    text = AMORPH_PROGRAM.format(n=500, rho=0.95)
    path = tmp_path / "glass.chaord"
    path.write_text(text)
    rng = np.random.default_rng(17)
    frame = build_program(load(path), dialect, rng=rng, physics=True)

    # an independent later frame of the same protocol continuation
    from chaord.realize.lj import LJ, run_md
    lj = LJ(frame.cell_diag, rc=2.5, skin=0.3)
    v = rng.normal(size=frame.pos.shape) * np.sqrt(0.05)
    r_later, _ = run_md(frame.pos.copy(), v, frame.cell_diag, 400, 0.005, 0.05, 0.5,
                        rng, lj=lj)
    later = Frame(pos=r_later, cell=frame.cell, symbols=frame.symbols)

    program = lift_frame(frame, dialect)
    ppath = tmp_path / "lifted.chaord"
    ppath.write_text(format_program(program))
    rng2 = np.random.default_rng(23)
    rebuilt = build_program(load(ppath), dialect, rng=rng2, physics=True)
    assert len(rebuilt) == 500  # never drop an atom

    oo, ol, o_re = observables(frame, dialect), observables(later, dialect), observables(rebuilt, dialect)
    floor = distance(oo, ol)
    d = distance(oo, o_re)
    for k in floor:
        print(f"{k}: {d[k]:.3f} vs floor {floor[k]:.3f} (x{d[k]/max(floor[k],1e-9):.2f})")
        assert d[k] <= 1.5 * max(floor[k], 1e-6), k


def test_amorphous_lift_program(dialect, tmp_path):
    from chaord.build import build_program
    text = AMORPH_PROGRAM.format(n=108, rho=0.9).replace(
        "melt 2 for 3000", "melt 2 for 300").replace(
        "anneal 0.01 for 2000", "anneal 0.01 for 200")
    path = tmp_path / "g2.chaord"
    path.write_text(text)
    rng = np.random.default_rng(5)
    frame = build_program(load(path), dialect, rng=rng, physics=True)
    text2 = format_program(lift_frame(frame, dialect, mode="amorphous"))
    assert "amorphous glass : all" in text2
    assert "conserve atoms X 108" in text2
    assert "state density 0.9" in text2


def test_shortest_controller(dialect):
    from chaord.check.shortest import shortest_program
    from chaord.build.fluid import build_fluid

    rng = np.random.default_rng(7)
    frame = Frame(pos=rng.uniform(0, 7, (200, 3)), cell=np.diag([7.0] * 3),
                  symbols=["X"] * 200)
    program = lift_frame(frame, load_dialect(("core", "lj")))
    name, text, _d = shortest_program(
        frame, program, load_dialect(("core", "lj")),
        build_fn=lambda p, r: build_fluid(p, load_dialect(("core", "lj")), r,
                                          physics=False))
    assert "conserve atoms X 200" in text  # conservation always survives
    assert name in ("full", "no-asserts", "minimal")
