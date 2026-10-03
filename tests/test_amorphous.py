"""M5 acceptance: protocols, restraints, glass statistics, shortest program.

Exit criteria (PLAN M5): constrained quantities held within tolerance;
amorphous held-out statistics within 1.5x the noise floor on >= 80% of cases.
"""
import inspect

import numpy as np
import pytest

from chaord.build.crystal import build_conventional
from chaord.cv.glass import ring_distribution, voronoi_index_distribution
from chaord.dialects import load_dialect
from chaord.io.frames import Frame
from chaord.lang.api import load
from chaord.lang.errors import ChaordError
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
    program rebuilds from.

    O10 (Review 7): the floor is the cross-quench scale of two INDEPENDENTLY
    PREPARED reference glasses (independent_glass_floor below -- fresh
    seeds, complete protocol, never the tested rebuild, its later frames,
    or any other quantity of the system under test; AGENTS.md: a floor never
    includes the tested system).  The previous gate took max(later-frame
    floor, distance between two rebuilds): a noisy rebuild then widened its
    own tolerance -- two rebuilds that landed in distant basins certified
    exactly that larger distance as acceptable.  The independent floor is
    the same construction A5's glass floor uses (cross-quench pairs)."""
    from chaord.build import build_program
    from chaord.cv.noise import observables, distance
    text = AMORPH_PROGRAM.format(n=500, rho=0.95)
    path = tmp_path / "glass.chaord"
    path.write_text(text)
    rng = np.random.default_rng(17)
    frame = build_program(load(path), dialect, rng=rng, physics=True)
    oo = observables(frame, dialect)

    program = lift_frame(frame, dialect)
    ppath = tmp_path / "lifted.chaord"
    ppath.write_text(format_program(program))
    rng2 = np.random.default_rng(23)
    rebuilt = build_program(load(ppath), dialect, rng=rng2, physics=True)
    assert len(rebuilt) == 500  # never drop an atom

    floor = independent_glass_floor(dialect)
    d = distance(oo, observables(rebuilt, dialect))
    for k in sorted(d):
        print(f"{k}: {d[k]:.3f} vs floor {floor[k]:.3f} "
              f"(x{d[k]/max(floor[k],1e-9):.2f})")
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


def test_amorphous_without_history_assumes_dialect_protocol(dialect, tmp_path):
    """Review 2 root cause: an amorphous program with no history line silently
    returned the random RSA packing -- a frame that never saw melt-quench.
    Instead the builder runs the dialect's default protocol (glass_melt/quench/
    anneal thresholds, the same ones the lifter states) and records that the
    protocol was assumed."""
    from chaord.build import build_program
    text = AMORPH_PROGRAM.format(n=30, rho=0.8).replace(
        "  history melt 2 for 3000 -> quench to 0.01 at 0.000332 -> anneal 0.01 for 2000\n",
        "")
    assert "history" not in text
    path = tmp_path / "no_history.chaord"
    path.write_text(text)
    packed = build_program(load(path), dialect,
                           rng=np.random.default_rng(2), physics=False)
    quenched = build_program(load(path), dialect,
                             rng=np.random.default_rng(2), physics=True)
    assert len(quenched) == 30  # never drop an atom
    assert quenched.info.get("assumed_history") == \
        "assumed default protocol from dialect"
    # the protocol actually ran: not the RSA start any more
    assert not np.allclose(packed.pos, quenched.pos)


def test_amorphous_without_history_unknown_dialect_raises(tmp_path):
    """A dialect that defines no default glass protocol cannot silently fall
    back to the packing either: the missing threshold is the error."""
    from chaord.build import build_program
    from chaord.lang.errors import ChaordError
    text = AMORPH_PROGRAM.format(n=30, rho=0.8).replace(
        "  history melt 2 for 3000 -> quench to 0.01 at 0.000332 -> anneal 0.01 for 2000\n",
        "")
    path = tmp_path / "no_history_lj.chaord"
    path.write_text(text.replace("dialect core + glass", "dialect core + lj"))
    with pytest.raises(ChaordError, match="glass_melt_T"):
        build_program(load(path), load_dialect(("core", "lj")),
                      rng=np.random.default_rng(2), physics=True)


def test_amorphous_lift_marks_assumed_history(dialect, tmp_path):
    """The amorphous lifter states a history line whose parameters are the
    dialect defaults, not anything measured from the frame: the program must
    say so (provenance note), so a reader knows the protocol was assumed."""
    from chaord.build import build_program
    # n=108, rho*=0.7, full melt: an honestly amorphous frame on every
    # platform (the old n=30/rho*=0.8 fixture crystallised under the
    # corrected first-shell q6 gate -- 30 atoms near the freezing density
    # order readily; measured solid fractions: 0.065 here vs 1.0 there.
    # The clean machine also failed the old shortened-melt draw)
    text = AMORPH_PROGRAM.format(n=108, rho=0.7)
    path = tmp_path / "g3.chaord"
    path.write_text(text)
    frame = build_program(load(path), dialect, rng=np.random.default_rng(5),
                          physics=True)
    text2 = format_program(lift_frame(frame, dialect, mode="amorphous"))
    assert "history melt" in text2
    assert 'note "assumed default protocol from dialect"' in text2


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


# ------------------------------------------------- O9: crystals never amorphous

def lj_crystal(proto, rho_star, rep, jitter=0.0, seed=0):
    """Perfect (or lightly jittered, wrapped) prototype crystal in LJ sigma
    units, X species: the glass dialect's own unit system (glass.yaml is
    exercised exclusively with the lj backend, whose sigma unit is the A)."""
    from ase.build import bulk
    if proto == "hcp":
        a = (2.0 / (0.8660254 * 2 * 1.633 * rho_star)) ** (1.0 / 3.0)
        at = bulk("X", "hcp", a=a, c=1.633 * a).repeat(rep)
    else:
        per_cell = {"fcc": 4, "bcc": 2, "diamond": 8, "sc": 1}[proto]
        a = (per_cell / rho_star) ** (1.0 / 3.0)
        at = bulk("X", proto, a=a, cubic=True).repeat(rep)
    at.set_masses(np.full(len(at), 1.0))
    pos = at.positions
    if jitter:
        pos = pos + np.random.default_rng(seed).normal(0, jitter, pos.shape)
    L = at.cell.lengths()
    # wrap: stored/reference frames are wrapped, and unwrapped jitter used to
    # crash typical_neighbor_distance inside network_edges (a crash the
    # amorphous gate then swallowed as a False -- a bug it must not rely on)
    return Frame(pos=np.mod(pos, L), cell=np.diag(L),
                 symbols=["X"] * len(at), pbc=(True,) * 3)


@pytest.mark.parametrize(
    "proto,rho", [("fcc", 0.85), ("fcc", 1.0), ("bcc", 0.85), ("bcc", 1.0),
                  ("hcp", 0.85), ("hcp", 1.0), ("diamond", 0.85),
                  ("diamond", 1.0), ("sc", 0.85)])
def test_perfect_crystal_never_lifts_as_amorphous(dialect, proto, rho):
    """O9 (Review 7): a crystal must never promote to the amorphous macrostate
    under any dialect.  The glass dialect's phase-indicator cutoff used to be
    3.0 in oxide-network A; in the LJ sigma units the dialect is exercised in
    it averaged q6 over ~3 neighbour shells, which erases the crystal signal
    (multi-shell q6bar of perfect fcc measures 0.021, far below q6_solid) and
    let a perfect fcc crystal pass is_amorphous.  Unit discipline: the glass
    dialect's q6_cutoff states the same first-shell sigma cutoff the lj
    dialect states (value change flagged for owner approval in glass.yaml)."""
    from chaord.lift.amorphous import is_amorphous
    frame = lj_crystal(proto, rho, (4, 4, 4) if proto == "hcp" else 4)
    assert not is_amorphous(frame, dialect), \
        f"perfect {proto} crystal at rho*={rho} promotes as amorphous"
    with pytest.raises(ChaordError, match="not a bonded disordered network"):
        lift_frame(frame, dialect, mode="amorphous")


@pytest.mark.parametrize("jitter", [0.03, 0.06])
@pytest.mark.parametrize("rho", [0.85, 1.0])
def test_thermal_crystal_never_lifts_as_amorphous(dialect, rho, jitter):
    """Same invariant for a thermally vibrating crystal (wrapped, as stored
    frames are): the arm the cascade reaches when the exact crystal engines
    refuse a noisy lattice must not be the amorphous one."""
    from chaord.lift.amorphous import is_amorphous
    frame = lj_crystal("fcc", rho, 5, jitter=jitter, seed=3)
    assert not is_amorphous(frame, dialect), \
        f"fcc crystal at rho*={rho} with {jitter} sigma jitter promotes as amorphous"


# --------------------------------------- O10: floor excludes the tested system

def test_round_trip_floor_excludes_the_system_under_test():
    """O10 (Review 7): the round-trip gate's floor must never include a
    quantity measured on the system under test.

    The gate in test_amorphous_round_trip_statistics used to take
    ``max(later-frame floor, distance between two rebuilds)``: a noisy rebuild
    then widened its own tolerance (AGENTS.md rule: a floor or tolerance never
    includes a quantity measured on the system under test).  The floor must
    come from two INDEPENDENTLY PREPARED reference glasses (fresh seeds, full
    protocol), the same cross-quench construction A5's glass floor uses.

    This test fails while the inclusion stands and passes only when the gate
    reads its floor from the independent-preparation helper alone."""
    src = inspect.getsource(test_amorphous_round_trip_statistics)
    assert "independent_glass_floor(" in src, \
        "the round-trip gate must take its floor from " \
        "independent_glass_floor(dialect), not from the tested preparation"
    for banned in ("cross[", "cross =", "max(f[k] for f in floors)",
                   "floors.append"):
        assert banned not in src, \
            f"the gate still folds '{banned}' (a quantity of the tested " \
            "system) into its own floor"
    helper = inspect.getsource(independent_glass_floor)
    for tested_seed in ("default_rng(17)", "default_rng(23)",
                        "default_rng(31)"):
        assert tested_seed not in helper, \
            "the floor helper reuses a seed of the tested preparation"


def independent_glass_floor(dialect, n=500, rho=0.95, seeds=(101, 102)):
    """O10 floor: the cross-quench scale from TWO independently prepared
    reference glasses -- complete melt-quench-anneal preparations with fresh
    seeds, never the tested rebuild, its later frames, or any other quantity
    of the system under test (AGENTS.md: a floor never includes the tested
    system).  Same construction as A5's cross-quench glass floor."""
    import tempfile
    from pathlib import Path

    from chaord.build import build_program
    from chaord.cv.noise import observables, distance
    text = AMORPH_PROGRAM.format(n=n, rho=rho)
    glasses = []
    with tempfile.TemporaryDirectory() as td:
        for seed in seeds:
            path = Path(td) / f"floor_glass_{seed}.chaord"
            path.write_text(text)
            glasses.append(build_program(
                load(path), dialect,
                rng=np.random.default_rng(seed), physics=True))
    a, b = (observables(g, dialect) for g in glasses)
    return distance(a, b)
