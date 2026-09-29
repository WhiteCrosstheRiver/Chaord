"""Deposit protocol and the cn core restraint (M5+).

`deposit X 20 for 400` grows the system: every steps/count integration steps one
atom of species X lands at a seeded random free site (min-image distance to all
existing atoms >= the dialect's deposit_min_gap) and MD continues. The cn
fallback of restrained_sample pulls a 108-atom LJ liquid to its target with a
temperature/relaxation loop and fails loudly (ChaordError) when it cannot.
"""
import numpy as np
import pytest

from chaord.build import build_program
from chaord.build.slab import rsa
from chaord.cv import measure
from chaord.dialects import load_dialect
from chaord.io.frames import Frame
from chaord.lang.api import load
from chaord.lang.errors import ChaordError
from chaord.lang.ir import Arrow, Name, Quantity, Statement
from chaord.realize.protocols import parse_history, restrained_sample, run_protocol

DEPOSIT_PROGRAM = """chaord 0.1
dialect core + glass

system {{
  pbc xyz
  conserve atoms X 100
}}

physics {{
  backend lj
}}

amorphous g : all {{
  state density {rho}
  history melt 0.9 for 100 -> deposit X 20 for 400
}}
"""


@pytest.fixture(scope="module")
def glass_dialect():
    return load_dialect(("core", "glass"))


@pytest.fixture(scope="module")
def lj_dialect():
    # glass first so its protocol/restraint keys load, lj last so LJ cutoffs win
    return load_dialect(("core", "glass", "lj"))


def _rsa_frame(n, rho, seed, dialect):
    """Overlap-free protocol start, the same way build_amorphous prepares one."""
    L = np.full(3, (n / rho) ** (1 / 3))
    rng = np.random.default_rng(seed)
    pos = rsa(np.zeros((0, 3)), L, L[0] / 2, L[0] / 2, n,
              float(dialect.threshold("fluid_rsa_dmin")), rng)
    return Frame(pos=pos, cell=np.diag(L), symbols=["X"] * n)


def _min_pair_distance(frame):
    from scipy.spatial import cKDTree
    L = frame.cell_diag
    pos = np.mod(frame.pos, L)
    tree = cKDTree(pos, boxsize=L)
    pairs = tree.query_pairs(1.5, output_type="ndarray")  # any cutoff >= d_NN works
    d = pos[pairs[:, 1]] - pos[pairs[:, 0]]
    d -= L * np.round(d / L)
    return float(np.linalg.norm(d, axis=1).min())


# ------------------------------------------------------------------ parsing --

def test_deposit_parse_leading_command(glass_dialect):
    stmt = Statement(kind="history", key="deposit", values=[
        Name(text="X"), Quantity(num="20"), Name(text="for"), Quantity(num="400"),
    ])
    assert parse_history(stmt) == [("deposit", "X", 20, 400)]


def test_deposit_parse_chained_after_arrow():
    stmt = Statement(kind="history", key="melt", values=[
        Quantity(num="0.9"), Name(text="for"), Quantity(num="100"), Arrow(),
        Name(text="deposit"), Name(text="Y"), Quantity(num="5"),
        Name(text="for"), Quantity(num="50"),
    ])
    steps = parse_history(stmt)
    assert steps[0] == ("melt", 0.9, 100)
    assert steps[1] == ("deposit", "Y", 5, 50)


def test_deposit_parse_needs_species():
    stmt = Statement(kind="history", key="deposit", values=[
        Quantity(num="3"), Name(text="for"), Quantity(num="30"),
    ])
    with pytest.raises(ChaordError, match="species"):
        parse_history(stmt)


def test_deposit_parse_from_source(glass_dialect, tmp_path):
    path = tmp_path / "dep.chaord"
    path.write_text(DEPOSIT_PROGRAM.format(rho=0.6))
    program = load(path)
    region = next(b for b in program.blocks if b.t == "region")
    history = [s for s in region.statements if s.kind == "history"]
    assert len(history) == 1
    assert parse_history(history[0]) == [
        ("melt", 0.9, 100), ("deposit", "X", 20, 400)]


# ---------------------------------------------------------------- execution --

def test_deposit_grows_system(glass_dialect):
    frame = _rsa_frame(100, 0.6, 7, glass_dialect)
    out = run_protocol(frame, [("deposit", "X", 20, 400)], glass_dialect,
                       np.random.default_rng(42))
    assert len(out) == 120                      # total count grows by `count`
    assert out.symbols.count("X") == 120        # species correct, none dropped
    L = out.cell_diag
    assert np.all(out.pos >= 0.0) and np.all(out.pos <= L)  # wrapped into the box
    # every insertion kept deposit_min_gap; Langevin motion at deposit_T then
    # softens the closest pair by at most ~0.1 sigma
    gap = float(glass_dialect.threshold("deposit_min_gap"))
    assert _min_pair_distance(out) >= 0.9 * gap


def test_deposit_seeded_reproducible(glass_dialect):
    frame = _rsa_frame(100, 0.6, 7, glass_dialect)
    a = run_protocol(frame, [("deposit", "X", 20, 400)], glass_dialect,
                     np.random.default_rng(42))
    b = run_protocol(frame, [("deposit", "X", 20, 400)], glass_dialect,
                     np.random.default_rng(42))
    c = run_protocol(frame, [("deposit", "X", 20, 400)], glass_dialect,
                     np.random.default_rng(43))
    assert np.array_equal(a.pos, b.pos) and a.symbols == b.symbols
    assert not np.array_equal(a.pos, c.pos)    # different seed, different film


def test_deposit_through_amorphous_builder(glass_dialect, tmp_path):
    path = tmp_path / "glass.chaord"
    path.write_text(DEPOSIT_PROGRAM.format(rho=0.6))
    frame = build_program(load(path), glass_dialect, rng=np.random.default_rng(9),
                          physics=True)
    assert len(frame) == 120
    assert set(frame.symbols) == {"X"}
    gap = float(glass_dialect.threshold("deposit_min_gap"))
    assert _min_pair_distance(frame) >= 0.9 * gap


def test_melt_quench_anneal_smoke(glass_dialect):
    frame = _rsa_frame(64, 0.8, 3, glass_dialect)
    out = run_protocol(frame, [("melt", 1.2, 60), ("quench", 1.0, 0.1, 0.05),
                               ("anneal", 0.1, 40)], glass_dialect,
                       np.random.default_rng(4))
    assert len(out) == 64 and set(out.symbols) == {"X"}   # nothing dropped or added
    assert np.isfinite(out.pos).all()
    L = out.cell_diag
    assert np.all(out.pos >= 0.0) and np.all(out.pos <= L)


# ------------------------------------------------------------ cn restraint --

@pytest.fixture(scope="module")
def lj_liquid(lj_dialect):
    """108-atom LJ liquid: RSA start, short melt on the protocol path."""
    frame = _rsa_frame(108, 0.70, 21, lj_dialect)
    return run_protocol(frame, [("melt", 1.2, 400)], lj_dialect,
                        np.random.default_rng(11))


def test_cn_restraint_pulls_liquid_to_target(lj_liquid, lj_dialect):
    cn0 = measure("cn", lj_liquid, lj_dialect)
    target = cn0 + 0.6                       # requires the loop, not the start value
    out = restrained_sample(lj_liquid, "cn", target, 0.5, lj_dialect,
                            np.random.default_rng(3))
    assert len(out) == len(lj_liquid)        # a restraint never adds or drops atoms
    assert abs(measure("cn", out, lj_dialect) - target) <= 0.5


def test_cn_restraint_far_target(lj_liquid, lj_dialect):
    cn0 = measure("cn", lj_liquid, lj_dialect)
    target = cn0 + 1.5                       # several cooling iterations
    out = restrained_sample(lj_liquid, "cn", target, 0.5, lj_dialect,
                            np.random.default_rng(3))
    assert abs(measure("cn", out, lj_dialect) - target) <= 0.5


def test_cn_restraint_unreachable_raises(lj_liquid, lj_dialect):
    cn0 = measure("cn", lj_liquid, lj_dialect)
    # the target must be unreachable by PHYSICS, not by trajectory luck: at
    # fixed box/density (rho = 0.70) the mean cn at cn_cutoff = 1.5 sigma of
    # even a fully random (ideal-gas) configuration is rho * 4/3*pi*rc^3 ~ 9.9,
    # and the hot liquid measures 9.2-9.9; cn0 - 2.5 +- 0.5 needs cn <= ~7.9,
    # a ~2-neighbour deficit no MD trajectory at this density can produce.
    # (cn0 - 1.5 was reached on some CI runners: seeded MD is not
    # bit-reproducible across runner ISAs, and that margin sat ~0.3 from
    # reachable -- a platform-dependent premise, not a law.)
    with pytest.raises(ChaordError, match="cn restraint"):
        restrained_sample(lj_liquid, "cn", cn0 - 2.5, 0.5, lj_dialect,
                          np.random.default_rng(5))
