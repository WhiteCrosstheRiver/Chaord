"""W1 (Review 8, docs/reviews/open_items_v2.md): crystal, glass and gas
detection must be unit-free and fail closed.

Red-first (AGENTS.md: the failing test is written before the fix).  Pre-fix
state, reproduced on this tree with the W1 reproducer (2026-10-02):

* 9 of 9 bench crystals were called amorphous by ``is_amorphous`` under
  ``core + glass``: glass.yaml's ``q6_cutoff`` 1.3 was meant in sigma but
  holds no neighbour on an Angstrom crystal (decision D10 rejected it as the
  fix);
* ``mode="fluid"`` refused the dilute argon bench gas ("solid-like fraction
  too high"): an atom with one or two neighbours inside an absolute cutoff
  has q6bar near 1;
* a missing threshold inside ``is_amorphous`` was swallowed
  (``except Exception: disordered = True``) and inside ``legacy_lift`` every
  arm fell through on every error, so the cascade could end in a program no
  arm stood behind.

The fix under test (W1 steps 1-4): the crystal lattice fit gates
``is_amorphous`` first (any prototype site coverage at or above
``crystal_site_coverage_min`` -> not amorphous) and its unit-free keys live
in core.yaml so the fit runs under every dialect stack; the local-order test
scales with the frame's own median d_NN (``q_cutoff_factor``), requires
``q_min_neighbours`` neighbours before an atom can be crystal-like
(q6bar > q6_solid OR q4bar > q4_solid); disordered is below
``amorphous_solid_frac_max``; errors propagate (``NotThisPhase`` is the only
fall-through in the cascade).

Random criteria are judged over DRAWS = 20 draws (the 20-draw rule of
AGENTS.md); every draw uses its own seed.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from chaord.dialects import Dialect, load_dialect          # noqa: E402
from chaord.io.frames import Frame, read_frame             # noqa: E402
from chaord.lang.errors import ChaordError                 # noqa: E402
from chaord.lang.fmt import format_program                 # noqa: E402
from chaord.lift import lift_frame                         # noqa: E402
from tools import acceptance as acc                        # noqa: E402

DRAWS = 20          # the 20-draw rule (AGENTS.md): one draw is not evidence
JITTER_DNN = 0.10   # W1 done-when: 0.10 d_NN Gaussian jitter (test constant)
LJ_RHOS = (0.85, 1.1)
LJ_PROTOS = ("fcc", "bcc", "hcp", "sc", "diamond")
REFERENCE = ROOT / "bench" / "reference"


@pytest.fixture(scope="module")
def glass_dialect():
    return load_dialect(("core", "glass"))


# --------------------------------------------------------------- LJ crystals --

def _lj_a(proto, rho_star):
    """Conventional lattice constant of an LJ prototype at rho* (sigma)."""
    per_cell = {"fcc": 4, "bcc": 2, "diamond": 8, "sc": 1, "hcp": 4}[proto]
    if proto == "hcp":
        # orthohexagonal cell (a, sqrt(3) a, c), 4 atoms, ideal c/a:
        # V_atom = (sqrt(3)/2) a^2 c = 0.7071 a^3 at c = 1.633 a
        return (2.0 / (0.8660254 * 2 * 1.633 * rho_star)) ** (1.0 / 3.0)
    return (per_cell / rho_star) ** (1.0 / 3.0)


def _lj_crystal(proto, rho_star, rep, jitter_dnn, seed):
    """A wrapped LJ prototype crystal in sigma units, X species, Gaussian
    jitter of `jitter_dnn` x d_NN (hcp ships in the orthohexagonal setting
    x [100] y [1 2 0] z [001], the box the hcp lift arm fits)."""
    from chaord.build.defects import typical_neighbor_distance
    if proto == "hcp":
        a = _lj_a(proto, rho_star)
        c = 1.633 * a
        # hcp in the orthohexagonal setting x [100] y [1 2 0] z [001]: the
        # ortho box (a, sqrt(3) a, c) holds 4 atoms -- two triangular-lattice
        # points per z-plane ((0,0) and e2) and the B-plane shifted by
        # (2a/3, a/sqrt(3), c/2).  Sites are enumerated over a generous
        # (m, n) range and folded into the box modulo the box vectors, which
        # are supercell vectors of the triangular lattice, then deduplicated
        # (a raw base+shift sum leaves B sites one cell outside the box).
        e1 = np.array([a, 0.0])
        e2 = np.array([0.5 * a, 3 ** 0.5 / 2 * a])
        b_shift = np.array([2.0 / 3.0 * a, a / 3 ** 0.5])
        Lx, Ly, Lz = rep * a, rep * 3 ** 0.5 * a, rep * c
        pos = []
        for k in range(rep):
            for fam, z in ((e1 * 0.0, 0.0), (b_shift, c / 2)):
                seen = set()
                for m in range(-2, 2 * rep + 2):
                    for n in range(-2, 2 * rep + 2):
                        p = np.mod(m * e1 + n * e2 + fam, (Lx, Ly))
                        key = (round(p[0], 6), round(p[1], 6))
                        if key in seen:
                            continue
                        seen.add(key)
                        pos.append([p[0], p[1], k * c + z])
                assert len(seen) == 2 * rep * rep, \
                    f"hcp plane holds {len(seen)} sites, want {2 * rep * rep}"
        pos = np.array(pos)
        L = np.array([Lx, Ly, Lz])
    else:
        from ase.build import bulk
        a = _lj_a(proto, rho_star)
        at = bulk("X", proto, a=a, cubic=True).repeat(rep)
        at.set_masses(np.full(len(at), 1.0))
        pos = at.positions
        L = at.cell.lengths()
    dnn = typical_neighbor_distance(Frame(pos=pos, cell=np.diag(L),
                                          symbols=["X"] * len(pos),
                                          pbc=(True,) * 3))
    if jitter_dnn:
        pos = pos + np.random.default_rng(seed).normal(
            0.0, jitter_dnn * dnn, pos.shape)
    return Frame(pos=np.mod(pos, L), cell=np.diag(L),
                 symbols=["X"] * len(pos), pbc=(True, True, True))


# ---- W1 done-when 1: the reproducer verdicts (0 of 9; argon accepted) -------

def test_reproducer_no_bench_crystal_is_amorphous(glass_dialect):
    """The W1 reproducer, first half: 0 of the 9 exact-roundtrip bench
    crystals may be called amorphous under core + glass (was 9 of 9)."""
    from chaord.lift.amorphous import is_amorphous
    bad = [c["id"] for c in acc._exact_roundtrip_cases(None)
           if is_amorphous(acc._a2a3_frame(c), glass_dialect)]
    assert bad == [], f"bench crystals called amorphous under core + glass: {bad}"


def test_reproducer_argon_gas_accepted_by_fluid_mode():
    """The W1 reproducer, second half: the dilute argon bench gas lifts with
    mode='fluid' (was refused: 'solid-like fraction too high')."""
    ar = acc.case_by_id("fluid/ar_gas_box25")
    frame = read_frame(ar["frames"][0])
    lift_frame(frame, load_dialect(ar["dialect"]), mode="fluid")  # must not raise


# ---- W1 done-when 2: crystals are not amorphous, 20 draws -------------------

@pytest.mark.parametrize("rho", LJ_RHOS)
@pytest.mark.parametrize("proto", LJ_PROTOS)
def test_lj_crystal_not_amorphous_glass_dialect(glass_dialect, proto, rho):
    """20 draws per (prototype, rho*): is_amorphous under core + glass is
    False on every draw (the q6/q4 rule scales with the frame's own d_NN, and
    the lattice fit catches the borderline lattices)."""
    from chaord.lift.amorphous import is_amorphous
    for draw in range(DRAWS):
        frame = _lj_crystal(proto, rho, 4, JITTER_DNN, seed=draw)
        assert not is_amorphous(frame, glass_dialect), \
            f"{proto} at rho*={rho} draw {draw} promotes as amorphous"


@pytest.mark.slow
@pytest.mark.parametrize("rho", LJ_RHOS)
@pytest.mark.parametrize("proto", LJ_PROTOS)
def test_lj_crystal_lift_prints_no_amorphous(glass_dialect, proto, rho):
    """20 draws per (prototype, rho*): the full lift under core + glass never
    prints an amorphous region (the cascade must reach a crystal arm or
    refuse, not fall through to the glass)."""
    for draw in range(DRAWS):
        frame = _lj_crystal(proto, rho, 4, JITTER_DNN, seed=draw)
        text = format_program(lift_frame(frame, glass_dialect))
        assert "amorphous" not in text, \
            f"{proto} at rho*={rho} draw {draw} lifts as amorphous"


def test_bench_crystal_lift_prints_no_amorphous(glass_dialect):
    """The 9 Angstrom bench crystals lift under core + glass without an
    amorphous region (their recorded dialect stack is checked too)."""
    for case in acc._exact_roundtrip_cases(None):
        frame = acc._a2a3_frame(case)
        for names in (("core", "glass"), case["dialect"]):
            text = format_program(lift_frame(frame, load_dialect(names)))
            assert "amorphous" not in text, \
                f"{case['id']} under {'+'.join(names)} lifts as amorphous"


# ---- W1 done-when 3: glasses stay amorphous, the liquid stays a fluid -------

def test_bench_rsa_glass_lifts_amorphous(glass_dialect):
    """The bench RSA glass frames stay amorphous under the unit-free rule."""
    case = acc.case_by_id("glass/lj_glass_rho085")
    for path in case["frames"]:
        text = format_program(lift_frame(read_frame(path), glass_dialect))
        assert "amorphous glass" in text, path


@pytest.mark.slow
def test_reference_glass_frames_amorphous(glass_dialect):
    """The 15 MD glass reference frames are amorphous (is_amorphous on every
    frame; the reviewer's matrix reads crystal-like 0.02-0.12 for them)."""
    from chaord.lift.amorphous import is_amorphous
    frames = sorted((REFERENCE / "lj_glass").glob("frame_*.npz"))
    assert len(frames) == 15
    for path in frames:
        frame = read_frame(path)
        assert is_amorphous(frame, glass_dialect), \
            f"{path.name} is no longer called amorphous"


def test_large_liquid_lifts_as_fluid():
    """The 2,048-atom liquid reference lifts as a fluid under core + lj."""
    frame = read_frame(REFERENCE / "lj_liquid_large" / "frame_0.npz")
    text = format_program(
        lift_frame(frame, load_dialect(("core", "lj")), mode="fluid"))
    assert "liquid fluid" in text
    assert "amorphous" not in text


# ---- W1 done-when 4: the molecular fluid modes pass --------------------------

@pytest.mark.parametrize("cid", ["fluid/ar_gas_box25", "fluid/n2_box22",
                                 "gases/co2_dense", "fluid/water_box15"])
def test_fluid_mode_accepts_the_fluid_cases(cid):
    """mode='fluid' lifts every named fluid/gas case (so A13 records no
    routing fallback for them)."""
    case = acc.case_by_id(cid)
    lift_frame(read_frame(case["frames"][0]),
               load_dialect(case["dialect"]), mode="fluid")  # must not raise


# ---- W1 step 2: the local-order rule itself ---------------------------------

def test_gas_atoms_with_few_neighbours_are_not_crystal_like():
    """An atom with 1-2 neighbours has q6bar near 1 but must not count as
    crystal-like: the unit-free rule requires q_min_neighbours first."""
    from chaord.lift.amorphous import crystal_like_fraction
    rng = np.random.default_rng(0)
    # 50 widely spaced atoms in a 25 A box: every atom has ~1-3 neighbours
    L = np.array([25.0, 25.0, 25.0])
    frame = Frame(pos=rng.uniform(0, 25, (50, 3)), cell=np.diag(L),
                  symbols=["Ar"] * 50, pbc=(True,) * 3)
    dialect = load_dialect(("core", "molecular"))
    frac = crystal_like_fraction(frame, dialect)
    assert frac < float(dialect.threshold("amorphous_solid_frac_max"))


# ---- W1 steps 3-4: fail closed ----------------------------------------------

def _stripped_dialect():
    """A core dialect with no thresholds at all: every lookup must raise."""
    return Dialect(("core",), {"thresholds": {}}, {"core": "0"})


def test_is_amorphous_missing_threshold_raises():
    """W1 step 3: the old ``except Exception: disordered = True`` swallowed
    the missing-threshold error; now it propagates."""
    from chaord.lift.amorphous import is_amorphous
    rng = np.random.default_rng(1)
    frame = Frame(pos=rng.uniform(0, 8, (200, 3)), cell=np.diag([8.0] * 3),
                  symbols=["X"] * 200, pbc=(True,) * 3)
    with pytest.raises(ChaordError, match="defines no threshold"):
        is_amorphous(frame, _stripped_dialect())


def test_legacy_missing_threshold_propagates():
    """W1 step 4: a dialect without the thresholds an arm needs is a
    configuration error; the cascade must raise, never launder it into a
    program (pre-fix: the frame below lifted as a fluid program)."""
    from chaord.lift.legacy import legacy_lift
    frame = read_frame(acc.case_by_id("crystals/fcc_cu")["frames"][0])
    with pytest.raises(ChaordError, match="defines no threshold"):
        legacy_lift(frame, _stripped_dialect(), None, "auto")


def test_legacy_arm_error_propagates(monkeypatch):
    """Only NotThisPhase (a genuine phase refusal) lets an arm fall through;
    any other error -- here a planted RuntimeError in the crystal arm --
    propagates out of the cascade."""
    from chaord.lift import legacy
    frame = read_frame(acc.case_by_id("crystals/fcc_cu")["frames"][0])
    dialect = load_dialect(("core", "metal"))

    def boom(frame, dialect):
        raise RuntimeError("planted engine failure")

    monkeypatch.setattr(legacy, "lift_crystal", boom)
    with pytest.raises(RuntimeError, match="planted engine failure"):
        legacy.legacy_lift(frame, dialect, None, "auto")


def test_legacy_refusal_still_falls_through(monkeypatch):
    """Positive control: a ChaordError refusal (the arms' not-this-phase
    channel) still falls through to the next arm and lifts."""
    from chaord.lift import legacy
    frame = read_frame(acc.case_by_id("crystals/fcc_cu")["frames"][0])
    dialect = load_dialect(("core", "metal"))

    def refuse(frame, dialect):
        raise ChaordError("not this phase: planted refusal")

    monkeypatch.setattr(legacy, "lift_crystal", refuse)
    program, unexplained = legacy.legacy_lift(frame, dialect, None, "auto")
    assert "amorphous" not in format_program(program)


def test_legacy_not_this_phase_exception_falls_through(monkeypatch):
    """An arm that raises NotThisPhase itself (the forward-looking contract)
    falls through; the exception is the ladder's one fall-through key."""
    from chaord.lift import legacy
    from chaord.lift.legacy import NotThisPhase
    frame = read_frame(acc.case_by_id("crystals/fcc_cu")["frames"][0])
    dialect = load_dialect(("core", "metal"))

    def refuse(frame, dialect):
        raise NotThisPhase("planted forward-looking refusal")

    monkeypatch.setattr(legacy, "lift_crystal", refuse)
    program, _ = legacy.legacy_lift(frame, dialect, None, "auto")
    assert "amorphous" not in format_program(program)


def test_threshold_lookup_error_is_not_a_refusal(monkeypatch):
    """A ChaordError raised BY a threshold lookup is a missing-configuration
    error even when it escapes from deep inside an arm: the cascade must not
    read it as 'not this phase'."""
    from chaord.lift import legacy
    frame = read_frame(acc.case_by_id("crystals/fcc_cu")["frames"][0])
    dialect = load_dialect(("core", "metal"))

    def missing_key(frame, dialect):
        raise ChaordError("dialect 'core + metal' defines no threshold 'x'")

    monkeypatch.setattr(legacy, "lift_crystal", missing_key)
    with pytest.raises(ChaordError, match="defines no threshold 'x'"):
        legacy.legacy_lift(frame, dialect, None, "auto")


# ---- W1 step 5: glass.yaml no longer states an absolute cutoff ---------------

def test_glass_dialect_has_no_absolute_q6_cutoff(glass_dialect):
    """D10: glass.yaml's q6_cutoff is deleted; the phase rule reads the
    unit-free keys (q_cutoff_factor, q_min_neighbours, q4_solid) instead."""
    import yaml
    text = (Path(chaord_dir()) / "glass.yaml").read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    assert "q6_cutoff" not in data["thresholds"]
    for key in ("q_cutoff_factor", "q_min_neighbours", "q4_solid"):
        assert glass_dialect.threshold(key) is not None


def chaord_dir():
    import chaord.dialects as d
    return str(Path(d.__file__).parent)
