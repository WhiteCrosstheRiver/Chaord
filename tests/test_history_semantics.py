"""X5 (Review 10, D19): a history must mean what it says.

Three gaps between the stated history and the realised one, each with its
red-first test (the v3 brief's reproducers):

1. SILENT CAP -- a quench longer than quench_max_steps was cut to 20,000
   steps with no error; D19(b): the builder never caps or clamps a stated
   history, it raises a ChaordError naming the requested and the allowed
   step count (the max(rate, 1e-6) guard is gone with it).
2. THERMOSTAT LAG -- with Langevin gamma = relax_gamma the system cannot
   follow the default ramp (reviewer: kinetic T 0.280 at the end of a
   bath-0.1 quench, 0.254/0.151/0.104/0.100 after 100/500/2000/4000 anneal
   steps at N = 500 -- the anneal absorbed the lag); D19(c): ramps run with
   a thermostat that follows the ramp and provenance records the kinetic
   temperature at the end of every stage.
3. RATE PER STEP -- `at <rate>` was per MD step, so a program's meaning
   depended on relax_dt; D19(a): rates are per TIME UNIT (T*/tau in LJ
   reduced units), with a one-time conversion of the printed rates.

D19(d): one protocol runner serves every engine (run_protocol grew the
mixture engine parameter; _run_mixture_history is deleted).
"""
import re

import numpy as np
import pytest

from chaord.build.slab import rsa
from chaord.dialects import load_dialect
from chaord.io.frames import Frame
from chaord.lang.errors import ChaordError
import chaord.realize.protocols as prot
from chaord.realize.protocols import run_protocol


@pytest.fixture(scope="module")
def dialect():
    return load_dialect(("core", "glass"))


def _frame(n, rho, seed, dialect):
    """Overlap-free protocol start, the same way build_amorphous prepares one."""
    L = np.full(3, (n / rho) ** (1 / 3))
    rng = np.random.default_rng(seed)
    pos = rsa(np.zeros((0, 3)), L, L[0] / 2, L[0] / 2, n,
              float(dialect.threshold("fluid_rsa_dmin")), rng)
    return Frame(pos=pos, cell=np.diag(L), symbols=["X"] * n, pbc=(True,) * 3)


class _CountingMd:
    """Monkeypatch shim (the brief's reproducer pattern): count integration
    steps instead of integrating, per call."""

    def __init__(self):
        self.counts = []

    def __call__(self, r, v, L, nsteps, *args, **kwargs):
        self.counts.append(int(nsteps))
        return r, v


# ------------------------------------------------- gap 1: the silent cap --

def test_quench_over_cap_raises_naming_requested_and_allowed(dialect,
                                                             monkeypatch):
    """The brief's reproducer: a quench slower than the dialect allows must
    raise, naming the requested and the allowed step counts -- never run a
    truncated 20,000-step history as if it were the stated one."""
    counter = _CountingMd()
    monkeypatch.setattr(prot, "run_md", counter)
    frame = Frame(pos=np.random.default_rng(0).uniform(0, 4, (32, 3)),
                  cell=np.diag([4.0] * 3), symbols=["X"] * 32,
                  pbc=(True, True, True))
    dt = float(dialect.threshold("md")["relax_dt"])
    cap = int(dialect.threshold("md")["quench_max_steps"])
    rate = 1e-5                                   # T*/tau: 1.9/(rate*dt) steps
    requested = int(np.ceil(1.9 / (rate * dt)))   # 38,000,000 at dt = 0.005
    with pytest.raises(ChaordError, match=rf"{requested}.*{cap}") as ei:
        run_protocol(frame, [("melt", 2.0, 10), ("quench", 0.1, 0.1, rate)],
                     dialect, np.random.default_rng(0))
    assert "quench" in str(ei.value)


def test_quench_exactly_at_cap_runs_it_all(dialect, monkeypatch):
    """A quench whose stated rate needs exactly quench_max_steps runs every
    one of them -- the cap is a refusal bound, not a truncation."""
    counter = _CountingMd()
    monkeypatch.setattr(prot, "run_md", counter)
    frame = Frame(pos=np.zeros((8, 3)), cell=np.diag([2.0] * 3),
                  symbols=["X"] * 8, pbc=(True,) * 3)
    dt = float(dialect.threshold("md")["relax_dt"])
    cap = int(dialect.threshold("md")["quench_max_steps"])
    rate = 1.9 / (cap * dt)                       # exactly cap steps
    out = run_protocol(frame, [("melt", 2.0, 4), ("quench", 0.1, 0.1, rate)],
                       dialect, np.random.default_rng(0))
    assert len(out) == 8
    assert sum(counter.counts[1:]) == cap         # nothing truncated


def test_quench_nonpositive_rate_refuses(dialect, monkeypatch):
    """The max(rate, 1e-6) guard is gone (D19(b)): a zero rate is a statement
    the runner cannot honour and must refuse, not clamp into 20,000 steps."""
    counter = _CountingMd()
    monkeypatch.setattr(prot, "run_md", counter)
    frame = Frame(pos=np.zeros((8, 3)), cell=np.diag([2.0] * 3),
                  symbols=["X"] * 8, pbc=(True,) * 3)
    with pytest.raises(ChaordError, match="rate"):
        run_protocol(frame, [("melt", 2.0, 4), ("quench", 0.1, 0.1, 0.0)],
                     dialect, np.random.default_rng(0))


# ------------------------------------------- gap 3: the rate is per tau --

def test_quench_step_count_uses_rate_per_time_unit(dialect, monkeypatch):
    """D19(a): `at r` is T*/tau, so the MD steps are |dT| / (r * relax_dt) --
    a program keeps its meaning through the dt the dialect happens to state."""
    counter = _CountingMd()
    monkeypatch.setattr(prot, "run_md", counter)
    frame = Frame(pos=np.zeros((8, 3)), cell=np.diag([2.0] * 3),
                  symbols=["X"] * 8, pbc=(True,) * 3)
    dt = float(dialect.threshold("md")["relax_dt"])
    out = run_protocol(frame, [("melt", 2.0, 4), ("quench", 1.0, 0.1, 4.0)],
                       dialect, np.random.default_rng(0))
    assert len(out) == 8
    # 0.9 T* over 4 T*/tau at dt: exactly 0.9/(4*dt) integration steps
    assert sum(counter.counts[1:]) == int(np.ceil(0.9 / (4.0 * dt)))


def _lifter_default_steps(n, dialect):
    """The history the amorphous lifter states at size n: the dialect's
    glass_* base counts scaled by (n/glass_protocol_ref_n)^(2/3)."""
    ref_n = float(dialect.threshold("glass_protocol_ref_n"))
    scale = (n / ref_n) ** (2.0 / 3.0)  # dialect-exempt: exact-geometry: diffusive t_mix ~ N^(2/3)
    return (int(round(int(dialect.threshold("glass_melt_steps")) * scale)),
            int(round(int(dialect.threshold("glass_quench_steps")) * scale)),
            int(round(int(dialect.threshold("glass_anneal_steps")) * scale)))


@pytest.fixture(scope="module")
def default_glass_n500(dialect):
    """The default-protocol glass at the done-when size (N = 500): RSA start,
    then the history the lifter states there (base counts x (500/2000)^(2/3)
    = 630/1260/1260, rate in T*/tau)."""
    n_melt, n_q, n_a = _lifter_default_steps(500, dialect)
    t_melt = float(dialect.threshold("glass_melt_T"))
    t_q = float(dialect.threshold("glass_quench_T"))
    t_a = float(dialect.threshold("glass_anneal_T"))
    dt = float(dialect.threshold("md")["relax_dt"])
    rate = (t_melt - t_q) / (n_q * dt)
    steps = [("melt", t_melt, n_melt), ("quench", t_q, t_q, rate),
             ("anneal", t_a, n_a)]
    return run_protocol(_frame(500, 0.95, 2024, dialect), steps, dialect,
                        np.random.default_rng(7))


def test_printed_rate_is_per_tau(default_glass_n500, dialect):
    """The one-time conversion (D19(a)): the lifted history states the cooling
    rate in T*/tau -- (2.0 - 0.1)/(1260 steps x relax_dt), not the per-step
    0.0015079 it printed before."""
    from chaord.lang.fmt import format_program
    from chaord.lift import lift_frame
    text = format_program(lift_frame(default_glass_n500, dialect,
                                     mode="amorphous"))
    m = re.search(r"quench to ([\d.]+) at ([\d.eE+-]+)", text)
    assert m, f"no quench rate in the lifted program:\n{text}"
    n_melt, n_q, n_a = _lifter_default_steps(500, dialect)
    t_melt = float(dialect.threshold("glass_melt_T"))
    t_q = float(dialect.threshold("glass_quench_T"))
    dt = float(dialect.threshold("md")["relax_dt"])
    rate = float(m.group(2))
    assert rate == pytest.approx((t_melt - t_q) / (n_q * dt), rel=1e-3), (
        f"printed rate {rate} is not (T_melt - T_quench)/(steps x dt) = "
        f"{(t_melt - t_q) / (n_q * dt):g} T*/tau")


# ----------------------------------------- gap 2: the thermostat follows --

def test_stage_kinetic_temperatures_within_10pct(default_glass_n500, dialect):
    """The done-when of gap 2, at N = 500: the kinetic temperature at the
    end of EVERY default-protocol stage is within 10% of the stated one, and
    the provenance records it (D19(c)).  Before the fix the quench ended at
    T_kin 0.615 against a stated 0.1 and the anneal inherited 0.134 -- the
    anneal was absorbing the ramp lag (reviewer: 0.280 on the reference
    quench)."""
    stages = default_glass_n500.info.get("history_stages")
    assert stages, "run_protocol records no kinetic temperature per stage"
    stated = {"melt": float(dialect.threshold("glass_melt_T")),
              "quench": float(dialect.threshold("glass_quench_T")),
              "anneal": float(dialect.threshold("glass_anneal_T"))}
    seen = {}
    for rec in stages:
        seen.setdefault(rec["stage"], rec)   # the END record of each stage
    assert set(seen) == {"melt", "quench", "anneal"}
    for stage, rec in seen.items():
        assert rec["T"] == pytest.approx(stated[stage]), rec
        assert abs(rec["T_kin"] - stated[stage]) <= 0.10 * stated[stage], (
            f"{stage}: kinetic temperature {rec['T_kin']:.4f} is not within "
            f"10% of the stated {stated[stage]:g}")


def test_ramp_gamma_stronger_than_relax_gamma(default_glass_n500, dialect):
    """D19(c)'s mechanism, visible in the record: melt and quench ran with a
    Langevin gamma at least ramp_gamma_tau_multiple / (stage duration in
    tau) and, on a ramp, ramp_gamma_rate_multiple x |dT/dt| -- never the
    equilibration-only relax_gamma 0.5."""
    md = dialect.threshold("md")
    dt = float(md["relax_dt"])
    tau_mult = float(dialect.threshold("ramp_gamma_tau_multiple"))
    rate_mult = float(dialect.threshold("ramp_gamma_rate_multiple"))
    n_melt, n_q, _n_a = _lifter_default_steps(500, dialect)
    t_melt = float(dialect.threshold("glass_melt_T"))
    t_q = float(dialect.threshold("glass_quench_T"))
    floors = {"melt": tau_mult / (n_melt * dt),
              "quench": max(tau_mult / (n_q * dt),
                            rate_mult * (t_melt - t_q) / (n_q * dt))}
    for rec in default_glass_n500.info["history_stages"]:
        if rec["stage"] in floors:
            assert rec["gamma"] >= floors[rec["stage"]] - 1e-9, rec


# --------------------------------------------- D19(d): one runner, one API --

def test_run_protocol_serves_the_mixture_engine():
    """D19(d): the merged runner executes a mixture history when handed the
    pair-resolved engine -- same semantics (fail-closed cap, per-tau rates,
    stage temperature records) as the monatomic path."""
    from chaord.realize.lj import LJMixture
    dl = load_dialect(("core", "glass", "lj_mixtures"))
    entry = dl.threshold("lj_mixtures")["kob_andersen"]
    L = np.full(3, (108 / 1.2) ** (1 / 3))
    rng = np.random.default_rng(11)
    k = 5                                   # 125 sites > 108 atoms
    sites = (np.array(np.meshgrid(*[np.arange(k)] * 3, indexing="ij"))
             .reshape(3, -1).T + 0.5) * (L / k)
    pos = sites[rng.choice(len(sites), 108, replace=False)]
    symbols = ["A"] * 86 + ["B"] * 22
    rng.shuffle(symbols)
    frame = Frame(pos=pos, cell=np.diag(L), symbols=symbols,
                  pbc=(True,) * 3)
    mix = LJMixture(L, symbols, entry["epsilon"], entry["sigma"],
                    float(entry["rc_factor"]), skin=float(
                        dl.threshold("md")["skin"]))
    out = run_protocol(frame, [("melt", 2.0, 60), ("quench", 0.1, 0.1, 1.9),
                               ("anneal", 0.1, 40)], dl,
                       np.random.default_rng(3), engine=mix)
    assert out.symbols.count("A") == 86 and out.symbols.count("B") == 22
    assert np.isfinite(out.pos).all()
    stages = out.info.get("history_stages")
    assert stages and {s["stage"] for s in stages} == {"melt", "quench",
                                                       "anneal"}


def test_mixture_history_runs_through_the_one_runner(tmp_path):
    """The duplicate is gone: build.amorphous has no private mixture-history
    runner any more (D19(d): one protocol runner serves every engine)."""
    import chaord.build.amorphous as amod
    assert not hasattr(amod, "_run_mixture_history"), (
        "_run_mixture_history survives: D19(d) merges it into "
        "realize.protocols.run_protocol")


def test_merged_runner_still_refuses_unknown_backend(dialect):
    frame = Frame(pos=np.zeros((4, 3)), cell=np.diag([2.0] * 3),
                  symbols=["X"] * 4, pbc=(True,) * 3)
    with pytest.raises(NotImplementedError):
        run_protocol(frame, [("melt", 1.0, 2)], dialect,
                     np.random.default_rng(0), backend="mace")
