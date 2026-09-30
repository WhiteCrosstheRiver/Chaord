"""A5 root cause (reference/lj_liquid_large): the lj fluid rebuild's
equilibration was sized for the 500-atom reference case and never revisited
when the 2048-atom case joined the bench.

The noise floor of a single-frame fluid observable tightens roughly as
1/sqrt(N): lj_liquid_large's cn_tv floor is 0.0174 against lj_liquid's 0.0602
(3.5x tighter), while the rebuild's equilibration was a fixed 2400 MD steps
(12 tau).  Erasing the RSA-packed start is a diffusive process, t ~ L^2, and
the cubic box edge grows as N^(1/3), so the step count must grow as
N^(2/3): 2400 x (2048/500)^(2/3) = 6144 steps for the large case.  The
dialect ships 6400 (fast + slow) -- flat in N on purpose; see the relax_steps
comment in src/chaord/dialects/lj.yaml for why the builder must not scale the
steps at runtime (the A5 verifier reads md_steps as a static dialect sum).
"""
import json
import time
from pathlib import Path

import numpy as np
import pytest

from chaord.build import build_program
from chaord.cv.noise import distance, observables
from chaord.dialects import load_dialect
from chaord.io.frames import read_frame
from chaord.lang.fmt import format_program
from chaord.lang.parser import parse_text
from chaord.lift import lift_frame

ROOT = Path(__file__).resolve().parents[1]

# the largest MD reference fluid the lj rebuild must equilibrate
# (bench/reference/lj_liquid_large, 2048 atoms -- A9's >=2000-atom case)
_N_LARGEST_REFERENCE = 2048

_TINY_LJ_LIQUID = (
    "chaord 0.1\n"
    "dialect core + lj\n"
    "\n"
    "system {\n"
    "  cell 4.6 4.6 4.6\n"
    "  pbc xyz\n"
    "  state T 0.70\n"
    "  conserve atoms X 60\n"
    "}\n"
    "\n"
    "physics {\n"
    "  backend lj\n"
    "}\n"
    "\n"
    "liquid bulk : all {\n"
    "}\n"
)


def _lj_md() -> dict:
    return load_dialect(("core", "lj")).threshold("md")


def test_default_equilibration_covers_diffusive_mixing():
    """Root-cause lock: the default (no md_steps override) equilibration must
    cover the largest MD reference case.  The basis is the dialect's own
    sizing parameters -- relax_steps_base steps equilibrate
    relax_steps_ref_n atoms, and diffusive mixing of the packed start costs
    (N/N_ref)^relax_steps_exp more -- so anything smaller reverts to the
    2400-step protocol that left the 2048-atom rebuild at 3.2x its cn_tv
    floor."""
    md = _lj_md()
    total = int(md["relax_steps_fast"]) + int(md["relax_steps"])
    base = int(md["relax_steps_base"])
    n_ref = float(md["relax_steps_ref_n"])
    exponent = float(md["relax_steps_exp"])
    required = base * (_N_LARGEST_REFERENCE / n_ref) ** exponent
    assert total >= required, (
        f"lj rebuild equilibration is {total} steps but diffusive mixing of "
        f"N={_N_LARGEST_REFERENCE} needs {required:.0f} "
        f"({base} steps at N_ref={n_ref:g}, exponent {exponent:g})")


def test_builder_runs_exactly_the_tables_md_steps(monkeypatch):
    """Honesty invariant: with no md_steps override the fluid builder runs
    exactly relax_steps_fast + relax_steps steps -- the arithmetic the A5
    verifier (_a5_rebuild_meta in tools/acceptance.py) reads off the dialect
    for a program without a history line.  If the builder ever ran more or
    fewer steps than the table sum, the verifier's md_steps metadata would
    stop describing the rebuild it scores."""
    import chaord.realize.lj as rlj

    calls = []

    def fake_run_md(r, v, L, nsteps, dt, T, gamma, rng, **kw):
        calls.append(dict(steps=int(nsteps), dt=float(dt),
                          T=float(T), gamma=float(gamma)))
        return r, v

    monkeypatch.setattr(rlj, "run_md", fake_run_md)
    md = _lj_md()
    build_program(parse_text(_TINY_LJ_LIQUID), load_dialect(("core", "lj")),
                  rng=np.random.default_rng(3), physics=True)
    assert len(calls) == 2, "fluid relax is exactly two stages: fast, slow"
    assert calls[0]["steps"] == int(md["relax_steps_fast"])
    assert calls[1]["steps"] == int(md["relax_steps"])
    assert calls[0]["dt"] == float(md["relax_dt_fast"])
    assert calls[1]["dt"] == float(md["relax_dt"])
    assert calls[0]["gamma"] == float(md["relax_gamma_fast"])
    assert calls[1]["gamma"] == float(md["relax_gamma"])
    assert calls[0]["T"] == pytest.approx(0.70)  # the program's state T
    assert (calls[0]["steps"] + calls[1]["steps"]
            == int(md["relax_steps_fast"]) + int(md["relax_steps"]))


@pytest.mark.slow
@pytest.mark.parametrize("case", ["lj_liquid", "lj_liquid_large"])
def test_lj_liquid_rebuild_within_150pct_floor(case):
    """Reproduction of the A5 distance failure: rebuild the reference frame
    exactly as the acceptance harness does (lift frame_0 -> build with rng
    seed 7, dialect-default md_steps) and hold every observable within 1.5x
    the measured noise floor.

    Before the protocol fix reference/lj_liquid_large failed on cn_tv:
    0.0557 against a 0.0174 floor (3.2x; the 1.5x gate is 0.0261) while
    gr_rms 0.0442 already passed.  lj_liquid (500 atoms) is the guard
    against regressing the case the old 2400-step protocol was sized for."""
    floors = json.loads(
        (ROOT / "reports" / "noise_floors.json").read_text(encoding="utf-8"))
    fl = floors[case]
    dl = load_dialect(("core", "lj"))
    frame = read_frame(ROOT / "bench" / "reference" / case / "frame_0.npz")
    text = format_program(lift_frame(frame, dl))
    t0 = time.perf_counter()
    rebuilt = build_program(parse_text(text), dl,
                            rng=np.random.default_rng(7), physics=True)
    secs = time.perf_counter() - t0
    dist = distance(observables(frame, dl), observables(rebuilt, dl),
                    dialect=dl)
    detail = "; ".join(
        f"{k} {dist[k]:.4f} vs floor {max(float(fl[f'{k}_mean']), float(fl.get(f'{k}_q90', 0.0))):.4f} "
        f"(x{dist[k] / max(float(fl[f'{k}_mean']), float(fl.get(f'{k}_q90', 1e-9))):.2f})"
        for k in ("gr_rms", "cn_tv"))
    for k in ("gr_rms", "cn_tv"):
        # same effective floor as the acceptance criterion (A5,
        # tools/acceptance.py::_a5_effective_floor): max(mean, P90 of the
        # measured pairs). A single rebuild draw is one sample of the
        # frame-pair distance distribution (measured; the pair max is
        # ~1.5x the pair mean), so a mean-only gate rejects ~20% of
        # perfectly equilibrated draws by construction -- the 2026-09-30
        # clean machine hit exactly that on liquid_large (draw 0.0337 vs
        # mean floor 0.0208, inside the acceptance gate 1.5x q90 = 0.0439)
        floor = max(float(fl[f"{k}_mean"]), float(fl.get(f"{k}_q90", 0.0)))
        assert dist[k] <= 1.5 * floor, (
            f"{case}: {k} {dist[k]:.4f} > 1.5x floor {floor:.4f} "
            f"(rebuild {secs:.1f} s; {detail})")
