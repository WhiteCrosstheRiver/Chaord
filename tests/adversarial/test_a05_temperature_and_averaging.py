"""A5 -- RED (known open item, quantified): wrong-temperature rebuilds PASS
the current gate, and neither the seed pair nor a 5-frame average closes it.

Numbers (lj_liquid, floor = max(mean, P90) from reports/noise_floors.json:
gr 0.107 / cn 0.076; gate 1.5x = gr 0.161 / cn 0.114; measured 2026-09-30,
rebuild = build_program(physics=True), exactly the A5 path):

  T          A5 rule (median of draws 7/13)      ratio vs floor   verdict
  0.52 (x0.8)  gr 0.095 / cn 0.053              x0.89 / x0.70    PASS  <- wrong
  0.65 (x1.0)  gr 0.077 / cn 0.045              x0.72 / x0.59    PASS
  0.8125(x1.25) gr 0.131 / cn 0.052             x1.22 / x0.68    PASS  <- wrong

Systematic (5-frame-averaged) distances: x0.8 -> gr 0.097, x1.0 -> 0.031,
x1.25 -> 0.078: the wrong-T systematic shift is 2.5-3.1x the correct-T one,
i.e. as large as the noise floor itself, and hides inside the single-frame
microstate noise the gate is calibrated to.

Seed / reference luck (measured spreads):
  * per-seed gr distance at correct T: 0.058-0.103 (factor 1.8);
  * reference-frame choice moves the seed-mean gr distance by up to 0.046
    (43% of the gr floor) at wrong T, 0.013 at correct T; the cn seed-mean
    moves by 0.033 (43% of the cn floor 0.076).

Averaging ablation (5 frames/5 draws): the split-half floor of the AVERAGED
statistic is gr ~0.071, so even a 5-frame-averaged gate at 1.5x (gr 0.106)
still passes the x1.25 systematic (0.078) -- with only 5 decorrelated frames
the averaged statistic is not tight enough either; ~20+ frames (noise ~
1/sqrt(k)) or a temperature-sensitive observable (kinetic T, energy) is
needed.  Also structural: the lifted program's `state T` comes from the
dialect's md_reference_T and is never measured, so the language cannot even
represent a wrong frame temperature.

The fast tests assert the cached experiment numbers (skipped if the cache is
absent); the slow test re-runs the x1.25 canary live through the exact A5
judging logic.
"""
import json
import re
from pathlib import Path

import numpy as np
import pytest

from tools import acceptance as acc

HERE = Path(__file__).parent
RESULTS = HERE / "a5_temperature_results.json"
AVG = HERE / "a5_averaging_results.json"


def _floors():
    fl = json.loads((acc.ROOT / "reports" / "noise_floors.json")
                    .read_text(encoding="utf-8"))["lj_liquid"]
    eff = acc._a5_effective_floor(fl)
    return eff


pytestmark = pytest.mark.skipif(
    not RESULTS.exists(),
    reason="cached experiment results not generated yet; run "
           "python tests/adversarial/a5_temperature_experiment.py")


def test_a5_wrong_temperature_rebuild_must_fail_the_gate():
    eff = _floors()
    rep = json.loads(RESULTS.read_text(encoding="utf-8"))
    for name in ("0.8x (T=0.52)", "1.25x (T=0.8125)"):
        t = rep["temperatures"][name]
        dist = t["a5_median_of_draws"]
        ok = all(dist[k] <= 1.5 * eff[k] for k in dist)
        assert not ok, (
            f"A5 RED: the {name} rebuild of lj_liquid must exceed 1.5x the "
            f"floor; cached measured distances {dist} vs floor {eff} "
            f"(systematic 5-frame average {t['systematic_avg5']})"
        )


def test_a5_wrong_t_systematic_exceeds_correct_t_by_2x():
    rep = json.loads(RESULTS.read_text(encoding="utf-8"))
    right = rep["temperatures"]["1x (T=0.65)"]["systematic_avg5"]["gr_rms"]
    for name in ("0.8x (T=0.52)", "1.25x (T=0.8125)"):
        wrong = rep["temperatures"][name]["systematic_avg5"]["gr_rms"]
        assert wrong >= 2.0 * right, (
            f"A5 RED: a gate that cannot separate {name} (systematic "
            f"{wrong:.3f}) from the correct temperature ({right:.3f}) by at "
            "least 2x is temperature-blind; today the ratio is "
            f"{wrong/right:.1f}x yet both PASS"
        )


def test_a5_reference_frame_choice_moves_distance_less_than_floor():
    """A single-frame gate judged against frames[0] only: the distance spread
    across the five reference frames must be small relative to the floor, or
    the verdict depends on which frame was picked."""
    rep = json.loads(RESULTS.read_text(encoding="utf-8"))
    eff = _floors()
    t = rep["temperatures"]["0.8x (T=0.52)"]
    per_frame = [s["vs_frame"] for s in t["seeds"]]
    means = [float(np.mean([pf[i]["gr_rms"] for pf in per_frame]))
             for i in range(5)]
    spread = max(means) - min(means)
    assert spread <= 0.25 * eff["gr_rms"], (
        f"A5 RED: choosing a different reference frame moves the gr distance "
        f"by {spread:.3f} ({spread/eff['gr_rms']:.0%} of the floor "
        f"{eff['gr_rms']:.3f}); the gate is judged on frames[0] alone "
        f"(per-frame means {[round(m,3) for m in means]})"
    )


def test_a5_averaged_gate_would_separate_wrong_temperature():
    if not AVG.exists():
        pytest.skip("averaging results not generated; run "
                    "python tests/adversarial/a5_averaging_experiment.py")
    avg = json.loads(AVG.read_text(encoding="utf-8"))
    rep = json.loads(RESULTS.read_text(encoding="utf-8"))
    gate = avg["gate_1p5x_floor_avg"]
    right = rep["temperatures"]["1x (T=0.65)"]["systematic_avg5"]
    wrong = rep["temperatures"]["1.25x (T=0.8125)"]["systematic_avg5"]
    assert (right["gr_rms"] <= gate["gr_rms"]
            and wrong["gr_rms"] > gate["gr_rms"]), (
        f"A5 RED (averaging ablation): with 5-frame-averaged observables the "
        f"gate 1.5x floor_avg {gate} does NOT separate correct T "
        f"({right['gr_rms']:.3f}) from x1.25 T ({wrong['gr_rms']:.3f}); "
        "either more decorrelated frames or a temperature-sensitive "
        "observable is required"
    )


@pytest.mark.slow
def test_a5_wrong_temperature_live_canary():
    """Live re-run of the x1.25 canary through the exact A5 pipeline
    (lift -> rewrite state T -> rebuild x2 draws -> distance vs floor)."""
    pytest.importorskip("chaord")
    from chaord.build import build_program
    from chaord.dialects import load_dialect
    from chaord.io.frames import read_frame
    from chaord.lift import lift_frame
    from chaord.lang.parser import parse_text
    from chaord.cv.noise import observables, distance

    dl = load_dialect(("core", "lj"))
    frame = read_frame(str(acc.ROOT / "bench/reference/lj_liquid/frame_0.npz"))
    text = acc.format_program_text(lift_frame(frame, dl))
    wrong = re.sub(r"(?m)^(\s*state T\s+)0.65", r"\g<1>0.8125", text, count=1)
    assert "state T 0.8125" in wrong
    obs_ref = observables(frame, dl)
    draws = []
    for seed in (7, 13):
        rb = build_program(parse_text(wrong), dl,
                           rng=np.random.default_rng(seed), physics=True)
        draws.append(distance(obs_ref, observables(rb, dl), dialect=dl))
    dist = {k: float(np.median([d[k] for d in draws])) for k in draws[0]}
    eff = _floors()
    ok = all(dist[k] <= 1.5 * eff[k] for k in dist)
    assert not ok, (
        f"A5 RED (live): x1.25-T rebuild of lj_liquid passes the gate "
        f"(distances {dist} vs floor {eff})"
    )
