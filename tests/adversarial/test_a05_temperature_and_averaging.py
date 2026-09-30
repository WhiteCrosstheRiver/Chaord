"""A5 -- wrong-temperature rebuilds vs the gate, from the red-team F3
experiment cache (regenerated 2026-09-30 under the review-3 averaged
protocol on lj_liquid_large, the case that carries the criterion's
temperature evidence).

Pre-fix state (500-atom lj_liquid, single-frame statistic, floor =
max(mean, P90) gr 0.107 / cn 0.076, gate 1.5x):

  T           A5 rule (median of draws 7/13)   ratio vs floor   verdict
  0.52 (x0.8)  gr 0.095 / cn 0.053             x0.89 / x0.70    PASS <- wrong
  0.65 (x1.0)  gr 0.077 / cn 0.045             x0.72 / x0.59    PASS
  0.8125(x1.25) gr 0.131 / cn 0.052            x1.22 / x0.68    PASS <- wrong

Post-fix protocol (check_a5 since review 3): the reference side is the MEAN
of the per-frame observables over the floor record's ref_frames (10 frames
here), the rebuild side the mean of THREE independent draws (seeds 7/13/29),
the lift states the PROVENANCE temperature T*=0.72 (never the 0.65 dialect
default), and the same wrong-temperature power mutations (temp_lo / temp_hi
= state T rewritten to 0.8x / 1.25x of the provenance temperature) must FAIL
the 1.5x gate.  Measured numbers live in a5_temperature_results.json
(regenerate: python tests/adversarial/a5_temperature_experiment.py).

Honest margin, kept on record (reports/redteam_findings.md, F3): the
500-atom lj_liquid case still does not separate -- under this protocol
x0.8 sits at ~x1.1 the floor (passes) and x1.25 at ~x1.6 with a 5% gate
margin smaller than the documented runner ISA/BLAS draw divergence; the
criterion's temperature evidence comes from the >= 2,000-atom case, and
tests/acceptance/test_mutations.py exercises the temp mutations on it.

The fast tests assert the cached experiment numbers (skipped if the cache
is absent); the slow test re-runs both temperature canaries live through
the exact A5 judging path.
"""
import json
import re
from pathlib import Path

import numpy as np
import pytest

from tools import acceptance as acc

HERE = Path(__file__).parent
RESULTS = HERE / "a5_temperature_results.json"


def _cache():
    return json.loads(RESULTS.read_text(encoding="utf-8"))


def _floors(rep):
    return rep["floor_effective"]


pytestmark = pytest.mark.skipif(
    not RESULTS.exists(),
    reason="cached experiment results not generated yet; run "
           "python tests/adversarial/a5_temperature_experiment.py")


def test_a5_wrong_temperature_rebuild_must_fail_the_gate():
    rep = _cache()
    eff = _floors(rep)
    for name, t in rep["temperatures"].items():
        if name.startswith("1x"):
            continue
        dist = t["systematic_avg5"]
        ok = all(dist[k] <= 1.5 * eff[k] for k in dist)
        assert not ok and not t["a5_verdict_pass"], (
            f"A5 RED: the {name} rebuild of {rep['case']} must exceed 1.5x "
            f"the averaged floor; cached measured distances {dist} vs floor "
            f"{eff} (ratios {t['worst_ratio']})"
        )


def test_a5_wrong_t_systematic_exceeds_correct_t_by_2x():
    rep = _cache()
    right = rep["temperatures"]["1x (T=0.72)"]["systematic_avg5"]["gr_rms"]
    for name, t in rep["temperatures"].items():
        if name.startswith("1x"):
            continue
        wrong = t["systematic_avg5"]["gr_rms"]
        assert wrong >= 2.0 * right, (
            f"A5 RED: a gate that cannot separate {name} (systematic "
            f"{wrong:.3f}) from the correct temperature ({right:.3f}) by at "
            "least 2x is temperature-blind; today the ratio is "
            f"{wrong/right:.1f}x yet both would sit inside the gate"
        )


def test_a5_reference_frame_choice_moves_distance_less_than_floor():
    """Averaged protocol: judging the same rebuild draws against the mean of
    frames 0-4 vs the mean of frames 5-9 (the floor record's own disjoint
    5v5 split) must not move the gr distance by more than a quarter of the
    floor, or the verdict depends on which half of the trajectory the
    reference was drawn from (the pre-fix single-frame gate moved by 43% of
    the floor -- F3)."""
    rep = _cache()
    eff = _floors(rep)
    for name, t in rep["temperatures"].items():
        spread = t["ref_half_sensitivity"]["spread"]["gr_rms"]
        assert spread <= 0.25 * eff["gr_rms"], (
            f"A5 RED: judging {name} against a different reference half "
            f"moves the gr distance by {spread:.3f} "
            f"({spread / eff['gr_rms']:.0%} of the floor "
            f"{eff['gr_rms']:.3f}); halves "
            f"{t['ref_half_sensitivity']['first']['gr_rms']:.3f} vs "
            f"{t['ref_half_sensitivity']['last']['gr_rms']:.3f}"
        )


def test_a5_averaged_gate_would_separate_wrong_temperature():
    """The gated statistic itself (averaged reference vs averaged rebuild)
    must separate the correct temperature from both wrong ones: correct T
    inside the 1.5x gate, every wrong T outside it."""
    rep = _cache()
    gate = rep["gate_1p5x"]
    right = rep["temperatures"]["1x (T=0.72)"]["systematic_avg5"]
    assert all(right[k] <= gate[k] for k in right), (
        f"A5 RED: the correct-temperature rebuild sits OUTSIDE the averaged "
        f"gate {gate} ({right}) -- the gate is miscalibrated"
    )
    for name, t in rep["temperatures"].items():
        if name.startswith("1x"):
            continue
        wrong = t["systematic_avg5"]
        assert any(wrong[k] > gate[k] for k in wrong), (
            f"A5 RED: with the averaged observables the gate 1.5x floor "
            f"{gate} does NOT separate correct T ({right['gr_rms']:.3f}) "
            f"from {name} ({wrong['gr_rms']:.3f} gr / {wrong['cn_tv']:.3f} "
            "cn); either more decorrelated frames or a temperature-"
            "sensitive observable is required"
        )


@pytest.mark.slow
def test_a5_wrong_temperature_live_canary():
    """Live re-run of both temperature power mutations through the exact A5
    pipeline (lift at the provenance temperature -> rewrite state T to
    0.8x / 1.25x -> 3-draw averaged rebuild -> distance vs the averaged
    floor) on lj_liquid_large; the same case clean must pass.  This is the
    criterion's temperature evidence (review 3 / red-team F3)."""
    clean = acc.check_a5(case_filter="lj_liquid_large")
    row = next(r for r in clean["details"]["rows"]
               if r["case"] == "reference/lj_liquid_large")
    assert row["status"] == "pass", row["note"]
    assert row["temperature"] == pytest.approx(0.72)   # provenance T*
    assert row["seeds"] == [7, 13, 29]
    assert len(row["ref_frames"]) >= 5

    for mutation, tag in (("temp_lo", "MUTATED(T x0.8)"),
                          ("temp_hi", "MUTATED(T x1.25)")):
        bad = acc.check_a5(mutation=mutation, case_filter="lj_liquid_large")
        row = next(r for r in bad["details"]["rows"]
                   if r["case"] == "reference/lj_liquid_large")
        assert row["status"] == "fail", row["note"]
        assert tag in row["note"], row["note"]
        assert row["temperature"] != pytest.approx(0.72)  # the T really moved
        assert not bad["passed"], bad["evidence"]
