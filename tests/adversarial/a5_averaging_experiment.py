"""Red-team follow-up for A5: would an averaged-observable gate separate a
wrong-temperature rebuild from a correct one?

The current gate compares ONE reference frame against ONE rebuild draw
(median of 2 draws).  The temperature experiment (a5_temperature_experiment)
showed both 0.8x and 1.25x-T rebuilds PASS that gate while their systematic
(5-frame-averaged) distance is 2.5-3.1x the correct-temperature systematic.

This script measures the noise floor of the AVERAGED statistic itself, so a
concrete repair direction can be priced:
  floor_avg ~ distance(mean(ref frames A), mean(ref frames B))  (split-half
  of the 5 reference frames) -- and the same split over rebuild draws.

Run: python tests/adversarial/a5_averaging_experiment.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

RESULTS = Path(__file__).with_name("a5_averaging_results.json")


def mean_obs(obs_list):
    out = {}
    for k in obs_list[0]:
        out[k] = np.array([o[k] for o in obs_list]).mean(axis=0)
    return out


def run(force=False):
    if RESULTS.exists() and not force:
        return json.loads(RESULTS.read_text(encoding="utf-8"))
    from chaord.build import build_program
    from chaord.dialects import load_dialect
    from chaord.io.frames import read_frame
    from chaord.lift import lift_frame
    from chaord.lang.parser import parse_text
    from chaord.cv.noise import observables, distance
    from tools import acceptance as acc

    dl = load_dialect(("core", "lj"))
    frames = [read_frame(str(p)) for p in
              sorted((ROOT / "bench" / "reference" / "lj_liquid").glob("frame_*.npz"))]
    obs_ref = [observables(f, dl) for f in frames]
    text0 = acc.format_program_text(lift_frame(frames[0], dl))
    floors = json.loads((ROOT / "reports" / "noise_floors.json")
                        .read_text(encoding="utf-8"))["lj_liquid"]
    fl = {k: max(floors[f"{k}_mean"], floors[f"{k}_q90"]) for k in ("gr_rms", "cn_tv")}

    t0 = time.perf_counter()
    obs_rb = []
    for seed in (7, 13, 101, 202, 303):
        rb = build_program(parse_text(text0), dl,
                           rng=np.random.default_rng(seed), physics=True)
        obs_rb.append(observables(rb, dl))
    build_s = time.perf_counter() - t0

    rep = {"build_seconds": round(build_s, 1), "single_frame_floor": fl}
    # noise of the averaged statistic: split-half of the reference frames
    # (5 frames -> 2+3 two ways) and split-half of the rebuild draws
    splits_ref = [([0, 1], [2, 3, 4]), ([0, 1, 2], [3, 4])]
    ref_halves = [distance(mean_obs([obs_ref[i] for i in a]),
                            mean_obs([obs_ref[i] for i in b]), dialect=dl)
                  for a, b in splits_ref]
    rb_halves = [distance(mean_obs([obs_rb[i] for i in a]),
                          mean_obs([obs_rb[i] for i in b]), dialect=dl)
                 for a, b in splits_ref]
    # cross: average reference half vs average rebuild half (equilibrated
    # independent draws -> this is what a perfect rebuild's distance is)
    cross = [distance(mean_obs([obs_ref[i] for i in a]),
                      mean_obs([obs_rb[i] for i in b]), dialect=dl)
             for a, b in splits_ref]
    rep["ref_splithalf"] = ref_halves
    rep["rebuild_splithalf"] = rb_halves
    rep["cross_splithalf"] = cross
    pool = [d for d in ref_halves + rb_halves + cross]
    floor_avg = {k: max(d[k] for d in pool) for k in pool[0]}
    rep["floor_avg_estimate_max"] = floor_avg
    rep["gate_1p5x_floor_avg"] = {k: 1.5 * v for k, v in floor_avg.items()}

    # verdicts under the averaged gate, using the systematic distances
    # measured by the temperature experiment
    temp = json.loads(Path(__file__).with_name("a5_temperature_results.json")
                      .read_text(encoding="utf-8"))
    verdicts = {}
    for name, t in temp["temperatures"].items():
        sysd = t["systematic_avg5"]
        verdicts[name] = {
            "systematic": sysd,
            "pass_averaged_gate": all(sysd[k] <= rep["gate_1p5x_floor_avg"][k]
                                      for k in sysd),
        }
    rep["verdicts_under_averaged_gate"] = verdicts
    RESULTS.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    return rep


if __name__ == "__main__":
    r = run(force="--force" in sys.argv)
    for k in ("single_frame_floor", "ref_splithalf", "rebuild_splithalf",
              "cross_splithalf", "floor_avg_estimate_max",
              "gate_1p5x_floor_avg", "verdicts_under_averaged_gate"):
        print(k, "=", json.dumps(r[k]))
