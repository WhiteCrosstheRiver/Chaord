"""Red-team experiment for A5 (statistical round trip): wrong-temperature
rebuilds, seed luck and the missing averaging of observables.

Question (known open item, quantified here): the gate compares ONE reference
frame against ONE rebuilt frame per draw and takes the median of 2 draws
(seeds 7/13).  How much systematic error can hide inside that single-frame
microstate noise?

Protocol (mirrors tools/acceptance.py check_a5 exactly where it matters):
  * lift bench/reference/lj_liquid/frame_0.npz with dialect core+lj;
  * rewrite `state T 0.65` in the lifted text to 0.8x and 1.25x of it;
  * rebuild with build_program(..., physics=True) for seeds 7, 13, 101, 202,
    303 (the first two are exactly A5's draws);
  * distances measured with chaord.cv.noise.observables/distance (the A5
    observable path for reference cases), against frame_0 and the other four
    reference frames;
  * verdicts against the CURRENT floor on record (max(mean, P90) of the
    decorrelated pairs in reports/noise_floors.json), gate 1.5x, PLAN's.

Also quantifies the averaging ablation:
  * single-frame statistic: distance(ref_frame_i, rebuild_seed_j) spread;
  * 5-frame-averaged statistic: distance(mean over 5 ref frames,
    mean over 5 rebuild observables) = the systematic component with the
    ~1/sqrt(5) microstate sampling noise removed.

Run standalone:  python tests/adversarial/a5_temperature_experiment.py
(the numbers land in tests/adversarial/a5_temperature_results.json; the
pytest module test_a5_open_items.py re-verifies them cheaply from the cache
and re-runs a reduced live version marked slow).
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

RESULTS = Path(__file__).with_name("a5_temperature_results.json")
SEEDS = (7, 13, 101, 202, 303)
A5_DRAWS = (7, 13)          # exactly the seeds tools/acceptance.py uses
CASE = "lj_liquid"
T_BASE = 0.65
T_FACTORS = (0.8, 1.0, 1.25)


def effective_floor(fl: dict) -> dict:
    """max(mean, P90 of the recorded pairs) -- copied from acceptance._a5_effective_floor."""
    def q90(vals):
        s = sorted(vals)
        return s[max(0, int(np.ceil(0.9 * len(s))) - 1)]
    out = {}
    for k in ("gr_rms", "cn_tv"):
        mean = float(fl[f"{k}_mean"])
        vals = [p[k] for p in fl.get("pairs", []) if k in p]
        out[k] = float(max(mean, q90(vals))) if vals else mean
    return out


def mean_observables(obs_list: list[dict]) -> dict:
    """Average every numeric-array key over frames (same bins: NVT case)."""
    out = {}
    for k in obs_list[0]:
        arr = np.array([o[k] for o in obs_list])
        out[k] = arr.mean(axis=0)
    return out


def run(force: bool = False) -> dict:
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
              sorted((ROOT / "bench" / "reference" / CASE).glob("frame_*.npz"))]
    text0 = acc.format_program_text(lift_frame(frames[0], dl))
    obs_ref = [observables(f, dl) for f in frames]
    floors = json.loads((ROOT / "reports" / "noise_floors.json")
                        .read_text(encoding="utf-8"))[CASE]
    fl = effective_floor(floors)

    re_t = re.compile(r"(?m)^(\s*state T\s+)([-+0-9.eE]+)")
    report = {"case": CASE, "t_base": T_BASE, "seeds": list(SEEDS),
              "floor_effective": fl, "gate_1p5x": {k: 1.5 * v for k, v in fl.items()},
              "temperatures": {}}
    for fac in T_FACTORS:
        t_new = T_BASE * fac
        text = re_t.sub(lambda m: m.group(1) + f"{t_new:.4g}", text0, count=1)
        assert f"state T {t_new:.4g}" in text
        rows = []
        t0 = time.perf_counter()
        obs_rb = []
        for seed in SEEDS:
            rb = build_program(parse_text(text), dl,
                               rng=np.random.default_rng(seed), physics=True)
            o = observables(rb, dl)
            obs_rb.append(o)
            per_frame = [distance(obs_ref[i], o, dialect=dl) for i in range(5)]
            rows.append(dict(seed=seed, vs_frame=per_frame))
        build_s = time.perf_counter() - t0
        # the exact A5 verdict: median of the two official draws vs frame 0
        a5_draws = [distance(obs_ref[0], obs_rb[SEEDS.index(s)], dialect=dl)
                    for s in A5_DRAWS]
        a5_dist = {k: float(np.median([d[k] for d in a5_draws])) for k in a5_draws[0]}
        a5_ok = all(a5_dist[k] <= 1.5 * fl[k] for k in a5_dist)
        # averaging statistic: systematic distance, microstate noise averaged out
        avg_ref = mean_observables(obs_ref)
        avg_rb = mean_observables(obs_rb)
        sys_dist = distance(avg_ref, avg_rb, dialect=dl)
        # seed spread at the frame-0 comparison
        d0 = [r["vs_frame"][0] for r in rows]
        report["temperatures"][f"{fac:g}x (T={t_new:.4g})"] = dict(
            build_seconds=round(build_s, 1),
            seeds=rows,
            a5_median_of_draws=a5_dist,
            a5_verdict_pass=a5_ok,
            worst_ratio={k: a5_dist[k] / fl[k] for k in a5_dist},
            systematic_avg5={k: float(v) for k, v in sys_dist.items()},
            frame0_spread={k: dict(min=float(min(d[k] for d in d0)),
                                   max=float(max(d[k] for d in d0)),
                                   mean=float(np.mean([d[k] for d in d0])))
                           for k in d0[0]},
        )
    RESULTS.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def print_report(rep: dict):
    print(f"case {rep['case']}, base T {rep['t_base']}")
    print(f"effective floor {rep['floor_effective']}  "
          f"gate 1.5x {rep['gate_1p5x']}")
    for name, t in rep["temperatures"].items():
        print(f"\n== {name} ==")
        print(f"  A5 rule (median of draws 7/13): {t['a5_median_of_draws']}"
              f" -> {'PASS' if t['a5_verdict_pass'] else 'FAIL'}")
        print(f"  ratio vs floor: {t['worst_ratio']}")
        print(f"  5-frame-averaged systematic distance: {t['systematic_avg5']}")
        print(f"  frame0 per-seed spread: {t['frame0_spread']}")


if __name__ == "__main__":
    print_report(run(force="--force" in sys.argv))
