"""Red-team experiment for A5 (statistical round trip): wrong-temperature
rebuilds, reference-frame sensitivity and the averaged protocol's power.

Original question (F3, 2026-09-30, on the 500-atom lj_liquid case): the gate
compared ONE reference frame against ONE rebuilt frame per draw -- how much
systematic error hides inside that single-frame microstate noise?  Answer:
all of it (x0.8 and x1.25 temperature rebuilds both PASSED).

Protocol since review 3 / F3 (this module mirrors tools/acceptance.py
check_a5 exactly where it matters, on the case that carries the criterion's
temperature evidence):
  * case lj_liquid_large (N = 2,048, 10 stored decorrelated reference
    frames); lift frame_0 with dialect core+lj at the PROVENANCE
    temperature (T* = 0.72 from units.temperatures_K.equilibrium and
    epsilon = 1 eV; the lift states T=0.72, never the 0.65 dialect
    default);
  * rewrite `state T` in the lifted text to 0.8x / 1.0x / 1.25x of that;
  * rebuild with build_program(..., physics=True) for seeds 7, 13, 29
    (exactly A5's averaged-protocol draws);
  * judged statistic: distance(mean observables over the floor record's
    ref_frames, mean observables over the 3 draws) -- both sides averaged,
    gate 1.5 x max(mean, P90) of the `avg` floor record (the PLAN's factor,
    the review-3 protocol);
  * reference-frame sensitivity: judge the same draws against the mean of
    frames 0-4 vs the mean of frames 5-9 (the floor record's own 5v5 split)
    -- the verdict must not depend on which half of the trajectory the
    reference is drawn from.

The 500-atom lj_liquid case is kept as the measured HONEST MARGIN: under
this protocol it separates x1.25 by only ~5% of the gate (documented in
reports/redteam_findings.md, F3); the criterion's temperature evidence
comes from the >= 2,000-atom case.

Run standalone:  python tests/adversarial/a5_temperature_experiment.py
(the numbers land in tests/adversarial/a5_temperature_results.json; the
pytest module test_a05_temperature_and_averaging.py re-verifies them
cheaply from the cache and re-runs the live canary marked slow).
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
SEEDS = (7, 13, 29)          # exactly the seeds tools/acceptance.py draws
CASE = "lj_liquid_large"
T_FACTORS = (0.8, 1.0, 1.25)


def mean_observables(obs_list: list[dict]) -> dict:
    """Average every key over frames (same bins: NVT case) -- mirrors
    tools.acceptance._mean_observables."""
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
    case_dir = ROOT / "bench" / "reference" / CASE
    frames = [read_frame(str(p)) for p in
              sorted(case_dir.glob("frame_*.npz"),
                     key=lambda p: int(re.search(r"(\d+)$", p.stem).group(1)))]
    prov = json.loads((case_dir / "provenance.json").read_text(encoding="utf-8"))
    t_base = acc._provenance_tstar(prov)
    assert t_base is not None, "provenance carries no structured temperature"
    text0 = acc.format_program_text(
        lift_frame(frames[0], dl, T=t_base))          # states T = 0.72
    obs_ref = [observables(f, dl) for f in frames]
    floor_rec = json.loads((ROOT / "reports" / "noise_floors.json")
                           .read_text(encoding="utf-8"))[CASE]["avg"]
    ref_frames = floor_rec["ref_frames"]
    fl = acc._a5_effective_floor(floor_rec)

    re_t = re.compile(r"(?m)^(\s*state T\s+)([-+0-9.eE]+)")
    m = re_t.search(text0)
    assert m and abs(float(m.group(2)) - t_base) < 1e-6, (
        f"the lift must state the provenance temperature {t_base}, not "
        f"{m and m.group(2)}")
    report = {"case": CASE, "t_base": t_base,
              "t_base_source": "reference provenance (review 3, step 3)",
              "protocol": ("averaged (review 3 / red-team F3): reference = "
                           "mean observables over the floor record's "
                           f"ref_frames {ref_frames}; rebuild = mean over "
                           f"draws {list(SEEDS)}; floor = max(mean, P90) of "
                           "the `avg` record; gate 1.5x"),
              "seeds": list(SEEDS), "floor_effective": fl,
              "gate_1p5x": {k: 1.5 * v for k, v in fl.items()},
              "reference_halves": {"first5": list(range(5)),
                                   "last5": list(range(5, 10))},
              "temperatures": {}}
    for fac in T_FACTORS:
        t_new = t_base * fac
        text = re_t.sub(lambda mm: mm.group(1) + f"{t_new:.4g}", text0, count=1)
        assert f"state T {t_new:.4g}" in text
        rows, obs_rb = [], []
        t0 = time.perf_counter()
        for seed in SEEDS:
            rb = build_program(parse_text(text), dl,
                               rng=np.random.default_rng(seed), physics=True)
            o = observables(rb, dl)
            obs_rb.append(o)
            rows.append(dict(seed=seed,
                             vs_ref_mean=distance(
                                 mean_observables(
                                     [obs_ref[i] for i in ref_frames]),
                                 o, dialect=dl)))
        build_s = time.perf_counter() - t0
        # the exact A5 averaged statistic: mean ref vs mean of the draws
        o_ref = mean_observables([obs_ref[i] for i in ref_frames])
        o_rb = mean_observables(obs_rb)
        sys_dist = distance(o_ref, o_rb, dialect=dl)
        ok = all(sys_dist[k] <= 1.5 * fl[k] for k in sys_dist)
        # reference-side sensitivity: the same draws judged against each
        # disjoint 5-frame half of the trajectory
        halves = {}
        for name, idx in (("first5", range(5)), ("last5", range(5, 10))):
            o_half = mean_observables([obs_ref[i] for i in idx])
            halves[name] = distance(o_half, o_rb, dialect=dl)
        spread = {k: abs(halves["first5"][k] - halves["last5"][k])
                  for k in halves["first5"]}
        # median of the per-draw distances vs the averaged reference (the
        # old single-draw statistic's closest analogue, kept for context)
        med = {k: float(np.median([r["vs_ref_mean"][k] for r in rows]))
               for k in rows[0]["vs_ref_mean"]}
        report["temperatures"][f"{fac:g}x (T={t_new:.4g})"] = dict(
            build_seconds=round(build_s, 1),
            seeds=rows,
            a5_median_of_draws=med,
            a5_verdict_pass=ok,
            worst_ratio={k: sys_dist[k] / fl[k] for k in sys_dist},
            systematic_avg5={k: float(v) for k, v in sys_dist.items()},
            ref_half_sensitivity={"first": halves["first5"],
                                  "last": halves["last5"],
                                  "spread": spread},
        )
    RESULTS.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def print_report(rep: dict):
    print(f"case {rep['case']}, provenance T {rep['t_base']}")
    print(f"protocol: {rep['protocol']}")
    print(f"effective floor {rep['floor_effective']}  "
          f"gate 1.5x {rep['gate_1p5x']}")
    for name, t in rep["temperatures"].items():
        print(f"\n== {name} ==")
        print(f"  A5 averaged statistic: {t['systematic_avg5']}"
              f" -> {'PASS' if t['a5_verdict_pass'] else 'FAIL'}")
        print(f"  ratio vs floor: {t['worst_ratio']}")
        print(f"  per-draw median: {t['a5_median_of_draws']}")
        print(f"  reference-half sensitivity: {t['ref_half_sensitivity']}")


if __name__ == "__main__":
    print_report(run(force="--force" in sys.argv))
