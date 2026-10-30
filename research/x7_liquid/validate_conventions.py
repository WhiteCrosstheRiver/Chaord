"""Validate the chaord-free conventions against the committed floors.

Runs BEFORE any new data is generated (the ka_convergence V3 precedent):
every check must reproduce the committed reports/noise_floors.json numbers
from the committed bench frames (bench/reference/lj_liquid,
bench/reference/lj_liquid_large) with the code in liq_obs.py.  Exit code 1
on any mismatch -- the new-floor analysis is only trusted after this passes.

Checks:
  V0  fcc start construction reproduces the committed box lengths
      (lj_fcc convention, generate_reference.py).
  V1  lj_liquid: all 45 frame-pair distances bit-exact vs all_pairs.
  V2  lj_liquid: main floor stats (lag >= 4 subset, 21 pairs) bit-exact.
  V3  lj_liquid: averaged floor (rotated 5v5 half-splits, 10 pairs --
      5 rotations x 2 orderings, the committed O8 pad) bit-exact.
  V4  lj_liquid_large: frame-pair floor (within-trajectory lag >= 2 plus
      cross pairs of the decorrelated halves, 21 pairs) bit-exact.
  V5  lj_liquid_large: averaged floor (5v5 cross-trajectory mean pair +
      2v3 cyclic rotations within each trajectory, 11 pairs) bit-exact.
  V6  P90 estimator: np.percentile(v, 90) on the committed pair values
      reproduces every gr_rms_q90 / cn_tv_q90 on record.
  V7  D15 estimator on the committed avg floors reproduces the A5
      effective floors quoted in reports/acceptance_details.md
      (lj_liquid gr 0.047 / cn 0.041; large gr 0.033 / cn 0.017).

Usage:  python validate_conventions.py
Writes validation_report.json and prints PASS/FAIL per check.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from liq_obs import frame_observables, distance, pair_distances, d15
from liq_params import CASES, LAG_MIN, fcc_start, RHO_STAR

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
BENCH = REPO / "bench" / "reference"
FLOORS = REPO / "reports" / "noise_floors.json"


def load_npz(path: Path):
    with np.load(path) as z:
        return (np.asarray(z["r"], float), np.asarray(z["L"], float),
                [str(s) for s in z["symbols"]])


def load_case_obs(case: str, n_frames: int):
    obs = []
    for k in range(n_frames):
        pos, L, _ = load_npz(BENCH / case / f"frame_{k}.npz")
        obs.append(frame_observables(pos, L))
    return obs


def committed_pairs(rec: dict):
    """{(min,max): (gr_rms, cn_tv)} from a committed pair list."""
    return {tuple(sorted(p["frames"])): (p["gr_rms"], p["cn_tv"])
            for p in rec["pairs"]}


def check(name, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return {"name": name, "pass": bool(ok), "detail": detail}


def main() -> int:
    rec = json.loads(FLOORS.read_text())
    results = []

    # ---- V0: start construction -------------------------------------------
    pos5, L5 = fcc_start(5, 5, 5, RHO_STAR)
    pos8, L8 = fcc_start(8, 8, 8, RHO_STAR)
    _, Lc5, _ = load_npz(BENCH / "lj_liquid" / "frame_0.npz")
    _, Lc8, _ = load_npz(BENCH / "lj_liquid_large" / "frame_0.npz")
    ok = (len(pos5) == 500 and np.allclose(L5, Lc5, atol=0, rtol=0)
          and len(pos8) == 2048 and np.allclose(L8, Lc8, atol=0, rtol=0))
    results.append(check(
        "V0 fcc start box", ok,
        f"L(5x5x5)={L5[0]:.8f} vs committed {Lc5[0]:.8f}; "
        f"L(8x8x8)={L8[0]:.8f} vs committed {Lc8[0]:.8f}"))

    # ---- V1/V2/V3: lj_liquid ----------------------------------------------
    obs = load_case_obs("lj_liquid", 10)
    mine = pair_distances(obs)
    all_rec = rec["lj_liquid"]["all_pairs"]
    ref = committed_pairs(all_rec)
    bad = [k for k, (g, c) in ref.items()
           if k not in {(tuple(d["frames"])) for d in mine}
           or next((d for d in mine if tuple(d["frames"]) == k), None)
           is None
           or (d := next(d for d in mine if tuple(d["frames"]) == k)) is None
           or d["gr_rms"] != g or d["cn_tv"] != c]
    n_match = sum(1 for d in mine
                  if tuple(d["frames"]) in ref
                  and ref[tuple(d["frames"])] == (d["gr_rms"], d["cn_tv"]))
    results.append(check(
        "V1 lj_liquid all 45 pairs", n_match == 45 and len(ref) == 45,
        f"{n_match}/45 pair distances bit-exact vs all_pairs"))

    main_rec = rec["lj_liquid"]
    lag_pairs = [d for d in mine if d["frames"][1] - d["frames"][0] >= LAG_MIN]
    ok = (len(lag_pairs) == main_rec["n_pairs"]
          and max(d["gr_rms"] for d in lag_pairs) == main_rec["gr_rms_max"]
          and np.mean([d["gr_rms"] for d in lag_pairs]) == main_rec["gr_rms_mean"]
          and np.percentile([d["gr_rms"] for d in lag_pairs], 90)
          == main_rec["gr_rms_q90"]
          and np.mean([d["cn_tv"] for d in lag_pairs]) == main_rec["cn_tv_mean"]
          and np.percentile([d["cn_tv"] for d in lag_pairs], 90)
          == main_rec["cn_tv_q90"])
    results.append(check(
        "V2 lj_liquid main floor (lag>=4, 21 pairs)", ok,
        f"mean {np.mean([d['gr_rms'] for d in lag_pairs]):.6f} / "
        f"q90 {np.percentile([d['gr_rms'] for d in lag_pairs], 90):.6f} vs "
        f"committed {main_rec['gr_rms_mean']:.6f} / {main_rec['gr_rms_q90']:.6f}"))

    avg_rec = rec["lj_liquid"]["avg"]
    avg_pairs = []
    rots10 = []
    for i in range(5):                     # 5 rotations, both orderings
        a = [(i + k) % 10 for k in range(5)]
        b = [f for f in range(10) if f not in a]
        rots10.append((a, b))
    for a, b in rots10:                    # committed order: 5 (a,b) then 5 (b,a)
        oa = {k: np.mean([obs[f][k] for f in a], axis=0) for k in ("gr", "rm", "cn_hist")}
        ob = {k: np.mean([obs[f][k] for f in b], axis=0) for k in ("gr", "rm", "cn_hist")}
        avg_pairs.append(distance(oa, ob))
    for a, b in rots10:
        oa = {k: np.mean([obs[f][k] for f in a], axis=0) for k in ("gr", "rm", "cn_hist")}
        ob = {k: np.mean([obs[f][k] for f in b], axis=0) for k in ("gr", "rm", "cn_hist")}
        avg_pairs.append(distance(ob, oa))
    # committed avg pairs carry group lists, not frame indices; match by order
    n_match = sum(1 for d, p in zip(avg_pairs, avg_rec["pairs"])
                  if d["gr_rms"] == p["gr_rms"] and d["cn_tv"] == p["cn_tv"])
    ok = (n_match == 10 and len(avg_pairs) == avg_rec["n_pairs"]
          and np.mean([d["gr_rms"] for d in avg_pairs]) == avg_rec["gr_rms_mean"]
          and max(d["gr_rms"] for d in avg_pairs) == avg_rec["gr_rms_max"])
    results.append(check(
        "V3 lj_liquid averaged floor (rotated 5v5, 10 pairs)", ok,
        f"{n_match}/10 bit-exact; mean {np.mean([d['gr_rms'] for d in avg_pairs]):.6f} "
        f"vs committed {avg_rec['gr_rms_mean']:.6f}"))

    # ---- V4/V5: lj_liquid_large (frames trajectory-major: 5*t+k) -----------
    obs2 = load_case_obs("lj_liquid_large", 10)
    rec2 = rec["lj_liquid_large"]
    ref4 = committed_pairs(rec2)
    mine4 = []
    for t in range(2):                     # within-trajectory, lag >= 2
        for k1 in range(5):
            for k2 in range(k1 + 2, 5):
                mine4.append((5 * t + k1, 5 * t + k2))
    for k1 in range(2, 5):                 # cross, decorrelated halves (k >= 2)
        for k2 in range(2, 5):
            mine4.append((k1, 5 + k2))
    n_match = 0
    for (i, j) in mine4:
        d = distance(obs2[i], obs2[j])
        if ref4.get((i, j)) == (d["gr_rms"], d["cn_tv"]):
            n_match += 1
    ok = n_match == 21 and len(mine4) == rec2["n_pairs"]
    results.append(check(
        "V4 large frame-pair floor (within lag>=2 + cross halves, 21 pairs)",
        ok, f"{n_match}/21 bit-exact vs committed"))

    avg2 = rec2["avg"]
    mine5 = []
    oa = {k: np.mean([obs2[f][k] for f in range(5)], axis=0)
          for k in ("gr", "rm", "cn_hist")}
    ob = {k: np.mean([obs2[f][k] for f in range(5, 10)], axis=0)
          for k in ("gr", "rm", "cn_hist")}
    mine5.append(distance(oa, ob))                     # the 5v5 cross pair
    rots = [((0, 1), (2, 3, 4)), ((1, 2), (0, 3, 4)), ((2, 3), (0, 1, 4)),
            ((3, 4), (0, 1, 2)), ((0, 4), (1, 2, 3))]   # committed 2v3 cyclic set
    for t in range(2):
        base = 5 * t
        for a_idx, b_idx in rots:
            oa = {k: np.mean([obs2[base + f][k] for f in a_idx], axis=0)
                  for k in ("gr", "rm", "cn_hist")}
            ob = {k: np.mean([obs2[base + f][k] for f in b_idx], axis=0)
                  for k in ("gr", "rm", "cn_hist")}
            mine5.append(distance(oa, ob))
    n_match = sum(1 for d, p in zip(mine5, avg2["pairs"])
                  if d["gr_rms"] == p["gr_rms"] and d["cn_tv"] == p["cn_tv"])
    ok = n_match == 11 and len(mine5) == avg2["n_pairs"]
    results.append(check(
        "V5 large averaged floor (5v5 cross + 2v3 rotations, 11 pairs)",
        ok, f"{n_match}/11 bit-exact vs committed"))

    # ---- V6: P90 estimator --------------------------------------------------
    bad6 = []
    for case in ("lj_liquid", "lj_liquid_large", "lj_solid_liquid",
                 "water_tip4p", "nacl_aq"):
        r = rec[case]
        v = [p["gr_rms"] for p in r["pairs"]]
        if np.percentile(v, 90) != r["gr_rms_q90"]:
            bad6.append(case + ":gr")
        v = [p["cn_tv"] for p in r["pairs"]]
        if np.percentile(v, 90) != r["cn_tv_q90"]:
            bad6.append(case + ":cn")
    results.append(check(
        "V6 P90 estimator (np.percentile 90, linear)", not bad6,
        "reproduces every committed gr_rms_q90/cn_tv_q90" if not bad6
        else f"mismatch: {bad6}"))

    # ---- V7: D15 on committed avg floors == A5 effective floors ------------
    want = {"lj_liquid": (0.047, 0.041), "lj_liquid_large": (0.033, 0.017)}
    bad7 = []
    for case, (g, c) in want.items():
        r = rec[case]["avg"]
        if abs(d15([p["gr_rms"] for p in r["pairs"]]) - g) > 5e-4:
            bad7.append(case + ":gr")
        if abs(d15([p["cn_tv"] for p in r["pairs"]]) - c) > 5e-4:
            bad7.append(case + ":cn")
    results.append(check(
        "V7 D15 max(mean,P90) == A5 effective floor", not bad7,
        "lj_liquid gr 0.047/cn 0.041, large gr 0.033/cn 0.017 (acceptance_details)"
        if not bad7 else f"mismatch: {bad7}"))

    out = {"checks": results, "passed": all(r["pass"] for r in results)}
    (HERE / "validation_report.json").write_text(json.dumps(out, indent=1))
    print(f"== validate_conventions: "
          f"{'ALL PASS' if out['passed'] else 'FAILURES PRESENT'} ==")
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
