"""X7 item-2 analysis: re-ground the liquid floors on >= 30 decorrelated
pairs with bootstrap 95% intervals (chaord-free).

Inputs : runs/<id>/frame_<k>.npz (bench convention; 6 trajectories = 2 cases
         x 3 independent runs, protocol per liq_params.py)
         reports/noise_floors.json (the committed floors being re-ground)
Output : liquid_floors.json (all numbers; the narrative report
         reports/convergence/liquid_floors.md is assembled from these)

Definitions (all validated bit-exact against the committed floors by
validate_conventions.py, 8/8):
  observables   liq_obs.py = src/chaord/cv/noise.py legacy set (all-species
                g(r), cn histogram) at the core+lj thresholds.
  frame-pair floor  D15 (max(mean, P90)) over the decorrelated frame pairs:
                cross-trajectory pairs with both frame indices >= LAG_MIN,
                pooled with within-trajectory pairs at index lag >= LAG_MIN
                (the committed pooling; LAG_MIN = 4 = 20 tau, the committed
                lj_liquid decorrelation lag).  Diagnostics: within-only,
                cross-only, and the lag-decay of pair distance (lags 1..8).
  averaged floor  the committed group conventions resampled over independent
                blocks (liq_params.py docstring "pair sets"):
                lj_liquid: rotated 5v5 half-splits of 12 disjoint 10-frame
                blocks + position-matched block-vs-block pairs across
                trajectories; lj_liquid_large: per 5-frame window the 5
                cyclic 2v3 rotations (x2 orderings? NO -- each rotation
                once, the committed large-case convention) over 9 disjoint
                windows per trajectory + position-matched 5v5 window pairs
                across trajectories.
  bootstrap     resample the pair list with replacement, NBOOT=1000 draws
                (stream brief), D15 per draw, CI = percentile [2.5, 97.5];
                one seeded generator (BOOT_SEED), floors in the fixed order
                recorded below.  Pairs share frames, so this is the brief's
                pair-resampling convention, not an independent-draw CI.
  comparison    new floor vs committed: ratio of D15 values, and whether the
                committed D15 falls inside the new 95% CI.

Usage:  python analyze_floors.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import liq_params as K
from liq_obs import frame_observables, distance, d15

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
REPO = HERE.parent.parent
FLOORS = REPO / "reports" / "noise_floors.json"

# fixed evaluation order for the shared bootstrap generator (recorded in the
# output; re-running the script reproduces every CI bit-exactly)
FLOOR_ORDER = ["lj_liquid.frame_pair", "lj_liquid.avg",
               "lj_liquid_large.frame_pair", "lj_liquid_large.avg"]


def load_traj_obs(case_idx: int):
    """Per-trajectory observable lists for one case."""
    out = []
    for t in range(K.N_TRAJ):
        rid = K.run_id(case_idx, t)
        obs = []
        for k in range(K.N_FRAMES):
            with np.load(RUNS / rid / f"frame_{k}.npz") as z:
                obs.append(frame_observables(np.asarray(z["r"], float),
                                             np.asarray(z["L"], float)))
        out.append(obs)
    return out



def mean_obs(obs, frames):
    """Mean observables over frame indices (the committed group-mean op)."""
    return {k: np.mean([obs[f][k] for f in frames], axis=0)
            for k in ("gr", "rm", "cn_hist")}


def pair_rec(frames, d):
    return {"frames": list(frames), "gr_rms": d["gr_rms"], "cn_tv": d["cn_tv"]}


def stats_block(pairs):
    """mean/max/q90/D15 per key, in the committed layout."""
    out = {"n_pairs": len(pairs)}
    for key in ("gr_rms", "cn_tv"):
        v = np.array([p[key] for p in pairs], float)
        b = {"mean": float(np.mean(v)),
             "max": float(np.max(v)),
             "q90": float(np.percentile(v, 90))}
        b["floor_d15"] = d15(v)
        out[key] = b
    return out


def bootstrap_d15(pairs, rng):
    """(lo, hi) 95% CI of the D15 floor per key, resampling pairs."""
    n = len(pairs)
    gr = np.array([p["gr_rms"] for p in pairs], float)
    cn = np.array([p["cn_tv"] for p in pairs], float)
    idx = rng.integers(0, n, (K.NBOOT, n))
    out = {}
    for key, v in (("gr_rms", gr), ("cn_tv", cn)):
        boots = np.maximum(v[idx].mean(axis=1),
                           np.percentile(v[idx], 90, axis=1))
        out[key] = [float(np.percentile(boots, 2.5)),
                    float(np.percentile(boots, 97.5))]
    return out


def frame_pair_floor(traj_obs, lag_diag=False):
    """Cross pairs (both idx >= LAG_MIN, each unordered pair once) pooled
    with within-trajectory pairs at index lag >= LAG_MIN."""
    n = K.N_FRAMES
    cross, within = [], []
    for a in range(K.N_TRAJ):
        for b in range(a + 1, K.N_TRAJ):
            for i in range(K.LAG_MIN, n):
                for j in range(K.LAG_MIN, n):
                    cross.append(pair_rec((a, i, b, j),
                                          distance(traj_obs[a][i],
                                                   traj_obs[b][j])))
    for a in range(K.N_TRAJ):
        obs_a = traj_obs[a]
        for i in range(n):
            for j in range(i + K.LAG_MIN, n):
                within.append(pair_rec((a, i, a, j),
                                       distance(obs_a[i], obs_a[j])))
    diag = {}
    if lag_diag:
        for lag in range(1, 9):
            ds = [distance(traj_obs[a][i], traj_obs[a][i + lag])
                  for a in range(K.N_TRAJ) for i in range(n - lag)]
            diag[f"lag{lag}"] = {
                "gr_rms_mean": float(np.mean([d["gr_rms"] for d in ds])),
                "cn_tv_mean": float(np.mean([d["cn_tv"] for d in ds]))}
    pooled = cross + within
    return {"pairs": pooled, "diag": diag, "n_cross": len(cross),
            "n_within": len(within), "cross_pairs": cross,
            "within_pairs": within}


def avg_floor_lj500(traj_obs):
    """Committed lj_liquid statistic resampled: rotated 5v5 half-splits of
    10-frame blocks + position-matched block pairs across trajectories."""
    within, cross = [], []
    for a in range(K.N_TRAJ):
        obs = traj_obs[a]
        for blk in range(K.N_BLOCKS_AVG):
            fr = [blk * K.AVG_BLOCK_LEN + k for k in range(K.AVG_BLOCK_LEN)]
            rots = []
            for i in range(5):
                a_idx = [(i + k) % K.AVG_BLOCK_LEN for k in range(5)]
                b_idx = [f for f in range(K.AVG_BLOCK_LEN) if f not in a_idx]
                rots.append((a_idx, b_idx))
            for a_idx, b_idx in rots:      # committed order: 5 then 5 reversed
                oa = mean_obs(obs, [fr[f] for f in a_idx])
                ob = mean_obs(obs, [fr[f] for f in b_idx])
                within.append(pair_rec((a, blk, "ab", a_idx[0]),
                                       distance(oa, ob)))
            for a_idx, b_idx in rots:
                oa = mean_obs(obs, [fr[f] for f in a_idx])
                ob = mean_obs(obs, [fr[f] for f in b_idx])
                within.append(pair_rec((a, blk, "ba", a_idx[0]),
                                       distance(ob, oa)))
    for blk in range(K.N_BLOCKS_AVG):      # cross-trajectory block pairs
        for a in range(K.N_TRAJ):
            for b in range(a + 1, K.N_TRAJ):
                oa = mean_obs(traj_obs[a], [blk * 10 + k for k in range(10)])
                ob = mean_obs(traj_obs[b], [blk * 10 + k for k in range(10)])
                cross.append(pair_rec((a, blk, b, blk), distance(oa, ob)))
    return {"pairs": within + cross, "n_within": len(within),
            "n_cross": len(cross)}


def avg_floor_lj2048(traj_obs):
    """Committed large-case statistic resampled: per 5-frame window the 5
    cyclic 2v3 rotations (each once) + position-matched 5v5 window pairs."""
    rots = [((0, 1), (2, 3, 4)), ((1, 2), (0, 3, 4)), ((2, 3), (0, 1, 4)),
            ((3, 4), (0, 1, 2)), ((0, 4), (1, 2, 3))]   # committed cyclic set
    n_win = K.N_FRAMES // K.WINDOW_AVG
    within, cross = [], []
    for a in range(K.N_TRAJ):
        obs = traj_obs[a]
        for w in range(n_win):
            base = w * K.WINDOW_AVG
            for a_idx, b_idx in rots:
                oa = mean_obs(obs, [base + f for f in a_idx])
                ob = mean_obs(obs, [base + f for f in b_idx])
                within.append(pair_rec((a, w, "2v3", a_idx[0]),
                                       distance(oa, ob)))
    for w in range(n_win):
        for a in range(K.N_TRAJ):
            for b in range(a + 1, K.N_TRAJ):
                oa = mean_obs(traj_obs[a], [w * 5 + f for f in range(5)])
                ob = mean_obs(traj_obs[b], [w * 5 + f for f in range(5)])
                cross.append(pair_rec((a, w, b, w), distance(oa, ob)))
    return {"pairs": within + cross, "n_within": len(within),
            "n_cross": len(cross)}


def committed_block(rec):
    """Committed floor values in the same layout for comparison."""
    out = {}
    for key in ("gr_rms", "cn_tv"):
        out[key] = {"mean": rec[f"{key}_mean"], "max": rec[f"{key}_max"],
                    "q90": rec[f"{key}_q90"],
                    "floor_d15": max(rec[f"{key}_mean"], rec[f"{key}_q90"]),
                    "n_pairs": rec["n_pairs"]}
    return out


def main() -> int:
    committed = json.loads(FLOORS.read_text())
    rng = np.random.default_rng(K.BOOT_SEED)
    out = {
        "order": "docs/reviews/open_items_v3.md X7 item 2 (liquid floors); "
                 "estimator D15 max(mean,P90); bootstrap pair resampling "
                 f"{K.NBOOT} draws, seed {K.BOOT_SEED}; conventions "
                 "validated bit-exact (validate_conventions.py 8/8)",
        "protocol": {c: K.protocol_record(c) for c in K.CASES},
        "runs": {},
        "floors": {},
        "committed_comparison": {},
    }
    man = json.loads((RUNS / "inputs_manifest.json").read_text())
    for rid, m in man["runs"].items():
        s = json.loads((RUNS / rid / "summary.json").read_text())
        out["runs"][rid] = {
            "case": m["case"], "seeds": m["seeds"],
            "n_frames": len(s["frames"]),
            "equil_T_star": s["stages"]["equil"]["T_star"],
            "run_end_pe_per_atom": s["run_end"]["pe_per_atom"],
            "density": s["density"],
        }

    bench = REPO / "bench" / "reference"
    for case_idx, case_name in ((0, "lj_liquid"), (1, "lj_liquid_large")):
        traj_obs = load_traj_obs(case_idx)
        committed_obs = []
        for k in range(10):
            with np.load(bench / case_name / f"frame_{k}.npz") as z:
                committed_obs.append(frame_observables(
                    np.asarray(z["r"], float), np.asarray(z["L"], float)))
        fp = frame_pair_floor(traj_obs, lag_diag=True)
        av = (avg_floor_lj500(traj_obs) if case_idx == 0
              else avg_floor_lj2048(traj_obs))
        case_out = {"frame_pair": {}, "avg": {}}
        for fname, blk in (("frame_pair", fp), ("avg", av)):
            sb = stats_block(blk["pairs"])
            sb["n_cross"] = blk.get("n_cross")
            sb["n_within"] = blk.get("n_within")
            sb["bootstrap_ci95"] = bootstrap_d15(blk["pairs"], rng)
            if fname == "frame_pair":
                # D15 of the two pair families separately (the committed
                # large-case note quantified cross vs within; here it is
                # measured on the new data), plus the D15 spread across the
                # three trajectory pairs (trajectory-level variability --
                # the pair bootstrap under-represents it because pairs
                # share frames)
                for fam in ("within_pairs", "cross_pairs"):
                    sb[fam[:-6] + "_only_d15"] = {
                        k: d15([q[k] for q in blk[fam]])
                        for k in ("gr_rms", "cn_tv")}
                tp = []
                for a in range(K.N_TRAJ):
                    for b in range(a + 1, K.N_TRAJ):
                        v = [q for q in blk["cross_pairs"]
                             if q["frames"][0] == a and q["frames"][2] == b]
                        tp.append({k: d15([q[k] for q in v])
                                   for k in ("gr_rms", "cn_tv")})
                sb["traj_pair_d15_spread"] = tp
                # pair lists omitted from the JSON: 7626 records, bit-exactly
                # reproducible from the committed observable series
            else:
                sb["pairs"] = blk["pairs"]
            case_out[fname] = sb
        case_out["frame_pair"]["lag_decay_diagnostic"] = fp["diag"]
        out["floors"][case_name] = case_out

        comp = {}
        for fname in ("frame_pair", "avg"):
            old = committed_block(committed[case_name]
                                  if fname == "frame_pair"
                                  else committed[case_name]["avg"])
            new = case_out[fname]
            for key in ("gr_rms", "cn_tv"):
                lo, hi = new["bootstrap_ci95"][key]
                nd, od = new[key]["floor_d15"], old[key]["floor_d15"]
                comp[f"{fname}.{key}"] = {
                    "committed_d15": od, "new_d15": nd,
                    "ratio_new_over_committed": nd / od,
                    "new_ci95": [lo, hi],
                    "committed_d15_inside_new_ci": bool(lo <= od <= hi)}
        # engine equivalence: the committed (ASE) frames vs the new (LAMMPS)
        # frames under the same observables -- if the new frames sat off the
        # committed distribution, new-committed would exceed both one-set
        # ranges; it sits between them instead
        def dset(ds, name):
            v = {k: np.array([d[k] for d in ds]) for k in ("gr_rms", "cn_tv")}
            return {name: {k: {"mean": float(np.mean(v[k])),
                               "q90": float(np.percentile(v[k], 90)),
                               "max": float(np.max(v[k])),
                               "n": len(ds)} for k in v}}
        eq = {}
        eq.update(dset([distance(committed_obs[i], committed_obs[j])
                        for i in range(len(committed_obs))
                        for j in range(i + 1, len(committed_obs))],
                       "committed_vs_committed"))
        eq.update(dset([distance(traj_obs[a][i], traj_obs[b][j])
                        for a in range(K.N_TRAJ)
                        for b in range(a + 1, K.N_TRAJ)
                        for i in range(K.LAG_MIN, K.N_FRAMES)
                        for j in range(K.LAG_MIN, K.N_FRAMES)],
                       "new_vs_new_cross"))
        eq.update(dset([distance(traj_obs[a][i], committed_obs[j])
                        for a in range(K.N_TRAJ)
                        for i in range(K.LAG_MIN, K.N_FRAMES)
                        for j in range(len(committed_obs))],
                       "new_vs_committed_cross"))
        out.setdefault("engine_equivalence", {})[case_name] = eq
        out["committed_comparison"][case_name] = comp

    (HERE / "liquid_floors.json").write_text(json.dumps(out, indent=1))
    for case_name, case_out in out["floors"].items():
        print(f"== {case_name} ==")
        for fname in ("frame_pair", "avg"):
            sb = case_out[fname]
            print(f"  {fname}: {sb['n_pairs']} pairs "
                  f"(cross {sb['n_cross']}, within {sb['n_within']})")
            for key in ("gr_rms", "cn_tv"):
                b, ci = sb[key], sb["bootstrap_ci95"][key]
                print(f"    {key}: mean {b['mean']:.4f} q90 {b['q90']:.4f} "
                      f"-> D15 {b['floor_d15']:.4f}  "
                      f"CI95 [{ci[0]:.4f}, {ci[1]:.4f}]")
    print("== committed comparison ==")
    for case_name, comp in out["committed_comparison"].items():
        for k, v in comp.items():
            print(f"  {case_name} {k}: committed {v['committed_d15']:.4f} "
                  f"-> new {v['new_d15']:.4f} "
                  f"(x{v['ratio_new_over_committed']:.2f}), "
                  f"CI {v['new_ci95'][0]:.4f}..{v['new_ci95'][1]:.4f}, "
                  f"committed inside CI: "
                  f"{v['committed_d15_inside_new_ci']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
