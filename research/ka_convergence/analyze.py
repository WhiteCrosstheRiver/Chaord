"""X7 analysis: energy + structure per decade of quench rate, with CIs,
against the repo's own cross-quench floors (chaord-free).

Inputs : runs/<id>/frame_<k>.npz (bench convention), runs/<id>/summary.json
         reports/noise_floors.json (ka_glass floors), bench/reference/ka_glass
Output : reports/convergence/ka_rates.json + ka_rates_tables.md (numbers only;
         the narrative report ka_rates.md is assembled from these).

Conventions (all cited in ka_obs.py, validated bit-exact in
validate_against_bench.py V3):
  per-quench observables = frame observables averaged over the LAST TWO
  (most-annealed) frames -- the glass floor's ref_frames convention
  (noise_floors.json ka_glass avg note); rate statistic = mean over the 10
  independent quenches; floor = distance between per-quench means across
  independent quenches (cross-quench pairs; 45 pairs at 10 quenches).
A5-style verdict: an inter-rate distance is "told apart" when it exceeds
1.5 x max(mean, P90) of the cross-quench floor pairs -- the acceptance
harness rule (noise.py within_floor factor = noise_floor_factor 1.5, and
the avg floor method note in noise_floors.json).
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

import ka_params as K
from ka_obs import frame_observables, distance

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
REPO = HERE.parent.parent
REPORTS = REPO / "reports"
BENCH = REPO / "bench" / "reference" / "ka_glass"

RNG = np.random.default_rng(20261025)
NBOOT = 10000
DIST_KEYS = ["gr_rms", "gr_rms_A-A", "gr_rms_A-B", "gr_rms_B-B", "cn_tv"]


def load_npz(path: Path):
    with np.load(path) as z:
        return np.asarray(z["r"], float), np.asarray(z["L"], float), \
            [str(s) for s in z["symbols"]]


def quench_observables(rid: str, frames=(3, 4)):
    """Per-quench mean observables over the last-two frames."""
    acc = None
    for k in frames:
        o = frame_observables(*load_npz(RUNS / rid / f"frame_{k}.npz"))
        if acc is None:
            acc = {k2: v.copy() for k2, v in o.items()}
        else:
            for k2 in acc:
                acc[k2] = acc[k2] + o[k2]
    return {k2: v / len(frames) for k2, v in acc.items()}


def mean_curve(obs_list):
    return {k2: np.mean([o[k2] for o in obs_list], axis=0) for k2 in obs_list[0]}


def bootstrap_ci(values, stat=np.mean):
    v = np.asarray(values, float)
    n = len(v)
    idx = RNG.integers(0, n, (NBOOT, n))
    boots = stat(v[idx], axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def welch_ci(a, b, conf=0.95):
    """Welch t CI for mean(a) - mean(b)."""
    from math import sqrt
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1) / na, b.var(ddof=1) / nb
    se = sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (na - 1) + vb ** 2 / (nb - 1))
    try:
        from scipy.stats import t as tdist
        tc = tdist.ppf(0.5 + conf / 2, df)
    except Exception:
        tc = 2.0
    d = a.mean() - b.mean()
    return float(d), float(d - tc * se), float(d + tc * se)


def main():
    floors_rec = json.loads((REPORTS / "noise_floors.json").read_text())
    kg_avg = floors_rec["ka_glass"]["avg"]
    kg_pair = floors_rec["ka_glass"]

    rates = K.RATES
    runs = {r: [K.run_id(i, q) for q in range(K.N_SEEDS)]
            for i, r in enumerate(K.RATE_TAGS)}
    have = {tag: [rid for rid in ids if (RUNS / rid / "DONE").exists()]
            for tag, ids in runs.items()}
    if any(len(v) < K.N_SEEDS for v in have.values()):
        print("WARNING incomplete:", {t: len(v) for t, v in have.items()})

    out = {"n_quenches": {t: len(v) for t, v in have.items()},
           "rates": rates, "dist_keys": DIST_KEYS, "n_boot": NBOOT}

    # ---- per-quench observables + energies
    obs = {}
    energy = {}
    for tag, ids in have.items():
        obs[tag] = [quench_observables(rid) for rid in ids]
        es = []
        for rid in ids:
            s = json.loads((RUNS / rid / "summary.json").read_text())
            pe5 = s["frame_pe_per_atom"]
            es.append({"thermal_pe_last2": float(np.mean(pe5[3:5])),
                       "is_energy": s["inherent_structure"]["pe_per_atom"],
                       "is_fnorm": s["inherent_structure"]["fnorm"]})
        energy[tag] = es

    # ---- per-rate energy statistics + decade differences
    est = {}
    for tag in have:
        if not energy[tag]:
            continue
        is_e = [e["is_energy"] for e in energy[tag]]
        pe_e = [e["thermal_pe_last2"] for e in energy[tag]]
        lo, hi = bootstrap_ci(is_e)
        lop, hip = bootstrap_ci(pe_e)
        pairs = [abs(a - b) for a, b in itertools.combinations(is_e, 2)]
        est[tag] = {
            "is_energy_mean": float(np.mean(is_e)),
            "is_energy_sd": float(np.std(is_e, ddof=1)),
            "is_energy_ci95": [lo, hi],
            "thermal_pe_last2_mean": float(np.mean(pe_e)),
            "thermal_pe_ci95": [lop, hip],
            "is_cross_quench_pairs": {
                "n": len(pairs), "mean": float(np.mean(pairs)),
                "p90": float(np.percentile(pairs, 90)),
                "max": float(np.max(pairs))},
        }
    dec = {}
    for a, b, name in [(0, 1, "r0_vs_r1"), (1, 2, "r1_vs_r2"), (0, 2, "r0_vs_r2")]:
        ta, tb = K.RATE_TAGS[a], K.RATE_TAGS[b]
        if len(have[ta]) < K.N_SEEDS or len(have[tb]) < K.N_SEEDS:
            continue
        d, lo, hi = welch_ci([e["is_energy"] for e in energy[ta]],
                             [e["is_energy"] for e in energy[tb]])
        dp, lop, hip = welch_ci([e["thermal_pe_last2"] for e in energy[ta]],
                                [e["thermal_pe_last2"] for e in energy[tb]])
        dec[name] = {"decades": b - a,
                     "rate_fast": rates[a], "rate_slow": rates[b],
                     "is_energy_diff": d, "is_energy_ci95": [lo, hi],
                     "thermal_pe_diff": dp, "thermal_pe_ci95": [lop, hip],
                     "is_significant": bool(lo > 0 or hi < 0)}
    out["energy"] = {"per_rate": est, "per_decade": dec,
                     "per_quench": {t: energy[t] for t in energy if energy[t]}}

    # ---- structure: rate means, cross-quench floors, inter-rate distances
    means = {t: mean_curve(obs[t]) for t in have if obs[t]}
    own_floors = {}
    for tag in have:
        if len(obs[tag]) < 2:
            continue
        ds = [distance(o1, o2) for o1, o2 in itertools.combinations(obs[tag], 2)]
        own_floors[tag] = {k: {"n": len(ds),
                               "mean": float(np.mean([d[k] for d in ds])),
                               "p90": float(np.percentile([d[k] for d in ds], 90)),
                               "max": float(np.max([d[k] for d in ds]))}
                           for k in DIST_KEYS}
    inter = {}
    for a, b, name in [(0, 1, "r0_vs_r1"), (1, 2, "r1_vs_r2"), (0, 2, "r0_vs_r2")]:
        ta, tb = K.RATE_TAGS[a], K.RATE_TAGS[b]
        if ta not in means or tb not in means:
            continue
        d = distance(means[ta], means[tb])
        # bootstrap over quenches for a CI on each distance
        ci = {}
        for k in DIST_KEYS:
            vals = []
            for _ in range(2000):
                ia = RNG.integers(0, len(obs[ta]), len(obs[ta]))
                ib = RNG.integers(0, len(obs[tb]), len(obs[tb]))
                vals.append(distance(mean_curve([obs[ta][i] for i in ia]),
                                     mean_curve([obs[tb][i] for i in ib]))[k])
            ci[k] = [float(np.percentile(vals, 2.5)),
                     float(np.percentile(vals, 97.5))]
        inter[name] = {"decades": b - a, "rate_fast": rates[a],
                       "rate_slow": rates[b], "distances": d, "ci95": ci}
    out["structure"] = {"rate_mean_distances": inter,
                        "own_cross_quench_floors": own_floors,
                        "bench_floors_avg": {k: kg_avg[f"{k}_mean"] for k in DIST_KEYS},
                        "bench_floors_avg_p90": {k: kg_avg[f"{k}_q90"] for k in DIST_KEYS},
                        "bench_floors_pair": {k: kg_pair[f"{k}_mean"] for k in DIST_KEYS}}

    # A5-style verdict per distance key: inter-rate distance vs
    # 1.5 x max(mean, P90) of OUR cross-quench floor (the honest floor at
    # each rate) and of the bench avg floor
    verdict = {}
    for name, blk in inter.items():
        v = {}
        for k in DIST_KEYS:
            thr_own = max(
                *[own_floors[t][k]["mean"] for t in own_floors],
                *[own_floors[t][k]["p90"] for t in own_floors]) * 1.5
            thr_bench = max(kg_avg[f"{k}_mean"], kg_avg[f"{k}_q90"]) * 1.5
            v[k] = {"distance": blk["distances"][k],
                    "threshold_1p5x_own_floor": thr_own,
                    "threshold_1p5x_bench_avg_floor": thr_bench,
                    "told_apart_own": bool(blk["distances"][k] > thr_own),
                    "told_apart_bench": bool(blk["distances"][k] > thr_bench)}
        verdict[name] = v
    out["verdict"] = verdict

    # ---- bridge: our fast-rate glasses vs the bench reference frames
    bench_obs = []
    for k in (3, 4, 8, 9, 13, 14):          # last-two of each bench quench
        bench_obs.append(frame_observables(*load_npz(BENCH / f"frame_{k}.npz")))
    bench_mean = mean_curve(bench_obs)
    bridge = {}
    for tag in means:
        bridge[tag] = distance(means[tag], bench_mean)
    out["bridge_vs_bench_reference"] = bridge

    # ---- peak/structure sanity snapshot per rate (AA first-peak height etc.)
    peaks = {}
    for tag, m in means.items():
        rm = m["rm"]
        pk = {}
        for key, win in (("A-A", (1.02, 1.16)), ("A-B", (0.80, 0.94)),
                         ("B-B", (1.28, 1.46))):
            g = m[f"gr_{key}"]
            msk = (rm >= win[0]) & (rm <= win[1])
            pk[key] = {"peak_r": float(rm[msk][int(np.argmax(g[msk]))]),
                       "peak_height": float(g[msk].max())}
        # mean coordination
        pk["cn_mean_from_hist"] = float(
            (m["cn_hist"] * np.arange(len(m["cn_hist"]))).sum())
        peaks[tag] = pk
    out["structure_snapshots"] = peaks

    dest = REPORTS / "convergence" / "ka_rates.json"
    dest.write_text(json.dumps(out, indent=1))

    # console digest
    print("== IS energy per atom ==")
    for tag in est:
        e = est[tag]
        print(f"  {tag} rate {rates[K.RATE_TAGS.index(tag)]:g}: "
              f"{e['is_energy_mean']:.4f} +- {e['is_energy_sd']:.4f} "
              f"CI95 {e['is_energy_ci95'][0]:.4f}..{e['is_energy_ci95'][1]:.4f} "
              f"(cross-quench |d| mean {e['is_cross_quench_pairs']['mean']:.4f})")
    print("== per-decade differences (Welch 95% CI) ==")
    for name, d in dec.items():
        print(f"  {name}: IS {d['is_energy_diff']:+.4f} "
              f"[{d['is_energy_ci95'][0]:+.4f},{d['is_energy_ci95'][1]:+.4f}] "
              f"sig={d['is_significant']} | thermal "
              f"{d['thermal_pe_diff']:+.4f}")
    print("== structure: rate-mean gr_rms / cn_tv ==")
    for name, blk in inter.items():
        print(f"  {name}: gr_rms {blk['distances']['gr_rms']:.4f} "
              f"[{blk['ci95']['gr_rms'][0]:.4f},{blk['ci95']['gr_rms'][1]:.4f}] "
              f"cn_tv {blk['distances']['cn_tv']:.4f} | "
              f"bench avg floor x1.5: gr {max(kg_avg['gr_rms_mean'], kg_avg['gr_rms_q90'])*1.5:.4f} "
              f"cn {max(kg_avg['cn_tv_mean'], kg_avg['cn_tv_q90'])*1.5:.4f}")
    print("== bridge vs bench reference (rate-mean vs bench 6-frame mean) ==")
    for tag, b in bridge.items():
        print(f"  {tag}: gr_rms {b['gr_rms']:.4f} cn_tv {b['cn_tv']:.4f} "
              f"B-B {b['gr_rms_B-B']:.4f}")
    print(f"\nwrote {dest}")
    return out


if __name__ == "__main__":
    main()
