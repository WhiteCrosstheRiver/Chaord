# -*- coding: utf-8 -*-
"""Print every number in docs/paper/draft.md and draw Figures 1, 2, 3 and 5.

Run from the repository root:  python docs/paper/scripts/make_paper_assets.py

Reads only committed evidence. Writes docs/paper/assets/paper_numbers.json,
docs/paper/assets/table1_acceptance.md and docs/paper/figures/fig*.{pdf,png}.
Fails closed: a recomputed statistic that disagrees with the stored one raises.
STANDALONE: numpy, scipy, matplotlib; does not import chaord.
"""
import json
import os

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

A = "research/cnt_electrolyte/analysis/"
OUT = "docs/paper/assets/"
FIG = "docs/paper/figures/"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
FIELD_MV = {0.01: "0.01", 0.02: "0.02"}

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
    "savefig.dpi": 300, "pdf.fonttype": 42})


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def welch(on, off):
    on, off = np.asarray(on), np.asarray(off)
    r = stats.ttest_ind(on, off, equal_var=False)
    se = np.sqrt(on.var(ddof=1) / len(on) + off.var(ddof=1) / len(off))
    df = float(r.df)
    tc = stats.t.ppf(0.975, df)
    d = float(on.mean() - off.mean())
    return dict(shift=d, t=float(r.statistic), df=df, p=float(r.pvalue),
                ci95=[d - tc * se, d + tc * se], n=len(on))


def close(a, b, what, tol=1e-9):
    if not abs(a - b) <= tol:
        raise RuntimeError(f"{what}: recomputed {a} != stored {b}")


def z4_numbers(c):
    J = c["jobs"]
    off = [np.mean([j["direct_p1_blocks"]["A"], j["direct_p1_blocks"]["B"]])
           for j in J.values() if j["field"] == 0]
    out = dict(off_per_run=[float(x) for x in off], off_mean=float(np.mean(off)),
               off_sem=float(np.std(off, ddof=1) / np.sqrt(len(off))), fields={})
    for E, key in FIELD_MV.items():
        on = [j["direct_p1_blocks"]["B"] for j in J.values() if j["field"] == E]
        w = welch(on, off)
        s = c["welch"]["fields"][key]
        for k in ("shift", "t", "p", "df"):
            close(w[k], s[k], f"Z4 {key} {k}", 1e-6)
        w.update(on_per_run=[float(x) for x in on], on_mean=float(np.mean(on)),
                 on_sem=float(np.std(on, ddof=1) / np.sqrt(len(on))),
                 n_positive=int(sum(x > 0 for x in on)))
        out["fields"][key] = w
    return out


def fidelity(c, l):
    cj = [j["p1_text_vs_direct_max_abs"] for j in c["jobs"].values()]
    lj = [j["p1_text_vs_direct_max_abs"] for j in l["jobs"].values()]
    conf = int(c["meta"]["lift_coverage"]["lifts_total"])
    full = int(l["meta"]["lift_subsampling"]["lifts_total"])
    return dict(confirmatory_lifts=conf, ladder_lifts=full,
                total_lifts=conf + full, max_confirmatory=max(cj),
                max_ladder=max(lj), max_all=max(cj + lj),
                failed_checks_confirmatory=c["meta"]["failed_static_checks_total"],
                per_job_confirmatory=cj, per_job_ladder=lj)


def filling(f):
    out = {}
    for name, tag in (("fill_cnt16", "16"), ("fill_cnt10", "10")):
        F = f[name]
        out[tag] = {s: dict(pilot_per_A=F["comparison"][s]["pilot_per_A"],
                            fill_per_A=F["comparison"][s]["fill_per_A"],
                            pct_diff=F["comparison"][s]["pct_diff"],
                            per_A_sem_blocks=F["stats"][s].get("per_A_sem_blocks"))
                    for s in ("EC", "EMC", "Li+", "PF6-")}
    per98 = {s: out["16"][s]["fill_per_A"] * 98.4 for s in out["16"]}
    out["reservoir_per_98p4A"] = per98
    if round(per98["EC"]) != 125 or round(per98["EMC"]) != 58:
        raise RuntimeError("R13 composition does not follow from Q2 uptake")
    return out


def acceptance_table(acc):
    rows = ["| ID | Criterion | Passed |", "| --- | --- | --- |"]
    n = 0
    for c in acc["criteria"]:
        n += bool(c["passed"])
        rows.append(f"| {c['id']} | {c['name']} | "
                    f"{'yes' if c['passed'] else 'NO'} |")
    with open(OUT + "table1_acceptance.md", "w", encoding="utf-8",
              newline="\n") as f:
        f.write("\n".join(rows) + "\n")
    return dict(passed=n, total=len(acc["criteria"]))


def save(fig, stem):
    fig.savefig(FIG + stem + ".pdf", bbox_inches="tight")
    fig.savefig(FIG + stem + ".png", bbox_inches="tight")
    plt.close(fig)


def fig2(l, z):
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.0),
                               gridspec_kw=dict(width_ratios=[1.25, 1]))
    style = {"16": (BLUE, "o", "(16,16), Stage B"),
             "10": (ORANGE, "s", "(10,10), Stage B")}
    for tube, (col, mk, lab) in style.items():
        cur = l["per_tube"][tube]["p1_curve"]["chaord_dipole_p1"]
        E = sorted(float(k) for k in cur)
        m = [cur[{0.0: "0"}.get(e, repr(e))]["mean"] for e in E]
        s = [cur[{0.0: "0"}.get(e, repr(e))]["sem"] for e in E]
        a.errorbar(E, m, yerr=s, color=col, marker=mk, ms=4.5, lw=1.4,
                   capsize=2, label=lab)
        i = E.index(0.05)
        a.annotate(lab.split(",")[0] + "," + lab.split(",")[1],
                   (E[i], m[i]), xytext=(-6, 8) if tube == "16" else (7, -10),
                   textcoords="offset points", color=INK, fontsize=8,
                   ha="right" if tube == "16" else "left")
    a.set_xscale("symlog", linthresh=0.005)
    a.set_xticks([0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
    a.set_xticklabels(["0", ".005", ".01", ".02", ".05", ".1", ".2"])
    a.set(xlabel="Field (V/Å)", ylabel="Blind P1, block B")
    a.set_title("a  Full ladder (n = 3, pilot filling)", loc="left", fontsize=9)
    a.legend(loc="upper left", fontsize=8)

    cur = l["per_tube"]["16"]["p1_curve"]["chaord_dipole_p1"]
    E = [0.0, 0.01, 0.02]
    k = {0.0: "0", 0.01: "0.01", 0.02: "0.02"}
    b.errorbar(np.array(E) - 0.0006, [cur[k[e]]["mean"] for e in E],
               yerr=[cur[k[e]]["sem"] for e in E], color=BLUE, marker="o",
               ms=4.5, lw=1.4, capsize=2, label="Stage B (n = 3)")
    zm = [z["off_mean"], z["fields"]["0.01"]["on_mean"], z["fields"]["0.02"]["on_mean"]]
    zs = [z["off_sem"], z["fields"]["0.01"]["on_sem"], z["fields"]["0.02"]["on_sem"]]
    b.errorbar(np.array(E) + 0.0006, zm, yerr=zs, color=AQUA, marker="D", ms=4.5,
               lw=1.4, capsize=2, label="Z4 (n = 6)")
    rng = np.random.default_rng(20261106)
    for e, vals in ((0.0, z["off_per_run"]),
                    (0.01, z["fields"]["0.01"]["on_per_run"]),
                    (0.02, z["fields"]["0.02"]["on_per_run"])):
        x = e + 0.0006 + rng.uniform(0.0002, 0.0012, len(vals))
        b.scatter(x, vals, s=11, color=AQUA, alpha=0.45, lw=0, zorder=1)
    b.set_xticks(E)
    b.set_xticklabels(["0", ".01", ".02"])
    b.set(xlabel="Field (V/Å)", ylabel="Blind P1")
    b.set_title("b  (16,16), packed below open-tube uptake", loc="left", fontsize=9)
    b.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    save(fig, "fig2_blind_p1_vs_field")


def fig3(c, fid):
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.9))
    tx, di = [], []
    for j in c["jobs"].values():
        tx.extend(j["p1_text"])
        di.extend(j["p1_direct"])
    tx, di = np.array(tx, float), np.array(di, float)
    if len(tx) != fid["confirmatory_lifts"]:
        raise RuntimeError(f"fig3: {len(tx)} frames")
    a.scatter(di, tx, s=2, color=BLUE, alpha=0.2, lw=0, rasterized=True)
    lim = [min(di.min(), tx.min()), max(di.max(), tx.max())]
    a.plot(lim, lim, color=MUTED, lw=0.8)
    a.set(xlabel="P1, direct calculation", ylabel="P1, Chaord statement")
    a.set_title(f"a  {len(tx):,} confirmatory frames", loc="left", fontsize=9)
    res = tx - di
    b.hist(res, bins=np.linspace(-5.5e-4, 5.5e-4, 45), color=BLUE, rwidth=0.9)
    for s in (-5e-4, 5e-4):
        b.axvline(s, color=ORANGE, lw=1.2, ls="--")
    b.set_ylim(0, b.get_ylim()[1] * 1.22)
    b.annotate("±5×10⁻⁴ print half-quantum", (0, b.get_ylim()[1] * 0.92),
               ha="center", color=INK, fontsize=8)
    b.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    b.set(xlabel="Statement − direct", ylabel="Frames")
    b.set_title("b  Residual", loc="left", fontsize=9)
    fig.tight_layout()
    save(fig, "fig3_fidelity")
    return dict(max_abs_residual_recomputed=float(np.abs(res).max()))


def fig5(fill):
    sp = ["EC", "EMC", "Li+", "PF6-"]
    lab = ["EC", "EMC", "Li⁺", "PF₆⁻"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for ax, tube, name in zip(axes, ("16", "10"), ("(16,16)", "(10,10)")):
        F = fill[tube]
        x = np.arange(4)
        p = ax.bar(x - 0.2, [F[s]["pilot_per_A"] for s in sp], 0.38, color=ORANGE,
                   label="Pilot packing")
        q = ax.bar(x + 0.2, [F[s]["fill_per_A"] for s in sp], 0.38, color=BLUE,
                   hatch="///", edgecolor="#fcfcfb", lw=0, label="Open-tube uptake")
        ax.set_xticks(x)
        ax.set_xticklabels(lab)
        ax.set(ylabel="Molecules per Å")
        ax.set_title(name, loc="left", fontsize=9)
        ax.grid(axis="x", visible=False)
        ax.bar_label(q, fmt="%.2f", fontsize=7, padding=1.5, color=INK)
        ax.bar_label(p, fmt="%.2f", fontsize=7, padding=1.5, color=INK)
        ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    save(fig, "fig5_filling")


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    c = load(A + "confirmatory_results.json")
    l = load(A + "full_ladder_results.json")
    f = load(A + "filling_results.json")
    acc = load("reports/acceptance.json")
    N = dict(acceptance=acceptance_table(acc), z4=z4_numbers(c),
             fidelity=fidelity(c, l), filling=filling(f),
             e_min={t: l["per_tube"][t]["e_min"] for t in ("16", "10")},
             stage_b_p1={t: {k: v for k, v in
                             l["per_tube"][t]["p1_curve"]["chaord_dipole_p1"].items()}
                         for t in ("16", "10")},
             q3=load("research/cnt_electrolyte/q3_evidence/q3_results.json"))
    fig2(l, N["z4"])
    N["fidelity"].update(fig3(c, N["fidelity"]))
    fig5(N["filling"])
    with open(OUT + "paper_numbers.json", "w", encoding="utf-8",
              newline="\n") as fh:
        json.dump(N, fh, indent=1)
    z = N["z4"]["fields"]
    print("acceptance", N["acceptance"])
    print("Z4 0.01", {k: round(z["0.01"][k], 4) for k in ("shift", "t", "df", "p")},
          [round(v, 3) for v in z["0.01"]["ci95"]], "positive", z["0.01"]["n_positive"])
    print("Z4 0.02", {k: round(z["0.02"][k], 6) for k in ("shift", "t", "df", "p")},
          [round(v, 3) for v in z["0.02"]["ci95"]], "positive", z["0.02"]["n_positive"])
    fd = N["fidelity"]
    print("lifts", fd["total_lifts"], "max", fd["max_all"],
          "residual", fd["max_abs_residual_recomputed"])
    print("per 98.4", {k: round(v, 2) for k, v in
                       N["filling"]["reservoir_per_98p4A"].items()})


if __name__ == "__main__":
    main()
