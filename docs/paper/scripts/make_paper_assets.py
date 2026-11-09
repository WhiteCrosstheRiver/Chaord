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
import re

import numpy as np
from scipy import stats
import matplotlib
import matplotlib.colors
matplotlib.use("Agg")
import matplotlib.pyplot as plt

A = "research/cnt_electrolyte/analysis/"
OUT = "docs/paper/assets/"
FIG = "docs/paper/figures/"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#7a4fd0"
R13 = "research/cnt_electrolyte/r13/r13_results.json"
CURVE = "research/program_length/results.json"
MACRO = "research/macrostate_selection/results.json"
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


def text_arm(J):
    """Secondary, not registered: the same Welch contrast on the means of the
    printed statement (dipole_p1_blocks) instead of the direct calculation."""
    off = [np.mean([j["dipole_p1_blocks"]["A"], j["dipole_p1_blocks"]["B"]])
           for j in J.values() if j["field"] == 0]
    out = {}
    for E, key in FIELD_MV.items():
        on = [j["dipole_p1_blocks"]["B"] for j in J.values() if j["field"] == E]
        out[key] = welch(on, off)
        out[key]["n_positive"] = int(sum(x > 0 for x in on))
    return out


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
    out["text_arm"] = text_arm(J)
    return out


def r13_numbers(r):
    """R13 (reservoir filling): the same Welch call as Z4, recomputed from the
    per-job blocks and checked against r13_results.json's stored d39."""
    J = r["jobs"]
    off = [np.mean([j["direct_p1_blocks"]["A"], j["direct_p1_blocks"]["B"]])
           for j in J.values() if j["field"] == 0]
    out = dict(off_per_run=[float(x) for x in off], off_mean=float(np.mean(off)),
               off_sem=float(np.std(off, ddof=1) / np.sqrt(len(off))), fields={},
               decision=r["decision"]["verdict"],
               section7_output=r["section7_output"],
               paired=r["build_paired_secondary"],
               chaining={b: v["identical_at_step_1000000"]
                         for b, v in r["chaining_check"].items()})
    for E, key in FIELD_MV.items():
        on = [j["direct_p1_blocks"]["B"] for j in J.values() if j["field"] == E]
        w = welch(on, off)
        s = r["d39"]["fields"][key]
        for k in ("shift", "t", "p", "df"):
            close(w[k], s[k], f"R13 {key} {k}", 1e-6)
        w.update(on_per_run=[float(x) for x in on], on_mean=float(np.mean(on)),
                 on_sem=float(np.std(on, ddof=1) / np.sqrt(len(on))),
                 n_positive=int(sum(x > 0 for x in on)))
        out["fields"][key] = w
    out["text_arm"] = text_arm(J)
    if (out["fields"]["0.01"]["p"] < 0.05) != (out["decision"] == "CONFIRMED"):
        raise RuntimeError("R13 stored verdict disagrees with the recomputed p")
    return out


def curve_numbers(r):
    """Program-length curve (research/program_length, PLAN registered 5b434b8):
    recompute pass/fail from the stored ratios, the shortest passing level per
    physics level, and the decision under both readings of "none" (RESULTS.md).
    Fails closed against the stored pass flags, shortest_passing and verdict."""
    gate, cells = r["gate"], r["cells"]
    for k, c in cells.items():
        ok = all(c["ratio"][m] <= gate for m in ("gr_rms", "cn_tv"))
        if ok != c["passed"] or c["refusals"]:
            raise RuntimeError(f"curve cell {k}: pass flag or refusal mismatch")
    cases, phys, prog = ("lj_liquid", "lj_liquid_large"), ("P0", "P1", "P2"), ("S0", "S1", "S2")
    short = {}
    for case in cases:
        short[case] = {}
        for p in phys:
            ok = [s for s in prog if cells[f"{case}|{p}|{s}"]["passed"]]
            short[case][p] = ok[0] if ok else "none"
            if short[case][p] != r["shortest_passing"][case][p]["level"]:
                raise RuntimeError(f"curve shortest {case} {p} disagrees")
    def verdict(none_is_longest):
        for case in cases:
            L = [r["shortest_passing"][case][p]["length"] for p in phys]
            if none_is_longest:
                L = [float("inf") if x is None else x for x in L]
            else:
                if L[2] is None:
                    return "NOT SUPPORTED"
                L = [x for x in L if x is not None]
                if not all(a >= b for a, b in zip(L, L[1:])) or max(L) <= L[-1]:
                    return "NOT SUPPORTED"
                continue
            if not (L[0] >= L[1] >= L[2] and max(L[0], L[1]) > L[2]):
                return "NOT SUPPORTED"
        return "SUPPORTED"
    a, b = verdict(True), verdict(False)
    if a != r["decision"]["verdict"]:
        raise RuntimeError("curve stored verdict disagrees with reading A")
    rng = lambda xs: [float(min(xs)), float(max(xs))]
    sel = lambda p, s, m: [cells[f"{c}|{p}|{s}"][m] if m == "angle_tv_rel_P2S0"
                           else cells[f"{c}|{p}|{s}"]["ratio"][m] for c in cases]
    s1_inert = all(cells[f"{c}|{p}|S1"]["ratio"] == cells[f"{c}|{p}|S0"]["ratio"]
                   for c in cases for p in phys)
    grid = {p: {s: dict(passed=[cells[f"{c}|{p}|{s}"]["passed"] for c in cases],
                        gr=rng([cells[f"{c}|{p}|{s}"]["ratio"]["gr_rms"] for c in cases]),
                        cn=rng([cells[f"{c}|{p}|{s}"]["ratio"]["cn_tv"] for c in cases]))
                for s in prog} for p in phys}
    return dict(commit=r["commit"], seeds=r["seeds"], gate=gate, grid=grid,
                shortest=short, verdict_reading_A=a, verdict_reading_B=b,
                s1_identical_to_s0=s1_inert,
                lengths={c: {s: cells[f"{c}|P0|{s}"]["length"] for s in prog}
                         for c in cases},
                p1_md_steps=cells["lj_liquid|P1|S0"]["md_steps"],
                p2_md_steps=cells["lj_liquid|P2|S0"]["md_steps"],
                p1_s0_gr=rng(sel("P1", "S0", "gr_rms")),
                p1_s0_cn=rng(sel("P1", "S0", "cn_tv")),
                p2_s0_gr=rng(sel("P2", "S0", "gr_rms")),
                p2_s0_cn=rng(sel("P2", "S0", "cn_tv")),
                p0_s0_gr=rng(sel("P0", "S0", "gr_rms")),
                p2_s2_gr=rng(sel("P2", "S2", "gr_rms")),
                p2_s2_cn=rng(sel("P2", "S2", "cn_tv")),
                p2_s2_angle_rel=rng(sel("P2", "S2", "angle_tv_rel_P2S0")),
                s2_angle_rel_all=rng([cells[f"{c}|{p}|S2"]["angle_tv_rel_P2S0"]
                                      for c in cases for p in phys]),
                wall_seconds=r["wall_seconds"])


def fidelity(c, l, r):
    cj = [j["p1_text_vs_direct_max_abs"] for j in c["jobs"].values()]
    lj = [j["p1_text_vs_direct_max_abs"] for j in l["jobs"].values()]
    rj = [j["p1_text_vs_direct_max_abs"] for j in r["jobs"].values()]
    conf = int(c["meta"]["lift_coverage"]["lifts_total"])
    full = int(l["meta"]["lift_subsampling"]["lifts_total"])
    res = sum(int(j["n_lifted"]) for j in r["jobs"].values())
    return dict(confirmatory_lifts=conf, ladder_lifts=full, r13_lifts=res,
                total_lifts=conf + full + res, max_confirmatory=max(cj),
                max_ladder=max(lj), max_r13=max(rj), max_all=max(cj + lj + rj),
                failed_checks_confirmatory=c["meta"]["failed_static_checks_total"],
                failed_checks_r13=sum(len(j["failed_checks"])
                                      for j in r["jobs"].values()),
                per_job_confirmatory=cj, per_job_ladder=lj, per_job_r13=rj)


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


def macrostate_numbers(r):
    """Macrostate selection (PLAN registered 8deeff2): recompute every match and
    the six registered conditions from the stored ratios; fails closed."""
    S, gate, cells = ("liquid", "fcc", "hcp", "coex"), r["gate"], r["cells"]
    for k, c in cells.items():
        if all(v <= gate for v in c["ratio"].values()) != c["match"]:
            raise RuntimeError(f"macrostate cell {k}: stored match disagrees")
    m = lambda i, kind, j: cells[f"{i}|{kind}|{j}"]["match"]
    checks = dict(no_refusals=not r["problems"],
                  references_valid=r["checks"]["references_valid"],
                  diagonal=all(m(s, "full", s) for s in S),
                  off_diagonal=not any(m(s, "full", j) for s in S for j in S if j != s),
                  control_liquid=all(m(s, "control", "liquid") for s in S),
                  control_not_source=not any(m(s, "control", s) for s in S[1:]))
    if checks != r["checks"]:
        raise RuntimeError("macrostate conditions disagree with the stored ones")
    verdict = "SUPPORTED" if all(checks.values()) else "NOT SUPPORTED"
    if verdict != r["verdict"]:
        raise RuntimeError("macrostate verdict disagrees")
    grid = {kind: {i: {j: max(cells[f"{i}|{kind}|{j}"]["ratio"].values()) for j in S}
                   for i in S} for kind in ("full", "control")}
    off = [grid["full"][i][j] for i in S for j in S if i != j]
    return dict(verdict=verdict, checks=checks, grid=grid, gate=gate, T=r["T"], rho=r["rho"],
                diagonal={s: grid["full"][s][s] for s in S}, off_min=min(off), off_max=max(off),
                control_to_liquid=[grid["control"][s]["liquid"] for s in S],
                coex_ratio=cells["coex|full|coex"]["ratio"],
                coex_control=grid["control"]["coex"]["coex"],
                hcp_ratio=cells["hcp|full|hcp"]["ratio"],
                lengths={s: [len(r["programs"][f"{s}|{i}"]["full"]) for i in range(3)] for s in S},
                control_lengths=[len(r["programs"][f"{s}|0"]["control"]) for s in S],
                coex_rebuild_fcc=[x[0] for x in r["rebuild_final_cna"]["coex|full"]],
                coex_reference_fcc=[x[0] for x in r["reference_final_cna"]["coex"]])


def fig6(mac):
    """Macrostate matrix: rows source program, columns reference, one hue on a
    log scale; outlined cells pass the gate, every cell carries its number."""
    S = ("liquid", "fcc", "hcp", "coex")
    labels = ("liquid", "fcc", "hcp", "slab")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)
    norm = matplotlib.colors.LogNorm(vmin=0.3, vmax=1300)
    for ax, kind, title in zip(axes, ("full", "control"),
                               ("(a) Lifted program", "(b) Control: cell, T, N, density only")):
        g = np.array([[mac["grid"][kind][i][j] for j in S] for i in S])
        im = ax.imshow(g, cmap="Blues", norm=norm)
        ax.grid(False)
        ax.set_xticks(range(4), labels)
        ax.set_yticks(range(4), labels)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xlabel("Reference state")
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        for y in range(4):
            for x in range(4):
                v = g[y, x]
                ax.text(x, y, f"{v:.1f}" if v < 100 else f"{v:.0f}", ha="center", va="center",
                        fontsize=8.5, color="white" if v > 40 else INK)
                if v <= mac["gate"]:
                    ax.add_patch(plt.Rectangle((x - .46, y - .46), .92, .92, fill=False,
                                               ec=ORANGE, lw=2))
    axes[0].set_ylabel("Source of program")
    cb = fig.colorbar(im, ax=axes, shrink=.85, pad=.02)
    cb.set_label("Largest distance / noise floor")
    cb.outline.set_visible(False)
    save(fig, "fig6_macrostate_matrix")


def a5_cases():
    """Per-case A5 ratios as tools/acceptance.py printed them (one decimal),
    and the cases it skipped for lack of a measured floor."""
    t = open("reports/acceptance_details.md", encoding="utf-8").read()
    sec, rows, skipped = t[t.index("## A5"):t.index("## A6")], {}, []
    for line in sec.splitlines():
        cells = [x.strip() for x in line.strip("|").split("|")]
        if len(cells) < 3 or "/" not in cells[0]:
            continue
        if cells[2].startswith("pass"):
            got = re.findall(r"(cn_tv|gr_rms) [0-9.]+ vs floor [0-9.]+ \(x([0-9.]+)", cells[2])
            rows[cells[0]] = dict(category=cells[1], **{k: float(v) for k, v in got})
        elif cells[2].startswith("no-floor"):
            skipped.append(cells[0])
        else:
            raise RuntimeError(f"A5 case not pass or skipped: {cells[0]}")
    return dict(scored=rows, skipped=skipped)


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


def fig2(l, z, r13):
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.0),
                               gridspec_kw=dict(width_ratios=[1.25, 1]))
    style = {"16": (BLUE, "o", "(16,16)"),
             "10": (ORANGE, "s", "(10,10)")}
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
    a.set_xticklabels(["0", ".005", ".01", ".02", ".05", ".1", ".2"], rotation=45, fontsize=8)
    a.set(xlabel="Field (V/Å)", ylabel="Blind P1, block B")
    a.set_title("a  Full ladder (n = 3, pilot filling)", loc="left", fontsize=9)
    a.legend(loc="upper left", fontsize=8)

    cur = l["per_tube"]["16"]["p1_curve"]["chaord_dipole_p1"]
    E = [0.0, 0.01, 0.02]
    k = {0.0: "0", 0.01: "0.01", 0.02: "0.02"}
    b.errorbar(np.array(E) - 0.0012, [cur[k[e]]["mean"] for e in E],
               yerr=[cur[k[e]]["sem"] for e in E], color=BLUE, marker="o",
               ms=4.5, lw=1.4, capsize=2, label="Field ladder (n = 3)")
    zm = [z["off_mean"], z["fields"]["0.01"]["on_mean"], z["fields"]["0.02"]["on_mean"]]
    zs = [z["off_sem"], z["fields"]["0.01"]["on_sem"], z["fields"]["0.02"]["on_sem"]]
    b.errorbar(np.array(E) + 0.0002, zm, yerr=zs, color=AQUA, marker="D", ms=4.5,
               lw=1.4, capsize=2, label="First test, pilot packing (n = 6)")
    rng = np.random.default_rng(20261106)
    for e, vals in ((0.0, z["off_per_run"]),
                    (0.01, z["fields"]["0.01"]["on_per_run"]),
                    (0.02, z["fields"]["0.02"]["on_per_run"])):
        x = e + rng.uniform(0.0004, 0.0010, len(vals))
        b.scatter(x, vals, s=11, color=AQUA, alpha=0.45, lw=0, zorder=1)
    rm = [r13["off_mean"], r13["fields"]["0.01"]["on_mean"],
          r13["fields"]["0.02"]["on_mean"]]
    rs = [r13["off_sem"], r13["fields"]["0.01"]["on_sem"],
          r13["fields"]["0.02"]["on_sem"]]
    b.errorbar(np.array(E) + 0.0016, rm, yerr=rs, color=VIOLET, marker="^",
               ms=5, lw=1.4, capsize=2, label="Second test, measured uptake (n = 6)")
    for e, vals in ((0.0, r13["off_per_run"]),
                    (0.01, r13["fields"]["0.01"]["on_per_run"]),
                    (0.02, r13["fields"]["0.02"]["on_per_run"])):
        x = e + rng.uniform(0.0018, 0.0024, len(vals))
        b.scatter(x, vals, s=11, color=VIOLET, alpha=0.45, lw=0, zorder=1)
    b.set_xticks(E)
    b.set_xticklabels(["0", ".01", ".02"])
    b.set(xlabel="Field (V/Å)", ylabel="Blind P1")
    b.set_title("b  (16,16): pilot filling vs reservoir filling", loc="left",
                fontsize=9)
    b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=7, handlelength=1.5, columnspacing=1.0)
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
    r = load(R13)
    N = dict(acceptance=acceptance_table(acc), a5_cases=a5_cases(), z4=z4_numbers(c),
             r13=r13_numbers(r), program_length=curve_numbers(load(CURVE)),
             macrostate=macrostate_numbers(load(MACRO)),
             fidelity=fidelity(c, l, r), filling=filling(f),
             e_min={t: l["per_tube"][t]["e_min"] for t in ("16", "10")},
             stage_b_p1={t: {k: v for k, v in
                             l["per_tube"][t]["p1_curve"]["chaord_dipole_p1"].items()}
                         for t in ("16", "10")},
             q3=load("research/cnt_electrolyte/q3_evidence/q3_results.json"))
    fig2(l, N["z4"], N["r13"])
    N["fidelity"].update(fig3(c, N["fidelity"]))
    fig5(N["filling"])
    fig6(N["macrostate"])
    with open(OUT + "paper_numbers.json", "w", encoding="utf-8",
              newline="\n") as fh:
        json.dump(N, fh, indent=1)
    z = N["z4"]["fields"]
    print("acceptance", N["acceptance"])
    print("Z4 0.01", {k: round(z["0.01"][k], 4) for k in ("shift", "t", "df", "p")},
          [round(v, 3) for v in z["0.01"]["ci95"]], "positive", z["0.01"]["n_positive"])
    print("Z4 0.02", {k: round(z["0.02"][k], 6) for k in ("shift", "t", "df", "p")},
          [round(v, 3) for v in z["0.02"]["ci95"]], "positive", z["0.02"]["n_positive"])
    rz = N["r13"]["fields"]
    for key in ("0.01", "0.02"):
        print("R13", key, {k: round(rz[key][k], 4) for k in ("shift", "t", "df", "p")},
              [round(v, 4) for v in rz[key]["ci95"]], "positive",
              rz[key]["n_positive"])
    print("R13 decision", N["r13"]["decision"], "chaining", N["r13"]["chaining"])
    fd = N["fidelity"]
    print("lifts", fd["total_lifts"], "max", fd["max_all"],
          "residual", fd["max_abs_residual_recomputed"])
    print("curve", N["program_length"])
    print("per 98.4", {k: round(v, 2) for k, v in
                       N["filling"]["reservoir_per_98p4A"].items()})


if __name__ == "__main__":
    main()
