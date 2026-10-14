# -*- coding: utf-8 -*-
"""
Equilibration-evidence extraction from LAMMPS logs (runs/<sys>/log.lammps).

Parses the three run sections (ramp / equilibration / production), records
the mobile-subsystem temperature (v_Tmob) and potential-energy trajectories,
and computes block-mean comparisons for the equilibration decision:
  * stage-2 PE: mean over quarters 1..4 of the 400 ps equilibration
  * PE drift over the last 200 ps (kcal/mol/ps, linear-fit slope)
  * same for Tmob
Also records per-section loop times (wall time) from the "Loop time" lines.

A plateau is *recorded*, not asserted: the numbers go into the provenance
JSON as-is.

Usage:  python analyze_equil.py <system> [...]
Writes  runs/<system>/equil_<system>.json
"""
import json
import os
import re
import sys

import numpy as np

import params as P

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_log(path):
    sections = []          # list of dict(header=[...], rows=[[floats]])
    wall_times = []        # (n_steps, seconds)
    cur = None
    with open(path) as f:
        for line in f:
            ls = line.strip()
            if ls.startswith("Step "):
                cur = dict(header=ls.split(), rows=[])
                sections.append(cur)
            elif cur is not None:
                m = re.match(r"^Loop time of ([0-9.]+) on (\d+) procs for "
                             r"(\d+) steps", line)
                if m:
                    wall_times.append((int(m.group(3)), float(m.group(1))))
                    cur = None
                else:
                    try:
                        cur["rows"].append([float(x) for x in line.split()])
                    except ValueError:
                        cur = None
    return sections, wall_times


def block_stats(step, val, t0, t1):
    m = (step >= t0) & (step <= t1)
    return float(np.mean(val[m])), float(np.std(val[m])), int(m.sum())


def analyze(name):
    mdir = os.path.join(HERE, "runs", name)
    sections, walls = parse_log(os.path.join(mdir, "log.lammps"))
    out = {"system": name, "n_sections": len(sections), "wall_times": []}
    for nsteps, sec in walls:
        out["wall_times"].append({"steps": nsteps, "wall_s": round(sec, 2)})
    out["total_wall_s"] = round(sum(w[1] for w in walls), 2)

    # identify sections: ramp=first, equil=second, prod=third
    tags = ["ramp", "equil", "prod"] + ["extra"] * max(0, len(sections) - 3)
    for tag, sec in zip(tags, sections):
        hdr = sec["header"]
        rows = np.array(sec["rows"])
        col = {c: i for i, c in enumerate(hdr)}
        i_step, i_pe, i_T = col["Step"], col["PotEng"], col["v_Tmob"]
        step = rows[:, i_step]
        pe, T = rows[:, i_pe], rows[:, i_T]
        # time in ps within the section (timestep from params)
        t = step * P.TIMESTEP_FS / 1000.0
        rec = {
            "steps": [int(step[0]), int(step[-1])],
            "Tmob_mean_K": round(float(T.mean()), 2),
            "Tmob_std_K": round(float(T.std()), 2),
            "PE_mean_kcal_mol": round(float(pe.mean()), 3),
            "PE_std_kcal_mol": round(float(pe.std()), 3),
            "n_samples": int(len(step)),
        }
        if tag == "equil":
            # thermodynamic-time block means (t runs 0..400 ps in-section)
            t0, t1 = t[0], t[-1]
            qs = []
            for k in range(4):
                a = t0 + (t1 - t0) * k / 4.0
                b = t0 + (t1 - t0) * (k + 1) / 4.0
                m, s, n = block_stats(t, pe, a, b)
                qs.append(round(m, 3))
            rec["PE_quarter_means_kcal_mol"] = qs
            # drift over the last half: linear slope
            m = t >= t0 + 0.5 * (t1 - t0)
            slope = float(np.polyfit(t[m], pe[m], 1)[0])
            rec["PE_slope_last_half_kcal_per_ps"] = round(slope, 5)
            mT, sT, _ = block_stats(t, T, t0 + 0.5 * (t1 - t0), t1)
            rec["Tmob_mean_last_half_K"] = round(mT, 2)
            rec["Tmob_std_last_half_K"] = round(sT, 2)
            # normalized drift: |slope| * 200 ps vs |PE q4 - q1|
            rec["PE_q4_minus_q1_kcal_mol"] = round(qs[3] - qs[0], 3)
            n_mol = (P.COMPOSITION[P.SYSTEMS[name]["n"]]["n_EC"]
                     + P.COMPOSITION[P.SYSTEMS[name]["n"]]["n_EMC"]
                     + 2 * P.COMPOSITION[P.SYSTEMS[name]["n"]]["n_LiPF6"])
            rec["PE_q4_minus_q1_per_molecule_kT"] = round(
                (qs[3] - qs[0]) / n_mol / 0.0019872041 / 300.0, 4)
        if tag == "prod":
            rec["Tmob_mean_K"] = round(float(T.mean()), 2)
            rec["PE_mean_kcal_mol"] = round(float(pe.mean()), 3)
        out[tag] = rec

    with open(os.path.join(mdir, f"equil_{name}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[equil] {name}: total wall {out['total_wall_s']} s")
    for tag in ("ramp", "equil", "prod"):
        if tag in out:
            r = out[tag]
            print(f"   {tag}: steps {r['steps']} Tmob {r['Tmob_mean_K']} "
                  f"+- {r['Tmob_std_K']} K, PE {r['PE_mean_kcal_mol']} "
                  f"+- {r['PE_std_kcal_mol']}")
    if "equil" in out:
        e = out["equil"]
        print(f"   PE quarter means {e['PE_quarter_means_kcal_mol']}, "
              f"q4-q1 = {e['PE_q4_minus_q1_kcal_mol']} kcal/mol "
              f"({e['PE_q4_minus_q1_per_molecule_kT']} kT/mol), "
              f"slope(last half) {e['PE_slope_last_half_kcal_per_ps']} "
              f"kcal/mol/ps")
    return out


def main():
    for name in (sys.argv[1:] or list(P.SYSTEMS)):
        analyze(name)


if __name__ == "__main__":
    main()
