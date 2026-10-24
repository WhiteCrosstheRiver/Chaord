"""IS-energy robustness pass: cascade minimisation (fire -> hftn -> fire) of
every run's FINAL sampled frame, vs the single-FIRE convention used in the
runs (which stalls at line-search alpha=0 in a few glasses, residual fnorm
up to ~1.1).  If the cascade agrees with the single-FIRE values within the
cross-quench spread, the D20 numbers are convention-robust.

Writes validation/is_robustness.json.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import numpy as np

import ka_params as K
from parse_runs import read_dump
from validate_against_bench import write_data, potential_setup

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
VALID = HERE / "validation"

IN = """\
units lj
boundary p p p
atom_style atomic
read_data {data}
pair_style lj/cut 2.5
pair_modify shift yes
pair_coeff 1 1 {eAA} {sAA} {rcAA}
pair_coeff 1 2 {eAB} {sAB} {rcAB}
pair_coeff 2 2 {eBB} {sBB} {rcBB}
thermo 1000
min_style fire
minimize 1e-12 1e-12 200000 400000
print "CASCADE fire1 $(pe:%.10f) $(fnorm:%.6e)"
min_style hftn
minimize 1e-12 1e-12 200000 400000
print "CASCADE hftn $(pe:%.10f) $(fnorm:%.6e)"
min_style fire
minimize 1e-12 1e-12 200000 400000
print "CASCADE fire2 $(pe:%.10f) $(fnorm:%.6e)"
"""


def main():
    VALID.mkdir(exist_ok=True)
    out = {}
    for r in range(3):
        for q in range(K.N_SEEDS):
            rid = K.run_id(r, q)
            step, types, pos, L = read_dump(RUNS / rid / "frame_4.lammpstrj")
            sym = ["B" if t == 2 else "A" for t in types]
            write_data(VALID / f"final_{rid}.data", pos, sym, L)
            inp = VALID / f"in.casc_{rid}"
            inp.write_text(IN.format(data=f"final_{rid}.data",
                                     **potential_setup()), newline="\n")
            res = subprocess.run(
                [K.LMP, "-in", inp.name, "-log", f"log.casc_{rid}.lammps",
                 "-screen", "none"], cwd=VALID, capture_output=True, text=True)
            assert res.returncode == 0, (rid, res.stderr[-500:])
            log = (VALID / f"log.casc_{rid}.lammps").read_text()
            stages = re.findall(
                r'CASCADE (\w+) ([-\d.]+) ([\d.e+-]+)', log)
            single = json.loads(
                (RUNS / rid / "summary.json").read_text())["inherent_structure"]
            out[rid] = {
                "single_fire": single["pe_per_atom"],
                "single_fire_fnorm": single["fnorm"],
                "cascade_stages": {s: {"pe": float(p), "fnorm": float(f)}
                                   for s, p, f in stages},
                "cascade_final": float(stages[-1][1]),
                "cascade_final_fnorm": float(stages[-1][2]),
                "shift": float(stages[-1][1]) - single["pe_per_atom"],
            }
    shifts = [abs(v["shift"]) for v in out.values()]
    rep = {"per_run": out,
           "max_abs_shift_vs_single_fire": float(max(shifts)),
           "mean_abs_shift": float(np.mean(shifts)),
           "max_cascade_fnorm": float(max(
               v["cascade_final_fnorm"] for v in out.values()))}
    (VALID / "is_robustness.json").write_text(json.dumps(rep, indent=1))
    for rid, v in out.items():
        print(f"{rid}: single {v['single_fire']:.4f} (F={v['single_fire_fnorm']:.2f})"
              f" -> cascade {v['cascade_final']:.4f} "
              f"(F={v['cascade_final_fnorm']:.1e}) shift {v['shift']:+.5f}")
    print("max |shift|", rep["max_abs_shift_vs_single_fire"],
          "mean |shift|", rep["mean_abs_shift"])


if __name__ == "__main__":
    main()
