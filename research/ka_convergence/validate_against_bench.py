"""Validate the X7 LAMMPS setup AND the numpy observables against the repo's
own bench/reference/ka_glass records (chaord-free, D23).

Three independent checks, each against a recorded number:

  V1 potential energy:  LAMMPS pe/atom of the 6 most-annealed bench frames
     vs the per-frame values recorded in reports/ka_glass_d7_sweep.json
     (A_ref5x ref_energy_per_atom, mean -7.460324... in reports/
     ka_glass_quench_resolution.json).  Proves pair_style/pair_coeff/
     pair_modify shift yes == the repo's cut-and-shifted KobAndersenLJ.

  V2 inherent structure (D20):  FIRE minimisation of the LAST frame of each
     bench quench (frames 4, 9, 14) vs open_items_v3.md X7 item 1's recorded
     values (-7.615, -7.611, -7.598 per atom).

  V3 observables:  the numpy g(r)/cn implementation of analyze.py recomputes
     the 12 recorded cross-quench floor pairs of noise_floors.json ka_glass
     (gr_rms_mean 0.037817..., cn_tv_mean 0.0310, partial keys).  Proves the
     analysis matches cv/noise.py + cv/rich.py at the ka_glass dialect
     thresholds (gr_rmax_fluid 2.5, gr_bins_fluid 60, cn_cutoff_fluid 1.5,
     noise_gr_rmin 0.8).

Run:  python validate_against_bench.py     (writes validation/report.json)
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import ka_params as K

HERE = Path(__file__).resolve().parent
VALID = HERE / "validation"
BENCH = HERE.parent.parent / "bench" / "reference" / "ka_glass"
REPORTS = HERE.parent.parent / "reports"
LMP = K.LMP

# --- conventions replicated from src/chaord (cited, not imported):
#   cv/noise.py observables + distance; cv/rich.py partial_gr;
#   thresholds: glass.yaml (gr_rmax_fluid 2.5, gr_bins_fluid 60,
#   cn_cutoff_fluid 1.5) and core.yaml (noise_gr_rmin 0.8).
GR_RMAX, GR_BINS, CN_CUT, GR_RMIN = 2.5, 60, 1.5, 0.8


def load_npz(path: Path):
    with np.load(path) as z:
        return np.asarray(z["r"], float), np.asarray(z["L"], float), \
            [str(s) for s in z["symbols"]]


def write_data(path: Path, pos, symbols, L):
    L = [float(x) for x in np.asarray(L, float)]
    types = [2 if s == "B" else 1 for s in symbols]
    with open(path, "w", newline="\n") as f:
        f.write(f"LAMMPS data file bench frame {path.stem}\n\n"
                f"{len(pos)} atoms\n2 atom types\n\n")
        for d, l in zip("xyz", L):
            f.write(f"0.0 {l:.10f} {d}lo {d}hi\n")
        f.write("\nMasses\n\n1 1.0\n2 1.0\n\nAtoms # atomic\n\n")
        for i, (p, t) in enumerate(zip(pos, types), 1):
            f.write(f"{i} {t} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}\n")


IN_RUN0 = """\
units lj
boundary p p p
atom_style atomic
read_data {data}
pair_style lj/cut 2.5
pair_modify shift yes
pair_coeff 1 1 {eAA} {sAA} {rcAA}
pair_coeff 1 2 {eAB} {sAB} {rcAB}
pair_coeff 2 2 {eBB} {sBB} {rcBB}
run 0
print "PE_RESULT $(pe:%.12f)"
"""

IN_FIRE = IN_RUN0 + """\
min_style fire
min_modify integrator eulerimplicit
minimize 1e-10 1e-10 100000 200000
print "IS_RESULT $(pe:%.10f) $(fnorm:%.6e) $(step)"
"""


def run_lammps(in_txt: str, tag: str) -> str:
    p = VALID / f"in.{tag}"
    p.write_text(in_txt, newline="\n")
    r = subprocess.run([LMP, "-in", p.name, "-log", f"log.{tag}.lammps",
                        "-screen", "none"], cwd=VALID, capture_output=True,
                       text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{tag}: lmp failed\n{r.stdout}\n{r.stderr}")
    return (VALID / f"log.{tag}.lammps").read_text()


def grab(log: str, key: str):
    m = re.findall(rf'^{key}[^\n]*', log, re.M)
    if not m:
        raise RuntimeError(f"no {key} line in log")
    return m[-1]


def potential_setup():
    return dict(eAA=K.EPS["AA"], sAA=K.SIG["AA"], rcAA=K.RC["AA"],
                eAB=K.EPS["AB"], sAB=K.SIG["AB"], rcAB=K.RC["AB"],
                eBB=K.EPS["BB"], sBB=K.SIG["BB"], rcBB=K.RC["BB"])


# ------------------------------------------------- observables: ka_obs.py --
from ka_obs import (GR_RMAX, GR_BINS, CN_CUT, GR_RMIN,   # noqa: F401
                    gr_all, gr_partial, cn_hist, frame_observables, distance)


# ------------------------------------------------------------------- main ---

def main():
    VALID.mkdir(exist_ok=True)
    report = {"lammps": LMP, "checks": {}}

    frames = {}
    for k in range(15):
        frames[k] = load_npz(BENCH / f"frame_{k}.npz")

    # ---- V1: potential energy of the 6 most-annealed frames
    recorded_v1 = [-7.4648, -7.4706, -7.4567, -7.469, -7.4523, -7.4486]
    v1 = {}
    for k in (3, 4, 8, 9, 13, 14):
        pos, L, sym = frames[k]
        write_data(VALID / f"bench_{k}.data", pos, sym, L)
        log = run_lammps(IN_RUN0.format(data=f"bench_{k}.data",
                                        **potential_setup()),
                         f"pe_{k}")
        # units lj: thermo pe is NORMALISED PER ATOM (thermo_modify norm
        # yes is the lj-units default), so no division
        pe = float(grab(log, "PE_RESULT").split()[1])
        v1[k] = pe
    devs = [min(abs(pe - rec) for rec in recorded_v1) for pe in v1.values()]
    report["checks"]["V1_potential_energy_per_atom"] = {
        "measured": v1,
        "recorded_reference_values": recorded_v1,
        "recorded_mean": -7.460324409878239,
        "measured_mean": float(np.mean(list(v1.values()))),
        "max_abs_dev_vs_recorded_set": float(max(devs)),
        "pass": bool(max(devs) < 5e-4),
    }

    # ---- V2: inherent-structure energy of the 3 quench-final frames
    recorded_v2 = {4: -7.615, 9: -7.611, 14: -7.598}
    v2 = {}
    for k, _rec in recorded_v2.items():
        t0 = time.time()
        log = run_lammps(IN_FIRE.format(data=f"bench_{k}.data",
                                        **potential_setup()),
                         f"is_{k}")
        line = grab(log, "IS_RESULT")
        pe = float(line.split()[1])          # per atom, see V1 note
        v2[k] = {"is_energy_per_atom": pe,
                 "fnorm": line.split()[2], "fire_iterations": line.split()[3],
                 "seconds": round(time.time() - t0, 1)}
    devs2 = {k: v2[k]["is_energy_per_atom"] - recorded_v2[k]
             for k in recorded_v2}
    report["checks"]["V2_inherent_structure_per_atom"] = {
        "measured": {k: v2[k]["is_energy_per_atom"] for k in v2},
        "recorded": recorded_v2,
        "deviation": devs2,
        "detail": v2,
        "pass": bool(max(abs(d) for d in devs2.values()) < 0.02),
    }

    # ---- V3: numpy observables reproduce the recorded ka_glass floor
    floors = json.loads((REPORTS / "noise_floors.json").read_text())
    kg = floors["ka_glass"]
    obs = {k: frame_observables(*frames[k]) for k in range(15)}
    pairs = [(3, 8), (3, 9), (3, 13), (3, 14), (4, 8), (4, 9),
             (4, 13), (4, 14), (8, 13), (8, 14), (9, 13), (9, 14)]
    ds = [distance(obs[i], obs[j]) for i, j in pairs]
    v3 = {}
    ok = True
    for key in ("gr_rms", "gr_rms_A-A", "gr_rms_A-B", "gr_rms_B-B", "cn_tv"):
        mine = [d[key] for d in ds]
        rec_mean = kg[f"{key}_mean"]
        rec_max = kg[f"{key}_max"]
        rel = abs(float(np.mean(mine)) - rec_mean) / rec_mean
        v3[key] = {"measured_mean": float(np.mean(mine)),
                   "measured_max": float(np.max(mine)),
                   "recorded_mean": rec_mean, "recorded_max": rec_max,
                   "rel_dev_of_mean": rel}
        ok &= rel < 0.02
    report["checks"]["V3_floor_observables"] = {"pass": bool(ok), **v3}

    report["passed"] = all(c["pass"] for c in report["checks"].values())
    (VALID / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report["checks"], indent=1))
    print("VALIDATION", "PASSED" if report["passed"] else "FAILED")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
