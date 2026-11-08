# -*- coding: utf-8 -*-
"""T3: the pre-registered R13 analysis (r13/PLAN.md sections 3 and 7).

Nothing new is measured here. The per-job inputs are produced by the Z4
machinery imported UNCHANGED from analysis/confirmatory_z4.py (arm 1: blind
Chaord lift of every thinned frame; arm 2: direct numpy P1; the fidelity
assert at the print half-quantum; the cross-build chaining check). Only the
R13 constants (PLAN section 1: 4872 atoms, EC 125 / EMC 58 / PF6 24 / Li 24)
and the R13 job paths are set. The test itself is the section-7 code block,
read out of PLAN.md and executed verbatim, its printed lines stored as-is.

Usage (repo root):  python research/cnt_electrolyte/r13/r13_analysis.py [--procs N]
Writes research/cnt_electrolyte/r13/r13_results.json.
"""
import argparse
import contextlib
import io
import json
import multiprocessing as mp
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent                  # .../r13
CNT = HERE.parent                                       # research/cnt_electrolyte
ROOT = CNT.parents[1]
for _p in (str(ROOT / "src"), str(CNT), str(CNT / "analysis")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import confirmatory_z4 as z4                            # noqa: E402

# ---- R13 constants (PLAN section 1), checked at run time by the Z4 asserts --
z4.N_ATOMS = 4872                                       # 2560 tube C + 2312 liquid
z4.EXPECTED_CENSUS = {"EC": 125, "EMC": 58, "PF6": 24, "Li": 24}
RUNS = HERE / "runs"
OUT = HERE / "r13_results.json"
FIELD_OF_MV = {0: 0.0, 10: 0.01, 20: 0.02}


def discover_jobs() -> list[dict]:
    jobs = []
    for mv, field in FIELD_OF_MV.items():
        for b in range(1, 7):
            name = f"r13_e{mv}_b{b}"
            d = RUNS / name
            dump, log, bd = d / "prod_thinned.dump", d / "log.lammps", d / "system.data"
            for p in (dump, log, bd, d / "final.data"):
                assert p.is_file() and p.stat().st_size > 0, (name, p)
            assert "Total wall time" in log.read_text(errors="replace"), name
            jobs.append({"name": name, "field": field, "mv": mv,
                         "build": f"r13_b{b}", "dump": dump, "build_data": bd,
                         "log": log})
    assert len(jobs) == 18
    return jobs


def one_job(job: dict) -> dict:
    t0 = time.perf_counter()
    dl = z4.composition_dialect()
    d = z4.direct_arm(job)
    c = z4.chaord_arm(job, d["prod_ps"], dl)
    rec = z4.job_record(job, d, c, time.perf_counter() - t0)
    rec["dump"] = str(Path(rec["dump"]).relative_to(ROOT).as_posix())
    rec["build_data"] = str(Path(rec["build_data"]).relative_to(ROOT).as_posix())
    print(f"[r13] {job['name']}: A {rec['direct_p1_blocks']['A']:+.4f} "
          f"B {rec['direct_p1_blocks']['B']:+.4f}  fidelity "
          f"{rec['p1_text_vs_direct_max_abs']:.2e}  {rec['job_wall_s']:.0f}s",
          flush=True)
    return rec


def section7_code() -> str:
    text = (HERE / "PLAN.md").read_text(encoding="utf-8")
    sec = text.split("## 7.", 1)[1]
    m = re.search(r"```python\n(.*?)```", sec, re.S)
    assert m, "PLAN.md section 7 has no python block"
    return m.group(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=18)
    args = ap.parse_args()
    jobs = discover_jobs()
    chaining = z4.chaining_check(jobs)
    with mp.Pool(args.procs) as pool:
        recs = pool.map(one_job, jobs, chunksize=1)
    J = {r["name"]: r for r in recs}
    d39 = z4.d39_welch(J)
    paired = z4.build_paired_secondary(J)
    res = {
        "plan": "research/cnt_electrolyte/r13/PLAN.md",
        "machinery": "analysis/confirmatory_z4.py (direct_arm, chaord_arm, "
                     "job_record, chaining_check, d39_welch, "
                     "build_paired_secondary), imported unchanged",
        "n_atoms": z4.N_ATOMS, "census": z4.EXPECTED_CENSUS,
        "chaining_check": chaining,
        "d39": d39,
        "build_paired_secondary": paired,
        "jobs": J,
    }
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    # the pre-registered call, verbatim, against the file just written
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(section7_code(), "PLAN.md#7", "exec"), {"__name__": "s7"})
    res["section7_output"] = buf.getvalue().splitlines()
    gate_p = d39["fields"]["0.01"]["p"]
    res["decision"] = {
        "rule": "0.01 V/A is CONFIRMED AT THE RESERVOIR FILLING iff its Welch "
                "p < 0.05 on this set alone (PLAN section 3)",
        "p": gate_p,
        "verdict": "CONFIRMED" if gate_p < z4.ALPHA else "NOT CONFIRMED",
    }
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("\n".join(res["section7_output"]))
    print(f"[r13] decision: {res['decision']['verdict']} (p {gate_p:.4f})")


if __name__ == "__main__":
    main()
