"""Parallel queue runner for the X7 runs (one serial LAMMPS process per job).

Usage:  python run_jobs.py r0 r1 r2 --jobs 10
Jobs are run dirs under runs/ with an in.<id> input; a DONE marker file makes
a completed run idempotent (skipped on re-invocation).
"""
from __future__ import annotations

import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ka_params as K

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"


def run_one(rid: str) -> tuple[str, float, int]:
    d = RUNS / rid
    if (d / "DONE").exists():
        return rid, 0.0, 0
    t0 = time.time()
    r = subprocess.run(
        [K.LMP, "-in", f"in.{rid}", "-log", "log.lammps", "-screen", "none"],
        cwd=d, capture_output=True, text=True)
    dt = time.time() - t0
    if r.returncode == 0:
        (d / "DONE").write_text(f"{rid} ok {dt:.1f}s\n")
    else:
        (d / "FAILED").write_text(r.stdout[-4000:] + r.stderr[-2000:])
    return rid, dt, r.returncode


def main(argv):
    tags = [a for a in argv[1:] if not a.startswith("--")]
    jobs = int((argv[argv.index("--jobs") + 1] if "--jobs" in argv else 10))
    ids = []
    for r in range(len(K.RATES)):
        if tags and K.RATE_TAGS[r] not in tags:
            continue
        for q in range(K.N_SEEDS):
            ids.append(K.run_id(r, q))
    print(f"{len(ids)} jobs, {jobs} parallel: {ids}")
    rc = 0
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(run_one, i): i for i in ids}
        for f in as_completed(futs):
            rid, dt, code = f.result()
            print(f"  {rid}: {dt:.1f}s rc={code}", flush=True)
            rc |= code
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))
