# -*- coding: utf-8 -*-
"""PILOT DIAGNOSTIC, pilot seeds only: regenerates the coex reference frames so
the lift's handling of them can be studied. Not part of any result."""
import multiprocessing as mp
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "src")]
import numpy as np                                        # noqa: E402
sys.argv = [sys.argv[0], sys.argv[1], "--pilot"]
import matrix as M                                        # noqa: E402


def job(i):
    r = M.reference(("coex", i))
    return i, r["last"], r.get("bins"), r.get("error")


if __name__ == "__main__":
    with mp.Pool(6) as pool:
        out = pool.map(job, range(6), chunksize=1)
    data = {}
    for i, last, bins, err in out:
        print(i, bins, err, flush=True)
        data[f"pos{i}"], data[f"box{i}"] = last
    np.savez_compressed("out/coex_frames.npz", **data)
