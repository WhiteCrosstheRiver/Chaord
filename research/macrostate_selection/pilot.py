# -*- coding: utf-8 -*-
"""PILOT ONLY, seed 9001: finds a (T, rho) where all four macrostates hold for
the run length, and whether lift accepts real MD frames. Nothing here enters a
registered result."""
import json
import multiprocessing as mp
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "src")]
import numpy as np                                        # noqa: E402
import cna                                                # noqa: E402
import states as S                                        # noqa: E402
from chaord.dialects import load_dialect                  # noqa: E402
from chaord.io.frames import Frame                        # noqa: E402
from chaord.lang.errors import ChaordError                # noqa: E402
from chaord.lang.fmt import format_program                # noqa: E402
from chaord.lift import lift_frame                        # noqa: E402

NP = int(sys.argv[1])
OUT = Path("out")
dl = load_dialect(("core", "lj"))


def one(job):
    T, rho, st = job
    d = Path("/tmp") / f"ms_{os.getpid()}_{st}_{T}_{rho}"
    d.mkdir(exist_ok=True)
    pos, box = S.lattice(st, rho)
    S.write_data(d / "start.data", pos, box)
    (d / "in.l").write_text(S.reference_input(st, T, 9001, box, 20000, 40000, 8000))
    r = S.run_lammps(d)
    if r.returncode:
        shutil.rmtree(d, ignore_errors=True)
        return f"{st}|{T}|{rho}", dict(error=r.stderr[-400:])
    fr = S.read_dump(d / "prod.dump")
    rc = 1.40 * (0.95 / rho) ** (1 / 3)
    fp = [cna.classify(p, b, rc).round(3).tolist() for p, b in fr]
    p, b = fr[-1]
    f = Frame(pos=np.mod(p, b), cell=np.diag(b), symbols=["X"] * len(p))
    try:
        txt = format_program(lift_frame(f, dl, T=T))
    except ChaordError as e:
        txt = "REFUSED: " + str(e)[:300]
    except Exception as e:                                # noqa: BLE001
        txt = "ERROR: " + repr(e)[:300]
    shutil.rmtree(d, ignore_errors=True)
    return f"{st}|{T}|{rho}", dict(n=len(p), cna=fp, program=txt)


if __name__ == "__main__":
    jobs = [(T, rho, st) for T, rho in ((1.0, 0.95), (0.85, 0.95), (1.0, 0.97), (0.75, 0.93))
            for st in S.STATES]
    with mp.Pool(NP) as pool:
        out = dict(pool.map(one, jobs, chunksize=1))
    json.dump(out, open(OUT / "pilot.json", "w"), indent=1)
    for k, v in out.items():
        print("##", k, v.get("n"), v.get("error", ""), "cna first/last",
              v.get("cna", [[]])[0], v.get("cna", [[]])[-1], flush=True)
        print(v.get("program", "")[:640], flush=True)
