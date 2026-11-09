# -*- coding: utf-8 -*-
"""Macrostate selection, as PLAN.md fixes it. Four macrostates share potential,
T and density. Each is lifted to a program; every program is compiled with
physics OFF and then given the SAME LAMMPS MD; the result is scored against
every reference. A control program keeps only what physics needs (cell, T,
atom count, density, backend).

All scoring here is chaord-free (cna.py, g_of_r, cn_hist). Chaord only lifts
and builds.

Usage (repo root): python research/macrostate_selection/matrix.py <procs> [--pilot]
"""
import itertools
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "src")]
import numpy as np                                        # noqa: E402
from scipy.spatial import cKDTree                         # noqa: E402
import cna                                                # noqa: E402
import states as S                                        # noqa: E402
from chaord.build import build_program                    # noqa: E402
from chaord.dialects import load_dialect                  # noqa: E402
from chaord.io.frames import Frame                        # noqa: E402
from chaord.lang.errors import ChaordError                # noqa: E402
from chaord.lang.fmt import format_program                # noqa: E402
from chaord.lang.parser import parse_text                 # noqa: E402
from chaord.lift import lift_frame                        # noqa: E402

T, RHO, GATE = 0.75, 0.90, 1.5                             # PLAN: state point, gate
EQUIL, PROD, EVERY = 20000, 80000, 8000                   # PLAN: run lengths
N_REF, N_PROG = 6, 3                                      # PLAN: replication
RMAX, NBIN, RC_CN, RC_CNA = 4.5, 90, 1.5, 1.40 * (0.95 / RHO) ** (1 / 3)
PILOT = "--pilot" in sys.argv
BASE_REF, BASE_BUILD = (9101, 9201) if PILOT else (1001, 2001)   # PLAN: seeds
METRICS = ("gr_rms", "cn_tv", "cna_tv")
KINDS = ("full", "control")


# ---- chaord-free observables ------------------------------------------------

def observe(frames):
    """Mean g(r), coordination histogram and CNA fractions over the frames."""
    grs, cns, cnas = [], [], []
    edges = np.linspace(0, RMAX, NBIN + 1)
    shell = 4 / 3 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    for p, b in frames:
        x = np.mod(p, b)
        x[x >= b] = 0.0
        tree = cKDTree(x, boxsize=b)
        pairs = tree.query_pairs(RMAX, output_type="ndarray")
        d = x[pairs[:, 0]] - x[pairs[:, 1]]
        d -= b * np.round(d / b)
        r = np.sqrt((d ** 2).sum(1))
        n = len(x)
        grs.append(np.histogram(r, edges)[0] * 2 / (n * shell * n / np.prod(b)))
        near = pairs[r < RC_CN]
        cn = np.bincount(near.ravel(), minlength=n)
        cns.append(np.bincount(np.minimum(cn, 20), minlength=21) / n)
        cnas.append(cna.classify(p, b, RC_CNA))
    return dict(gr=np.mean(grs, 0), cn=np.mean(cns, 0), cna=np.mean(cnas, 0))


def distance(a, b):
    return dict(gr_rms=float(np.sqrt(np.mean((a["gr"] - b["gr"]) ** 2))),
                cn_tv=float(0.5 * np.abs(a["cn"] - b["cn"]).sum()),
                cna_tv=float(0.5 * np.abs(a["cna"] - b["cna"]).sum()))


def jsonable(o):
    return {k: v.tolist() for k, v in o.items()}


# ---- the MD runs ------------------------------------------------------------

def _workdir(tag):
    d = Path("/tmp") / f"ms_{os.getpid()}_{tag}"
    d.mkdir(exist_ok=True)
    return d


def _md(d):
    r = S.run_lammps(d)
    if r.returncode or not (d / "prod.dump").exists():
        raise RuntimeError("LAMMPS failed: " + (r.stderr or "")[-300:])
    return S.read_dump(d / "prod.dump")


def reference(job):
    st, i = job
    d, t0 = _workdir(f"ref_{st}_{i}"), time.perf_counter()
    try:
        pos, box = S.lattice(st, RHO)
        S.write_data(d / "start.data", pos, box)
        (d / "in.l").write_text(S.reference_input(st, T, BASE_REF + 10 * S.STATES.index(st) + i,
                                                  box, EQUIL, PROD, EVERY))
        fr = _md(d)
        return dict(state=st, i=i, obs=observe(fr), last=fr[-1], n=len(pos),
                    final_cna=cna.classify(*fr[-1], RC_CNA).tolist(),
                    bins=bin_fcc(*fr[-1]), seconds=time.perf_counter() - t0)
    except Exception:                                     # noqa: BLE001
        return dict(state=st, i=i, error=traceback.format_exc()[-500:])
    finally:
        shutil.rmtree(d, ignore_errors=True)


def bin_fcc(p, b):
    """fcc fraction in each of 10 slices along z (checks that a slab is a slab)."""
    kinds = cna.classify(p, b, RC_CNA, labels=True)
    idx = np.minimum((np.mod(p[:, 2], b[2]) / b[2] * 10).astype(int), 9)
    return [float(np.mean(kinds[idx == k] == 0)) if np.any(idx == k) else 0.0 for k in range(10)]


def control_text(prog_text, n, box):
    """PLAN: the control keeps cell, T, atom count, density and backend only."""
    return (f"chaord 0.1\ndialect core + lj\n\nsystem {{\n  cell {box[0]:.3f} {box[1]:.3f} "
            f"{box[2]:.3f}\n  pbc xyz\n  state T {T:.2f}\n  conserve atoms X {n}\n}}\n\n"
            f"physics {{\n  backend lj\n}}\n\nliquid fluid : all {{\n  state density "
            f"{RHO:.3f}\n}}\n\nresidual none\n")


def rebuild(job):
    st, kind, i, text = job
    dl = load_dialect(("core", "lj"))
    d, t0 = _workdir(f"reb_{st}_{kind}_{i}"), time.perf_counter()
    seed = BASE_BUILD + 100 * S.STATES.index(st) + 10 * KINDS.index(kind) + i
    key = (st, kind, i)
    try:
        f = build_program(parse_text(text), dl, rng=np.random.default_rng(seed), physics=False)
        box = np.diag(np.asarray(f.cell, float))
        stated = int(text.split("conserve atoms X")[1].split()[0])
        if len(f.pos) != stated:
            raise ChaordError(f"build returned {len(f.pos)} atoms, program states {stated}")
        S.write_data(d / "start.data", np.mod(np.asarray(f.pos, float), box), box)
        (d / "in.l").write_text(S.rescore_input(T, seed, EQUIL, PROD, EVERY))
        fr = _md(d)
        return dict(key=key, obs=observe(fr), final_cna=cna.classify(*fr[-1], RC_CNA).tolist(),
                    seconds=time.perf_counter() - t0)
    except ChaordError as exc:
        return dict(key=key, refused=str(exc)[:300])
    except Exception:                                     # noqa: BLE001
        return dict(key=key, error=traceback.format_exc()[-500:])
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---- main -------------------------------------------------------------------

def q90(v):
    s = sorted(v)
    return s[max(0, int(np.ceil(0.9 * len(s))) - 1)]


def mean_obs(obs):
    return {k: np.mean([o[k] for o in obs], 0) for k in obs[0]}


def main():
    procs = int(sys.argv[1])
    OUT = Path("out")
    t0 = time.perf_counter()
    dl = load_dialect(("core", "lj"))
    with mp.Pool(procs) as pool:
        refs = pool.map(reference, [(s, i) for s in S.STATES for i in range(N_REF)], chunksize=1)
    bad = [r for r in refs if "error" in r]
    if bad:
        json.dump(dict(reference_errors=bad), open(OUT / "results.json", "w"), indent=1)
        print("reference failed", bad[0]["error"])
        return
    by = {(r["state"], r["i"]): r for r in refs}

    # references: mean over N_REF trajectories, floor from their pairwise distances
    ref_mean, floor, validity = {}, {}, {}
    for s in S.STATES:
        traj = [by[(s, i)]["obs"] for i in range(N_REF)]
        ref_mean[s] = mean_obs(traj)
        ds = [distance(a, b) for a, b in itertools.combinations(traj, 2)]
        floor[s] = {m: max(float(np.mean([d[m] for d in ds])), q90([d[m] for d in ds]))
                    for m in METRICS}
        validity[s] = [by[(s, i)]["final_cna"] for i in range(N_REF)]

    # programs: lifted from the last frame of trajectories 0..N_PROG-1
    programs, tasks, refused_lift = {}, [], []
    for s in S.STATES:
        for i in range(N_PROG):
            p, b = by[(s, i)]["last"]
            f = Frame(pos=np.mod(p, b), cell=np.diag(b), symbols=["X"] * len(p))
            try:
                full = format_program(lift_frame(f, dl, T=T))
            except ChaordError as exc:
                refused_lift.append(dict(state=s, i=i, reason=str(exc)[:300]))
                continue
            ctrl = format_program(parse_text(control_text(full, len(p), b)))
            programs[f"{s}|{i}"] = dict(full=full, control=ctrl)
            tasks += [(s, "full", i, full), (s, "control", i, ctrl)]
    with mp.Pool(procs) as pool:
        done = pool.map(rebuild, tasks, chunksize=1)

    groups, problems = {}, list(refused_lift)
    for d in done:
        st, kind, i = d["key"]
        if "obs" in d:
            groups.setdefault((st, kind), []).append(d)
        else:
            problems.append(dict(state=st, kind=kind, i=i,
                                 reason=d.get("refused") or d.get("error")))

    cells = {}
    for (st, kind), ds in groups.items():
        if len(ds) != N_PROG:
            continue
        mean = mean_obs([d["obs"] for d in ds])
        for j in S.STATES:
            dist = distance(mean, ref_mean[j])
            ratio = {m: dist[m] / max(floor[j][m], 1e-9) for m in METRICS}
            cells[f"{st}|{kind}|{j}"] = dict(distance=dist, ratio=ratio,
                                             match=all(r <= GATE for r in ratio.values()))
    report = {k: [d["final_cna"] for d in v] for k, v in
              (("|".join(g), groups[g]) for g in groups)}

    def m(i, kind, j):
        return cells.get(f"{i}|{kind}|{j}", {}).get("match")

    # PLAN: decision rule
    non_liquid = [s for s in S.STATES if s != "liquid"]
    checks = dict(
        no_refusals=not problems,
        references_valid=(
            all(max(x[0], x[1]) < 0.01 for x in validity["liquid"])
            and all(x[0] > 0.10 and x[1] < 0.01 for x in validity["fcc"])
            and all(x[1] > 0.10 and x[0] < 0.01 for x in validity["hcp"])
            and all(max(by[("coex", i)]["bins"]) >= 0.10
                    and sum(v < 0.01 for v in by[("coex", i)]["bins"]) >= 3
                    for i in range(N_REF))),
        diagonal=all(m(s, "full", s) is True for s in S.STATES),
        off_diagonal=all(m(s, "full", j) is False
                         for s in S.STATES for j in S.STATES if j != s),
        control_liquid=all(m(s, "control", "liquid") is True for s in S.STATES),
        control_not_source=all(m(s, "control", s) is False for s in non_liquid),
    )
    verdict = "SUPPORTED" if all(checks.values()) else "NOT SUPPORTED"
    res = dict(plan="research/macrostate_selection/PLAN.md", pilot=PILOT, T=T, rho=RHO, gate=GATE,
               seeds=dict(reference=BASE_REF, build=BASE_BUILD), programs=programs,
               floor=floor, cells=cells, problems=problems, reference_final_cna=validity,
               rebuild_final_cna=report, checks=checks, verdict=verdict,
               lammps="PyPI wheel lammps 2024.8.29.3.0, serial",
               wall_seconds=round(time.perf_counter() - t0, 1))
    json.dump(res, open(OUT / ("pilot2.json" if PILOT else "results.json"), "w"), indent=1)

    print("floors", json.dumps(floor))
    for kind in KINDS:
        print(f"--- {kind}: rows source, columns reference; max ratio over metrics")
        print(f"{'':8s}" + "".join(f"{j:>10s}" for j in S.STATES))
        for s in S.STATES:
            row = [cells.get(f"{s}|{kind}|{j}") for j in S.STATES]
            print(f"{s:8s}" + "".join(
                (f"{max(c['ratio'].values()):9.1f}{'*' if c['match'] else ' '}" if c else "      none")
                for c in row))
    print("problems", json.dumps(problems)[:1500])
    print("checks", json.dumps(checks))
    print("verdict", verdict, "wall", res["wall_seconds"])


if __name__ == "__main__":
    main()
