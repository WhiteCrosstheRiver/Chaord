# -*- coding: utf-8 -*-
"""The program-length curve, exactly as research/program_length/PLAN.md
(registered in 5b434b8) fixes it. Nothing here is a free choice; each line
points at the PLAN section it implements.

Grid: 2 cases x 3 physics levels x 3 program levels, each cell rebuilt with
seeds 7, 13, 29 and scored with tools/acceptance.py's own A5 helpers
(`_reference_cases`, `_a5_obs`, `_mean_observables`, `_a5_effective_floor`)
and chaord.cv.noise.distance. Writes results.json (every cell: length,
ratios, angle_tv, refusals, seconds) and prints the PLAN decision.

Usage (repo root):  python research/program_length/curve.py --procs N [--commit SHA]
"""
import argparse
import json
import multiprocessing as mp
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT / "src"), str(ROOT / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import acceptance as A                                   # noqa: E402
from chaord.build import build_program                   # noqa: E402
from chaord.cv.noise import distance, observables        # noqa: E402
from chaord.dialects import load_dialect                 # noqa: E402
from chaord.io.frames import read_frame                  # noqa: E402
from chaord.lang.errors import ChaordError               # noqa: E402
from chaord.lang.fmt import format_program               # noqa: E402
from chaord.lang.parser import parse_text                # noqa: E402

CASES = ("lj_liquid", "lj_liquid_large")                 # PLAN: Systems
SEEDS = (7, 13, 29)                                      # PLAN: Measurement
GATE = 1.5                                               # PLAN: Measurement
PHYS = ("P0", "P1", "P2")                                # PLAN: Physics levels
PROG = ("S0", "S1", "S2")                                # PLAN: Program ladder


# ---- program ladder (PLAN: Program ladder) ----------------------------------

def ladder(lifted: str) -> dict:
    """S0/S1/S2 from the lifted text: asserts and the provenance block are
    removed at every level; S1 adds `constrain cn` copied verbatim from the
    lifted `assert cn`; S2 adds `constrain gr_peak <r> height <h>` from the
    lifted `assert gr_peak` (height tolerance left to the dialect default)."""
    m_cn = re.search(r"^\s*assert cn (\S+) \+- (\S+) cutoff (\S+)\s*$", lifted, re.M)
    m_gr = re.search(r"^\s*assert gr_peak (\S+) height (\S+)\s*$", lifted, re.M)
    if not (m_cn and m_gr):
        raise ChaordError("lifted program lacks the assert cn / gr_peak lines")
    base = re.sub(r"\nprovenance \{.*?\n\}\n?", "\n", lifted, flags=re.S)
    base = "\n".join(l for l in base.splitlines()
                     if not l.strip().startswith("assert "))
    cn = f"  constrain cn {m_cn.group(1)} +- {m_cn.group(2)} cutoff {m_cn.group(3)}"
    gr = f"  constrain gr_peak {m_gr.group(1)} height {m_gr.group(2)}"

    def with_lines(extra):
        # insert into the liquid block, right after its `state density` line
        out, done = [], False
        for line in base.splitlines():
            out.append(line)
            if not done and line.strip().startswith("state density"):
                out.extend(extra)
                done = True
        if not done:
            raise ChaordError("no liquid `state density` line to anchor to")
        return format_program(parse_text("\n".join(out) + "\n"))

    return {"S0": with_lines([]), "S1": with_lines([cn]), "S2": with_lines([cn, gr])}


# ---- one rebuild -------------------------------------------------------------

def md_steps_of(level: str, dl):
    """PLAN: P0 physics off; P1 10% of md.relax_steps; P2 dialect default."""
    if level == "P1":
        return int(round(0.1 * int(dl.threshold("md")["relax_steps"])))
    return None


def one_build(task):
    case, phys, prog, seed, text, dial = task
    dl = load_dialect(tuple(dial))
    t0 = time.perf_counter()
    try:
        frame = build_program(parse_text(text), dl, rng=np.random.default_rng(seed),
                              physics=(phys != "P0"), md_steps=md_steps_of(phys, dl))
    except ChaordError as exc:
        return dict(key=(case, phys, prog, seed), refused=str(exc)[:300],
                    seconds=time.perf_counter() - t0)
    except Exception:                                    # noqa: BLE001
        return dict(key=(case, phys, prog, seed),
                    error=traceback.format_exc()[-600:],
                    seconds=time.perf_counter() - t0)
    obs = A._a5_obs(frame, dl)
    ang = observables(frame, dl, selection=("gr", "angle"))
    return dict(key=(case, phys, prog, seed), obs=obs, ang=ang,
                seconds=time.perf_counter() - t0)


# ---- scoring -----------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--commit", default="unknown")
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--smoke", action="store_true",
                    help="P0 only, one seed: pipeline test, never for the record")
    args = ap.parse_args()

    floors = json.loads(A.NOISE_FLOORS.read_text(encoding="utf-8"))
    refs = {c["id"].split("/")[-1]: c
            for c in A._reference_cases(ROOT / "bench" / "reference", floors)}
    phys_levels = ("P0",) if args.smoke else PHYS
    seeds = (7,) if args.smoke else SEEDS

    texts, ref_obs, ref_ang, floor_eff, dials = {}, {}, {}, {}, {}
    for case in CASES:
        c = refs[case]
        dl = load_dialect(c["dialect"])
        dials[case] = c["dialect"]
        lifted = (HERE / "lifted" / f"{case}.chaord").read_text(encoding="utf-8")
        texts[case] = ladder(lifted)
        avg = c["floor"]["avg"]                           # averaged A5 protocol
        frames = [read_frame(c["frames"][i]) for i in avg["ref_frames"]]
        ref_obs[case] = A._mean_observables([A._a5_obs(f, dl) for f in frames])
        ref_ang[case] = A._mean_observables(
            [observables(f, dl, selection=("gr", "angle")) for f in frames])
        floor_eff[case] = A._a5_effective_floor(avg)

    tasks = [(case, p, s, seed, texts[case][s], dials[case])
             for case in CASES for p in phys_levels for s in PROG for seed in seeds]
    print(f"[curve] {len(tasks)} rebuilds on {args.procs} processes", flush=True)
    t0 = time.perf_counter()
    with mp.Pool(args.procs) as pool:
        done = pool.map(one_build, tasks, chunksize=1)
    by = {d["key"]: d for d in done}

    cells = {}
    for case in CASES:
        dl = load_dialect(dials[case])
        for p in phys_levels:
            for s in PROG:
                ds = [by[(case, p, s, seed)] for seed in seeds]
                cell = dict(case=case, physics=p, program=s,
                            length=len(texts[case][s]),
                            md_steps=(0 if p == "P0" else md_steps_of(p, dl)
                                      or int(dl.threshold("md")["relax_steps"])),
                            seconds=round(sum(d["seconds"] for d in ds), 1))
                bad = [d for d in ds if "obs" not in d]
                if bad:
                    cell.update(passed=False,
                                refusals=[d.get("refused") or d.get("error")
                                          for d in bad])
                else:
                    o_rb = A._mean_observables([d["obs"] for d in ds])
                    dist = distance(ref_obs[case], o_rb, dialect=dl)
                    ratio = {k: dist[k] / max(floor_eff[case][k], 1e-6)
                             for k in ("gr_rms", "cn_tv")}
                    a_rb = A._mean_observables([d["ang"] for d in ds])
                    atv = distance(ref_ang[case], a_rb, dialect=dl,
                                   keys=("angle_tv",))["angle_tv"]
                    cell.update(distance={k: dist[k] for k in ("gr_rms", "cn_tv")},
                                floor=floor_eff[case], ratio=ratio,
                                passed=all(ratio[k] <= GATE for k in ratio),
                                angle_tv=atv, refusals=[])
                cells[f"{case}|{p}|{s}"] = cell
                print(f"[curve] {case} {p} {s}: len {cell['length']} "
                      + (f"gr x{cell['ratio']['gr_rms']:.2f} cn x{cell['ratio']['cn_tv']:.2f}"
                         f" angle_tv {cell['angle_tv']:.4f}" if "ratio" in cell
                         else f"REFUSED {cell['refusals'][0][:80]}")
                      + f"  -> {'PASS' if cell['passed'] else 'FAIL'}", flush=True)

    # held-out check relative to P2/S0 (PLAN: Known weakness)
    for case in CASES:
        base = cells.get(f"{case}|P2|S0", {}).get("angle_tv")
        for p in phys_levels:
            for s in PROG:
                c = cells[f"{case}|{p}|{s}"]
                if base and "angle_tv" in c:
                    c["angle_tv_rel_P2S0"] = c["angle_tv"] / base

    # decision (PLAN: Outcome and decision rule)
    shortest = {}
    for case in CASES:
        shortest[case] = {}
        for p in phys_levels:
            ok = [s for s in PROG if cells[f"{case}|{p}|{s}"]["passed"]]
            shortest[case][p] = (dict(level=ok[0],
                                      length=cells[f"{case}|{p}|{ok[0]}"]["length"])
                                 if ok else dict(level="none", length=None))

    def as_num(x):            # "none" (no level passes) sorts as longest
        return float("inf") if x["length"] is None else x["length"]

    verdict = None
    if not args.smoke:
        per_case = {}
        for case in CASES:
            L = [as_num(shortest[case][p]) for p in PHYS]
            per_case[case] = dict(non_increasing=L[0] >= L[1] >= L[2],
                                  some_level_longer_than_P2=max(L[0], L[1]) > L[2])
        supported = all(v["non_increasing"] and v["some_level_longer_than_P2"]
                        for v in per_case.values())
        verdict = dict(per_case=per_case,
                       verdict="SUPPORTED" if supported else "NOT SUPPORTED")

    res = dict(plan="research/program_length/PLAN.md (registered 5b434b8)",
               commit=args.commit, smoke=args.smoke, seeds=list(seeds), gate=GATE,
               programs={c: texts[c] for c in CASES}, floors=floor_eff,
               cells=cells, shortest_passing=shortest, decision=verdict,
               wall_seconds=round(time.perf_counter() - t0, 1))
    out = Path(args.out) / ("results_smoke.json" if args.smoke else "results.json")
    out.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps(shortest))
    print(f"[curve] decision: {verdict and verdict['verdict']}")


if __name__ == "__main__":
    main()
