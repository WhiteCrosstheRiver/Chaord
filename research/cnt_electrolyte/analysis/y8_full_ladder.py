# -*- coding: utf-8 -*-
"""Y8 go/no-go analysis -- the FULL stage-B field ladder
(docs/reviews/open_items_v4.md Y8; D31/D32 binding; floor estimator D15;
block protocol of reports/research/stage_b_plan.md section 2-3).

This is the full-ladder pass that supersedes the tranche-1 analysis
(y8_block_analysis.py, kept untouched as the committed template and
reproducer of tranche1_results.json).  The two arms, the statement
feature parser, the D15 floor estimator and the LORO classifier are
imported from it unchanged, so the full-ladder numbers are produced by
the same code paths tranche 1 was.

THE RECORD (verified at run time, see discover_jobs):

  42 jobs = 2 tubes x 7 fields (0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2 V/A)
  x 3 independent builds b1/b2/b3.  Two provenance chains produced it:

  - 12 LOCAL full-density runs (stageB/manifests/sb_cnt10_{e0,e10,e50,
    e100}_r{1,2,3}.json, DONE markers, non-smoke): stageB/jobs/<name>/
    prod.dump, 12,001 frames at 1 ps.  Local LAMMPS 22 Jul 2025.
  - 33 CLUSTER thinned runs (stageB/sb_<tube>_e<mv>_r<k>/): the same
    generated inputs, thinned record -- prod_thinned.dump holds EVERY
    12th production frame (12 ps spacing, 1,001 frames covering both
    6-ns blocks; timesteps 0..12,000,000).  Cluster LAMMPS 3 Mar 2020.
    sb_cnt10_e100 exists in BOTH chains (identical seeds and step-0
    thermo, different LAMMPS builds); the LOCAL full-density copy is the
    canonical source and the cluster copy is used as an independent
    cross-check (same job, different build -> a determinism check).

LIFT SUBSAMPLING (stated, measured, not assumed): a lift costs 0.10 s
(cnt10, 971 atoms) to 0.17 s (cnt16, 2058 atoms) on this box (measured),
so the task brief's 20-30 s/lift worst case does not bind: EVERY frame
of the committed record is lifted at the 12-ps thinned cadence --
1,001 lifts per job, 500 in block A + 501 in block B, 42,042 lifts
total.  No analysis subsampling beyond the record's own thinning is
applied.  For the 12 local jobs the record is 12x denser than the
analysis cadence; their full 12,001-frame density is used by arm 2
(dense), and the dense-vs-subsampled block-mean agreement is recorded,
which quantifies what the 12-ps cadence costs.  Every block mean rests
on >= 500 frames (arm 2 local dense: 6,000).

BLOCKS (stage_b_plan.md section 2, verified against the dump timesteps):
production time runs 0..12,000 ps after the 1-ns settle; block A =
t < 6000 ps, block B = t >= 6000 ps.  Block B is PRIMARY (section 3.2:
block A of a field-on run retains ~27-31% of the switch-on transient);
block A is disclosed as transient-biased and used in the sensitivity
checks only.  The e0 runs get identical treatment (field stays 0, both
blocks clean) and are the floor side.

FLOORS (D15 estimator, D31 pair set -- NOW LIKE-FOR-LIKE): per tube and
statement, over the e0 BLOCK MEANS of THIS batch (6 ns x 2 blocks x 3
runs per tube): every pair of blocks from DIFFERENT runs (12 cross-run
pairs per tube), floor = max(mean |Delta|, P90 |Delta|).  These
supersede Stage A's 1-ns zf floors (tranche-1 finding: short-block
scatter inflated them ~sqrt(6)); the e0 blocks have exactly the length
and cadence of the field-on blocks, so no length correction remains.
ALSO computed: the build-paired secondary view (per build: field-on
block-B mean minus the SAME build's e0 all-block mean) -- disclosed,
never the ordered gate.

DETECTION (two criteria, BOTH reported -- the choice between them is
the owner's, see go_no_go.md):

  A. the ordered unpaired test (D31/D32 as tranche 1 ran it):
     |mean_on(B) - mean_off| vs 3 x floor, runs the units (sem over the
     3 runs), floor = D15 cross-run e0 floor.
  B. the build-paired contrast: per build, on(B) - e0 all-block mean of
     the same build; detected when |mean| exceeds 3x its sem over the 3
     builds (2x also recorded).  Removes the shared per-build zero-field
     offset; assumes the offset survives into the field-on run.

Both criteria are computed for the Chaord arm (blind dipole_p1
statement) and the direct arm (P1 without chaord, dense) -> E_min per
tube per arm per criterion.  LORO classifier accuracies per field.
Linear response over 0.005-0.02 V/A (three field points + the zero).

Every number is deterministic.  Output:
research/cnt_electrolyte/analysis/full_ladder_results.json plus a
console summary.  Per-job progress is checkpointed to
full_ladder_checkpoint.jsonl (one JSON line per completed job) so an
interrupted run resumes without recomputing finished jobs.

Usage (from the repository root):
    python research/cnt_electrolyte/analysis/y8_full_ladder.py
    python research/cnt_electrolyte/analysis/y8_full_ladder.py --smoke
        # 4 frames per job, pipeline test only, never for the record
    python research/cnt_electrolyte/analysis/y8_full_ladder.py --fresh
        # ignore any existing checkpoint
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parents[1]          # research/cnt_electrolyte
for _p in (str(ROOT / "src"), str(HERE), str(HERE / "analysis")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# the tranche-1 machinery, unchanged (arms, parser, estimator, classifier)
from y8_block_analysis import (                        # noqa: E402
    classify_frame_census, build_file_census, d15_floor, loro_classifier,
    stream_frames, y8_statement_features)
from analyze_first_result import (                     # noqa: E402
    composition_dialect, lift_one_frame)
from chaord.io.lammps import read_lammps_data, read_lammps_dump  # noqa: E402

OUT_JSON = Path(__file__).resolve().parent / "full_ladder_results.json"
CHECKPOINT = Path(__file__).resolve().parent / "full_ladder_checkpoint.jsonl"

# ---- full-ladder constants (verified against the dumps at run time) ----
TUBES = (10, 16)
FIELDS = (0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2)   # V/A; e<mv> = 1000x
FIELD_OF_MV = {0: 0.0, 5: 0.005, 10: 0.01, 20: 0.02, 50: 0.05,
               100: 0.1, 200: 0.2}
WORKING_FIELD = 0.01                                 # D30 default
CADENCE_PS = 12.0                                    # the thinned record
LOCAL_CADENCE_PS = 1.0                               # the local full dumps
SPLIT_PS = 6000.0                                    # block A | B
THINNED_FRAMES = 1001                                # per job, both blocks
LOCAL_FRAMES = 12001                                 # per local job, 1 ps
DT_PS = 1.0                                          # production dump base
CHARGE_TOL = 1e-6
N_RUNS = 3
ATOMS_PER = {"EC": 10, "EMC": 15, "PF6": 7, "Li": 1}  # stageA convention


# ---------------------------------------------------------------------------
# provenance: which jobs, where their canonical record lives
# ---------------------------------------------------------------------------

def discover_jobs() -> list[dict]:
    """The 42 ladder jobs with their canonical dump paths.

    Canonical assignment FROZEN to the Y8 work-order state (the record
    the go/no-go is written against):

    - LOCAL full-density: cnt10 fields {0, 0.01, 0.05, 0.1} x 3 runs
      (stageB/manifests + DONE non-smoke): stageB/jobs/<name>/prod.dump,
      12,001 frames at 1 ps.  Local LAMMPS 22 Jul 2025.
    - CLUSTER thinned: the 33 stageB/sb_* dirs with prod_thinned.dump
      (EVERY 12th production frame, 12 ps, 1,001 frames) and a finished
      log.lammps.  Cluster LAMMPS 3 Mar 2020.

    A job present in BOTH chains takes the frozen canonical source and
    uses the other copy as a determinism cross-check.  Anomaly handled
    explicitly: a concurrent local stageB queue (pid recorded in
    stageB/QUEUE_RUNNING) completed cnt10 e5 locally WHILE this analysis
    ran (new DONE markers + manifests appeared mid-run); per the freeze,
    its canonical source stays the cluster thinned copy and the local
    full-density copy is the cross-check.  The queue's e20/e200 local
    runs are still in flight -- they are not part of any record here.
    """
    index = json.loads((HERE / "stageB" / "jobs_index.json").read_text())
    by_name = {e["name"]: e for e in index}
    assert len(by_name) == 2 * len(FIELDS) * N_RUNS, len(by_name)

    zfA = json.loads((HERE / "stageA" / "jobs_index.json").read_text())
    build_data = {}
    for e in zfA:
        if e["kind"] == "zf" and not e["name"].endswith("m"):
            build_data[(e["tube"], e["build"])] = HERE / e["source"]

    def local_done_of(name: str) -> bool:
        p = HERE / "stageB" / "jobs" / name / "DONE"
        return p.is_file() and not json.loads(p.read_text())["smoke"]

    def cluster_ok(d: Path) -> bool:
        return (d / "prod_thinned.dump").is_file() and "Total wall time" \
            in (d / "log.lammps").read_text(errors="replace")

    jobs = []
    n_local = n_cluster = n_extra_local = 0
    for tube in TUBES:
        for mv, field in FIELD_OF_MV.items():
            for run in range(1, N_RUNS + 1):
                name = f"sb_cnt{tube}_e{mv}_r{run}"
                e = by_name.pop(name)
                assert e["tube"] == tube and e["field"] == field, name
                assert e["n_blocks"] == 2 and e["block_ps"] == 6000.0, name
                bd = build_data[(tube, e["build"])]
                assert bd.is_file(), (name, bd)

                local_dir = HERE / "stageB" / "jobs" / name
                cluster_dir = HERE / "stageB" / f"sb_cnt{tube}_e{mv}_r{run}"
                has_local, has_cluster = local_done_of(name), \
                    cluster_ok(cluster_dir)
                # the freeze: local full-density only for the cnt10 fields
                # that were local when the work order was written
                canonical_local = has_local and tube == 10 and \
                    field in (0.0, 0.01, 0.05, 0.1)
                if canonical_local:
                    man = json.loads((HERE / "stageB" / "manifests"
                                      / f"{name}.json").read_text())
                    assert man["job"] == name and man["build"] == e["build"], \
                        name
                    assert float(man["field"]) == field, name
                    dump, dup = local_dir / "prod.dump", None
                    if has_cluster:
                        dup = cluster_dir / "prod_thinned.dump"
                    src = "local_full"
                    n_local += 1
                else:
                    assert has_cluster, f"{name}: no finished cluster record"
                    dump, dup = cluster_dir / "prod_thinned.dump", None
                    if has_local:
                        dup = local_dir / "prod.dump"   # e.g. the late
                        n_extra_local += 1              # local e5 finishes
                    src = "cluster_thinned"
                    n_cluster += 1
                jobs.append({
                    "name": name, "tube": tube, "field": field,
                    "run": run, "build": e["build"], "kind": "sb",
                    "source_kind": src, "dump": dump, "build_data": bd,
                    "local": src == "local_full",
                    "duplicate": dup,
                    "duplicate_kind": ("cluster_thinned" if src == "local_full"
                                       else "local_full") if dup else None,
                })
    assert not by_name, sorted(by_name)
    assert len(jobs) == 2 * len(FIELDS) * N_RUNS
    assert (n_local, n_cluster) == (12, 30), (n_local, n_cluster)   # 33 cluster dirs, 3 of them the e100 duplicates
    print(f"[y8] canonical record: {n_local} local full + {n_cluster} "
          f"cluster thinned; {n_extra_local} extra local duplicate(s) "
          f"cross-checked, not canonical", flush=True)
    return jobs


def lmp_version(log_path: Path) -> str:
    with open(log_path, errors="replace") as f:
        for line in f:
            if line.startswith("LAMMPS ("):
                return line.strip()
    return "unknown"


# ---------------------------------------------------------------------------
# arm 2 -- direct P1 without chaord (imported streamer; adapted driver)
# ---------------------------------------------------------------------------

def direct_arm(job: dict, sub: list[int]) -> dict:
    """P1 of the solvent on EVERY frame of the canonical dump (dense),
    with the per-frame census/charge verification; plus the series at
    exactly the lift arm's frame indices (the 12-ps cadence)."""
    want_atoms, want_census, type_mass = build_file_census(job["build_data"])
    p1, p1_ec, p1_emc, prod_ps = [], [], [], []
    t_first = t_last = None
    for k, t, box, a in stream_frames(job["dump"]):
        if not p1:
            t_first = t
        t_last = t
        assert len(a) == want_atoms, (job["name"], k, len(a), want_atoms)
        assert abs(a[:, 3].sum()) < CHARGE_TOL, (job["name"], k, "charge")
        census = classify_frame_census(a[:, 1].astype(int), a[:, 2],
                                       type_mass)
        assert census == want_census, (job["name"], k, census)
        L = box[:, 1] - box[:, 0]
        cos, cos_ec, cos_emc = [], [], []
        for m in np.unique(a[:, 1].astype(int)):
            idx = np.where(a[:, 1].astype(int) == m)[0]
            n_at = len(idx)
            if n_at not in (ATOMS_PER["EC"], ATOMS_PER["EMC"]):
                continue
            d = a[idx, 4:7] - a[idx[0], 4:7]
            d -= L * np.round(d / L)
            mu = (a[idx, 3][:, None] * d).sum(0)
            c = float(mu[2] / np.linalg.norm(mu))
            cos.append(c)
            (cos_ec if n_at == ATOMS_PER["EC"] else cos_emc).append(c)
        p1.append(float(np.mean(cos)))
        p1_ec.append(float(np.mean(cos_ec)))
        p1_emc.append(float(np.mean(cos_emc)))
        prod_ps.append(t / 1000.0)
    n_frames = len(p1)
    if job["source_kind"] == "local_full":
        assert n_frames == LOCAL_FRAMES, (job["name"], n_frames)
        dts = np.diff(prod_ps)
        assert np.allclose(dts, LOCAL_CADENCE_PS), (job["name"], dts[:5])
        assert sub == list(range(0, n_frames, int(CADENCE_PS))) or len(sub) \
            < 10, (job["name"], "lift indices not on the 12-ps grid")
    else:
        assert n_frames == THINNED_FRAMES, (job["name"], n_frames)
        dts = np.diff(prod_ps)
        assert np.allclose(dts, CADENCE_PS), (job["name"], dts[:5])
        assert prod_ps[0] == 0.0 and prod_ps[-1] == 12000.0, job["name"]
        assert sub == list(range(n_frames)) or len(sub) < 10, job["name"]
    return {
        "n_frames": n_frames, "t_first": t_first, "t_last": t_last,
        "prod_ps": np.array(prod_ps),
        "census": want_census, "n_atoms": want_atoms,
        "p1_dense": np.array(p1), "p1_ec_dense": np.array(p1_ec),
        "p1_emc_dense": np.array(p1_emc),
        "sub": sub,
        "p1_sub": np.array(p1)[sub],
    }


def direct_arm_duplicate(job: dict) -> dict:
    """Arm-2 P1 on the NON-canonical copy of a job run in both chains
    (same seeds and start, different LAMMPS builds) -- the determinism
    cross-check.  Handles either thinned (12 ps) or full (1 ps) dumps;
    returns its dense series on its own prod-ps grid."""
    p1, prod_ps = [], []
    for k, t, box, a in stream_frames(job["duplicate"]):
        assert abs(a[:, 3].sum()) < CHARGE_TOL, (job["name"], k, "charge")
        L = box[:, 1] - box[:, 0]
        cos = []
        for m in np.unique(a[:, 1].astype(int)):
            idx = np.where(a[:, 1].astype(int) == m)[0]
            if len(idx) not in (ATOMS_PER["EC"], ATOMS_PER["EMC"]):
                continue
            dvec = a[idx, 4:7] - a[idx[0], 4:7]
            dvec -= L * np.round(dvec / L)
            mu = (a[idx, 3][:, None] * dvec).sum(0)
            cos.append(float(mu[2] / np.linalg.norm(mu)))
        p1.append(float(np.mean(cos)))
        prod_ps.append(t / 1000.0)
    return {"n_frames": len(p1), "prod_ps": np.array(prod_ps),
            "p1_dense": np.array(p1)}


# ---------------------------------------------------------------------------
# arm 1 -- the Chaord lift (Y4 reader -> blind lift -> statement features)
# ---------------------------------------------------------------------------

def chaord_arm(job: dict, sub: list[int], prod_ps: np.ndarray, dl) -> dict:
    """Arm 1 on one job: Y4-read the 12-ps-cadence frames, lift each with
    identical blind options, parse the statement features.  sub are dump
    frame ordinals; prod_ps their production times (ps)."""
    tmap = read_lammps_data(str(job["build_data"]))[1]["type_map"]
    frames = read_lammps_dump(str(job["dump"]), frame_indices=sub,
                              type_map=tmap)
    unknown = tmap.apply(frames, strict=True)
    assert not unknown, f"{job['name']}: unmapped type ids {unknown}"
    assert len(frames) == len(sub)
    feats_list, consts_list, failed = [], [], []
    p1_text = []
    for fr, ps in zip(frames, prod_ps):
        assert abs(fr.info["timestep"] / 1000.0 - ps) < 1e-9, \
            (job["name"], fr.info["timestep"], ps)
        assert abs(np.asarray(fr.info["charges"]).sum()) < CHARGE_TOL
        text, info = lift_one_frame(fr, None, dl=dl)   # T=300, blind, auto
        feats, consts = y8_statement_features(text)
        feats_list.append(feats)
        consts_list.append(json.dumps(consts, sort_keys=True))
        if info["failed_checks"]:
            failed.append({"prod_ps": float(ps),
                           "checks": info["failed_checks"]})
        p1_text.append(feats["dipole_p1"])
    if failed:
        print(f"[y8] WARNING {job['name']}: {len(failed)} lifted frame(s) "
              f"with failed static checks -- recorded in the JSON",
              file=sys.stderr)
    assert len(set(consts_list)) == 1, \
        f"{job['name']}: the program's non-moving statements move between " \
        f"frames"
    keys = sorted(set().union(*[set(f) for f in feats_list]))
    counts = {k2: sum(k2 in f for f in feats_list) for k2 in keys}
    assert all(c == len(frames) for c in counts.values()), \
        f"{job['name']}: statement presence varies across frames: {counts}"
    return {
        "n_lifted": len(frames),
        "prod_ps": [float(x) for x in prod_ps],
        "consts": json.loads(consts_list[0]),
        "features": {k2: [f[k2] for f in feats_list] for k2 in keys},
        "failed_checks": failed,
        "p1_text": p1_text,
    }


# ---------------------------------------------------------------------------
# blocks, levels, paired view, detection
# ---------------------------------------------------------------------------

def blocks_from_ps(values: np.ndarray, prod_ps: np.ndarray) -> dict:
    """Block means over frames: A = t < SPLIT_PS, B = t >= SPLIT_PS."""
    a = values[prod_ps < SPLIT_PS]
    b = values[prod_ps >= SPLIT_PS]
    return {"A": float(np.nanmean(a)), "B": float(np.nanmean(b)),
            "nA": int(np.sum(~np.isnan(a))), "nB": int(np.sum(~np.isnan(b)))}


def run_level(per_run_B: list[float]) -> dict:
    v = np.array(per_run_B, float)
    return {"mean": float(v.mean()),
            "sem": float(v.std(ddof=1) / math.sqrt(len(v))),
            "per_run": [float(x) for x in v]}


def e0_level(e0_res: list[dict], key: str) -> dict:
    """mean over the 6 e0 blocks + the run-level sem (3 run means)."""
    blocks, per_run = [], []
    for r in e0_res:
        b = r["blocks"][key]
        blocks += [b["A"], b["B"]]
        per_run.append((b["A"] * b["nA"] + b["B"] * b["nB"])
                       / (b["nA"] + b["nB"]))
    v = np.array(per_run, float)
    return {"mean": float(np.mean(blocks)),
            "sem": float(v.std(ddof=1) / math.sqrt(len(v))),
            "per_run": [float(x) for x in v],
            "blocks": [float(x) for x in blocks]}


def detection_entry(on: dict, off: dict, floor: float) -> dict:
    delta = abs(on["mean"] - off["mean"])
    return {
        "mean_on_B": on["mean"], "sem_on_B": on["sem"],
        "per_run_B": on["per_run"],
        "mean_off": off["mean"], "sem_off": off["sem"],
        "per_run_off": off["per_run"],
        "delta": delta, "floor": floor,
        "floor_zero": bool(floor == 0),
        "ratio": delta / floor if floor > 0 else float("inf"),
        "ratio_to_3x_floor": delta / (3 * floor) if floor > 0
        else float("inf"),
        "pass_3x": bool(delta > 3 * floor),
        "z": delta / math.hypot(on["sem"], off["sem"])
        if (on["sem"] or off["sem"]) else float("inf"),
    }


def paired_view(sb_res: list[dict], e0_res: list[dict], key: str) -> dict:
    """Build-paired contrast: run rK's block-B mean minus build K's e0
    all-block mean.  Secondary diagnostic -- the criterion choice is the
    owner's; both the 2-sigma and 3-sigma detections are recorded, plus
    the strict D15-style floor over the paired deltas themselves
    (|paired_i - paired_j| across builds, 3 pairs)."""
    e0_by_build = {}
    for r in e0_res:
        b = r["blocks"][key]
        e0_by_build[r["build"]] = ((b["A"] * b["nA"] + b["B"] * b["nB"])
                                   / (b["nA"] + b["nB"]))
    pairs = [r["blocks"][key]["B"] - e0_by_build[r["build"]] for r in sb_res]
    v = np.array(pairs, float)
    mean = float(v.mean())
    sem = float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else 0.0
    dd = [abs(v[i] - v[j]) for i in range(len(v))
          for j in range(i + 1, len(v))]
    pfloor = float(max(np.mean(dd), np.percentile(dd, 90))) if dd else 0.0
    return {
        "per_build": [float(x) for x in v],
        "builds": [r["build"] for r in sb_res],
        "e0_by_build": {k3: float(x) for k3, x in e0_by_build.items()},
        "mean": mean, "sem": sem,
        "z": mean / sem if sem > 0 else float("inf"),
        "pass_2s": bool(sem > 0 and abs(mean) > 2 * sem),
        "pass_3s": bool(sem > 0 and abs(mean) > 3 * sem),
        "paired_floor": pfloor,
        "ratio_to_paired_floor": abs(mean) / pfloor if pfloor > 0
        else float("inf"),
    }


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------

def job_record(job: dict, d: dict, c: dict) -> dict:
    """The per-job store: blocks for every statement, both arms' series."""
    sub_idx = np.array(c["prod_ps"], float)
    blocks = {k: blocks_from_ps(np.array(v), sub_idx)
              for k, v in c["features"].items()}
    agree = np.abs(np.array(c["p1_text"]) - d["p1_sub"])
    dense_ps = d["prod_ps"]
    selA, selB = dense_ps < SPLIT_PS, dense_ps >= SPLIT_PS
    p1_blocks = {"A": float(d["p1_dense"][selA].mean()),
                 "B": float(d["p1_dense"][selB].mean()),
                 "nA": int(selA.sum()), "nB": int(selB.sum())}
    sub_ps = np.array(c["prod_ps"], float)
    sselA, sselB = sub_ps < SPLIT_PS, sub_ps >= SPLIT_PS
    p1_sub_blocks = {"A": float(d["p1_sub"][sselA].mean()),
                     "B": float(d["p1_sub"][sselB].mean()),
                     "nA": int(sselA.sum()), "nB": int(sselB.sum())}
    return {
        "name": job["name"], "tube": job["tube"], "field": job["field"],
        "run": job["run"], "build": job["build"],
        "source": job["source_kind"],
        "dump": str(job["dump"]), "build_data": str(job["build_data"]),
        "duplicate": str(job["duplicate"]) if job["duplicate"] else None,
        "duplicate_kind": job["duplicate_kind"],
        "n_atoms": d["n_atoms"], "census": d["census"],
        "consts": c["consts"], "n_lifted": c["n_lifted"],
        "n_frames_dense": d["n_frames"],
        "cadence_ps": (LOCAL_CADENCE_PS if job["local"] else CADENCE_PS),
        "blocks": blocks,
        "failed_checks": c["failed_checks"],
        "p1_text_vs_direct_max_abs": float(agree.max()),
        "direct_p1_blocks": p1_blocks,
        "direct_p1_sub_blocks": p1_sub_blocks,
        "p1_dense": [round(x, 6) for x in d["p1_dense"]],
        "p1_ec_dense": [round(x, 6) for x in d["p1_ec_dense"]],
        "p1_emc_dense": [round(x, 6) for x in d["p1_emc_dense"]],
        "p1_sub": [round(x, 6) for x in d["p1_sub"]],
    }


def aggregate(recs: list[dict]) -> dict:
    """All per-tube statistics from the per-job records."""
    out = {}
    for tube in TUBES:
        trecs = [r for r in recs if r["tube"] == tube]
        e0_res = [r for r in trecs if r["field"] == 0.0]
        assert len(e0_res) == N_RUNS, tube
        keys = sorted(set().union(*[set(r["blocks"]) for r in trecs]))
        assert all(set(r["blocks"]) == set(keys) for r in trecs), \
            f"cnt{tube}: statement sets differ across jobs"

        floors = {}
        for k in keys:
            vals, runs = [], []
            for r in e0_res:
                vals += [r["blocks"][k]["A"], r["blocks"][k]["B"]]
                runs += [r["name"], r["name"]]
            floors[k] = d15_floor(np.array(vals), runs)

        fields = sorted({r["field"] for r in trecs if r["field"] != 0.0})

        def direct_res(recs2: list[dict], which: str) -> list[dict]:
            """Records viewed through the direct-P1 arm (which in
            {"dense", "sub"}), as single-statement records."""
            out2 = []
            for r in recs2:
                b = r["direct_p1_blocks" if which == "dense"
                       else "direct_p1_sub_blocks"]
                out2.append({**r, "blocks": {"direct_p1": dict(b)}})
            return out2

        detection, detection_direct, paired, moves, classifier, ab_check \
            = {}, {}, {}, {}, {}, {}
        for f in fields:
            rs = [r for r in trecs if r["field"] == f]
            det, detD = {}, {}
            for k in keys:
                on = run_level([r["blocks"][k]["B"] for r in rs])
                off = e0_level(e0_res, k)
                det[k] = detection_entry(on, off, floors[k]["floor"])
            # direct P1 (dense = every frame of the record; sub = the
            # 12-ps cadence the lift arm uses)
            onD = run_level([r["direct_p1_blocks"]["B"] for r in rs])
            offD = e0_level(direct_res(e0_res, "dense"), "direct_p1")
            floorD = d15_floor(np.array(offD["blocks"]),
                               [r["name"] for r in e0_res
                                for _ in (0, 1)])["floor"]
            detD["direct_p1"] = detection_entry(onD, offD, floorD)
            onS = run_level([r["direct_p1_sub_blocks"]["B"] for r in rs])
            offS = e0_level(direct_res(e0_res, "sub"), "direct_p1")
            floorS = d15_floor(np.array(offS["blocks"]),
                               [r["name"] for r in e0_res
                                for _ in (0, 1)])["floor"]
            detD["direct_p1_sub"] = detection_entry(onS, offS, floorS)
            detection[f"{f:g}"] = det
            detection_direct[f"{f:g}"] = detD

            paired[f"{f:g}"] = {k: paired_view(rs, e0_res, k) for k in keys}
            paired[f"{f:g}"]["direct_p1"] = paired_view(
                direct_res(rs, "dense"), direct_res(e0_res, "dense"),
                "direct_p1")

            # D32 step-7 move list: every non-dipole_p1 statement beyond
            # 2x its floor (delta > 0), per field; <= working field is
            # the reportable list, other fields recorded for context.
            m2 = sorted(k for k, v in det.items()
                        if k != "dipole_p1" and v["ratio"] > 2
                        and v["delta"] > 0)
            m3 = sorted(k for k, v in det.items()
                        if k != "dipole_p1" and v["pass_3x"])
            structural = sorted(k for k, v in det.items()
                                if v["floor_zero"] and v["delta"] == 0)
            moves[f"{f:g}"] = {
                "beyond_2x_floor": m2, "beyond_3x_floor": m3,
                "constant_by_construction": structural,
                "at_or_below_working_field": bool(f <= WORKING_FIELD)}

            # classifier: dipole_p1 block values, LORO over runs, chaord
            # statement + direct P1 (dense / 12-ps subsample)
            def e0_block_vals(getter, all_blocks=True):
                vals, runs = [], []
                for r in e0_res:
                    blks = ([r["blocks"]["dipole_p1"]["A"],
                             r["blocks"]["dipole_p1"]["B"]]
                            if all_blocks else
                            [r["blocks"]["dipole_p1"]["B"]])
                    for b in blks:
                        vals.append(getter(b, r))
                        runs.append(r["name"])
                return np.array(vals), runs

            on_b = np.array([r["blocks"]["dipole_p1"]["B"] for r in rs])
            off_ab, off_runs = e0_block_vals(lambda b, r: b)
            off_b, _ = e0_block_vals(lambda b, r: b, all_blocks=False)
            on_ab = np.array([r["blocks"]["dipole_p1"][w] for r in rs
                              for w in ("A", "B")])
            on_dense = np.array([r["direct_p1_blocks"]["B"] for r in rs])
            off_dense, off_dense_runs = [], []
            for r in e0_res:
                off_dense += [r["direct_p1_blocks"]["A"],
                              r["direct_p1_blocks"]["B"]]
                off_dense_runs += [r["name"], r["name"]]
            on_sub = np.array([r["direct_p1_sub_blocks"]["B"] for r in rs])
            off_sub, off_sub_runs = [], []
            for r in e0_res:
                off_sub += [r["direct_p1_sub_blocks"]["A"],
                            r["direct_p1_sub_blocks"]["B"]]
                off_sub_runs += [r["name"], r["name"]]
            classifier[f"{f:g}"] = {
                "feature": "dipole_p1 (the blind statement's value; the "
                           "direct arm uses the same quantity without "
                           "chaord)",
                "primary_blockB_on_vs_e0_allblocks": loro_classifier(
                    on_b, [r["name"] for r in rs], off_ab, off_runs),
                "sensitivity_allblocks": loro_classifier(
                    on_ab, [r["name"] for r in rs for _ in (0, 1)],
                    off_ab, off_runs),
                "direct_p1_dense": loro_classifier(
                    on_dense, [r["name"] for r in rs],
                    np.array(off_dense), off_dense_runs),
                "direct_p1_dense_allblocks_on": loro_classifier(
                    np.array([r["direct_p1_blocks"][w] for r in rs
                              for w in ("A", "B")]),
                    [r["name"] for r in rs for _ in (0, 1)],
                    np.array(off_dense), off_dense_runs),
                "direct_p1_subsampled": loro_classifier(
                    on_sub, [r["name"] for r in rs],
                    np.array(off_sub), off_sub_runs),
            }

            # block A vs B transient check (stage_b_plan.md section 3.3)
            rows = {}
            for k in keys:
                mA = float(np.mean([r["blocks"][k]["A"] for r in rs]))
                mB = float(np.mean([r["blocks"][k]["B"] for r in rs]))
                run_sem = run_level(
                    [(r["blocks"][k]["A"] * r["blocks"][k]["nA"]
                      + r["blocks"][k]["B"] * r["blocks"][k]["nB"])
                     / (r["blocks"][k]["nA"] + r["blocks"][k]["nB"])
                     for r in rs])["sem"]
                rows[k] = {"mean_A": mA, "mean_B": mB,
                           "offset_exceeds_2x_run_sem":
                               bool(run_sem > 0
                                    and abs(mA - mB) > 2 * run_sem)}
            ab_check[f"{f:g}"] = rows

        # linear response, both arms (block B primary on the field-on
        # side; the zero-field point is the e0 runs, both blocks clean)
        linresp = {}

        def zero_point(getter_all):
            v = np.array([getter_all(r) for r in e0_res], float)
            return (float(v.mean()),
                    float(v.std(ddof=1) / math.sqrt(len(v))))

        for arm, getter, zp in (
                ("chaord_dipole_p1_statement",
                 lambda r: r["blocks"]["dipole_p1"]["B"],
                 zero_point(lambda r: (r["blocks"]["dipole_p1"]["A"]
                                       * r["blocks"]["dipole_p1"]["nA"]
                                       + r["blocks"]["dipole_p1"]["B"]
                                       * r["blocks"]["dipole_p1"]["nB"])
                           / (r["blocks"]["dipole_p1"]["nA"]
                              + r["blocks"]["dipole_p1"]["nB"]))),
                ("direct_p1_dense",
                 lambda r: r["direct_p1_blocks"]["B"],
                 zero_point(lambda r: (r["direct_p1_blocks"]["A"]
                                       * r["direct_p1_blocks"]["nA"]
                                       + r["direct_p1_blocks"]["B"]
                                       * r["direct_p1_blocks"]["nB"])
                           / (r["direct_p1_blocks"]["nA"]
                              + r["direct_p1_blocks"]["nB"])))):
            zero_mean, zero_sem = zp
            pts = {0.0: {"mean": zero_mean, "sem": zero_sem}}
            for f in fields:
                vals = [getter(r) for r in trecs
                        if r["field"] == f]
                v = np.array(vals, float)
                pts[f] = {"mean": float(v.mean()),
                          "sem": float(v.std(ddof=1) / math.sqrt(len(v))),
                          "per_run": [float(x) for x in v]}
            slopes = {}
            for f in fields:
                num = pts[f]["mean"] - pts[0.0]["mean"]
                den = (math.hypot(pts[f]["sem"], pts[0.0]["sem"]) / f
                       if f > 0 else None)
                slopes[f"{f:g}"] = {"slope": num / f, "sem": den}
            lin_pts = [f for f in fields if 0 < f <= 0.02]
            pairs = [(a1, b1) for i, a1 in enumerate(lin_pts)
                     for b1 in lin_pts[i + 1:]]
            agree = []
            for a1, b1 in pairs:
                sa, sb = slopes[f"{a1:g}"], slopes[f"{b1:g}"]
                agree.append({
                    "fields": [a1, b1],
                    "agree_within_2sigma":
                        bool(abs(sa["slope"] - sb["slope"])
                             <= 2 * math.hypot(sa["sem"], sb["sem"]))})
            linresp[arm] = {
                "P1_vs_field": {f"{f:g}": pts[f] for f in sorted(pts)},
                "slopes_per_V_per_A": slopes,
                "linear_range_checks_0.005_0.02": agree}

        def e_min(getter_pass):
            hits = [f for f in fields if getter_pass(f)]
            return hits[0] if hits else None

        e_min_tab = {
            "criterion_A_3x_floor": {
                "chaord_dipole_p1": e_min(
                    lambda f: detection[f"{f:g}"]["dipole_p1"]["pass_3x"]),
                "direct_p1_dense": e_min(
                    lambda f: detection_direct[f"{f:g}"]["direct_p1"]
                    ["pass_3x"]),
                "direct_p1_subsampled": e_min(
                    lambda f: detection_direct[f"{f:g}"]["direct_p1_sub"]
                    ["pass_3x"])},
            "criterion_B_paired_3sigma": {
                "chaord_dipole_p1": e_min(
                    lambda f: paired[f"{f:g}"]["dipole_p1"]["pass_3s"]),
                "direct_p1_dense": e_min(
                    lambda f: paired[f"{f:g}"]["direct_p1"]["pass_3s"])},
            "criterion_B_paired_2sigma": {
                "chaord_dipole_p1": e_min(
                    lambda f: paired[f"{f:g}"]["dipole_p1"]["pass_2s"]),
                "direct_p1_dense": e_min(
                    lambda f: paired[f"{f:g}"]["direct_p1"]["pass_2s"])},
            "loro_perfect_separation": e_min(
                lambda f: classifier[f"{f:g}"]
                ["primary_blockB_on_vs_e0_allblocks"]["accuracy"] == 1.0),
        }

        out[str(tube)] = {
            "floors": floors, "detection": detection,
            "detection_direct": detection_direct, "paired": paired,
            "statement_moves": moves, "classifier": classifier,
            "linear_response": linresp, "block_AB_check": ab_check,
            "e_min": e_min_tab,
            "p1_curve": {
                "chaord_dipole_p1": linresp["chaord_dipole_p1_statement"]
                ["P1_vs_field"]},
        }
    return out


def run(smoke: bool = False, fresh: bool = False) -> dict:
    t0 = time.perf_counter()
    jobs = discover_jobs()
    dl = composition_dialect()
    print(f"[y8] {len(jobs)} ladder jobs "
          f"({sum(j['source_kind'] == 'local_full' for j in jobs)} local "
          f"full-density, "
          f"{sum(j['source_kind'] == 'cluster_thinned' for j in jobs)} "
          f"cluster thinned)",
          flush=True)

    done = {}
    if not smoke and CHECKPOINT.is_file() and not fresh:
        for line in CHECKPOINT.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["name"]] = r
        print(f"[y8] checkpoint: {len(done)} job(s) already done",
              flush=True)
    ckpt = open(CHECKPOINT, "a", encoding="utf-8") if not smoke else None

    recs = []
    for i, job in enumerate(jobs):
        if job["name"] in done:
            recs.append(done[job["name"]])
            continue
        tj = time.perf_counter()
        n_full = LOCAL_FRAMES if job["source_kind"] == "local_full" \
            else THINNED_FRAMES
        if smoke:
            sub = [0, n_full // 2, n_full // 2 + 1, n_full - 1]
        elif job["source_kind"] == "local_full":
            sub = list(range(0, n_full, int(CADENCE_PS)))
        else:
            sub = list(range(n_full))
        prod_ps = np.array(sub, float) * (LOCAL_CADENCE_PS
                                          if job["source_kind"] == "local_full"
                                          else CADENCE_PS)
        print(f"[y8] ({i + 1}/{len(jobs)}) {job['name']} "
              f"(field {job['field']:g} V/A, {job['source_kind']}): "
              f"arm 2 dense stream...", flush=True)
        d = direct_arm(job, sub)
        print(f"[y8]   {d['n_frames']} frames read, census+charge verified "
              f"every frame", flush=True)
        print(f"[y8]   arm 1: Y4 read + blind lift of {len(sub)} frames...",
              flush=True)
        c = chaord_arm(job, sub, prod_ps, dl)
        rec = job_record(job, d, c)
        # the determinism cross-check: the same job re-run in the other
        # chain (same seeds and step-0 thermo, different LAMMPS builds)
        if job["duplicate"] is not None:
            dd = direct_arm_duplicate(job)
            ps = dd["prod_ps"]
            # both series on the shared 12-ps grid (both grids are
            # increasing, so the masked selections stay aligned)
            canon = d["p1_dense"][np.isin(d["prod_ps"], ps)]
            dupv = dd["p1_dense"][np.isin(ps, d["prod_ps"])]
            assert len(canon) == len(dupv) > 0, \
                (job["name"], len(canon), len(dupv))
            rec["duplicate_check"] = {
                "dump": str(job["duplicate"]),
                "kind": job["duplicate_kind"],
                "n_frames": dd["n_frames"],
                "n_shared_frames": int(len(canon)),
                "p1_blocks": {
                    "A": float(dd["p1_dense"][ps < SPLIT_PS].mean()),
                    "B": float(dd["p1_dense"][ps >= SPLIT_PS].mean())},
                "max_abs_p1_diff_vs_canonical_on_shared_grid":
                    float(np.abs(dupv - canon).max()),
            }
            print(f"[y8]   duplicate-check ({job['duplicate_kind']}): "
                  f"B {rec['duplicate_check']['p1_blocks']['B']:+.4f} "
                  f"vs canonical {rec['direct_p1_blocks']['B']:+.4f}",
                  flush=True)
        assert not rec["failed_checks"], (job["name"], rec["failed_checks"])
        print(f"[y8]   lift ok; dipole_p1 printed-vs-direct max |diff| "
              f"{rec['p1_text_vs_direct_max_abs']:.2e}; job wall "
              f"{time.perf_counter() - tj:.0f} s", flush=True)
        recs.append(rec)
        if ckpt is not None:
            ckpt.write(json.dumps(rec) + "\n")
            ckpt.flush()

    if ckpt is not None:
        ckpt.close()

    results = {"meta": {}, "jobs": {r["name"]: r for r in recs},
               "per_tube": aggregate(recs)}
    n_lift_total = sum(r["n_lifted"] for r in recs)
    n_frames_total = sum(r["n_frames_dense"] for r in recs)
    fields_by_tube = {
        str(t): sorted({r["field"] for r in recs if r["tube"] == t})
        for t in TUBES}
    results["meta"] = {
        "work_order": "docs/reviews/open_items_v4.md Y8 (full-ladder pass)",
        "decisions": "D31 (ensembles/floors/LORO), D32 (two arms, verdict), "
                     "D15 (floor estimator), D29 (blind dipole_p1), "
                     "D30 (working field 0.01 V/A)",
        "ladder": "2 tubes x 7 fields (0, 0.005, 0.01, 0.02, 0.05, 0.1, "
                  "0.2 V/A) x 3 independent builds; 42 jobs",
        "record_sources": {
            "freeze": "canonical assignment FROZEN to the Y8 work-order "
                      "state: local full-density for cnt10 fields {0, "
                      "0.01, 0.05, 0.1}, cluster thinned for the 33 "
                      "stageB/sb_* dirs",
            "local_full": "12 jobs: stageB/manifests/sb_cnt10_{e0,e10,e50,"
                          "e100}_r{1,2,3}.json, DONE non-smoke, prod.dump "
                          "12,001 frames at 1 ps (LAMMPS 22 Jul 2025)",
            "cluster_thinned": "33 jobs: stageB/sb_*/prod_thinned.dump, "
                               "EVERY 12th production frame (12 ps, 1,001 "
                               "frames), finished log.lammps (LAMMPS "
                               "3 Mar 2020)",
            "duplicate": "sb_cnt10_e100 exists in both chains (same seeds, "
                         "same step-0 thermo); the LOCAL full-density copy "
                         "is canonical, the cluster copy is the "
                         "determinism cross-check (different LAMMPS "
                         "builds)",
            "anomaly_concurrent_queue": "a local stageB queue (pid in "
                                        "stageB/QUEUE_RUNNING) completed "
                                        "sb_cnt10_e5_r{1,2,3} locally "
                                        "while this analysis ran (new "
                                        "DONE markers + manifests); per "
                                        "the freeze their canonical "
                                        "source stays the cluster thinned "
                                        "copy and the local full-density "
                                        "copies are cross-checks; the "
                                        "queue's cnt10 e20/e200 local "
                                        "runs were still in flight and "
                                        "are part of no record here"},
        "lift_subsampling": {
            "rule": "EVERY frame of the committed record at the 12-ps "
                    "thinned cadence -- NO analysis subsampling beyond the "
                    "record's own thinning; local 1-ps records are "
                    "downsampled to the same 12-ps cadence (uniform "
                    "protocol), their full density used by arm 2 (dense) "
                    "with the dense-vs-subsampled agreement recorded",
            "lifts_per_job": THINNED_FRAMES,
            "lifts_total": n_lift_total,
            "block_frames_lift_arm": [500, 501],
            "block_frames_arm2_dense_local": [6000, 6001],
            "measured_lift_cost_ms": {"cnt10_971_atoms": 103,
                                      "cnt16_2058_atoms": 174},
            "note": "the 20-30 s/lift worst case in the task brief was "
                    "measured at 0.10-0.17 s/lift on this box, so full "
                    "coverage is affordable and used"},
        "block_protocol": {
            "split_ps": SPLIT_PS, "block_ps": 6000.0,
            "verified": "thinned timesteps 0..12,000,000 (12 ps); local "
                        "prod.dump timesteps 0..12,000,000 (1 ps)",
            "primary": "block B (stage_b_plan.md section 3.2: block A "
                       "retains ~27-31% of the switch-on transient)",
            "e0_blocks": "2 x 6 ns per run, both clean (field stays 0)"},
        "floors": "D15 max(mean, P90) over |Delta| between e0 block pairs "
                  "from different runs of THIS batch (12 cross-run pairs "
                  "per tube; 6 ns blocks -- like-for-like, supersedes the "
                  "Stage A 1-ns floors)",
        "detection_criteria": {
            "A_ordered_3x_floor": "|mean_on(B) - mean_off| > 3 x D15 "
                                  "cross-run e0 floor (unpaired; runs the "
                                  "units)",
            "B_build_paired": "|mean(on(B) - same-build e0 all-block mean)|"
                              " > 3 x sem over the 3 builds (2x also "
                              "recorded); secondary, owner's call"},
        "smoke": bool(smoke),
        "n_frames_read_total": n_frames_total,
        "wall_seconds": round(time.perf_counter() - t0, 1),
    }

    if not smoke:
        OUT_JSON.write_text(json.dumps(results, indent=1))
        print(f"[y8] wrote {OUT_JSON}")
    print(f"[y8] wall {time.perf_counter() - t0:.1f} s")
    return results


def print_summary(res: dict) -> None:
    for tube in TUBES:
        pt = res["per_tube"][str(tube)]
        fl = pt["floors"]["dipole_p1"]
        print(f"\n== cnt{tube}: D15 floors from the stage-B e0 runs "
              f"(12 cross-run pairs, 6-ns blocks) ==")
        for k in ("dipole_p1", "fluid.li_shell.C3H4O3",
                  "fluid.li_shell.C4H8O3", "fluid.li_shell.F6P",
                  "fluid.ion_pairs.CIP", "fluid.nr.Li+.peak"):
            if k in pt["floors"]:
                f = pt["floors"][k]
                print(f"  {k:32s} floor {f['floor']:.4f} (mean "
                      f"{f['mean']:.4f}, P90 {f['p90']:.4f})")
        print(f"\n== cnt{tube}: dipole_p1 P1-vs-E curve (block B, mean +- "
              f"sem over 3 runs) ==")
        for f, p in pt["linear_response"]["chaord_dipole_p1_statement"][
                "P1_vs_field"].items():
            print(f"  E = {f:>5s} V/A: P1 {p['mean']:+.4f} +- {p['sem']:.4f}")
        print(f"\n== cnt{tube}: detection per criterion ==")
        print(f"  e_min table: {json.dumps(pt['e_min'])}")
        for f, det in pt["detection"].items():
            p1 = det["dipole_p1"]
            print(f"  field {f} V/A: delta {p1['delta']:.4f} = "
                  f"x{p1['ratio']:.2f} floor ({p1['floor']:.4f}) -> "
                  f"{'PASS' if p1['pass_3x'] else 'fail'} at 3x "
                  f"(z {p1['z']:.1f}); moves>2x: "
                  f"{pt['statement_moves'][f]['beyond_2x_floor'] or 'none'}")
            pr = pt["paired"][f]["dipole_p1"]
            print(f"    paired {pr['mean']:+.4f} +- {pr['sem']:.4f} "
                  f"(z {pr['z']:.1f}; 3s {'PASS' if pr['pass_3s'] else 'fail'}"
                  f", 2s {'PASS' if pr['pass_2s'] else 'fail'})")
        print(f"\n== cnt{tube}: LORO classifier (primary variant) ==")
        for f, c in pt["classifier"].items():
            pr = c["primary_blockB_on_vs_e0_allblocks"]
            print(f"  field {f} V/A: {pr['n_correct']}/{pr['n_blocks']} "
                  f"blocks (on {pr['on_blocks_correct']}/3, off "
                  f"{pr['off_blocks_correct']}/6)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true",
                    help="4 frames per job; pipeline test, never for the "
                         "record (no JSON written)")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore any existing checkpoint")
    args = ap.parse_args()
    res = run(smoke=args.smoke, fresh=args.fresh)
    if not args.smoke:
        print_summary(res)


if __name__ == "__main__":
    main()
