# -*- coding: utf-8 -*-
"""Z4 confirmatory analysis -- the Z3 confirmatory set, run EXACTLY as
pre-registered (docs/reviews/open_items_v5.md work order Z4; D39/D41
binding; research/cnt_electrolyte/confirmatory/PLAN.md, written 2026-10-31
BEFORE any confirmatory job was built, run or submitted).

THE CALL (PLAN.md section 2, verbatim contract): lift every frame of the
18 thinned production series blind (lift_frame(mode="auto"), no field=
kwarg -- the Y8 machinery, imported unchanged) into per-job
direct_p1_blocks {A, B} (block A = production t in [0, 6000) ps, block
B = [6000, 12000] ps, verified against the dump timesteps), then the D39
test: zero-field run means (blocks A and B averaged) vs field-on block-B
values, Welch's t two-sided alpha = 0.05, n = 6 per group.

    0.01 V/A is CONFIRMED iff its Welch p < 0.05 on the confirmatory set
    alone.  No other test is added after seeing the data.

Computed by the same pre-registered call (PLAN section 2): the 0.02 V/A
consistency point, reported as secondary (Stage B detected it at
p = 0.0109 with n = 3; D40's E_min for (16,16) is 0.02 V/A), and the
build-paired contrast (per build: field-on block B minus that build's e0
all-block mean) -- secondary, never the gate.

TWO ARMS, exactly as Y8 ran them (D32's second arm):

  Arm 1 (Chaord): Y4 reader (chaord.io.lammps) with the type map of the
  BUILD data file (confirmatory/jobs/<name>/system.data -- never
  final.data), ONE blind lift per frame via
  analyze_first_result.lift_one_frame(frame, None, dl) -- mode="auto",
  T = 300.0, one composition dialect for all 18,018 frames -- statement
  features parsed from the formatted program text with
  y8_block_analysis.y8_statement_features (the same parser tranche 1 and
  the full ladder used).

  Arm 2 (direct P1 without chaord): numpy/stdlib only --
  y8_block_analysis.stream_frames reads EVERY frame of the thinned dump;
  P1 = mean solvent mu_z/|mu| over EC (10 atoms) and EMC (15 atoms)
  molecules by mol id, minimum-image offsets from each molecule's first
  atom, the dump's own charges; per-frame atom-count, charge and census
  verification.

FIDELITY (work order step 2): the blind statement's printed dipole_p1 vs
the direct numpy P1 on the same frames must agree to the print quantum.
The statement prints P1 with "%.3f" (src/chaord/lift/__init__.py
_attach_dipole_p1), so the tolerance is the half-quantum 5e-4 (Y8's
full-ladder max over 42,042 frames was 4.9999e-4 -- exactly this bound).

PROTOCOL VERIFICATION (PLAN sections 3/8, honest record, no test added):
- blocks verified against the dump timesteps: thinned timesteps
  0 .. 12,000,000 at exactly 12,000-step (12 ps) spacing; block A = the
  500 frames with t < 6,000,000, block B = the 501 frames with
  t >= 6,000,000.
- cross-build chaining check (PLAN section 3): on the ONE cluster LAMMPS
  build, each build's three field jobs share the thermalization seeds,
  so their end-of-stage-2 thermo row (step 1000000) must be IDENTICAL;
  a mismatch is reported, never silently absorbed.
- every lifted program's non-moving statements must be identical across
  a job's 1,001 lifts; failed static checks are recorded per frame --
  never substituted, never dropped.

Every number is deterministic.  Output:
research/cnt_electrolyte/analysis/confirmatory_results.json plus a
console summary.  Per-job progress is checkpointed to
confirmatory_z4_checkpoint.jsonl (one JSON line per completed job) so an
interrupted run resumes without recomputing finished jobs.

Usage (from the repository root):
    python research/cnt_electrolyte/analysis/confirmatory_z4.py
    python research/cnt_electrolyte/analysis/confirmatory_z4.py --smoke 2 4
        # first 2 jobs, 4 frames each; pipeline test only, never for the
        # record (no JSON written)
    python research/cnt_electrolyte/analysis/confirmatory_z4.py --fresh
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
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parents[1]          # research/cnt_electrolyte
for _p in (str(ROOT / "src"), str(HERE), str(HERE / "analysis")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# the Y8 machinery, unchanged (arms, parser, streamer, census)
from y8_block_analysis import (                        # noqa: E402
    build_file_census, classify_frame_census, stream_frames,
    y8_statement_features)
from analyze_first_result import (                     # noqa: E402
    composition_dialect, lift_one_frame)
from chaord.io.lammps import read_lammps_data, read_lammps_dump  # noqa: E402

OUT_JSON = Path(__file__).resolve().parent / "confirmatory_results.json"
CHECKPOINT = (Path(__file__).resolve().parent /
              "confirmatory_z4_checkpoint.jsonl")

CONF = HERE / "confirmatory"

# ---- confirmatory constants (PLAN.md sections 1-3, verified at run time) --
BUILDS = tuple(range(1, 7))                  # cf16_b1 .. cf16_b6
FIELD_OF_MV = {0: 0.0, 10: 0.01, 20: 0.02}   # V/A; e<mv> = 1000x
N_JOBS = 18
CADENCE_PS = 12.0                            # the thinned record
SPLIT_PS = 6000.0                            # block A | B
THINNED_FRAMES = 1001                        # per job, both blocks
N_ATOMS = 4116                               # 2560 tube C + 1556 liquid
EXPECTED_CENSUS = {"EC": 74, "EMC": 48, "PF6": 12, "Li": 12}   # PLAN sec. 6
ATOMS_PER = {"EC": 10, "EMC": 15, "PF6": 7, "Li": 1}           # stageA conv.
CHARGE_TOL = 1e-6                            # cell neutrality (e)
ALPHA = 0.05                                 # D39, two-sided
PRINT_QUANTUM = 1e-3                         # the statement prints "%.3f"
FIDELITY_TOL = PRINT_QUANTUM / 2 + 1e-9      # half-quantum + float slack


# ---------------------------------------------------------------------------
# provenance: the 18 fixed jobs of PLAN.md section 1
# ---------------------------------------------------------------------------

def discover_jobs() -> list[dict]:
    jobs = []
    for mv, field in FIELD_OF_MV.items():
        for b in BUILDS:
            name = f"sb16L_e{mv}_b{b}"
            d = CONF / name
            dump = d / "prod_thinned.dump"
            log = d / "log.lammps"
            bd = CONF / "jobs" / name / "system.data"
            for p in (dump, log, bd, d / "final.data", d / "equil.dump"):
                assert p.is_file() and p.stat().st_size > 0, (name, p)
            assert "Total wall time" in log.read_text(errors="replace"), name
            jobs.append({"name": name, "field": field, "mv": mv,
                         "build": f"cf16_b{b}",
                         "dump": dump, "build_data": bd, "log": log})
    assert len(jobs) == N_JOBS
    return jobs


# ---------------------------------------------------------------------------
# arm 2 -- direct P1 without chaord (imported streamer; confirmatory asserts)
# ---------------------------------------------------------------------------

def direct_arm(job: dict) -> dict:
    """P1 of the solvent on EVERY frame of the thinned dump (dense -- the
    record IS the 12-ps cadence; no analysis subsampling), with the
    per-frame census/charge verification and the timestep verification."""
    want_atoms, want_census, type_mass = build_file_census(job["build_data"])
    assert want_atoms == N_ATOMS, (job["name"], want_atoms)
    assert want_census == EXPECTED_CENSUS, (job["name"], want_census)
    p1, p1_ec, p1_emc, prod_ps, timesteps = [], [], [], [], []
    for k, t, box, a in stream_frames(job["dump"]):
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
        timesteps.append(t)
    assert len(p1) == THINNED_FRAMES, (job["name"], len(p1))
    ts = np.array(timesteps)
    assert ts[0] == 0 and ts[-1] == 12_000_000, (job["name"], ts[0], ts[-1])
    dts = np.diff(ts)
    assert np.all(dts == 12_000), (job["name"], np.unique(dts))
    prod_ps = np.array(prod_ps)
    assert np.allclose(np.diff(prod_ps), CADENCE_PS), job["name"]
    selA, selB = prod_ps < SPLIT_PS, prod_ps >= SPLIT_PS
    assert int(selA.sum()) == 500 and int(selB.sum()) == 501, \
        (job["name"], int(selA.sum()), int(selB.sum()))
    return {
        "n_frames": len(p1), "t_first": int(ts[0]), "t_last": int(ts[-1]),
        "prod_ps": prod_ps, "census": want_census, "n_atoms": want_atoms,
        "p1_dense": np.array(p1), "p1_ec_dense": np.array(p1_ec),
        "p1_emc_dense": np.array(p1_emc),
        "direct_p1_blocks": {
            "A": float(np.array(p1)[selA].mean()),
            "B": float(np.array(p1)[selB].mean()),
            "nA": int(selA.sum()), "nB": int(selB.sum())},
    }


# ---------------------------------------------------------------------------
# arm 1 -- the Chaord lift (Y4 reader -> blind lift -> statement features)
# ---------------------------------------------------------------------------

def chaord_arm(job: dict, prod_ps: np.ndarray, dl) -> dict:
    """Arm 1 on one job: Y4-read all 1,001 frames, lift each with identical
    blind options (mode="auto", NO field=, T=300), parse the statement
    features."""
    _, data_meta = read_lammps_data(str(job["build_data"]))
    tmap = data_meta["type_map"]
    frames = read_lammps_dump(str(job["dump"]),
                              frame_indices=list(range(THINNED_FRAMES)),
                              type_map=tmap)
    unknown = tmap.apply(frames, strict=True)
    assert not unknown, f"{job['name']}: unmapped type ids {unknown}"
    assert len(frames) == THINNED_FRAMES, (job["name"], len(frames))
    feats_list, consts_list, failed = [], [], []
    p1_text = []
    t_lift = 0.0
    for fr, ps in zip(frames, prod_ps):
        assert abs(fr.info["timestep"] / 1000.0 - ps) < 1e-9, \
            (job["name"], fr.info["timestep"], ps)
        assert abs(np.asarray(fr.info["charges"]).sum()) < CHARGE_TOL
        t0 = time.perf_counter()
        text, info = lift_one_frame(fr, None, dl=dl)   # T=300, blind, auto
        t_lift += time.perf_counter() - t0
        feats, consts = y8_statement_features(text)
        feats_list.append(feats)
        consts_list.append(json.dumps(consts, sort_keys=True))
        if info["failed_checks"]:
            failed.append({"prod_ps": float(ps),
                           "checks": info["failed_checks"]})
        p1_text.append(feats["dipole_p1"])
    if failed:
        print(f"[z4] WARNING {job['name']}: {len(failed)} lifted frame(s) "
              f"with failed static checks -- recorded in the JSON, never "
              f"substituted", file=sys.stderr)
    assert len(set(consts_list)) == 1, \
        f"{job['name']}: the program's non-moving statements move between " \
        f"frames"
    consts = json.loads(consts_list[0])
    assert consts.get("p1_axis") == "z", (job["name"], consts.get("p1_axis"))
    assert consts.get("chirality") == [16, 16], (job["name"], consts)
    keys = sorted(set().union(*[set(f) for f in feats_list]))
    counts = {k2: sum(k2 in f for f in feats_list) for k2 in keys}
    assert all(c == len(frames) for c in counts.values()), \
        f"{job['name']}: statement presence varies across frames: {counts}"
    return {
        "n_lifted": len(frames),
        "prod_ps": [float(x) for x in prod_ps],
        "consts": consts,
        "features": {k2: [f[k2] for f in feats_list] for k2 in keys},
        "failed_checks": failed,
        "p1_text": p1_text,
        "lift_seconds": t_lift,
    }


# ---------------------------------------------------------------------------
# protocol verification: the cross-build chaining check (PLAN section 3)
# ---------------------------------------------------------------------------

# a thermo row at step 1000000 (equil end) / 100000 (ramp end):
# step atoms Tmob temp pe etotal press vol
_THERMO_1M = re.compile(
    r"^\s*1000000\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$")
_THERMO_100K = re.compile(
    r"^\s*100000\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$")
_THERMO_COLS = ("atoms", "Tmob", "temp", "pe", "etotal", "press", "vol")


def first_thermo_row(log_path: Path, pattern) -> dict:
    """The FIRST thermo row at the given step of a finished log.  At step
    1000000 that is the end of the shared stage-2 thermalization (stage
    3's initial thermo prints the same step again and is not the one PLAN
    names); at step 100000 it is the end of the shared stage-1 ramp."""
    with open(log_path, errors="replace") as f:
        for line in f:
            m = pattern.match(line)
            if m:
                return dict(zip(_THERMO_COLS, m.groups()))
    raise AssertionError(f"{log_path.name}: no matching thermo row")


def chaining_check(jobs: list[dict]) -> dict:
    """Per build: the three field jobs' step-1000000 thermo rows must be
    IDENTICAL (one LAMMPS build, shared thermalization seeds).  A mismatch
    is reported, never absorbed (PLAN section 3).  The step-100000 row
    (end of the shared ramp) localizes where a divergence began."""
    by_build: dict[str, dict[str, dict]] = {}
    ramp: dict[str, dict[str, dict]] = {}
    for j in jobs:
        by_build.setdefault(j["build"], {})[j["name"]] = \
            first_thermo_row(j["log"], _THERMO_1M)
        ramp.setdefault(j["build"], {})[j["name"]] = \
            first_thermo_row(j["log"], _THERMO_100K)
    out = {}
    for build in sorted(by_build):
        per_job = by_build[build]
        vals = {name: tuple(row.values()) for name, row in per_job.items()}
        identical = len(set(vals.values())) == 1
        # which jobs share a bit-identical thermalization path
        groups: dict[tuple, list[str]] = {}
        for name, v in vals.items():
            groups.setdefault(v, []).append(name)
        ramp_job = ramp[build]
        ramp_vals = {name: tuple(row.values())
                     for name, row in ramp_job.items()}
        ramp_groups: dict[tuple, list[str]] = {}
        for name, v in ramp_vals.items():
            ramp_groups.setdefault(v, []).append(name)
        # does the divergent member already differ at the ramp end?
        diverges_in_ramp = len(set(
            tuple(ramp_job[n].values()) for n in per_job)) > 1
        out[build] = {
            "jobs": per_job,
            "ramp_end_step_100000": ramp_job,
            "identical_at_step_1000000": identical,
            "chain_groups_step_1000000": [sorted(v) for v in groups.values()],
            "chain_groups_step_100000": [sorted(v)
                                         for v in ramp_groups.values()],
            "divergence_begins_in_stage_1_ramp": bool(diverges_in_ramp),
        }
        print(f"[z4] chaining {build}: "
              f"{'IDENTICAL' if identical else 'MISMATCH'} at step 1000000 "
              f"(groups {sorted(sorted(v) for v in groups.values())}; "
              f"ramp-end groups "
              f"{sorted(sorted(v) for v in ramp_groups.values())})",
              flush=True)
    return out


# ---------------------------------------------------------------------------
# the pre-registered D39 test (PLAN section 2, verbatim call)
# ---------------------------------------------------------------------------

def d39_welch(J: dict) -> dict:
    """Exactly the PLAN section 2 script: zero-field run means (blocks A
    and B averaged, unweighted as written) vs field-on block-B values,
    stats.ttest_ind(on, off, equal_var=False), n = 6 per group."""
    off = np.array([np.mean([j["direct_p1_blocks"]["A"],
                             j["direct_p1_blocks"]["B"]])
                    for j in J.values() if j["field"] == 0.0])   # n = 6
    assert len(off) == 6, len(off)
    out = {"alpha": ALPHA, "n_per_group": 6,
           "off_mean": float(off.mean()),
           "off_sem": float(off.std(ddof=1) / math.sqrt(len(off))),
           "off_per_run": [float(x) for x in off],
           "fields": {}}
    for E in (0.01, 0.02):
        on = np.array([j["direct_p1_blocks"]["B"] for j in J.values()
                       if j["field"] == E])                      # n = 6
        assert len(on) == 6, (E, len(on))
        r = stats.ttest_ind(on, off, equal_var=False)
        # Welch-Satterthwaite df, for the record
        s2o, s2f = on.var(ddof=1) / len(on), off.var(ddof=1) / len(off)
        df = (s2o + s2f) ** 2 / (s2o ** 2 / (len(on) - 1)
                                 + s2f ** 2 / (len(off) - 1))
        out["fields"][f"{E:g}"] = {
            "on_mean_B": float(on.mean()),
            "on_sem_B": float(on.std(ddof=1) / math.sqrt(len(on))),
            "on_per_run_B": [float(x) for x in on],
            "shift": float(on.mean() - off.mean()),
            "t": float(r.statistic), "p": float(r.pvalue),
            "df": float(df),
            "secondary": bool(E == 0.02),
        }
        print(f"(16,16) {E:g}: shift {on.mean() - off.mean():+.3f}  "
              f"t {r.statistic:4.1f}  p {r.pvalue:.4f}", flush=True)
    gate = out["fields"]["0.01"]
    out["gate"] = {
        "field": 0.01,
        "rule": "0.01 V/A is CONFIRMED iff its Welch p < 0.05 on the "
                "confirmatory set alone (two-sided, alpha = 0.05); the "
                "pre-registered single test -- no post-hoc additions",
        "p": gate["p"],
        "verdict": "CONFIRMED" if gate["p"] < ALPHA else "NOT CONFIRMED",
    }
    return out


def build_paired_secondary(J: dict) -> dict:
    """The build-paired contrast (PLAN section 2, secondary, never the
    gate): per build, field-on block B minus that build's e0 all-block
    mean; mean +- sem over the 6 builds."""
    e0_by_build = {j["build"]: np.mean([j["direct_p1_blocks"]["A"],
                                        j["direct_p1_blocks"]["B"]])
                   for j in J.values() if j["field"] == 0.0}
    assert len(e0_by_build) == 6
    out = {}
    for E in (0.01, 0.02):
        pairs = []
        builds = []
        for j in J.values():
            if j["field"] == E:
                pairs.append(float(j["direct_p1_blocks"]["B"]
                                   - e0_by_build[j["build"]]))
                builds.append(j["build"])
        v = np.array(pairs)
        sem = float(v.std(ddof=1) / math.sqrt(len(v)))
        out[f"{E:g}"] = {
            "per_build": pairs, "builds": builds,
            "mean": float(v.mean()), "sem": sem,
            "z": float(v.mean() / sem) if sem > 0 else None,
            "secondary": True, "never_the_gate": True,
        }
    return out


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------

def job_record(job: dict, d: dict, c: dict, wall: float,
               sub: list[int] | None = None) -> dict:
    """The per-job store: both arms' block means, fidelity, series.
    sub (smoke only) restricts the fidelity comparison to the lifted
    frames; the record run lifts every frame, so there sub is None."""
    sub_ps = np.array(c["prod_ps"], float)
    sselA, sselB = sub_ps < SPLIT_PS, sub_ps >= SPLIT_PS
    p1_text_arr = np.array(c["p1_text"], float)
    text_blocks = {"A": float(p1_text_arr[sselA].mean()),
                   "B": float(p1_text_arr[sselB].mean()),
                   "nA": int(sselA.sum()), "nB": int(sselB.sum())}
    direct_cmp = d["p1_dense"] if sub is None else d["p1_dense"][sub]
    agree = np.abs(p1_text_arr - direct_cmp)
    max_diff = float(agree.max())
    assert max_diff <= FIDELITY_TOL, \
        (job["name"], max_diff, "statement-vs-direct beyond the print "
                                "quantum -- FIDELITY FAILURE, reported")
    return {
        "name": job["name"], "field": job["field"], "build": job["build"],
        "dump": str(job["dump"]), "build_data": str(job["build_data"]),
        "n_atoms": d["n_atoms"], "census": d["census"],
        "consts": c["consts"], "n_lifted": c["n_lifted"],
        "n_frames": d["n_frames"],
        "t_first": d["t_first"], "t_last": d["t_last"],
        "cadence_ps": CADENCE_PS,
        "direct_p1_blocks": d["direct_p1_blocks"],
        "dipole_p1_blocks": text_blocks,
        "p1_text_vs_direct_max_abs": max_diff,
        "failed_checks": c["failed_checks"],
        "lift_seconds": round(c["lift_seconds"], 1),
        "lifts_per_s": round(c["n_lifted"] / c["lift_seconds"], 3),
        "job_wall_s": round(wall, 1),
        "p1_direct": [round(x, 6) for x in d["p1_dense"]],
        "p1_ec_direct": [round(x, 6) for x in d["p1_ec_dense"]],
        "p1_emc_direct": [round(x, 6) for x in d["p1_emc_dense"]],
        "p1_text": [round(x, 6) for x in c["p1_text"]],
    }


def run(smoke_jobs: int | None, smoke_frames: int, fresh: bool) -> dict:
    t0 = time.perf_counter()
    jobs = discover_jobs()
    dl = composition_dialect()
    print(f"[z4] {len(jobs)} confirmatory jobs (PLAN.md section 1, fixed): "
          f"fields 0 / 0.01 / 0.02 V/A x builds cf16_b1..b6", flush=True)

    smoke = smoke_jobs is not None
    done = {}
    if not smoke and CHECKPOINT.is_file() and not fresh:
        for line in CHECKPOINT.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["name"]] = r
        print(f"[z4] checkpoint: {len(done)} job(s) already done",
              flush=True)
    ckpt = None if smoke else open(CHECKPOINT, "a", encoding="utf-8")

    sel_jobs = jobs if not smoke else jobs[:smoke_jobs]
    recs = []
    for i, job in enumerate(sel_jobs):
        if job["name"] in done:
            recs.append(done[job["name"]])
            continue
        tj = time.perf_counter()
        print(f"[z4] ({i + 1}/{len(sel_jobs)}) {job['name']} "
              f"(field {job['field']:g} V/A): arm 2 dense stream...",
              flush=True)
        d = direct_arm(job)
        print(f"[z4]   {d['n_frames']} frames read, timesteps "
              f"{d['t_first']}..{d['t_last']} step 12000, census+charge "
              f"verified every frame", flush=True)
        print(f"[z4]   arm 1: Y4 read + blind lift of "
              f"{smoke_frames if smoke else d['n_frames']} frames...",
              flush=True)
        if smoke:
            sub = list(range(0, THINNED_FRAMES,
                             max(THINNED_FRAMES // smoke_frames,
                                 1)))[:smoke_frames]
            prod_ps = d["prod_ps"][sub]
            c = chaord_arm_smoke(job, sub, prod_ps, dl)
        else:
            c = chaord_arm(job, d["prod_ps"], dl)
        rec = job_record(job, d, c, time.perf_counter() - tj,
                         sub=sub if smoke else None)
        print(f"[z4]   lift ok; dipole_p1 printed-vs-direct max |diff| "
              f"{rec['p1_text_vs_direct_max_abs']:.2e} "
              f"(tol {FIDELITY_TOL:.1e}); job wall "
              f"{rec['job_wall_s']:.0f} s ({rec['lifts_per_s']} lifts/s)",
              flush=True)
        recs.append(rec)
        if ckpt is not None:
            ckpt.write(json.dumps(rec) + "\n")
            ckpt.flush()
    if ckpt is not None:
        ckpt.close()

    if smoke:
        print(f"[z4] SMOKE run -- pipeline test only, no JSON written; "
              f"wall {time.perf_counter() - t0:.1f} s")
        return {}

    J = {r["name"]: {"field": r["field"],
                     "direct_p1_blocks": r["direct_p1_blocks"],
                     "build": r["build"]}
         for r in recs}
    welch = d39_welch(J)
    paired = build_paired_secondary(J)
    chaining = chaining_check(jobs)

    n_lift_total = sum(r["n_lifted"] for r in recs)
    fid_max = max(r["p1_text_vs_direct_max_abs"] for r in recs)
    fid_min = min(r["p1_text_vs_direct_max_abs"] for r in recs)
    n_failed = sum(len(r["failed_checks"]) for r in recs)
    lift_s = sum(r["lift_seconds"] for r in recs)
    results = {
        "meta": {
            "work_order": "docs/reviews/open_items_v5.md Z4 (Review 11 "
                          "round; D39/D41 binding)",
            "plan": "research/cnt_electrolyte/confirmatory/PLAN.md "
                    "(pre-registered 2026-10-31, before any confirmatory "
                    "job was built, run or submitted); run EXACTLY as "
                    "registered, no other test added after seeing the data",
            "path_note": "PLAN.md's verbatim snippet reads the jobs dict "
                         "from confirmatory_results.json; the file lives "
                         "at research/cnt_electrolyte/analysis/"
                         "confirmatory_results.json per the Z4 work order "
                         "(same structure, same call)",
            "set": "18 jobs: fields 0 / 0.01 / 0.02 V/A x 6 fresh packed "
                   "builds cf16_b1..b6, armchair (16,16) 98.4 A tubes, "
                   "74 EC / 48 EMC / 12 LiPF6 = 1.101 M; 1 ns in-job "
                   "thermalization + 1 ns settle + 2 x 6 ns blocks; ONE "
                   "cluster LAMMPS build (3 Mar 2020, 8 ranks) for all 18",
            "lift_options": 'lift_frame(mode="auto"), NO field= (blind), '
                            "T=300.0, one composition dialect for every "
                            "frame -- the Y8 machinery, imported unchanged",
            "lift_coverage": {
                "rule": "EVERY frame of the committed thinned record -- "
                        "1,001 lifts per job, 18,018 total; no analysis "
                        "subsampling beyond the record's own thinning",
                "lifts_total": n_lift_total,
                "block_frames": [500, 501],
                "measured_lift_seconds_total": round(lift_s, 1),
                "measured_lift_rate_mean": round(n_lift_total / lift_s, 3)},
            "block_protocol": {
                "split_ps": SPLIT_PS, "block_ps": 6000.0,
                "verified": "thinned timesteps 0..12,000,000 at exactly "
                            "12,000 steps (12 ps) in every job; block A = "
                            "t < 6,000 ps (500 frames), block B = "
                            "t >= 6,000 ps (501 frames)",
                "primary": "the D39 gate uses field-on block B and the "
                           "zero-field all-block run means, per PLAN "
                           "section 2 verbatim"},
            "arms": "arm 1 = the blind Chaord statement (dipole_p1); "
                    "arm 2 = direct P1 without chaord (numpy, mol id, "
                    "minimum image, dump charges); the D39 test is run on "
                    "arm 2's direct_p1_blocks per PLAN section 2",
            "fidelity": {
                "criterion": "printed dipole_p1 vs direct P1 on the same "
                             "frames, within the statement's 3-decimal "
                             "print half-quantum 5e-4",
                "max_abs_over_all_frames": fid_max,
                "per_job_min": fid_min,
                "n_frames_compared": n_lift_total},
            "failed_static_checks_total": n_failed,
            "chaining_check": "PLAN section 3: per build, the three field "
                              "jobs share the thermalization seeds on ONE "
                              "LAMMPS build -> identical step-1000000 "
                              "thermo; see meta.chaining and the chaining "
                              "section -- mismatches are reported, never "
                              "absorbed",
            "chaining_consequence": "where a build's three jobs do NOT "
                                    "chain, the build-paired secondary "
                                    "contrast loses its 'identical "
                                    "ladder-t=0 state' premise for the "
                                    "divergent member; the primary D39 "
                                    "gate is UNPAIRED (runs are the "
                                    "units) and is not affected",
            "lift_wall_seconds_total": round(
                sum(r["job_wall_s"] for r in recs), 1),
            "smoke": False,
            "wall_seconds": round(time.perf_counter() - t0, 1),
        },
        "chaining": chaining,
        "welch": welch,
        "secondary_build_paired": paired,
        "jobs": {r["name"]: r for r in recs},
    }
    results["meta"]["chaining_all_identical"] = all(
        v["identical_at_step_1000000"] for v in chaining.values())

    OUT_JSON.write_text(json.dumps(results, indent=1))
    print(f"[z4] wrote {OUT_JSON}")
    print(f"[z4] wall {time.perf_counter() - t0:.1f} s "
          f"(lift time {lift_s:.1f} s)")
    return results


def chaord_arm_smoke(job: dict, sub: list[int], prod_ps: np.ndarray,
                     dl) -> dict:
    """Smoke variant: lift a few frames; same code path, no JSON."""
    _, data_meta = read_lammps_data(str(job["build_data"]))
    tmap = data_meta["type_map"]
    frames = read_lammps_dump(str(job["dump"]), frame_indices=sub,
                              type_map=tmap)
    unknown = tmap.apply(frames, strict=True)
    assert not unknown, f"{job['name']}: unmapped type ids {unknown}"
    assert len(frames) == len(sub)
    feats_list, consts_list, failed = [], [], []
    p1_text = []
    t_lift = 0.0
    for fr, ps in zip(frames, prod_ps):
        assert abs(fr.info["timestep"] / 1000.0 - ps) < 1e-9, \
            (job["name"], fr.info["timestep"], ps)
        assert abs(np.asarray(fr.info["charges"]).sum()) < CHARGE_TOL
        t0 = time.perf_counter()
        text, info = lift_one_frame(fr, None, dl=dl)
        t_lift += time.perf_counter() - t0
        feats, consts = y8_statement_features(text)
        feats_list.append(feats)
        consts_list.append(json.dumps(consts, sort_keys=True))
        if info["failed_checks"]:
            failed.append({"prod_ps": float(ps),
                           "checks": info["failed_checks"]})
        p1_text.append(feats["dipole_p1"])
    assert len(set(consts_list)) == 1, \
        f"{job['name']}: consts move between frames"
    return {
        "n_lifted": len(frames),
        "prod_ps": [float(x) for x in prod_ps],
        "consts": json.loads(consts_list[0]),
        "features": {k: [f[k] for f in feats_list]
                     for k in sorted(set().union(*[set(f)
                                                   for f in feats_list]))},
        "failed_checks": failed,
        "p1_text": p1_text,
        "lift_seconds": t_lift,
    }


def print_summary(res: dict) -> None:
    w = res["welch"]
    print("\n== Z4 confirmatory set -- the pre-registered D39 call "
          "(PLAN.md section 2 verbatim) ==")
    print(f"  zero-field run means (blocks A+B averaged), n = 6: "
          f"{w['off_mean']:+.4f} +- {w['off_sem']:.4f}")
    for E, d in w["fields"].items():
        tag = " [secondary, dose-response context]" if d["secondary"] else ""
        print(f"  (16,16) {E} V/A vs 0: shift {d['shift']:+.4f} "
              f"(on-B {d['on_mean_B']:+.4f} +- {d['on_sem_B']:.4f}, "
              f"n = 6)  t {d['t']:.2f}  df {d['df']:.1f}  "
              f"p {d['p']:.4f}{tag}")
    g = w["gate"]
    print(f"  GATE 0.01 V/A: p = {g['p']:.4f} vs alpha = {w['alpha']} -> "
          f"** {g['verdict']} **")
    for E, p in res["secondary_build_paired"].items():
        print(f"  build-paired secondary {E} V/A: "
              f"{p['mean']:+.4f} +- {p['sem']:.4f} "
              f"(z {p['z']:.1f}) -- secondary, never the gate")
    print(f"  fidelity: printed dipole_p1 vs direct P1, max |diff| over "
          f"{sum(r['n_lifted'] for r in res['jobs'].values())} frames = "
          f"{res['meta']['fidelity']['max_abs_over_all_frames']:.2e} "
          f"(print half-quantum 5e-4)")
    print(f"  chaining: per build, chain groups at step 1000000 = "
          f"{[c['chain_groups_step_1000000'] for c in res['chaining'].values()]} "
          f"(PLAN section 3: mismatches reported, never absorbed; "
          f"divergence begins in the shared stage-1 ramp where a job "
          f"diverges)")
    print(f"  failed static checks: "
          f"{res['meta']['failed_static_checks_total']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", nargs=2, type=int, metavar=("JOBS", "FRAMES"),
                    help="first JOBS jobs, FRAMES frames each; pipeline "
                         "test only, never for the record (no JSON)")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore any existing checkpoint")
    args = ap.parse_args()
    res = run(args.smoke[0] if args.smoke else None,
              args.smoke[1] if args.smoke else 4,
              fresh=args.fresh)
    if res:
        print_summary(res)


if __name__ == "__main__":
    main()
