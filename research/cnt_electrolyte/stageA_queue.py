# -*- coding: utf-8 -*-
"""
Stage A queue runner (Review 10, work order Y6).

A plain-Python (stdlib only, no chaord, no numpy) sequential runner for
the Stage A job list in stageA/jobs_index.json:

  * executes jobs in dependency order (zf -> wf/sf -> off; fills last),
  * stages each job's system.data (copy of data/<build>.data, or the
    dependency's final.data) so every job directory is self-contained
    and cluster-portable,
  * logs wall times to stageA/queue_log.jsonl,
  * writes a per-run sha256 manifest to stageA/manifests/<job>.json
    covering the input, the staged data, the log, final.data and every
    raw dump (dumps are LISTED but stay out of git via stageA/.gitignore),
  * skips completed jobs (DONE marker), so it can be stopped and
    restarted at any time.

The LAMMPS invocation is overridable:  --lmp /path/to/lmp  (or the LMP
environment variable). Default is the owner's local serial build from
docs/reviews/owner_inputs.md. On a cluster, run this script (or simply
the same in.lammps files) with the cluster lmp binary; nothing else
changes because all paths inside the inputs are relative to the job
directory.

Usage:
    python stageA_queue.py [--dry-run] [--only JOB ...] [--smoke]
                           [--lmp PATH]
"""
import argparse
import datetime as _dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STAGEA = os.path.join(HERE, "stageA")
JOBS = os.path.join(STAGEA, "jobs")
MANIFESTS = os.path.join(STAGEA, "manifests")
QUEUE_LOG = os.path.join(STAGEA, "queue_log.jsonl")

DEFAULT_LMP = os.environ.get(
    "LMP", "C:/ProgramData/anaconda3/Scripts/lmp.exe")

KIND_RANK = {"zf": 0, "wf": 1, "sf": 2, "off": 3, "fill": 4}


def log_event(event, **kw):
    rec = dict(event=event,
               ts=_dt.datetime.now().isoformat(timespec="seconds"))
    rec.update(kw)
    with open(QUEUE_LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print("[queue]", json.dumps(rec), flush=True)


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_manifest(job, wall_s, lmp):
    d = os.path.join(JOBS, job["name"])
    man = dict(
        job=job["name"], kind=job["kind"], tube=job["tube"],
        build=job["build"], field=job["field"],
        lmp=os.path.abspath(lmp),
        finished=_dt.datetime.now().isoformat(timespec="seconds"),
        wall_seconds=round(wall_s, 1),
        files=[],
    )
    for fname in ("in.lammps", "system.data", "log.lammps", "screen.log",
                  "final.data", "DONE"):
        p = os.path.join(d, fname)
        if os.path.isfile(p):
            man["files"].append(dict(role=fname, path=os.path.relpath(p, HERE),
                                     bytes=os.path.getsize(p),
                                     sha256=sha256(p)))
    # raw dumps: listed with hashes, kept OUT of git (stageA/.gitignore)
    for fname in sorted(os.listdir(d)):
        if fname.endswith((".dump", ".data")) and fname != "final.data" \
                and fname != "system.data":
            p = os.path.join(d, fname)
            man["files"].append(dict(role="raw dump", path=os.path.relpath(p, HERE),
                                     bytes=os.path.getsize(p),
                                     sha256=sha256(p)))
    out = os.path.join(MANIFESTS, job["name"] + ".json")
    with open(out, "w") as f:
        json.dump(man, f, indent=2)
    return out


def run_job(job, lmp, smoke):
    name = job["name"]
    d = os.path.join(JOBS, name)
    done_name = "DONE.smoke" if smoke else "DONE"
    done = os.path.join(d, done_name)
    if os.path.isfile(done):
        log_event("skip", job=name, reason=done_name + " exists")
        return "done"

    if job.get("depends"):
        dep_done = os.path.join(JOBS, job["depends"], done_name)
        if not os.path.isfile(dep_done):
            log_event("skip", job=name,
                      reason="dependency not done: " + job["depends"])
            return "blocked"

    # stage system.data (job["source"] is relative to stageA/ for chained
    # final.data sources, to research/cnt_electrolyte/ for build data files)
    base = STAGEA if job["source"].startswith("jobs/") else HERE
    src = os.path.join(base, job["source"])
    dst = os.path.join(d, "system.data")
    if not os.path.isfile(src):
        log_event("fail", job=name, reason="missing source " + job["source"])
        return "failed"
    shutil.copyfile(src, dst)

    cmd = [lmp, "-in", "in.lammps"]
    if smoke:
        cmd += ["-var", "smoke", "1"]
    log_event("start", job=name, atoms=job["atoms"],
              steps_ps=job["steps_ps"],
              est_wall_min=job["est_wall_min"],
              cmd=" ".join(cmd), smoke=bool(smoke))
    t0 = time.time()
    with open(os.path.join(d, "screen.log"), "w") as out:
        rc = subprocess.call(cmd, cwd=d, stdout=out,
                             stderr=subprocess.STDOUT)
    wall = time.time() - t0
    if rc != 0:
        log_event("fail", job=name, rc=rc, wall_s=round(wall, 1))
        return "failed"
    with open(done, "w") as f:
        json.dump(dict(exit_code=rc, wall_seconds=round(wall, 1),
                       lmp=os.path.abspath(lmp),
                       finished=_dt.datetime.now().isoformat(
                           timespec="seconds"),
                       smoke=bool(smoke)), f, indent=2)
    if not smoke:
        man = write_manifest(job, wall, lmp)
        log_event("end", job=name, wall_s=round(wall, 1), manifest=man)
    else:
        log_event("end", job=name, wall_s=round(wall, 1), note="smoke")
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmp", default=DEFAULT_LMP,
                    help="path to the lmp executable (default: the owner's "
                         "local serial build, or $LMP)")
    ap.add_argument("--only", nargs="+", default=None,
                    help="run only these jobs (by name)")
    ap.add_argument("--smoke", action="store_true",
                    help="pass -var smoke 1 (500-step self-tests)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the execution plan and exit")
    args = ap.parse_args()

    os.makedirs(MANIFESTS, exist_ok=True)
    with open(os.path.join(STAGEA, "jobs_index.json")) as f:
        jobs = json.load(f)
    jobs.sort(key=lambda j: (KIND_RANK.get(j["kind"], 9), j["tube"],
                             j["build"] or "", j["name"]))
    if args.only:
        jobs = [j for j in jobs if j["name"] in set(args.only)]
        if not jobs:
            sys.exit("no matching jobs")

    if args.dry_run:
        for j in jobs:
            print(f"{j['name']:<22} kind={j['kind']:<4} atoms={j['atoms']:<6} "
                  f"steps={j['steps_ps']:>5} ps  "
                  f"est={j['est_wall_min']:>7} min  "
                  f"deps={j['depends'] or '-'}")
        total = sum(j["est_wall_min"] for j in jobs)
        print(f"total estimated serial wall time: {total:.0f} min "
              f"= {total / 60:.1f} h = {total / 1440:.1f} days")
        return

    log_event("queue_start", lmp=os.path.abspath(args.lmp),
              n_jobs=len(jobs), smoke=bool(args.smoke))
    counts = {}
    for j in jobs:
        r = run_job(j, args.lmp, args.smoke)
        counts[r] = counts.get(r, 0) + 1
        if counts.get("failed", 0) >= 3:
            log_event("abort", reason="3 or more failed jobs")
            break
    log_event("queue_end", **counts)


if __name__ == "__main__":
    main()
