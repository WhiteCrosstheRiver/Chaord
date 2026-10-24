# -*- coding: utf-8 -*-
"""
Stage B queue runner (Review 10, Y7: tau_M-limited field ladder).

Same pattern as stageA_queue.py (stdlib only, no chaord):

  * executes jobs from stageB/jobs_index.json in priority order
    (cnt10 before cnt16; within a tube: e10, e50, e0, e100, e5, e20,
    e200 - the go/no-go core fields first),
  * stages each job's system.data from the DONE Stage A zero-field
    final (stageA/jobs/zf_cntN_bK/final.data) - job dirs stay
    self-contained and cluster-portable,
  * logs wall times to stageB/queue_log.jsonl,
  * writes a per-run sha256 manifest to stageB/manifests/<job>.json,
  * skips completed jobs (DONE marker) - restartable at any time,
  * aborts after 3 failed jobs.

CORE COORDINATION (owner order 2026-10-25): this box has ONE serial
core for lmp. Before starting, the runner refuses to launch if
  (a) an lmp.exe process is already running (tasklist, best-effort,
      non-Windows skips the check), or
  (b) a queue_start with no matching queue_end is found in
      stageA/queue_log.jsonl (the fills queue) or this queue's own log
      (i.e. a Stage B queue is already alive).
Override with --force ONLY if the detected run is known-dead (a stale
marker from a killed queue; verify the pid first).

Usage:
    python stageB_queue.py [--dry-run] [--only JOB ...] [--smoke]
                           [--lmp PATH] [--force]
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

HERE = os.path.dirname(os.path.abspath(__file__))      # .../stageB
ROOT = os.path.dirname(HERE)                            # .../cnt_electrolyte
JOBS = os.path.join(HERE, "jobs")
MANIFESTS = os.path.join(HERE, "manifests")
QUEUE_LOG = os.path.join(HERE, "queue_log.jsonl")
QUEUE_RUNNING = os.path.join(HERE, "QUEUE_RUNNING")
STAGEA_QUEUE_LOG = os.path.join(ROOT, "stageA", "queue_log.jsonl")

DEFAULT_LMP = os.environ.get(
    "LMP", "C:/ProgramData/anaconda3/Scripts/lmp.exe")

# execution priority: cheapest results that feed the D32 comparison first
TUBE_RANK = {10: 0, 16: 1}
FIELD_RANK = {10: 0, 50: 1, 0: 2, 100: 3, 5: 4, 20: 5, 200: 6}


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


# ------------------------------------------------------------ coordination
def lmp_running():
    """True if an lmp.exe process exists (Windows tasklist, best-effort;
    returns None when the check is unavailable). Byte-level check: the
    image name is ASCII, and the surrounding text is locale-encoded
    (GBK here) - decoding it as text crashes on a UTF-8-default Python
    (found on first launch 2026-10-25)."""
    try:
        out = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq lmp.exe"],
            capture_output=True, timeout=30).stdout
    except Exception:
        return None
    return b"lmp.exe" in out.lower()


def queue_log_open_start(path):
    """True if the jsonl queue log's last start event has no end event
    (a queue that considers itself running - or a stale marker)."""
    if not os.path.isfile(path):
        return False, None
    running = False
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("event") == "queue_start":
                running, rec_start = True, rec
            elif rec.get("event") == "queue_end":
                running = False
    return running, (locals().get("rec_start") if running else None)


def coordination_check(force):
    if force:
        log_event("coordination", note="--force: start checks skipped")
        return
    blockers = []
    lmp = lmp_running()
    if lmp:
        blockers.append("an lmp.exe process is already running")
    elif lmp is None:
        log_event("coordination", note="tasklist unavailable - "
                                     "lmp process check skipped")
    for label, path in (("the fills/stageA queue", STAGEA_QUEUE_LOG),
                        ("a stageB queue", QUEUE_LOG)):
        running, rec = queue_log_open_start(path)
        if running:
            blockers.append(f"{label} has an open queue_start "
                            f"(started {rec.get('ts')}, pid "
                            f"{rec.get('pid')}) in {path}")
    if blockers:
        for b in blockers:
            print("[queue] BLOCKED: " + b, file=sys.stderr, flush=True)
        sys.exit("refusing to start: one serial core, and it is busy. "
                 "Verify the pid above is alive; use --force only for a "
                 "known-dead stale marker.")


# ------------------------------------------------------------------- jobs
def write_manifest(job, wall_s, lmp):
    d = os.path.join(JOBS, job["name"])
    man = dict(
        job=job["name"], kind=job["kind"], tube=job["tube"],
        build=job["build"], run=job["run"], field=job["field"],
        field_mv=job["field_mv"],
        lmp=os.path.abspath(lmp),
        finished=_dt.datetime.now().isoformat(timespec="seconds"),
        wall_seconds=round(wall_s, 1),
        protocol=dict(settle_ps=job["settle_ps"], block_ps=job["block_ps"],
                      n_blocks=job["n_blocks"],
                      start_from=job["depends"], tranche=job["tranche"]),
        files=[],
    )
    for fname in ("in.lammps", "system.data", "log.lammps", "screen.log",
                  "final.data", "DONE"):
        p = os.path.join(d, fname)
        if os.path.isfile(p):
            man["files"].append(dict(role=fname, path=os.path.relpath(p, ROOT),
                                     bytes=os.path.getsize(p),
                                     sha256=sha256(p)))
    # raw dumps: listed with hashes, kept OUT of git (stageB/.gitignore)
    for fname in sorted(os.listdir(d)):
        if fname.endswith((".dump", ".data")) and fname != "final.data" \
                and fname != "system.data":
            p = os.path.join(d, fname)
            man["files"].append(dict(role="raw dump", path=os.path.relpath(p, ROOT),
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

    # dependency: the Stage A zero-field final must be complete
    dep_done = os.path.join(ROOT, job["source"])
    if not os.path.isfile(dep_done):
        log_event("skip", job=name,
                  reason="missing start state " + job["source"])
        return "blocked"
    shutil.copyfile(dep_done, os.path.join(d, "system.data"))

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
    ap.add_argument("--force", action="store_true",
                    help="skip the one-core coordination check (stale "
                         "markers only - verify the pid first)")
    args = ap.parse_args()

    os.makedirs(MANIFESTS, exist_ok=True)
    with open(os.path.join(HERE, "jobs_index.json")) as f:
        jobs = json.load(f)
    jobs.sort(key=lambda j: (TUBE_RANK.get(j["tube"], 9),
                             FIELD_RANK.get(j["field_mv"], 9), j["run"]))
    if args.only:
        jobs = [j for j in jobs if j["name"] in set(args.only)]
        if not jobs:
            sys.exit("no matching jobs")

    if args.dry_run:
        for j in jobs:
            print(f"{j['name']:<20} tube={j['tube']:<3} field={j['field']:<6} "
                  f"run={j['run']} atoms={j['atoms']:<6} "
                  f"steps={j['steps_ps']:>6} ps  "
                  f"est={j['est_wall_min']:>7} min  "
                  f"tranche={j['tranche'] or '-'}")
        total = sum(j["est_wall_min"] for j in jobs)
        print(f"total estimated serial wall time: {total:.0f} min "
              f"= {total / 60:.1f} h = {total / 1440:.1f} days")
        return

    if not args.smoke:
        coordination_check(args.force)

    log_event("queue_start", lmp=os.path.abspath(args.lmp),
              pid=os.getpid(), n_jobs=len(jobs), smoke=bool(args.smoke),
              only=args.only or "all")
    if not args.smoke:
        with open(QUEUE_RUNNING, "w") as f:
            json.dump(dict(pid=os.getpid(),
                           ts=_dt.datetime.now().isoformat(
                               timespec="seconds"),
                           jobs=[j["name"] for j in jobs]), f, indent=2)
    counts = {}
    try:
        for j in jobs:
            r = run_job(j, args.lmp, args.smoke)
            counts[r] = counts.get(r, 0) + 1
            if counts.get("failed", 0) >= 3:
                log_event("abort", reason="3 or more failed jobs")
                break
    finally:
        if not args.smoke and os.path.isfile(QUEUE_RUNNING):
            os.remove(QUEUE_RUNNING)
    log_event("queue_end", **counts)


if __name__ == "__main__":
    main()
