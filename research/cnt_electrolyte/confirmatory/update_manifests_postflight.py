# -*- coding: utf-8 -*-
"""Postflight: complete the per-job manifests after the cluster evidence is
downloaded (confirmatory/cluster_results/<name>/). Hashes the returned files,
reads prod_thinned.sha256, extracts the cluster LAMMPS identity + wall time
from log.lammps, and runs the cross-build chaining check (the three field
jobs of each build must show IDENTICAL thermo at the end of the shared
thermalization, step 1000000 -- PLAN.md section 3). REPORT-only: any
mismatch is recorded, never absorbed.

STANDALONE: does not import chaord.
"""
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "cluster_results")


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def log_identity(logpath):
    """(LAMMPS version string, total wall time) from a log.lammps."""
    version = wall = None
    with open(logpath, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if version is None and line.startswith("LAMMPS ("):
                version = line.strip()
            if "Total wall time" in line:
                wall = line.split("Total wall time")[1].strip(" :=\n")
                break
    return version, wall


THERMO_ROW = re.compile(
    r"^\s*1000000\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)")


def step1e6_rows(logpath):
    """All thermo rows at step 1000000 (end of the shared thermalization)."""
    rows = []
    with open(logpath, encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = THERMO_ROW.match(line)
            if m:
                rows.append(tuple(m.groups()))
    return rows


def main():
    index_path = os.path.join(HERE, "jobs_index.json")
    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)
    # group by build for the chaining check
    by_build = {}
    for j in index:
        by_build.setdefault(j["build"], []).append(j["name"])

    chaining = {}
    for build, names in sorted(by_build.items()):
        rows = {}
        for n in names:
            lp = os.path.join(RESULTS, n, "log.lammps")
            rows[n] = step1e6_rows(lp) if os.path.isfile(lp) else None
        vals = set()
        for n in names:
            if rows[n]:
                vals.update(rows[n])
        chaining[build] = dict(
            jobs=names,
            identical_at_step_1000000=(len(vals) == 1),
            n_distinct_rows=len(vals),
            rows={n: rows[n] for n in names},
        )

    for j in index:
        name = j["name"]
        d = os.path.join(RESULTS, name)
        if not os.path.isdir(d):
            print(f"[postflight] {name}: no cluster_results dir - skipped")
            continue
        files = []
        for fn in ("final.data", "log.lammps", "equil.dump",
                   "prod_thinned.dump", "prod_thinned.sha256", "sbatch.sh"):
            p = os.path.join(d, fn)
            if os.path.isfile(p):
                files.append(dict(role=fn, path=os.path.relpath(p, HERE),
                                  bytes=os.path.getsize(p), sha256=sha256(p)))
        outs = sorted(glob.glob(os.path.join(d, "*.out")))
        for p in outs:
            files.append(dict(role="slurm out", path=os.path.relpath(p, HERE),
                              bytes=os.path.getsize(p), sha256=sha256(p)))
        thin = None
        tp = os.path.join(d, "prod_thinned.sha256")
        if os.path.isfile(tp):
            with open(tp, encoding="utf-8") as f:
                thin = json.load(f)
        version = wall = None
        lp = os.path.join(d, "log.lammps")
        if os.path.isfile(lp):
            version, wall = log_identity(lp)
        man_path = os.path.join(HERE, "manifests", name + ".json")
        with open(man_path, encoding="utf-8") as f:
            man = json.load(f)
        man["postflight"] = dict(
            files=files, prod_thinned=thin,
            cluster_lammps=version, wall_time=wall,
            chaining_check=chaining[j["build"]],
        )
        with open(man_path, "w", encoding="utf-8") as f:
            json.dump(man, f, indent=2)
        print(f"[postflight] {name}: {len(files)} files, LAMMPS "
              f"{version or '?'}, wall {wall or '?'}")
    print("[postflight] cross-build chaining (step-1000000 thermo identity):")
    for build, c in chaining.items():
        flag = "OK" if c["identical_at_step_1000000"] else "MISMATCH"
        print(f"  {build}: {flag} ({c['n_distinct_rows']} distinct rows)")


if __name__ == "__main__":
    main()
