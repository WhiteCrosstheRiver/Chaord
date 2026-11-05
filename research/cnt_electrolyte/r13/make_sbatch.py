# -*- coding: utf-8 -*-
"""Write one sbatch.sh per R13 job plus the queue scripts (the verified
Stage B/confirmatory template). RUN BEFORE UPLOAD; nothing is submitted
here (pre-registration: sbatch happens only after the orchestrator's go).

Cluster conventions (all /vol02 -- compute nodes cannot see /dev/shm):
  -p cpu -N 1 -n 8 -t 48:00:00, module load lammps/3Mar20-icc.2022,
  mpirun lmp_mpi -in in.lammps, %j.out inside the job dir,
  post-run thin (12 ps series) and raw-dump deletion on success only.

STANDALONE: does not import chaord.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
JOBS = os.path.join(HERE, "jobs")
CLUSTER = "/vol02/hdd1/ime001/chaord_r13"     # $HOME = /vol02/hdd1/ime001
HOST = "172.28.191.241"

SBATCH = """\
#!/bin/bash
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 48:00:00
#SBATCH -o {jobdir}/%j.out
#SBATCH -J {name}
module load lammps/3Mar20-icc.2022
cd {jobdir}
mpirun lmp_mpi -in in.lammps
RC=$?
if [ $RC -eq 0 ] && [ -f final.data ]; then
  # 只算不存: thin the 1-ps record to the 12-ps series, then raw dies here
  python3 {cluster}/thin_prod.py . >> thin.log 2>&1
  if [ -f THIN_DONE ]; then
    rm -f prod.dump
    echo "prod.dump thinned (12 ps series) and deleted on cluster (只算不存)"
  fi
fi
exit $RC
"""

# 9 jobs per wave (disk-bound: each running job holds a ~2.9 GB raw dump)
WAVE1 = [f"r13_e0_b{k}" for k in range(1, 7)] + \
        [f"r13_e10_b{k}" for k in range(1, 4)]
WAVE2 = [f"r13_e10_b{k}" for k in range(4, 7)] + \
        [f"r13_e20_b{k}" for k in range(1, 7)]
ALL = WAVE1 + WAVE2


def main():
    with open(os.path.join(HERE, "jobs_index.json"), encoding="utf-8") as f:
        index = json.load(f)
    names = [j["name"] for j in index]
    assert sorted(ALL) == sorted(names), "wave split != run list"
    for j in index:
        jobdir = f"{CLUSTER}/jobs/{j['name']}"
        with open(os.path.join(JOBS, j["name"], "sbatch.sh"), "w",
                  encoding="utf-8", newline="\n") as f:
            f.write(SBATCH.format(jobdir=jobdir, name=j["name"],
                                  cluster=CLUSTER))
    print(f"[make-sbatch] wrote {len(index)} sbatch.sh files")

    w1 = " ".join(WAVE1)
    w2 = " ".join(WAVE2)
    done_list = " ".join(ALL)
    submit = f"""\
#!/bin/bash
# R13 reservoir-filling submission (r13/PLAN.md). Waves are DISK-bound:
# each running job holds a ~2.9 GB raw 1-ps prod.dump until it self-thins.
# Usage: ./submit.sh wave1 | wave2 | status | done
set -e
cd "$HOME/chaord_r13/jobs"
case "${{1:-status}}" in
  wave1)
    echo "[submit] wave 1 (9 jobs): {w1}"
    for j in {w1}; do sbatch "$j/sbatch.sh"; done ;;
  wave2)
    echo "[submit] wave 2 (9 jobs): {w2}"
    for j in {w2}; do sbatch "$j/sbatch.sh"; done ;;
  status)
    squeue -u ime001 | head -30 ;;
  done)
    n=0
    for j in {done_list}; do
      if [ -f "$j/final.data" ] && [ -f "$j/THIN_DONE" ]; then
        n=$((n+1)); else echo "NOT DONE: $j"; fi
    done
    echo "DONE: $n / 18 (final.data + THIN_DONE per job dir)" ;;
  *) echo "usage: ./submit.sh wave1|wave2|status|done"; exit 2 ;;
esac
squeue -u ime001 | head -30
"""
    with open(os.path.join(HERE, "submit.sh"), "w",
              encoding="utf-8", newline="\n") as f:
        f.write(submit)

    upload = f"""\
#!/bin/bash
# Upload the staged R13 inputs to the cluster (inputs only; NO sbatch here
# -- submission is the orchestrator's go, via ./submit.sh).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
ssh {HOST} 'mkdir -p chaord_r13/jobs'
tar czf - -C "$HERE" jobs thin_prod.py submit.sh |
  ssh {HOST} 'tar xzf - -C chaord_r13 && chmod +x chaord_r13/submit.sh'
echo "[upload] cluster inventory:"
ssh {HOST} 'ls chaord_r13/jobs | sort | tr "\\n" " "; echo; \\
  ls chaord_r13/jobs/*/in.lammps | wc -l'
"""
    with open(os.path.join(HERE, "upload_r13.sh"), "w",
              encoding="utf-8", newline="\n") as f:
        f.write(upload)
    print("[make-sbatch] wrote submit.sh + upload_r13.sh")


if __name__ == "__main__":
    main()
