# -*- coding: utf-8 -*-
"""Write one sbatch.sh per confirmatory job (Stage B/fills template, verified
on ime001) plus the queue scripts. RUN BEFORE UPLOAD; nothing is submitted
here (pre-registration: sbatch happens only after the orchestrator's go).

Cluster conventions (all /vol02 -- compute nodes cannot see /dev/shm, and
sbatch -o paths must be on /vol02):
  -p cpu -N 1 -n 8 -t 48:00:00, module load lammps/3Mar20-icc.2022,
  mpirun lmp_mpi -in in.lammps, %j.out inside the job dir,
  post-run thin (12 ps series) and raw-dump deletion on success only.

STANDALONE: does not import chaord.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
JOBS = os.path.join(HERE, "jobs")
CLUSTER = "/vol02/hdd1/ime001/chaord_confirmatory"     # $HOME = /vol02/hdd1/ime001
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

WAVE1 = [f"sb16L_e{f}_b{k}" for f in ("0",) for k in range(1, 7)] + \
        [f"sb16L_e10_b{k}" for k in range(1, 4)]
WAVE2 = [f"sb16L_e10_b{k}" for k in range(4, 7)] + \
        [f"sb16L_e20_b{k}" for k in range(1, 7)]


def main():
    with open(os.path.join(HERE, "jobs_index.json"), encoding="utf-8") as f:
        index = json.load(f)
    names = [j["name"] for j in index]
    assert sorted(WAVE1 + WAVE2) == sorted(names), "wave split != run list"
    for j in index:
        jobdir = f"{CLUSTER}/jobs/{j['name']}"
        with open(os.path.join(JOBS, j["name"], "sbatch.sh"), "w",
                  encoding="utf-8", newline="\n") as f:
            f.write(SBATCH.format(jobdir=jobdir, name=j["name"],
                                  cluster=CLUSTER))
    print(f"[make-sbatch] wrote {len(index)} sbatch.sh files")

    w1 = " ".join(WAVE1)
    w2 = " ".join(WAVE2)
    submit = f"""\
#!/bin/bash
# Z3 confirmatory submission (PLAN.md section 7). Waves are DISK-bound:
# each running job holds a ~2.4 GB raw 1-ps prod.dump until it self-thins.
# Usage: ./submit.sh wave1 | wave2 | status | done
set -e
cd "$HOME/chaord_confirmatory/jobs"
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
    for j in sb16L_e0_b1 sb16L_e0_b2 sb16L_e0_b3 sb16L_e0_b4 sb16L_e0_b5 \\
             sb16L_e0_b6 sb16L_e10_b1 sb16L_e10_b2 sb16L_e10_b3 \\
             sb16L_e10_b4 sb16L_e10_b5 sb16L_e10_b6 sb16L_e20_b1 \\
             sb16L_e20_b2 sb16L_e20_b3 sb16L_e20_b4 sb16L_e20_b5 \\
             sb16L_e20_b6; do
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
# Upload the staged confirmatory inputs to the cluster (inputs only; NO
# sbatch here -- submission is the orchestrator's go, via ./submit.sh).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
ssh {HOST} 'mkdir -p chaord_confirmatory/jobs'
tar czf - -C "$HERE" jobs thin_prod.py submit.sh |
  ssh {HOST} 'tar xzf - -C chaord_confirmatory && chmod +x chaord_confirmatory/submit.sh'
echo "[upload] cluster inventory:"
ssh {HOST} 'ls chaord_confirmatory/jobs | sort | tr "\\n" " "; echo; \\
  ls chaord_confirmatory/jobs/*/in.lammps | wc -l'
"""
    with open(os.path.join(HERE, "upload_confirmatory.sh"), "w",
              encoding="utf-8", newline="\n") as f:
        f.write(upload)

    download = f"""\
#!/bin/bash
# Download the 只算不存 evidence set per completed job into cluster_results/
# (final.data, log.lammps, equil.dump, prod_thinned.dump + its sha256,
# THIN_DONE, sbatch.sh, %j.out). Run AFTER the wave reports DONE.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
for j in {w1} {w2}; do
  d="$HERE/cluster_results/$j"
  mkdir -p "$d"
  if ssh {HOST} "test -f chaord_confirmatory/jobs/$j/THIN_DONE"; then
    scp -q "{HOST}:chaord_confirmatory/jobs/$j/final.data" \\
       "{HOST}:chaord_confirmatory/jobs/$j/log.lammps" \\
       "{HOST}:chaord_confirmatory/jobs/$j/equil.dump" \\
       "{HOST}:chaord_confirmatory/jobs/$j/prod_thinned.dump" \\
       "{HOST}:chaord_confirmatory/jobs/$j/prod_thinned.sha256" \\
       "{HOST}:chaord_confirmatory/jobs/$j/THIN_DONE" \\
       "{HOST}:chaord_confirmatory/jobs/$j/sbatch.sh" "$d/"
    scp -q "{HOST}:chaord_confirmatory/jobs/$j/*.out" "$d/" || true
    echo "[download] $j ok"
  else
    echo "[download] $j NOT DONE - skipped"
  fi
done
"""
    with open(os.path.join(HERE, "download_confirmatory.sh"), "w",
              encoding="utf-8", newline="\n") as f:
        f.write(download)
    print("[make-sbatch] wrote submit.sh, upload_confirmatory.sh, "
          "download_confirmatory.sh")
    print(f"[make-sbatch] wave 1: {w1}")
    print(f"[make-sbatch] wave 2: {w2}")


if __name__ == "__main__":
    main()
