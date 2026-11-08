#!/bin/bash
# R13 reservoir-filling submission (r13/PLAN.md). Waves are DISK-bound:
# each running job holds a ~2.9 GB raw 1-ps prod.dump until it self-thins.
# Usage: ./submit.sh wave1 | wave2 | status | done
set -e
cd "/vol02/hdd1/ime001/zmfeng/chaord_r13/code/r13/jobs"
case "${1:-status}" in
  wave1)
    echo "[submit] wave 1 (9 jobs): r13_e0_b1 r13_e0_b2 r13_e0_b3 r13_e0_b4 r13_e0_b5 r13_e0_b6 r13_e10_b1 r13_e10_b2 r13_e10_b3"
    for j in r13_e0_b1 r13_e0_b2 r13_e0_b3 r13_e0_b4 r13_e0_b5 r13_e0_b6 r13_e10_b1 r13_e10_b2 r13_e10_b3; do sbatch "$j/sbatch.sh"; done ;;
  wave2)
    echo "[submit] wave 2 (9 jobs): r13_e10_b4 r13_e10_b5 r13_e10_b6 r13_e20_b1 r13_e20_b2 r13_e20_b3 r13_e20_b4 r13_e20_b5 r13_e20_b6"
    for j in r13_e10_b4 r13_e10_b5 r13_e10_b6 r13_e20_b1 r13_e20_b2 r13_e20_b3 r13_e20_b4 r13_e20_b5 r13_e20_b6; do sbatch "$j/sbatch.sh"; done ;;
  status)
    squeue -u ime001 | head -30 ;;
  done)
    n=0
    for j in r13_e0_b1 r13_e0_b2 r13_e0_b3 r13_e0_b4 r13_e0_b5 r13_e0_b6 r13_e10_b1 r13_e10_b2 r13_e10_b3 r13_e10_b4 r13_e10_b5 r13_e10_b6 r13_e20_b1 r13_e20_b2 r13_e20_b3 r13_e20_b4 r13_e20_b5 r13_e20_b6; do
      if [ -f "$j/final.data" ] && [ -f "$j/THIN_DONE" ]; then
        n=$((n+1)); else echo "NOT DONE: $j"; fi
    done
    echo "DONE: $n / 18 (final.data + THIN_DONE per job dir)" ;;
  *) echo "usage: ./submit.sh wave1|wave2|status|done"; exit 2 ;;
esac
squeue -u ime001 | head -30
