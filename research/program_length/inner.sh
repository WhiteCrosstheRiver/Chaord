#!/bin/bash
# Runs on mgmt inside the allocation (see launch.sh).
set -uo pipefail
TAG=$1; shift
J=$SLURM_JOB_ID
sbcast -f "$HOME/chaord_bundle_$TAG.tgz" "/tmp/cb_$J.tgz"
sbcast -f research/program_length/node.sh "/tmp/cn_$J.sh"
srun -n 1 bash "/tmp/cn_$J.sh" "$@" > "results/$TAG.tgz"
rc=$?
srun -n 1 rm -f "/tmp/cb_$J.tgz" "/tmp/cn_$J.sh"
echo "job $J tag $TAG rc $rc"
exit $rc
