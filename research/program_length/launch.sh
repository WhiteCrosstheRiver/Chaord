#!/bin/bash
# Run one Python script on a gerui compute node and bring its results back.
# Compute nodes cannot see /share/home, so: bundle -> sbcast to node /tmp ->
# run -> stream results back as a tarball -> delete everything on the node.
# Usage (on mgmt, from ~/chaord_work): bash research/program_length/launch.sh \
#        <partition> <cpus> <time> <tag> <script.py> [args...]
set -euo pipefail
cd "$HOME/chaord_work"
PART=$1; CPUS=$2; TIME=$3; TAG=$4; shift 4
mkdir -p results logs
tar czf "$HOME/chaord_bundle_$TAG.tgz" --exclude=./wheels --exclude=./logs --exclude=./results --exclude="./*.sh" .
salloc -p "$PART" -N 1 -n 1 -c "$CPUS" -t "$TIME" -J "chaord_$TAG" \
  bash research/program_length/inner.sh "$TAG" "$@" > "logs/$TAG.launch.log" 2>&1
rc=$?
rm -f "$HOME/chaord_bundle_$TAG.tgz"
exit $rc
