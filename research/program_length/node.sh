#!/bin/bash
# Runs on the compute node: unpack, run, stream results to stdout, clean up.
J=$SLURM_JOB_ID
D=/tmp/cw_$J
mkdir -p "$D" && tar xzf "/tmp/cb_$J.tgz" -C "$D" && cd "$D" || exit 90
export PYTHONPATH="$D/pylib:$D/src:$D" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p out
/apps/softwares/anaconda3/bin/python3 "$@" > out/run.out 2> out/run.err
rc=$?
echo "$rc" > out/rc
tar czf - out
cd / && rm -rf "$D"
exit $rc
