#!/bin/bash
# T3 on one gerui compute node. Data arrive on stdin (tar.gz streamed from the
# analysis machine) straight into node-local /tmp -- nothing is staged in
# /share/home (root fs, little space). Results stream back to stdout's file;
# the node copy is deleted at the end. Run inside salloc.
set -u
J=$SLURM_JOB_ID; D=/tmp/r13_$J; W=$HOME/chaord_work
tar czf "$W/pl_$J.tgz" -C "$W" pylib && sbcast "$W/pl_$J.tgz" "/tmp/pl_$J.tgz"; rm -f "$W/pl_$J.tgz"
srun -n1 bash -c "mkdir -p $D && tar xzf - -C $D && tar xzf /tmp/pl_$J.tgz -C $D && rm -f /tmp/pl_$J.tgz && du -sh $D >&2"
srun -n1 bash -c "cd $D && PYTHONPATH=$D/pylib:$D/src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /apps/softwares/anaconda3/bin/python3 research/cnt_electrolyte/r13/r13_analysis.py --procs 18 > r13.out 2> r13.err; echo \$? > rc" < /dev/null
mkdir -p "$W/results"
srun -n1 bash -c "cd $D && tar czf - rc r13.out r13.err \$(ls research/cnt_electrolyte/r13/r13_results.json 2>/dev/null)" < /dev/null > "$W/results/r13_T3.tgz"
srun -n1 rm -rf "$D" "/tmp/pl_$J.tgz" < /dev/null
echo "job $J T3 done"
