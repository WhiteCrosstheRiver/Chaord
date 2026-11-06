#!/bin/bash
# Submit the staged build of one or more R13 builds (no production jobs here).
# Usage: bash submit_staged.sh r13_b1 [r13_b2 ...]    run from code/r13/
#
# PY is an absolute path on purpose: `module load lammps/3Mar20-icc.2022`
# puts Intel Python 3.9 / numpy 1.21.4 first on PATH, and sbatch exports the
# caller's environment, so a bare python3 silently changes interpreter.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/vol02/hdd1/ime001/software/anaconda3/bin/python3
mkdir -p "$HERE/logs"
for b in "$@"; do
  pre=$(sbatch --parsable -p cpu -N 1 -n 1 -t 12:00:00 -J ${b}_pre \
    -o "$HERE/logs/${b}_pre_%j.out" --wrap \
    "cd $HERE && $PY r13_staged.py prepack $b && $PY r13_staged.py input $b")
  sbatch --parsable -p cpu -N 1 -n 8 -t 2:00:00 -J ${b}_rlx \
    --dependency=afterok:$pre -o "$HERE/logs/${b}_rlx_%j.out" --wrap \
    "module load lammps/3Mar20-icc.2022; cd $HERE/relax/$b && mpirun lmp_mpi -in in.relax && cd $HERE && $PY r13_staged.py finish $b"
  echo "$b prepack job $pre"
done
