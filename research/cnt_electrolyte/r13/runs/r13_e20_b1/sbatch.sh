#!/bin/bash
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 48:00:00
#SBATCH -o /vol02/hdd1/ime001/zmfeng/chaord_r13/code/r13/jobs/r13_e20_b1/%j.out
#SBATCH -J r13_e20_b1
module load lammps/3Mar20-icc.2022
cd /vol02/hdd1/ime001/zmfeng/chaord_r13/code/r13/jobs/r13_e20_b1
mpirun lmp_mpi -in in.lammps
RC=$?
if [ $RC -eq 0 ] && [ -f final.data ]; then
  # 只算不存: thin the 1-ps record to the 12-ps series, then raw dies here
  /vol02/hdd1/ime001/software/anaconda3/bin/python3 /vol02/hdd1/ime001/zmfeng/chaord_r13/code/r13/thin_prod.py . >> thin.log 2>&1
  if [ -f THIN_DONE ]; then
    rm -f prod.dump
    echo "prod.dump thinned (12 ps series) and deleted on cluster (只算不存)"
  fi
fi
exit $RC
