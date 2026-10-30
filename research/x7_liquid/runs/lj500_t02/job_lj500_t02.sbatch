#!/bin/bash
#SBATCH -J x7lq_lj500_t02
#SBATCH -o /vol02/hdd1/ime001/chaord_x7/liquid_floors/runs/lj500_t02/sbatch.out
#SBATCH -p cpu
#SBATCH -w node07
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -t 00:30:00
module load lammps/3Mar20-icc.2022
cd /vol02/hdd1/ime001/chaord_x7/liquid_floors/runs/lj500_t02
echo "node $(hostname) start $(date +%s)"
lmp -in in.lj500_t02 -log log.lammps -screen none
rc=$?
echo "rc=$rc end $(date +%s)"
echo $rc > rc.txt
