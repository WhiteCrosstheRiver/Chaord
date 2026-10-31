# Z3 confirmatory queue — monitoring and completion note

Cluster: ime001 (`ssh 172.28.191.241`, slurm). Jobs live on shared storage at
`/vol02/hdd1/ime001/chaord_confirmatory/jobs/<name>/` ($HOME is on /vol02;
NOT /dev/shm — compute nodes cannot see it, and sbatch -o paths must be on
/vol02). 18 jobs, 2 waves of 9 (disk-bound, PLAN.md section 7), 8 ranks each,
wall limit 48 h (expected ~7 h).

## Submission (after the orchestrator's go — NOTHING is submitted before)

    bash upload_confirmatory.sh        # push 18 staged job dirs + scripts
    ssh 172.28.191.241 'chaord_confirmatory/submit.sh wave1'
    ssh 172.28.191.241 'chaord_confirmatory/submit.sh status'   # squeue check
    # wave 2 after wave 1 drains (see below)
    ssh 172.28.191.241 'chaord_confirmatory/submit.sh wave2'

## Progress checks (run any time)

    ssh 172.28.191.241 'chaord_confirmatory/submit.sh status'   # squeue
    ssh 172.28.191.241 'chaord_confirmatory/submit.sh done'     # DONE count:
    # a job is DONE iff its dir has final.data AND THIN_DONE (thinned to the
    # 12-ps series; raw prod.dump deleted on cluster — 只算不存)

Per-job health while running (thermo every 1 ps, flushed):

    ssh 172.28.191.241 'tail -3 chaord_confirmatory/jobs/<name>/log.lammps'
    # Tmob should sit near 300 K (thermalization) / 300 K (ladder);
    # step should advance ~1000/min... (8 ranks, 4116 atoms: ~250-300 ps/h)

Disk guard (why waves exist): each running job holds a ~2.4 GB raw prod.dump
until its own sbatch thins and deletes it at completion. Check with
`ssh 172.28.191.241 'df -h /vol02'`. If free space drops below ~10 GB while
jobs run, STOP submitting more; never delete another user's files.

## Failure handling

- A job that dies leaves no final.data; its slurm `<jobid>.out` and
  `log.lammps` say why. Resubmit that one job:
  `ssh 172.28.191.241 'cd chaord_confirmatory/jobs/<name> && sbatch sbatch.sh'`
  (LAMMPS overwrites prod.dump/lagging outputs on restart; the run is
  deterministic from system.data + in.lammps on the same cluster build).
- Repeated failure of the same stage = stop and report; do NOT patch inputs
  on the cluster (parameter drift). Any input change goes through the repo,
  PLAN.md amendment first, re-upload, re-smoke.

## Completion (Z3 "Done when": all 18 DONE with manifests)

1. `submit.sh done` reports 18 / 18.
2. `bash download_confirmatory.sh` — pulls per job into
   `cluster_results/<name>/`: final.data, log.lammps, equil.dump (settle),
   prod_thinned.dump (12-ps series) + prod_thinned.sha256, THIN_DONE,
   sbatch.sh, %j.out. Raw 1-ps dumps never leave the cluster.
3. `python update_manifests_postflight.py` — completes
   `manifests/<name>.json` (postflight: hashes, cluster LAMMPS identity
   "LAMMPS (3 Mar 2020)", wall time, thinned-series bytes/sha256) and runs
   the cross-build chaining check: for each build, the three field jobs
   (e0/e10/e20) share the thermalization seeds, so their step-1000000 thermo
   rows must be IDENTICAL on the one cluster LAMMPS build. A MISMATCH is
   reported in the manifest and flagged to the orchestrator — never
   silently absorbed.
4. Analysis handoff: S5/Z4 runs exactly the PLAN.md section 2 Welch call on
   the 18 thinned series (no other test — D39/D41; the 0.01 V/A decision
   rule is already fixed there).
