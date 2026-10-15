# Owner inputs (written 2026-10-15, per Review 10 "Do now" step 3)

The owner has not yet supplied the two facts below; every default the briefs
define is recorded explicitly so the streams are unblocked today and the
owner can amend this file at any time.

```yaml
working_field_V_per_A: unknown     # -> the D30 default applies: 0.01 V/A
model: not-yet-provided            # -> Stage A uses the pilot model (D37 note below)
charge_scaling: 1.0                # the pilot model's own charges, unscaled
cluster_job_template: not-yet-provided
lammps_build: C:/ProgramData/anaconda3/Scripts/lmp.exe   # local, serial
```

## What runs where (honest statement)

There is no cluster job script yet. All simulation runs execute locally on
the owner's Windows box with the anaconda LAMMPS build (`lmp.exe`, version
`22 Jul 2025 - Update 4`, serial, 1 MPI rank). The pilot proved the needed
packages are compiled in: `fix rigid/nvt/small` (RIGID+MOLECULE) and PPPM
(KSPACE). Measured serial throughput on the pilot systems: 550 ps of
971-2058 atoms took 12-52 minutes, so a 2 ns run costs roughly 0.8-2 hours
and Stage A's ~13 runs fit in 1-2 days of local serial compute; Stage B's
~50 runs of 10 ns do NOT fit locally at that rate and will need either the
owner's cluster (submit the inputs S3 writes) or a scaled-down local
variant (recorded honestly as such if used).

## Model note (D37)

Until the owner provides the Moltemplate tube+electrolyte model, Stage A
and B use the pilot model (research/cnt_electrolyte/params.py: OPLS-AA-based
rigid EC/EMC from Nutakki 2023 Tables II-III, Joung-Cheatham-recorded Li+,
Werder sp2-C tube carbons, Lorentz-Berthelot mixing recorded as a deviation
from OPLS geometric sigma, full +-1 ion charges). The D37 paper-grade
upgrades (native mixing rule, stated charge scaling, reservoir filling,
10-significant-digit dumps, sha256 manifests with raw trajectories out of
git) apply to every NEW run regardless.

## Working field

No working value named yet. Per D30 the working field is **0.01 V/A** and
the ladder is 0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2 V/A. If the owner names
a value before Stage B starts, add it to the ladder (D30).
