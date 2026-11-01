# Corrections — commit messages that named files they did not contain

Per Review 12 D50: over-claiming commit messages are listed here; history is
not rewritten. The D50 check (`tools/check_commit_claims.py`, wired into CI)
fails any future commit whose message names a file path absent from that
commit's tree.

| commit | the message claimed | what the commit actually contained | corrected by |
| --- | --- | --- | --- |
| `b994426` | "Per job: final.data + log.lammps committed" (the Y6 fills evidence) | only `fills_cluster/fill_series_manifest.json`; the four `fills_cluster/fill_{cnt10,cnt16}/{final.data,log.lammps}` files were absent (fenced by the directory-level gitignore) | `7e5fe1d` |
| `1957a08` | "every job's final.data + equil.dump + prod_thinned.dump … retrieved to the local working tree" as committed evidence | only `confirmatory/series_manifest.json` + `.gitignore`; the 18 `confirmatory/jobs/sb16L_*/{final.data,log.lammps,equil.dump}` evidence files were absent (fenced) | `4cc4dab` |
| `6f889bd` | "18/18 job inputs generated and smoke-tested … staged self-contained" | the 18 `jobs/sb16L_*/{in.lammps,system.data}` inputs WERE contained — but the same commit also added 18 `jobs/*/prod.dump` smoke-era full dumps (~10 MB each, 180 MB) that D50 excludes; they were later untracked | `4cc4dab` (the dumps untracked; the inputs remain) |
