# Bookkeeping runs

Nothing here is citable evidence. `AGENTS.md` allows reports to quote numbers only
from clean-runner artifacts; a self-run may be shown only as labelled bookkeeping.
`reports/acceptance.json` remains the clean-runner artifact.

## Self-run at d06374a (2026-11-10, job 115392, gerui `dell-cpu`, tag `gate_head`)

- `tools/acceptance.py`: 14 of 14 (`acceptance_d06374a.json`,
  `acceptance_details_d06374a.md`). 19,382 s.
- pytest: 1346 passed, 3 failed, 2 skipped, 270 deselected (`pytest_d06374a.log`).
  All three failures are `tests/acceptance/test_o2_o3.py`, `ModuleNotFoundError: No
  module named 'pymatgen'`; those tests did not exercise the code on this machine.
- The run came from an export without git metadata, so `provenance.commit` is empty.
  The 79 files under `src/` and `tools/` on the node were compared one by one by
  SHA-256 with `git archive d06374a` and are identical. Other directories were not
  compared.

## Why this is not clean-machine evidence

The last clean-runner pass is at `ba6b594` (dispatch run 37868982335; scheduled runs
on 2026-10-11 and 2026-10-12). The scheduled clean-machine acceptance job, which runs
on `master`, then failed every day from 2026-10-16 to 2026-11-09 (25 runs), in about
2 to 3 hours and so not at the 6-hour job limit. The failing criterion is not known
here: job logs and artifacts require a signed-in GitHub account. This self-run passing
does not explain those failures: it used a different machine and later code.
