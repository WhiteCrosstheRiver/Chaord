"""A13 -- RED: NaN input kills the process itself, not with an exception.

lift_frame on a frame with NaN positions (or a NaN cell diagonal) crashes
the interpreter with a native segfault (measured exit code 139 on Windows
Git Bash, 2026-09-30) inside scipy's cKDTree -- no exception is raised, so
no harness can record it: A13's 'zero unhandled exceptions' evidence cannot
see inputs like this at all, and a single NaN in one bench frame would take
the whole acceptance run down with it.

The lift must instead refuse with a ChaordError after validating its input.
This test runs the lift in a child process and asserts the child either
succeeds or dies with a Python-level exception (both visible to the
harness), never a signal.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CHILD = r"""
import sys
sys.path.insert(0, {root!r}/src)
sys.path.insert(0, {root!r})
import numpy as np
from chaord.dialects import load_dialect
from chaord.io.frames import read_frame, Frame
from chaord.lift import lift_frame
from tools import acceptance as acc

good = read_frame(acc.case_by_id("crystals/fcc_cu")["frames"][0])
dl = load_dialect(("core",))
probe = {probe}
if probe == "nan_pos":
    f = Frame(pos=np.full((8, 3), np.nan), cell=good.cell,
              symbols=["Cu"] * 8, pbc=good.pbc)
else:
    f = Frame(pos=np.full((8, 3), 5.0), cell=np.diag([np.nan] * 3),
              symbols=["Cu"] * 8, pbc=good.pbc)
try:
    lift_frame(f, dl)
    print("returned")
except Exception as e:
    print("raised", type(e).__name__)
"""


def _run(probe: str):
    return subprocess.run(
        [sys.executable, "-c", CHILD.format(root=str(ROOT), probe=probe)],
        capture_output=True, text=True, timeout=180,
        cwd=str(ROOT))


def test_a13_nan_positions_refuse_instead_of_segfault():
    r = _run("nan_pos")
    assert r.returncode == 0 and ("raised" in r.stdout or "returned" in r.stdout), (
        f"A13 RED: lifting a frame with NaN positions terminates the child "
        f"process with returncode {r.returncode} (139 = SIGSEGV measured); "
        f"stdout={r.stdout.strip()!r} stderr={r.stderr.strip()[-120:]!r} -- "
        "a native crash no exception-based harness can record"
    )


def test_a13_nan_cell_refuse_instead_of_segfault():
    r = _run("nan_cell")
    assert r.returncode == 0 and ("raised" in r.stdout or "returned" in r.stdout), (
        f"A13 RED: lifting a frame with a NaN cell terminates the child "
        f"process with returncode {r.returncode} (139 = SIGSEGV measured); "
        f"stdout={r.stdout.strip()!r} stderr={r.stderr.strip()[-120:]!r}"
    )
