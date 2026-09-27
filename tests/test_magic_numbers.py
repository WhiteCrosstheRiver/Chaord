"""CI gate: no bare float literals in lift/build code (AGENTS.md rule)."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def test_no_magic_numbers():
    r = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "check_magic_numbers.py")],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
