"""Nightly statistical benchmark: slow round-trip suite + acceptance A1-A14.

Usage: python tools/nightly.py [--skip-slow]
Writes reports/nightly_<date>.md summarising both stages. The slow stage runs
the statistical round trips (minutes); --skip-slow runs acceptance only, for
debugging the harness itself.
"""
from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def run(cmd: list[str]) -> tuple[int, str]:
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    return r.returncode, (r.stdout + r.stderr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-slow", action="store_true")
    args = ap.parse_args()

    date = datetime.date.today().isoformat()
    lines = [f"# Nightly statistical benchmark — {date}", ""]

    slow_ok, slow_tail = True, "(skipped)"
    if not args.skip_slow:
        rc, out = run([PY, "-m", "pytest", "tests", "-m", "slow", "-q"])
        slow_ok = rc == 0
        slow_tail = [l for l in out.splitlines() if l.strip()][-1] if out.strip() else "(no output)"
    lines += ["## Slow statistical suite", "",
              f"- status: {'PASS' if slow_ok else 'FAIL'}", f"- summary: `{slow_tail}`", ""]

    acc_path = ROOT / "reports" / "acceptance.json"
    rc, out = run([PY, str(ROOT / "tools" / "acceptance.py"),
                   "-o", str(acc_path)])
    acc_ok = rc == 0
    criteria = []
    if acc_path.exists():
        try:
            report = json.loads(acc_path.read_text(encoding="utf-8"))
            criteria = report.get("criteria", [])
            lines += ["## Acceptance A1-A14", "",
                      f"- {report.get('passed')}/{report.get('total')} criteria pass", ""]
            for c in criteria:
                lines.append(f"- {'PASS' if c['passed'] else 'FAIL'} {c['id']} {c['name']}: {c['evidence']}")
        except Exception as e:  # pragma: no cover
            lines.append(f"- could not parse acceptance report: {e}")
    else:
        lines.append(f"- acceptance runner did not produce a report (exit {rc})")
    lines.append("")

    overall = slow_ok and acc_ok
    lines.insert(1, f"**Overall: {'PASS' if overall else 'FAIL'}**")
    out_path = ROOT / "reports" / f"nightly_{date}.md"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"nightly report: {out_path} (overall {'PASS' if overall else 'FAIL'})")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
