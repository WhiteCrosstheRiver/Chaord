"""Physical sanity checker for the bench/reference cases.

CLI:  python bench/reference/check_sanity.py [--root DIR] [--case NAME]

Re-measures, from the stored frames alone, the checks recorded in each
case's provenance.json (density or per-region density, minimum pair
distances with the case's hard cores, first g(r) peak windows, frame
distinctness) and prints one PASS/FAIL line per check. Writes
sanity_report.json next to the cases and exits non-zero on any failure.

This tool is part of the reference-data package and must not depend on chaord
(circular-validation ban, AGENTS.md).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import ref_common as rc  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(HERE),
                    help="directory holding the case folders")
    ap.add_argument("--case", action="append",
                    help="restrict to one or more cases")
    args = ap.parse_args(argv)

    root = Path(args.root)
    cases = [c for c in sorted(p.name for p in root.iterdir()
                               if (p / "provenance.json").is_file())
             if not args.case or c in args.case]
    if not cases:
        print("no cases found under", root)
        return 2

    all_ok = True
    reports = {}
    for case in cases:
        rep = rc.evaluate_case(root / case)
        reports[case] = rep
        print(f"[{case}] {'PASS' if rep['passed'] else 'FAIL'} "
              f"({rep['n_frames']} frames)")
        for name, chk in rep["checks"].items():
            extra = {k: v for k, v in chk.items() if k != "pass"}
            print(f"    {'ok ' if chk['pass'] else 'FAIL'} {name}: {extra}")
        all_ok &= rep["passed"]

    (root / "sanity_report.json").write_text(
        json.dumps(reports, indent=2), encoding="utf-8")
    print("wrote", root / "sanity_report.json")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
