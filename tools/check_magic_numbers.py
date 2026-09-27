"""CI check: no bare float literals in pass or builder code.

AGENTS.md rule: thresholds live in dialect YAML files, read by name. Exceptions
need a `# dialect-exempt: <reason>` tag on the line, or a
`# dialect-exempt-begin: <reason>` ... `# dialect-exempt-end` block (for exact
geometry data tables such as the prototype registry). Integers are allowed.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

FLOAT_RE = re.compile(r"""
    (?<![\w.])                    # not part of an identifier or another number
    -?\d+\.\d+                    # 1.5, 0.32
    (?:[eE][+-]?\d+)?
    (?![\w.])
    |
    (?<![\w.])-?\d+[eE][+-]?\d+(?![\w.])   # 1e-9
    |
    (?<![\w.])-?\d+/\d+(?![\w.])           # 1/3
""", re.VERBOSE)

EXEMPT = "# dialect-exempt:"


def scan(path: Path):
    bad = []
    in_block = False
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("# dialect-exempt-begin:"):
            in_block = True
            continue
        if stripped.startswith("# dialect-exempt-end"):
            in_block = False
            continue
        if in_block or EXEMPT in line:
            continue
        code = line.split("#")[0]
        if not code.strip():
            continue
        for m in FLOAT_RE.finditer(line):
            bad.append((lineno, line.strip(), m.group()))
    return bad


def main() -> int:
    root = Path(__file__).resolve().parent.parent / "src" / "chaord"
    total = 0
    for sub in ("lift", "build"):
        for path in sorted((root / sub).rglob("*.py")):
            for lineno, line, lit in scan(path):
                print(f"{path.relative_to(root.parent.parent)}:{lineno}: bare literal {lit!r}: {line}")
                total += 1
    if total:
        print(f"\n{total} bare float literal(s) in pass/builder code; "
              f"move them into a dialect YAML or tag '# dialect-exempt: <reason>'")
        return 1
    print("magic-number check: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
