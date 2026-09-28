"""CI check: no bare float literals in pass, builder, CV or backend code.

AGENTS.md rule: thresholds live in dialect YAML files, read by name. Exceptions
need a `# dialect-exempt: <reason>` tag on the line, or a
`# dialect-exempt-begin: <reason>` ... `# dialect-exempt-end` block (for exact
geometry data tables such as the prototype registry). Integers are allowed.
Lines inside module/function docstrings are prose, not code, and are skipped
(the docstring tracker below is deliberately simple: it only understands
triple-quoted string spans). `src/chaord/check` is not scanned yet — it is
being reworked on the mainline and will join this check afterwards.
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

_TRIPLE_QUOTE = re.compile(r'"""|\x27\x27\x27')

EXEMPT = "# dialect-exempt:"


def _code_segments(line: str, state: str | None):
    """Segments of `line` outside a triple-quoted string span.

    `state` is the still-open triple-quote marker carried over from the
    previous line (or None). Returns (segments, new_state)."""
    segments = []
    pos = 0
    while pos < len(line):
        if state is not None:
            idx = line.find(state, pos)
            if idx == -1:
                return segments, state       # still inside the docstring
            pos = idx + len(state)
            state = None
        else:
            m = _TRIPLE_QUOTE.search(line, pos)
            if m is None:
                segments.append(line[pos:])
                return segments, state
            segments.append(line[pos:m.start()])
            state = m.group()
            pos = m.end()
    return segments, state


def scan(path: Path):
    bad = []
    in_block = False
    doc_state = None
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("# dialect-exempt-begin:"):
            in_block = True
            continue
        if stripped.startswith("# dialect-exempt-end"):
            in_block = False
            continue
        segments, doc_state = _code_segments(line, doc_state)
        if in_block or EXEMPT in line:
            continue
        code_line = " ".join(seg for seg in segments if seg)
        code = code_line.split("#")[0]
        if not code.strip():
            continue
        for m in FLOAT_RE.finditer(code_line):
            bad.append((lineno, line.strip(), m.group()))
    return bad


def main() -> int:
    root = Path(__file__).resolve().parent.parent / "src" / "chaord"
    total = 0
    for sub in ("lift", "build", "cv", "realize"):
        for path in sorted((root / sub).rglob("*.py")):
            for lineno, line, lit in scan(path):
                print(f"{path.relative_to(root.parent.parent)}:{lineno}: bare literal {lit!r}: {line}")
                total += 1
    if total:
        print(f"\n{total} bare float literal(s) in pass/builder/CV/backend code; "
              f"move them into a dialect YAML or tag '# dialect-exempt: <reason>'")
        return 1
    print("magic-number check: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
