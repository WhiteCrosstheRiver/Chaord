"""A1 (parse and format) -- RED: formatting swallows string semantics.

`species { atom X = file "C:\\data\\x.xyz" }` parses to the StrVal
'C:datax.xyz': _unquote maps every unrecognized escape to the bare character
(dropping the backslash) and the known ones to control characters, silently.
The A1 property laws (parse==IR, fmt idempotent) hold on the ALREADY
CORRUPTED value, so the criterion passes while the program's file reference
was changed.  Windows paths are the natural victim.

Measured (2026-09-30, this repro): source "C:\\data\\x.xyz" -> StrVal
'C:datax.xyz'; source "C:\\temp\\n.txt" -> StrVal containing a TAB and a LF.

Desired behaviour: either the lexer rejects non-(\\\\|\\"|\\n|\\t) escapes
with a line-numbered ChaordSyntaxError, or unrecognized escapes are preserved
literally.  This test is red until then.
"""
import pytest  # noqa: F401  (kept: the red tests below assert plain equality)

from chaord.lang.parser import parse_text
from chaord.lang.fmt import format_program

def _ref_of(path_literal: str) -> str:
    src = (
        "chaord 0.1\n"
        "dialect core\n"
        "\n"
        'species {\n  atom X = file ' + path_literal + "\n}\n"
        "\n"
        "system {\n  cell 10 10 10\n  pbc xyz\n}\n"
        "\n"
        "crystal c : all {\n  lattice fcc\n  a 4.0\n}\n"
        "\n"
        "residual none\n"
    )
    prog = parse_text(src)
    return prog.blocks[0].defs[0].ref


def test_a1_backslash_path_survives_parse_fmt_parse():
    literal = '"C:\\data\\x.xyz"'
    want = "C:\\data\\x.xyz"
    got = _ref_of(literal)
    assert got == want, (
        f"A1 RED: the STRING literal {literal} parses to {got!r}; the "
        "backslashes were silently swallowed, and parse->fmt->parse is "
        "idempotent on the corrupted value so the A1 laws stay green"
    )


def test_a1_tab_newline_escapes_corrupt_windows_paths():
    # 'C:\\temp\\n.txt': the \\t and \\n are REAL escapes, but a Windows user
    # means directories.  The language must at least round-trip what it was
    # given: whatever rule is chosen, the current output mixes a TAB and a LF
    # into the path.
    got = _ref_of('"C:\\temp\\n.txt"')
    assert "\t" not in got and "\n" not in got, (
        f"A1 RED: source path 'C:\\temp\\n.txt' parsed to {got!r} "
        "(control characters inside a file path)"
    )
