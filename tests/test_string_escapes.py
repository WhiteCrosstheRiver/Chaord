"""F7 (red-team reports/redteam_findings.md, 2026-09-30): string literals
swallowed backslashes.

`atom X = file "C:\\data\\x.xyz"` parsed to the StrVal 'C:datax.xyz': the
escape decoder mapped every unrecognized escape to its bare character,
deleting the backslash, and mapped \\t/\\n to real TAB/LF -- so a Windows
path changed meaning while the A1 laws (parse==IR, fmt idempotent) stayed
green on the already-corrupted value.

The rule these tests pin: only \\\\ and \\" are escapes (the inverse of fmt's
_quote); every other backslash sequence is preserved literally, and fmt
re-escapes on output, so source -> IR -> fmt -> IR is lossless for file
paths. Red before the fix (2026-09-30) except the last two pins, green
before and after."""
import pytest  # noqa: F401  (plain equality asserts below)

from chaord.lang.fmt import format_program
from chaord.lang.parser import parse_text


def _ref_of(path_literal: str) -> str:
    src = ("chaord 0.1\n"
           "dialect core\n"
           "\n"
           'species {\n  atom X = file ' + path_literal + "\n}\n"
           "\n"
           "system {\n  cell 10 10 10\n  pbc xyz\n}\n"
           "\n"
           "residual none\n")
    return parse_text(src).blocks[0].defs[0].ref


def test_backslash_path_survives_parse():
    got = _ref_of('"C:\\data\\x.xyz"')
    assert got == "C:\\data\\x.xyz", (
        f"F7: the STRING literal \"C:\\data\\x.xyz\" parses to {got!r}; the "
        "backslashes were silently swallowed")


def test_backslash_path_survives_parse_fmt_parse():
    literal = '"C:\\temp\\results v2\\x.xyz"'
    want = "C:\\temp\\results v2\\x.xyz"
    src = ("chaord 0.1\ndialect core\n\n"
           'species {\n  atom X = file ' + literal + "\n}\n"
           "\nsystem {\n  cell 10 10 10\n  pbc xyz\n}\n\nresidual none\n")
    prog = parse_text(src)
    assert prog.blocks[0].defs[0].ref == want
    reparsed = parse_text(format_program(prog))
    assert reparsed.blocks[0].defs[0].ref == want, (
        "F7: fmt -> parse changed the file reference"
    )
    assert format_program(reparsed) == format_program(prog), "fmt not idempotent"


def test_tab_newline_sequences_stay_literal_in_paths():
    got = _ref_of('"C:\\temp\\n.txt"')
    assert "\t" not in got and "\n" not in got, (
        f"F7: source path 'C:\\temp\\n.txt' parsed to {got!r} (control "
        "characters inside a file path)")


def test_statement_level_strings_round_trip():
    """physics/model strings take the same path as species refs."""
    src = ("chaord 0.1\ndialect core\n\n"
           "system {\n  cell 10 10 10\n  pbc xyz\n}\n\n"
           'physics {\n  potential "D:\\pots\\Cu.eam.alloy"\n}\n'
           "\nresidual none\n")
    prog = parse_text(src)
    val = prog.blocks[1].statements[0].values[0]
    assert val.text == "D:\\pots\\Cu.eam.alloy"
    assert parse_text(format_program(prog)).blocks[1].statements[0].values[0].text \
        == val.text


def test_recognized_escapes_still_decode():          # pin: green before+after
    assert _ref_of('"a\\\\b"') == "a\\b"      # backslash-backslash -> one
    assert _ref_of('"q\\"z"') == 'q"z'        # escaped quote inside string
