import sys
from pathlib import Path

import pytest

from chaord.lang.errors import ChaordSyntaxError
from chaord.lang.fmt import format_program
from chaord.lang.ir import ir_equal
from chaord.lang.parser import parse_text

EXAMPLES = sorted((Path(__file__).parent.parent / "spec" / "examples").glob("*.chaord"))


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_parses_and_round_trips(path):
    text = path.read_text(encoding="utf-8")
    program = parse_text(text)
    out = format_program(program)
    assert out == text, f"{path.name} is not byte-stable under fmt"
    again = parse_text(out)
    assert ir_equal(again, program)


def test_all_seven_examples():
    # the original seven plus 08_deposit (the deposit protocol example)
    assert len(EXAMPLES) == 8


def test_sketch_check_still_passes():
    import subprocess
    root = Path(__file__).parent.parent
    r = subprocess.run(
        [sys.executable, str(root / "tools" / "sketch_check.py"),
         *[str(p) for p in EXAMPLES]],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_parse_gives_line_numbers():
    program = parse_text("chaord 0.1\n\nsystem {\n  cell 1 2 3\n}\n")
    system = program.blocks[0]
    assert system.t == "system"
    assert system.statements[0].line == 4


def test_comments_preserved():
    text = "chaord 0.1\n\nsystem {\n  seed 7  # my seed\n}\n"
    program = parse_text(text)
    assert program.blocks[0].statements[0].comment == "my seed"
    assert format_program(program) == text


def test_hash_inside_string_is_not_a_comment():
    text = 'chaord 0.1\n\nsystem {\n  note "a # b"\n}\n'
    program = parse_text(text)
    assert program.blocks[0].statements[0].values[0].text == "a # b"
    assert format_program(program) == text


def test_residual_multiple_atom_statements():
    """Regression: consecutive `atom` statements in a residual body must
    re-parse (the lifter writes one line per unexplained atom)."""
    NL = chr(10)
    text = (f"chaord 0.1{NL}{NL}system {{{NL}  pbc xyz{NL}}}{NL}{NL}residual {{{NL}"
            f"  atom X 1.0 2.0 3.0{NL}  atom X 4.0 5.0 6.0{NL}}}{NL}")
    program = parse_text(text)
    residual = next(b for b in program.blocks if b.t == "residual")
    assert len(residual.statements) == 2
    assert all(s.key == "atom" for s in residual.statements)
    assert format_program(program) == text


def test_species_block_round_trip():
    NL = chr(10)
    text = (f"chaord 0.1{NL}{NL}species {{{NL}"
            "  molecule EC = smiles \"C1COC(=O)O1\""
            f"{NL}  ion Li+{NL}  atom Ni{NL}}}{NL}")
    program = parse_text(text)
    defs = next(b for b in program.blocks if b.t == "species").defs
    assert [(d.k, d.name, d.source) for d in defs] == [
        ("molecule", "EC", "smiles"), ("ion", "Li+", None), ("atom", "Ni", None)]
    assert format_program(program) == text
