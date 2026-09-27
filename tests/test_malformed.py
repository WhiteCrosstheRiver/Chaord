"""Thirty malformed programs; each must report the right source line."""
import re

import pytest

from chaord.lang.errors import ChaordSyntaxError
from chaord.lang.parser import parse_text

# (source, expected error line)
MALFORMED = [
    ("", 1),
    ("system {}\n", 1),
    ("chaord\n", 1),
    ("chaord 0.1 extra\ndialect core\n", 1),
    ("chaord 0.1\ndialect core , lj\n", 2),
    ("chaord 0.1\ndialect core +\n\nsystem {\n}\n", 2),
    ("chaord 0.1\ndialect\nsystem {\n}\n", 2),
    ("chaord 0.1\nfoo {}\n", 2),
    ("chaord 0.1\nsystem {\n", 2),
    ("chaord 0.1\nsystem }\n", 2),
    ("chaord 0.1\nsystem {\n  12\n}\n", 3),
    ("chaord 0.1\nsystem {\n  state\n}\n", 3),
    ("chaord 0.1\nsystem {\n  a 1 +- x\n}\n", 3),
    ("chaord 0.1\nsystem {\n  a .. 2\n}\n", 3),
    ("chaord 0.1\nsystem {\n  a 1}\n", 3),
    ("chaord 0.1\nsystem {\n  a [001] <110\n}\n", 3),
    ("chaord 0.1\nsystem {\n  a 1st x\n}\n", 3),
    ("chaord 0.1\nsystem {\n  a \"unterminated\n}\n", 3),
    ("chaord 0.1\ncrystal A slab z 1 .. 2 {\n}\n", 2),
    ("chaord 0.1\ncrystal A : cone 1 {\n}\n", 2),
    ("chaord 0.1\ncrystal A : slab q 1 .. 2 {\n}\n", 2),
    ("chaord 0.1\ncrystal A : slab z 1 ..\n}\n", 2),
    ("chaord 0.1\ncrystal A : slab z 1 .. 2 and {\n}\n", 2),
    ("chaord 0.1\ncrystal A : box 1 .. 2 3 .. 4 {\n}\n", 2),
    ("chaord 0.1\ncrystal A : sphere center 1 2 3 4 {\n}\n", 2),
    ("chaord 0.1\ncrystal A : cylinder axis q center 1 2 radius 3 {\n}\n", 2),
    ("chaord 0.1\ninterface A B {\n}\n", 2),
    ("chaord 0.1\nspecies {\n  molecule = smiles \"X\"\n}\n", 3),
    ("chaord 0.1\nspecies {\n  molecule X = file\n}\n", 3),
    ("chaord 0.1\nresidual none extra\n", 2),
    ("chaord 0.1\nsolid x : all {\n}\n", 2),
]


@pytest.mark.parametrize("text,expected_line", MALFORMED)
def test_malformed_reports_line(text, expected_line):
    with pytest.raises(ChaordSyntaxError) as exc:
        parse_text(text)
    m = re.search(r"line (\d+)", str(exc.value))
    assert m, f"no line number in {exc.value!r}"
    assert int(m.group(1)) == expected_line, f"{text!r}: {exc.value}"


def test_thirty_cases():
    assert len(MALFORMED) >= 30
