"""Text -> IR -> JSON -> IR -> text is lossless."""
from pathlib import Path

import pytest

from chaord.lang.api import from_json, to_json
from chaord.lang.fmt import format_program
from chaord.lang.ir import ir_equal
from chaord.lang.parser import parse_text

EXAMPLES = sorted((Path(__file__).parent.parent / "spec" / "examples").glob("*.chaord"))


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_json_round_trip(path):
    program = parse_text(path.read_text(encoding="utf-8"))
    j = to_json(program)
    back = from_json(j)
    assert ir_equal(back, program)
    assert format_program(back) == format_program(program)


def test_json_schema_compiles():
    from chaord.lang.ir import Program
    schema = Program.model_json_schema()
    assert schema["title"] == "Program"
    assert "Block" in str(schema)


def test_ir_equality_ignores_lines():
    a = parse_text("chaord 0.1\n\nsystem {\n  seed 7\n}\n")
    b = parse_text("chaord 0.1\n\n# extra comment line\n\nsystem {\n  seed 7\n}\n")
    assert ir_equal(a, b)
