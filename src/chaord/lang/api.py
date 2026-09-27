"""Convenience API for the language layer: parse, format, JSON round trip."""
from __future__ import annotations

from pathlib import Path

from .fmt import format_program
from .ir import Program
from .parser import parse_file, parse_text


def load(path) -> Program:
    return parse_file(path)


def save(program: Program, path) -> None:
    Path(path).write_text(format_program(program), encoding="utf-8")


def to_json(program: Program) -> str:
    return program.model_dump_json()


def from_json(text: str) -> Program:
    return Program.model_validate_json(text)
