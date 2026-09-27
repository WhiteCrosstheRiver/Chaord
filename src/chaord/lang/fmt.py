"""Canonical formatter: IR -> text, printed by `chaord fmt`.

Canonical form (v0.1):
- two-space indent inside bodies, one statement per line, ";" never printed;
- blocks appear in program order, separated by one blank line;
- a comment is appended to its own statement/header line after two spaces;
- number tokens are printed exactly as parsed (lift writes canonical tokens).
"""
from __future__ import annotations

from .ir import (
    At, Arrow, Direction, Eq, Family, GeoChain, KVDefect, Name, Plane, Plus,
    Program, Quantity, RangeVal, Statement, StrVal, Tol, Wood,
)


def fmt_value(v) -> str:
    t = v.t
    if t == "q":
        return f"{v.num} {v.unit}" if v.unit else v.num
    if t == "r":
        return f"{fmt_value(v.lo)} .. {fmt_value(v.hi)}"
    if t == "tol":
        return f"+- {fmt_value(v.value)}"
    if t == "s":
        return '"' + v.text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\t", "\\t") + '"'
    if t == "arrow":
        return "->"
    if t == "at":
        return "@"
    if t == "plus":
        return "+"
    if t == "eq":
        return "="
    return v.text  # n, dir, fam, pl, wood, kv


def fmt_statement(s: Statement) -> str:
    parts = []
    if s.kind != "build":
        parts.append(s.kind)
    parts.append(s.key)
    for v in s.values:
        parts.append(fmt_value(v))
    line = " ".join(parts)
    if s.comment:
        line += f"  # {s.comment}"
    return line


def fmt_shape(sh) -> str:
    t = sh.t
    if t == "all":
        return "all"
    if t == "rest":
        return "rest"
    if t == "slab":
        return f"slab {sh.axis} {fmt_value(sh.rng)}"
    if t == "box":
        return f"box {fmt_value(sh.xs)} {fmt_value(sh.ys)} {fmt_value(sh.zs)}"
    if t == "sphere":
        return (f"sphere center {sh.cx.num} {sh.cy.num} {sh.cz.num} "
                f"radius {fmt_value(sh.radius)}")
    if t == "cylinder":
        return (f"cylinder axis {sh.axis} center {sh.cx.num} {sh.cy.num} "
                f"radius {fmt_value(sh.radius)}")
    raise ValueError(f"unknown shape {t!r}")


def fmt_geometry(g: GeoChain) -> str:
    out = fmt_shape(g.parts[0])
    for op, part in zip(g.ops, g.parts[1:]):
        out += f" {op} {fmt_shape(part)}"
    return out


def fmt_block(b) -> list[str]:
    t = b.t
    if t in ("system", "physics", "provenance"):
        lines = [f"{t} {{"]
        lines += [f"  {fmt_statement(s)}" for s in b.statements]
        lines.append("}")
        if b.comment:
            lines[0] += f"  # {b.comment}"
        return lines
    if t == "species":
        lines = ["species {"]
        for d in b.defs:
            line = f"  {d.k} {d.name}"
            if d.source:
                line += f" = {d.source} {_quote(d.ref)}"
            if d.comment:
                line += f"  # {d.comment}"
            lines.append(line)
        lines.append("}")
        return lines
    if t == "region":
        header = f"{b.phase} {b.name} : {fmt_geometry(b.geometry)} {{"
        if b.comment:
            header += f"  # {b.comment}"
        lines = [header]
        lines += [f"  {fmt_statement(s)}" for s in b.statements]
        lines.append("}")
        return lines
    if t == "interface":
        header = f"interface {b.a} | {b.b} {{"
        if b.comment:
            header += f"  # {b.comment}"
        lines = [header]
        lines += [f"  {fmt_statement(s)}" for s in b.statements]
        lines.append("}")
        return lines
    if t == "residual":
        if b.none:
            return ["residual none"]
        lines = ["residual {"]
        lines += [f"  {fmt_statement(s)}" for s in b.statements]
        lines.append("}")
        return lines
    raise ValueError(f"unknown block {t!r}")


def _quote(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\t", "\\t") + '"'


def format_program(p: Program) -> str:
    head = [f"chaord {p.version}"]
    if p.dialects:
        head.append("dialect " + " + ".join(p.dialects))
    chunks = ["\n".join(head)]
    chunks += ["\n".join(fmt_block(b)) for b in p.blocks]
    return "\n\n".join(chunks) + "\n"
