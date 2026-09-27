"""Parser: text -> IR, with line-numbered errors and comment preservation.

Comments are extracted before lexing (replaced by blanks so Lark's line and column
numbers are unaffected) and re-attached to the statement or region header that
shares their line. Standalone comment-only lines are not part of the canonical
form and are dropped (canonical files never contain them).
"""
from __future__ import annotations

from functools import lru_cache
from importlib.resources import files as _files

from lark import Lark, Token, Tree, UnexpectedInput

from .errors import ChaordSyntaxError
from .ir import (
    At, Arrow, Direction, Eq, Family, GeoChain, InterfaceBlock, KVDefect, Name,
    Plane, Plus, Program, Quantity, RangeVal, RegionBlock, ResidualBlock,
    ShAll, ShBox, ShCylinder, ShSlab, ShSphere, ShRest, SpecDef, SpeciesBlock,
    Statement, StrVal, SystemBlock, PhysicsBlock, ProvenanceBlock, Tol, Wood,
)

_GRAMMAR = (_files("chaord.lang") / "chaord.lark").read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _parser() -> Lark:
    return Lark(
        _GRAMMAR,
        parser="lalr",
        start="start",
        propagate_positions=True,
        maybe_placeholders=False,
    )


# ------------------------------------------------------------------ comments --

def split_comments(text: str) -> tuple[str, dict[int, str]]:
    """Return (text with comments blanked out, {line: comment body})."""
    out = list(text)
    comments: dict[int, str] = {}
    i, n = 0, len(text)
    in_string = False
    line = 1
    while i < n:
        c = text[i]
        if c == "\n":
            line += 1
            in_string = False
            i += 1
            continue
        if in_string:
            if c == "\\":
                i += 2
                continue
            if c == '"':
                in_string = False
            i += 1
            continue
        if c == '"':
            in_string = True
            i += 1
            continue
        if c == "#":
            j = text.find("\n", i)
            if j == -1:
                j = n
            body = text[i + 1:j].strip()
            if body:
                comments[line] = body
            for k in range(i, j):
                out[k] = " "
            i = j
            continue
        i += 1
    return "".join(out), comments


# -------------------------------------------------------------------- parse --

def parse_text(text: str, units=None) -> Program:
    """Parse Chaord source into the IR. `units` is the set of unit words used to
    decide whether a WORD directly after a NUMBER is its unit (dialect-supplied;
    defaults to the core dialect's unit list)."""
    if units is None:
        from ..dialects import load_dialect
        units = load_dialect(("core",)).units
    units = frozenset(units)

    cleaned, comments = split_comments(text)
    if cleaned and not cleaned.endswith("\n"):
        cleaned += "\n"
    try:
        tree = _parser().parse(cleaned)
    except UnexpectedInput as e:
        ctx = ""
        try:
            ctx = e.get_context(cleaned).splitlines()[0].strip()
        except Exception:
            pass
        raise ChaordSyntaxError(
            f"line {e.line}: cannot parse near {ctx[:60]!r}" if ctx
            else f"line {e.line}: cannot parse input"
        ) from e

    header = tree.children[0]
    version = header.children[0].value  # NUMBER token
    dialects: list[str] = []
    for child in header.children[1:]:
        if isinstance(child, Tree) and child.data == "dialect_line":
            dialects = [tok.value for tok in child.children
                        if isinstance(tok, Token) and tok.type == "WORD"]

    blocks = []
    for blk in tree.find_data("block"):
        inner = blk.children[0]
        blocks.append(_block(inner, comments, units))
    return Program(version=version, dialects=dialects, blocks=blocks)


def parse_file(path, units=None) -> Program:
    from pathlib import Path
    return parse_text(Path(path).read_text(encoding="utf-8"), units=units)


# ------------------------------------------------------------------ helpers --

def _tok_line(tok: Token) -> int:
    return getattr(tok, "line", 0) or 0


def _stmt_tokens(tree: Tree) -> list[Token]:
    return [c for c in tree.children
            if isinstance(c, Token) and c.type not in ("NL",)]


def _statement(tree: Tree, comments: dict[int, str], units: frozenset) -> Statement:
    toks = _stmt_tokens(tree)
    pos = 0
    kind = "build"
    if toks and toks[0].type == "KIND":
        kind = toks[0].value
        pos = 1
    key_tok = toks[pos]
    pos += 1

    values = []

    def q(tok: Token, i: int) -> tuple[Quantity, int]:
        unit = None
        if i + 1 < len(toks) and toks[i + 1].type == "WORD" and toks[i + 1].value in units:
            unit = toks[i + 1].value
            i += 1
        return Quantity(num=tok.value, unit=unit, line=_tok_line(tok)), i

    while pos < len(toks):
        tok = toks[pos]
        t = tok.type
        if t == "NUMBER":
            val, pos = q(tok, pos)
            values.append(val)
            pos += 1
        elif t == "PM":
            if pos + 1 >= len(toks) or toks[pos + 1].type != "NUMBER":
                raise ChaordSyntaxError(
                    f"line {_tok_line(tok)}: '+-' must be followed by a number"
                )
            val, i = q(toks[pos + 1], pos + 1)
            values.append(Tol(value=val, line=_tok_line(tok)))
            pos = i + 1
        elif t == "DOTS":
            if not values or not isinstance(values[-1], Quantity):
                raise ChaordSyntaxError(
                    f"line {_tok_line(tok)}: '..' must follow a number"
                )
            if pos + 1 >= len(toks) or toks[pos + 1].type != "NUMBER":
                raise ChaordSyntaxError(
                    f"line {_tok_line(tok)}: range needs a number after '..'"
                )
            lo = values.pop()
            hi, i = q(toks[pos + 1], pos + 1)
            values.append(RangeVal(lo=lo, hi=hi, line=_tok_line(tok)))
            pos = i + 1
        elif t == "WORD":
            values.append(Name(text=tok.value, line=_tok_line(tok)))
            pos += 1
        elif t == "STRING":
            values.append(StrVal(text=_unquote(tok.value), line=_tok_line(tok)))
            pos += 1
        elif t == "DIRECTION":
            values.append(Direction(text=tok.value, line=_tok_line(tok)))
            pos += 1
        elif t == "FAMILY":
            values.append(Family(text=tok.value, line=_tok_line(tok)))
            pos += 1
        elif t == "PLANE":
            values.append(Plane(text=tok.value, line=_tok_line(tok)))
            pos += 1
        elif t == "WOOD":
            values.append(Wood(text=tok.value, line=_tok_line(tok)))
            pos += 1
        elif t == "KV":
            values.append(KVDefect(text=tok.value, line=_tok_line(tok)))
            pos += 1
        elif t == "ARROW":
            values.append(Arrow(line=_tok_line(tok)))
            pos += 1
        elif t == "AT":
            values.append(At(line=_tok_line(tok)))
            pos += 1
        elif t == "PLUS":
            values.append(Plus(line=_tok_line(tok)))
            pos += 1
        elif t == "EQ":
            values.append(Eq(line=_tok_line(tok)))
            pos += 1
        else:  # pragma: no cover - grammar guarantees the token set
            raise ChaordSyntaxError(f"line {_tok_line(tok)}: unexpected token {tok.value!r}")

    line = _tok_line(key_tok)
    return Statement(
        kind=kind, key=key_tok.value, values=values,
        comment=comments.get(line), line=line,
    )


def _unquote(s: str) -> str:
    body = s[1:-1]
    out, i = [], 0
    while i < len(body):
        c = body[i]
        if c == "\\" and i + 1 < len(body):
            nxt = body[i + 1]
            mapped = {"n": "\n", "t": "\t", '"': '"', "\\": "\\"}.get(nxt, nxt)
            out.append(mapped)
            i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _body_statements(body: Tree, comments: dict[int, str], units: frozenset) -> list[Statement]:
    return [
        _statement(t, comments, units)
        for t in body.children
        if isinstance(t, Tree) and t.data == "statement"
    ]


def _quantity(tree: Tree) -> Quantity:
    toks = [c for c in tree.children if isinstance(c, Token)]
    num = toks[0]
    unit = toks[1].value if len(toks) > 1 else None
    return Quantity(num=num.value, unit=unit, line=_tok_line(num))


def _range(tree: Tree) -> RangeVal:
    quants = [c for c in tree.children if isinstance(c, Tree)]
    lo = _quantity(quants[0])
    hi = _quantity(quants[1])
    return RangeVal(lo=lo, hi=hi, line=lo.line)


def _geometry(tree: Tree) -> GeoChain:
    parts, ops = [], []
    for child in tree.children:
        if isinstance(child, Token):  # and / or / minus keyword
            ops.append(child.value)
        else:
            parts.append(_shape(child))
    return GeoChain(parts=parts, ops=ops)


def _shape(tree: Tree):
    data = tree.data
    if data == "all_shape":
        return ShAll()
    if data == "rest_shape":
        return ShRest()
    if data == "slab_shape":
        axis = tree.children[0].value
        rng = _range(tree.children[1])
        return ShSlab(axis=axis, rng=rng)
    if data == "box_shape":
        return ShBox(xs=_range(tree.children[0]), ys=_range(tree.children[1]), zs=_range(tree.children[2]))
    if data == "sphere_shape":
        c = tree.children  # [NUMBER, NUMBER, NUMBER, quantity]
        return ShSphere(
            cx=Quantity(num=c[0].value), cy=Quantity(num=c[1].value),
            cz=Quantity(num=c[2].value), radius=_quantity(c[3]),
        )
    if data == "cylinder_shape":
        c = tree.children  # [AXIS, NUMBER, NUMBER, quantity]
        return ShCylinder(
            axis=c[0].value,
            cx=Quantity(num=c[1].value), cy=Quantity(num=c[2].value),
            radius=_quantity(c[3]),
        )
    raise ChaordSyntaxError(f"unknown shape {data!r}")  # pragma: no cover


def _block(inner: Tree, comments: dict[int, str], units: frozenset):
    data = inner.data
    if data in ("system_block", "physics_block", "provenance_block"):
        body = inner.children[0]
        line = getattr(getattr(body, "meta", None), "line", 0) or 0
        stmts = _body_statements(body, comments, units)
        cls = {"system_block": SystemBlock, "physics_block": PhysicsBlock,
               "provenance_block": ProvenanceBlock}[data]
        return cls(statements=stmts, comment=comments.get(line))
    if data == "species_block":
        defs = []
        for t in inner.children:
            if isinstance(t, Tree) and t.data in ("mol_def", "ion_def", "atom_def"):
                toks = [c for c in t.children if isinstance(c, Token)]
                k = {"mol_def": "molecule", "ion_def": "ion", "atom_def": "atom"}[t.data]
                name = toks[0]
                source = ref = None
                if len(toks) >= 3 and toks[1].type == "EQ":
                    source = "smiles" if toks[2].type == "SMILES" else "file"
                    ref = _unquote(toks[3].value)
                line = _tok_line(name)
                defs.append(SpecDef(
                    k=k, name=name.value, source=source, ref=ref,
                    comment=comments.get(line), line=line))
        return SpeciesBlock(defs=defs)
    if data.endswith("_region"):
        phase = data[:-len("_region")]
        toks = [c for c in inner.children if isinstance(c, Token)]
        name = toks[0]
        geo_tree = next(c for c in inner.children if isinstance(c, Tree) and c.data == "geometry")
        body = next(c for c in inner.children if isinstance(c, Tree) and c.data == "body")
        # the '{' shares the last header-token line; attach that comment
        header_line = max([_tok_line(t) for t in toks] + [getattr(geo_tree.meta, "end_line", 0)])
        return RegionBlock(
            phase=phase, name=name.value, geometry=_geometry(geo_tree),
            statements=_body_statements(body, comments, units),
            comment=comments.get(header_line))
    if data == "interface_block":
        toks = [c for c in inner.children if isinstance(c, Token) and c.type == "WORD"]
        body = next(c for c in inner.children if isinstance(c, Tree) and c.data == "body")
        return InterfaceBlock(
            a=toks[0].value, b=toks[1].value,
            statements=_body_statements(body, comments, units),
            comment=comments.get(_tok_line(toks[0])))
    if data == "residual_none":
        return ResidualBlock(none=True)
    if data == "residual_body":
        body = next(c for c in inner.children if isinstance(c, Tree) and c.data == "body")
        return ResidualBlock(none=False, statements=_body_statements(body, comments, units))
    raise ChaordSyntaxError(f"unknown block {data!r}")  # pragma: no cover
