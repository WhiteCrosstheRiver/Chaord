"""Structural checker for the Chaord v0.1 grammar sketch (spec/grammar.ebnf).
It validates block structure, region headers, geometry and statement shape.
It does NOT know the vocabulary (that is the dialects' job). Usage: python sketch_check.py FILE..."""
import re, sys

TOKENS = [
    ("COMMENT", r"#[^\n]*"), ("NL", r"\n"), ("WS", r"[ \t\r]+"),
    ("STRING", r'"(?:[^"\\]|\\.)*"'),
    ("DIRECTION", r"\[\s*-?\d(?:\s*-?\d){2,3}\s*\]"),
    ("FAMILY", r"<\s*-?\d(?:\s*-?\d){2,3}\s*>"),
    ("WOOD", r"[pc]?\(\s*\w+\s*x\s*\w+\s*\)(?:R\d+)?"),
    ("PLANE", r"\(\s*-?\d(?:\s*-?\d){2,3}\s*\)"),
    ("KV", r"(?:V|[A-Z][a-z]?)_(?:[A-Z][a-z]?|i)(?:\^(?:x|\.+|'+))?(?![A-Za-z0-9_])"),
    ("ARROW", r"->"), ("RANGE", r"\.\."), ("PM", r"\+-"),
    ("NUMBER", r"[+-]?(?:\d+/\d+|(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)(?![A-Za-z_])"),
    ("SYM", r"[{}:|@=;+,]"),
    ("WORD", r"[A-Za-z_%][A-Za-z0-9_+\-/%^.]*"),
]
LEX = re.compile("|".join(f"(?P<{n}>{p})" for n, p in TOKENS))
PHASES = {"crystal", "amorphous", "liquid", "gas", "fluid", "cluster", "vacuum"}
KINDS = {"state", "constrain", "assert", "history", "conserve"}
VALUE = {"NUMBER", "WORD", "STRING", "DIRECTION", "FAMILY", "PLANE", "WOOD", "KV", "RANGE", "PM", "ARROW"}

def lex(src):
    pos, line, out = 0, 1, []
    while pos < len(src):
        m = LEX.match(src, pos)
        if not m: raise SyntaxError(f"line {line}: cannot read {src[pos:pos+20]!r}")
        kind, text = m.lastgroup, m.group()
        if kind == "NL": out.append(("NL", "\n", line)); line += 1
        elif kind not in ("WS", "COMMENT"): out.append((kind, text, line))
        pos = m.end()
    out.append(("EOF", "", line)); return out

class P:
    def __init__(s, toks): s.t, s.i = toks, 0
    def peek(s, k=0): return s.t[s.i + k]
    def take(s, kind=None, text=None):
        tk = s.peek()
        if (kind and tk[0] != kind) or (text and tk[1] != text):
            raise SyntaxError(f"line {tk[2]}: expected {text or kind}, got {tk[1]!r}")
        s.i += 1; return tk
    def nls(s):
        while s.peek()[0] == "NL": s.i += 1
    def program(s):
        s.nls(); s.take("WORD", "chaord"); s.take("NUMBER"); s.take("NL"); s.nls()
        if s.peek()[1] == "dialect":
            s.i += 1; s.take("WORD")
            while s.peek()[1] == "+": s.i += 1; s.take("WORD")
            s.take("NL")
        blocks = []
        s.nls()
        while s.peek()[0] != "EOF": blocks.append(s.block()); s.nls()
        return blocks
    def block(s):
        w = s.peek()[1]
        if w in ("system", "physics", "provenance"): s.i += 1; return (w, s.body())
        if w == "species":
            s.i += 1; s.take("SYM", "{"); n = 0
            while True:
                s.nls()
                if s.peek()[1] == "}": s.i += 1; return ("species", n)
                s.take("WORD"); 
                if s.peek()[0] not in ("WORD", "KV"): raise SyntaxError(f"line {s.peek()[2]}: species name expected")
                s.i += 1
                if s.peek()[1] == "=":
                    s.i += 1
                    if s.peek()[1] in ("smiles", "file"): s.i += 1
                    s.take("STRING")
                s.take("NL"); n += 1
        if w in PHASES:
            s.i += 1; name = s.take("WORD")[1]; s.take("SYM", ":"); s.geometry()
            return (w, name, s.body())
        if w == "interface":
            s.i += 1; a = s.take("WORD")[1]; s.take("SYM", "|"); b = s.take("WORD")[1]
            return ("interface", a, b, s.body())
        if w == "residual":
            s.i += 1
            if s.peek()[1] == "none": s.i += 1; return ("residual", 0)
            return ("residual", s.body())
        raise SyntaxError(f"line {s.peek()[2]}: unknown block {w!r}")
    def qty(s):
        s.take("NUMBER")
        if s.peek()[0] == "WORD" and s.peek()[1] not in ("and", "or", "minus", "radius", "center"): s.i += 1
    def rng(s): s.qty(); s.take("RANGE"); s.qty()
    def geometry(s):
        s.shape()
        while s.peek()[1] in ("and", "or", "minus"): s.i += 1; s.shape()
    def shape(s):
        w = s.take("WORD")[1]
        if w in ("all", "rest"): return
        if w == "slab": s.take("WORD"); s.rng(); return
        if w == "box": s.rng(); s.rng(); s.rng(); return
        if w == "sphere":
            s.take("WORD", "center"); [s.take("NUMBER") for _ in range(3)]; s.take("WORD", "radius"); s.qty(); return
        if w == "cylinder":
            s.take("WORD", "axis"); s.take("WORD"); s.take("WORD", "center"); s.take("NUMBER"); s.take("NUMBER")
            s.take("WORD", "radius"); s.qty(); return
        raise SyntaxError(f"line {s.peek()[2]}: unknown shape {w!r}")
    def body(s):
        s.take("SYM", "{"); stmts = []
        while True:
            s.nls()
            if s.peek()[1] == "}": s.i += 1; return stmts
            stmts.append(s.statement())
    def statement(s):
        kind = "build"
        if s.peek()[1] in KINDS: kind = s.take("WORD")[1]
        key = s.peek()
        if key[0] not in ("WORD", "KV"): raise SyntaxError(f"line {key[2]}: statement key expected, got {key[1]!r}")
        s.i += 1; vals = []
        while s.peek()[0] in VALUE or s.peek()[1] in ("@", "+", "=", ","):
            tk = s.take()
            if tk[0] == "PM" and s.peek()[0] != "NUMBER": raise SyntaxError(f"line {tk[2]}: '+-' needs a number")
            if tk[0] == "RANGE" and (not vals or s.peek()[0] != "NUMBER"): raise SyntaxError(f"line {tk[2]}: bad range")
            vals.append(tk[1])
        if s.peek()[1] == ";": s.i += 1
        elif s.peek()[0] != "NL" and s.peek()[1] != "}":
            raise SyntaxError(f"line {s.peek()[2]}: unexpected {s.peek()[1]!r} in statement {key[1]!r}")
        return (kind, key[1], vals)

def check(path):
    blocks = P(lex(open(path).read())).program()
    n = sum(len(b[-1]) for b in blocks if isinstance(b[-1], list))
    kinds = sorted({st[0] for b in blocks if isinstance(b[-1], list) for st in b[-1]})
    return len(blocks), n, kinds

if __name__ == "__main__":
    bad = 0
    for f in sys.argv[1:]:
        try:
            nb, ns, kinds = check(f); print(f"OK   {f.split('/')[-1]:<28} {nb} blocks, {ns} statements, kinds: {', '.join(kinds)}")
        except SyntaxError as e:
            bad += 1; print(f"FAIL {f.split('/')[-1]:<28} {e}")
    sys.exit(1 if bad else 0)
