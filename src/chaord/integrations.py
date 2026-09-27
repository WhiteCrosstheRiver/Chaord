"""Integrations: LLM tool schema, scripting layer, Laya state encoder.

The scripting layer is a thin sugar over the IR: every constructor lowers to
the same canonical text `chaord fmt` prints. The LLM tool schema exposes the
typed IR for tool calling; programs come back as JSON and are validated by the
same models. The Laya encoder packs a program DIFF into a short token budget
(320) for state tracking across an agent conversation."""
from __future__ import annotations

import difflib
import json

from .lang.api import format_program, from_json, parse_text, to_json
from .lang.errors import ChaordError
from .lang.ir import (
    At, Arrow, Direction, Eq, Family, GeoChain, InterfaceBlock, KVDefect, Name,
    Plane, Plus, Program, ProvenanceBlock, Quantity, RegionBlock, ResidualBlock,
    ShAll, ShBox, ShCylinder, ShRest, ShSlab, ShSphere, SpecDef, SpeciesBlock,
    Statement, StrVal, SystemBlock, PhysicsBlock, Tol, Wood,
)

# ---------------------------------------------------------------- LLM schema --

def program_schema() -> dict:
    """JSON schema of a Program: the tool-call contract for LLM writers."""
    return Program.model_json_schema()


def program_from_json(data) -> Program:
    """Validate an LLM-produced program (dict or JSON text) into the IR."""
    if isinstance(data, str):
        return from_json(data)
    return Program.model_validate(data)


def example_program_for_prompt() -> str:
    """A canonical example shipped to prompt writers."""
    return """chaord 0.1
dialect core + metal

system {
  cell 21.432 21.432 21.432 A
  pbc xyz
  seed 7
  state T 900 K
  conserve atoms Ni 646 Al 217
}

physics {
  backend eam
  potential "NiAl.eam.alloy"
}

crystal matrix : all {
  prototype L1_2
  composition Ni3Al
  a 3.572 A
  orient x [100] y [010] z [001]
  defect V_Ni count 1
}

residual none
"""

# ------------------------------------------------------------ scripting API --

def system(*statements, cell=None, pbc="xyz", seed=None, conserve=None) -> SystemBlock:
    stmts = []
    if cell is not None:
        stmts.append(Statement(kind="build", key="cell",
                               values=[Quantity(num=str(v)) for v in cell]))
    if pbc:
        stmts.append(Statement(kind="build", key="pbc", values=[Name(text=pbc)]))
    if seed is not None:
        stmts.append(Statement(kind="build", key="seed",
                               values=[Quantity(num=str(seed))]))
    stmts.extend(statements)
    if conserve:
        vals = []
        for species, n in conserve.items():
            vals += [Name(text=species), Quantity(num=str(n))]
        stmts.append(Statement(kind="conserve", key="atoms", values=vals))
    return SystemBlock(statements=stmts)


def physics(backend="lj", **params) -> PhysicsBlock:
    stmts = [Statement(kind="build", key="backend", values=[Name(text=backend)])]
    for k, v in params.items():
        value = StrVal(text=str(v)) if isinstance(v, str) else Quantity(num=str(v))
        stmts.append(Statement(kind="build", key=k, values=[value]))
    return PhysicsBlock(statements=stmts)


def crystal(name, geometry="all", prototype=None, lattice=None, a=None, c=None,
            composition=None, defects=()) -> RegionBlock:
    stmts = []
    if prototype:
        stmts.append(Statement(kind="build", key="prototype",
                               values=[Name(text=prototype)]))
    if lattice:
        stmts.append(Statement(kind="build", key="lattice",
                               values=[Name(text=lattice)]))
    if composition:
        stmts.append(Statement(kind="build", key="composition",
                               values=[Name(text=composition)]))
    if a is not None:
        stmts.append(Statement(kind="build", key="a",
                               values=[Quantity(num=f"{a}", unit="A")]))
    if c is not None:
        stmts.append(Statement(kind="build", key="c",
                               values=[Quantity(num=f"{c}", unit="A")]))
    stmts.extend(defects)
    return RegionBlock(phase="crystal", name=name,
                       geometry=GeoChain(parts=[ShAll()], ops=[]),
                       statements=stmts)


def defect(tok, count=1, **kv) -> Statement:
    vals = [KVDefect(text=tok), Name(text="count"), Quantity(num=str(count))]
    for k, v in kv.items():
        vals += [Name(text=k), Quantity(num=str(v))]
    return Statement(kind="build", key="defect", values=vals)


def program(*blocks, version="0.1", dialects=("core", "metal")) -> Program:
    return Program(version=version, dialects=list(dialects), blocks=list(blocks))


# ------------------------------------------------------------- Laya encoder --
#
# Patch format (v2, exact): a patch is a sequence of difflib opcodes over the
# state-carrying lines of the two texts. Every hunk starts with a control line
#
#     <op>@ <i1> <i2> <j1> <j2>
#
# where <op> is "-" when the hunk removes old lines (delete/replace) and "+"
# when it only adds them (insert), i1:i2 is the old-line range [i1, i2), and
# j1:j2 the new-line range. Control lines are followed by the removed lines
# verbatim, each prefixed "-", then the added lines, each prefixed "+". Line
# numbers are indices into the provenance-free line list, so decode is exact:
# decode(old, encode(old, new)) reproduces `new` line for line except inside
# provenance blocks (dialect / lift versions never describe state). The no-op
# is the single token "=0". Canonical program lines never start with "@", so a
# "+"/"-" line whose second character is "@" is unambiguously a control line.
#
# The 320-token Laya state budget is enforced through fits_budget().

LAYA_NOOP = "=0"
LAYA_BUDGET = 320


def _laya_lines(text: str | None) -> list[str]:
    """State-carrying lines of a program text: everything except the provenance
    block (from `provenance {` to its matching `}`) and the blank separator
    line directly above it — provenance never describes state."""
    if text is None:
        return []
    out: list[str] = []
    in_prov = False
    for line in text.splitlines():
        stripped = line.strip()
        if in_prov:
            if stripped == "}":
                in_prov = False
            continue
        if stripped.startswith("provenance {"):
            if out and not out[-1].strip():
                out.pop()
            in_prov = True
            continue
        out.append(line)
    return out


def laya_encode(old_text: str | None, new_text: str) -> str:
    """Encode a program transition as an exact line patch (Laya state token).

    Hunks carry old/new line ranges from difflib opcodes; the removed and added
    lines follow verbatim. Provenance blocks are dropped on both sides: they
    never describe state, so a transition that only touches provenance encodes
    to the no-op token."""
    a = _laya_lines(old_text)
    b = _laya_lines(new_text)
    if a == b:
        return LAYA_NOOP
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    out: list[str] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        op = "-" if i2 > i1 else "+"
        out.append(f"{op}@ {i1} {i2} {j1} {j2}")
        out.extend("-" + a[i] for i in range(i1, i2))
        out.extend("+" + b[j] for j in range(j1, j2))
    return "\n".join(out) if out else LAYA_NOOP


def _laya_hunks(patch: str) -> list[tuple[int, int, list[str], list[str]]]:
    """Parse a patch into (i1, i2, removed, added) hunks, validating shape."""
    lines = patch.splitlines()
    hunks: list[tuple[int, int, list[str], list[str]]] = []
    i = 0
    while i < len(lines):
        head = lines[i]
        if len(head) < 2 or head[0] not in "+-" or head[1] != "@":
            raise ChaordError(
                f"laya patch line {i + 1}: expected a '<op>@ i1 i2 j1 j2' header")
        try:
            i1, i2, _j1, _j2 = (int(x) for x in head[2:].split())
        except ValueError:
            raise ChaordError(
                f"laya patch line {i + 1}: header needs four integers") from None
        removed: list[str] = []
        added: list[str] = []
        i += 1
        while i < len(lines) and not (len(lines[i]) >= 2 and lines[i][1] == "@"):
            line = lines[i]
            if not line or line[0] not in "+-":
                raise ChaordError(
                    f"laya patch line {i + 1}: every diff line must start with + or -")
            (removed if line[0] == "-" else added).append(line[1:])
            i += 1
        hunks.append((i1, i2, removed, added))
    return hunks


def laya_decode(old_text: str | None, patch: str) -> str:
    """Apply a Laya patch to a previous canonical text.

    Reconstructs the new text exactly (outside provenance blocks, which the
    codec never encodes). Raises ChaordError when the patch is malformed or was
    computed against a different old text (stale state token)."""
    a = _laya_lines(old_text)
    if patch == LAYA_NOOP:
        return "\n".join(a) + "\n" if a else ""
    out: list[str] = []
    consumed = 0
    for i1, i2, removed, added in _laya_hunks(patch):
        if i1 < consumed or i2 < i1 or i2 > len(a):
            raise ChaordError(
                "laya patch does not apply: hunk ranges overlap or run past "
                "the old text (stale state token?)")
        if removed != a[i1:i2]:
            raise ChaordError(
                f"laya patch does not apply: lines {i1}..{i2} of the old text "
                "differ from the hunk (stale state token?)")
        out.extend(a[consumed:i1])
        out.extend(added)
        consumed = i2
    out.extend(a[consumed:])
    return "\n".join(out) + "\n" if out else ""


def laya_tokens(patch: str) -> int:
    """Approximate token count of a patch (whitespace-split, conservative)."""
    return max(len(patch.split()), 1)


def fits_budget(patch: str, budget: int = LAYA_BUDGET) -> bool:
    """True when a Laya patch fits the state-token budget (default 320)."""
    return laya_tokens(patch) <= budget
