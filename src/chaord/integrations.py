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
        stmts.append(Statement(kind="build", key=k,
                               values=[Quantity(num=str(v))]))
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

# compact field codes for the diff codec (stable vocabulary)
_CODES = {
    "chaord": "v", "dialect": "d", "system": "S", "physics": "P",
    "provenance": "N", "species": "Y", "interface": "I", "residual": "R",
    "crystal": "C", "amorphous": "A", "liquid": "L", "gas": "G",
    "fluid": "F", "cluster": "K", "vacuum": "V",
}


def laya_encode(old_text: str | None, new_text: str) -> str:
    """Encode a program transition as a compact line diff (Laya state token).

    Line-level unified-diff hunks with short op codes: `+line`, `-line`;
    context-free. Provenance lines are dropped: they never describe state."""
    def lines(text):
        if text is None:
            return []
        return [l for l in text.splitlines()
                if l.strip() and "provenance" not in l and not l.strip().startswith("dialects ")
                and not l.strip().startswith("lift_version")]

    a, b = lines(old_text), lines(new_text)
    if a == b:
        return "=0"
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    out = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("delete", "replace"):
            out.extend("-" + a[i] for i in range(i1, i2))
        if tag in ("insert", "replace"):
            out.extend("+" + b[j] for j in range(j1, j2))
    return "\n".join(out) if out else "=0"


def laya_decode(old_text: str | None, patch: str) -> str:
    """Apply a Laya patch to a previous canonical text."""
    if patch == "=0":
        return old_text or ""
    a = [l for l in (old_text or "").splitlines()
         if l.strip() and "provenance" not in l]
    out = []
    for line in patch.splitlines():
        op, content = line[0], line[1:]
        if op == "-":
            if content in a:
                a.remove(content)
        elif op == "+":
            out.append(content)
    return "\n".join(a + out) + "\n"


def laya_tokens(patch: str) -> int:
    """Approximate token count of a patch (whitespace-split, conservative)."""
    return max(len(patch.split()), 1)
