"""IR models for Chaord programs (Pydantic v2).

The IR is the single representation shared by build and lift. Every value keeps
its source token text so that text -> IR -> text is lossless and fmt is idempotent;
number normalisation for lifted programs happens when lift constructs the tokens,
not in the printer. Source line numbers are tracked for error reporting but are
excluded from JSON serialisation.
"""
from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field

# ---------------------------------------------------------------- values ----

Line = Field(default=0, exclude=True)


class Quantity(BaseModel):
    t: Literal["q"] = "q"
    num: str
    unit: Optional[str] = None
    line: int = Line


class RangeVal(BaseModel):
    t: Literal["r"] = "r"
    lo: Quantity
    hi: Quantity
    line: int = Line


class Tol(BaseModel):
    """An inline tolerance `+- value [unit]`; position within the statement is kept."""
    t: Literal["tol"] = "tol"
    value: Quantity
    line: int = Line


class Name(BaseModel):
    t: Literal["n"] = "n"
    text: str
    line: int = Line


class StrVal(BaseModel):
    t: Literal["s"] = "s"
    text: str
    line: int = Line


class Direction(BaseModel):
    t: Literal["dir"] = "dir"
    text: str
    line: int = Line


class Family(BaseModel):
    t: Literal["fam"] = "fam"
    text: str
    line: int = Line


class Plane(BaseModel):
    t: Literal["pl"] = "pl"
    text: str
    line: int = Line


class Wood(BaseModel):
    t: Literal["wood"] = "wood"
    text: str
    line: int = Line


class KVDefect(BaseModel):
    """A Kröger–Vink style token such as V_O^.. or Al_Ni."""
    t: Literal["kv"] = "kv"
    text: str
    line: int = Line


class Arrow(BaseModel):
    t: Literal["arrow"] = "arrow"
    line: int = Line


class At(BaseModel):
    t: Literal["at"] = "at"
    line: int = Line


class Plus(BaseModel):
    t: Literal["plus"] = "plus"
    line: int = Line


class Eq(BaseModel):
    t: Literal["eq"] = "eq"
    line: int = Line


Value = Annotated[
    Union[
        Quantity, RangeVal, Tol, Name, StrVal, Direction, Family, Plane, Wood,
        KVDefect, Arrow, At, Plus, Eq,
    ],
    Field(discriminator="t"),
]


def value_line(v: Value) -> int:
    return getattr(v, "line", 0)


def ir_equal(a: BaseModel, b: BaseModel) -> bool:
    """Structural equality ignoring source line numbers."""
    return _strip(a) == _strip(b)


def _strip(m):
    if isinstance(m, BaseModel):
        return (type(m), tuple((k, _strip(v)) for k, v in sorted(m.__dict__.items()) if k != "line"))
    if isinstance(m, list):
        return [_strip(x) for x in m]
    return m


# ---------------------------------------------------------------- shapes ----

Axis = Literal["x", "y", "z"]


class ShAll(BaseModel):
    t: Literal["all"] = "all"


class ShRest(BaseModel):
    t: Literal["rest"] = "rest"


class ShSlab(BaseModel):
    t: Literal["slab"] = "slab"
    axis: Axis
    rng: RangeVal


class ShBox(BaseModel):
    t: Literal["box"] = "box"
    xs: RangeVal
    ys: RangeVal
    zs: RangeVal


class ShSphere(BaseModel):
    t: Literal["sphere"] = "sphere"
    cx: Quantity
    cy: Quantity
    cz: Quantity
    radius: Quantity


class ShCylinder(BaseModel):
    t: Literal["cylinder"] = "cylinder"
    axis: Axis
    cx: Quantity
    cy: Quantity
    radius: Quantity


Shape = Annotated[
    Union[ShAll, ShRest, ShSlab, ShBox, ShSphere, ShCylinder],
    Field(discriminator="t"),
]


class GeoChain(BaseModel):
    """A geometry expression: shapes joined by and/or/minus (left to right)."""
    t: Literal["chain"] = "chain"
    parts: list[Shape]
    ops: list[Literal["and", "or", "minus"]]


# ------------------------------------------------------------- statements ----

Kind = Literal["build", "state", "constrain", "assert", "history", "conserve"]
Phase = Literal["crystal", "amorphous", "liquid", "gas", "fluid", "cluster", "vacuum"]


class Statement(BaseModel):
    kind: Kind = "build"
    key: str
    values: list[Value] = Field(default_factory=list)
    comment: Optional[str] = None
    line: int = Line


class SpecDef(BaseModel):
    k: Literal["molecule", "ion", "atom"]
    name: str
    source: Optional[Literal["smiles", "file"]] = None
    ref: Optional[str] = None
    comment: Optional[str] = None
    line: int = Line


# ----------------------------------------------------------------- blocks ----

class SystemBlock(BaseModel):
    t: Literal["system"] = "system"
    statements: list[Statement] = Field(default_factory=list)
    comment: Optional[str] = None


class PhysicsBlock(BaseModel):
    t: Literal["physics"] = "physics"
    statements: list[Statement] = Field(default_factory=list)
    comment: Optional[str] = None


class ProvenanceBlock(BaseModel):
    t: Literal["provenance"] = "provenance"
    statements: list[Statement] = Field(default_factory=list)
    comment: Optional[str] = None


class SpeciesBlock(BaseModel):
    t: Literal["species"] = "species"
    defs: list[SpecDef] = Field(default_factory=list)
    comment: Optional[str] = None


class RegionBlock(BaseModel):
    t: Literal["region"] = "region"
    phase: Phase
    name: str
    geometry: GeoChain
    statements: list[Statement] = Field(default_factory=list)
    comment: Optional[str] = None


class InterfaceBlock(BaseModel):
    t: Literal["interface"] = "interface"
    a: str
    b: str
    statements: list[Statement] = Field(default_factory=list)
    comment: Optional[str] = None


class ResidualBlock(BaseModel):
    t: Literal["residual"] = "residual"
    none: bool = True
    statements: list[Statement] = Field(default_factory=list)
    comment: Optional[str] = None


Block = Annotated[
    Union[
        SystemBlock, PhysicsBlock, ProvenanceBlock, SpeciesBlock,
        RegionBlock, InterfaceBlock, ResidualBlock,
    ],
    Field(discriminator="t"),
]


class Program(BaseModel):
    version: str
    dialects: list[str] = Field(default_factory=list)
    blocks: list[Block] = Field(default_factory=list)
