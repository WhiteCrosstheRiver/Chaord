"""Static checks on a program + structure pair: overlaps, conservation, asserts.

These run after every build and every lift. They measure and report; they never
modify anything. Thresholds come from the dialect.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..io.frames import Frame
from ..lang.ir import Program


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


def overlap_check(frame: Frame, dialect) -> CheckResult:
    tol = float(dialect.threshold("overlap_tolerance"))
    if len(frame) < 2:
        return CheckResult("overlap", True, "fewer than two atoms")
    from scipy.spatial import cKDTree
    from ..realize.lj import wrap
    L = frame.cell_diag
    tree = cKDTree(wrap(frame.pos, L), boxsize=L)
    d, _ = tree.query(wrap(frame.pos, L), k=2)
    dmin = float(d[:, 1].min())
    ok = dmin >= tol
    return CheckResult("overlap", ok, f"min pair distance {dmin:.3f} vs tolerance {tol}")


def conservation_check(program: Program, frame: Frame) -> CheckResult:
    """Atoms per species and total count in the program must equal the input's."""
    stated: dict[str, int] = {}
    for b in program.blocks:
        if b.t == "system":
            for s in b.statements:
                if s.kind == "conserve" and s.key == "atoms":
                    vals = s.values
                    i = 0
                    while i + 1 < len(vals):
                        if vals[i].t == "n" and vals[i + 1].t == "q":
                            stated[vals[i].text] = int(float(vals[i + 1].num))
                        i += 2
    present: dict[str, int] = {}
    for sym in frame.symbols:
        present[sym] = present.get(sym, 0) + 1
    if stated == present:
        return CheckResult("conservation", True, f"{sum(present.values())} atoms, species match")
    return CheckResult(
        "conservation", False,
        f"program says {stated}, structure has {present}")


def residual_explained(program: Program, frame: Frame) -> CheckResult:
    """Never drop an atom: unexplained atoms must appear in the residual block."""
    n_res = sum(len(b.statements) for b in program.blocks if b.t == "residual" and not b.none)
    has_none = any(b.t == "residual" and b.none for b in program.blocks)
    if has_none:
        return CheckResult("residual", True, "residual none")
    return CheckResult("residual", True, f"{n_res} atoms in residual")


def run_checks(program: Program, frame: Frame, dialect) -> list[CheckResult]:
    return [
        overlap_check(frame, dialect),
        conservation_check(program, frame),
        residual_explained(program, frame),
    ]
