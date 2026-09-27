#!/usr/bin/env python
"""50-prompt LLM validation suite for Chaord program synthesis (M8).

Every case is a natural-language prompt plus a machine-checkable expectation
of the program an LLM should write for it: the answer must parse into the IR,
be byte-stable under `chaord fmt`, contain the key statements the prompt asks
for, and — where a builder exists — compile with `chaord build --no-physics`
(`build_program(..., physics=False)`) while conserving atoms. Families cover
the seven spec example systems plus crystal / defect / solid-solution /
liquid / gas / surface / amorphous / bicrystal variants.

Writers (--writer):
- deterministic  the built-in correct program stands in for the LLM answer;
                 this validates the checking pipeline itself and must pass
                 100% (the M8 gate asks for >= 90% of 50 prompts).
- file           reads a JSON file mapping {"prompt text" | "prompt id":
                 "program text"} produced by a real LLM run and checks every
                 answer it finds (missing prompts count as failures).

No network API is called anywhere: file mode is the only bridge to an
external model. A case passes on the FIRST compile attempt — no repair,
no retry; the suite reports the first-compile pass rate.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

# the tool lives in tools/; make `import chaord` work next to the source tree
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from chaord.build import build_program  # noqa: E402
from chaord.check.statics import conservation_check  # noqa: E402
from chaord.dialects import dialect_from_program  # noqa: E402
from chaord.integrations import (  # noqa: E402
    crystal, defect, format_program, physics, program, program_from_json,
    system, to_json,
)
from chaord.lang.ir import ir_equal  # noqa: E402
from chaord.lang.parser import parse_text  # noqa: E402

PASS_RATE_TARGET = 0.90        # PLAN M8: LLM programs compile first-try >= 90%
NO_PHYSICS = False             # every build runs with --no-physics semantics


# ------------------------------------------------------------------ types ----

@dataclass
class Case:
    """One prompt with its machine-checkable expectation of the answer."""
    pid: str
    family: str
    prompt: str
    points: tuple[str, ...]                  # what a correct program must say
    writer: Callable[[], str] | str          # deterministic stand-in answer
    statements: tuple[str, ...] = ()         # fragments required in canon text
    build: bool = False                      # build_program(physics=False)
    conserve: bool = False                   # conservation_check vs the frame
    atoms: Optional[dict] = None             # expected element census
    schema: bool = False                     # LLM JSON schema round trip
    custom: Optional[Callable] = None        # custom(builder) check, raises


@dataclass
class CaseResult:
    pid: str
    family: str
    ok: bool
    stages: str = ""
    error: str = ""


@dataclass
class SuiteResult:
    results: list = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.ok)

    @property
    def rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def failures(self) -> list:
        return [r for r in self.results if not r.ok]

    def by_family(self) -> dict:
        out: dict[str, list[int]] = {}
        for r in self.results:
            p, t = out.get(r.family, (0, 0))
            out[r.family] = (p + (1 if r.ok else 0), t + 1)
        return out


# ------------------------------------------------------------ check engine ---

def _seed_of(program) -> int:
    for b in program.blocks:
        if b.t == "system":
            for s in b.statements:
                if s.key == "seed":
                    return int(float(s.values[0].num))
    return 0


def run_case(case: Case, program_text: Optional[str]) -> CaseResult:
    """First-compile check of one answer; any stage failure fails the case."""
    result = CaseResult(case.pid, case.family, ok=False)
    if program_text is None:
        result.error = "no answer for this prompt (missing from the file)"
        return result
    stages = []
    stage = "parse"
    try:
        prog = parse_text(program_text)
        canonical = format_program(prog)
        # fmt is a fixed point: canonical text re-parses to the same text
        if format_program(parse_text(canonical)) != canonical:
            raise ValueError("canonical text is not a fmt fixed point")
        stages.append("parse")

        stage = "statements"
        if case.statements:
            missing = [frag for frag in case.statements if frag not in canonical]
            if missing:
                raise ValueError(f"missing key statements: {missing}")
        stages.append("statements")

        if case.schema:
            stage = "schema"
            back = program_from_json(to_json(prog))
            if not ir_equal(back, prog):
                raise ValueError("JSON schema round trip changed the program")
            stages.append("schema")

        frame = None
        if case.build or case.conserve or case.atoms:
            stage = "build"
            dialect = dialect_from_program(prog)
            rng = np.random.default_rng(_seed_of(prog))
            frame = build_program(prog, dialect, rng, physics=NO_PHYSICS)
            stages.append("build(no-physics)")

        if case.conserve:
            stage = "conserve"
            chk = conservation_check(prog, frame)
            if not chk.passed:
                raise ValueError(f"conservation failed: {chk.detail}")
            stages.append("conserve")

        if case.atoms is not None:
            stage = "atoms"
            census = dict(Counter(frame.symbols))
            if census != case.atoms:
                raise ValueError(f"atom census {census} != expected {case.atoms}")
            stages.append("atoms")

        if case.custom is not None:
            stage = "custom"
            case.custom(case, prog)
            stages.append("custom")
    except Exception as exc:  # noqa: BLE001 — report every failure kind
        result.error = f"[{stage}] {type(exc).__name__}: {exc}"
        result.stages = "+".join(stages)
        return result
    result.stages = "+".join(stages)
    result.ok = True
    return result


def run_suite(writer: str = "deterministic", file: Optional[str] = None,
              cases=None) -> SuiteResult:
    """Run the 50 prompts; `writer` is 'deterministic' or 'file' (JSON path)."""
    if writer not in ("deterministic", "file"):
        raise ValueError(f"unknown writer {writer!r}")
    answers: dict = {}
    if writer == "file":
        if not file:
            raise ValueError("--file is required with --writer file")
        answers = json.loads(Path(file).read_text(encoding="utf-8"))
    out = SuiteResult()
    for case in cases if cases is not None else CASES:
        text: Optional[str]
        if writer == "deterministic":
            text = case.writer() if callable(case.writer) else case.writer
        else:
            text = answers.get(case.prompt, answers.get(case.pid))
        out.results.append(run_case(case, text))
    return out


# ------------------------------------------------- custom checks (extended) --

def _grain_boundary_check(case, prog):
    """CSL bicrystals compile through the M6 grain-boundary builder."""
    from chaord.build.extended import build_grain_boundary
    region = next(b for b in prog.blocks if b.t == "region")
    dialect = dialect_from_program(prog)
    sigma = int(float(next(
        v for s in region.statements if s.key == "grain_boundary"
        for v in s.values if v.t == "q").num))
    frame = build_grain_boundary(region, {}, dialect, np.random.default_rng(5))
    census = dict(Counter(frame.symbols))
    expected = {"Cu": 1536}          # fcc 8x8x6 conventional cells
    if census != expected:
        raise ValueError(f"bicrystal census {census} != expected {expected}")
    if sigma not in (5, 13, 17):
        raise ValueError(f"unknown CSL Sigma {sigma}")


def _dislocation_check(case, prog):
    """Edge dislocations compile through the M6 Volterra builder."""
    from chaord.build.extended import build_dislocation
    region = next(b for b in prog.blocks if b.t == "region")
    dialect = dialect_from_program(prog)
    frame = build_dislocation(region, {}, dialect, np.random.default_rng(5))
    census = dict(Counter(frame.symbols))
    expected = {"Cu": 576}           # fcc 6x6x4 conventional cells
    if census != expected:
        raise ValueError(f"dislocation census {census} != expected {expected}")


# ------------------------------------------------------ scripted writers -----

def _w_fcc_cu() -> str:
    return format_program(program(
        system(cell=(14.46, 14.46, 14.46), seed=11, conserve={"Cu": 256}),
        physics(backend="eam"),
        crystal("bulk", lattice="fcc", a=3.615),
    ))


def _w_bcc_fe() -> str:
    return format_program(program(
        system(cell=(11.48, 11.48, 11.48), seed=3, conserve={"Fe": 128}),
        physics(backend="eam"),
        crystal("bulk", lattice="bcc", a=2.87),
    ))


def _w_ni3al_vacancy() -> str:
    return format_program(program(
        system(seed=7, conserve={"Ni": 2, "Al": 1}),
        physics(backend="eam"),
        crystal("matrix", prototype="L1_2", composition="Ni3Al", a=3.572,
                defects=[defect("V_Ni", count=1)]),
    ))


def _w_ni3al_antisite() -> str:
    return format_program(program(
        system(seed=4, conserve={"Ni": 2, "Al": 2}),
        physics(backend="eam"),
        crystal("matrix", prototype="L1_2", composition="Ni3Al", a=3.572,
                defects=[defect("Al_Ni", count=1)]),
    ))


def _w_cu_interstitials() -> str:
    return format_program(program(
        system(seed=6, conserve={"Cu": 6}),
        physics(backend="eam"),
        crystal("bulk", lattice="fcc", a=3.615,
                defects=[defect("Cu_i", count=2)]),
    ))


# ------------------------------------------------------------- the 50 cases --

def _t(text: str) -> str:
    """Dedent a canonical program text (keeps the case table readable)."""
    lines = text.strip("\n").splitlines()
    return "\n".join(ln[4:] if ln.startswith("    ") else ln for ln in lines) + "\n"


CRYSTAL_CASES = [
    Case(
        pid="P01", family="crystal",
        prompt="Build a 4x4x4 supercell of fcc copper, lattice constant "
               "a = 3.615 A, EAM backend with the Cu potential, random seed "
               "11. All 256 Cu atoms must be conserved.",
        points=("fcc lattice, a = 3.615 A", "4x4x4 tiling = 256 atoms",
                "EAM physics backend"),
        writer=_w_fcc_cu,
        statements=("lattice fcc", "a 3.615 A", "conserve atoms Cu 256"),
        build=True, conserve=True),
    Case(
        pid="P02", family="crystal",
        prompt="A 4x4x4 bcc iron supercell, a = 2.87 A, EAM backend, seed 3. "
               "Conserve all 128 Fe atoms.",
        points=("bcc lattice, a = 2.87 A", "128 Fe atoms conserved"),
        writer=_w_bcc_fe,
        statements=("lattice bcc", "conserve atoms Fe 128"),
        build=True, conserve=True),
    Case(
        pid="P03", family="crystal",
        prompt="Simple-cubic polonium, a = 2.76 A, four conventional cells "
               "along each axis, seed 5. 64 Po atoms in total.",
        points=("sc lattice, a = 2.76 A", "64 Po atoms"),
        writer=_t("""
            chaord 0.1
            dialect core + metal

            system {
              cell 11.04 11.04 11.04 A
              pbc xyz
              seed 5
              conserve atoms Po 64
            }

            physics {
              backend eam
            }

            crystal bulk : all {
              lattice sc
              a 2.76 A
            }

            residual none
            """),
        statements=("lattice sc", "conserve atoms Po 64"),
        build=True, conserve=True),
    Case(
        pid="P04", family="crystal",
        prompt="Diamond-structure silicon, a = 5.431 A, tiled 4x4x4 "
               "conventional cells, seed 2. Machine-learning backend "
               "mace-mp-0. 512 Si atoms.",
        points=("diamond lattice, a = 5.431 A", "512 Si atoms"),
        writer=_t("""
            chaord 0.1
            dialect core + metal

            system {
              cell 21.724 21.724 21.724 A
              pbc xyz
              seed 2
              conserve atoms Si 512
            }

            physics {
              backend mlp
              model "mace-mp-0"
            }

            crystal bulk : all {
              lattice diamond
              a 5.431 A
            }

            residual none
            """),
        statements=('lattice diamond', 'model "mace-mp-0"',
                    "conserve atoms Si 512"),
        build=True, conserve=True),
    Case(
        pid="P05", family="crystal",
        prompt="One conventional hcp magnesium cell, a = 3.209 A and "
               "c = 5.211 A, EAM backend, seed 8. Two Mg atoms.",
        points=("hcp lattice with a and c", "2 Mg atoms"),
        writer=_t("""
            chaord 0.1
            dialect core + metal

            system {
              pbc xyz
              seed 8
              conserve atoms Mg 2
            }

            physics {
              backend eam
            }

            crystal bulk : all {
              lattice hcp
              a 3.209 A
              c 5.211 A
            }

            residual none
            """),
        statements=("lattice hcp", "a 3.209 A", "c 5.211 A",
                    "conserve atoms Mg 2"),
        build=True, conserve=True),
    Case(
        pid="P06", family="crystal",
        prompt="Rock-salt NaCl, a = 4.212 A, one conventional cell, "
               "classical backend, seed 1. 4 Na and 4 Cl.",
        points=("rocksalt prototype NaCl, a = 4.212 A", "Na 4 Cl 4"),
        writer=_t("""
            chaord 0.1
            dialect core + metal

            system {
              pbc xyz
              seed 1
              conserve atoms Cl 4 Na 4
            }

            physics {
              backend classical
            }

            crystal bulk : all {
              prototype rocksalt
              composition NaCl
              a 4.212 A
            }

            residual none
            """),
        statements=("prototype rocksalt", "composition NaCl",
                    "conserve atoms Cl 4 Na 4"),
        build=True, conserve=True),
    Case(
        pid="P07", family="crystal",
        prompt="Fluorite-structure uranium dioxide, a = 5.47 A, one "
               "conventional cell, seed 9: 4 U and 8 O atoms.",
        points=("fluorite prototype UO2, a = 5.47 A", "U 4 O 8"),
        writer=_t("""
            chaord 0.1
            dialect core + metal

            system {
              pbc xyz
              seed 9
              conserve atoms O 8 U 4
            }

            physics {
              backend classical
            }

            crystal bulk : all {
              prototype fluorite
              composition UO2
              a 5.47 A
            }

            residual none
            """),
        statements=("prototype fluorite", "composition UO2",
                    "conserve atoms O 8 U 4"),
        build=True, conserve=True),
    Case(
        pid="P08", family="crystal",
        prompt="Cubic perovskite SrTiO3, a = 3.905 A, one unit cell, "
               "seed 4: 1 Sr, 1 Ti, 3 O.",
        points=("perovskite prototype SrTiO3", "Sr 1 Ti 1 O 3"),
        writer=_t("""
            chaord 0.1
            dialect core + metal

            system {
              pbc xyz
              seed 4
              conserve atoms O 3 Sr 1 Ti 1
            }

            physics {
              backend classical
            }

            crystal bulk : all {
              prototype perovskite
              composition SrTiO3
              a 3.905 A
            }

            residual none
            """),
        statements=("prototype perovskite", "composition SrTiO3",
                    "conserve atoms O 3 Sr 1 Ti 1"),
        build=True, conserve=True),
]

DEFECT_CASES = [
    Case(
        pid="P09", family="defects",
        prompt="One L1_2 Ni3Al conventional cell (a = 3.572 A) containing a "
               "single nickel vacancy V_Ni, EAM backend, seed 7. Remaining "
               "atoms: Ni 2, Al 1.",
        points=("L1_2 prototype Ni3Al", "defect V_Ni count 1",
                "Ni 2 Al 1 after removal"),
        writer=_w_ni3al_vacancy,
        statements=("prototype L1_2", "defect V_Ni count 1",
                    "conserve atoms Ni 2 Al 1"),
        build=True, conserve=True),
    Case(
        pid="P10", family="defects",
        prompt="L1_2 Ni3Al (a = 3.572 A) with one antisite defect: aluminium "
               "sitting on a nickel site (Al_Ni), seed 4. Final census "
               "Ni 2 Al 2.",
        points=("antisite Al_Ni count 1", "Ni 2 Al 2"),
        writer=_w_ni3al_antisite,
        statements=("defect Al_Ni count 1", "conserve atoms Ni 2 Al 2"),
        build=True, conserve=True),
    Case(
        pid="P11", family="defects",
        prompt="A single fcc copper cell (a = 3.615 A) with two copper "
               "self-interstitials (Cu_i), seed 6. 6 Cu atoms in total.",
        points=("interstitial Cu_i count 2", "6 Cu atoms"),
        writer=_w_cu_interstitials,
        statements=("defect Cu_i count 2", "conserve atoms Cu 6"),
        build=True, conserve=True),
    Case(
        pid="P12", family="defects",
        prompt="One fcc copper cell (a = 3.615 A) containing a single "
               "Frenkel pair (one atom kicked off its site into an "
               "interstitial position), seed 2. Atom count stays 4.",
        points=("frenkel_pair count 1", "4 Cu atoms (count preserved)"),
        writer=_t("""
            chaord 0.1
            dialect core + metal

            system {
              pbc xyz
              seed 2
              conserve atoms Cu 4
            }

            physics {
              backend eam
            }

            crystal bulk : all {
              lattice fcc
              a 3.615 A
              defect frenkel_pair count 1
            }

            residual none
            """),
        statements=("defect frenkel_pair count 1", "conserve atoms Cu 4"),
        build=True, conserve=True),
    Case(
        pid="P13", family="defects",
        prompt="Rutile TiO2 (a = 4.594 A, c = 2.959 A), one cell, with two "
               "doubly-charged oxygen vacancies V_O^.. on the surface layer, "
               "seed 3. Ti 2 O 2 remain.",
        points=("rutile prototype with a and c", "charged vacancy V_O^..",
                "Ti 2 O 2"),
        writer=_t("""
            chaord 0.1
            dialect core + metal + ionic

            system {
              pbc xyz
              seed 3
              conserve atoms O 2 Ti 2
            }

            physics {
              backend classical
            }

            crystal bulk : all {
              prototype rutile
              composition TiO2
              a 4.594 A
              c 2.959 A
              defect V_O^.. count 2 layer surface
            }

            residual none
            """),
        statements=("prototype rutile", "defect V_O^.. count 2 layer surface",
                    "conserve atoms O 2 Ti 2"),
        build=True, conserve=True),
    Case(
        pid="P14", family="defects",
        prompt="L1_2 Ni3Al (a = 3.572 A) with a nickel vacancy and one "
               "aluminium antisite on a nickel site, seed 5. Final census "
               "Ni 1 Al 2.",
        points=("V_Ni and Al_Ni together", "Ni 1 Al 2"),
        writer=_t("""
            chaord 0.1
            dialect core + metal

            system {
              pbc xyz
              seed 5
              conserve atoms Al 2 Ni 1
            }

            physics {
              backend eam
            }

            crystal matrix : all {
              prototype L1_2
              composition Ni3Al
              a 3.572 A
              defect V_Ni count 1
              defect Al_Ni count 1
            }

            residual none
            """),
        statements=("defect V_Ni count 1", "defect Al_Ni count 1",
                    "conserve atoms Al 2 Ni 1"),
        build=True, conserve=True),
    Case(
        pid="P15", family="defects",
        prompt="A 2x2x2 fcc copper supercell (a = 3.615 A) with three random "
               "vacancies V_Cu, seed 10. 29 Cu atoms remain.",
        points=("2x2x2 = 32 sites minus 3 vacancies", "29 Cu atoms"),
        writer=_t("""
            chaord 0.1
            dialect core + metal

            system {
              cell 7.23 7.23 7.23 A
              pbc xyz
              seed 10
              conserve atoms Cu 29
            }

            physics {
              backend eam
            }

            crystal bulk : all {
              lattice fcc
              a 3.615 A
              defect V_Cu count 3
            }

            residual none
            """),
        statements=("defect V_Cu count 3", "conserve atoms Cu 29"),
        build=True, conserve=True),
    Case(
        pid="P16", family="defects",
        prompt="Rock-salt NaCl (a = 4.212 A), one cell, containing one "
               "Schottky defect: a Na vacancy and a Cl vacancy pair, seed 6. "
               "Na 3 Cl 3 remain.",
        points=("Schottky pair V_Na + V_Cl", "Na 3 Cl 3"),
        writer=_t("""
            chaord 0.1
            dialect core + metal + ionic

            system {
              pbc xyz
              seed 6
              conserve atoms Cl 3 Na 3
            }

            physics {
              backend classical
            }

            crystal bulk : all {
              prototype rocksalt
              composition NaCl
              a 4.212 A
              defect V_Na count 1
              defect V_Cl count 1
            }

            residual none
            """),
        statements=("defect V_Na count 1", "defect V_Cl count 1",
                    "conserve atoms Cl 3 Na 3"),
        build=True, conserve=True),
]

SOLUTION_CASES = [
    Case(
        pid="P17", family="solution",
        prompt="A 4x4x4 fcc medium-entropy alloy CrCoNi, a = 3.56 A, with "
               "equal random occupancy of the three species on every site, "
               "seed 12. 256 atoms: Cr 85, Co 85, Ni 86.",
        points=("fcc with occupancy Cr/Co/Ni 1/3 each", "256 sites"),
        writer=_t("""
            chaord 0.1
            dialect core + metal

            system {
              cell 14.24 14.24 14.24 A
              pbc xyz
              seed 12
              conserve atoms Co 85 Cr 85 Ni 86
            }

            physics {
              backend mlp
              model "mace-mp-0"
            }

            crystal mea : all {
              lattice fcc
              a 3.56 A
              occupancy Cr 1/3 Co 1/3 Ni 1/3
            }

            residual none
            """),
        statements=("occupancy Cr 1/3 Co 1/3 Ni 1/3",
                    "conserve atoms Co 85 Cr 85 Ni 86"),
        build=True, conserve=True),
    Case(
        pid="P18", family="solution",
        prompt="A 2x2x2 fcc CrCoNi cell (a = 3.56 A) with equal occupancy and "
               "a short-range-order restraint on the first Cr-Cr shell: "
               "Warren-Cowley alpha1 = +0.10 +- 0.02 (mild Cr clustering), "
               "seed 3. 32 sites: Cr 11, Co 11, Ni 10.",
        points=("occupancy thirds", "constrain sro alpha1 Cr-Cr +0.10"),
        writer=_t("""
            chaord 0.1
            dialect core + metal

            system {
              cell 7.12 7.12 7.12 A
              pbc xyz
              seed 3
              conserve atoms Co 11 Cr 11 Ni 10
            }

            physics {
              backend mlp
            }

            crystal mea : all {
              lattice fcc
              a 3.56 A
              occupancy Cr 1/3 Co 1/3 Ni 1/3
              constrain sro alpha1 Cr-Cr +0.10 +- 0.02
            }

            residual none
            """),
        statements=("constrain sro alpha1 Cr-Cr +0.10 +- 0.02",
                    "conserve atoms Co 11 Cr 11 Ni 10"),
        build=True, conserve=True),
    Case(
        pid="P19", family="solution",
        prompt="A disordered 2x2x2 fcc Cu-Ni solid solution, a = 3.59 A, "
               "50/50 random occupancy, seed 7. 16 Cu and 16 Ni.",
        points=("binary 50/50 occupancy", "Cu 16 Ni 16"),
        writer=_t("""
            chaord 0.1
            dialect core + metal

            system {
              cell 7.18 7.18 7.18 A
              pbc xyz
              seed 7
              conserve atoms Cu 16 Ni 16
            }

            physics {
              backend eam
            }

            crystal alloy : all {
              lattice fcc
              a 3.59 A
              occupancy Cu 1/2 Ni 1/2
            }

            residual none
            """),
        statements=("occupancy Cu 1/2 Ni 1/2", "conserve atoms Cu 16 Ni 16"),
        build=True, conserve=True),
    Case(
        pid="P20", family="solution",
        prompt="Dilute fcc Al-2at%Zn: 2x2x2 supercell a = 4.05 A, occupancy "
               "Zn 0.02 Al 0.98, seed 15. 31 Al and 1 Zn.",
        points=("dilute occupancy 2%", "Al 31 Zn 1"),
        writer=_t("""
            chaord 0.1
            dialect core + metal

            system {
              cell 8.1 8.1 8.1 A
              pbc xyz
              seed 15
              conserve atoms Al 31 Zn 1
            }

            physics {
              backend eam
            }

            crystal alloy : all {
              lattice fcc
              a 4.05 A
              occupancy Zn 0.02 Al 0.98
            }

            residual none
            """),
        statements=("occupancy Zn 0.02 Al 0.98", "conserve atoms Al 31 Zn 1"),
        build=True, conserve=True),
    Case(
        pid="P21", family="solution",
        prompt="A quinary high-entropy-ish fcc alloy FeCoCrNiCu, a = 3.60 A, "
               "4x4x4 supercell, equimolar occupancy (1/5 each), seed 21. "
               "Fe 51, Co 51, Cr 51, Ni 51, Cu 52.",
        points=("five-species occupancy 1/5", "256 sites"),
        writer=_t("""
            chaord 0.1
            dialect core + metal

            system {
              cell 14.4 14.4 14.4 A
              pbc xyz
              seed 21
              conserve atoms Co 51 Cr 51 Cu 52 Fe 51 Ni 51
            }

            physics {
              backend mlp
            }

            crystal hea : all {
              lattice fcc
              a 3.6 A
              occupancy Fe 1/5 Co 1/5 Cr 1/5 Ni 1/5 Cu 1/5
            }

            residual none
            """),
        statements=("occupancy Fe 1/5 Co 1/5 Cr 1/5 Ni 1/5 Cu 1/5",
                    "conserve atoms Co 51 Cr 51 Cu 52 Fe 51 Ni 51"),
        build=True, conserve=True),
]

AMORPHOUS_CASES = [
    Case(
        pid="P22", family="amorphous",
        prompt="Amorphous silicon: 216 Si atoms at mass density 2.28 g/cm3, "
               "melt-quench history from 3000 K, seed 5. Assert tetrahedral "
               "order: cn 4.0 +- 0.1 at cutoff 2.85 A.",
        points=("amorphous region, density + history", "Si 216 conserved"),
        writer=_t("""
            chaord 0.1
            dialect core + glass

            system {
              cell auto cubic
              pbc xyz
              seed 5
              conserve atoms Si 216
            }

            physics {
              backend mlp
              model "mace-mp-0"
            }

            amorphous aSi : all {
              composition Si
              state density 2.28 g/cm3
              history melt 3000 K for 20 ps -> quench to 300 K at 1 K/ps -> anneal 300 K for 50 ps
              assert cn 4.0 +- 0.1 cutoff 2.85 A
            }

            residual none
            """),
        statements=("composition Si", "state density 2.28 g/cm3",
                    "conserve atoms Si 216"),
        build=True, conserve=True),
    Case(
        pid="P23", family="amorphous",
        prompt="A small amorphous silicon sample: 64 atoms, density "
               "2.28 g/cm3, melt-quench protocol, seed 9.",
        points=("64 Si atoms", "same protocol as P22"),
        writer=_t("""
            chaord 0.1
            dialect core + glass

            system {
              cell auto cubic
              pbc xyz
              seed 9
              conserve atoms Si 64
            }

            physics {
              backend mlp
            }

            amorphous aSi : all {
              composition Si
              state density 2.28 g/cm3
              history melt 3000 K for 20 ps -> quench to 300 K at 1 K/ps
            }

            residual none
            """),
        statements=("conserve atoms Si 64", "state density 2.28 g/cm3"),
        build=True, conserve=True),
    Case(
        pid="P24", family="amorphous",
        prompt="Amorphous carbon at 2.00 g/cm3: 125 C atoms, quench from "
               "4000 K to 300 K, seed 13.",
        points=("amorphous C, 125 atoms", "density 2.00 g/cm3"),
        writer=_t("""
            chaord 0.1
            dialect core + glass

            system {
              cell auto cubic
              pbc xyz
              seed 13
              conserve atoms C 125
            }

            physics {
              backend mlp
            }

            amorphous aC : all {
              composition C
              state density 2.00 g/cm3
              history melt 4000 K for 20 ps -> quench to 300 K at 1 K/ps
            }

            residual none
            """),
        statements=("composition C", "conserve atoms C 125"),
        build=True, conserve=True),
    Case(
        pid="P25", family="amorphous",
        prompt="A 100-atom amorphous silicon model at 2.28 g/cm3 with "
               "assertions on the bond angle (109.0 deg mean +- 1.5) and "
               "coordination (4.0 +- 0.1), seed 17.",
        points=("angle_mean and cn asserts", "Si 100"),
        writer=_t("""
            chaord 0.1
            dialect core + glass

            system {
              cell auto cubic
              pbc xyz
              seed 17
              conserve atoms Si 100
            }

            physics {
              backend mlp
            }

            amorphous aSi : all {
              composition Si
              state density 2.28 g/cm3
              assert cn 4.0 +- 0.1 cutoff 2.85 A
              assert angle_mean 109.0 +- 1.5 deg
            }

            residual none
            """),
        statements=("assert cn 4.0 +- 0.1 cutoff 2.85 A",
                    "assert angle_mean 109.0 +- 1.5 deg",
                    "conserve atoms Si 100"),
        build=True, conserve=True),
]

LIQUID_CASES = [
    Case(
        pid="P26", family="liquid",
        prompt="Liquid water: 60 H2O molecules in a 15 A cubic box at 300 K, "
               "classical backend, seed 3. Assert mean O-O coordination "
               "around 4.4 within 3.1 A.",
        points=("liquid region, molecules H2O 60", "cell 15 A cube"),
        writer=_t("""
            chaord 0.1
            dialect core + molecular

            system {
              cell 15 15 15 A
              pbc xyz
              seed 3
              state T 300 K
              conserve charge 0
            }

            physics {
              backend classical
            }

            liquid water : all {
              molecules H2O 60
              assert cn 4.4 +- 0.6 cutoff 3.10 A
            }

            residual none
            """),
        statements=("molecules H2O 60", "cell 15 15 15 A"),
        build=True, atoms={"H": 120, "O": 60}),
    Case(
        pid="P27", family="liquid",
        prompt="Aqueous NaCl brine: 100 water molecules plus 10 Na+ and 10 "
               "Cl- ions in a 25 A box at 300 K, seed 11, neutral cell.",
        points=("H2O 100, Na+ 10, Cl- 10", "charge neutrality stated"),
        writer=_t("""
            chaord 0.1
            dialect core + molecular

            system {
              cell 25 25 25 A
              pbc xyz
              seed 11
              state T 300 K
              conserve charge 0
            }

            physics {
              backend classical
            }

            liquid brine : all {
              molecules H2O 100 Na+ 10 Cl- 10
            }

            residual none
            """),
        statements=("molecules H2O 100 Na+ 10 Cl- 10", "conserve charge 0"),
        build=True, atoms={"H": 200, "O": 100, "Na+": 10, "Cl-": 10}),
    Case(
        pid="P28", family="liquid",
        prompt="Liquid argon in Lennard-Jones reduced units: 108 atoms in a "
               "6x6x6 box at T* = 0.71, LJ backend (epsilon 1, sigma 1, "
               "cutoff 2.5), seed 1.",
        points=("LJ units fluid", "conserve atoms Ar 108"),
        writer=_t("""
            chaord 0.1
            dialect core + lj

            system {
              units lj
              cell 6 6 6
              pbc xyz
              seed 1
              state T 0.71
              conserve atoms Ar 108
            }

            physics {
              backend lj
              epsilon 1
              sigma 1
              cutoff 2.5
            }

            liquid argon : all {
              state density 0.84
            }

            residual none
            """),
        statements=("units lj", "conserve atoms Ar 108", "backend lj"),
        build=True, conserve=True),
    Case(
        pid="P29", family="liquid",
        prompt="A low-density water box for cavitation studies: 40 H2O "
               "molecules at mass density 0.5 g/cm3 (auto cubic cell), "
               "300 K, seed 8.",
        points=("auto cell from state density", "40 waters"),
        writer=_t("""
            chaord 0.1
            dialect core + molecular

            system {
              pbc xyz
              seed 8
              state T 300 K
            }

            physics {
              backend classical
            }

            liquid water : all {
              molecules H2O 40
              state density 0.5 g/cm3
            }

            residual none
            """),
        statements=("molecules H2O 40", "state density 0.5 g/cm3"),
        build=True, atoms={"H": 80, "O": 40}),
    Case(
        pid="P30", family="liquid",
        prompt="Liquid CO2: 20 molecules in an 18 A cubic box at 250 K, "
               "seed 5.",
        points=("CO2 20 in 18 A box", "C 20 O 40 atoms"),
        writer=_t("""
            chaord 0.1
            dialect core + molecular

            system {
              cell 18 18 18 A
              pbc xyz
              seed 5
              state T 250 K
            }

            physics {
              backend classical
            }

            liquid co2 : all {
              molecules CO2 20
            }

            residual none
            """),
        statements=("molecules CO2 20", "cell 18 18 18 A"),
        build=True, atoms={"C": 20, "O": 40}),
]

GAS_CASES = [
    Case(
        pid="P31", family="gas",
        prompt="An ideal-ish air mixture: 40 N2 and 10 O2 molecules in a "
               "40 A cubic box at 300 K and 1 bar, classical TrapPE-UA "
               "backend, seed 2. Assert compressibility near 1.",
        points=("gas region, N2 40 O2 10", "Z = 1.00 +- 0.01"),
        writer=_t("""
            chaord 0.1
            dialect core + molecular

            system {
              cell 40 40 40 A
              pbc xyz
              seed 2
              state T 300 K
              state P 1 bar
            }

            physics {
              backend classical
              forcefield "trappe-ua"
            }

            gas air : all {
              molecules N2 40 O2 10
              assert compressibility 1.00 +- 0.01
            }

            residual none
            """),
        statements=("molecules N2 40 O2 10", 'forcefield "trappe-ua"',
                    "assert compressibility 1.00 +- 0.01"),
        build=True, atoms={"N": 80, "O": 20}),
    Case(
        pid="P32", family="gas",
        prompt="Pure argon gas: 50 atoms in a 30 A cubic box at 300 K, "
               "seed 4.",
        points=("Ar 50 monatomic gas"),
        writer=_t("""
            chaord 0.1
            dialect core + molecular

            system {
              cell 30 30 30 A
              pbc xyz
              seed 4
              state T 300 K
            }

            physics {
              backend classical
            }

            gas argon : all {
              molecules Ar 50
            }

            residual none
            """),
        statements=("molecules Ar 50", "cell 30 30 30 A"),
        build=True, atoms={"Ar": 50}),
    Case(
        pid="P33", family="gas",
        prompt="Carbon-dioxide gas: 15 CO2 molecules in a 35 A box at "
               "323 K, seed 6.",
        points=("CO2 15", "C 15 O 30"),
        writer=_t("""
            chaord 0.1
            dialect core + molecular

            system {
              cell 35 35 35 A
              pbc xyz
              seed 6
              state T 323 K
            }

            physics {
              backend classical
            }

            gas co2 : all {
              molecules CO2 15
            }

            residual none
            """),
        statements=("molecules CO2 15"),
        build=True, atoms={"C": 15, "O": 30}),
    Case(
        pid="P34", family="gas",
        prompt="Humid nitrogen: 30 N2 molecules and 5 water molecules in a "
               "40 A box at 300 K, seed 7.",
        points=("N2 30 + H2O 5", "element census N 60 O 5 H 10"),
        writer=_t("""
            chaord 0.1
            dialect core + molecular

            system {
              cell 40 40 40 A
              pbc xyz
              seed 7
              state T 300 K
            }

            physics {
              backend classical
            }

            gas humid : all {
              molecules N2 30 H2O 5
            }

            residual none
            """),
        statements=("molecules N2 30 H2O 5"),
        build=True, atoms={"N": 60, "O": 5, "H": 10}),
]

SURFACE_CASES = [
    Case(
        pid="P35", family="surface",
        prompt="A clean Cu(001) surface slab: fcc copper, a = 3.615 A, "
               "(001) cut with vacuum above, EAM backend, seed 3. 36 Cu "
               "atoms in the slab.",
        points=("fcc Cu, surface (001) top", "36 atoms + vacuum"),
        writer=_t("""
            chaord 0.1
            dialect core + metal + surface

            system {
              cell 10.845 10.845 22.0 A
              pbc xyz
              seed 3
              conserve atoms Cu 36
            }

            physics {
              backend eam
            }

            crystal slab : slab z 0 .. 10.0 A {
              lattice fcc
              a 3.615 A
              surface (001) top
            }

            vacuum gap : slab z 10.0 .. 22.0 A {
            }

            residual none
            """),
        statements=("surface (001) top", "conserve atoms Cu 36"),
        build=True, conserve=True),
    Case(
        pid="P36", family="surface",
        prompt="A clean Cu(110) surface slab (same parameters as Cu(001) "
               "but the (110) cut), seed 5. 36 Cu atoms.",
        points=("fcc Cu, surface (110) top", "36 atoms"),
        writer=_t("""
            chaord 0.1
            dialect core + metal + surface

            system {
              cell 10.225 7.23 22.0 A
              pbc xyz
              seed 5
              conserve atoms Cu 36
            }

            physics {
              backend eam
            }

            crystal slab : slab z 0 .. 10.0 A {
              lattice fcc
              a 3.615 A
              surface (110) top
            }

            vacuum gap : slab z 10.0 .. 22.0 A {
            }

            residual none
            """),
        statements=("surface (110) top", "conserve atoms Cu 36"),
        build=True, conserve=True),
    Case(
        pid="P37", family="surface",
        prompt="A Ni(111) surface slab, fcc a = 3.615 A, vacuum on top, "
               "seed 3, EAM. 36 Ni atoms.",
        points=("fcc Ni, surface (111) top", "36 atoms"),
        writer=_t("""
            chaord 0.1
            dialect core + metal + surface

            system {
              cell 7.669 6.641 18.26 A
              pbc xyz
              seed 3
              conserve atoms Ni 36
            }

            physics {
              backend eam
            }

            crystal slab : slab z 0 .. 6.3 A {
              lattice fcc
              a 3.615 A
              surface (111) top
            }

            vacuum gap : slab z 6.3 .. 18.26 A {
            }

            residual none
            """),
        statements=("surface (111) top", "conserve atoms Ni 36"),
        build=True, conserve=True),
    Case(
        pid="P38", family="surface",
        prompt="Oxygen adsorbed on Ni(111): fcc nickel slab a = 3.615 A with "
               "4 O atoms on top sites (2.0 A above the top layer), seed 3. "
               "36 Ni + 4 O.",
        points=("adsorb O count 4 site top", "Ni 36 O 4"),
        writer=_t("""
            chaord 0.1
            dialect core + metal + surface

            system {
              cell 7.669 6.641 18.26 A
              pbc xyz
              seed 3
              conserve atoms Ni 36 O 4
            }

            physics {
              backend eam
            }

            crystal slab : slab z 0 .. 6.3 A {
              lattice fcc
              a 3.615 A
              surface (111) top
              adsorb O count 4 site top
            }

            vacuum gap : slab z 6.3 .. 18.26 A {
            }

            residual none
            """),
        statements=("surface (111) top", "adsorb O count 4 site top",
                    "conserve atoms Ni 36 O 4"),
        build=True, conserve=True),
    Case(
        pid="P39", family="surface",
        prompt="A quarter-monolayer of oxygen on Pt(001): fcc platinum "
               "a = 3.924 A, adsorbate coverage 0.25 ML on the top layer, "
               "seed 7. 36 Pt + 2 O.",
        points=("adsorb O coverage 0.25 ML", "Pt 36 O 2"),
        writer=_t("""
            chaord 0.1
            dialect core + metal + surface

            system {
              cell 11.772 11.772 22.0 A
              pbc xyz
              seed 7
              conserve atoms Pt 36 O 2
            }

            physics {
              backend eam
            }

            crystal slab : slab z 0 .. 10.0 A {
              lattice fcc
              a 3.924 A
              surface (001) top
              adsorb O coverage 0.25 ML
            }

            vacuum gap : slab z 10.0 .. 22.0 A {
            }

            residual none
            """),
        statements=("adsorb O coverage 0.25 ML", "conserve atoms Pt 36 O 2"),
        build=True, conserve=True),
    Case(
        pid="P40", family="surface",
        prompt="A clean Si(001) diamond slab, a = 5.431 A, vacuum above, "
               "seed 2, ML backend. 36 Si atoms.",
        points=("diamond Si, surface (001) top", "36 atoms"),
        writer=_t("""
            chaord 0.1
            dialect core + glass + surface

            system {
              cell 16.293 16.293 26.0 A
              pbc xyz
              seed 2
              conserve atoms Si 36
            }

            physics {
              backend mlp
              model "mace-mp-0"
            }

            crystal slab : slab z 0 .. 14.0 A {
              lattice diamond
              a 5.431 A
              surface (001) top
            }

            vacuum gap : slab z 14.0 .. 26.0 A {
            }

            residual none
            """),
        statements=("lattice diamond", "surface (001) top",
                    "conserve atoms Si 36"),
        build=True, conserve=True),
]

SLAB_CASES = [
    Case(
        pid="P41", family="slab",
        prompt="An LJ solid-liquid coexistence slab: fcc solid (a* = 1.609) "
               "filling z 15..8 and liquid at density 0.854 filling z 8..15 "
               "of an 8x8x20 box at T* = 0.65, with one vacancy in the solid "
               "at depth 6.0 and +0.9% zz strain. 1000 X atoms, seed 7. "
               "Interfaces at z = 8 and z = 15.",
        points=("crystal+liquid z slabs", "LJ physics block",
                "conserve atoms X 1000"),
        writer=_t("""
            chaord 0.1
            dialect core + lj

            system {
              units lj
              cell 8 8 20
              pbc xyz
              seed 7
              state T 0.65
              conserve atoms X 1000
            }

            physics {
              backend lj
              epsilon 1
              sigma 1
              cutoff 2.5
            }

            crystal A : slab z 15.0 .. 8.0 {
              lattice fcc
              a 1.609
              constrain strain zz +0.9 % +- 0.2
              defect vacancy count 1 depth 6.0
              assert sites_matched 99.9 %
            }

            liquid B : slab z 8.0 .. 15.0 {
              state density 0.854
            }

            interface A | B {
              at z 8.0
              width 1.1
            }

            interface B | A {
              at z 15.0
              width 1.9
            }

            residual none
            """),
        statements=("constrain strain zz +0.9 % +- 0.2",
                    "defect vacancy count 1 depth 6.0",
                    "interface A | B {", "conserve atoms X 1000"),
        build=True, conserve=True),
    Case(
        pid="P42", family="slab",
        prompt="A clean LJ solid-liquid interface: fcc solid (a* = 1.609) in "
               "z 14.5..8, liquid z 8..14.5 of the 8x8x20 box, T* = 0.65, no "
               "defects, no strain. 1000 X atoms, seed 3.",
        points=("plain two-slab program", "no defect statements"),
        writer=_t("""
            chaord 0.1
            dialect core + lj

            system {
              units lj
              cell 8 8 20
              pbc xyz
              seed 3
              state T 0.65
              conserve atoms X 1000
            }

            physics {
              backend lj
              epsilon 1
              sigma 1
              cutoff 2.5
            }

            crystal A : slab z 14.5 .. 8.0 {
              lattice fcc
              a 1.609
              assert sites_matched 100.0 %
            }

            liquid B : slab z 8.0 .. 14.5 {
              state density 0.854
              assert cn 12.0 +- 1.1 cutoff 1.50
            }

            interface A | B {
              at z 8.0
              width 1.0
            }

            residual none
            """),
        statements=("lattice fcc", "a 1.609", "conserve atoms X 1000"),
        build=True, conserve=True),
    Case(
        pid="P43", family="slab",
        prompt="An LJ slab with a split divacancy in the solid: crystal "
               "z 15..8 (a* = 1.609) strained -1.5% in zz, one divacancy at "
               "depth 6.2, liquid z 8..15 at density 0.854, 8x8x20 box, "
               "T* = 0.65, 1000 X atoms, seed 9.",
        points=("negative strain", "defect divacancy count 1 depth 6.2"),
        writer=_t("""
            chaord 0.1
            dialect core + lj

            system {
              units lj
              cell 8 8 20
              pbc xyz
              seed 9
              state T 0.65
              conserve atoms X 1000
            }

            physics {
              backend lj
              epsilon 1
              sigma 1
              cutoff 2.5
            }

            crystal A : slab z 15.0 .. 8.0 {
              lattice fcc
              a 1.609
              constrain strain zz -1.5 % +- 0.2
              defect divacancy count 1 depth 6.2 form split
            }

            liquid B : slab z 8.0 .. 15.0 {
              state density 0.854
            }

            interface A | B {
              at z 8.0
              width 1.1
            }

            residual none
            """),
        statements=("constrain strain zz -1.5 % +- 0.2",
                    "defect divacancy count 1 depth 6.2 form split",
                    "conserve atoms X 1000"),
        build=True, conserve=True),
]

EXTENDED_CASES = [
    Case(
        pid="P44", family="bicrystal",
        prompt="A Sigma-5 [001] tilt grain boundary in fcc copper: two "
               "grains rotated about z on a CSL lattice, a = 3.615 A, EAM "
               "backend, seed 5. 1536 Cu atoms.",
        points=("grain_boundary sigma 5", "fcc Cu host"),
        writer=_t("""
            chaord 0.1
            dialect core + metal

            system {
              pbc xyz
              seed 5
              conserve atoms Cu 1536
            }

            physics {
              backend eam
            }

            crystal bicrystal : all {
              lattice fcc
              a 3.615 A
              grain_boundary sigma 5
            }

            residual none
            """),
        statements=("lattice fcc", "grain_boundary sigma 5",
                    "conserve atoms Cu 1536"),
        custom=_grain_boundary_check),
    Case(
        pid="P45", family="bicrystal",
        prompt="A Sigma-13 [001] CSL bicrystal of fcc copper, a = 3.615 A, "
               "EAM, seed 6. 1536 Cu atoms.",
        points=("grain_boundary sigma 13"),
        writer=_t("""
            chaord 0.1
            dialect core + metal

            system {
              pbc xyz
              seed 6
              conserve atoms Cu 1536
            }

            physics {
              backend eam
            }

            crystal bicrystal : all {
              lattice fcc
              a 3.615 A
              grain_boundary sigma 13
            }

            residual none
            """),
        statements=("grain_boundary sigma 13", "conserve atoms Cu 1536"),
        custom=_grain_boundary_check),
    Case(
        pid="P46", family="bicrystal",
        prompt="A single edge dislocation in fcc copper (Burgers vector "
               "a/2<110>, line along z), a = 3.615 A, EAM, seed 2. "
               "576 Cu atoms.",
        points=("dislocation edge", "fcc Cu host"),
        writer=_t("""
            chaord 0.1
            dialect core + metal

            system {
              pbc xyz
              seed 2
              conserve atoms Cu 576
            }

            physics {
              backend eam
            }

            crystal bulk : all {
              lattice fcc
              a 3.615 A
              dislocation edge
            }

            residual none
            """),
        statements=("dislocation edge", "conserve atoms Cu 576"),
        custom=_dislocation_check),
]

ELECTROLYTE_CASES = [
    Case(
        pid="P47", family="electrolyte",
        prompt="A lithium-ion battery electrolyte: 600 ethylene carbonate "
               "(SMILES C1COC(=O)O1), 45 Li+ and 45 PF6- ions at 300 K and "
               "1 bar (about 1 M), neutral charge, seed 11.",
        points=("species block with SMILES", "molecules EC 600 Li+ 45 PF6- 45"),
        writer=_t("""
            chaord 0.1
            dialect core + molecular

            system {
              cell auto cubic
              pbc xyz
              seed 11
              state T 300 K
              state P 1 bar
              conserve charge 0
            }

            physics {
              backend mlp
              model "mace-mp-0"
            }

            species {
              molecule EC = smiles "C1COC(=O)O1"
              ion Li+ = smiles "[Li+]"
              ion PF6- = smiles "F[P-](F)(F)(F)(F)F"
            }

            liquid electrolyte : all {
              molecules EC 600 Li+ 45 PF6- 45
            }

            residual none
            """),
        statements=('molecule EC = smiles "C1COC(=O)O1"',
                    "molecules EC 600 Li+ 45 PF6- 45", "conserve charge 0"),
        schema=True),
    Case(
        pid="P48", family="electrolyte",
        prompt="A reactive rutile TiO2(110)-water interface: rutile slab "
               "(a = 4.594 A, c = 2.959 A) in z 0..18, water slab z 18..45 "
               "with 620 molecules, vacuum above, two bridging-oxygen "
               "vacancies, 9 water dissociations OH @ Ti_5c + H @ O_br, "
               "330 K, seed 9.",
        points=("rutile + water + vacuum slabs", "dissociate statement"),
        writer=_t("""
            chaord 0.1
            dialect core + ionic + surface + molecular

            system {
              cell 26.63 25.99 60.0 A
              pbc xyz
              seed 9
              state T 330 K
            }

            physics {
              backend mlp
              model "mace-mp-0"
            }

            crystal rutile : slab z 0 .. 18 A {
              prototype rutile
              composition TiO2
              a 4.594 A
              c 2.959 A
              orient x [001] y [1-10] z [110]
              surface (110) top
              defect V_O^.. count 2 layer surface
            }

            liquid water : slab z 18 .. 45 A {
              molecules H2O 620
              state density 1.0 g/cm3
            }

            vacuum gap : slab z 45 .. 60 A {
            }

            interface rutile | water {
              dissociate H2O -> OH @ Ti_5c + H @ O_br count 9
              assert coverage OH 0.25 ML +- 0.05
            }

            residual none
            """),
        statements=("prototype rutile", "surface (110) top",
                    "dissociate H2O -> OH @ Ti_5c + H @ O_br count 9",
                    "vacuum gap : slab z 45 .. 60 A {"),
        schema=True),
    Case(
        pid="P49", family="electrolyte",
        prompt="A LiCl solution: 70 water molecules with 7 Li+ and 7 Cl- "
               "ions in a 22 A box at 300 K, neutral, seed 13.",
        points=("H2O 70, Li+ 7, Cl- 7"),
        writer=_t("""
            chaord 0.1
            dialect core + molecular

            system {
              cell 22 22 22 A
              pbc xyz
              seed 13
              state T 300 K
              conserve charge 0
            }

            physics {
              backend classical
            }

            liquid licl : all {
              molecules H2O 70 Li+ 7 Cl- 7
            }

            residual none
            """),
        statements=("molecules H2O 70 Li+ 7 Cl- 7"),
        build=True, atoms={"H": 140, "O": 70, "Li+": 7, "Cl-": 7}),
    Case(
        pid="P50", family="electrolyte",
        prompt="Carbonated water (soda): 50 H2O and 5 CO2 molecules in a "
               "20 A box at 298 K, seed 14.",
        points=("H2O 50 + CO2 5"),
        writer=_t("""
            chaord 0.1
            dialect core + molecular

            system {
              cell 20 20 20 A
              pbc xyz
              seed 14
              state T 298 K
            }

            physics {
              backend classical
            }

            liquid soda : all {
              molecules H2O 50 CO2 5
            }

            residual none
            """),
        statements=("molecules H2O 50 CO2 5"),
        build=True, atoms={"H": 100, "C": 5, "O": 60}),
]

CASES = (CRYSTAL_CASES + DEFECT_CASES + SOLUTION_CASES + AMORPHOUS_CASES
         + LIQUID_CASES + GAS_CASES + SURFACE_CASES + SLAB_CASES
         + EXTENDED_CASES + ELECTROLYTE_CASES)

FAMILIES = ("crystal", "defects", "solution", "amorphous", "liquid", "gas",
            "surface", "slab", "bicrystal", "electrolyte")


# ------------------------------------------------------------------- main ----

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--writer", choices=("deterministic", "file"),
                    default="deterministic",
                    help="answer source: built-in correct programs, or a JSON "
                         "file {prompt: program_text} from a real LLM run")
    ap.add_argument("--file", help="JSON answers for --writer file")
    ap.add_argument("--case", action="append",
                    help="run only this prompt id (repeatable)")
    args = ap.parse_args(argv)

    cases = CASES
    if args.case:
        wanted = set(args.case)
        cases = [c for c in CASES if c.pid in wanted]
        missing = wanted - {c.pid for c in cases}
        if missing:
            print(f"unknown case ids: {sorted(missing)}")
            return 2
    result = run_suite(writer=args.writer, file=args.file, cases=cases)

    for r in result.results:
        line = f"{r.pid} {r.family:<11} {'ok  ' if r.ok else 'FAIL'} {r.stages}"
        if not r.ok:
            line += f"\n     {r.error[:300]}"
        print(line)

    print()
    for fam in FAMILIES:
        if fam in result.by_family():
            p, t = result.by_family()[fam]
            print(f"  {fam:<11} {p}/{t}")
    pct = 100 * result.rate
    verdict = "PASS" if result.rate >= PASS_RATE_TARGET else "FAIL"
    print(f"\nfirst-compile pass rate: {result.passed}/{result.total} "
          f"({pct:.1f}%)  [target >= {100 * PASS_RATE_TARGET:.0f}%]  {verdict}")
    return 0 if result.rate >= PASS_RATE_TARGET else 1


if __name__ == "__main__":
    sys.exit(main())
