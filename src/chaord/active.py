"""Active-learning hooks: local uncertainty report and frame proposals.

`uncertainty_report(program, frame, dialect)` summarises how well a program
explains one frame, using only local signals — never a network call:

- `assert` statements are re-measured on the frame when a local CV measure
  exists (cn, angle_mean); asserts without a local measure, or without a
  stated tolerance, count as *unverified* (they contribute uncertainty);
- the residual block counts the atoms the program could not explain;
- `assert sites_matched <x> %` is read as the lattice-fit score of the lift;
- atom conservation compares the stated census (`conserve atoms`, or the
  element census derivable from `molecules` through the local templates)
  with what the frame actually contains.

`propose_frames(...)` ranks frames by that uncertainty and labels each with
the next action: send to labeling, escalate to a higher-accuracy backend, or
accept. Both functions are pure-local heuristics; the actual labeling /
escalation is left to an external active-learning bridge (every result is a
JSON-ready dataclass with `to_dict()`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

from .cv import measure as _cv_measure
from .io.frames import Frame
from .lang.ir import Program

# next-action bands for propose_frames; a dialect may override them through
# the thresholds `active_label_min` / `active_escalate_min` (none ships yet)
_LABEL_MIN = 0.5
_ESCALATE_MIN = 0.2

# how much each local signal weighs in the total score (sums to 1.0; the fit
# term is only added when the program states sites_matched)
_W_ASSERT_FAILURES = 0.40
_W_UNVERIFIED = 0.10
_W_RESIDUAL = 0.25
_W_CONSERVATION = 0.15
_W_FIT_DEFICIT = 0.10

# assert keys that have a local CV measure usable for re-checking a frame
_LOCALLY_CHECKABLE = ("cn", "angle_mean")

ACTION_LABEL = "label"        # worth sending to a human / oracle for labeling
ACTION_ESCALATE = "escalate"  # worth recomputing with a higher-accuracy backend
ACTION_ACCEPT = "accept"      # the local explanation is good enough


# ----------------------------------------------------------------- report ----

@dataclass
class UncertaintyReport:
    """Local uncertainty of one (program, frame) pair. All fields JSON-ready."""

    frame_atoms: int = 0
    asserts_total: int = 0
    asserts_checked: int = 0
    asserts_failed: int = 0
    asserts_unverified: int = 0
    failures: list[dict] = field(default_factory=list)
    unverified: list[str] = field(default_factory=list)
    residual_atoms: int = 0
    residual_fraction: float = 0.0
    stated_atoms: dict = field(default_factory=dict)
    present_atoms: dict = field(default_factory=dict)
    conservation_mismatch: int = 0
    atom_census_known: bool = True
    sites_matched: Optional[float] = None
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Plain dict for the active-learning bridge (JSON serialisable)."""
        return {
            "frame_atoms": self.frame_atoms,
            "asserts": {"total": self.asserts_total,
                        "checked": self.asserts_checked,
                        "failed": self.asserts_failed,
                        "unverified": self.asserts_unverified},
            "failures": list(self.failures),
            "unverified": list(self.unverified),
            "residual_atoms": self.residual_atoms,
            "residual_fraction": self.residual_fraction,
            "stated_atoms": dict(self.stated_atoms),
            "present_atoms": dict(self.present_atoms),
            "conservation_mismatch": self.conservation_mismatch,
            "atom_census_known": self.atom_census_known,
            "sites_matched": self.sites_matched,
            "score": self.score,
            "reasons": list(self.reasons),
        }


def _num(value) -> float:
    text = getattr(value, "num", None)
    if text is None:
        return float("nan")
    if "/" in text:
        lo, hi = text.split("/")
        return float(lo) / float(hi)
    return float(text)


def _assert_values(stmt):
    """(target, tolerance, cutoff) of an assert statement, when extractable."""
    target = tol = cutoff = None
    values = stmt.values
    for i, v in enumerate(values):
        if v.t == "q" and target is None:
            target = _num(v)
        elif v.t == "tol" and tol is None:
            tol = _num(v.value)
        elif v.t == "n" and v.text == "cutoff" and i + 1 < len(values) \
                and values[i + 1].t == "q":
            cutoff = _num(values[i + 1])
    return target, tol, cutoff


def _asserts(program: Program):
    for block in program.blocks:
        for stmt in getattr(block, "statements", ()):
            if stmt.kind == "assert":
                yield stmt


def _present_atoms(frame: Frame) -> dict:
    census: dict[str, int] = {}
    for sym in frame.symbols:
        census[sym] = census.get(sym, 0) + 1
    return census


def _stated_atoms(program: Program):
    """(census, mode) with mode in {'conserve', 'molecules', 'unknown'}."""
    for block in program.blocks:
        if block.t != "system":
            continue
        for stmt in block.statements:
            if stmt.kind == "conserve" and stmt.key == "atoms":
                census: dict[str, int] = {}
                vals = stmt.values
                i = 0
                while i + 1 < len(vals):
                    if vals[i].t == "n" and vals[i + 1].t == "q":
                        census[vals[i].text] = int(_num(vals[i + 1]))
                    i += 1
                if census:
                    return census, "conserve"
    from .build.molecules import TEMPLATES
    census = {}
    unknown = False
    for block in program.blocks:
        if block.t != "region":
            continue
        for stmt in block.statements:
            if stmt.key != "molecules":
                continue
            vals = stmt.values
            i = 0
            while i < len(vals):
                if vals[i].t != "n":
                    i += 1
                    continue
                name = vals[i].text
                count = 1
                if i + 1 < len(vals) and vals[i + 1].t == "q":
                    count = int(_num(vals[i + 1]))
                    i += 2
                else:
                    i += 1
                if name in TEMPLATES:
                    for sym in TEMPLATES[name]["symbols"]:
                        census[sym] = census.get(sym, 0) + count
                else:
                    unknown = True
    if census and not unknown:
        return census, "molecules"
    return {}, "unknown"


def uncertainty_report(program: Program, frame: Frame, dialect) -> UncertaintyReport:
    """Summarise how certain a lifted program is about one frame (pure local).

    Signals: assert failures re-measured where a local CV exists, unverified
    asserts, residual (unexplained) atoms, the sites_matched fit score of the
    lift, and atom conservation between the program and the frame. The score
    is a weighted combination in [0, 1]; higher means more uncertain."""
    report = UncertaintyReport(frame_atoms=len(frame))
    present = _present_atoms(frame)
    report.present_atoms = present

    # --- assert statements --------------------------------------------------
    for stmt in _asserts(program):
        report.asserts_total += 1
        if stmt.key == "sites_matched":
            target, _, _ = _assert_values(stmt)
            if target is not None:
                report.sites_matched = max(0.0, min(target / 100.0, 1.0))
            report.unverified.append(stmt.key)
            report.asserts_unverified += 1
            continue
        target, tol, cutoff = _assert_values(stmt)
        if stmt.key not in _LOCALLY_CHECKABLE or target is None or tol is None:
            report.unverified.append(stmt.key)
            report.asserts_unverified += 1
            continue
        try:
            kwargs = {"cutoff": cutoff} if cutoff is not None else {}
            measured = float(_cv_measure(stmt.key, frame, dialect, **kwargs))
        except Exception:
            report.unverified.append(stmt.key)
            report.asserts_unverified += 1
            continue
        report.asserts_checked += 1
        if not (abs(measured - target) <= tol):  # NaN fails closed: unexplained
            report.asserts_failed += 1
            report.failures.append({
                "key": stmt.key, "asserted": target, "measured": measured,
                "tolerance": tol,
                "cutoff": cutoff if cutoff is not None else "dialect"})

    # --- residual (unexplained) atoms ---------------------------------------
    report.residual_atoms = sum(
        len(b.statements) for b in program.blocks
        if b.t == "residual" and not b.none)
    report.residual_fraction = (report.residual_atoms / report.frame_atoms
                                if report.frame_atoms else 0.0)

    # --- atom conservation ---------------------------------------------------
    stated, mode = _stated_atoms(program)
    report.stated_atoms = stated
    if mode == "unknown":
        report.atom_census_known = False
        report.reasons.append(
            "no local atom census: species defined outside the local templates")
    else:
        report.conservation_mismatch = sum(
            abs(stated.get(sp, 0) - present.get(sp, 0))
            for sp in set(stated) | set(present))

    # --- score ---------------------------------------------------------------
    fail_frac = (report.asserts_failed / report.asserts_checked
                 if report.asserts_checked else 0.0)
    unver_frac = (report.asserts_unverified / report.asserts_total
                  if report.asserts_total else 0.0)
    cons_frac = (report.conservation_mismatch / report.frame_atoms
                 if report.frame_atoms else 0.0)
    score = (_W_ASSERT_FAILURES * fail_frac
             + _W_UNVERIFIED * unver_frac
             + _W_RESIDUAL * report.residual_fraction
             + _W_CONSERVATION * cons_frac)
    if report.sites_matched is not None:
        score += _W_FIT_DEFICIT * (1.0 - report.sites_matched)
    report.score = max(0.0, min(score, 1.0))

    # --- human-readable reasons ----------------------------------------------
    if report.asserts_failed:
        for f in report.failures:
            report.reasons.append(
                f"assert {f['key']} failed: stated {f['asserted']:.3g}, "
                f"measured {f['measured']:.3g} (+- {f['tolerance']:.3g})")
    if report.asserts_unverified:
        report.reasons.append(
            f"{report.asserts_unverified} of {report.asserts_total} asserts "
            "have no local measure")
    if report.residual_atoms:
        report.reasons.append(
            f"{report.residual_atoms} residual atoms "
            f"({100 * report.residual_fraction:.1f}%) unexplained")
    if report.conservation_mismatch:
        report.reasons.append(
            f"conservation mismatch: program states {stated}, frame has {present}")
    if report.sites_matched is not None and report.sites_matched < 1.0:
        report.reasons.append(
            f"site fit explains {100 * report.sites_matched:.1f}% of the sites")
    if not report.reasons:
        report.reasons.append("no local uncertainty signal")
    return report


# --------------------------------------------------------------- proposals ---

@dataclass
class FrameProposal:
    """One frame's uncertainty and the next action an active-learning loop
    should take with it."""

    frame_id: str
    score: float
    action: str
    reasons: list[str]
    report: Optional[UncertaintyReport] = None

    @property
    def label_worthy(self) -> bool:
        return self.action == ACTION_LABEL

    @property
    def escalate_worthy(self) -> bool:
        return self.action == ACTION_ESCALATE

    def to_dict(self) -> dict:
        return {"frame_id": self.frame_id, "score": self.score,
                "action": self.action, "reasons": list(self.reasons),
                "report": self.report.to_dict() if self.report is not None else None}


def _band_thresholds(dialect):
    """Action bands from the dialect when it defines them, else the defaults."""
    label_min, escalate_min = _LABEL_MIN, _ESCALATE_MIN
    if dialect is not None:
        try:
            label_min = float(dialect.threshold("active_label_min"))
        except Exception:
            pass
        try:
            escalate_min = float(dialect.threshold("active_escalate_min"))
        except Exception:
            pass
    return label_min, escalate_min


def propose_frames(candidates: Iterable, dialect=None, top: Optional[int] = None,
                   ) -> list[FrameProposal]:
    """Rank frames by uncertainty and propose the next action for each.

    `candidates` yields either `(frame_id, UncertaintyReport)` pairs (reports
    computed earlier) or `(frame_id, program, frame[, dialect])` tuples, which
    are scored here with `uncertainty_report`. Results are sorted by score
    (most uncertain first); `top` keeps only the first k. Actions: `label`
    (>= label band), `escalate` (>= escalate band), else `accept`."""
    proposals: list[FrameProposal] = []
    for item in candidates:
        frame_id, rest = item[0], item[1:]
        if rest and isinstance(rest[0], UncertaintyReport):
            report = rest[0]
            d = rest[1] if len(rest) > 1 else dialect
        else:
            program, frame = rest[0], rest[1]
            d = rest[2] if len(rest) > 2 else dialect
            report = uncertainty_report(program, frame, d)
        label_min, escalate_min = _band_thresholds(d)
        if report.score >= label_min:
            action = ACTION_LABEL
        elif report.score >= escalate_min:
            action = ACTION_ESCALATE
        else:
            action = ACTION_ACCEPT
        proposals.append(FrameProposal(
            frame_id=frame_id, score=report.score, action=action,
            reasons=list(report.reasons), report=report))
    proposals.sort(key=lambda p: p.score, reverse=True)
    return proposals[:top] if top is not None else proposals
