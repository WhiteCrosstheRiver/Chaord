"""M8 completion: prompt suite pass rate, active-learning hooks, Laya codec.

Covers the three M8 deliverables beyond the base acceptance in
test_integrations.py: the 50-prompt LLM validation suite (deterministic
writer must be 100%, the PLAN gate asks >= 90%), the local uncertainty
report and frame proposals, and the exact (opcode-based) Laya patch round
trip over random program pairs.
"""
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from chaord.active import (
    ACTION_ACCEPT, ACTION_ESCALATE, ACTION_LABEL,
    propose_frames, uncertainty_report,
)
from chaord.build import build_program
from chaord.dialects import dialect_from_program
from chaord.integrations import (
    fits_budget, laya_decode, laya_encode, laya_tokens,
)
from chaord.lang.errors import ChaordError
from chaord.lang.parser import parse_text

ROOT = Path(__file__).resolve().parent.parent


def _load_suite():
    spec = importlib.util.spec_from_file_location(
        "llm_prompt_suite", ROOT / "tools" / "llm_prompt_suite.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module     # dataclasses needs the module registered
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def suite():
    return _load_suite()


@pytest.fixture(scope="module")
def suite_run(suite):
    return suite.run_suite(writer="deterministic")


# ------------------------------------------------------------ prompt suite ---

def test_prompt_suite_deterministic_is_100_percent(suite_run):
    # the built-in answers validate the checking pipeline itself: they must
    # pass on the first compile attempt, well above the 90% M8 gate
    assert suite_run.total == 50
    assert suite_run.rate >= 0.90
    assert suite_run.rate == 1.0
    assert suite_run.failures == []


def test_prompt_suite_shape_and_families(suite):
    assert len(suite.CASES) == 50
    assert len({c.pid for c in suite.CASES}) == 50          # unique ids
    assert set(suite.FAMILIES) == {c.family for c in suite.CASES}
    for case in suite.CASES:
        # every prompt carries a natural-language ask and expected points
        assert case.prompt and len(case.prompt.split()) >= 8
        assert case.points
        # and at least one machine-checkable expectation beyond parsing
        assert (case.statements or case.build or case.conserve
                or case.atoms or case.schema or case.custom)


def test_prompt_suite_file_mode_bridges_real_llm_answers(suite, tmp_path):
    good = next(c for c in suite.CASES if c.pid == "P01")
    other = next(c for c in suite.CASES if c.pid == "P02")
    answers = {
        good.pid: good.writer(),            # keyed by prompt id
        other.prompt: other.writer(),       # keyed by prompt text
        "P03": "this is not a chaord program",  # a real LLM can be wrong
    }
    path = tmp_path / "answers.json"
    path.write_text(json.dumps(answers), encoding="utf-8")
    result = suite.run_suite(writer="file", file=str(path))
    assert result.total == 50
    assert result.passed == 2                      # two answers, one compile each
    p03 = next(r for r in result.results if r.pid == "P03")
    assert not p03.ok and "parse" in p03.error
    missing = [r for r in result.results if "missing" in r.error]
    assert len(missing) == 47                      # unanswered prompts fail


def test_prompt_suite_cli_reports_rate(suite, capsys):
    code = suite.main(["--writer", "deterministic", "--case", "P01",
                       "--case", "P02"])
    out = capsys.readouterr().out
    assert code == 0
    assert "first-compile pass rate: 2/2 (100.0%)" in out


# ------------------------------------------------------- active-learning -----

CU_PROGRAM = """chaord 0.1
dialect core + metal

system {{
  cell {cell} {cell} {cell} A
  pbc xyz
  seed 7
  conserve atoms Cu {n}
}}

physics {{
  backend eam
}}

crystal bulk : all {{
  lattice fcc
  a 3.615 A
  assert cn 12.0 +- 0.5 cutoff 3.5 A
  assert sites_matched 100.0 %
}}

residual none
"""


def _cu_frame_and_program(n=32, cell=7.23, conserve=32):
    text = CU_PROGRAM.format(cell=cell, n=conserve)
    prog = parse_text(text)
    dialect = dialect_from_program(prog)
    frame = build_program(prog, dialect, np.random.default_rng(7))
    assert len(frame) == n
    return prog, frame, dialect


def test_uncertainty_report_fields():
    prog, frame, dialect = _cu_frame_and_program()
    report = uncertainty_report(prog, frame, dialect)
    # the documented report contract
    for field in ("frame_atoms", "asserts_total", "asserts_checked",
                  "asserts_failed", "asserts_unverified", "failures",
                  "unverified", "residual_atoms", "residual_fraction",
                  "stated_atoms", "present_atoms", "conservation_mismatch",
                  "sites_matched", "score", "reasons"):
        assert hasattr(report, field)
    assert report.frame_atoms == 32
    assert report.asserts_total == 2
    # cn is measurable locally and holds for a perfect crystal
    assert report.asserts_checked == 1
    assert report.asserts_failed == 0
    assert report.sites_matched == 1.0
    assert report.residual_atoms == 0
    assert report.stated_atoms == {"Cu": 32}
    assert report.present_atoms == {"Cu": 32}
    assert report.conservation_mismatch == 0
    assert 0.0 <= report.score <= 1.0
    assert report.reasons and isinstance(report.reasons, list)
    # JSON-ready for the active-learning bridge
    as_dict = report.to_dict()
    assert as_dict["frame_atoms"] == 32
    assert json.loads(json.dumps(as_dict)) == as_dict


def test_uncertainty_report_assert_failure_and_residual():
    # NOTE: a residual block with two `atom ...` statements does not re-parse
    # (pre-existing parser quirk: the specdef `atom` literal confuses the
    # contextual lexer after the first statement), so this program uses one.
    text = """chaord 0.1
dialect core + metal

system {
  cell 7.23 7.23 7.23 A
  pbc xyz
  seed 7
  conserve atoms Cu 30
}

physics {
  backend eam
}

crystal bulk : all {
  lattice fcc
  a 3.615 A
  assert cn 99.0 +- 0.5 cutoff 3.5 A
}

residual {
  atom Cu 1.0 2.0 3.0
}
"""
    prog, frame, dialect = _cu_frame_and_program()
    bad = parse_text(text)
    report = uncertainty_report(bad, frame, dialect)
    assert report.asserts_total == 1
    assert report.asserts_checked == 1
    assert report.asserts_failed == 1
    failure = report.failures[0]
    assert failure["key"] == "cn"
    assert failure["asserted"] == 99.0
    assert failure["measured"] == pytest.approx(12.0, abs=0.5)
    assert failure["tolerance"] == 0.5
    assert report.residual_atoms == 1
    assert report.residual_fraction == pytest.approx(1 / 32)
    # 2 atoms stated missing vs the frame
    assert report.conservation_mismatch == 2
    assert report.score > 0.25
    assert any("cn" in r for r in report.reasons)
    assert any("residual" in r for r in report.reasons)
    assert any("conservation" in r for r in report.reasons)


def test_uncertainty_report_molecular_census():
    text = """chaord 0.1
dialect core + molecular

system {
  cell 15 15 15 A
  pbc xyz
  seed 3
  state T 300 K
}

physics {
  backend classical
}

liquid water : all {
  molecules H2O 60
}
"""
    prog = parse_text(text)
    dialect = dialect_from_program(prog)
    frame = build_program(prog, dialect, np.random.default_rng(3),
                          physics=False)
    report = uncertainty_report(prog, frame, dialect)
    assert report.stated_atoms == {"H": 120, "O": 60}
    assert report.present_atoms == {"H": 120, "O": 60}
    assert report.conservation_mismatch == 0
    assert report.atom_census_known
    assert report.score == 0.0
    assert report.asserts_total == 0


def test_propose_frames_orders_by_uncertainty_and_bands():
    prog, frame, dialect = _cu_frame_and_program()
    good = uncertainty_report(prog, frame, dialect)

    text = CU_PROGRAM.format(cell=7.23, n=30).replace(
        "assert cn 12.0 +- 0.5 cutoff 3.5 A",
        "assert cn 99.0 +- 0.5 cutoff 3.5 A")
    bad_prog = parse_text(text)
    uncertain = uncertainty_report(bad_prog, frame, dialect)
    assert uncertain.score > good.score

    proposals = propose_frames(
        [("frame-good", prog, frame, dialect),
         ("frame-bad", bad_prog, frame, dialect),
         ("frame-cached", uncertain)],
        dialect)
    scores = [p.score for p in proposals]
    assert scores == sorted(scores, reverse=True)
    assert proposals[0].frame_id == "frame-bad"
    by_id = {p.frame_id: p for p in proposals}
    assert by_id["frame-bad"].action in (ACTION_LABEL, ACTION_ESCALATE)
    assert by_id["frame-cached"].action == by_id["frame-bad"].action
    assert by_id["frame-good"].action == ACTION_ACCEPT
    for p in proposals:
        assert p.action in (ACTION_LABEL, ACTION_ESCALATE, ACTION_ACCEPT)
        assert isinstance(p.reasons, list) and p.reasons
        # every proposal is JSON-ready for the bridge
        d = p.to_dict()
        assert json.loads(json.dumps(d)) == d
    assert propose_frames([("a", good)], dialect, top=0) == []


def test_propose_frames_labels_the_worst_cases():
    prog, frame, dialect = _cu_frame_and_program()
    worst = CU_PROGRAM.format(cell=7.23, n=8).replace(
        "assert cn 12.0 +- 0.5 cutoff 3.5 A",
        "assert cn 99.0 +- 0.5 cutoff 3.5 A"
    ).replace("residual none\n",
              "residual {\n  atom Cu 1.0 2.0 3.0\n}\n")
    bad_prog = parse_text(worst)
    report = uncertainty_report(bad_prog, frame, dialect)
    assert report.score >= 0.5   # failed assert + residual + big census gap
    proposals = propose_frames([("f0", prog, frame, dialect),
                                ("f1", bad_prog, frame, dialect)])
    assert proposals[0].frame_id == "f1"
    assert proposals[0].action == ACTION_LABEL
    assert proposals[0].label_worthy
    assert not proposals[0].escalate_worthy


# ------------------------------------------------------------- Laya codec ----

def _strip_provenance(text):
    """Independent normalizer: drop each provenance block and the blank
    separator line above it, keep everything else line for line."""
    out, skip = [], False
    for line in text.splitlines():
        s = line.strip()
        if skip:
            if s == "}":
                skip = False
            continue
        if s.startswith("provenance {"):
            if out and not out[-1].strip():
                out.pop()
            skip = True
            continue
        out.append(line)
    return "\n".join(out) + "\n" if out else ""


_SYS_POOL = ["  cell 7.23 7.23 7.23 A", "  cell 10.845 10.845 10.845 A",
             "  pbc xyz", "  seed 7", "  seed 11", "  conserve atoms Cu 32",
             "  state T 300 K"]
_PHYS_POOL = ["  backend eam", '  potential "Cu.eam"', "  epsilon 1",
              "  cutoff 2.5"]
_REGION_POOL = ["  lattice fcc", "  a 3.615 A", "  orient x [100] y [010] z [001]",
                "  defect V_Cu count 1", "  defect Cu_i count 2",
                "  occupancy Cu 1/2 Ni 1/2", "  assert sites_matched 99.2 %",
                "  assert cn 12.0 +- 0.5 cutoff 3.5 A"]
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def _random_program_lines(rng, n_system=4, n_region=4):
    lines = ["chaord 0.1", "dialect core + metal", "", "system {"]
    lines += list(rng.choice(_SYS_POOL, size=n_system, replace=False))
    lines += ["}", "", "physics {"]
    lines += list(rng.choice(_PHYS_POOL, size=2, replace=False))
    lines += ["}", "", "crystal bulk : all {"]
    lines += ["  lattice fcc", "  a 3.615 A"]
    lines += list(rng.choice(_REGION_POOL, size=n_region, replace=False))
    lines += ["}", "", "residual none"]
    return lines


def _mutate(lines, rng):
    out = list(lines)
    for _ in range(int(rng.integers(1, 4))):
        op = rng.random()
        body = [i for i, ln in enumerate(out)
                if ln.startswith("  ") and " {" not in ln and ln != "}"]
        if op < 0.35 and body:      # change a number token
            i = int(rng.choice(body))
            nums = list(_NUM_RE.finditer(out[i]))
            if nums:
                m = nums[int(rng.integers(len(nums)))]
                new = f"{float(m.group()) * (1 + 0.1 * rng.integers(1, 9)):.3f}"
                out[i] = out[i][:m.start()] + new + out[i][m.end():]
        elif op < 0.6 and body:     # delete a statement line
            out.pop(int(rng.choice(body)))
        elif op < 0.85:             # insert a statement line
            i = int(rng.choice([i for i, ln in enumerate(out)
                                if ln.startswith("  ")] or [7]))
            out.insert(i, str(rng.choice(_REGION_POOL)))
        elif len(out) > 12:         # swap two statement lines
            a = int(rng.choice(body))
            b = int(rng.choice(body))
            out[a], out[b] = out[b], out[a]
    return out


def _provenance(rng):
    return ["", "provenance {",
            f'  dialects "metal 0.{int(rng.integers(1, 9))} + core 0.1"',
            f'  lift_version "0.{int(rng.integers(1, 9))}.0"',
            "}"]


def test_laya_strict_round_trip_on_20_random_program_pairs():
    rng = np.random.default_rng(20260928)
    for k in range(20):
        old_lines = _random_program_lines(rng)
        new_lines = _mutate(old_lines, rng)
        old_text = "\n".join(old_lines) + "\n"
        new_text = "\n".join(new_lines) + "\n"
        if k % 3 == 0:      # provenance blocks appear and are allowed to differ
            old_text += "\n".join(_provenance(rng)) + "\n"
            new_text += "\n".join(_provenance(rng)) + "\n"
        patch = laya_encode(old_text, new_text)
        assert laya_decode(old_text, patch) == _strip_provenance(new_text), \
            f"pair {k} did not round trip"


def test_laya_provenance_only_transition_is_noop():
    rng = np.random.default_rng(3)
    base = "\n".join(_random_program_lines(rng)) + "\n"
    with_prov = base + "\n".join(_provenance(rng)) + "\n"
    assert laya_encode(base, with_prov) == "=0"
    assert laya_encode(with_prov, base) == "=0"


def test_laya_round_trip_from_empty_state():
    rng = np.random.default_rng(7)
    text = "\n".join(_random_program_lines(rng)) + "\n"
    patch = laya_encode(None, text)
    assert patch != "=0"
    assert laya_decode(None, patch) == _strip_provenance(text)


def test_laya_patch_format_carries_line_numbers_and_ops():
    old = "chaord 0.1\n\nsystem {\n  seed 7\n}\n\nresidual none\n"
    new = ("chaord 0.1\n\nsystem {\n  seed 8\n  state T 300 K\n}\n\n"
           "residual none\n")
    patch = laya_encode(old, new)
    head = patch.splitlines()[0]
    # control line: op char + '@' + old/new line ranges
    assert re.fullmatch(r"[-+]@ \d+ \d+ \d+ \d+", head)
    assert all(line[0] in "+-" for line in patch.splitlines())
    assert laya_tokens(patch) <= 320 and fits_budget(patch)
    assert fits_budget("=0") and laya_tokens("=0") == 1
    assert not fits_budget("x " * 400)


def test_laya_stale_patch_is_rejected():
    old = "chaord 0.1\n\nsystem {\n  seed 7\n}\n"
    other = "chaord 0.1\n\nsystem {\n  seed 9\n}\n"
    new = "chaord 0.1\n\nsystem {\n  seed 8\n}\n"
    patch = laya_encode(old, new)
    with pytest.raises(ChaordError):
        laya_decode(other, patch)     # token computed against a different state
    with pytest.raises(ChaordError):
        laya_decode(old, "+not a header")   # malformed patch
