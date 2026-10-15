"""Generate the acceptance-results section of reports/gate_a_prime.md from
the committed artifacts (D22, Review 9: the criteria table is GENERATED, so
it cannot go stale against a newer acceptance run).

Sources, read-only:
  reports/acceptance.json       the clean-runner artifact (per-criterion
                                verdicts + evidence + run provenance)
  reports/acceptance_details.md the per-case tables the runner writes

The section is written between the BEGIN/END markers in reports/gate_a_prime.md
and must only ever be changed by this script (tests can pin that with --check).

Usage:
  python tools/gen_gate_report.py            regenerate the section in place
  python tools/gen_gate_report.py --check    exit 1 with a diff if the
                                             committed section is stale
"""
import difflib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "gate_a_prime.md"
ACCEPT = ROOT / "reports" / "acceptance.json"
DETAILS = ROOT / "reports" / "acceptance_details.md"

BEGIN = "<!-- BEGIN GENERATED acceptance results: tools/gen_gate_report.py, do not edit by hand -->"
END = "<!-- END GENERATED acceptance results -->"

# Evidence clauses are '; '-separated. The report table carries the first
# clauses up to this budget; the full evidence stays in the artifacts.
EVIDENCE_BUDGET = 200
PAREN_STRIP_LEN = 120          # a clause longer than this loses its asides


def _short_evidence(text: str) -> str:
    """First '; '-clauses of an evidence string within the budget."""
    out = []
    for clause in text.split("; "):
        if len(clause) > PAREN_STRIP_LEN:
            clause = re.sub(r"\s*\([^()]*\)", "", clause)
        if out and len("; ".join(out)) + 2 + len(clause) > EVIDENCE_BUDGET:
            break
        if not out:
            out.append(clause)
        elif len(clause) <= EVIDENCE_BUDGET - len("; ".join(out)) - 2:
            out.append(clause)
        else:
            break
    joined = "; ".join(out).strip()
    if len(text) > len(joined):
        joined += "; ..."
    return joined


def _criteria_table(acc: dict) -> str:
    lines = ["| criterion | verdict | key evidence (first clauses; full "
             "evidence in acceptance_details.md) |",
             "| --- | --- | --- |"]
    for c in acc["criteria"]:
        verdict = "**PASS**" if c["passed"] else "**FAIL**"
        ev = _short_evidence(c["evidence"]).replace("|", "\\|")
        lines.append(f"| {c['id']} {c['name']} | {verdict} | {ev} |")
    return "\n".join(lines)


def _a5_table(details: str) -> str:
    """The per-case A5 table from the runner-written details file."""
    section = re.search(r"^## A5 .*?(?=^## )", details, re.M | re.S)
    if not section:
        raise SystemExit("gen_gate_report: no '## A5' section in "
                         f"{DETAILS.relative_to(ROOT)}")
    rows = []
    for line in section.group(0).splitlines():
        if not line.startswith("|") or set(line) <= {"|", " ", "-"}:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells[0] in ("case", ""):
            continue
        case, category, status, backend = cells[0], cells[1], cells[2], cells[4]
        ratios = {}
        for key in ("cn_tv", "gr_rms"):
            m = re.search(key + r" [\d.]+ vs floor [\d.]+ \(x([\d.]+)",
                          status)
            if m:
                ratios[key] = "x" + m.group(1)
        if ratios:
            cells_out = (f"| {case} | {category} | "
                         f"{ratios.get('cn_tv', '-')} | "
                         f"{ratios.get('gr_rms', '-')} | {backend} |")
        else:
            cells_out = (f"| {case} | {category} | no floor on record "
                         f"(skipped, honestly) | - | {backend} |")
        rows.append(cells_out)
    head = ("| case | category | cn_tv xfloor | gr_rms xfloor | backend |\n"
            "| --- | --- | --- | --- | --- |")
    return head + "\n" + "\n".join(rows)


def generate() -> str:
    if not ACCEPT.exists():
        raise SystemExit(f"gen_gate_report: {ACCEPT.relative_to(ROOT)} "
                         "is missing")
    acc = json.loads(ACCEPT.read_text(encoding="utf-8"))
    prov = acc.get("provenance", {})
    if prov.get("run_kind") != "clean-runner":
        raise SystemExit(
            "gen_gate_report: refusing to generate the Gate A' table from a "
            f"non-clean-runner artifact (run_kind={prov.get('run_kind')!r}); "
            "the gate report cites the clean-runner run only")
    details = DETAILS.read_text(encoding="utf-8")
    sha = prov.get("commit", "?")[:7]
    url = prov.get("run_url", "?")
    when = prov.get("timestamp", "?")
    banner = (
        f"Artifact: clean-runner run "
        f"[{url.rsplit('/', 1)[-1]}]({url}) on `{sha}` ({when}) -- "
        f"**{acc['passed']}/{acc['total']} criteria PASS**; the runner's "
        f"OS was `{prov.get('runner', '?')}`, working tree "
        f"{'clean' if not prov.get('working_tree_dirty') else 'DIRTY'}. "
        "The table below is generated from this artifact "
        "(`tools/gen_gate_report.py`); the full per-case tables are in "
        "[acceptance_details.md](acceptance_details.md).")
    return "\n".join([BEGIN, "", banner, "", "### Criteria", "",
                      _criteria_table(acc), "",
                      "### A5 statistical round trip, per case "
                      "(x floor; gate 1.5x)", "", _a5_table(details), "",
                      END])


def _marker_span(lines):
    """Line-index span (begin, end) of the generated section, None if absent."""
    begins = [i for i, l in enumerate(lines) if l.strip() == BEGIN]
    ends = [i for i, l in enumerate(lines) if l.strip() == END]
    if len(begins) != 1 or len(ends) != 1 or ends[0] < begins[0]:
        return None
    return begins[0], ends[0]


def main() -> int:
    mode = "check" if "--check" in sys.argv[1:] else "write"
    fresh = generate().splitlines()
    lines = REPORT.read_text(encoding="utf-8").splitlines()
    span = _marker_span(lines)
    current = lines[span[0]:span[1] + 1] if span else None
    if mode == "check":
        if current == fresh:
            print("gen_gate_report: committed section is current")
            return 0
        print("gen_gate_report: STALE -- the acceptance-results section of "
              f"{REPORT.relative_to(ROOT)} does not match the artifacts; "
              "re-run the script")
        for line in difflib.unified_diff(current or [], fresh, "committed",
                                         "regenerated", lineterm=""):
            print(line)
        return 1
    if span is None:
        raise SystemExit(
            f"gen_gate_report: markers not found in {REPORT.relative_to(ROOT)}"
            " -- the report must contain the BEGIN/END section")
    lines[span[0]:span[1] + 1] = fresh
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8",
                      newline="\n")
    n_crit = sum(1 for l in fresh if l.startswith("| A"))
    print(f"gen_gate_report: wrote {n_crit} criteria rows + the A5 per-case "
          f"table into {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
