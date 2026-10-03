"""O12a (Reviews 4-7): strict xfail is allowed only for registered red-team
findings awaiting the owner's decision.

The FOILED registry in tests/adversarial/conftest.py is the single sanctioned
home for strict xfails: each id is an open finding the owner has not yet
decided. A strict xfail anywhere else is a silently-expected failure with no
registered finding behind it -- this test forbids that."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = ROOT / "tests" / "adversarial" / "conftest.py"


def test_strict_xfail_only_in_red_team_registry():
    offenders = []
    for p in sorted((ROOT / "tests").rglob("test_*.py")):
        text = p.read_text(encoding="utf-8")
        for m in re.finditer(
                r"(xfail\s*\(\s*(?:[^)]*)?strict\s*=\s*True|"
                r"xfail\s*\(\s*strict\s*,)", text, re.S):
            offenders.append(str(p.relative_to(ROOT)))
            break
        # the registry's own marker application lives in conftest, not here
    assert not offenders, (
        f"strict xfail outside the red-team registry: {sorted(set(offenders))} "
        "-- register the finding in tests/adversarial/conftest.py FOILED or "
        "fix the test")


def test_registry_ids_reference_existing_tests():
    """Every FOILED id must name a real test (a stale id silently stops
    guarding anything)."""
    sys_path = ROOT / "tests" / "adversarial" / "conftest.py"
    text = sys_path.read_text(encoding="utf-8")
    ids = set(re.findall(r'"(tests/adversarial/[^"]+)"', text))
    assert ids, "registry empty"
    for nid in ids:
        path, _, name = nid.partition("::")
        assert (ROOT / path).exists(), f"{nid}: file missing"
        assert name in (ROOT / path).read_text(encoding="utf-8"), \
            f"{nid}: test name not found in file"
