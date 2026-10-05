"""Red-team (T0) adversarial suite: inputs that make a criterion PASS while
the result is wrong.  Each test states which criterion it attacks, carries the
measured numbers in its assertion messages, and is either RED (the defence is
currently fooled; the test fails until the gap is fixed) or GREEN (the defence
holds; a regression asset).  See reports/redteam_findings.md for the full
write-up with every number's provenance."""
import sys

import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)


# ---- post Review 3 registry (2026-09-30) ------------------------------------
# The tests below currently FAIL: each is an input where a criterion passes
# while the result is wrong (reports/redteam_findings.md). Registered as
# xfail(strict=True): the suite stays green while the finding is open, and
# the moment a fix lands the test XPASSes, CI goes red on the strict marker,
# and the id is removed here -- a fix cannot silently bypass its own
# adversarial test.
#
# Closed on 2026-09-30 (acceptance-side verification stream; numbers in
# reports/redteam_findings.md): F12 (A4 mixed-cell metric + mixed cells in
# the plan), F4 (A6 slab region arithmetic), F9 (A7 core scope disclosure +
# gate), F11 (A9 heterogeneous gated set), F2's lift_mode half (bench loop
# honours the recorded mode), F8 (A14 structured coverage), F5's O half
# (verifier charge tables ION_CHARGES ∪ OXIDE_ELEMENTS), F3/A5 (averaged
# protocol on lj_liquid_large: wrong-T power mutations fail the gate).
FOILED = {
}


def pytest_collection_modifyitems(items):
    for item in items:
        if item.nodeid in FOILED:
            item.add_marker(
                pytest.mark.xfail(
                    strict=True,
                    reason="red team: criterion currently fooled "
                           "(reports/redteam_findings.md); remove this id "
                           "when the fix lands"))
