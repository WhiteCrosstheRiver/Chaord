"""Top-level tests conftest.

O12a (Reviews 4-7) registry: strict xfail is allowed only for REGISTERED
open findings awaiting the owner's decision.  The registry lives HERE
(top level) so its collection hook sees every test in the tree -- a
subdirectory conftest's hook only sees its own subtree, which is why
this moved out of tests/adversarial/conftest.py on 2026-10-09.
Registration classes: red-team findings (a criterion currently fooled,
reports/redteam_findings.md) and recorded open items from a review work
order awaiting the reviewer's sign-off (the last one was the W5
strained-box identity gap, closed by X2 on 2026-10-14; tests/
test_strained_boxes.py _GAP is the readable mirror machinery, kept in
sync with this registry by test_gap_set_matches_registry).  Strictness is
the enforcement: the moment a registered test passes, the XPASS fails the
suite and the id must come out -- a fix cannot silently bypass its own
registered failure.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)


# Strained-box identity gap (registered 2026-10-09, CLOSED by X2 on
# 2026-10-14): the 94 matrix configs of tests/test_strained_boxes.py
# whose first lift (defect arm, scan-fitted constant) and re-lift
# (spglib arm, idealised constant) disagreed on the a/c line were
# registered here per O12a; with the D17 lift-side snap
# (box_lattice_constant: every arm snaps its fitted constant to the
# constants the STATED cell implies, and the builder accepts strained
# cells by the same function) all 94 XPASSed, the strict markers turned
# them red, and the ids came out -- the designed forcing function. The
# registry stays empty until a new registered finding needs it; the
# O12a machinery (strict xfail only here, applied at collection)
# remains in force.
FOILED = {}


def pytest_collection_modifyitems(items):
    for item in items:
        if item.nodeid in FOILED:
            reason = FOILED[item.nodeid]
            if not isinstance(reason, str):
                reason = ("red team: criterion currently fooled "
                          "(reports/redteam_findings.md); remove this id "
                          "when the fix lands")
            item.add_marker(pytest.mark.xfail(strict=True, reason=reason))
