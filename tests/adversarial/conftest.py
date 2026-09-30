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
FOILED = {
    "tests/adversarial/test_a05_temperature_and_averaging.py::test_a5_averaged_gate_would_separate_wrong_temperature",
    "tests/adversarial/test_a05_temperature_and_averaging.py::test_a5_reference_frame_choice_moves_distance_less_than_floor",
    "tests/adversarial/test_a05_temperature_and_averaging.py::test_a5_wrong_temperature_live_canary",
    "tests/adversarial/test_a05_temperature_and_averaging.py::test_a5_wrong_temperature_rebuild_must_fail_the_gate",
    "tests/adversarial/test_a06_a13_hcp_misroute.py::test_hcp_ground_truth_lift_mode_is_honored",
    "tests/adversarial/test_a06_conservation_blindspots.py::test_a06_acceptance_charge_arithmetic_covers_multivalent_elements",
    "tests/adversarial/test_a06_conservation_blindspots.py::test_a6_slab_region_arithmetic_is_audited",
    "tests/adversarial/test_a07_judged_core_scope.py::test_a07_targeted_label_corruption_is_caught",
    "tests/adversarial/test_a09_evidence_class.py::test_a09_gated_set_includes_a_heterogeneous_case",
    "tests/adversarial/test_a12_a10_a14_statics.py::test_a12_lj_overlap_tolerance_covers_the_sanity_hard_core",
    "tests/adversarial/test_a14_prose_coverage.py::test_a14_prose_only_reference_is_rejected",
    "tests/adversarial/test_power_mutations.py::test_a4_mixed_cell_mutation",
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
