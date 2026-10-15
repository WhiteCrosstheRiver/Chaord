"""First-result evidence pins (open_items_v2.md section 5, item 3).

The slow test runs the full first-result pipeline end-to-end on the
committed research data (4 systems x 21 production frames, ~40 s) and pins
the headline numbers of research/cnt_electrolyte/RESULTS.md: the P1
distributions, the noise floors, the cross-state separation multiples and
the statement-only classifier accuracies.  A number that drifts fails the
test; a number that was tuned would have shown up here.

The fast tests pin the classifier's anti-circularity properties without
touching the data: the statement features can never see the `state field`
line or the provenance notes, and the absent-P1 encoding is the a-priori
zero of an unaligned liquid.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src"), str(ROOT / "research" / "cnt_electrolyte")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import analyze_first_result as A                     # noqa: E402

# RESULTS.md tables (pipeline run on the committed runs/ data, 2026-10-14).
PIN_P1 = {  # system -> (mean, std) of the 21-frame P1 distribution
    "cnt10_field0": (-0.1207, 0.0759),
    "cnt10_field1": (0.7247, 0.0777),
    "cnt16_field0": (0.1521, 0.0299),
    "cnt16_field1": (0.6060, 0.0532),
}
PIN_FLOOR = {  # system -> max(mean, P90) over the 210 within-system pairs
    "cnt10_field0": 0.1762,
    "cnt10_field1": 0.1733,
    "cnt16_field0": 0.0708,
    "cnt16_field1": 0.1291,
}
PIN_SEPARATION = {  # diameter -> (cross mean, cross worst pair)
    "cnt10": (0.845, 0.580),
    "cnt16": (0.454, 0.269),
}
PIN_COMPOSITION = {  # system -> provenance composition (verified per frame)
    "cnt10_field0": {"EC": 8, "EMC": 5, "Li": 2, "PF6": 2},
    "cnt10_field1": {"EC": 8, "EMC": 5, "Li": 2, "PF6": 2},
    "cnt16_field0": {"EC": 37, "EMC": 24, "Li": 6, "PF6": 6},
    "cnt16_field1": {"EC": 37, "EMC": 24, "Li": 6, "PF6": 6},
}
PIN_CHIRALITY = {"cnt10": (10, 10), "cnt16": (16, 16)}
TOL = 1e-3


# --------------------------------------------------------------- fast pins --

def test_statement_features_cannot_see_the_field_line_or_notes():
    """Anti-circularity: the classifier's features are blind to the field
    statement and the provenance notes; only assert/state numeric
    statements of the region and system blocks are read."""
    a = ("system {\n  state T 300.00\n  state field E 0.2 V/A along z\n"
         "  assert dipole_p1 0.800 +- 0.277 along z\n}\n"
         'provenance {\n  note "field assumed (caller-supplied E = 0.2 V/A'
         ' along z); not measured"\n}\n')
    b = a.replace("0.2 V/A", "9.9 V/A")
    assert A.statement_features(a) == A.statement_features(b)
    feats = A.statement_features(a)
    assert feats["p1"] == 0.800
    assert "0.2" not in str(feats.values()) and "9.9" not in str(feats.values())


def test_absent_p1_encodes_as_the_apriori_zero():
    records = [{"field": False, "feats": {"p1": None}},
               {"field": True, "feats": {"p1": 0.8}}]
    values, labels = A.encode_features(records)
    assert values.tolist() == [0.0, 0.8]
    assert labels.tolist() == [0, 1]


def test_threshold_fit_is_the_midpoint_of_class_means():
    v = np.array([0.0, 0.1, 0.6, 0.8])
    y = np.array([0, 0, 1, 1])
    assert A.threshold_fit(v, y) == pytest.approx(0.375)


# -------------------------------------------------------------- slow pins --

@pytest.mark.slow
def test_first_result_pins_the_headline_numbers():
    result = A.run_pipeline()
    s = result["summary"]

    # -- inputs verified against provenance, chirality recovered exactly
    for system, want in PIN_COMPOSITION.items():
        d = s["systems"][system]
        assert d["verification"]["composition"] == want, system
        assert d["verification"]["n_frames"] == 21, system
        assert d["verification"]["census_frame_stable"], system
        assert d["chirality"] == list(PIN_CHIRALITY[system[:5]]), system
        assert d["chirality_matches_provenance"], system
        assert d["max_tube_drift_A"] < 1e-3, system

    # -- P1 distributions (mean +- std, RESULTS.md section 3)
    for system, (mean, std) in PIN_P1.items():
        d = s["systems"][system]
        assert abs(d["p1_mean"] - mean) <= TOL, (system, d["p1_mean"])
        assert abs(d["p1_std"] - std) <= TOL, (system, d["p1_std"])
    # the states' P1 ranges do not overlap in either diameter
    for diam in ("cnt10", "cnt16"):
        off = s["systems"][f"{diam}_field0"]
        on = s["systems"][f"{diam}_field1"]
        assert off["p1_max"] < on["p1_min"], (diam, off["p1_max"], on["p1_min"])

    # -- noise floors (D15 estimator: max(mean, P90), 210 pairs per system)
    for system, floor in PIN_FLOOR.items():
        d = s["systems"][system]["p1_floor"]
        assert d["n_pairs"] == 210, system
        assert d["floor"] == pytest.approx(max(d["mean"], d["p90"]))
        assert abs(d["floor"] - floor) <= TOL, (system, d["floor"])

    # -- separation beyond the floors (RESULTS.md section 4)
    for diam, (cross_mean, cross_worst) in PIN_SEPARATION.items():
        d = s["separation"][diam]
        assert d["n_cross_pairs"] == 441, diam
        assert abs(d["cross_mean"] - cross_mean) <= TOL, (diam, d["cross_mean"])
        assert abs(d["cross_min"] - cross_worst) <= TOL, (diam, d["cross_min"])
        assert d["multiple_worst_pair"] > 2.0, (diam, d["multiple_worst_pair"])

    # -- the classifier (>= 95% is the reviewer's bar; pinned at 100%)
    c = s["classifier"]
    assert c["n_frames"] == 84
    assert c["loo_accuracy"] >= 0.95
    assert c["loo_accuracy"] == 1.0
    assert c["cross_diameter"]["train_cnt10_test_cnt16"] >= 0.95
    assert c["cross_diameter"]["train_cnt16_test_cnt10"] >= 0.95
    assert c["per_diameter"]["cnt10"]["acc"] == 1.0
    assert c["per_diameter"]["cnt16"]["acc"] == 1.0
    # the honest negative: the shared statements do NOT separate (chance)
    ctl = c["control_shared_statements_only"]
    assert ctl["loo_accuracy"] <= 0.70, ctl
    assert 0.40 <= ctl["train_cnt10_test_cnt16"] <= 0.65, ctl

    # -- static checks pass after every composed lift
    assert s["failed_static_checks"] == []

    # -- statement-level sanity of all 84 programs
    from chaord.lang.fmt import format_program
    from chaord.lang.parser import parse_text
    units = sorted(A.load_dialect(("core", "molecular", "carbon", "research")).units
                   | {"V/A", "A", "g/cm3", "K"})
    for r in result["records"]:
        text = r["text"]
        if r["field"]:
            assert "state field E 0.2 V/A along z" in text
        else:
            assert "field" not in text, text[:200]
        # D29 (blind lift): every program carries the measured assert,
        # field or no field -- the classifier feature is always present
        assert "assert dipole_p1" in text
        # every composed program parses back and is its own canonical form
        # (rule 3: one structure, one text)
        program = parse_text(text, units=units)
        assert format_program(program) == text, r["system"]
        # the classifier feature agrees with the text it came from
        feats = A.statement_features(text)
        assert feats["p1"] is not None
