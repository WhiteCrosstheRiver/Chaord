"""Mutation canaries for the A1-A14 acceptance checks (tools/acceptance.py).

Every criterion gets one seeded mutation injected into the checked input; the
acceptance function must return passed=False on the mutated input.  Where the
full-scope clean run is an honest FAIL (A2/A3 on the random-alloy case, A5 with
no noise floor, A9 with no >= 1,000-atom bench frame, A13 on refused frames,
A14 on uncovered keys), the canary first runs a *scoped* clean configuration
that passes, then the same configuration with the mutation -- so a PASS->FAIL
flip is demonstrated, not just a persistent FAIL.

The full-scope acceptance run (the honest per-criterion evidence) is
`python tools/acceptance.py`; these tests only guard the checker itself.
"""
import pytest

from tools import acceptance as acc


# --------------------------------------------------------------------- A1 ----
def test_a1_fmt_idempotence_and_parse():
    clean = acc.check_a1(max_examples=25)
    assert clean["passed"], clean["evidence"]
    bad = acc.check_a1(mutation="malformed_example", max_examples=25)
    assert not bad["passed"], bad["evidence"]
    assert any("parse failures" in s for s in [bad["evidence"]])


# --------------------------------------------------------------------- A2 ----
def test_a2_canonical_invariance():
    clean = acc.check_a2(case_filter="fcc_cu")
    assert clean["passed"], clean["evidence"]
    bad = acc.check_a2(mutation="scale_lattice", case_filter="fcc_cu")
    assert not bad["passed"], bad["evidence"]


# --------------------------------------------------------------------- A3 ----
def test_a3_round_trip_and_structure_matcher():
    clean = acc.check_a3(case_filter="fcc_cu")
    assert clean["passed"], clean["evidence"]
    bad = acc.check_a3(mutation="displace_rebuilt", case_filter="fcc_cu")
    assert not bad["passed"], bad["evidence"]


def test_a3_random_solution_species_canaries():
    """Random solid solutions are matched species-blind (the labeling is a
    microstate); the species arrangement is judged by Warren-Cowley alphas
    against the relabeling noise floor (human approval 2026-09-29).

    segregate: sites and composition stay right, so the failure must come
    from the species statistic itself."""
    clean = acc.check_a3(case_filter="crconi")
    assert clean["passed"], clean["evidence"]

    bad = acc.check_a3(mutation="displace_rebuilt", case_filter="crconi")
    assert not bad["passed"], bad["evidence"]

    bad = acc.check_a3(mutation="segregate", case_filter="crconi")
    assert not bad["passed"], bad["evidence"]
    row = bad["details"]["rows"][0]
    assert "EXCEEDS 1.5x floor" in row["note"], row["note"]
    assert not row["geometry"], "species check must report the segregation"

    bad = acc.check_a3(mutation="composition", case_filter="crconi")
    assert not bad["passed"], bad["evidence"]


def test_a3_ordered_case_forced_to_occupancy_still_fails():
    """An ordered L1_2 program rewritten as random occupancy rebuilds with a
    random species arrangement: the species-aware matcher must reject it."""
    bad = acc.check_a3(mutation="force_occupancy", case_filter="l12_ni3al")
    assert not bad["passed"], bad["evidence"]
    row = bad["details"]["rows"][0]
    assert not row["geometry"], "ordered cases keep the species-aware matcher"


# --------------------------------------------------------------------- A4 ----
def test_a4_defect_precision_and_recall():
    clean = acc.check_a4(temps=("room",))
    assert clean["passed"], clean["evidence"]
    bad = acc.check_a4(mutation="false_defect", temps=("room",))
    assert not bad["passed"]
    # one false vacancy among 6 planted: precision 6/7 = 0.857 < 0.95
    worst = min(bad["details"]["matrix"], key=lambda m: m["precision"])
    assert worst["precision"] < 0.95 and worst["fp"] >= 1


# --------------------------------------------------------------------- A5 ----
def test_a5_noise_floor_gate():
    # scoped to a synthetic case with NO floor: it must be skipped, and a
    # fabricated generous floor must make it pass; a distorted rebuild fail
    base = acc.check_a5(case_filter="water_box15")
    row = next(r for r in base["details"]["rows"]
               if r["case"] == "fluid/water_box15")
    assert row["status"].startswith("no-floor")      # no floor on record
    floor = {k: 2.0 * v for k, v in row["distance"].items()}
    floors = {"fluid/water_box15": floor}
    clean = acc.check_a5(case_filter="water_box15", floors=floors)
    assert clean["passed"], clean["evidence"]
    bad = acc.check_a5(mutation="distort_rebuild", case_filter="water_box15",
                       floors=floors)
    assert not bad["passed"], bad["evidence"]


def test_a5_physics_off_rebuild_fails():
    """Review-2 mutation: rebuilding lj_liquid with physics=False (the packing
    prior alone, no MD relaxation) must FAIL A5, while the physics rebuild of
    the same scoped case passes.  Done when: the physics-off mutation fails;
    lj_liquid is within 1.5x the floor on both observables."""
    clean = acc.check_a5(case_filter="lj_liquid")
    row = next(r for r in clean["details"]["rows"]
               if r["case"] == "reference/lj_liquid")
    assert row["status"] == "pass", row["note"]
    assert row["backend"] == "lj"
    assert row["temperature"] == 0.65          # program `state T`
    assert row["md_steps"] > 0                 # MD relaxation ran (physics on)
    # both observables within 1.5x the floor
    assert "cn_tv" in row["note"] and "gr_rms" in row["note"]
    assert "MUTATED" not in row["note"]

    bad = acc.check_a5(mutation="physics_off", case_filter="lj_liquid")
    row = next(r for r in bad["details"]["rows"]
               if r["case"] == "reference/lj_liquid")
    assert row["status"] == "fail", row["note"]
    assert row["md_steps"] == 0                # the MD prior really dropped
    assert "MUTATED(physics-off rebuild)" in row["note"]
    assert not bad["passed"], bad["evidence"]


# --------------------------------------------------------------------- A6 ----
def test_a6_three_way_conservation():
    clean = acc.check_a6(frame_limit=6)
    assert clean["passed"], clean["evidence"]
    bad = acc.check_a6(mutation="drop_atom", frame_limit=6)
    assert not bad["passed"], bad["evidence"]


# --------------------------------------------------------------------- A7 ----
def test_a7_phase_labels():
    clean = acc.check_a7(frames=(0,))
    assert clean["passed"], clean["evidence"]
    bad = acc.check_a7(mutation="flip_labels", frames=(0,))
    assert not bad["passed"], bad["evidence"]


# --------------------------------------------------------------------- A8 ----
def test_a8_reactive_census_independent_construction():
    clean = acc.check_a8()
    assert clean["passed"], clean["evidence"]
    bad = acc.check_a8(mutation="extra_oh")
    assert not bad["passed"], bad["evidence"]


# --------------------------------------------------------------------- A9 ----
def test_a9_compression_gate():
    case = dict(case="synthetic-2000", n_atoms=2000,
                prog_bytes=10_000, xyz_bytes=1_000_000)
    clean = acc.check_a9(cases_override=[case])
    assert clean["passed"], clean["evidence"]
    bad = acc.check_a9(mutation="inflate_program", cases_override=[case])
    assert not bad["passed"], bad["evidence"]


# -------------------------------------------------------------------- A10 ----
def test_a10_determinism():
    clean = acc.check_a10()
    assert clean["passed"], clean["evidence"]
    bad = acc.check_a10(mutation="perturb_build")
    assert not bad["passed"], bad["evidence"]


# -------------------------------------------------------------------- A11 ----
def test_a11_speed_limit():
    clean = acc.check_a11()
    assert clean["passed"], clean["evidence"]
    bad = acc.check_a11(elapsed_override=120.5)
    assert not bad["passed"], bad["evidence"]


# -------------------------------------------------------------------- A12 ----
def test_a12_all_four_seed_errors_caught():
    clean = acc.check_a12()
    assert clean["passed"], clean["evidence"]


@pytest.mark.parametrize("stub", ["stub_overlap", "stub_charge"])
def test_a12_missing_checker_stub_flips_verdict(stub):
    bad = acc.check_a12(mutation=stub)
    assert not bad["passed"], bad["evidence"]
    assert "uncaught" in bad["evidence"]


# -------------------------------------------------------------------- A13 ----
def test_a13_no_crashes():
    clean = acc.check_a13(frame_limit=3)
    assert clean["passed"], clean["evidence"]
    bad = acc.check_a13(mutation="poison_frame", frame_limit=3)
    assert not bad["passed"], bad["evidence"]


# -------------------------------------------------------------------- A14 ----
def test_a14_reference_coverage():
    base = acc.check_a14()
    keys = base["details"]["keys"]
    assert keys, "no dialect keys found"
    full_reference = " ".join(keys)       # a reference that covers every key
    clean = acc.check_a14(reference_text=full_reference)
    assert clean["passed"], clean["evidence"]
    bad = acc.check_a14(mutation="hide_key", reference_text=full_reference)
    assert not bad["passed"], bad["evidence"]
    assert "epsilon" in bad["details"]["missing"]


# ------------------------------------------------------------- runner shape --
def test_runner_report_schema():
    results = acc.run_all(only=["A8"])
    assert len(results) == 1
    r = results[0]
    assert set(r) >= {"id", "name", "passed", "evidence", "details"}
    assert r["id"] == "A8" and isinstance(r["passed"], bool)
