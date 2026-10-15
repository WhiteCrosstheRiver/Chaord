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
    # D21 (X4): 20 draws per case, stated in the evidence.  Red on the old
    # implementation: details n_transforms == 2 and the evidence read
    # "2 per case" -- two draws judged canonical invariance (rule 2 breach).
    assert clean["details"]["n_transforms"] == 20
    assert "20 draws per case, D21" in clean["evidence"], clean["evidence"]
    assert "20/20 rigid transforms" in clean["evidence"], clean["evidence"]
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
    lj_liquid is within 1.5x the floor on both observables.  Review 3
    (2026-09-30): the lift states the reference PROVENANCE temperature
    (T* = 0.72, from units.temperatures_K.equilibrium 8355.256 K and
    epsilon = 1 eV), not the dialect default 0.65; the verdict is the
    averaged protocol (>= 5 reference frames, 3 rebuild draws)."""
    clean = acc.check_a5(case_filter="lj_liquid")
    row = next(r for r in clean["details"]["rows"]
               if r["case"] == "reference/lj_liquid")
    assert row["status"] == "pass", row["note"]
    assert row["backend"] == "lj"
    assert row["temperature"] == pytest.approx(0.72)   # provenance T*, not the 0.65 default
    assert row["md_steps"] > 0                 # MD relaxation ran (physics on)
    assert row["seeds"] == [7, 13, 29]         # 3-draw averaged protocol
    assert len(row["ref_frames"]) >= 5         # >= 5-frame reference average
    # both observables within 1.5x the averaged floor
    assert "cn_tv" in row["note"] and "gr_rms" in row["note"]
    assert "MUTATED" not in row["note"]

    bad = acc.check_a5(mutation="physics_off", case_filter="lj_liquid")
    row = next(r for r in bad["details"]["rows"]
               if r["case"] == "reference/lj_liquid")
    assert row["status"] == "fail", row["note"]
    assert row["md_steps"] == 0                # the MD prior really dropped
    assert "MUTATED(physics-off rebuild)" in row["note"]
    assert not bad["passed"], bad["evidence"]


def test_a5_temperature_power_mutations_fail():
    """Review-3 / red-team F3 power mutations: a physically wrong but
    otherwise perfect rebuild -- the program's `state T` rewritten to 0.8x /
    1.25x of the reference PROVENANCE temperature -- must FAIL A5, while the
    correct-temperature rebuild of the same case passes.  Judged on
    lj_liquid_large (2,048 atoms, 10 reference frames): with the averaged
    protocol the wrong-temperature systematic no longer hides inside the
    single-frame microstate noise (measured 2026-09-30, this machine:
    correct T* = 0.72 -> gr x0.7 floor; x0.8 -> gr x2.4, cn x1.9 -> both
    over the 1.5x gate; the 500-atom lj_liquid case has no robust power --
    x0.8 passes at x1.1 the floor, x1.25 sits at x1.6 with a 5% gate margin
    smaller than the documented ISA/BLAS draw divergence).  Runtime ~10 min:
    three scoped runs x three rebuilds of a 2,048-atom system."""
    clean = acc.check_a5(case_filter="lj_liquid_large")
    row = next(r for r in clean["details"]["rows"]
               if r["case"] == "reference/lj_liquid_large")
    assert row["status"] == "pass", row["note"]
    assert row["temperature"] == pytest.approx(0.72)   # provenance T*
    assert row["seeds"] == [7, 13, 29]
    assert len(row["ref_frames"]) >= 5

    for mutation, tag in (("temp_lo", "MUTATED(T x0.8)"),
                          ("temp_hi", "MUTATED(T x1.25)")):
        bad = acc.check_a5(mutation=mutation, case_filter="lj_liquid_large")
        row = next(r for r in bad["details"]["rows"]
                   if r["case"] == "reference/lj_liquid_large")
        assert row["status"] == "fail", row["note"]
        assert tag in row["note"], row["note"]
        assert row["temperature"] != pytest.approx(0.72)  # the T really moved
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


def test_a7_core_scope_gate():
    """Red team F9: the judged core must not silently shrink.  Every clean
    row reports its judged fraction, and the criterion FAILS when a frame's
    judged fraction drops below 90% of what its interface geometry leaves
    available (the widen_band seeded fault halves the available core)."""
    clean = acc.check_a7(frames=(0,))
    for row in clean["details"]["rows"]:
        assert "judged_frac" in row and "geo_floor" in row
        assert row["judged_frac"] >= 0.9 * row["geo_floor"]
    assert "judged fraction disclosed per frame" in clean["evidence"]
    bad = acc.check_a7(mutation="widen_band", frames=(0,))
    assert not bad["passed"], bad["evidence"]
    assert "SCOPE VIOLATIONS" in bad["evidence"]


# --------------------------------------------------------------------- A8 ----
def test_a8_reactive_census_independent_construction():
    # scoped fast lane (1 draw x 1 translation, 1 clean draw): the criterion
    # NAMES its reduced scope in the evidence; the full 20 x 20 + 20 evidence
    # is the nightly acceptance run's (pinned by the slow test below)
    clean = acc.check_a8(site_draws=1, site_translations=1, clean_draws=1)
    assert clean["passed"], clean["evidence"]
    assert "1 draws x 1 random translations (1 frames" in clean["evidence"]
    assert "clean-slab false positives 0 of 6" in clean["evidence"]
    bad = acc.check_a8(mutation="extra_oh", site_draws=1, site_translations=1,
                       clean_draws=1)
    assert not bad["passed"], bad["evidence"]


def test_a8_adsorption_sites_power_mutation():
    """PLAN A8's second half (adsorption sites >= 90% correct), scoped to a
    named fast lane (2 draws x 2 translations): the clean draws read 100%
    (5 top + 3 bridge claimed exactly per frame); shifting the top-site
    adsorbates laterally by 1 A -- the planted-fault stand-in for a
    mis-assigned site -- moves every shifted O onto the nearest bridge
    feature (claims 8 bridge vs planted 5 top + 3 bridge; with the O >=
    2 a_NN apart, a 1 A shift can no longer land on a neighbouring top) and
    the criterion FAILs.  The full 20 x 20 scope is pinned by the slow
    full-scope test below."""
    clean = acc.check_a8(site_draws=2, site_translations=2, clean_draws=2)
    assert clean["passed"], clean["evidence"]
    assert "2 draws x 2 random translations (4 frames, 32 adsorbates" \
        in clean["evidence"], clean["evidence"]
    site = clean["details"]["adsorption"]
    assert site["correct"] and site["claims"] == {"top": 20, "bridge": 12}, site
    bad = acc.check_a8(mutation="wrong_site", site_draws=2,
                       site_translations=2, clean_draws=2)
    assert not bad["passed"], bad["evidence"]
    assert "38% correct" in bad["evidence"], bad["evidence"]
    assert "claims {'bridge': 32}" in bad["evidence"], bad["evidence"]


def test_a8_clean_slab_false_positive_wiring_mutation():
    """D21's new half: the clean-slab false-positive count must be wired, not
    decorative -- planting one REAL Pt adatom on one clean draw (the X1
    regression's observable) must make the count nonzero and the criterion
    FAIL.  Scoped to 1 clean draw set; the count is named in the evidence."""
    clean = acc.check_a8(site_draws=1, site_translations=1, clean_draws=2)
    assert clean["passed"], clean["evidence"]
    bad = acc.check_a8(mutation="planted_adatom", site_draws=1,
                       site_translations=1, clean_draws=2)
    assert not bad["passed"], bad["evidence"]
    assert "clean-slab false positives 1 of 12" in bad["evidence"], \
        bad["evidence"]


@pytest.mark.slow
def test_a8_full_scope_evidence_d21():
    """The nightly-shape evidence at the DEFAULT scope (D21 / X4 done-when):
    20 draws x 20 random translations judged (400 frames, 3,200 adsorbates)
    and 0 false adsorbates on 120 clean slabs (2 surfaces x 3 amplitudes x
    20 draws).  Slow on purpose: this is the acceptance run's own scope."""
    r = acc.check_a8()
    assert r["passed"], r["evidence"]
    assert "20 draws x 20 random translations (400 frames, 3200 " \
        "adsorbates" in r["evidence"], r["evidence"]
    assert "clean-slab false positives 0 of 120" in r["evidence"], \
        r["evidence"]
    assert r["details"]["adsorption"]["claims"] == {"top": 2000,
                                                    "bridge": 1200}


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
    """D21 (X4): A11 REPORTS every lift arm at 10,000 and 100,000 atoms (the
    gate stays PLAN's -- fluid at the largest size <= 120 s).  This canary
    runs a NAMED fast lane (all six arms at 4,096 atoms; the scope appears
    in the evidence, never silent) and pins the default scope; the full
    per-arm table is the nightly acceptance run's evidence.

    Red on the old implementation: check_a11(n_atoms=...) accepted no
    sizes/arms, timed only the fluid arm at one size, and its evidence named
    no per-arm seconds."""
    import inspect

    clean = acc.check_a11(sizes=(4_096,))
    assert clean["passed"], clean["evidence"]
    assert "at 4,096 atoms" in clean["evidence"], clean["evidence"]
    arms_seen = {r["arm"] for r in clean["details"]["rows"]}
    assert arms_seen == set(acc.A11_ARMS), arms_seen
    assert all(r["n_atoms"] == 4_096 for r in clean["details"]["rows"])
    bad = acc.check_a11(sizes=(4_096,), arms=("fluid",),
                        elapsed_override=120.5)
    assert not bad["passed"], bad["evidence"]
    # the default scope is pinned: six arms, both sizes (no silent shrink)
    sig = inspect.signature(acc.check_a11)
    assert sig.parameters["arms"].default == acc.A11_ARMS
    assert sig.parameters["sizes"].default == acc.A11_SIZES
    assert acc.A11_ARMS == ("crystal", "defects", "surface", "slab",
                            "fluid", "amorphous")
    assert acc.A11_SIZES == (10_000, 100_000)


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
def _structured_reference(keys):
    """A reference that covers every key the structured way: an
    entry-shaped heading plus an ACTUAL spec/examples line where one exists
    (red team F8: prose word occurrences are not coverage)."""
    import re
    examples = {}
    for p in sorted((acc.ROOT / "spec" / "examples").glob("*.chaord")):
        for line in p.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^\s*((?:(?:state|constrain|assert|history|"
                         r"conserve)\s+)?)([A-Za-z_][A-Za-z0-9_]*)\b", line)
            if not m:
                continue
            kind, key = m.group(1).strip(), m.group(2)
            for token in ((kind, key) if kind else (key,)):
                examples.setdefault(token, line.strip())
    out = []
    for k in keys:
        out.append(f"### `{k}`")
        out.append(f"{k}: definition line.")
        if k in examples:
            out.append(f"example: `{examples[k]}`")
    return "\n".join(out)


def test_a14_reference_coverage():
    base = acc.check_a14()
    keys = base["details"]["keys"]
    assert keys, "no dialect keys found"
    full_reference = _structured_reference(keys)
    clean = acc.check_a14(reference_text=full_reference)
    assert clean["passed"], clean["evidence"]
    bad = acc.check_a14(mutation="hide_key", reference_text=full_reference)
    assert not bad["passed"], bad["evidence"]
    assert "epsilon" in bad["details"]["missing"]
    # prose is not coverage (red team F8): bare words in sentences must fail
    prose = " ".join(f"We {k} things carefully." for k in keys)
    prose_result = acc.check_a14(reference_text=prose)
    assert not prose_result["passed"], prose_result["evidence"]
    assert len(prose_result["details"]["missing"]) == len(keys)
    # leave docs/reference_generated.md regenerated against the real doc
    acc.check_a14()


# ------------------------------------------------------------- runner shape --
def test_runner_report_schema():
    # A14, not A8: D21 took A8's default scope off the fast lane (20 x 20
    # site draws plus 120 clean-slab draws); the schema shape does not care
    # which criterion run_all executes
    results = acc.run_all(only=["A14"])
    assert len(results) == 1
    r = results[0]
    assert set(r) >= {"id", "name", "passed", "evidence", "details"}
    assert r["id"] == "A14" and isinstance(r["passed"], bool)
