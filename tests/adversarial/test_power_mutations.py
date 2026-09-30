"""Power mutations: for each criterion, the "most likely REAL error" an
implementation could make, injected beyond the 22 canaries already in
tests/acceptance/test_mutations.py.  Review step 6 asked specifically for
temperature, averaging, heterogeneous A9 and a hot SRO frame -- those live
here, each with its measured number.

Index (full write-ups in reports/redteam_findings.md):
  A1  string escape corruption ................ test_a01_string_escapes (RED)
  A2  rotation of a THERMAL frame ............. test_a02_a03_thermal_frames (RED)
  A3  lift->build on the stored thermal frames  test_a02_a03_thermal_frames (RED)
  A4  mixed defect types in one cell .......... test_a4_mixed_cell_mutation (RED metric)
  A5  wrong temperature (x0.8 / x1.25) ........ test_a05_temperature_and_averaging (RED)
  A5  missing averaging / reference-frame luck  test_a05_temperature_and_averaging (RED)
  A6  slab region arithmetic .................. test_a06_conservation_blindspots (RED)
  A6  multivalent-ion charge .................. test_a06_conservation_blindspots (RED)
  A7  targeted label corruption outside core .. test_a07_judged_core_scope (RED)
  A9  heterogeneous >=1,000-atom evidence ..... test_a09_evidence_class (RED)
  A10 hash-seed nondeterminism ............... test_defended_greens (GREEN)
  A12 overlap tolerance vs 0.8 sigma core .... test_a12_a10_a14_statics (RED)
  A13 NaN input (native crash) ............... test_a13_nan_crash (RED)
  A14 prose-coverage drift ................... test_a14_prose_coverage (RED)
  A2  hcp misroute (cascade + cutoff) ......... test_a06_a13_hcp_misroute (RED)
  SRO hot frame (W1-A/W1-B territory) ........ test_sro_hot_frame_power_mutation
      below -- xfail(strict=False): the SRO/defect lift is being reworked by
      the parallel W1-A/W1-B streams; do not confuse this with their
      pre-fix red tests.
"""
import re

import pytest

from chaord.build import build_program
from chaord.dialects import load_dialect
from chaord.io.frames import read_frame
from chaord.lang.parser import parse_text
from chaord.lift import lift_frame
from tools import acceptance as acc

from helpers_frames import mixed_l12_defects, thermal

METAL = ("core", "metal")


def _detected(text):
    return {t: int(c) for t, c in re.findall(r"defect (\S+) count (\d+)", text)}


def test_a4_mixed_cell_mutation():
    """Power mutation for A4: plant MIXED defect types in one cell (the
    realistic case; the acceptance plan used to plant exactly one type per
    cell).  The lift detects the planted multiset exactly (measured:
    V_Ni 3 + Al_Ni 4 + Ni_i 3 -> identical dict, a perfect lift), and since
    the 2026-09-30 fix the criterion's per-cell P/R arithmetic scores the
    planted MULTISET (P = correct detections / total detections,
    R = correct detections / total planted, bound per Kröger-Vink token), so
    the perfect mixed cell scores P = R = 1.0 instead of the pre-fix
    P = 0.30/0.40/0.30 (every other type's true detection used to count as a
    false positive).  The acceptance grid itself now carries mixed cells."""
    metal = load_dialect(METAL)
    frame, edge = mixed_l12_defects()
    d = acc.median_nn_distance(frame.pos, frame.cell_diag)
    hot = thermal(frame, float(metal.threshold("thermal_test_amplitude")) * d, 11)
    det = _detected(acc.format_program_text(lift_frame(hot, metal, mode="defects")))
    # lift side must be exact (this is the green half; see greens file)
    planted = {"V_Ni": 3, "Al_Ni": 4, "Ni_i": 3}
    assert det == planted, det
    p, r, counts = acc._pr_for_cell(planted, dict(det))
    assert p >= 0.95 and r >= 0.95, (
        f"A4 power mutation RED: the per-cell metric scores the perfectly "
        f"detected mixed cell at P={p:.2f} R={r:.2f} -- mixed cells are "
        "outside the metric's expressive range, so the criterion is only "
        "demonstrated on single-type plantings"
    )
    # ...and the acceptance plan itself must plant mixed cells (F12's other
    # half: the criterion must be DEMONSTRATED on mixed cells, not merely
    # able to score them)
    mixed_plans = [planting for host in acc.A4_HOSTS
                   for dtype, planting in acc.A4_PLANS[host[0]]
                   if dtype == "mixed"]
    assert len(mixed_plans) >= 2, (
        f"A4 RED: the acceptance grid plants {len(mixed_plans)} mixed cells; "
        "the criterion must be demonstrated on mixed-kind plantings"
    )
    assert all(len(pl) >= 2 for pl in mixed_plans), (
        "A4 RED: a 'mixed' cell in the acceptance grid plants a single kind"
    )


@pytest.mark.xfail(strict=False, reason=(
    "waiting W1-A/W1-B: the SRO/defect lift (lift/defect_program, "
    "lift/defects, lift/extended) is being reworked by the parallel repair "
    "streams; recorded as the measured pre-fix state, not as a red test "
    "against their in-flight work"))
def test_sro_hot_frame_power_mutation():
    """A3's species check for random solutions runs on perfect rebuilds; on
    the STORED thermal fcc_crconi frame the round trip's species arrangement
    is measured at max |dAlpha| = 0.393 (rebuild seed 5) against a perfect
    -frame relabel floor of order 0.1: the thermal-frame SRO round trip is
    not demonstrated today.  Needs the W1 SRO rework to settle; the number
    is recorded here so the repair has a target."""
    dl = load_dialect(METAL)
    case = acc.case_by_id("crystals/fcc_crconi")
    frame = read_frame(case["frames"][0])
    bands = acc._alpha_shell_bands(case)
    assert bands is not None
    t1 = acc.format_program_text(lift_frame(frame, dl))
    rb = build_program(parse_text(t1), dl, rng=np.random.default_rng(5))
    delta = acc._max_alpha_delta(frame, rb, bands)
    floor = acc._relabel_noise_floor(case, bands)
    assert delta <= 1.5 * floor, (
        f"thermal-frame SRO round trip: max |dAlpha| {delta:.3f} vs relabel "
        f"floor {floor:.3f} (x{delta / max(floor, 1e-9):.1f})"
    )
