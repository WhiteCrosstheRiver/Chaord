"""A6/A9/A13 -- RED: every stored hcp_mg bench frame lifts to garbage that
all three criteria accept.

The case (32-atom thermal hcp Mg supercell, orthohexagonal cell 6.42 x 11.12
x 10.42, amplitude 0.06 d_NN -- the same thermal level as the crystal cases
that lift fine) lifts in auto mode to:

    liquid fluid : all {
      molecules Mg32 1
      state density 1.736 g/cm3
      assert cn 0.0 +- 0.0 cutoff 3.10 A
    }

Root-cause chain (measured):
 1. lift_crystal raises 'no prototype matches the standardised structure'
    (mode='crystal', the case's own ground-truth lift_mode, propagates a raw
    ValueError -- a non-ChaordError);
 2. the auto cascade swallows that exception and reroutes;
 3. is_single_phase sees zero bonds because the metal dialect's
    cn_cutoff_fluid (3.10 A, Cu-scale) is BELOW d_NN(Mg) = 3.21 A, so the
    frame counts as an unbonded fluid;
 4. the census invents the molecule 'Mg32' (formula-parsed as 32 Mg), so
    conservation holds EXACTLY on all three sides and A6 records a perfect
    three-way match; A13 records a clean lift with zero residual; A9 measures
    the case's compression.

The acceptance bench-frame loop lifts everything in auto mode and ignores the
ground truth's recorded lift_mode ('crystal'), so the routing contract in the
bench metadata is never exercised.
"""
import re

import numpy as np
import pytest

from chaord.dialects import load_dialect
from chaord.io.frames import read_frame
from chaord.lift import lift_frame
from tools import acceptance as acc

METAL = ("core", "metal")


def _lift_hcp(k=0):
    case = acc.case_by_id("crystals/hcp_mg")
    frame = read_frame(case["frames"][k])
    dl = load_dialect(METAL)
    return acc.format_program_text(lift_frame(frame, dl)), case


def test_hcp_thermal_frame_lifts_as_crystal():
    text, case = _lift_hcp(0)
    phases = re.findall(r"(?m)^(crystal|liquid|gas|fluid|amorphous)\s+\S+\s*:",
                        text)
    assert phases == ["crystal"], (
        f"A6/A13 RED: hcp_mg frame_0 (a CRYSTAL case) auto-lifts as "
        f"{phases} with a 'molecules Mg32 1' census line; the program is "
        "semantically garbage yet passes A6 (three-way counts), A9 and A13"
    )


def test_hcp_conservation_row_holds_for_the_right_reason():
    """Post-fix rewrite (the original F2 red test became obsolete): the lift
    is a crystal program now, so the conservation row passing is CORRECT --
    and it must hold through the sites-derived arithmetic (32 hcp sites),
    never through an invented-molecule formula. A regression back to a
    'molecules Mg32' text must fail the molecule-free assertion, and if the
    census guard ever regresses the lift itself raises."""
    text, case = _lift_hcp(0)
    assert "Mg32" not in text, (
        "A6 RED: the census invented a pseudo-molecule again "
        "('Mg32' present in the lifted text)")
    frame = read_frame(case["frames"][0])
    row = acc._conservation_row("hcp_mg", 0, frame.symbols, text)
    assert row["counts_ok"] and row["derivation_ok"], (
        f"A6 regression: the correct crystal program fails its conservation "
        f"row ({row['derivation'][:60]})")
    assert "sites" in row["derivation"], (
        "A6 RED: the row passes only via the degraded two-way check "
        f"({row['derivation'][:60]})")


def test_hcp_ground_truth_lift_mode_is_honored():
    """The bench records lift_mode 'crystal' for hcp_mg; the acceptance
    bench-frame loop must lift with it (today lift_all_bench_frames always
    uses auto).  Lifting with the recorded mode raises a raw ValueError
    today, which is itself an A13-class defect (non-ChaordError)."""
    case = acc.case_by_id("crystals/hcp_mg")
    assert case["mode"] == "crystal"
    frame = read_frame(case["frames"][0])
    dl = load_dialect(METAL)
    with pytest.raises(Exception) as ei:
        lift_frame(frame, dl, mode="crystal")
    assert type(ei.value).__name__ == "ChaordError", (
        f"A13 RED: mode='crystal' on hcp_mg frame_0 escapes as a raw "
        f"{type(ei.value).__name__} ('{str(ei.value)[:60]}'), not a "
        "ChaordError refusal"
    )


def test_metal_fluid_cutoff_must_exceed_mg_nn_distance():
    """Post-fix rewrite: with the hcp crystal arm first in the cascade, Mg
    frames route to the crystal lifter BEFORE any fluid/q6 gate, so the
    routing must not depend on the fluid cutoffs at all -- and a single
    constant could not satisfy both constraints anyway (>= d_NN(Mg) = 3.21 A
    vs <= the cu_water liquid half-height ~3.0 A that the thin-film floor
    needs). The pin is now: the cutoff stays at the dialect's value AND the
    Mg frame still lifts as a crystal through the auto cascade."""
    dl = load_dialect(METAL)
    assert float(dl.threshold("q6_cutoff")) == 3.0, (
        "the metal q6_cutoff moved: re-check the cu_water thin-film floor "
        "(half-height ~3.0 A) before changing it -- one constant cannot "
        "serve both d_NN(Mg) and the film statistics")
    text_, case = _lift_hcp(0)
    assert re.search(r"(?m)^crystal\s+\S+\s*:", text_), (
        "F2 regression: the Mg frame no longer lifts as a crystal through "
        "the auto cascade with the dialect cutoffs unchanged")



def test_all_five_hcp_frames_misroute():
    dl = load_dialect(METAL)
    case = acc.case_by_id("crystals/hcp_mg")
    bad = 0
    for path in case["frames"]:
        frame = read_frame(path)
        text = acc.format_program_text(lift_frame(frame, dl))
        if re.search(r"(?m)^liquid\s+\S+\s*:", text) and "Mg32" in text:
            bad += 1
    assert bad == 0, (
        f"A6/A13 RED: {bad}/5 hcp_mg frames lift as 'liquid molecules Mg32'"
    )
