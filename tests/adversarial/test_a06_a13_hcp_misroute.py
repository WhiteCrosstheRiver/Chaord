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


def test_hcp_misroute_conservation_row_is_blind():
    """The garbage program must not satisfy A6's three-way check once the
    census invents molecules.  Today it does (derive_counts: 'unknown
    molecule Mg32' degrades to the two-way check; chaord's own
    conservation_check formula-parses Mg32 as 32 Mg and agrees exactly)."""
    text, case = _lift_hcp(0)
    frame = read_frame(case["frames"][0])
    row = acc._conservation_row("hcp_mg", 0, frame.symbols, text)
    assert not (row["counts_ok"] and row["derivation_ok"]), (
        f"A6 RED: a 'liquid molecules Mg32' program of a crystal frame "
        f"passes the conservation row ({row['derivation'][:60]})"
    )


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
    """Root cause 3, stated as a threshold invariant: a dialect's fluid
    coordination cutoff must exceed the largest nearest-neighbour distance of
    the systems it covers, else every bond vanishes and crystals route to
    'fluid'.  d_NN(Mg) = c/a-scaled 3.21 A here vs cutoff 3.10 A."""
    dl = load_dialect(METAL)
    cutoff = float(dl.threshold("cn_cutoff_fluid"))
    mg_a = 3.21
    assert cutoff > mg_a, (
        f"metal dialect cn_cutoff_fluid={cutoff} A < d_NN(Mg hcp)={mg_a} A: "
        "the Mg crystal has zero bonds under the fluid classifier and "
        "misroutes (measured: all 5 hcp_mg frames lift as 'liquid')"
    )


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
