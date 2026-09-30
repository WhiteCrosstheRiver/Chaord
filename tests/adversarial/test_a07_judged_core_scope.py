"""A7 -- the phase-segmentation accuracy is judged on a core that used to
exclude 64.5% of the atoms while never reporting it: '95% correct' silently
meant '95% of 35.5%' (red team F9).

Measured (interface/lj_solid_liquid frame_0, 512 atoms, boundary 6.35,
d_NN 1.007): a labeler that flips EVERY label outside the judged core scores
accuracy 1.000 on the A7 metric while being 64.5%-wrong overall.

Post-fix closure (2026-09-30), three defences, each asserted here:

1. PRINCIPLED MASK -- the core excludes exactly the 2 x d_NN band around
   every interface plane (the ground-truth boundary AND its translate at
   boundary +- L_z/2 in a periodic two-phase slab cell) plus, only where z
   is not periodic, the real surfaces.  Measured on the bench frames this is
   numerically the old mask (the old z-edge cut coincided with the periodic
   interface plane), but it now holds by RULE: no solid/liquid-interior atom
   beyond the bands is excluded.  Every mislabelled atom of the bench frames
   lies inside those bands (0 outside, both cases, 5 frames each).
2. DISCLOSURE -- every row and the criterion evidence report the judged
   fraction (0.355-0.395 on lj_solid_liquid: a 12.7 A cell where 2 x d_NN
   around each of the two interface planes covers ~64% of the atoms -- a
   fact the gate report now states instead of hiding).
3. TEETH -- a core that audits less than the interface geometry leaves
   available FAILS the criterion (judged fraction >= 90% of the
   geometry-implied availability); the widen_band mutation (band x 2)
   demonstrates the gate, so a silent scope shrink cannot pass.
"""
import numpy as np
import pytest

from chaord.dialects import load_dialect
from chaord.io.frames import read_frame
from chaord.lift.segment import phase_labels
from tools import acceptance as acc


def _core(case, frame):
    gt = case["gt"]
    boundary = next((f.get("boundary_z") for f in gt["frames"]
                     if "boundary_z" in f),
                    gt["expected"].get("boundary_z"))
    L = frame.cell_diag
    d_nn = acc.median_nn_distance(frame.pos, L)
    core, floor = acc._a7_core_mask(frame.pos[:, 2], float(L[2]),
                                    float(boundary), 2.0 * d_nn,
                                    bool(frame.pbc[2]))
    return core, boundary, d_nn, floor


def test_a07_targeted_label_corruption_is_caught():
    case = acc.case_by_id("interface/lj_solid_liquid")
    dl = load_dialect(("core", "lj"))
    frame = read_frame(case["frames"][0])
    core, boundary, d_nn, floor = _core(case, frame)
    labels = phase_labels(frame, dl)
    truth = frame.pos[:, 2] > boundary
    # the original attack, direction fixed: correct means label == ~truth, so
    # corrupting the unjudged atoms means setting their labels TO truth --
    # every excluded atom wrong, the judged core untouched ...
    targeted = labels.copy()
    outside = ~core
    targeted[outside] = truth[outside]         # every excluded atom wrong
    acc_core = float((targeted[core] == (~truth[core])).mean())
    overall = float((targeted == (~truth)).mean())
    assert acc_core >= 0.95 and overall < 0.95, (
        "attack premise changed: the corrupted labeler must still ace the "
        "judged core while being wrong overall"
    )
    # ... so the defences are principled scope, disclosure and teeth:
    #
    # (1) principled mask: every EXCLUDED atom is within 2 x d_NN of an
    # interface plane (boundary or its periodic translate) -- nothing of the
    # solid/liquid interior beyond the bands is silently unaudited
    z = frame.pos[:, 2]
    L2 = float(frame.cell_diag[2])
    bands = [boundary, (boundary + L2 / 2) % L2]
    for p in bands:
        dz = np.minimum(np.abs(z - p), L2 - np.abs(z - p))
        outside &= dz <= 2.0 * d_nn
    assert outside.sum() == 0, (
        f"A7 RED: {int(outside.sum())} excluded atoms are further than "
        f"2 x d_NN from every interface plane -- the core mask audits less "
        "than the interface geometry requires (interior atoms excluded)"
    )
    # (2) disclosure: the criterion reports the judged fraction, so the
    # '95% correct' claim states its coverage instead of hiding it
    res = acc.check_a7(frames=(0,))
    row = next(r for r in res["details"]["rows"]
               if r["case"] == "interface/lj_solid_liquid" and r["frame"] == 0)
    assert row["judged_frac"] == pytest.approx(float(core.mean()), abs=1e-4), (
        "A7 RED: the disclosed judged fraction disagrees with an "
        "independent recomputation of the mask"
    )
    assert "judged fraction disclosed per frame" in res["evidence"], (
        "A7 RED: the criterion evidence does not state how much of the "
        f"system it judged (core = {core.mean():.1%} of atoms; the labeler "
        f"wrong on every excluded atom still scores {acc_core:.3f})"
    )
    # (3) teeth: an audit narrower than the interface geometry implies
    # FAILS the criterion (insufficient evidence) -- widen_band halves the
    # available core and must not pass
    bad = acc.check_a7(mutation="widen_band", frames=(0,))
    assert not bad["passed"], (
        "A7 RED: a doubled exclusion band (the audit silently judging less) "
        "still passes the criterion -- the judged-fraction gate has no teeth"
    )
    # and the honest floor is what the geometry leaves available, so the
    # clean run passes with the disclosure in place
    assert res["passed"], res["evidence"]
