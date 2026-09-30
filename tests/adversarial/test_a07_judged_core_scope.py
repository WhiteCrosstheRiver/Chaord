"""A7 -- RED: the phase-segmentation accuracy is judged on a core that
excludes 64.5% of the atoms; a labeler that is wrong ONLY inside the excluded
zones passes with accuracy 1.0.

Measured (interface/lj_solid_liquid frame_0, 512 atoms, boundary 6.35,
d_NN 1.007, exclusion band 2 x d_NN plus the periodic z edges):
  judged core = 182 atoms (35.5%); excluded = 330 atoms (64.5%).
A labeler that flips EVERY label in the excluded 330 atoms and is perfect in
the core scores accuracy 1.000 on the A7 metric (100% of judged atoms
correct).  The seeded canary (flip 20% of labels globally) is caught because
its flips land inside the core; a targeted one is not.

The exclusion band itself is physically motivated (interface atoms are
ambiguous); the hole is that the metric never reports how much of the system
it judged, so '95% correct' silently means '95% of 35.5%'.
"""
import numpy as np

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
    z = frame.pos[:, 2]
    band = 2.0 * d_nn
    return ((np.abs(z - boundary) > band) & (z > band)
            & (z < L[2] - band)), boundary, d_nn


def test_a07_targeted_label_corruption_is_caught():
    case = acc.case_by_id("interface/lj_solid_liquid")
    dl = load_dialect(("core", "lj"))
    frame = read_frame(case["frames"][0])
    core, boundary, d_nn = _core(case, frame)
    labels = phase_labels(frame, dl)
    rng = np.random.default_rng(0)
    truth = frame.pos[:, 2] > boundary
    # corrupt EXACTLY the atoms the metric never judges
    targeted = labels.copy()
    outside = ~core
    targeted[outside] = ~truth[outside]        # every excluded atom wrong
    acc_core = float((targeted[core] == (~truth[core])).mean())
    overall = float((targeted == (~truth)).mean())
    assert not (acc_core >= 0.95), (
        f"A7 RED: a labeler wrong on {int(outside.sum())}/{len(frame)} atoms "
        f"({100 * outside.mean():.1f}% of the system, every atom outside the "
        f"judged core) passes A7 with core accuracy {acc_core:.3f} while the "
        f"overall accuracy is {overall:.3f}; the core is only "
        f"{int(core.sum())}/{len(frame)} atoms "
        f"({100 * core.mean():.1f}%), and the metric never reports how much "
        "of the system it judged"
    )
