"""A9 -- scope + RED: the >=1,000-atom gate used to be evidenced by exactly
one system, a perfect L1_2 crystal (0.49%), plus a tiled supplementary of
the same kind; compression's hard cases were absent from the gated set.

Measured ratios (program bytes / extxyz bytes, this suite, 2026-09-30):
  homogeneous liquid   bench/reference/lj_liquid_large (2048)  0.40%
  perfect crystal      bench crystals/l12_ni3al (1372)         0.49%  <- gated
  ionic solution       bench/reference/nacl_aq (1640)          0.51%
  heterogeneous iface  bench/reference/lj_solid_liquid (2304)  0.55%  <- gated
  heterogeneous iface  tiled lj_solid_liquid 2x2 (2048)        0.68%
  heterogeneous iface  stored lj_solid_liquid (832)            1.50%
Sub-1,000-atom systems are ungated and reach 34.29% (bcc_fe, 16 atoms).

Post-fix closure (2026-09-30): check_a9 measures the independent-MD
reference frames (bench/reference/*) alongside the bench cases, so the
2,304-atom heterogeneous solid-liquid frame and the 1,640-atom ionic
solution are gated measurements, and the evidence reports the worst
crystal-class and worst heterogeneous-class ratios separately.

RED assertion: the gated (>=1,000-atom) measured set must include at least
one heterogeneous case (interface/solution), not only a perfect crystal.
GREEN assertion: the heterogeneous tiled case itself stays under the 2% gate
(a defence that holds; regression asset).
"""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from chaord.dialects import load_dialect
from chaord.io.frames import read_frame, write_frame, Frame
from chaord.lift import lift_frame
from tools import acceptance as acc


def _ratio(frame, text):
    with tempfile.TemporaryDirectory() as td:
        q = Path(td) / "f.extxyz"
        write_frame(q, frame)
        return 100.0 * len(text.encode()) / q.stat().st_size, q.stat().st_size


@pytest.mark.slow
def test_a09_gated_set_includes_a_heterogeneous_case():
    res = acc.check_a9()
    gated = [m for m in res["details"]["measurements"]
             if m.get("n_atoms", 0) >= 1000 and m.get("source") in
             ("bench", "reference")]
    het = [m for m in gated
           if m.get("category") in acc.A9_HETEROGENEOUS_CLASSES]
    assert het, (
        f"A9 RED: the >=1,000-atom gated evidence set is "
        f"{[(m['case'], m.get('category')) for m in gated]} -- a single "
        "homogeneous perfect crystal; every heterogeneous category is "
        "absent, so the 2% gate has never been demonstrated on the systems "
        "where compression is hard"
    )
    assert res["passed"], res["evidence"]
    # the evidence names the worst ratios per class, not just one number
    assert "worst crystal-class" in res["evidence"]
    assert "heterogeneous-class" in res["evidence"]
    worst_het = max(het, key=lambda m: m["ratio"])
    assert worst_het["ratio"] <= 2.0, (
        f"A9 RED: the heterogeneous gated case {worst_het['case']} measures "
        f"{worst_het['ratio']:.2f}% -- over the 2% gate"
    )


def test_a09_heterogeneous_tiled_2048_stays_under_gate():
    lj = load_dialect(("core", "lj"))
    case = acc.case_by_id("interface/lj_solid_liquid")
    frame = read_frame(case["frames"][0])
    L = frame.cell_diag
    pos = np.vstack([frame.pos + np.array([i * L[0], j * L[1], 0.0])
                     for i in (0, 1) for j in (0, 1)])
    cell = frame.cell.copy()
    cell[0] *= 2
    cell[1] *= 2
    tiled = Frame(pos=pos, cell=cell, symbols=frame.symbols * 4,
                  pbc=frame.pbc)
    text = acc.format_program_text(lift_frame(tiled, lj))
    ratio, xyz_b = _ratio(tiled, text)
    assert ratio <= 2.0, (
        f"A9 (green defence): the heterogeneous 2,048-atom interface frame "
        f"compresses to {ratio:.2f}% ({len(text.encode())} B program / "
        f"{xyz_b} B extxyz) -- under the gate; keep this regression"
    )
