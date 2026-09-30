"""A9 -- scope + RED: the >=1,000-atom gate is evidenced by exactly one
system, a perfect L1_2 crystal (0.49%), plus a tiled supplementary of the
same kind; compression's hard cases are absent from the gated set.

Measured ratios (program bytes / extxyz bytes, this suite):
  homogeneous liquid   bench/reference/lj_liquid_large (2048)  0.33%
  perfect crystal      bench crystals/l12_ni3al (1372)         0.49%  <- gated
  ionic solution       bench/reference/nacl_aq (1640)          0.51%
  heterogeneous iface  tiled lj_solid_liquid 2x2 (2048)        0.68%
  heterogeneous iface  stored lj_solid_liquid (832)            1.50%
Sub-1,000-atom systems are ungated and reach 34.29% (bcc_fe, 16 atoms).

RED assertion: the gated (>=1,000-atom) measured set must include at least
one heterogeneous case (interface/solution), not only a perfect crystal --
today it does not.
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
    by_case = {}
    for rec in acc.lift_all_bench_frames():
        if rec["ok"]:
            by_case.setdefault(rec["case"], rec)
    big = {cid for cid, rec in by_case.items()
           if len(read_frame(acc.case_by_id(cid)["frames"][rec["frame"]])) >= 1000}
    het = {cid for cid in big if acc.case_by_id(cid)["category"]
           in ("interface", "interfaces", "solutions", "surface", "surfaces",
               "reactive")}
    assert het, (
        f"A9 RED: the >=1,000-atom gated evidence set is {sorted(big)} -- a "
        "single homogeneous perfect crystal; every heterogeneous category is "
        "absent, so the 2% gate has never been demonstrated on the systems "
        "where compression is hard"
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
