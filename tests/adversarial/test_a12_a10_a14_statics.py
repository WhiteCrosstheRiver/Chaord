"""A12 -- RED: the overlap checker's tolerances are inconsistent with the
repo's own physics.  A12's evidence is four hand-seeded errors; the boundary
behaviour around each tolerance is not part of the criterion and hides two
measured holes:

  * lj dialect overlap_tolerance = 0.7 (sigma units) is LOOSER than the
    reference-data sanity hard core 'no pair closer than 0.8 sigma'
    (AGENTS.md / PLAN.md): an LJ frame with every pair at 0.75 sigma passes
    the static overlap check while failing the repo's own sanity rule.
  * metal dialect overlap_tolerance = 0.5 A is absolute: that is 0.20 x
    d_NN(Cu) = 2.556 A.  A Cu pair 1.78 A apart (0.70 d_NN -- a catastrophic
    overlap) passes.

GREEN (defence that holds): the lattice-mismatch check catches a wrong cell
from +1.0% upward and correctly accepts an exact fit (kept as regression).
"""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from chaord.build import build_program
from chaord.check.statics import overlap_check
from chaord.dialects import load_dialect
from chaord.io.frames import Frame
from chaord.lang.api import load as load_prog
from chaord.lang.errors import ChaordError
from tools import acceptance as acc


def _pair_at(sep, box):
    return Frame(pos=np.array([[5.0, 5.0, 5.0], [5.0 + sep, 5.0, 5.0]]),
                 cell=np.diag([box] * 3), symbols=["X", "X"],
                 pbc=(True, True, True))


def test_a12_lj_overlap_tolerance_covers_the_sanity_hard_core():
    lj = load_dialect(("core", "lj"))
    tol = float(lj.threshold("overlap_tolerance"))
    assert tol >= 0.8, (
        f"A12 RED: lj overlap_tolerance = {tol} sigma < the 0.8 sigma hard "
        "core the reference sanity checks enforce; a frame with all pairs at "
        "0.75 sigma passes the static check (measured: pair at 0.70 sigma "
        "passes today)"
    )


def test_a12_overlap_pair_below_hard_core_is_flagged_in_every_dialect():
    for names, d_nn in ((( "core", "lj"), 1.0), (("core", "metal"), 2.556)):
        dl = load_dialect(names)
        r = overlap_check(_pair_at(0.75 * d_nn, 10 * d_nn), dl)
        assert not r.passed, (
            f"A12 RED: a pair at 0.75 x d_NN ({0.75 * d_nn:.2f} in "
            f"{'+'.join(names)}) passes overlap_check ({r.detail}); "
            "AGENTS' reference rule flags anything under 0.8 sigma"
        )


@pytest.mark.parametrize("cellv,caught", [(7.23, False), (7.30, True),
                                          (7.50, True)])
def test_a12_lattice_mismatch_boundary(cellv, caught):
    metal = load_dialect(("core", "metal"))
    src = (f"chaord 0.1\n\nsystem {{\n  cell {cellv} {cellv} {cellv}\n"
           "  pbc xyz\n}\n\nphysics {\n  backend eam\n}\n\n"
           "crystal bulk : all {\n  lattice fcc\n  a 3.615 A\n}\n")
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "m.chaord"
        p.write_text(src)
        try:
            build_program(load_prog(p), metal)
            raised = False
        except ChaordError:
            raised = True
    assert raised == caught, (
        f"lattice-mismatch boundary moved: fcc a=3.615 in cell {cellv} "
        f"(+{100 * (cellv / 7.23 - 1):.1f}%) raised={raised}, expected {caught}"
    )
