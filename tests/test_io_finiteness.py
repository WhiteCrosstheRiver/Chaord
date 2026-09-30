"""F10 (red-team reports/redteam_findings.md, 2026-09-30): NaN input killed
the process instead of refusing.

A NaN in the positions (or the cell) reached scipy's cKDTree inside the lift
and crashed the interpreter natively (SIGSEGV, exit code 139) -- no exception
was raised, so no harness could record it. The fix: the Frame constructor
(the chokepoint shared by read_frame, from_ase and every caller) validates
finiteness and refuses with a ChaordError. Red before the fix (2026-09-30)
except the last pin, green before and after."""
import numpy as np
import pytest

from chaord.io.frames import Frame, read_frame
from chaord.lang.errors import ChaordError


def _finite_pair():
    return Frame(pos=np.array([[1.0, 1.0, 1.0], [3.0, 1.0, 1.0]]),
                 cell=np.diag([10.0] * 3), symbols=["Cu", "Cu"],
                 pbc=(True,) * 3)


def test_nan_positions_refuse_with_chaord_error():
    with pytest.raises(ChaordError, match="non-finite"):
        Frame(pos=np.full((4, 3), np.nan), cell=np.diag([10.0] * 3),
              symbols=["Cu"] * 4, pbc=(True,) * 3)


def test_one_nan_position_refuses():
    pos = np.ones((4, 3))
    pos[2, 1] = np.nan
    with pytest.raises(ChaordError, match="positions"):
        Frame(pos=pos, cell=np.diag([10.0] * 3), symbols=["Cu"] * 4,
              pbc=(True,) * 3)


def test_inf_positions_refuse():
    with pytest.raises(ChaordError, match="non-finite"):
        Frame(pos=np.full((4, 3), np.inf), cell=np.diag([10.0] * 3),
              symbols=["Cu"] * 4, pbc=(True,) * 3)


def test_nan_cell_refuses():
    with pytest.raises(ChaordError, match="cell"):
        Frame(pos=np.ones((4, 3)), cell=np.diag([np.nan] * 3),
              symbols=["Cu"] * 4, pbc=(True,) * 3)


def test_read_frame_refuses_nan_npz(tmp_path):
    """The io entry point itself must refuse, not pass NaN downstream."""
    p = tmp_path / "bad.npz"
    r = np.ones((8, 3))
    r[3] = np.nan
    np.savez(p, r=r, L=np.array([12.0] * 3))
    with pytest.raises(ChaordError, match="non-finite"):
        read_frame(p)


def test_finite_frame_still_constructs():            # pin: green before+after
    f = _finite_pair()
    assert len(f) == 2
    assert np.allclose(f.cell_diag, 10.0)
    # zero cell axes are finite: the existing ValueError path is unchanged
    g = Frame(pos=np.zeros((2, 3)), cell=np.zeros((3, 3)),
              symbols=["X", "X"])
    assert len(g) == 2
