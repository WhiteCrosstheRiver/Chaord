"""A frame that is not one fluid and has no resolvable slab must refuse in
auto mode, never print a homogeneous liquid."""
import numpy as np
import pytest

import chaord.lift.fluid as fluid
import chaord.lift.slab as slab
from chaord.dialects import load_dialect
from chaord.io.frames import Frame
from chaord.lang.errors import ChaordError
from chaord.lift import lift_frame


def _thin(*args, **kwargs):
    raise ChaordError("crystal region has no interior atoms")


def test_auto_refuses_when_not_single_phase_and_slab_fails(monkeypatch):
    monkeypatch.setattr(fluid, "is_single_phase", lambda *a, **k: False)
    monkeypatch.setattr(slab, "decompile", _thin)
    rng = np.random.default_rng(3)
    box = np.array([10.0, 10.0, 22.0])
    frame = Frame(pos=rng.random((1900, 3)) * box, cell=np.diag(box), symbols=["X"] * 1900)
    with pytest.raises(ChaordError, match="neither one fluid nor a resolvable slab"):
        lift_frame(frame, load_dialect(("core", "lj")), T=0.75)
