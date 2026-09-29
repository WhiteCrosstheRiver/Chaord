"""Fluid-lifter routing: the molecular arm of is_single_phase must not absorb
extended bonded components as pseudo-molecules.

Root cause (acceptance run 2026-09-29, A5): interfaces/cu_water lifted as
`liquid fluid : all` whose census blob 'Cu384' then failed the rebuild's
species lookup -- the M3-era shortcut (any bond -> single-phase fluid) predates
interface segmentation and swallowed whole metal slabs."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chaord.build.molecules import bond_graph  # noqa: E402
from chaord.dialects import load_dialect  # noqa: E402
from chaord.io.frames import Frame, read_frame  # noqa: E402
from chaord.lift.fluid import bonded_single_phase, is_single_phase  # noqa: E402

BENCH = ROOT / "bench" / "data"


def _cu_water_frame() -> Frame:
    return read_frame(BENCH / "interfaces" / "cu_water" / "frame_0.npz")


def test_cu_water_slab_is_not_single_phase():
    """A Cu crystal slab coexisting with water must not lift as one fluid.

    Failed before the fix: the bonded arm returned True for any frame with
    bonds, so the 384-atom Cu slab passed as a single-phase molecular fluid."""
    metal = load_dialect(("core", "metal"))
    assert is_single_phase(_cu_water_frame(), metal) is False


def test_extended_component_fails_bonded_arm():
    """A bonded chain far above the dialect bound is a second phase."""
    molecular = load_dialect(("core", "molecular"))
    # 20 C atoms 1.3 A apart: a covalently bonded chain (2 x r_cov(C) = 1.52 A)
    n = 20
    pos = np.zeros((n, 3))
    pos[:, 0] = np.arange(n) * 1.3
    frame = Frame(pos=pos, cell=np.diag([40.0, 8.0, 8.0]),
                  symbols=["C"] * n, pbc=(False, False, False))
    edges = bond_graph(frame, molecular)
    assert edges, "chain must actually be bonded for the test to bite"
    assert bonded_single_phase(frame, edges, molecular) is False
    assert is_single_phase(frame, molecular) is False


def test_small_molecules_stay_single_phase():
    """Reference molecular fluids keep their fluid routing (no over-fix)."""
    molecular = load_dialect(("core", "molecular"))
    for case in ("water_tip4p", "nacl_aq"):
        frame = read_frame(ROOT / "bench" / "reference" / case / "frame_0.npz")
        assert is_single_phase(frame, molecular) is True, case
