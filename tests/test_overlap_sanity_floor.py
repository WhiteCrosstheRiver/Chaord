"""F6 (red-team reports/redteam_findings.md, 2026-09-30): the static overlap
check's tolerance sat below the repo's own sanity hard core.

AGENTS.md requires reference data to pass 'no pair closer than 0.8 sigma (or
the potential's hard core)'. The static check allowed less: lj
overlap_tolerance = 0.70 sigma, and the metal value (core's 0.5 A absolute)
is 0.20 x d_NN(Cu). The check therefore passed pairs the repo's own sanity
rule forbids (measured: a pair at 0.75 sigma, and a Cu pair at 0.70 d_NN =
1.78 A, both passed).

The fix: overlap_check never runs below the sanity fraction (0.8, or the
dialect's overlap_sanity_fraction override once approved) of the system's
natural length scale -- 1 sigma for LJ reduced units, the reference
nearest-neighbour distance for pure-metal frames. Frames that contain
species outside the dialect's atomic tables (molecular water against a metal
slab) keep the dialect tolerance alone: their intramolecular bonds are legal
sub-floor distances. The tests below were red before the fix (except the
molecular pin, green before and after)."""
import numpy as np
import pytest

from chaord.check.statics import overlap_check
from chaord.dialects import load_dialect
from chaord.io.frames import Frame

DNN_CU = 3.6149 / np.sqrt(2.0)      # metal dialect eam_potentials Cu a_ref


def _pair(sep, box, symbols=("X", "X")):
    return Frame(pos=np.array([[5.0, 5.0, 5.0], [5.0 + sep, 5.0, 5.0]]),
                 cell=np.diag([box] * 3), symbols=list(symbols),
                 pbc=(True, True, True))


def test_lj_pair_below_the_hard_core_is_flagged():
    lj = load_dialect(("core", "lj"))
    r = overlap_check(_pair(0.75, 10.0), lj)
    assert not r.passed, (
        f"F6: a pair at 0.75 sigma passes the LJ static overlap check "
        f"({r.detail}); the AGENTS sanity hard core is 0.8 sigma")


def test_lj_pair_above_the_hard_core_still_passes():
    lj = load_dialect(("core", "lj"))
    r = overlap_check(_pair(0.85, 10.0), lj)
    assert r.passed, r.detail


def test_metal_pair_below_the_hard_core_fraction_is_flagged():
    """0.75 x d_NN(Cu) = 1.917 A: a catastrophic overlap that passed at the
    core dialect's absolute 0.5 A (0.20 d_NN)."""
    metal = load_dialect(("core", "metal"))
    r = overlap_check(_pair(0.75 * DNN_CU, 10 * DNN_CU), metal)
    assert not r.passed, (
        f"F6: a Cu pair at {0.75 * DNN_CU:.3f} A (0.75 d_NN) passes the "
        f"metal static overlap check ({r.detail}); the sanity rule is "
        "no pair closer than 0.8 d_NN")


def test_metal_lattice_pair_still_passes():
    metal = load_dialect(("core", "metal"))
    r = overlap_check(_pair(DNN_CU, 10 * DNN_CU, symbols=("Cu", "Cu")), metal)
    assert r.passed, r.detail


def test_molecular_species_do_not_get_the_atomic_floor():
    """Pin (green before and after): a metal-dialect frame that also contains
    molecular species (the cu_water interface class) keeps the dialect
    tolerance alone -- the O-H bond at 0.96 A is a legal distance, and an
    atomic hard-sphere floor would flag every water molecule."""
    metal = load_dialect(("core", "metal"))
    frame = Frame(
        pos=np.array([[0.0, 0.0, 0.0], [0.958, 0.0, 0.0],      # O-H bond
                      [5.0, 5.0, 5.0], [5.0 + DNN_CU, 5.0, 5.0]]),
        cell=np.diag([25.0] * 3), symbols=["O", "H", "Cu", "Cu"],
        pbc=(True, True, True))
    r = overlap_check(frame, metal)
    assert r.passed, r.detail
