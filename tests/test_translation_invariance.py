"""O1 (Review 6, P0): the defect-lift crystal fit is not translation-invariant.

Red-first tests (AGENTS: the failing test is written before the fix; see
docs/reviews/open_items.md O1). Pre-fix state, reproduced 2026-10-02:

  * the three named reproducer displacements below all change the lifted
    text AND the shifted program is refused by Chaord's own builder
    ("system cell ... is not an integer multiple of the fcc/bcc lattice
    vectors" for fcc_cu / bcc_fe, "no neighbours within cutoff" for
    l12_ni3al);
  * 40 random translations per reproducer case change the text for 1-2 of
    the 40 draws (seed `|o1-trans`; rotation-only and reordering-only draws
    never do).

Root cause (lift/defects.py::fit_crystal): (a) the origin-anchored fit is
kept unless re-anchoring improves the mean matched distance by 30 %
(`lattice_anchor_improvement`), and a wrong-lattice-constant origin fit is
"equally good" under the generous site tolerance, so the improvement gate
rejects the correct re-anchored fit; (b) the re-anchor is seeded from only 4
lexsorted atoms; (c) the lattice-constant scan accepts constants that cannot
tile the periodic box (snap_a_to_cell then cannot rescue the fit because the
gates are re-measured with the wrong anchor).

The fix under test: the fit enumerates the box's own integer-tiling lattice
constants (the same `lattice_match_tolerance` cell contract the builder
enforces) and anchors from a deterministic all-atom offset pool, then
canonicalises the offset by the mean matched displacement -- so the fitted
(site lattice, constant) pair is a function of rigid invariants alone and the
lift is translation-invariant by construction.

Test-side constants (not pass code): N_RANDOM_SHIFTS = 40 is O1's own scan
size and N_RIGID_TRANSFORMS = 50 its done-when size (both above the 20-draw
rule of AGENTS.md).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from chaord.build import build_program                     # noqa: E402
from chaord.dialects import load_dialect                   # noqa: E402
from chaord.io.frames import Frame                         # noqa: E402
from chaord.lift import lift_frame                         # noqa: E402
from tools import acceptance as acc                        # noqa: E402

# O1's three named reproducer displacements (docs/reviews/open_items.md):
# fcc_cu lifts at a wrong non-tiling constant with 2 spurious frenkel pairs,
# bcc_fe the same with 3, l12_ni3al loses the lattice fit entirely.
NAMED = [
    ("crystals/fcc_cu", (-2.17, -2.18, -4.15)),
    ("crystals/bcc_fe", (-0.04, -2.54, 3.38)),
    ("crystals/l12_ni3al", (3.86, 1.97, -1.74)),
]
N_RANDOM_SHIFTS = 40       # O1's own scan size (>= 20: the 20-draw rule)
N_RIGID_TRANSFORMS = 50    # O1's done-when size per case
THERMAL_CASES = [c["id"] for c in acc._exact_roundtrip_cases(None)]


def _case(cid):
    return next(c for c in acc._exact_roundtrip_cases(None) if c["id"] == cid)


def _dialect_of(case):
    return load_dialect(case["dialect"])


def shifted(f, t):
    """The same frame, rigidly translated by t (A) and wrapped into the box."""
    return Frame(pos=np.mod(f.pos + np.asarray(t, float), f.cell_diag),
                 cell=f.cell, symbols=f.symbols, pbc=f.pbc)


# ---- O1 reproducer: the three named displacements ---------------------------

@pytest.mark.parametrize("cid,t", NAMED, ids=[c.split("/")[-1] for c, _ in NAMED])
def test_o1_named_translation_same_text_and_builds(cid, t):
    """The O1 reproducer verbatim: a rigid translation of the stored thermal
    frame must lift to byte-identical text and the lifted program must build
    (pre-fix: text differs and the builder refuses its own program)."""
    case = _case(cid)
    dl = _dialect_of(case)
    f = acc._a2a3_frame(case)
    g = shifted(f, t)
    t_ref = acc.format_program_text(lift_frame(f, dl))
    assert acc.format_program_text(lift_frame(g, dl)) == t_ref, (
        f"{cid}: the lifted text changes under the rigid translation "
        f"{t} (O1)")
    build_program(lift_frame(g, dl), dl)   # the O1 snippet builds as-is


# ---- O1 evidence: 40 random translations per reproducer case -----------------

@pytest.mark.parametrize("cid", [c for c, _ in NAMED],
                         ids=[c.split("/")[-1] for c, _ in NAMED])
def test_o1_random_translation_scan_40_draws_defect_lift(cid):
    """O1's scan: 40 seeded random translations per reproducer case must all
    lift to the untransformed frame's text (pre-fix: 1-6 of 40 differ, each
    a wrong non-tiling lattice constant and/or spurious defect tokens)."""
    case = _case(cid)
    dl = _dialect_of(case)
    f = acc._a2a3_frame(case)
    t_ref = acc.format_program_text(lift_frame(f, dl))
    rng = np.random.default_rng(acc._stable_seed(f"{cid}|o1-trans"))
    for k in range(N_RANDOM_SHIFTS):
        t = rng.uniform(-5, 5, 3)
        text = acc.format_program_text(lift_frame(shifted(f, t), dl))
        assert text == t_ref, (
            f"{cid}: draw {k} (translation {np.round(t, 3).tolist()}) lifts "
            f"to different text -- the crystal fit follows the origin (O1)")


# ---- O1 done-when: 50 rigid transforms over all 9 thermal crystal cases ------

@pytest.mark.parametrize("cid", THERMAL_CASES)
def test_o1_rigid_transform_50_draws_all_thermal_crystal_cases(cid):
    """O1's done-when matrix: 50 of A2's own rigid transforms (rotation +
    translation + re-imaging + re-ordering, acc.rigid_transform, used
    read-only) per case over all 9 exact-roundtrip thermal frames must lift
    to byte-identical text, and every lifted program must build."""
    case = _case(cid)
    dl = _dialect_of(case)
    frame = acc._a2a3_frame(case)
    prog_ref = lift_frame(frame, dl)
    t_ref = acc.format_program_text(prog_ref)
    build_program(prog_ref, dl, physics=False,
                  rng=np.random.default_rng(5))
    rng = np.random.default_rng(acc._stable_seed(f"{cid}|o1-rigid"))
    for k in range(N_RIGID_TRANSFORMS):
        g = acc.rigid_transform(frame, rng)
        prog = lift_frame(g, dl)
        text = acc.format_program_text(prog)
        assert text == t_ref, (
            f"{cid}: rigid transform {k} lifts to different text (O1 "
            f"done-when: 50 transforms per case byte-identical)")
        build_program(prog, dl, physics=False,
                      rng=np.random.default_rng(5))
