"""X1 (Review 9, docs/reviews/open_items_v3.md; carried by v4 section 3): a
clean thermal Pt slab must never lift an ``adsorb`` statement of its own
species.

W10 made ``_aligned_z`` (chaord/lift/surface.py) take the slab's rotation
point from the ROUNDED z of the stack's lowest atom (``zs =
np.sort(quantize_coords(z[idx]))`` then ``rot = zs[...]``).  When rounding
goes up, that atom ends at ``z - rot < 0``, wraps to the top of the box and
lifts as a fake adatom (site top on ABCA Pt(111), hollow on ABAB Pt(100)).
Every A8 frame carries a foreign adsorbate, so no criterion caught it.

Fix under test: the quantized array may only provide the canonical SORT
KEYS; the values (including the rotation point) come from the unrounded
coordinates.  These are the brief's Pt(111) reproducer (3000 + draw seeds,
the 0.08 A case verbatim) plus the same on the W9 reviewer Pt(100)
parametrisation, at 0.05, 0.08 and 0.10 A noise, 20 draws each (rule 2).
Measured red before the fix: Pt(111) 7/9/12 of 20 at 0.05/0.08/0.10 A
(the brief's own numbers); Pt(100) 14/9/9 of 20.  Done: 0 of 20 everywhere.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src"), str(ROOT / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from acceptance import format_program_text  # noqa: E402
from chaord.dialects import load_dialect  # noqa: E402
from chaord.io.frames import Frame  # noqa: E402
from chaord.lift import lift_frame  # noqa: E402

A_PT = 3.92                      # Pt lattice constant, A (reviewer geometry)
ANN = A_PT / np.sqrt(2.0)        # fcc in-plane NN distance
D111 = A_PT / np.sqrt(3.0)       # (111) interlayer spacing

NOISE_AMP = (0.05, 0.08, 0.10)   # A; the brief's three amplitudes


def _pt111_base():
    """The X1 brief's Pt(111) slab geometry, verbatim: 4 layers of an ABCA
    fcc(111) stacking (240 Pt, nx = 6 rows x 10 row-positions per layer)."""
    ann, d111 = ANN, D111
    nx, nrow, nl = 6, 10, 4
    Lx, Ly = nx * ann, (nrow // 2) * ann * np.sqrt(3)
    step = ann * np.sqrt(3) / 6
    P = []
    for k in range(nl):
        r = k % 3
        while r * step < Ly - 1e-9:
            off = ann / 2 if r % 2 else 0.0
            P += [[(i * ann + off) % Lx, (r * step) % Ly, k * d111]
                  for i in range(nx)]
            r += 3
    return np.array(P), np.array([Lx, Ly, 24.0])


def _pt100_base():
    """The W9 reviewer Pt(100) slab parametrisation, clean (no adsorbates):
    4 layers of a 6x6 square net (side = the in-plane NN distance), alternate
    layers shifted by half a NN distance (fcc(100) ABAB stacking)."""
    a, ann, n, nl = A_PT, ANN, 6, 4
    P = np.array([[((i + 0.5 * (k % 2)) * ann) % (n * ann),
                   ((j + 0.5 * (k % 2)) * ann) % (n * ann), k * a / 2]
                  for k in range(nl) for i in range(n) for j in range(n)])
    return P, np.array([n * ann, n * ann, 22.0])


_SURFACES = {"pt111": (_pt111_base, 3000), "pt100": (_pt100_base, 4000)}


@pytest.fixture(scope="module")
def dialect():
    return load_dialect(("core", "metal", "surface"))


@pytest.mark.parametrize("amp", NOISE_AMP)
@pytest.mark.parametrize("surface", sorted(_SURFACES))
def test_clean_thermal_slab_lifts_no_adatom(surface, amp, dialect):
    """20 draws of a clean Pt slab at ``amp`` A noise: no program may carry
    an ``adsorb`` line -- the slab's own thermally displaced atoms are never
    adsorbates (X1 done-when: 0 of 20 at every amplitude on both surfaces)."""
    base, seed0 = _SURFACES[surface]
    P, L = base()
    bad = []
    for d in range(20):
        pos = P + np.random.default_rng(seed0 + d).normal(scale=amp,
                                                          size=P.shape)
        pos[:, 2] += 3.0
        text = format_program_text(lift_frame(
            Frame(pos=np.mod(pos, L), cell=np.diag(L),
                  symbols=["Pt"] * len(P), pbc=(True,) * 3), dialect))
        hits = re.findall(r"\badsorb\b.*", text)
        if hits:
            bad.append((d, hits))
    assert not bad, (
        f"{surface} at {amp} A noise: {len(bad)} of 20 clean slabs grew a "
        f"fake adatom (X1): {bad}")
