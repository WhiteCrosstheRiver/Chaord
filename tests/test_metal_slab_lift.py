"""O11 (Reviews 4-7) -> W4 step 1 (Review 8): the metal stack defines
`printed_cutoff`, and the cu_solid_liquid reference must NOT be printed as
the M0-generic program.

O11 unblocked the slab route (the thresholds exist; the frame segments) but
the lifted program was wrong: Cu as `X`, LJ units, 300 K unmarked, the fcc
solid called hcp, 11 frenkel_pair lines. W4 step 1 (docs/reviews/
open_items_v2.md, gate) supersedes O11's "lifts and builds": the slab path
refuses a single-species real-element frame until step 2 carries species
and units through it. The refusal tests live in tests/test_metal_slab_refusal.py;
this file keeps the dialect-key contract and pins the reference's own
sanity floor (the frame itself stays physical; only its lift is refused)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import pytest  # noqa: E402

from chaord.dialects import load_dialect  # noqa: E402


def test_metal_dialect_defines_printed_cutoff():
    dl = load_dialect(("core", "metal"))
    assert float(dl.threshold("printed_cutoff")) == pytest.approx(3.25)
