"""Regression: a large orthohexagonal hcp tiling keeps every atom, because the
translation search bounds come from the cell corners, not extent / |a|."""
import numpy as np

from chaord.build import build_program
from chaord.dialects import load_dialect
from chaord.lang.parser import parse_text

PROGRAM = """chaord 0.1
dialect core + lj

system {
  cell 13.607 13.748 11.110
  pbc xyz
  conserve atoms X 2016
}

physics {
  backend eam
}

crystal bulk : all {
  lattice hcp
  a 1.134 A
  c 1.852 A
  orient x [100] y [1 2 0] z [001]
}

residual none
"""


def test_large_hcp_keeps_every_atom():
    frame = build_program(parse_text(PROGRAM), load_dialect(("core", "lj")),
                          rng=np.random.default_rng(1), physics=False)
    assert len(frame.pos) == 2016
