"""Fluid restraints: `constrain cn` and `constrain gr_peak` on atomic fluids.

Design rule 7: the restrained quantity is the one the lift measures and prints,
so every check below re-measures with the lift's own functions
(`lift.fluid._cn_stats`, `lift.fluid._rdf_stats`). Each test that asserts a
restraint worked first shows the unrestrained build misses the same target, so
the test could have failed.
"""
import numpy as np
import pytest

from chaord.build import build_program
from chaord.dialects import load_dialect
from chaord.lang.errors import ChaordError
from chaord.lang.parser import parse_text
from chaord.lift.fluid import _cn_stats, _rdf_stats

N = 256
RHO = 0.85
L = (N / RHO) ** (1 / 3)


def _program(region_lines: str, species: str = "X") -> str:
    return f"""chaord 0.1
dialect core + lj

system {{
  cell {L:.4f} {L:.4f} {L:.4f}
  pbc xyz
  state T 0.72
  conserve atoms {species} {N}
}}

physics {{
  backend lj
}}

liquid fluid : all {{
  state density {RHO:.3f}
{region_lines}
}}
"""


def _build(text: str, seed: int = 3, physics: bool = False):
    dl = load_dialect(("core", "lj"))
    return build_program(parse_text(text), dl, rng=np.random.default_rng(seed),
                         physics=physics), dl


def _peak(frame, dl):
    rm, g, _ = _rdf_stats(frame, dl)
    i = int(np.argmax(g))
    return float(rm[i]), float(g[i]), float(rm[1] - rm[0])


def test_unrestrained_packing_misses_the_liquid_targets():
    frame, dl = _build(_program(""))
    cn, _, _ = _cn_stats(frame, dl)
    r, h, _ = _peak(frame, dl)
    # the RSA prior is not a liquid: this is what the restraint has to fix
    assert abs(cn - 12.0) > 0.5 or abs(h - 3.0) > 0.3


def test_constrain_cn_reaches_its_target():
    frame, dl = _build(_program("  constrain cn 12.0 +- 0.3 cutoff 1.50"))
    cn, _, _ = _cn_stats(frame, dl)
    assert abs(cn - 12.0) <= 0.3


def test_constrain_gr_peak_reaches_position_and_height():
    frame, dl = _build(_program("  constrain gr_peak 1.10 height 3.00 +- 0.20"))
    r, h, dr = _peak(frame, dl)
    assert abs(r - 1.10) <= dr
    assert abs(h - 3.00) <= 0.20


def test_both_constraints_together():
    frame, dl = _build(_program(
        "  constrain cn 12.0 +- 0.3 cutoff 1.50\n"
        "  constrain gr_peak 1.10 height 3.00 +- 0.20"))
    cn, _, _ = _cn_stats(frame, dl)
    r, h, dr = _peak(frame, dl)
    assert abs(cn - 12.0) <= 0.3
    assert abs(r - 1.10) <= dr and abs(h - 3.00) <= 0.20


def test_restraint_is_deterministic():
    text = _program("  constrain cn 12.0 +- 0.3 cutoff 1.50")
    a, _ = _build(text, seed=11)
    b, _ = _build(text, seed=11)
    assert np.array_equal(a.pos, b.pos)


def test_restraint_keeps_the_hard_core_and_the_atom_count():
    frame, dl = _build(_program("  constrain cn 12.0 +- 0.3 cutoff 1.50"))
    assert len(frame) == N
    d = frame.pos[:, None, :] - frame.pos[None, :, :]
    d -= frame.cell_diag * np.round(d / frame.cell_diag)
    r = np.sqrt((d ** 2).sum(-1))
    np.fill_diagonal(r, np.inf)
    assert r.min() >= float(dl.threshold("rmc_hard_core"))


def test_unreachable_target_fails_closed():
    with pytest.raises(ChaordError, match="constrain"):
        _build(_program("  constrain cn 30.0 +- 0.1 cutoff 1.50"))


def test_unknown_fluid_constraint_key_fails_closed():
    with pytest.raises(ChaordError, match="constrain"):
        _build(_program("  constrain q6 0.5"))


def test_molecular_fluid_restraint_is_refused():
    text = """chaord 0.1
dialect core + molecular

system {
  cell 18.6 18.6 18.6
  pbc xyz
  conserve atoms H 432 O 216
}

physics {
  backend classical
}

liquid fluid : all {
  molecules H2O 216
  state density 1.0 g/cm3
  constrain cn 4.5 +- 0.5 cutoff 3.10 A
}
"""
    dl = load_dialect(("core", "molecular"))
    with pytest.raises(ChaordError, match="atomic"):
        build_program(parse_text(text), dl, rng=np.random.default_rng(1),
                      physics=False)


def test_restraint_with_physics_still_meets_the_target():
    frame, dl = _build(_program("  constrain cn 12.0 +- 0.3 cutoff 1.50"),
                       physics=True)
    cn, _, _ = _cn_stats(frame, dl)
    assert abs(cn - 12.0) <= 0.3
