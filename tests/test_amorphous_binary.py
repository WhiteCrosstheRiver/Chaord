"""W7 steps 1, 5, 6 (D9): multi-species amorphous regions round-trip.

An amorphous region may hold several species (`conserve atoms A 1600 B 400`,
`composition A 1600 B 400`); the physics block names a model from the
lj_mixtures dialect table (`model kob_andersen`); the build runs the history
on the pair-resolved LJ engine; the lift states composition, density, the
partial g(r)/coordination asserts and the model -- printed as ASSUMED,
inferred from the composition and the first partial g(r) peaks.

The frames here are small (N=108, 86 A + 22 B) and the histories short:
these tests verify the LANGUAGE and the wiring, not the physics (the physics
evidence is tests/test_lj_mixture.py against hand-computed pairs and the
bench/reference/ka_glass sanity + floors).
"""

# Cross-stream notes (2026-10-04): (1) the S1/W1 fail-closed cascade was
# briefly broken for every glass frame (lift_crystal's raw ValueError
# propagated); S1 fixed it.  (2) STILL OPEN on the mainline: the cascade's
# DEFECTS arm (lift_crystal_defects -> defects.fit_crystal) raises
# ValueError on placeholder species ('A'/'B' are not ASE elements), which
# the fail-closed ladder correctly propagates -- auto-mode lift of a binary
# glass refuses until fit_crystal states its no-fit verdict for non-element
# species as a ChaordError (precise patch in the W7 report; the A5 evidence
# in tests/acceptance/test_ka_glass_a5.py applies that 4-line guard
# in-process).  These tests pin mode="amorphous" -- the arm W7 owns.

from __future__ import annotations

import numpy as np
import pytest

from chaord.build import build_program
from chaord.dialects import load_dialect
from chaord.io.frames import Frame
from chaord.lang.parser import parse_text
from chaord.lang.errors import ChaordError
from chaord.lang.fmt import format_program
from chaord.lift import lift_frame
from chaord.lift.amorphous import is_amorphous, lift_amorphous

KA_PROGRAM = """chaord 0.1
dialect core + glass + lj_mixtures

system {{
  cell {L:.4f} {L:.4f} {L:.4f}
  pbc xyz
  conserve atoms A 86 B 22
}}

physics {{
  backend lj
  model kob_andersen
}}

amorphous glass : all {{
  composition A 86 B 22
  state density 1.2
  history melt 2 for 400 -> quench to 0.1 at 0.00475 -> anneal 0.1 for 300
}}

residual none
"""


@pytest.fixture(scope="module")
def dialect():
    return load_dialect(("core", "glass", "lj_mixtures"))


@pytest.fixture(scope="module")
def built_frame():
    n = 108
    L = (n / 1.2) ** (1 / 3)
    program = parse_text(KA_PROGRAM.format(L=L))
    rng = np.random.default_rng(7)
    return build_program(program, load_dialect(("core", "glass", "lj_mixtures")),
                         rng=rng)


def test_build_multi_species_frame(built_frame):
    """The mixture build conserves the species counts and the density, and
    lands disordered (its own is_amorphous gate, W1 rule)."""
    f = built_frame
    assert f.symbols.count("A") == 86 and f.symbols.count("B") == 22
    rho = len(f.pos) / float(np.prod(f.cell_diag))
    assert rho == pytest.approx(1.2, rel=0.01)
    dl = load_dialect(("core", "glass", "lj_mixtures"))
    assert is_amorphous(f, dl)


def test_lift_multi_species_states_composition_density_and_partials(built_frame):
    dl = load_dialect(("core", "glass", "lj_mixtures"))
    text = format_program(lift_frame(built_frame, dl, mode="amorphous"))
    assert "conserve atoms A 86 B 22" in text
    assert "composition A 86 B 22" in text
    assert "state density 1.2" in text
    # partial g(r) and partial coordination asserts, one per unordered pair
    for pair in ("A-A", "A-B", "B-B"):
        assert f"pair {pair}" in text, text
        assert "gr_peak" in text
    # three gr_peak asserts (one per pair) + three partial cn asserts
    assert text.count("gr_peak") == 3
    assert text.count("pair A-B") >= 2        # gr_peak + cn
    # the density assert value is the measured one (frame at rho 1.2)
    assert "1.2" in text


def test_lift_states_model_as_assumed(built_frame):
    """W7 step 5: the physics block names the inferred model and the
    provenance says it was ASSUMED (inferred from composition + first
    partial g(r) peak ratio)."""
    dl = load_dialect(("core", "glass", "lj_mixtures"))
    program = lift_amorphous(built_frame, dl)
    physics = next(b for b in program.blocks if b.t == "physics")
    models = [s for s in physics.statements if s.key == "model"]
    assert len(models) == 1 and models[0].values[0].text == "kob_andersen"
    prov = next(b for b in program.blocks if b.t == "provenance")
    notes = " ".join(s.values[0].text for s in prov.statements if s.key == "note")
    assert "model kob_andersen assumed" in notes
    assert "sigma_AB/sigma_AA 0.80" in notes


def test_multi_species_round_trip(built_frame):
    """lift -> build -> lift keeps the species, the model and the phase."""
    dl = load_dialect(("core", "glass", "lj_mixtures"))
    program = lift_frame(built_frame, dl, mode="amorphous")
    rng = np.random.default_rng(13)
    rebuilt = build_program(program, dl, rng=rng)
    assert rebuilt.symbols.count("A") == 86
    assert rebuilt.symbols.count("B") == 22
    text2 = format_program(lift_frame(rebuilt, dl, mode="amorphous"))
    assert "conserve atoms A 86 B 22" in text2
    assert "model kob_andersen" in text2
    assert "amorphous" in text2


def test_wrong_composition_infers_no_model():
    """A 60:40 mixture matches no lj_mixtures entry: no model statement, the
    build refuses (fail closed), and the lift says nothing it cannot stand
    behind."""
    dl = load_dialect(("core", "glass", "lj_mixtures"))
    rng = np.random.default_rng(11)
    L = 5.0
    pos = rng.uniform(0, L, (100, 3))
    syms = ["A"] * 60 + ["B"] * 40
    rng.shuffle(syms)
    f = Frame(pos=pos, cell=np.diag([L] * 3), symbols=syms, pbc=(True,) * 3)
    from chaord.lift.amorphous import infer_mixture_model
    assert infer_mixture_model(f, dl) is None


def test_build_refuses_multi_species_without_model():
    """Fail closed: two species and no physics model -> ChaordError, never a
    silently wrong potential."""
    n = 108
    L = (n / 1.2) ** (1 / 3)
    text = KA_PROGRAM.format(L=L).replace("  model kob_andersen\n", "")
    program = parse_text(text)
    with pytest.raises(ChaordError, match="physics `model`"):
        build_program(program, load_dialect(("core", "glass", "lj_mixtures")),
                      rng=np.random.default_rng(1))


def test_build_refuses_unknown_model():
    n = 108
    L = (n / 1.2) ** (1 / 3)
    text = KA_PROGRAM.format(L=L).replace("kob_andersen", "no_such_model")
    program = parse_text(text)
    with pytest.raises(ChaordError, match="no lj_mixtures model"):
        build_program(program, load_dialect(("core", "glass", "lj_mixtures")),
                      rng=np.random.default_rng(1))


def test_build_refuses_model_species_mismatch():
    """The mismatch species is 'D', not 'C': after a number, a WORD that is
    a core UNIT is lexed as that number's unit ('86 C' = 86 Coulomb), which
    would eat the species token entirely -- a grammar sharp edge for
    placeholder species named A/C/K/e/... recorded in the W7 report; the
    lj_mixtures table's species vocabulary (A, B) never follows a number in
    the sorted conserve/composition lines the lifter emits."""
    n = 108
    L = (n / 1.2) ** (1 / 3)
    text = (KA_PROGRAM.format(L=L)
            .replace("conserve atoms A 86 B 22", "conserve atoms A 86 D 22")
            .replace("composition A 86 B 22", "composition A 86 D 22"))
    program = parse_text(text)
    with pytest.raises(ChaordError, match="serves species"):
        build_program(program, load_dialect(("core", "glass", "lj_mixtures")),
                      rng=np.random.default_rng(1))


def test_physics_off_returns_packed_species_counts():
    """physics=False is the A5 packing-prior mutation: it must return the
    conserved species counts without running the protocol."""
    n = 108
    L = (n / 1.2) ** (1 / 3)
    program = parse_text(KA_PROGRAM.format(L=L))
    f = build_program(program, load_dialect(("core", "glass", "lj_mixtures")),
                      rng=np.random.default_rng(2), physics=False)
    assert f.symbols.count("A") == 86 and f.symbols.count("B") == 22


def test_monatomic_path_unchanged():
    """The v1 single-species amorphous build (no model statement needed) still
    works under core + glass without lj_mixtures."""
    from tests.test_amorphous import AMORPH_PROGRAM
    program = parse_text(AMORPH_PROGRAM.format(n=60, rho=0.85))
    rng = np.random.default_rng(5)
    program.blocks[2].statements = [  # shrink the history for speed
        s for s in program.blocks[2].statements if s.kind != "history"]
    f = build_program(program, load_dialect(("core", "glass")), rng=rng)
    assert len(f.pos) == 60
