"""W4 step 1 (gate) and W11 (assumed T) for the slab lift path.

W4 step 1 (docs/reviews/open_items_v2.md, gate): the single-species slab
program is the M0 text -- `units lj`, `backend lj`, epsilon 1 / sigma 1,
`conserve atoms X <N>`. For a frame whose one species is a real element
(the Cu solid-liquid reference: 832 Cu at 1335 K) every one of those lines
is wrong: Cu printed as `X`, Angstrom geometry printed in sigma, and an
unmarked dialect-default `state T 300.00` (W11). Pre-fix reproducer output
(commit f4fde65, re-run locally as bookkeeping):

    units lj / state T 300.00 / conserve atoms X 832 / backend lj /
    lattice hcp / 11 frenkel_pair lines -- all wrong.

The gate: `program_from_result` refuses such frames with the W4 message
until step 2 carries species and units through the slab path. Frames that
must keep lifting: pseudo-species `X` (the LJ bench), multi-species real
elements (the multi arm names each element and prints Angstrom units), and
`symbols=None` (the byte-identical M0 contract the CLI roundtrip uses).

W11: the slab path states the dialect-default T when the caller gives none;
the program's provenance must flag it assumed, exactly as the fluid path
does (`fluid.AssumedT`, note "T assumed (dialect default ...)"), and must
NOT flag a caller-given T.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np  # noqa: E402

import acceptance as acc  # noqa: E402
from chaord.dialects import load_dialect  # noqa: E402
from chaord.io.frames import read_frame  # noqa: E402
from chaord.lang.errors import ChaordError  # noqa: E402
from chaord.lift import lift_frame  # noqa: E402
from chaord.lift.slab import decompile, program_from_result  # noqa: E402

METAL = ("core", "metal")
LJ = ("core", "lj")

# the W4 reproducer frame: bench/reference/cu_solid_liquid/frame_0.npz,
# 832 Cu at 1335 K (the snippet loads it through load_ref; read_frame is
# the same npz reader with the same symbols)
CU_REF = ROOT / "bench" / "reference" / "cu_solid_liquid" / "frame_0.npz"
# single-species pseudo-species X slab (the LJ bench) and multi-species
# metal interface: both must keep lifting
LJ_BENCH = ROOT / "bench" / "data" / "interface" / "lj_solid_liquid" / "frame_0.npz"
CU_WATER = ROOT / "bench" / "data" / "interfaces" / "cu_water" / "frame_0.npz"

W4_MESSAGE = "single-species metal solid\u2013liquid interfaces are not supported yet"


def _provenance_notes(text):
    """The `note "..."` lines of the provenance block, as plain strings."""
    import re
    return re.findall(r'note "([^"]*)"', text)


# ------------------------------------------------------------------ W4 gate --

def test_w4_reproducer_single_species_metal_slab_refused():
    """The W4 reproducer verbatim: lift_frame on the Cu reference (mode auto,
    the snippet's call) must raise instead of printing Cu as `X` in LJ units.
    Red before the gate: the lift returned the wrong program
    (units lj / state T 300.00 / conserve atoms X 832 / lattice hcp /
    11 frenkel_pair lines).

    The refusal surfaces as a ChaordError; with today's legacy cascade the
    visible message is the fluid fallback's own guard (the slab arm's
    `except Exception` re-routes to lift_fluid, which refuses the extended
    bonded crystal) -- the W4 message itself is asserted by the pinned-mode
    and direct-call tests below, and W1's fail-closed cascade will surface
    it in auto too."""
    frame = read_frame(CU_REF)
    with pytest.raises(ChaordError):
        lift_frame(frame, load_dialect(METAL))


def test_w4_pinned_slab_mode_refuses():
    """The pinned arm refuses loudly (not only through the auto cascade)."""
    frame = read_frame(CU_REF)
    with pytest.raises(ChaordError, match=W4_MESSAGE):
        lift_frame(frame, load_dialect(METAL), mode="slab")


def test_w4_gate_sits_in_program_from_result():
    """decompile still measures the frame (its numbers are not the lie); the
    refusal fires where the X/LJ text would be assembled. This pins the
    gate's location for W4 step 2, which removes it."""
    frame = read_frame(CU_REF)
    res = decompile(frame.pos, frame.cell_diag, 300.0, load_dialect(METAL),
                    symbols=list(frame.symbols))
    assert res["N"] == 832
    with pytest.raises(ChaordError, match=W4_MESSAGE):
        program_from_result(res, load_dialect(METAL))


def test_w4_lj_x_species_slab_still_lifts():
    """The refusal condition is 'real element printed as X in LJ units': the
    LJ bench's pseudo-species `X` frames keep lifting byte-identically."""
    frame = read_frame(LJ_BENCH)
    text = acc.format_program_text(lift_frame(frame, load_dialect(LJ), mode="slab"))
    assert "units lj" in text
    assert "conserve atoms X 512" in text


def test_w4_multispecies_real_element_slab_still_lifts():
    """A multi-species interface (Cu under water) takes the multi arm, which
    names each element and prints Angstrom units: no refusal."""
    frame = read_frame(CU_WATER)
    text = acc.format_program_text(lift_frame(frame, load_dialect(METAL), mode="slab"))
    assert "conserve atoms Cu 384" in text
    assert "molecules H2O 136" in text


def test_w4_symbols_none_m0_path_still_lifts():
    """symbols=None is the byte-identical M0 contract (the CLI roundtrip
    calls decompile this way): the lift does not know the species, so the
    gate does not fire. The species-aware entry point (lift_frame) is what
    W4 step 1 closes."""
    frame = read_frame(CU_REF)
    res = decompile(frame.pos, frame.cell_diag, 300.0, load_dialect(METAL))
    program = program_from_result(res, load_dialect(METAL))
    text = acc.format_program_text(program)
    assert "conserve atoms X 832" in text


# ------------------------------------------------------------- W11 assumed T --

def test_w11_slab_flags_default_t_as_assumed():
    """Red before W11: the slab path printed `state T 0.65` with no mark.
    After: the dialect-default T is stated AND its provenance carries the
    same note the fluid path emits."""
    frame = read_frame(LJ_BENCH)
    text = acc.format_program_text(lift_frame(frame, load_dialect(LJ), mode="slab"))
    assert "state T 0.65" in text
    notes = _provenance_notes(text)
    assert any(n.startswith("T assumed (dialect default 0.65)")
               and "not measured from the frame" in n for n in notes), notes


def test_w11_slab_caller_given_t_is_not_flagged():
    """A T the caller passed is a fact about the program, not an assumption:
    no assumed-note may appear (and the stated value is the caller's)."""
    frame = read_frame(LJ_BENCH)
    text = acc.format_program_text(
        lift_frame(frame, load_dialect(LJ), T=0.7, mode="slab"))
    assert "state T 0.70" in text
    assert not any("T assumed" in n for n in _provenance_notes(text))


def test_w11_multispecies_slab_flags_default_t_as_assumed():
    """The multi-species slab arm (cu_water, core + metal) states the metal
    dialect's 300 K default; it must be flagged assumed the same way."""
    frame = read_frame(CU_WATER)
    text = acc.format_program_text(lift_frame(frame, load_dialect(METAL), mode="slab"))
    assert "state T 300.00" in text
    notes = _provenance_notes(text)
    assert any(n.startswith("T assumed (dialect default 300)")
               and "not measured from the frame" in n for n in notes), notes
