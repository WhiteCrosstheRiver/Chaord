"""W4 (Review 8): what the slab lift still refuses after step 2.

Step 1 (gate): the single-species slab program was the M0 text -- `units
lj`, `backend lj`, epsilon 1 / sigma 1, `conserve atoms X <N>` -- so a
real-element frame (the Cu solid-liquid reference: 832 Cu at 1335 K) was
refused outright: Cu laundered into `X`, Angstrom geometry in sigma, an
unmarked dialect-default `state T 300.00` (W11), the fcc solid called hcp,
11 spurious frenkel_pair lines. Pre-fix reproducer output (commit f4fde65,
re-run locally as bookkeeping):

    units lj / state T 300.00 / conserve atoms X 832 / backend lj /
    lattice hcp / 11 frenkel_pair lines -- all wrong.

Step 2 carries species and units through the single-species metal arm (the
done-when reproducer is green in tests/test_metal_slab_lift.py: `conserve
atoms Cu 832`, A/K statements, `backend eam`, `lattice fcc`, zero defect
lines, builds). This file pins what is STILL refused, honestly narrowed to
the frames whose program the lift cannot stand behind:

1. a real-element frame under a dialect without metal units (core + lj):
   the program would state LJ/reduced units for a metal;
2. a real element the metal dialect's eam_potentials does not parameterise
   (Au): no metal physics exists to name in the physics block (the build's
   physics gate refuses the same species).

Frames that must keep lifting: pseudo-species `X` (the LJ bench),
multi-species real elements (the multi arm names each element and prints
Angstrom units), `symbols=None` (the byte-identical M0 contract the CLI
roundtrip uses), and -- since step 2 -- the parameterised single-species
metals (Cu/Fe/Ni) under core + metal.

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
from chaord.io.frames import Frame, read_frame  # noqa: E402
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


def _au_slab():
    """The Cu reference geometry with every atom relabelled Au: a real
    element the metal dialect does not parameterise (eam_potentials: Cu,
    Fe, Ni). The geometry is Cu's; only the species matters to the gate."""
    cu = read_frame(CU_REF)
    return Frame(pos=cu.pos, cell=cu.cell, symbols=["Au"] * len(cu.pos),
                 pbc=cu.pbc)


# -------------------------------------------------- W4 step 2: refused set --

def test_w4_real_element_under_lj_dialect_refuses():
    """Refusal 1: a real-element frame under a dialect without metal units.
    The step-1 message keeps its first line verbatim; the reason names the
    actual wrong (LJ units for a metal)."""
    frame = read_frame(CU_REF)
    with pytest.raises(ChaordError, match=W4_MESSAGE):
        lift_frame(frame, load_dialect(LJ), mode="slab")
    with pytest.raises(ChaordError, match="would be printed in LJ units"):
        lift_frame(frame, load_dialect(LJ), mode="slab")


def test_w4_unparameterised_element_under_metal_refuses():
    """Refusal 2: Au has no EAM parameters in the metal dialect -- the
    program would state `backend eam` for a species the dialect cannot
    realize (the build's physics gate refuses it too; the lift fails
    closed first)."""
    with pytest.raises(ChaordError, match=W4_MESSAGE):
        lift_frame(_au_slab(), load_dialect(METAL), mode="slab")
    with pytest.raises(ChaordError, match="no EAM parameters"):
        lift_frame(_au_slab(), load_dialect(METAL), mode="slab")


def test_w4_gate_sits_in_program_from_result():
    """decompile still measures the frame (its numbers are not the lie);
    the refusal fires where the program text would be assembled. Same
    location as the step-1 gate (W4 step 2 pins it for the narrowed set)."""
    frame = _au_slab()
    res = decompile(frame.pos, frame.cell_diag, 300.0, load_dialect(METAL),
                    symbols=list(frame.symbols))
    assert res["N"] == 832
    with pytest.raises(ChaordError, match=W4_MESSAGE):
        program_from_result(res, load_dialect(METAL))


def test_w4_parameterised_single_species_metal_now_lifts():
    """The step-1 refusal on this frame is GONE (step 2): the Cu reference
    lifts through the metal arm. The done-when assertions live in
    tests/test_metal_slab_lift.py; this pins the direction of the narrowing
    right next to the refusal set it shrank."""
    frame = read_frame(CU_REF)
    text = acc.format_program_text(lift_frame(frame, load_dialect(METAL)))
    assert "conserve atoms Cu 832" in text
    assert "units lj" not in text


def test_w4_lj_x_species_slab_still_lifts():
    """The refusal condition is about real elements: the LJ bench's
    pseudo-species `X` frames keep lifting byte-identically."""
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
    W4 governs."""
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
