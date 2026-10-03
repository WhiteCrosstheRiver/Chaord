"""O11 (Reviews 4-7): the metal stack must define `printed_cutoff`, and the
cu_solid_liquid reference must lift (some mode) and build.

Before the key: every mode failed -- `auto` routed the frame to the
molecular-fluid check (extended bonded component, 832 Cu atoms) and the slab
route died on the missing threshold lookup before any fitting happened.
After: the slab route lifts and builds (auto falls through to it). Recorded
honestly: the lifted program is the M0-generic form (`X`, LJ units) -- a
species-aware metal interface lift is future work, not part of O11."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import acceptance as acc  # noqa: E402
from chaord.build import build_program  # noqa: E402
from chaord.dialects import load_dialect  # noqa: E402
from chaord.io.frames import read_frame  # noqa: E402
from chaord.lang.parser import parse_text  # noqa: E402
from chaord.lift import lift_frame  # noqa: E402

FRAME = ROOT / "bench" / "reference" / "cu_solid_liquid" / "frame_0.npz"


def test_metal_dialect_defines_printed_cutoff():
    dl = load_dialect(("core", "metal"))
    assert float(dl.threshold("printed_cutoff")) == pytest.approx(3.25)


def test_cu_solid_liquid_lifts_and_builds():
    dl = load_dialect(("core", "metal"))
    frame = read_frame(FRAME)
    text = acc.format_program_text(lift_frame(frame, dl, mode="slab"))
    assert "crystal" in text and "liquid" in text
    rebuilt = build_program(parse_text(text), dl,
                            rng=np.random.default_rng(5), physics=False)
    assert len(rebuilt) == 832


def test_cu_solid_liquid_auto_lifts():
    """Auto mode must not die on the molecular-fluid guard for a pure metal:
    it falls through to the slab route (which the printed_cutoff key
    unblocked)."""
    dl = load_dialect(("core", "metal"))
    frame = read_frame(FRAME)
    text = acc.format_program_text(lift_frame(frame, dl))
    assert len(text) > 0
