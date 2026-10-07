"""W12 (leftovers): `_a5_rebuild_meta` crashed on the cu_solid_liquid case.

The crash (reproduced 2026-10-02, pre-fix, both shapes below):

    chaord.lang.errors.ChaordError: dialect 'core + metal' defines no
    threshold 'md'

`_a5_rebuild_meta` counts a rebuild's MD steps by looking up the dialect's
own protocol tables (`md` / `eam_md`, and for history protocols
`quench_default_rate` + `md["quench_max_steps"]`). The metal dialect defines
only `eam_md` -- no `md`, no quench rate, no quench cap -- so any
core+metal program whose physics block names the `lj` or `eam` backend with a
history line kills the metadata extraction with a bare configuration error
from deep inside a threshold lookup. That is exactly the cu_solid_liquid
shape: the M0-generic interface lift of the Cu frame printed an LJ-unit
program (`backend lj`, the species printed as 'X' -- see the W4 step 2
refusal in lift/slab.py) under the core+metal dialect the case's floor
record names; the two call sites (check_a5's reference loop and bench loop,
~lines 2052/2171) sit outside any try/except, so the whole criterion died
instead of recording a row. Today the lift itself refuses the case (W4 step
2) and the provenance's known_limitation excludes it from A5, so the crash
is latent -- but the function is still a landmine for the first metal
program that reaches it.

W12 accepts either a fix or a clean refusal; a real fix needs MD protocol
tables in the metal dialect (a dialect-threshold change, human approval per
AGENTS.md), so the function now raises ONE ChaordError that states the
reason: the program's backend has no protocol table under this dialect, MD
metadata cannot be stated, and what to do about it.

The cu_solid_liquid program text below is the recorded shape of the case:
its cell (14.46 x 14.46 x 49.872 A), atom count (832 Cu) and equilibrium
temperature (1335 K) come from bench/reference/cu_solid_liquid's frames and
provenance.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools import acceptance as acc                    # noqa: E402
from chaord.dialects import load_dialect              # noqa: E402
from chaord.lang.errors import ChaordError            # noqa: E402

METAL = load_dialect(("core", "metal"))
LJ = load_dialect(("core", "lj"))

# the cu_solid_liquid shape: an interface program under core + metal whose
# physics block names an MD backend (the M0-generic lift printed LJ units)
CU_SOLID_LIQUID_LJ = (
    "chaord 0.1\ndialect core + metal\n\n"
    "system {\n"
    "  cell 14.460 14.460 49.872\n"
    "  pbc xyz\n"
    "  state T 1335.00\n"
    "  conserve atoms Cu 832\n"
    "}\n\n"
    "physics {\n"
    "  backend lj\n"
    "}\n\n"
    "interface cu : all {\n"
    "  ...\n"
    "}\n\n"
    "residual none\n")

# the same dialect with the OTHER crash surface: an eam backend plus a
# history protocol -- _a5_history_md_steps reads quench_default_rate and
# md["quench_max_steps"], neither of which exists in the metal dialect
CU_SOLID_LIQUID_EAM_HISTORY = CU_SOLID_LIQUID_LJ.replace(
    "backend lj", "backend eam").replace(
    "  ...\n",
    "  ...\n  history melt 2800 for 500 -> quench to 1335\n")


def test_a5_rebuild_meta_cu_solid_liquid_backend_lj_raises_reason():
    """The W12 crash, shape 1: `backend lj` under core + metal (no `md`
    table). Must refuse with a ChaordError STATING THE REASON -- not the
    bare 'defines no threshold' configuration error from the lookup."""
    with pytest.raises(ChaordError) as ei:
        acc._a5_rebuild_meta(CU_SOLID_LIQUID_LJ, METAL, physics_on=True)
    msg = str(ei.value)
    assert "cu_solid_liquid" in msg or "rebuild" in msg.lower(), msg
    assert "backend 'lj'" in msg, msg
    assert "md" in msg and "protocol" in msg.lower(), (
        "the reason must name the missing MD protocol table")


def test_a5_rebuild_meta_cu_solid_liquid_eam_history_raises_reason():
    """The W12 crash, shape 2: `backend eam` + history under core + metal
    (quench_default_rate / quench cap missing). Same clean refusal."""
    with pytest.raises(ChaordError) as ei:
        acc._a5_rebuild_meta(CU_SOLID_LIQUID_EAM_HISTORY, METAL,
                             physics_on=True)
    msg = str(ei.value)
    assert "backend 'eam'" in msg, msg
    assert "protocol" in msg.lower(), msg


@pytest.mark.parametrize("text", [CU_SOLID_LIQUID_LJ,
                                  CU_SOLID_LIQUID_EAM_HISTORY],
                         ids=["backend-lj", "eam-history"])
def test_a5_rebuild_meta_physics_off_needs_no_tables(text):
    """The physics-off mutation reads no protocol tables (md_steps is 0 by
    construction), so the cu_solid_liquid shape must NOT raise there --
    before AND after the fix."""
    meta = acc._a5_rebuild_meta(text, METAL, physics_on=False)
    assert meta["temperature"] == pytest.approx(1335.0)
    assert meta["md_steps"] == 0


def test_a5_rebuild_meta_still_reads_working_dialects():
    """Regression pins: the lj path (dialect HAS the tables) keeps extracting
    temperature, backend and step counts -- relax table without history, and
    the history arithmetic with a stated quench rate."""
    prog = ("chaord 0.1\ndialect core + lj\n\n"
            "system {\n  cell 20 20 20\n  pbc xyz\n  state T 0.72\n}\n\n"
            "physics {\n  backend lj\n}\n")
    meta = acc._a5_rebuild_meta(prog, LJ, physics_on=True)
    assert meta["temperature"] == pytest.approx(0.72)
    assert meta["backend"] == "lj"
    md = LJ.threshold("md")
    assert meta["md_steps"] == (int(md.get("relax_steps_fast", 0))
                                + int(md.get("relax_steps", 0)))
    hist = prog.replace("physics {\n  backend lj\n}",
                        "physics {\n  backend lj\n}\n\n"
                        "liquid x : all {\n"
                        "  history melt 0.90 for 300 -> quench to 0.10 "
                        "at 0.01\n}")
    steps = acc._a5_rebuild_meta(hist, LJ, physics_on=True)["md_steps"]
    assert steps == 300 + 80, \
        "melt for 300 + quench |0.9-0.1|/0.01 = 80 (rate stated, no default)"
