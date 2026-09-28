"""M6 acceptance: the reactive interface lift (slab + molecular overlayer).

Exit criteria (PLAN M6): the species census is exact on planted cases; the
lifted program names the liquid overlayer region, its molecules, the adsorbed
single atoms and the dissociation events on the slab | overlayer interface.
"""
import re

import numpy as np
import pytest

from chaord.build import build_program
from chaord.build.molecules import TEMPLATES, random_rotation
from chaord.check.statics import conservation_check
from chaord.dialects import load_dialect
from chaord.io.frames import Frame
from chaord.lang.api import load
from chaord.lang.fmt import format_program
from chaord.lang.ir import ir_equal
from chaord.lang.parser import parse_text
from chaord.lift import lift_frame

RUTILE_110_PROGRAM = """chaord 0.1
dialect core + molecular + surface

system {
  cell 6.497 2.959 23.727
  pbc xyz
  conserve atoms Ti 8 O 16
}

physics {
  backend eam
}

crystal slab : slab z 0 .. 11.7 {
  prototype rutile
  composition TiO2
  a 4.594 A
  c 2.959 A
  surface (110) top
  termination bridging_O
}

vacuum gap : slab z 11.7 .. 23.7 {
}
"""

# overlayer geometry (A, relative to the slab top): the molecule atoms sit
# between OVER_GAP (a vacuum gap the lifter can see) and CELL_OVERHEAD minus
# OVER_CLEAR (clear of the slab bottom through the periodic wrap)
OVER_GAP = 6.5
OVER_CLEAR = 3.2
CELL_OVERHEAD = 12.7


@pytest.fixture(scope="module")
def dialect():
    return load_dialect(("core", "molecular", "surface"))


def _tile(frame, nx, ny):
    cell = frame.cell.copy()
    pos, syms = [], []
    for i in range(nx):
        for j in range(ny):
            pos.append(frame.pos + i * cell[0] + j * cell[1])
            syms.extend(frame.symbols)
    new = cell.copy()
    new[0] *= nx
    new[1] *= ny
    return Frame(pos=np.vstack(pos), cell=new, symbols=syms, pbc=frame.pbc)


@pytest.fixture(scope="module")
def rutile_slab(dialect, tmp_path_factory):
    """3x5 tiled rutile(110) slab (~20 x 14 A footprint) with space above it."""
    path = tmp_path_factory.mktemp("reactive") / "rutile110.chaord"
    path.write_text(RUTILE_110_PROGRAM)
    frame = build_program(load(path), dialect, rng=np.random.default_rng(2))
    frame = _tile(frame, 3, 5)
    cell = frame.cell.copy()
    cell[2][2] = float(frame.pos[:, 2].max()) + CELL_OVERHEAD
    return Frame(pos=frame.pos, cell=cell, symbols=frame.symbols, pbc=frame.pbc)


def _overlayer_frame(slab, counts, seed, dialect):
    """slab + randomly packed molecules floating OVER_GAP above its top layer.

    Same random-sequential rule as chaord.build.molecules.pack_molecules, but
    the centres are drawn inside a z-window of the final cell so every
    molecule stays contiguous (pack_molecules wraps atoms into its own box,
    which would split them across the box faces)."""
    rng = np.random.default_rng(seed)
    L = slab.cell_diag
    top_z = float(slab.pos[:, 2].max())
    extent = max(float(np.linalg.norm(TEMPLATES[n]["rel"], axis=1).max())
                 for n in counts)
    zc0 = top_z + OVER_GAP + extent
    zc1 = top_z + CELL_OVERHEAD - OVER_CLEAR - extent
    gap = float(dialect.threshold("packing_gap"))
    centers, radii, pos, syms = [], [], [], []
    for name, n in counts.items():
        t = TEMPLATES[name]
        placed = tries = 0
        while placed < n:
            tries += 1
            assert tries < 1000 * max(n, 1), f"cannot pack {n} {name}"
            c = np.array([rng.uniform(0, L[0]), rng.uniform(0, L[1]),
                          rng.uniform(zc0, zc1)])
            too_close = False
            for c2, r2 in zip(centers, radii):
                v = c - c2
                v -= L * np.round(v / L)
                if np.linalg.norm(v) < t["radius"] + r2 + gap:
                    too_close = True
                    break
            if too_close:
                continue
            centers.append(c)
            radii.append(t["radius"])
            pos.append(t["rel"] @ random_rotation(rng).T + c)
            syms.extend(t["symbols"])
            placed += 1
    return Frame(pos=np.vstack([slab.pos, np.vstack(pos)]), cell=slab.cell,
                 symbols=slab.symbols + syms, pbc=slab.pbc)


def _block_skeleton(program):
    """Block sequence as (kind, identity) tuples."""
    out = []
    for b in program.blocks:
        if b.t == "region":
            out.append(("region", b.phase, b.name))
        elif b.t == "interface":
            out.append(("interface", b.a, b.b))
        else:
            out.append(b.t)
    return out


def test_reactive_overlayer_lift(rutile_slab, dialect):
    """rutile(110) + H2O/OH/H overlayer lifts slab + overlayer + interface."""
    frame = _overlayer_frame(rutile_slab, {"H2O": 8, "OH": 3, "H": 3}, 5, dialect)
    program = lift_frame(frame, dialect, mode="surface")
    text = format_program(program)

    # the slab itself still lifts through the prototype route
    assert "prototype rutile" in text
    assert "surface (110) top" in text
    assert "termination bridging_O" in text

    # block order: system, physics, species, crystal slab, liquid overlayer,
    # vacuum, interface, residual, provenance
    assert _block_skeleton(program) == [
        "system", "physics", "species",
        ("region", "crystal", "slab"),
        ("region", "liquid", "overlayer"),
        ("region", "vacuum", "gap"),
        ("interface", "slab", "overlayer"),
        "residual", "provenance",
    ]

    # overlayer molecules: census formulas (Hill order writes hydroxyl HO)
    # display under their conventional names
    assert "liquid overlayer : slab z 11.7 .. 24.4" in text
    assert "molecules H2O 8" in text
    assert "molecules OH 3" in text or "molecules HO 3" in text
    # species definitions carry no source: the census identified them
    assert "molecule H2O" in text
    assert "molecule OH" in text

    # the dissociated single H atoms keep the adsorb route (one line per
    # site class), and the interface explains min(OH, H) = 3 events
    assert sum(int(n) for n in re.findall(r"adsorb H count (\d+)", text)) == 3
    assert "dissociate H2O -> OH @ surface + H @ surface count 3" in text

    # conservation is per element over the whole frame: slab + overlayer
    assert conservation_check(program, frame).passed
    counts: dict[str, int] = {}
    for s in frame.symbols:
        counts[s] = counts.get(s, 0) + 1
    expected = "conserve atoms " + " ".join(
        f"{el} {counts[el]}" for el in sorted(counts))
    assert expected in text

    # canonical text is a fixed point: fmt -> parse reproduces the IR
    assert ir_equal(program, parse_text(text, units=dialect.units))


def test_pure_water_overlayer_no_dissociation(rutile_slab, dialect):
    """Pure water over the same slab: overlayer yes, dissociate no."""
    frame = _overlayer_frame(rutile_slab, {"H2O": 12}, 5, dialect)
    program = lift_frame(frame, dialect, mode="surface")
    text = format_program(program)

    assert "liquid overlayer : slab z 11.7 .. 24.4" in text
    assert "molecules H2O 12" in text
    assert "molecule H2O" in text
    assert "dissociate" not in text
    assert "adsorb" not in text
    assert "interface slab | overlayer" in text
    assert "interface slab | gap" not in text
    assert conservation_check(program, frame).passed
    assert ir_equal(program, parse_text(text, units=dialect.units))
