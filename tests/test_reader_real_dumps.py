"""D34 (docs/reviews/open_items_v4.md, work order Y4): the reader reads
real dumps.

Red-first suite against the binding decision:

1. Wrap positions on periodic axes (fractional modulo 1), record the
   largest excursion in box lengths in ``frame.info``, refuse only beyond
   half a box length (the corrupt-data bound).  LAMMPS rigid-body dumps
   legitimately carry atoms a few 1e-3 box lengths outside the box between
   re-neighbouring rewraps.
2. Accept the unwrapped columns ``xu yu zu`` / ``xsu ysu zsu``: wrap them
   and keep the unwrapped copy in ``frame.info`` for transport analysis.
3. Ambiguous masses raise unless an explicit ``type_map`` is given (both
   readers accept one): two elements inside the 0.05 amu window, or a
   united-atom CHn mass that collides with an element (CH2 14.027 vs N
   14.007; CH4 16.043 vs O 15.999).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]

from chaord.io.lammps import read_lammps_data, read_lammps_dump  # noqa: E402
from chaord.lang.errors import ChaordError                       # noqa: E402

RUNS = ROOT / "research" / "cnt_electrolyte" / "runs"
SYSTEMS = ("cnt10_field0", "cnt10_field1", "cnt16_field0", "cnt16_field1")


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return p


# ------------------------------------------------------- the real prod dumps --

@pytest.mark.parametrize("system", SYSTEMS)
def test_prod_dump_reads_21_wrapped_frames(system):
    """The Y4 reproducer, all four systems: the committed production dumps
    read without any scratch preprocessing (today the reader refuses on
    out-of-box coordinates between rewraps)."""
    frames = read_lammps_dump(RUNS / system / "prod.dump")
    assert len(frames) == 21
    L = np.array([f.cell_diag for f in frames])
    for f in frames:
        # wrapped invariant: everything inside the cell (float-text fuzz ok)
        assert (f.pos >= -1e-8).all() and (f.pos <= f.cell_diag + 1e-8).all()
        e = f.info["max_fractional_excursion"]
        assert 0.0 <= e < 0.01, e    # measured max over the four dumps: 8.2e-3
    assert len(set(L[:, 0].tolist())) == 1   # NVT/NPT-free pilot: static cell


# ------------------------------------------------------------- wrapping rules

DUMP_WRAPPED_OUT = """\
ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0.0 10.0
0.0 10.0
0.0 10.0
ITEM: ATOMS id type x y z
1 1 1.0 1.0 1.0
2 2 12.5 -0.25 5.0
"""

DUMP_WRAPPED_FAR_OUT = """\
ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS pp pp pp
0.0 10.0
0.0 10.0
0.0 10.0
ITEM: ATOMS id type x y z
1 1 16.0 1.0 1.0
"""

DUMP_XU_BEYOND = """\
ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
2.0 12.0
0.0 10.0
0.0 10.0
ITEM: ATOMS id type xu yu zu
1 1 1.5 2.0 3.0
2 2 23.0 -4.0 5.0
"""

DUMP_XSU_BEYOND = """\
ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
2.0 12.0
0.0 10.0
0.0 10.0
ITEM: ATOMS id type xsu ysu zsu
1 1 -0.05 0.2 0.3
2 2 2.1 1.4 0.5
"""


def test_wrapped_columns_wrap_and_record_the_excursion(tmp_path):
    frames = read_lammps_dump(_write(tmp_path, "w.dump", DUMP_WRAPPED_OUT))
    f = frames[0]
    # 12.5 -> 2.5, -0.25 -> 9.75 (fractional mod 1 on periodic axes)
    assert np.allclose(f.pos, [[1.0, 1.0, 1.0], [2.5, 9.75, 5.0]])
    assert f.info["max_fractional_excursion"] == pytest.approx(0.25)


def test_wrapped_columns_refuse_beyond_half_a_box(tmp_path):
    with pytest.raises(ChaordError, match="half a box"):
        read_lammps_dump(_write(tmp_path, "far.dump", DUMP_WRAPPED_FAR_OUT))


def test_xu_columns_round_trip(tmp_path):
    """xu/yu/zu: positions wrap into the cell, the unwrapped copy rides in
    frame.info for transport analysis (D34: dumps with image-aware
    unwrapped coordinates are now first-class input).  Frame coordinates
    carry no origin (lo is subtracted), so the wrapped x of atom 1 is
    (1.5 - 2) mod 10 = 9.5."""
    frames = read_lammps_dump(_write(tmp_path, "xu.dump", DUMP_XU_BEYOND))
    f = frames[0]
    assert np.allclose(f.pos, [[9.5, 2.0, 3.0], [1.0, 6.0, 5.0]])
    assert (f.pos >= -1e-8).all() and (f.pos <= f.cell_diag + 1e-8).all()
    # the unwrapped copy, in the frame's origin convention (shifted by -lo)
    assert np.allclose(f.info["unwrapped_pos"], [[-0.5, 2.0, 3.0],
                                                 [21.0, -4.0, 5.0]])


def test_xsu_columns_round_trip(tmp_path):
    frames = read_lammps_dump(_write(tmp_path, "xsu.dump", DUMP_XSU_BEYOND))
    f = frames[0]
    # frac -0.05 wraps to 0.95 -> 9.5 in the origin-less frame coordinates
    assert np.allclose(f.pos, [[9.5, 2.0, 3.0], [1.0, 4.0, 5.0]])
    assert np.allclose(f.info["unwrapped_pos"], [[-0.5, 2.0, 3.0],
                                                 [21.0, 14.0, 5.0]])


def test_data_file_wraps_and_records(tmp_path):
    """Data files get the same periodic wrap (their boundary is assumed
    periodic anyway) with the excursion recorded in frame.info."""
    data = """\
slightly out
1 atoms
1 atom types

0.0 10.0 xlo xhi
0.0 10.0 ylo yhi
0.0 10.0 zlo zhi

Masses

1 12.011

Atoms # full

1 1 1 0.0 10.2 5.0 5.0
"""
    frame, _ = read_lammps_data(_write(tmp_path, "d.data", data))
    assert np.allclose(frame.pos, [[0.2, 5.0, 5.0]])
    assert frame.info["max_fractional_excursion"] == pytest.approx(0.02)


# ------------------------------------------------------------ ambiguous masses

DATA_CH2_MASS = """\
united-atom CH2 force field
2 atoms
2 atom types

0.0 30.0 xlo xhi
0.0 30.0 ylo yhi
0.0 30.0 zlo zhi

Masses

1 14.027
2 1.008

Atoms # full

1 1 1 0.0 5.0 5.0 5.0
2 1 2 0.0 7.0 5.0 5.0
"""


def test_ch2_mass_raises_without_type_map(tmp_path):
    """CH2 (14.027) is 0.020 amu from N (14.007): today it silently becomes
    nitrogen; D34 makes it refuse and ask for the map."""
    with pytest.raises(ChaordError, match="type_map"):
        read_lammps_data(_write(tmp_path, "ch2.data", DATA_CH2_MASS))


def test_ch2_mass_reads_with_explicit_type_map(tmp_path):
    frame, meta = read_lammps_data(_write(tmp_path, "ch2.data", DATA_CH2_MASS),
                                   type_map={1: "C", 2: "H"})
    assert frame.symbols == ["C", "H"]
    assert meta["type_map"].symbol(1) == "C"   # shareable with dumps


def test_ch4_mass_raises_without_type_map(tmp_path):
    bad = DATA_CH2_MASS.replace("1 14.027", "1 16.043")
    with pytest.raises(ChaordError, match="type_map"):
        read_lammps_data(_write(tmp_path, "ch4.data", bad))
    frame, _ = read_lammps_data(_write(tmp_path, "ch4.data", bad),
                                type_map={1: "C", 2: "H"})
    assert frame.symbols == ["C", "H"]


def test_two_elements_inside_the_window_raise(tmp_path):
    """Bi (208.9804) and Po (208.98243) are 0.002 amu apart in the ase
    table: a mass between them has two candidates inside the window."""
    two = DATA_CH2_MASS.replace("1 14.027", "1 208.9812")
    with pytest.raises(ChaordError, match="Bi.*Po|Po.*Bi"):
        read_lammps_data(_write(tmp_path, "bipo.data", two))
    frame, _ = read_lammps_data(_write(tmp_path, "bipo.data", two),
                                type_map={1: "Po", 2: "H"})
    assert frame.symbols == ["Po", "H"]


def test_exact_element_mass_still_wins_over_the_united_atom(tmp_path):
    """A data file that writes true nitrogen (14.007, the ase value) maps
    to N, not an ambiguity: exact table matches always win (the reader
    must not refuse the elements it is named after)."""
    ok = DATA_CH2_MASS.replace("1 14.027", "1 14.007")
    frame, _ = read_lammps_data(_write(tmp_path, "n.data", ok))
    assert frame.symbols == ["N", "H"]


def test_dump_reader_accepts_type_map(tmp_path):
    """Both readers accept type_map (D34 step 3): the dump reader writes
    the symbols at read time instead of leaving X."""
    frames = read_lammps_dump(_write(tmp_path, "w.dump", DUMP_WRAPPED_OUT),
                              type_map={1: "O", 2: "H"})
    assert frames[0].symbols == ["O", "H"]
    with pytest.raises(ChaordError, match="symbol"):
        read_lammps_dump(_write(tmp_path, "w2.dump", DUMP_WRAPPED_OUT),
                         type_map={1: "NotAnElement", 2: "H"})
