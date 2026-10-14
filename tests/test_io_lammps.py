"""LAMMPS data/dump readers (docs/reviews/open_items_v2.md section 5,
item 1: "a LAMMPS data and dump reader that maps atom types to elements
from the data file's masses and handles triclinic and NPT boxes").

Red-first discipline: this file was written BEFORE the module existed;
the first run failed on ModuleNotFoundError (chaord.io.lammps).

Fixtures are hand-written minimal LAMMPS files (orthogonal, triclinic,
NPT varying bounds, scaled coordinates, image flags, unmapped masses,
malformed refusals) plus one end-to-end test that runs the locally
installed LAMMPS binary, generates a real dump, and reads it back.

Frame triclinic evidence: src/chaord/io/frames.py stores cell as a full
(3, 3) row-vector matrix, so tilt factors are representable and must be
populated, not dropped.
"""
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from chaord.io.lammps import LammpsTypeMap, read_lammps_data, read_lammps_dump
from chaord.lang.errors import ChaordError

LMP = Path(os.environ.get(
    "CHAORD_LMP", r"C:\ProgramData\anaconda3\Scripts\lmp.exe"))


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return p


# ---------------------------------------------------------------- fixtures

DATA_ORTHOGONAL = """\
water cluster, SPC/E-like charges, atom_style full

6 atoms
4 bonds
2 angles
2 atom types

0.0 20.0 xlo xhi
0.0 22.0 ylo yhi
0.0 20.0 zlo zhi

Masses

1 15.999
2 1.008

Atoms # full

1 1 1 -0.834 2.0 2.0 2.0
2 1 2 0.417 3.0 2.0 2.6
3 1 2 0.417 1.0 2.0 2.6
4 1 1 -0.834 6.0 6.0 6.0
5 1 2 0.417 7.0 6.0 6.6
6 1 2 0.417 5.0 6.0 6.6

Velocities

1 0.0 0.0 0.0
2 0.0 0.0 0.0
3 0.0 0.0 0.0
4 0.0 0.0 0.0
5 0.0 0.0 0.0
6 0.0 0.0 0.0

Bonds

1 1 1 2
2 1 1 3
3 1 4 5
4 1 4 6

Angles

1 1 2 1 3
2 1 5 4 6
"""

# Nonzero lower corner + tilt factors xy=1.5, xz=-0.5, yz=2.0.
# Expected cell rows (ASE/LAMMPS lower-triangular, xhi-xlo etc.):
#   [[10, 0, 0], [1.5, 11, 0], [-0.5, 2.0, 12]]
# Expected positions shifted by -(xlo, ylo, zlo) = -(1, -2, 0.5).
DATA_TRICLINIC = """\
triclinic cell fixture

2 atoms
2 atom types

1.0 11.0 xlo xhi
-2.0 9.0 ylo yhi
0.5 12.5 zlo zhi
1.5 -0.5 2.0 xy xz yz

Masses

1 12.011
2 63.546

Atoms # full

1 1 1 0.0 4.0 5.0 6.0
2 1 2 1.0 10.0 3.0 7.0
"""

# Type 2 mass 128.5 amu: nearest element (Xe, 131.293) is 2.79 amu away,
# far beyond the 0.05 amu tolerance -> unmapped, reported, not a crash.
DATA_UNMAPPED = """\
one mapped, one unmapped mass

1 atoms
2 atom types

0.0 10.0 xlo xhi
0.0 10.0 ylo yhi
0.0 10.0 zlo zhi

Masses

1 12.011
2 128.5

Atoms # full

1 1 2 0.0 1.0 1.0 1.0
"""

DUMP_NPT = """\
ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0.0 20.0
0.0 20.0
0.0 20.0
ITEM: ATOMS id type q x y z
1 1 0.5 1.0 1.0 1.0
2 2 -0.5 5.0 5.0 5.0
ITEM: TIMESTEP
100
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
-0.2 20.4
0.0 21.0
0.0 19.5
ITEM: ATOMS id type q x y z
1 1 0.5 1.1 1.0 1.0
2 2 -0.5 5.2 5.0 5.0
"""

# Third snapshot added on top of DUMP_NPT for frame_indices tests.
DUMP_THREE = DUMP_NPT + """\
ITEM: TIMESTEP
200
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
-0.4 21.0
0.0 22.0
0.0 19.0
ITEM: ATOMS id type q x y z
1 1 0.5 1.2 1.0 1.0
2 2 -0.5 5.4 5.0 5.0
"""

# Tilt 1.5 / -0.5 / 2.0; scaled coords exercise the shear maths:
# xs=0.25 ys=0.5 zs=0.75 -> 0.25*(10,0,0)+0.5*(1.5,11,0)+0.75*(-0.5,2,12)
#                          = (2.875, 7.0, 9.0)
DUMP_TRICLINIC = """\
ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp xy xz yz
0.0 10.0 1.5
0.0 11.0 -0.5
0.0 12.0 2.0
ITEM: ATOMS id type xs ys zs
1 1 0.25 0.5 0.75
2 2 1.25 0.0 0.0
"""

DUMP_SCALED_ORTHO = """\
ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS pp pp pp
0.0 20.0
0.0 22.0
0.0 20.0
ITEM: ATOMS id type xs ys zs
1 1 0.25 0.5 0.75
"""

DUMP_Q_IMAGES = """\
ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS pp pp pp
0.0 20.0
0.0 20.0
0.0 20.0
ITEM: ATOMS id type q x y z ix iy iz
1 1 0.5 3.0 4.0 5.0 -1 0 2
"""

DUMP_XU = """\
ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS pp pp pp
0.0 20.0
0.0 20.0
0.0 20.0
ITEM: ATOMS id type xu yu zu
1 1 3.0 4.0 5.0
"""

DUMP_LABELS = """\
ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS ff pp mp
0.0 20.0
0.0 20.0
0.0 20.0
ITEM: ATOMS id type x y z
1 1 1.0 1.0 1.0
"""

# Types 1/2 exist in DATA_ORTHOGONAL's Masses; type 3 does not.
DUMP_EXTRA_TYPE = """\
ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
3
ITEM: BOX BOUNDS pp pp pp
0.0 20.0
0.0 20.0
0.0 20.0
ITEM: ATOMS id type q x y z
1 1 0.0 1.0 1.0 1.0
2 2 0.0 2.0 2.0 2.0
3 3 0.0 3.0 3.0 3.0
"""

LMP_INPUT = """\
units real
atom_style full
boundary p p p
read_data start.data
pair_style lj/cut 5.0
pair_coeff 1 1 0.01 3.2
pair_coeff 2 2 0.02 2.9
pair_coeff 1 2 0.015 3.0
velocity all create 50.0 4928459 mom yes rot no dist gaussian
fix int all nvt temp 50.0 50.0 25.0
timestep 1.0
dump trj all custom 5 traj.dump id type q x y z ix iy iz
dump_modify trj sort id format float %.15g
run 20
"""

# 4 carbons (+0.3 e) and 2 oxygens (-0.6 e): charges sum to zero, all
# pair separations > 3 A (no overlap), all positions inside the box.
LMP_DATA = """\
Chaord e2e round-trip fixture

6 atoms
2 atom types

0.0 18.0 xlo xhi
0.0 16.0 ylo yhi
0.0 14.0 zlo zhi

Masses

1 12.011
2 15.999

Atoms # full

1 1 1 0.30 3.0 3.0 3.0
2 1 1 0.30 9.0 3.5 4.0
3 1 1 0.30 4.0 10.0 5.0
4 1 1 0.30 12.0 12.0 7.0
5 1 2 -0.60 8.0 9.0 10.0
6 1 2 -0.60 14.0 4.0 11.0
"""


# ------------------------------------------------------------- data file

def test_read_data_orthogonal_full_style(tmp_path):
    frame, meta = read_lammps_data(_write(tmp_path, "w.data", DATA_ORTHOGONAL))
    assert len(frame) == 6
    assert frame.symbols == ["O", "H", "H", "O", "H", "H"]
    assert np.allclose(frame.cell_diag, [20.0, 22.0, 20.0])
    assert frame.pos[0].tolist() == [2.0, 2.0, 2.0]
    assert np.allclose(meta["charges"], [-0.834, 0.417, 0.417,
                                         -0.834, 0.417, 0.417])
    assert meta["bonds"] == [(1, 1, 1, 2), (2, 1, 1, 3),
                             (3, 1, 4, 5), (4, 1, 4, 6)]
    assert meta["angles"] == [(1, 1, 2, 1, 3), (2, 1, 5, 4, 6)]
    assert np.allclose(meta["velocities"], np.zeros((6, 3)))
    assert meta["atom_style"] == "full"
    assert meta["molecule_ids"] == [1, 1, 1, 1, 1, 1]
    assert meta["image_flags"] is None
    assert meta["skipped_sections"] == []


def test_read_data_triclinic_cell_and_shift(tmp_path):
    frame, meta = read_lammps_data(
        _write(tmp_path, "t.data", DATA_TRICLINIC))
    expected_cell = np.array([[10.0, 0.0, 0.0],
                              [1.5, 11.0, 0.0],
                              [-0.5, 2.0, 12.0]])
    assert meta["triclinic"] is True
    assert np.allclose(frame.cell, expected_cell)
    # Frame carries no origin: positions are shifted by the lower corner.
    assert np.allclose(frame.pos, [[3.0, 7.0, 5.5],
                                   [9.0, 5.0, 6.5]])
    assert tuple(meta["box_lo"]) == (1.0, -2.0, 0.5)
    assert frame.symbols == ["C", "Cu"]


def test_read_data_unmapped_type_reported_not_crash(tmp_path):
    frame, meta = read_lammps_data(
        _write(tmp_path, "u.data", DATA_UNMAPPED))
    assert frame.symbols == ["X"]
    assert meta["unmapped_types"] == [2]
    assert meta["type_map"].symbol(1) == "C"
    assert meta["type_map"].symbol(2) == "X"


def test_type_map_exact_mass_and_tolerance():
    tm = LammpsTypeMap({1: 207.2, 2: 12.0, 3: 128.5})
    assert tm.symbol(1) == "Pb"     # exact table match
    assert tm.symbol(2) == "C"      # 12.0 vs 12.011: 0.011 amu off, in tol
    assert tm.symbol(3) == "X"      # nothing within 0.05 amu
    assert tm.unmapped == [3]


def test_data_atom_count_mismatch_refuses(tmp_path):
    bad = DATA_ORTHOGONAL.replace("6 atoms", "3 atoms")
    with pytest.raises(ChaordError, match="atoms"):
        read_lammps_data(_write(tmp_path, "bad.data", bad))


def test_data_malformed_atom_row_refuses(tmp_path):
    bad = DATA_ORTHOGONAL.replace(
        "6 1 2 0.417 5.0 6.0 6.6", "6 1 2 0.417")
    with pytest.raises(ChaordError, match="line 24"):
        read_lammps_data(_write(tmp_path, "bad2.data", bad))


def test_data_unparsed_sections_counted(tmp_path):
    extra = DATA_ORTHOGONAL.replace(
        "Angles\n\n1 1 2 1 3\n2 1 5 4 6\n",
        "Angles\n\n1 1 2 1 3\n2 1 5 4 6\n\n"
        "Dihedrals\n\n1 1 2 1 3 4\n\n"
        "Pair Coeffs\n\n1 0.01 3.0\n2 0.02 2.9\n")
    frame, meta = read_lammps_data(_write(tmp_path, "d.data", extra))
    # recognised-but-unstored sections are counted, never silently dropped
    assert ("Dihedrals", 1) in meta["skipped_sections"]
    assert ("Pair Coeffs", 2) in meta["skipped_sections"]
    assert len(frame) == 6


# ------------------------------------------------------------------ dump

def test_dump_npt_varying_bounds(tmp_path):
    frames = read_lammps_dump(_write(tmp_path, "npt.dump", DUMP_NPT))
    assert len(frames) == 2
    assert [f.info["timestep"] for f in frames] == [0, 100]
    assert np.allclose(frames[0].cell_diag, [20.0, 20.0, 20.0])
    assert np.allclose(frames[1].cell_diag, [20.6, 21.0, 19.5])
    # lower corner shift: snapshot 2 xlo = -0.2
    assert np.allclose(frames[1].pos[0], [1.3, 1.0, 1.0])
    assert np.allclose(frames[0].info["charges"], [0.5, -0.5])
    assert frames[0].symbols == ["X", "X"]   # dumps carry no elements


def test_dump_triclinic(tmp_path):
    frames = read_lammps_dump(_write(tmp_path, "t.dump", DUMP_TRICLINIC))
    expected_cell = np.array([[10.0, 0.0, 0.0],
                              [1.5, 11.0, 0.0],
                              [-0.5, 2.0, 12.0]])
    assert frames[0].info["triclinic"] is True
    assert np.allclose(frames[0].cell, expected_cell)
    # xs=0.25 ys=0.5 zs=0.75 -> (2.875, 7.0, 9.0) including shear terms
    assert np.allclose(frames[0].pos[0], [2.875, 7.0, 9.0])


def test_dump_scaled_orthogonal(tmp_path):
    frames = read_lammps_dump(
        _write(tmp_path, "s.dump", DUMP_SCALED_ORTHO))
    assert np.allclose(frames[0].pos, [[5.0, 11.0, 15.0]])


def test_dump_scaled_wraps_outside_unit_range(tmp_path):
    frames = read_lammps_dump(_write(tmp_path, "t.dump", DUMP_TRICLINIC))
    # xs=1.25 wraps to 0.25 -> (2.5, 0.0, 0.0)
    assert np.allclose(frames[0].pos[1], [2.5, 0.0, 0.0])


def test_dump_charges_and_image_flags(tmp_path):
    frames = read_lammps_dump(_write(tmp_path, "qi.dump", DUMP_Q_IMAGES))
    assert np.allclose(frames[0].info["charges"], [0.5])
    assert np.array_equal(frames[0].info["image_flags"], [[-1, 0, 2]])
    assert np.allclose(frames[0].pos, [[3.0, 4.0, 5.0]])


def test_dump_unwrapped_columns_refuse(tmp_path):
    with pytest.raises(ChaordError, match="unwrapped"):
        read_lammps_dump(_write(tmp_path, "xu.dump", DUMP_XU))


def test_dump_frame_indices(tmp_path):
    p = _write(tmp_path, "three.dump", DUMP_THREE)
    assert len(read_lammps_dump(p)) == 3
    sel = read_lammps_dump(p, frame_indices=[0, 2])
    assert [f.info["timestep"] for f in sel] == [0, 200]
    sl = read_lammps_dump(p, frame_indices=slice(None, None, 2))
    assert [f.info["timestep"] for f in sl] == [0, 200]
    one = read_lammps_dump(p, frame_indices=1)
    assert [f.info["timestep"] for f in one] == [100]
    with pytest.raises(ChaordError, match="out of range"):
        read_lammps_dump(p, frame_indices=5)
    with pytest.raises(ChaordError, match="out of range"):
        read_lammps_dump(p, frame_indices=[0, -7])


def test_dump_pbc_labels(tmp_path):
    frames = read_lammps_dump(_write(tmp_path, "lb.dump", DUMP_LABELS))
    assert frames[0].pbc == (False, True, False)
    # mixed labels (mp) cannot be expressed by a bool; recorded, not dropped
    assert frames[0].info["box_boundary_labels"] == ["ff", "pp", "mp"]


def test_dump_atom_count_mismatch_refuses(tmp_path):
    bad = DUMP_NPT.replace("ITEM: NUMBER OF ATOMS\n2",
                           "ITEM: NUMBER OF ATOMS\n3", 1)
    with pytest.raises(ChaordError, match="atoms"):
        read_lammps_dump(_write(tmp_path, "bad.dump", bad))


# ------------------------------------------------- type map shared to dump

def test_type_map_applies_to_dump_frames(tmp_path):
    _, meta = read_lammps_data(_write(tmp_path, "w.data", DATA_ORTHOGONAL))
    tmap = meta["type_map"]
    frames = read_lammps_dump(_write(tmp_path, "x.dump", DUMP_EXTRA_TYPE))
    unmapped = tmap.apply(frames)
    assert frames[0].symbols == ["O", "H", "X"]
    assert unmapped == [3]
    # strict mode refuses rather than leaving a hole
    with pytest.raises(ChaordError, match="type id.*3"):
        tmap.apply(read_lammps_dump(_write(tmp_path, "x2.dump",
                                           DUMP_EXTRA_TYPE)), strict=True)


# --------------------------------------------------------- end-to-end run

def test_lammps_end_to_end_round_trip(tmp_path):
    if not LMP.exists():
        pytest.skip(f"LAMMPS binary not found at {LMP}; "
                    "set CHAORD_LMP to run the end-to-end test")
    _write(tmp_path, "start.data", LMP_DATA)
    _write(tmp_path, "in.lmp", LMP_INPUT)
    proc = subprocess.run([str(LMP), "-in", "in.lmp"], cwd=tmp_path,
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, (proc.stdout[-2000:] + proc.stderr[-2000:])

    ref, meta = read_lammps_data(tmp_path / "start.data")
    tmap = meta["type_map"]
    frames = read_lammps_dump(tmp_path / "traj.dump")
    assert len(frames) == 5                       # dump every 5, run 20
    assert [f.info["timestep"] for f in frames] == [0, 5, 10, 15, 20]
    for f in frames:
        assert len(f) == 6
        assert np.allclose(f.cell_diag, [18.0, 16.0, 14.0])   # NVT: static
        assert np.allclose(f.info["charges"],
                           [0.30] * 4 + [-0.60] * 2, atol=1e-12)
        # all positions inside the (wrapped) cell
        assert (f.pos >= -1e-9).all() and (f.pos <= f.cell_diag + 1e-9).all()
    unmapped = tmap.apply(frames)
    assert unmapped == []
    assert frames[0].symbols == ["C", "C", "C", "C", "O", "O"]
    # dump at timestep 0 is the same physical state the data file describes
    assert np.allclose(frames[0].pos, ref.pos, atol=1e-8)
    # NVT at 50 K with a seeded velocity draw: the atoms actually moved
    drift = np.linalg.norm(frames[-1].pos - frames[0].pos, axis=1).max()
    assert drift > 1e-3, f"atoms did not move (max drift {drift} A)"
