"""I/O round trips through ASE: extxyz, LAMMPS data, POSCAR, CIF, plus npz."""
from pathlib import Path

import numpy as np
import pytest

from chaord.io.frames import Frame, read_frame, write_frame


@pytest.fixture
def small():
    return Frame(
        pos=np.array([[0.0, 0.0, 0.0], [1.8, 0.0, 0.0], [0.0, 1.8, 0.1]]),
        cell=np.diag([8.0, 8.0, 10.0]),
        symbols=["Cu", "Al", "Cu"],
    )


@pytest.mark.parametrize("suffix,fmt,atol,symbols_exact", [
    ("extxyz", None, 1e-8, True),
    ("vasp", None, 1e-8, True),
    ("lmp", None, 1e-6, False),   # ASE maps lammps-data species via masses: lossy
    ("cif", None, 5e-4, True),
])
def test_round_trip(small, tmp_path, suffix, fmt, atol, symbols_exact):
    path = tmp_path / f"out.{suffix}"
    write_frame(path, small)
    back = read_frame(path)
    assert len(back) == 3
    if symbols_exact:
        assert back.symbols == small.symbols
    assert np.allclose(back.pos, small.pos, atol=atol)
    assert np.allclose(back.cell_diag, small.cell_diag, atol=atol)


def test_npz_round_trip(small, tmp_path):
    path = tmp_path / "out.npz"
    write_frame(path, small)
    back = read_frame(path)
    assert np.allclose(back.pos, small.pos)
    assert back.symbols == small.symbols


def test_lammps_dump_read(small, tmp_path):
    # ASE has no lammps-dump writer; hand-write a minimal dump and read it back.
    path = tmp_path / "dump.lammpstrj"
    lines = ["ITEM: TIMESTEP", "0", "ITEM: NUMBER OF ATOMS", "3",
             "ITEM: BOX BOUNDS pp pp pp",
             "0.0 8.0", "0.0 8.0", "0.0 10.0",
             "ITEM: ATOMS id type x y z"]
    for i, (p, s) in enumerate(zip(small.pos, small.symbols), 1):
        lines.append(f"{i} {1 if s == 'Cu' else 2} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}")
    path.write_text("\n".join(lines) + "\n")
    back = read_frame(path)
    assert len(back) == 3
    assert np.allclose(back.pos, small.pos, atol=1e-5)
    assert np.allclose(back.cell_diag, small.cell_diag, atol=1e-8)
