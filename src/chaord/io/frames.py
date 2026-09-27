"""Structure I/O: one Frame type, ASE underneath, prototype npz supported."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Frame:
    pos: np.ndarray            # (N, 3)
    cell: np.ndarray           # (3, 3) row vectors
    symbols: list
    pbc: tuple = (True, True, True)
    info: dict = field(default_factory=dict)

    @property
    def cell_diag(self) -> np.ndarray:
        """Diagonal of the (assumed orthogonal) cell."""
        return np.array([self.cell[0, 0], self.cell[1, 1], self.cell[2, 2]])

    def __len__(self):
        return len(self.pos)


def from_ase(atoms) -> Frame:
    return Frame(
        pos=np.asarray(atoms.get_positions(), float),
        cell=np.asarray(atoms.get_cell().array, float),
        symbols=list(atoms.get_chemical_symbols()),
        pbc=tuple(bool(x) for x in atoms.get_pbc()),
        info=dict(atoms.info),
    )


def to_ase(frame: Frame):
    from ase import Atoms
    return Atoms(
        symbols=frame.symbols,
        positions=frame.pos,
        cell=frame.cell,
        pbc=frame.pbc,
    )


def _read_npz(path: Path) -> Frame:
    """Prototype snapshot format: arrays r (N,3), L (3,), optional v."""
    z = np.load(path)
    r = np.asarray(z["r"], float)
    L = np.asarray(z["L"], float)
    symbols = ["X"] * len(r)
    if "symbols" in z:
        symbols = [str(s) for s in z["symbols"]]
    return Frame(pos=r, cell=np.diag(L), symbols=symbols,
                 pbc=(True, True, True))


def read_frame(path, index: int = -1) -> Frame:
    path = Path(path)
    if path.suffix == ".npz":
        return _read_npz(path)
    from ase.io import read as ase_read
    fmt = _FORMATS.get(path.suffix.lower())
    atoms = ase_read(str(path), index=index, format=fmt)
    if isinstance(atoms, list):
        atoms = atoms[index]
    return from_ase(atoms)


_FORMATS = {
    ".extxyz": "extxyz",
    ".xyz": "extxyz",
    ".lammpstrj": "lammps-dump-text",
    ".dump": "lammps-dump-text",
    ".lmp": "lammps-data",
    ".data": "lammps-data",
    ".vasp": "vasp",
    ".poscar": "vasp",
    ".cif": "cif",
}


def write_frame(path, frame: Frame, fmt: str | None = None) -> None:
    path = Path(path)
    if path.suffix == ".npz":
        np.savez(path, r=frame.pos, L=frame.cell_diag,
                 symbols=np.array(frame.symbols, dtype="U8"))
        return
    from ase.io import write as ase_write
    fmt = fmt or _FORMATS.get(path.suffix.lower())
    atoms = to_ase(frame)
    if fmt in ("lammps-data", "lammps-dump-text"):
        atoms.set_chemical_symbols([s if s != "X" else "X" for s in frame.symbols])
    ase_write(str(path), atoms, format=fmt)
