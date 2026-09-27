"""Crystal prototypes: one registry, exact construction, no fitting.

Each prototype carries its conventional-cell basis (fractional positions with
species slots), free lattice parameters, and space group. Bases are given as
FULL conventional cells (fcc = 4 sites, fluorite = 12, ...). The rational
coordinates below are exact geometry, not thresholds, so the lines carrying
them have dialect-exempt tags per AGENTS.md. Builders construct from these
tables; the lifter identifies structures by matching against them in
Niggli-reduced fractional space.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Species slots: integers index into the composition's species list.


@dataclass
class Prototype:
    name: str                    # canonical name used in `prototype` statements
    family: str                  # lattice family keyword for `lattice` statements
    params: tuple                # free parameters, e.g. ("a",) or ("a", "c")
    spacegroup: int              # international number of the conventional cell
    default_slots: tuple = ()    # slot species when no composition is given
    n_atoms: int = 0             # atoms per conventional cell


PROTOTYPES: dict[str, Prototype] = {}


def _register(name, family, params, sg, slots, n):
    PROTOTYPES[name] = Prototype(name, family, params, sg, slots, n)
    return PROTOTYPES[name]


_register("sc", "sc", ("a",), 221, ("X",), 1)
_register("bcc", "bcc", ("a",), 229, ("X",), 2)
_register("fcc", "fcc", ("a",), 225, ("X",), 4)
_register("hcp", "hcp", ("a", "c"), 194, ("X",), 2)
_register("diamond", "diamond", ("a",), 227, ("X",), 8)
_register("rocksalt", "rocksalt", ("a",), 225, ("Na", "Cl"), 8)
_register("cscl", "cscl", ("a",), 221, ("Cs", "Cl"), 2)
_register("zincblende", "zincblende", ("a",), 216, ("Zn", "S"), 8)
_register("wurtzite", "wurtzite", ("a", "c"), 186, ("Zn", "S"), 4)
_register("fluorite", "fluorite", ("a",), 225, ("Ca", "F"), 12)
_register("perovskite", "perovskite", ("a",), 221, ("Sr", "Ti", "O"), 5)
_register("L1_2", "fcc", ("a",), 221, ("Ni", "Al"), 4)
_register("rutile", "rutile", ("a", "c"), 136, ("Ti", "O"), 6)

# dialect-exempt-begin: exact prototype geometry (rational basis, not thresholds)
_FCC = ((0.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5))
# dialect-exempt-end


def cell_matrix(name: str, params: dict) -> np.ndarray:
    if name in ("hcp", "wurtzite"):
        a, c = params["a"], params["c"]
        return np.array([[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, c]])
    if name == "rutile":
        return np.diag([params["a"], params["a"], params["c"]])
    return np.diag([params["a"]] * 3)


def basis(name: str, params: dict) -> tuple[np.ndarray, tuple]:
    pos, slots = _basis_raw(name, params)
    return np.mod(pos, 1.0), slots  # dialect-exempt: fractional wrap to [0,1)


# dialect-exempt-begin: exact prototype geometry (rational basis, not thresholds)
def _basis_raw(name: str, params: dict) -> tuple[np.ndarray, tuple]:
    """(fractional positions, species slots) of one conventional cell."""
    if name == "sc":
        return np.zeros((1, 3)), (0,)
    if name == "bcc":
        return np.array([[0, 0, 0], [0.5, 0.5, 0.5]]), (0, 0)  # dialect-exempt: exact basis
    if name == "fcc":
        return np.array(_FCC), (0,) * 4  # dialect-exempt: exact basis
    if name == "hcp":
        return np.array([[0, 0, 0], [1 / 3, 2 / 3, 0.5]]), (0, 0)  # dialect-exempt: exact basis
    if name == "diamond":
        pos = [tuple(np.array(f) + np.array([0.25, 0.25, 0.25])) for f in _FCC]  # dialect-exempt: exact basis
        return np.array(list(_FCC) + pos), (0,) * 8  # dialect-exempt: exact basis
    if name == "rocksalt":
        return np.array(list(_FCC) + [tuple(np.array(f) + np.array([0.5, 0.5, 0.5])) for f in _FCC]), \
            (0,) * 4 + (1,) * 4  # dialect-exempt: exact basis
    if name == "cscl":
        return np.array([[0, 0, 0], [0.5, 0.5, 0.5]]), (0, 1)  # dialect-exempt: exact basis
    if name == "zincblende":
        pos = [tuple(np.array(f) + np.array([0.25, 0.25, 0.25])) for f in _FCC]  # dialect-exempt: exact basis
        return np.array(list(_FCC) + pos), (0,) * 4 + (1,) * 4  # dialect-exempt: exact basis
    if name == "wurtzite":
        u = float(params.get("u", 0.375))  # dialect-exempt: ideal wurtzite u
        return (np.array([[1 / 3, 2 / 3, 0], [2 / 3, 1 / 3, 0.5],
                          [1 / 3, 2 / 3, u], [2 / 3, 1 / 3, 0.5 + u]]),
                (0, 0, 1, 1))  # dialect-exempt: exact basis in terms of u
    if name == "fluorite":
        f1 = [tuple(np.array(f) + np.array([0.25, 0.25, 0.25])) for f in _FCC]  # dialect-exempt: exact basis
        f2 = [tuple(np.array(f) + np.array([0.75, 0.75, 0.75])) for f in _FCC]  # dialect-exempt: exact basis
        return np.array(list(_FCC) + f1 + f2), (0,) * 4 + (1,) * 8  # dialect-exempt: exact basis
    if name == "perovskite":
        return (np.array([[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]]),
                (0, 1, 2, 2, 2))  # dialect-exempt: exact basis
    if name == "L1_2":
        # corner = slot 1 (minority), faces = slot 0 (majority): Ni3Al has Al at corners
        return np.array(list(_FCC)), (1, 0, 0, 0)  # dialect-exempt: exact basis
    if name == "rutile":
        u = float(params.get("u", 0.305))  # dialect-exempt: TiO2 rutile u
        return (np.array([[0, 0, 0], [0.5, 0.5, 0.5],
                          [u, u, 0], [1 - u, 1 - u, 0],
                          [0.5 + u, 0.5 - u, 0.5], [0.5 - u, 0.5 + u, 0.5]]),
                (0, 0, 1, 1, 1, 1))  # dialect-exempt: exact basis in terms of u
    raise KeyError(f"unknown prototype {name!r}")
# dialect-exempt-end
