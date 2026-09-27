"""M1 acceptance: builders, exact crystal round trip, and the invariance suite.

Exit criteria (PLAN M1): exact round trip on every crystal case; identical text
under rotation, translation, re-ordering and re-imaging.
"""
import numpy as np
import pytest
import spglib

from chaord.build import build_program
from chaord.build.crystal import build_conventional
from chaord.build.prototypes import PROTOTYPES
from chaord.dialects import load_dialect
from chaord.io.frames import Frame
from chaord.lang.api import load, save
from chaord.lang.fmt import format_program
from chaord.lang.parser import parse_text
from chaord.lift import lift_frame

CASES = [
    ("fcc", {"a": 3.615}, ("Cu",)),
    ("bcc", {"a": 2.87}, ("Fe",)),
    ("hcp", {"a": 3.21, "c": 5.21}, ("Mg",)),
    ("diamond", {"a": 5.43}, ("Si",)),
    ("rocksalt", {"a": 5.64}, ("Na", "Cl")),
    ("cscl", {"a": 4.11}, ("Cs", "Cl")),
    ("zincblende", {"a": 5.41}, ("Zn", "S")),
    ("wurtzite", {"a": 3.25, "c": 5.2}, ("Zn", "S")),
    ("fluorite", {"a": 5.46}, ("Ca", "F")),
    ("perovskite", {"a": 3.905}, ("Sr", "Ti", "O")),
    ("L1_2", {"a": 3.572}, ("Ni", "Al")),
    ("rutile", {"a": 4.594, "c": 2.959}, ("Ti", "O")),
]


@pytest.fixture(scope="module")
def dialect():
    return load_dialect(("core", "metal"))


def _built(name, params, slots, reps=(2, 2, 2)):
    return build_conventional(name, params, slots, reps)


@pytest.mark.parametrize("name,params,slots", CASES, ids=[c[0] for c in CASES])
def test_builder_atom_counts(name, params, slots):
    f = _built(name, params, slots)
    assert len(f) == PROTOTYPES[name].n_atoms * 8


@pytest.mark.parametrize("name,params,slots", CASES, ids=[c[0] for c in CASES])
def test_spacegroup_cross_check(name, params, slots, dialect):
    """Cross-check with spglib (nightly cross-check layer, run here per case)."""
    f = _built(name, params, slots)
    frac = f.pos @ np.linalg.inv(f.cell)
    numbers = [PROTOTYPES and __import__("ase.data", fromlist=["chemical_symbols"]).chemical_symbols.index(s)
               for s in f.symbols]
    ds = spglib.get_symmetry_dataset((f.cell.tolist(), frac.tolist(), numbers),
                                     symprec=1e-4)
    assert ds is not None
    assert ds.number == PROTOTYPES[name].spacegroup, f"{name}: {ds.number}"


@pytest.mark.parametrize("name,params,slots", CASES, ids=[c[0] for c in CASES])
def test_exact_round_trip(name, params, slots, dialect, tmp_path):
    """lift -> build -> lift gives byte-identical canonical text."""
    f = _built(name, params, slots)
    text1 = format_program(lift_frame(f, dialect))
    path = tmp_path / "p.chaord"
    path.write_text(text1)
    f2 = build_program(load(path), dialect)
    text2 = format_program(lift_frame(f2, dialect))
    assert text2 == text1


def _random_rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _transform(frame, rng):
    R = _random_rotation(rng)
    t = rng.uniform(-5, 5, 3)
    pos = frame.pos @ R.T + t
    cell = frame.cell @ R.T
    pos = pos @ np.linalg.inv(cell)          # re-image (fractional wrap)
    pos = np.mod(pos, 1.0) @ cell
    perm = rng.permutation(len(pos))         # re-order
    return Frame(pos=pos[perm], cell=cell, symbols=[frame.symbols[i] for i in perm])


@pytest.mark.parametrize("name,params,slots", CASES, ids=[c[0] for c in CASES])
def test_invariance_rotation_translation_reorder_reimage(name, params, slots, dialect):
    f = _built(name, params, slots)
    text1 = format_program(lift_frame(f, dialect))
    rng = np.random.default_rng(hash(name) % (2**32))
    for _ in range(3):
        f2 = _transform(f, rng)
        text2 = format_program(lift_frame(f2, dialect))
        assert text2 == text1


def test_replication_scales_counts(dialect, tmp_path):
    """Replicating the crystal scales counts; intensive lines stay unchanged."""
    f = _built("fcc", {"a": 3.615}, ("Cu",), (2, 2, 2))
    g = _built("fcc", {"a": 3.615}, ("Cu",), (4, 4, 4))
    t1 = format_program(lift_frame(f, dialect))
    t2 = format_program(lift_frame(g, dialect))
    def line(text, key):
        return next(l for l in text.splitlines() if l.strip().startswith(key))
    assert line(t1, "conserve") != line(t2, "conserve")          # counts scale
    assert line(t1, "lattice") == line(t2, "lattice")            # intensive lines fixed
    assert line(t1, "a ") == line(t2, "a ")


def test_lattice_mismatch_is_a_static_error(dialect, tmp_path):
    text = ("chaord 0.1\n\nsystem {\n  cell 7.5 7.5 7.5\n  pbc xyz\n}\n\n"
            "physics {\n  backend eam\n}\n\n"
            "crystal bulk : all {\n  lattice fcc\n  a 3.615 A\n}\n")
    path = tmp_path / "bad.chaord"
    path.write_text(text)
    from chaord.lang.errors import ChaordError
    with pytest.raises(ChaordError, match="integer multiple"):
        build_program(load(path), dialect)


def test_lift_via_cli_end_to_end(tmp_path, dialect):
    import subprocess, sys
    from pathlib import Path
    root = Path(__file__).parent.parent
    f = _built("L1_2", {"a": 3.572}, ("Ni", "Al"))
    struct = tmp_path / "ni3al.extxyz"
    from chaord.io.frames import write_frame
    write_frame(struct, f)
    out = tmp_path / "lifted.chaord"
    r = subprocess.run([sys.executable, "-m", "chaord.cli", "lift", str(struct),
                        "-o", str(out), "--dialect", "core+metal", "--mode", "crystal"],
                       capture_output=True, text=True, cwd=root)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "prototype L1_2" in out.read_text()
    assert "composition Ni3Al" in out.read_text()
