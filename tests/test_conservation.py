"""Real conservation: three-way census (frame vs program-implied vs conserve).

The old check was circular - it compared the program's conserve line against
the very frame that line was written from, and residual_explained returned
True on both branches. Now:

* implied_atom_count derives per-species counts from the program text alone
  (crystal lattice parameters x supercell sites, compositions, occupancy
  fractions x site count, Kröger-Vink defect net effects, molecules lines
  via the template table or name formulas, adsorb counts, residual atom
  lines);
* conservation_check requires the frame census, the implied census and the
  conserve declaration to agree, and prints all three numbers. Density-only
  regions (amorphous composition+density, atomic liquids, ASE-cut Miller
  slabs, the M0 strain/depth grid) are declared buckets: the conserve line
  stands for their atoms and the remainder must be explainable - non-negative
  per species and attributable to a region whose statements name them;
* lift_frame registers a lift diagnostic (unexplained atom count, failing
  checks) with every program it returns, and residual_explained audits the
  residual block against it: `residual none` needs zero unexplained atoms,
  a residual block must list exactly the unexplained ones.
"""
import numpy as np
import pytest

from chaord.build import build_program
from chaord.build.crystal import build_conventional
from chaord.build.defects import apply_defects
from chaord.build.molecules import pack_molecules
from chaord.check.statics import (
    conservation_check,
    get_lift_diag,
    implied_atom_count,
    record_lift_diag,
    residual_explained,
    run_checks,
)
from chaord.dialects import load_dialect
from chaord.io.frames import Frame
from chaord.lang.api import load
from chaord.lang.ir import KVDefect, Name, Quantity, Statement
from chaord.lift import lift_frame


def _kv(tok, count):
    return Statement(kind="build", key="defect",
                     values=[KVDefect(text=tok), Name(text="count"),
                             Quantity(num=str(count))])


@pytest.fixture(scope="module")
def metal():
    return load_dialect(("core", "metal"))


@pytest.fixture(scope="module")
def molecular():
    return load_dialect(("core", "molecular"))


@pytest.fixture(scope="module")
def crystal_defect(metal):
    """L1_2 Ni3Al 3x3x3 supercell with V_Ni 2, Al_Ni 1, Ni_i 1 planted."""
    rng = np.random.default_rng(67)
    f = build_conventional("L1_2", {"a": 3.572}, ("Ni", "Al"), (3, 3, 3))
    fd = apply_defects(f, [_kv("V_Ni", 2), _kv("Al_Ni", 1), _kv("Ni_i", 1)],
                       rng, metal)
    program = lift_frame(fd, metal, mode="defects")
    return program, fd


@pytest.fixture(scope="module")
def water(molecular):
    rng = np.random.default_rng(13)
    return pack_molecules({"H2O": 60}, [15.0] * 3, rng, molecular)


@pytest.fixture(scope="module")
def surf111(tmp_path_factory):
    """fcc(111) Ni slab + 4 O adsorbates, built from a program (M4 route)."""
    dialect = load_dialect(("core", "metal", "surface"))
    text = ("chaord 0.1\ndialect core + metal + surface\n\n"
            "system {\n  cell 7.669 6.641 18.26\n  pbc xyz\n"
            "  conserve atoms Ni 36\n}\n\n"
            "physics {\n  backend eam\n}\n\n"
            "crystal slab : slab z 0 .. 6.3 {\n  lattice fcc\n  a 3.615 A\n"
            "  surface (111) top\n  adsorb O count 4 site top\n}\n\n"
            "vacuum gap : slab z 6.3 .. 18.26 {\n}\n")
    path = tmp_path_factory.mktemp("conservation") / "surf111.chaord"
    path.write_text(text)
    frame = build_program(load(path), dialect, rng=np.random.default_rng(3))
    return frame, dialect


@pytest.fixture(scope="module")
def glass(tmp_path_factory):
    """108-atom LJ glass: melt -> quench -> anneal, lifted amorphous."""
    dialect = load_dialect(("core", "glass"))
    text = ("chaord 0.1\ndialect core + glass\n\n"
            "system {\n  pbc xyz\n  conserve atoms X 108\n}\n\n"
            "physics {\n  backend lj\n}\n\n"
            "amorphous glass : all {\n  state density 0.9\n"
            "  history melt 1.2 for 150 -> quench to 0.01 at 0.01 -> "
            "anneal 0.01 for 100\n}\n")
    path = tmp_path_factory.mktemp("conservation") / "glass.chaord"
    path.write_text(text)
    frame = build_program(load(path), dialect, rng=np.random.default_rng(5))
    return frame, dialect


def _all_pass(program, frame, dialect):
    results = run_checks(program, frame, dialect)
    for r in results:
        assert r.passed, f"{r.name}: {r.detail}"
    return results


# ------------------------------------------------------------ normal paths ----

def test_crystal_defects_three_way_pass(crystal_defect, metal):
    """Sites x supercell x composition, minus defect net effects, equals the
    conserve line and the frame: Ni 81-2-1+1=79, Al 27+1=28."""
    program, fd = crystal_defect
    assert implied_atom_count(program) == {"Ni": 79, "Al": 28}
    results = _all_pass(program, fd, metal)
    con = next(r for r in results if r.name == "conservation")
    for part in ("frame", "implied", "conserve"):
        assert part in con.detail
    assert "Ni: 79" in con.detail and "Al: 28" in con.detail


def test_fluid_molecules_three_way_pass(water, molecular):
    """The molecules line pins the census: 60 H2O -> H 120, O 60."""
    frame = water
    program = lift_frame(frame, molecular, mode="fluid")
    assert implied_atom_count(program) == {"H": 120, "O": 60}
    _all_pass(program, frame, molecular)


def test_surface_adsorption_three_way_pass(surf111):
    """Adsorb counts are implied exactly; the ASE-cut slab body is a declared
    bucket whose remainder (Ni 36) must be explainable by its statements."""
    frame, dialect = surf111
    program = lift_frame(frame, dialect, mode="surface")
    assert implied_atom_count(program) == {"O": 4}
    con = conservation_check(program, frame, dialect)
    assert con.passed, con.detail
    assert "Ni: 36" in con.detail            # the declared-only remainder
    _all_pass(program, frame, dialect)


def test_amorphous_declared_bucket_pass(glass):
    """A glass has no text-derivable count (density rounds): the conserve line
    stands as the declared value and the check still runs the three censes."""
    frame, dialect = glass
    program = lift_frame(frame, dialect, mode="amorphous")
    assert implied_atom_count(program) == {}   # nothing exact to pin
    con = conservation_check(program, frame, dialect)
    assert con.passed, con.detail
    assert "X: 108" in con.detail
    _all_pass(program, frame, dialect)


def test_lift_registers_diag_and_auto_runs_checks(water, molecular):
    """Every lift registers a diagnostic; check failures are attached to it
    instead of raising (here: a deliberate overlap fails the overlap check)."""
    frame = water
    broken = Frame(pos=frame.pos.copy(), cell=frame.cell,
                   symbols=list(frame.symbols), pbc=frame.pbc)
    broken.pos[1] = broken.pos[0]             # two coincident atoms: overlap
    program = lift_frame(broken, molecular, mode="fluid")
    diag = get_lift_diag(program)
    assert diag is not None and diag["unexplained"] == 0
    assert any(f.startswith("overlap") for f in diag["failed_checks"])
    # conservation itself is unaffected by the overlap (same census)
    assert conservation_check(program, broken, molecular).passed


# -------------------------------------------------------------- mutations ----

def _conserve_stmt(program):
    for b in program.blocks:
        if b.t == "system":
            for s in b.statements:
                if s.kind == "conserve" and s.key == "atoms":
                    return s
    raise AssertionError("no conserve atoms statement")


def test_mutation_conserve_minus_one_fails(crystal_defect, metal):
    """One atom removed from the conserve line: the implied census (derived
    from the region statements) no longer matches the declaration."""
    program, fd = crystal_defect
    stmt = _conserve_stmt(program)
    first_count = next(v for v in stmt.values if v.t == "q")
    species = stmt.values[stmt.values.index(first_count) - 1].text
    first_count.num = str(int(first_count.num) - 1)
    result = conservation_check(program, fd, metal)
    assert not result.passed
    assert species in result.detail
    assert "imply" in result.detail and "conserve states" in result.detail
    # and the full check suite fails on it too
    assert not all(r.passed for r in run_checks(program, fd, metal))


def test_mutation_frame_minus_one_atom_fails(crystal_defect, metal):
    """One atom removed from the frame: conserve (and implied) disagree with
    what the structure actually contains."""
    program, fd = crystal_defect
    i = fd.symbols.index("Ni")
    short = Frame(pos=np.delete(fd.pos, i, axis=0), cell=fd.cell,
                  symbols=fd.symbols[:i] + fd.symbols[i + 1:], pbc=fd.pbc)
    result = conservation_check(program, short, metal)
    assert not result.passed
    assert "frame has" in result.detail
    assert "Ni: 78" in result.detail


def test_mutation_residual_none_with_unexplained_fails(crystal_defect):
    """residual none while the lift diagnostic says 3 atoms went unexplained."""
    program, fd = crystal_defect
    record_lift_diag(program, {"unexplained": 3})
    result = residual_explained(program, fd)
    assert not result.passed
    assert "3" in result.detail and "unexplained" in result.detail


def test_mutation_residual_lines_mismatch_unexplained(crystal_defect):
    """A residual block must list exactly the atoms the lift could not
    explain: 2 lines against 3 unexplained is a conservation failure."""
    from chaord.lang.ir import Quantity as Q, ResidualBlock
    program, fd = crystal_defect
    lines = [Statement(kind="build", key="atom",
                       values=[Name(text="Ni"), Q(num="0.00"), Q(num="0.00"),
                               Q(num="0.00")])
             for _ in range(2)]
    program.blocks = [ResidualBlock(none=False, statements=lines)
                      if b.t == "residual" else b for b in program.blocks]
    record_lift_diag(program, {"unexplained": 3})
    result = residual_explained(program, fd)
    assert not result.passed
    assert "2 residual atoms" in result.detail and "3 unexplained" in result.detail
