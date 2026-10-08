"""W8 (docs/reviews/open_items_v2.md, section W8): molecular contact floor
and the builder's grid + minimise fallback.

Step 2's done-when, verbatim from the review: 60 CO2 at 1.00 g/cm3 -- the
lifted co2_dense program with its cell and density rewritten, the review's
own reproducer pattern -- builds on 20 of 20 seeds with physics off (before
the fix the whole-molecule RSA jams on 0 of 3 seeds: 'cannot place 60 CO2
at this density'), AND every intermolecular contact of every built frame
clears the step-1 floor (0.75 x the Bondi sum on heavy-atom pairs,
H...O/N >= 1.5 A; both read from the molecular dialect by name; radii:
the Bondi set shipped in ase.data.vdw_radii, A. Bondi, J. Phys. Chem. 68,
441 (1964)).

The floor's one definition is chaord.build.molecules.contact_floor_target
(house rule 7): the builder's pack-then-relax placement, the bench sanity
check of tests/test_bench_sanity.py and these tests all ask the same
function.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chaord.build import build_program  # noqa: E402
from chaord.build.molecules import (  # noqa: E402
    PackingJam, bond_graph, contact_floor_target, grid_pack_molecules,
    molecule_census, molecular_mass, pack_molecules)
from chaord.dialects import load_dialect  # noqa: E402
from chaord.io.frames import read_frame  # noqa: E402
from chaord.lang.ir import Quantity  # noqa: E402
from chaord.lift import lift_frame  # noqa: E402

CO2 = ROOT / "bench" / "data" / "gases" / "co2_dense"


def _components(frame, dialect):
    edges = bond_graph(frame, dialect)
    n = len(frame)
    if not edges:
        return np.arange(n)
    rows = [e[0] for e in edges] + [e[1] for e in edges]
    cols = [e[1] for e in edges] + [e[0] for e in edges]
    _, labels = connected_components(
        coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)),
        directed=False)
    return labels


def _worst_intermolecular_margin(frame, dialect) -> tuple[float, str]:
    """(worst d - floor over constrained intermolecular pairs, pair name)."""
    L = frame.cell_diag
    pos = np.mod(frame.pos, L)
    pos = np.minimum(pos, L * (1 - 1e-9))
    syms = frame.symbols
    labels = _components(frame, dialect)
    present = sorted(set(syms))
    rmax = max(contact_floor_target(a, b, dialect) or 0.0
               for a in present for b in present) + 1e-6
    tree = cKDTree(pos, boxsize=L)
    pairs = tree.query_pairs(rmax, output_type="ndarray")
    worst, pair = 0.0, "no constrained intermolecular pair"
    for i, j in pairs:
        if labels[i] == labels[j]:
            continue
        target = contact_floor_target(syms[i], syms[j], dialect)
        if target is None:
            continue
        d = pos[j] - pos[i]
        d -= L * np.round(d / L)
        margin = float(np.linalg.norm(d)) - target
        if margin < worst:
            worst, pair = margin, f"{syms[i]}-{syms[j]}"
    return worst, pair


def _co2_program_at(dl, density_g_cm3: float):
    """The lifted co2_dense program rewritten to the same 60 CO2 at the given
    mass density (the review's reproducer: cell scaled from the frame's, the
    density statement restated)."""
    frame = read_frame(CO2 / "frame_0.npz")
    gt = json.loads((CO2 / "ground_truth.json").read_text(encoding="utf-8"))
    stated = float(gt["density_target_g_cm3"])
    program = lift_frame(frame, dl, mode="fluid")
    L1 = float(frame.cell_diag[0]) * (stated / density_g_cm3) ** (1 / 3)
    for block in program.blocks:
        for s in getattr(block, "statements", []):
            if s.key == "cell":
                s.values = [Quantity(num=f"{L1:.3f}")
                            if isinstance(v, Quantity) else v
                            for v in s.values]
            if s.key == "density":
                s.values = [Quantity(num=f"{density_g_cm3:.2f}",
                                     unit=getattr(v, "unit", None))
                            if isinstance(v, Quantity) else v
                            for v in s.values]
    return program


def test_whole_molecule_rsa_still_jams_at_liquid_co2_density():
    """The red precondition, kept green: 60 CO2 at 1.00 g/cm3 is past the
    whole-molecule RSA ceiling (the census-safe sphere packing fraction
    exceeds the random-sequential saturation limit), so pack_molecules must
    still refuse -- the grid + minimise fallback is the recovery, not a
    loosened RSA."""
    dl = load_dialect(("core", "molecular"))
    edge = (60 * molecular_mass("CO2") / 1.00 / 0.6022140857) ** (1 / 3)
    with pytest.raises(PackingJam):
        pack_molecules({"CO2": 60}, [edge] * 3,
                       np.random.default_rng(0), dl)


def test_dense_co2_builds_on_20_seeds_and_clears_the_contact_floor():
    """W8 step 2 done-when: 60 CO2 at 1.00 g/cm3 build on 20 of 20 seeds
    (physics off), every intermolecular contact above the 0.75x-Bondi floor,
    and the census of every built frame is exact (the fallback moves rigid
    molecules, it never fuses or drops one)."""
    dl = load_dialect(("core", "molecular"))
    program = _co2_program_at(dl, 1.00)
    worst_overall = np.inf
    for seed in range(20):
        built = build_program(program, dl,
                              rng=np.random.default_rng(seed), physics=False)
        assert molecule_census(built, dl) == {"CO2": 60}, f"seed {seed}"
        margin, pair = _worst_intermolecular_margin(built, dl)
        worst_overall = min(worst_overall, margin)
        assert margin >= -1e-9, (
            f"seed {seed}: intermolecular {pair} pair sits {abs(margin):.3f} "
            "A BELOW the W8 contact floor")
    assert worst_overall < np.inf    # the scan actually constrained pairs


def test_grid_packing_is_deterministic_and_census_exact():
    """Rule 10: same counts + box + seed -> byte-identical frame; the grid +
    minimise fallback conserves every molecule."""
    dl = load_dialect(("core", "molecular"))
    edge = (60 * molecular_mass("CO2") / 1.00 / 0.6022140857) ** (1 / 3)
    a = grid_pack_molecules({"CO2": 60}, [edge] * 3,
                            np.random.default_rng(5), dl)
    b = grid_pack_molecules({"CO2": 60}, [edge] * 3,
                            np.random.default_rng(5), dl)
    assert np.array_equal(a.pos, b.pos) and a.symbols == b.symbols
    assert molecule_census(a, dl) == {"CO2": 60}


def test_grid_packing_serves_liquid_water_too():
    """The fallback must not be CO2-tuned: 256 H2O at 0.997 g/cm3 (the
    water_tip4p macrostate, the density the plain packer reaches only
    through its census-safe sphere) packs through grid + minimise with an
    exact census and a floor-clean contact set."""
    dl = load_dialect(("core", "molecular"))
    edge = (256 * molecular_mass("H2O") / 0.997 / 0.6022140857) ** (1 / 3)
    frame = grid_pack_molecules({"H2O": 256}, [edge] * 3,
                                np.random.default_rng(7), dl)
    assert molecule_census(frame, dl) == {"H2O": 256}
    margin, pair = _worst_intermolecular_margin(frame, dl)
    assert margin >= -1e-9, f"{pair} below the W8 contact floor"


def test_grid_packing_fails_closed_on_an_impossible_density():
    """No silent sub-floor frame: 60 CO2 at 3 g/cm3 (past dry ice) exhausts
    the relaxation and the fallback refuses."""
    from chaord.lang.errors import ChaordError
    dl = load_dialect(("core", "molecular"))
    edge = (60 * molecular_mass("CO2") / 3.0 / 0.6022140857) ** (1 / 3)
    with pytest.raises(ChaordError, match="contact"):
        grid_pack_molecules({"CO2": 60}, [edge] * 3,
                            np.random.default_rng(0), dl)
