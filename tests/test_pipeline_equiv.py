"""Stage-1 acceptance of the pipeline lift (docs/design/lift_build_v2.md 4.1).

Three gates:
- equivalence: for one frame of every bench family the legacy cascade covers
  (crystal, defects, fluid, glass, interface/slab), the pipeline lift's fmt
  text equals the legacy lift's byte for byte -- including the shipped slab
  golden (tests/golden/lj_slab.chaord);
- degradation: a frame whose crystal region cannot be fitted (the fit needs
  metal-dialect keys the frame's core+lj dialect does not define) still lifts:
  the region's atoms land in the residual block, conservation holds exactly, a
  provenance note records the degradation, nothing raises (design 2.5);
- performance: the pipeline on an lj_liquid frame stays within the 1.3x
  budget of the legacy cascade (risk R5); the single cached qbar and the
  frame-level single-phase gate keep the two paths at parity.

The bench frames are the shipped, seeded bench/data snapshots (schema
chaord-bench/2); prototype/snap.npz is the golden LJ slab fixture.
"""
import time
from pathlib import Path

import numpy as np
import pytest

from chaord.check.statics import conservation_check, get_lift_diag
from chaord.dialects import load_dialect
from chaord.io.frames import Frame, read_frame
from chaord.lang.fmt import format_program
from chaord.lang.parser import parse_text
from chaord.lift import lift_frame
from chaord.lift.legacy import legacy_lift
from chaord.lift.pipeline import pipeline_lift, pipeline_lift_detailed

ROOT = Path(__file__).parent.parent
BENCH = ROOT / "bench" / "data"
GOLDEN_SLAB = ROOT / "tests" / "golden" / "lj_slab.chaord"

# one frame per bench family, with the dialect that family lifts under
EQUIV_CASES = [
    ("crystals/fcc_cu", ("core", "metal")),            # exact crystal arm
    ("crystals/rocksalt_nacl", ("core", "metal")),     # multi-species host
    ("defects/fcc_cu_vacancies", ("core", "metal")),   # Wigner-Seitz arm
    ("defects/l12_ni3al_vac_antisite", ("core", "metal")),
    ("fluid/water_box15", ("core", "molecular")),      # molecular fluid
    ("fluid/ar_gas_box25", ("core", "molecular")),     # noisy order parameter
    ("glass/lj_glass_rho085", ("core", "glass")),      # amorphous arm
    ("interface/lj_solid_liquid", ("core", "lj")),     # mixed frame
]


def _case_id(path):
    return path.replace("/", "_")


@pytest.fixture(scope="module")
def equiv_texts():
    """{(case id | 'snap'): (legacy_text, pipeline_text, unexplained)}."""
    out = {}
    for case, dialect_names in EQUIV_CASES:
        frame = read_frame(BENCH / case / "frame_0.npz")
        dialect = load_dialect(dialect_names)
        legacy, _ = legacy_lift(frame, dialect, None, "auto")
        pipe, unexplained = pipeline_lift_detailed(frame, dialect, None)
        out[case] = (format_program(legacy), format_program(pipe), unexplained)
    snap = read_frame(ROOT / "prototype" / "snap.npz")
    dialect = load_dialect(("core", "lj"))
    legacy, _ = legacy_lift(snap, dialect, None, "auto")
    pipe, unexplained = pipeline_lift_detailed(snap, dialect, None)
    out["snap"] = (format_program(legacy), format_program(pipe), unexplained)
    return out


@pytest.mark.parametrize("case", [c for c, _ in EQUIV_CASES] + ["snap"],
                         ids=_case_id)
def test_pipeline_matches_legacy_byte_for_byte(equiv_texts, case):
    legacy_text, pipe_text, unexplained = equiv_texts[case]
    assert pipe_text == legacy_text
    assert unexplained == 0


def test_pipeline_reproduces_slab_golden(equiv_texts):
    """The shipped lj_slab golden text is byte-identical under the pipeline."""
    _, pipe_text, _ = equiv_texts["snap"]
    assert pipe_text == GOLDEN_SLAB.read_text(encoding="utf-8")


def test_mode_pipeline_is_routed_and_auto_stays_legacy():
    """lift_frame wiring: mode='pipeline' runs the pipeline; mode='auto' keeps
    the legacy cascade until the flip (design 2.6)."""
    frame = read_frame(BENCH / "crystals" / "fcc_cu" / "frame_0.npz")
    dialect = load_dialect(("core", "metal"))
    via_mode = format_program(lift_frame(frame, dialect, mode="pipeline"))
    via_api = format_program(pipeline_lift(frame, dialect))
    legacy, _ = legacy_lift(frame, dialect, None, "auto")
    assert via_mode == via_api == format_program(legacy)
    auto = format_program(lift_frame(frame, dialect))       # default mode
    assert auto == format_program(legacy)


# ------------------------------------------------------------- degradation --

def _degraded_frame():
    """A frame one of whose regions cannot be fitted under its dialect.

    An LJ fcc crystal block (a = 1.6, 500 atoms) and a liquid droplet
    (300 atoms), separated by a gap wider than the segmentation graph: two
    regions, no contact, so the composed path runs. The crystal fit's engines
    need metal-dialect thresholds (site_match_tol_fraction & co.) that
    core + lj does not define, so region A's fit necessarily fails and its
    atoms must degrade to the residual (never abort, design 2.5)."""
    rng = np.random.default_rng(41)
    a = 1.6
    idx = np.array([[x, y, z] for x in range(10) for y in range(10)
                    for z in range(10) if (x + y + z) % 2 == 0])
    block = idx * (a / 2)
    directions = rng.normal(size=(300, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    radii = 3.0 * rng.uniform(0, 1, 300) ** (1 / 3)
    droplet = np.array([8.0, 8.0, 12.0]) + directions * radii[:, None]
    pos = np.vstack([block, droplet])
    return Frame(pos=pos, cell=np.diag([16.0] * 3),
                 symbols=["X"] * len(pos), pbc=(True, True, True))


@pytest.fixture(scope="module")
def degraded():
    frame = _degraded_frame()
    dialect = load_dialect(("core", "lj"))
    program = lift_frame(frame, dialect, mode="pipeline")  # must not raise
    return frame, dialect, program


def test_degraded_region_atoms_go_to_residual(degraded):
    frame, _, program = degraded
    residual = next(b for b in program.blocks if b.t == "residual")
    assert not residual.none
    atoms = [s for s in residual.statements if s.key == "atom"]
    assert len(atoms) == 500                     # the crystal region, whole
    regions = [b for b in program.blocks if b.t == "region"]
    assert [r.phase for r in regions] == ["liquid"]   # the fit did not land
    assert get_lift_diag(program)["unexplained"] == 500


def test_degradation_is_recorded_in_provenance(degraded):
    _, _, program = degraded
    notes = [s for b in program.blocks if b.t == "provenance"
             for s in b.statements if s.key == "note"]
    assert any("region A" in n.values[0].text and "residual" in n.values[0].text
               for n in notes)


def test_degraded_lift_conserves_every_atom(degraded):
    frame, dialect, program = degraded
    system = next(b for b in program.blocks if b.t == "system")
    conserve = next(s for s in system.statements if s.key == "atoms")
    stated = {conserve.values[0].text: int(conserve.values[1].num)}
    assert stated == {"X": len(frame)}            # residual atoms included
    result = conservation_check(program, frame, dialect)
    assert result.passed, result.detail


def test_degraded_program_reparses(degraded):
    _, _, program = degraded
    text = format_program(program)
    assert parse_text(text).version == program.version
    from chaord.lang.ir import ir_equal
    assert ir_equal(parse_text(text), program)


# ------------------------------------------------- composed multi-region path --

@pytest.fixture(scope="module")
def composed():
    """A crystal + liquid + vacuum frame: three regions, so neither the
    single-phase gate nor the two-region slab engine can take it -- the
    composed path (S3-S7) runs: named regions, a vacuum region, interfaces
    only between regions that actually fitted, residual for the rest."""
    rng = np.random.default_rng(7)
    a = 3.6
    idx = np.array([[x, y, z] for x in range(6) for y in range(6)
                    for z in range(4) if (x + y + z) % 2 == 0])
    slab = idx * (a / 2)
    dirs = rng.normal(size=(60, 3))
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    radii = 2.6 * rng.uniform(0, 1, 60) ** (1 / 3)
    drop = np.array([5.4, 5.4, 7.0]) + dirs * radii[:, None]
    frame = Frame(pos=np.vstack([slab, drop]),
                  cell=np.diag([21.6, 21.6, 20.0]),
                  symbols=["Cu"] * len(slab) + ["X"] * len(drop),
                  pbc=(True, True, True))
    dialect = load_dialect(("core", "metal", "surface"))
    return frame, dialect, lift_frame(frame, dialect, mode="pipeline")


def test_composed_program_structure_and_names(composed):
    """One program: regions in canonical name order, an atom-less vacuum
    region, and no interface block naming a region that is not in the
    program (a degraded region has no block, so no interface may cite it)."""
    frame, dialect, program = composed
    blocks = program.blocks
    names = [b.name for b in blocks if b.t == "region"]
    assert names == sorted(names)                      # canonical order
    assert "vacuum" in [b.phase for b in blocks if b.t == "region"]
    for b in blocks:
        if b.t == "interface":
            assert {b.a, b.b} <= set(names)
    text = format_program(program)
    assert parse_text(text).version == program.version  # grammar-valid
    assert conservation_check(program, frame, dialect).passed


# ------------------------------------------------------------- performance --

def test_pipeline_within_performance_budget_on_lj_liquid():
    """Pipeline <= 1.3x legacy on an lj_liquid frame (risk R5). Both paths
    attempt the same engines here; the pipeline's cached qbar and the
    frame-level gate keep it at parity, and the best of three runs each side
    rules out scheduler noise."""
    rng = np.random.default_rng(11)
    n_atoms, rho = 2048, 0.85
    L = np.full(3, (n_atoms / rho) ** (1 / 3))
    pos = rng.uniform(0, L[0], (n_atoms, 3))
    frame = Frame(pos=pos, cell=np.diag(L), symbols=["X"] * n_atoms,
                  pbc=(True, True, True))
    dialect = load_dialect(("core", "lj"))

    def best_of(three, fn):
        times = []
        for _ in range(three):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        return min(times)

    t_legacy = best_of(3, lambda: legacy_lift(frame, dialect, None, "auto"))
    t_pipeline = best_of(3, lambda: pipeline_lift_detailed(frame, dialect, None))
    assert t_pipeline <= 1.3 * t_legacy, \
        f"pipeline {t_pipeline:.3f}s vs legacy {t_legacy:.3f}s"
