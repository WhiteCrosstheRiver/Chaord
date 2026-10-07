"""O11 (Reviews 4-7) -> W4 steps 1 and 2 (Review 8): the metal stack defines
`printed_cutoff`, and the cu_solid_liquid reference must be lifted CORRECTLY.

O11 unblocked the slab route (the thresholds exist; the frame segments) but
the lifted program was wrong: Cu as `X`, LJ units, 300 K unmarked, the fcc
solid called hcp, 11 frenkel_pair lines. W4 step 1 (docs/reviews/
open_items_v2.md, gate) made the slab path refuse that frame; step 2 carries
species and units through it (Cu, A, eV, K), states T from the caller or as
assumed (W11), holds the fcc/hcp call at 0.9-1.0 Tm and prints defect lines
only above a floor measured on defect-free thermal frames. The remaining
refusal set is tested in tests/test_metal_slab_refusal.py; this file keeps
the dialect-key contract and the step-2 done-when reproducer."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import acceptance as acc  # noqa: E402
from chaord.dialects import load_dialect  # noqa: E402
from chaord.io.frames import read_frame  # noqa: E402
from chaord.lang.api import load, save  # noqa: E402
from chaord.lift import lift_frame  # noqa: E402
from chaord.lift.passes import pairs_within  # noqa: E402
from chaord.lift.slab import chi0_mean  # noqa: E402

METAL = ("core", "metal")

# the W4 reproducer frame: bench/reference/cu_solid_liquid/frame_0.npz,
# 832 Cu at 1335 K (provenance.json: FBD-1986 Cu_u3 EAM, a0 = 3.615 A,
# Tm(FBD Cu) ~ 1330 K, equilibrated at 1335 K ~ 1.0 Tm)
CU_REF = ROOT / "bench" / "reference" / "cu_solid_liquid" / "frame_0.npz"

# FBD-1986 Cu_u3 lattice constant (potential file / provenance parameters):
# a0 = 3.615 A at the calibration state; linear thermal expansion to the
# frame's 1335 K gives a(1335 K) ~ 3.676 A (alpha ~ 16.5e-6/K over 1035 K),
# and the lifted constant must sit within 2% of that value
A0_FBD_CU = 3.615
A_1335K_FBD_CU = 3.676


def _provenance_notes(text):
    """The `note "..."` lines of the provenance block, as plain strings."""
    return re.findall(r'note "([^"]*)"', text)


def _defect_lines(text):
    """The program's defect lines (`  defect ...`, one per complex)."""
    return re.findall(r"^  defect\b.*$", text, re.M)


def test_metal_dialect_defines_printed_cutoff():
    dl = load_dialect(METAL)
    assert float(dl.threshold("printed_cutoff")) == pytest.approx(3.25)


# ------------------------------------------------------- W4 step 2 done-when --

def test_w4_step2_reproducer_prints_cu_in_metal_units():
    """The W4 step-2 reproducer verbatim (docs/reviews/open_items_v2.md):
    `conserve atoms Cu 832`, metal units, `lattice fcc`, a within 2% of the
    potential's lattice constant at 1335 K, at most one defect line.

    Red before step 2: the step-1 gate refused this lift (and pre-gate the
    program printed units lj / conserve atoms X 832 / lattice hcp /
    11 frenkel_pair lines)."""
    frame = read_frame(CU_REF)
    text = acc.format_program_text(lift_frame(frame, load_dialect(METAL)))
    lines = [l.strip() for l in text.splitlines()
             if l.strip().startswith(("units", "conserve", "state T", "backend", "lattice"))]

    # species and units carried through: no reduced-units laundering anywhere
    assert not any(l.startswith("units lj") for l in lines), lines
    assert "conserve atoms Cu 832" in text
    # metal units: physical-unit statements and the metal backend
    assert "backend eam" in lines, lines
    m = re.search(r"^  cell (\S+) (\S+) (\S+) A$", text, re.M)
    assert m, text[:400]
    assert [float(m.group(i)) for i in (1, 2, 3)] == pytest.approx(
        [14.460, 14.460, 49.872], abs=5e-3)
    # T: the caller gave none -> the dialect default is stated AND flagged
    # assumed (W11), never laundered into a measurement of this 1335 K frame
    assert "state T 300.00 K" in text
    assert any(n.startswith("T assumed (dialect default 300)")
               and "not measured from the frame" in n
               for n in _provenance_notes(text)), _provenance_notes(text)

    # the fcc/hcp call holds at ~1.0 Tm: the solid is fcc, not hcp
    assert "lattice fcc" in lines, lines

    # the lattice constant within 2% of the potential's a at 1335 K
    m = re.search(r"^  a (\S+) A$", text, re.M)
    assert m, text[:400]
    a = float(m.group(1))
    assert abs(a - A_1335K_FBD_CU) / A_1335K_FBD_CU <= 0.02, a
    assert abs(a - A0_FBD_CU) / A0_FBD_CU <= 0.02, a

    # at most one defect line (pre-fix: 12 complexes, 11 frenkel_pair lines)
    assert len(_defect_lines(text)) <= 1, _defect_lines(text)


def test_w4_step2_caller_T_stated_without_assumed_note():
    """Done-when arm 2: a caller-given T is a fact about the program -- stated
    as the provenance temperature (1335 K of the reference protocol) with no
    assumed note."""
    frame = read_frame(CU_REF)
    text = acc.format_program_text(
        lift_frame(frame, load_dialect(METAL), T=1335.0))
    assert "state T 1335.00 K" in text
    assert not any("T assumed" in n for n in _provenance_notes(text))


def test_w4_step2_program_builds_and_conserves(tmp_path):
    """Done-when arm 3: the lifted Cu program builds (never drops an atom;
    every built atom is Cu) and passes the static checks."""
    from chaord.build.slab import build_slab
    from chaord.check.statics import run_checks

    frame = read_frame(CU_REF)
    program = lift_frame(frame, load_dialect(METAL))
    path = tmp_path / "cu_solid_liquid.chaord"
    save(program, path)
    dialect = load_dialect(METAL)
    built = build_slab(load(path), dialect,
                       rng=np.random.default_rng(1), physics=False)
    assert len(built) == 832                    # never drop an atom
    assert set(built.symbols) == {"Cu"}         # species carried through
    checks = run_checks(load(path), built, dialect)
    failed = [f"{c.name}: {c.detail}" for c in checks if not c.passed]
    assert not failed, failed


# ----------------------------------------------- fcc/hcp discriminator pin --

# RMS |u| per atom measured 2026-10-02 on the chi0 calibration frames at
# 1.0 Tm (metal.yaml chi0_fcc_min comment: ASE Langevin + FBD-1986 Cu_u3,
# seeds 424242/434343): fcc 0.315-0.327 A, hcp 0.380-0.504 A
_AMP_FCC_1TM = 0.32


def _ideal_fcc():
    """Perfect fcc(100) block, 4x4x6 cells at a0 = 3.615 A (the calibration
    frames' geometry), via the realize builder."""
    from chaord.realize.lj import fcc as fcc_block
    a0 = 3.615
    pos = fcc_block(4, 4, 6, a0)
    return pos, np.array([4 * a0, 4 * a0, 6 * a0])


def _ideal_ortho_hcp():
    """Perfect hcp block in the orthohexagonal setting (a, sqrt(3) a, c),
    a = a0/sqrt(2), ideal c/a -- the setting of the calibration frames."""
    a0 = 3.615
    a = a0 / np.sqrt(2)
    c = a * np.sqrt(8 / 3)
    s3 = np.sqrt(3)
    basis = np.array([[0, 0, 0], [a / 2, s3 * a / 2, 0],
                      [a / 2, s3 * a / 6, c / 2], [0, 2 * s3 * a / 3, c / 2]])
    pos = np.array([b + [i * a, j * s3 * a, k * c]
                    for i in range(4) for j in range(3) for k in range(7)
                    for b in basis])
    L = np.array([4 * a, 3 * s3 * a, 7 * c])
    return pos, L


def _classify(pos, L, dialect):
    """decompile's chi0 discrimination on a bulk frame (all atoms are
    crystal interior): the statistic, then the same fcc/hcp/other call."""
    pairs = pairs_within(pos, L, rc=float(dialect.threshold("q6_cutoff")))
    chi0 = chi0_mean(pos, np.ones(len(pos), bool), pairs, L, dialect)
    fcc_min = float(dialect.threshold("chi0_fcc_min"))
    hcp_min = float(dialect.threshold("chi0_hcp_min"))
    ctype = "fcc" if chi0 > fcc_min else ("hcp" if chi0 > hcp_min else "other")
    return chi0, ctype


def test_w4_step2_discriminator_holds_at_1tm_thermal_amplitude():
    """The fcc/hcp call must hold at 0.9-1.0 Tm (the W4 failure: thermal fcc
    at ~1.0 Tm read hcp under core's chi0_fcc_min 4.5). What this mechanism
    test honestly pins with synthetic frames:

    - the wiring: ideal fcc reads exactly 6.00 antiparallel pairs per atom
      and classifies fcc; ideal orthohexagonal hcp reads exactly 3.00 and
      classifies hcp, through the same chi0_mean + thresholds decompile
      uses;
    - the W4 failure mode: an fcc lattice displaced by seeded Gaussian
      noise at the RMS amplitude MEASURED on the 1.0 Tm calibration frames
      (0.32 A, metal.yaml chi0_fcc_min comment) still classifies fcc over
      five draws. Independent Gaussian noise is HARSHER than real thermal
      motion (measured MD band 5.06-5.39 vs 3.96-4.51 under Gaussian noise
      at the same RMS -- real vibrations are correlated), so this is a
      lower bound on the real margin.

    The hcp-at-temperature side is pinned by the measured MD band
    (2.78-2.98, metal.yaml) and the band-split test below, NOT by synthetic
    noise: at the measured hcp RMS (0.44 A) uncorrelated noise reads chi0
    1.65-1.85, below the MD band -- a synthetic displacement cannot stand
    in for the correlated thermal motion the calibration measured."""
    dialect = load_dialect(METAL)
    rng = np.random.default_rng(20261002)
    for name, make in (("fcc", _ideal_fcc), ("hcp", _ideal_ortho_hcp)):
        pos, L = make()
        chi0_0, ctype_0 = _classify(pos, L, dialect)
        assert ctype_0 == name
        assert chi0_0 == pytest.approx(6.0 if name == "fcc" else 3.0)

    pos, L = _ideal_fcc()
    for draw in range(5):
        disp = rng.normal(0, _AMP_FCC_1TM / np.sqrt(3), pos.shape)
        chi0, ctype = _classify(pos + disp, L, dialect)
        assert ctype == "fcc", (
            f"fcc at the measured 1.0 Tm thermal amplitude, draw {draw}: "
            f"chi0 {chi0:.2f} classified {ctype!r} (chi0_fcc_min "
            f"{float(dialect.threshold('chi0_fcc_min'))})")


def test_w4_step2_metal_chi0_threshold_is_calibrated_band_split():
    """The metal chi0_fcc_min must sit strictly between the MEASURED bands
    (hcp <= 2.98 < threshold < 4.43 <= fcc, metal.yaml comment): a threshold
    outside the gap misclassifies a measured frame. Pins the numbers the
    comment cites so the calibration cannot silently drift."""
    dialect = load_dialect(METAL)
    fcc_min = float(dialect.threshold("chi0_fcc_min"))
    assert 2.98 < fcc_min < 4.43
