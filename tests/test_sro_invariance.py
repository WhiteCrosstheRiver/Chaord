"""SRO at scale: 50 markings x 10 orderings, and the thermal-frame SRO cutoff.

Two external-review-3 findings against the audited tree are pinned here.

1. A2 order invariance, at property scale. The exact join-count null
   (``_join_count_null``, Cliff-Ord randomisation moments) is a symmetric
   function of the labelling, so the SRO emission band -- and the emitted
   text -- cannot depend on the atom ordering. The review's report of one
   marking lifting to two different texts over 20 reorderings was the
   pre-fix bootstrap band; the 50 x 10 matrix here is the at-scale evidence,
   and NEAR-THRESHOLD markings (alpha1 within a few hundredths of the
   emission band edge) are included because an order-dependent band flips
   the text exactly there.

2. The SRO first-shell cutoff root cause (review 3, issue 7): the lift drew
   the Warren-Cowley cutoff from ``nearest_neighbor_distance(frame)`` -- the
   MINIMUM pair distance of the atom cloud. Thermal jitter collapses that
   minimum far below the lattice spacing (bench fcc_crconi at the 0.8 Tm
   amplitude: 1.80 A vs the 2.52 A ideal fcc spacing), so the cutoff
   1.2 x 1.80 = 2.16 A fell below the first shell and every atom kept ~0.4
   of its 12 first-shell neighbours: alpha1 read the sparse-graph artefact
   (-1.0 on Co-Co of a RANDOM marking) and no thermal frame's SRO was
   meaningful. The cutoff now comes from the FITTED lattice constant
   (``_sro_first_shell_cutoff``): on the same frames the first shell is
   intact (11-13 neighbours per atom), a random marking reads alpha1 ~ 0
   inside the exact null band (and stays silent), and a genuinely ordered
   marking emits its measured alpha1. Before the fix the first-shell and
   alpha assertions error on the missing helper, and the ordered-marking
   test fails on value: the pre-fix lift read alpha1(Co-Co) = +0.51 on a
   +0.32 marking and stayed silent under the inflated sparse-graph band
   (0.77), so no `constrain sro` line appeared at all.
"""
import re
from pathlib import Path

import numpy as np
import pytest

from chaord.build.crystal import build_conventional
from chaord.build.defects import warren_cowley_alpha1
from chaord.dialects import load_dialect
from chaord.io.frames import Frame, read_frame
from chaord.lang.fmt import format_program
from chaord.lift import lift_frame
from chaord.lift.defect_program import _join_count_null, _neighbor_degrees

ROOT = Path(__file__).parent.parent
CRCONI_FRAME = ROOT / "bench" / "data" / "crystals" / "fcc_crconi" / "frame_0.npz"

SPECIES = ("Co", "Cr", "Ni")
N_MARKINGS = 50          # property matrix: 50 markings x 10 atom orderings
N_ORDERINGS = 10
N_RANDOM = 40            # the remaining 10 are driven near the band edge
# near-threshold offsets from the emission band edge (fractions of alpha1):
# half the driven markings land just inside (silent), half just outside
EDGE_OFFSETS = (-0.06, -0.03, -0.01, 0.01, 0.03)


@pytest.fixture(scope="module")
def dialect():
    return load_dialect(("core", "metal"))


def _lift_text(frame, dialect) -> str:
    return format_program(lift_frame(frame, dialect, mode="defects"))


def _ideal_fcc_solution_sites(a=3.56, reps=(3, 3, 3)):
    """Exact fcc sites of the fcc_crconi construction (108 atoms, 10.68 A box)."""
    return build_conventional("fcc", {"a": a}, ("Cr",), reps)


def _random_marking(rng) -> list[str]:
    marking = [s for s in SPECIES for _ in range(36)]
    rng.shuffle(marking)
    return marking


def _alpha(sites, cell, pbc, marking, species, cutoff) -> float:
    frame = Frame(pos=sites, cell=cell, symbols=list(marking), pbc=pbc)
    return warren_cowley_alpha1(frame, species, species, cutoff)


def _drive_alpha(sites, cell, pbc, marking, species, cutoff, target, rng,
                 tol=0.005, max_steps=4000) -> list[str]:
    """Seeded greedy species swaps driving alpha1(species, species) to target."""
    marking = list(marking)
    best = _alpha(sites, cell, pbc, marking, species, cutoff)
    for _ in range(max_steps):
        if abs(best - target) < tol:
            break
        i = int(rng.choice([k for k, s in enumerate(marking) if s == species]))
        j = int(rng.choice([k for k, s in enumerate(marking) if s != species]))
        marking[i], marking[j] = marking[j], marking[i]
        trial = _alpha(sites, cell, pbc, marking, species, cutoff)
        if abs(trial - target) < abs(best - target):
            best = trial
        else:
            marking[i], marking[j] = marking[j], marking[i]
    return marking


def _emission_band(degrees, n_species_atoms, dialect) -> float:
    """The lift's significance gate: max(dialect floor, noise factor x null sd)."""
    return max(float(dialect.threshold("sro_emit_threshold")),
               float(dialect.threshold("sro_emit_noise_factor"))
               * _join_count_null(degrees, n_species_atoms))


def _geometric_cutoff(a, dialect) -> float:
    """First-shell cutoff of the ground-truth lattice (a/sqrt(2) between shells)."""
    return float(dialect.threshold("sro_shell1_factor")) * a / np.sqrt(2)


# ------------------------------------------------- 50 markings x 10 orderings --

def test_marking_ordering_invariance_50x10(dialect):
    """Every marking lifts to exactly ONE text over 10 random atom orderings.

    Random markings (40) must stay silent (a random solution's alpha1 is a
   finite-sample fluctuation inside the 3.5-sigma exact-null band; rule 1),
   and the near-threshold markings (10, alpha1 driven to within +/-0.06 of
   the band edge) must split between emitting and silent -- the knife edge
   where an order-dependent band would flip the `constrain sro` line and
   break rule 3. On the pre-A2-fix bootstrap band this matrix produced
   markings with two different texts; against the exact join-count null it
   must be 50/50 matrices of one text each. (Run against the pre-cutoff-fix
   tree: all 50 markings WERE byte-stable over their 10 orderings -- the
   order-invariance evidence -- and the only failing assertion was the
   emission split, because the jitter-collapsed cutoff silenced every
   marking, which is root cause 2 below.)"""
    if not CRCONI_FRAME.is_file():
        pytest.skip("bench data not generated (bench/generate.py --out bench/data)")
    base = read_frame(CRCONI_FRAME)
    # ground truth of the case: 3x3x3 tiling of a = 3.56 (cell 10.68 A)
    a_true = float(base.cell_diag[0]) / 3
    cutoff = _geometric_cutoff(a_true, dialect)
    degrees = _neighbor_degrees(base.pos, base.cell_diag, cutoff)
    band = _emission_band(degrees, 36, dialect)

    n_unique_total, n_sro_texts, emitted = 0, 0, []
    for k in range(N_MARKINGS):
        rng = np.random.default_rng(3000 + k)
        if k < N_RANDOM:
            marking = _random_marking(rng)
        else:
            target = band + EDGE_OFFSETS[(k - N_RANDOM) % len(EDGE_OFFSETS)]
            marking = _drive_alpha(base.pos, base.cell, base.pbc,
                                   _random_marking(rng), "Co", cutoff, target, rng)
            achieved = _alpha(base.pos, base.cell, base.pbc, marking, "Co", cutoff)
            assert abs(achieved - target) <= 0.01, \
                f"marking {k}: drove alpha1(Co) to {achieved:.3f}, target {target:.3f}"
        texts = set()
        rng_perm = np.random.default_rng(70000 + k)
        for _ in range(N_ORDERINGS):
            perm = rng_perm.permutation(len(base))
            frame = Frame(pos=base.pos[perm], cell=base.cell,
                          symbols=[marking[i] for i in perm], pbc=base.pbc)
            texts.add(_lift_text(frame, dialect))
        assert len(texts) == 1, \
            f"marking {k}: {len(texts)} distinct texts over {N_ORDERINGS} orderings"
        n_unique_total += len(texts)
        has_sro = "constrain sro" in next(iter(texts))
        emitted.append(has_sro)
        n_sro_texts += int(has_sro)
        if k < N_RANDOM:
            assert not has_sro, \
                f"random marking {k} emitted an SRO line (alpha1 inside the band)"
    # the knife edge is exercised on BOTH sides of the emission gate
    assert 0 < n_sro_texts < N_MARKINGS, \
        f"near-threshold markings all on one side of the band ({n_sro_texts})"
    assert n_unique_total == N_MARKINGS


# ----------------------------------------------- thermal-frame SRO root cause --

def test_thermal_random_marking_full_first_shell_and_alpha_near_zero(dialect):
    """Root cause (review 3, issue 7), random marking on a jittered solution.

    A thermally jittered fcc solid solution (ideal sites + 0.06 d_NN per-axis
    Gaussian, seeded) measured with the lift's own first-shell cutoff must
    keep the first shell intact -- 11-13 neighbors per atom, mean ~12 (the
    pre-fix cutoff, 1.2 x the jitter-collapsed MINIMUM pair distance, kept
    ~0.4 neighbors per atom) -- and a random marking must read alpha1 ~ 0
    inside the exact join-count band (the pre-fix sparse-graph readout was
    alpha1(Co-Co) = -1.0), so no `constrain sro` line is emitted."""
    from chaord.lift.defects import fit_crystal
    from chaord.lift.defect_program import _sro_first_shell_cutoff

    rng = np.random.default_rng(1003)
    sites = _ideal_fcc_solution_sites()
    d_ideal = 3.56 / np.sqrt(2)
    amp = float(dialect.threshold("thermal_test_amplitude")) * d_ideal
    marking = _random_marking(rng)
    hot = Frame(pos=np.mod(sites.pos + rng.normal(size=sites.pos.shape) * amp,
                           sites.cell_diag),
                cell=sites.cell, symbols=marking, pbc=sites.pbc)

    # the cutoff the lift itself derives: fitted lattice constant -> d_NN
    name, a_fit, _slot, _score, _sites, _site_species = fit_crystal(hot, dialect)
    cutoff = _sro_first_shell_cutoff(name, a_fit, dialect)
    degrees = _neighbor_degrees(hot.pos, hot.cell_diag, cutoff)
    assert degrees.min() >= 11 and degrees.max() <= 13, \
        f"first shell damaged: degrees in [{degrees.min()}, {degrees.max()}]"
    assert 11.5 <= float(degrees.mean()) <= 12.5, float(degrees.mean())
    for s in SPECIES:
        alpha = warren_cowley_alpha1(hot, s, s, cutoff)
        band = _emission_band(degrees, marking.count(s), dialect)
        assert abs(alpha) < band, f"alpha1({s}-{s}) = {alpha:.3f} vs band {band:.3f}"

    assert "constrain sro" not in _lift_text(hot, dialect)


def test_thermal_ordered_marking_emits_measured_alpha(dialect):
    """Root cause (review 3, issue 7), genuinely ordered marking: the lift must
    emit the SRO constrain with the measured alpha1 even on a thermally
    jittered frame. Failed before the fix on value: with the jitter-collapsed
    cutoff the same marking read alpha1(Co-Co) = +0.51 against an inflated
    0.77 band and emitted nothing."""
    rng = np.random.default_rng(20260930)
    sites = _ideal_fcc_solution_sites()
    d_ideal = 3.56 / np.sqrt(2)
    amp = float(dialect.threshold("thermal_test_amplitude")) * d_ideal
    marking = _random_marking(rng)
    hot = Frame(pos=np.mod(sites.pos + rng.normal(size=sites.pos.shape) * amp,
                           sites.cell_diag),
                cell=sites.cell, symbols=marking, pbc=sites.pbc)
    cutoff = _geometric_cutoff(3.56, dialect)
    band = _emission_band(_neighbor_degrees(hot.pos, hot.cell_diag, cutoff),
                          36, dialect)
    target = band + 0.05                      # safely outside the emission band
    driven = _drive_alpha(hot.pos, hot.cell, hot.pbc, marking, "Co",
                          cutoff, target, rng)
    measured = _alpha(hot.pos, hot.cell, hot.pbc, driven, "Co", cutoff)
    assert abs(measured - target) <= 0.01, measured

    text = _lift_text(Frame(pos=hot.pos, cell=hot.cell, symbols=driven,
                            pbc=hot.pbc), dialect)
    m = re.search(r"constrain sro alpha1 Co-Co ([+-]\d+\.\d+)", text)
    assert m, "ordered marking on a thermal frame emitted no SRO constrain"
    assert abs(float(m.group(1)) - measured) <= 0.03, \
        f"printed alpha1 {m.group(1)} vs measured {measured:.3f}"
    # the other two species are random fluctuations and must stay silent
    assert "Cr-Cr" not in text and "Ni-Ni" not in text
