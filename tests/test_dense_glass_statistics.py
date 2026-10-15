"""X6 (Review 10, D27): dense-glass statistics mean what they say.

The lifted Kob-Andersen program used to print statistics with weak meaning
(measured on bench/reference/ka_glass/frame_0.npz, before this work):

    assert cn 10.52 +- 0.30 cutoff 1.19     <- cuts THROUGH the A-A first
                                                peak (1.06; its first minimum
                                                is near 1.4 sigma)
    assert angle_mean 96.8 +- 3.0 deg        <- network-glass statistic on a
                                                dense molecular glass
    assert ring_mode 3                       <- same (true of any dense
                                                packing)
    assert gr_peak 1.69 ... pair B-B         <- the GLOBAL maximum of a
                                                double-peaked noisy function,
                                                not the first peak (jumps
                                                1.35..1.73 sigma across the
                                                15 reference frames)

D27 (binding): per-pair coordination cutoffs at each partial g(r)'s FIRST
MINIMUM; ring_mode and angle_mean print ONLY for network glasses (mean
coordination <= 6); first-peak search WINDOWS per pair for gr_peak.

Every measured number cited here is reproducible on the 15 reference frames
(3 quenches x 5) with the helpers in this file -- the dialect values are
pinned to the data by test_first_minimum_cutoffs_match_reference_measurement
and test_every_printed_assert_within_tolerance_across_reference.
"""
from __future__ import annotations

import re
from itertools import combinations_with_replacement

import numpy as np
import pytest

from chaord.dialects import load_dialect
from chaord.io.frames import Frame
from chaord.lang.fmt import format_program
from chaord.lift.amorphous import lift_amorphous

REF = "bench/reference/ka_glass"


def load_ref(path):
    """A bench/reference frame (npz with r, L, symbols)."""
    z = np.load(path, allow_pickle=True)
    L = np.asarray(z["L"], float)
    return Frame(pos=np.asarray(z["r"], float),
                 cell=np.diag(L) if L.ndim == 1 else L,
                 symbols=[str(s) for s in z["symbols"]],
                 pbc=(True, True, True))


@pytest.fixture(scope="module")
def dialect():
    return load_dialect(("core", "glass", "lj_mixtures"))


@pytest.fixture(scope="module")
def ref_frames():
    return [load_ref(f"{REF}/frame_{k}.npz") for k in range(15)]


def _assert_index(program):
    """{(key, pair-or-None): (asserted quantities, tolerances, cutoff)} for
    every assert in the program.

    The pair label is the text after the `pair` marker when the assert
    carries one (the per-pair partial asserts), else None.  Asserted
    quantities are the ones with a tolerance; the number after a `cutoff`
    marker is a stated parameter, not an asserted value."""
    out = {}
    for block in program.blocks:
        for stmt in getattr(block, "statements", ()):
            if stmt.kind != "assert":
                continue
            pair = None
            nums, tols = [], []
            skip_next_q = False
            for i, v in enumerate(stmt.values):
                if v.t == "n" and v.text == "pair" and i + 1 < len(stmt.values):
                    pair = stmt.values[i + 1].text
                elif v.t == "n" and v.text == "cutoff":
                    skip_next_q = True
                elif v.t == "q" and skip_next_q:
                    skip_next_q = False
                elif v.t == "q":
                    nums.append(v)
                elif v.t == "tol":
                    tols.append(v)
            out[(stmt.key, pair)] = (nums, tols)
    return out


# ------------------------------------------------ cutoffs: per-pair minima --

def test_ka_reference_uses_per_pair_first_minimum_cutoffs(dialect, ref_frames):
    """D27: the coordination asserts cut at each partial g(r)'s FIRST MINIMUM,
    not at one geometric radius through the A-A first peak.

    The old global line (`assert cn 10.52 +- 0.30 cutoff 1.19`, the single
    amorphous_bond_factor x d_NN radius applied to every pair) is gone; each
    unordered pair states its own cutoff, and the A-A cutoff (1.40 sigma)
    sits past the A-A first peak (1.06 sigma) at the measured minimum."""
    program = lift_amorphous(ref_frames[0], dialect)
    idx = _assert_index(program)
    entry = dialect.threshold("lj_mixtures")["kob_andersen"]
    cuts = entry["coordination_cutoffs"]
    for pair, key in (("A-A", "AA"), ("A-B", "AB"), ("B-B", "BB")):
        assert ("cn", pair) in idx, f"no cn assert for pair {pair}"
        # the cutoff is the quantity after the `cutoff` marker on the line
        text = format_program(program)
        m = re.search(rf"assert cn [\d.]+ \+- [\d.]+ cutoff ([\d.]+) pair "
                      rf"{re.escape(pair)}\b", text)
        assert m, f"pair {pair}: no cutoff on its cn assert"
        assert float(m.group(1)) == pytest.approx(float(cuts[key])), \
            f"pair {pair}: cutoff {m.group(1)} is not the dialect value"
    # no bare total cn assert survives for a multi-species glass (its cutoff
    # would have to be one radius over per-pair first minima -- meaningless)
    assert ("cn", None) not in idx, idx.keys()
    # the A-A cutoff must not cut the A-A first peak
    aa_peak = float(idx[("gr_peak", "A-A")][0][0].num)
    aa_cut = float(cuts["AA"])
    assert aa_cut > aa_peak, "the A-A cutoff cuts through the first peak"


def test_first_minimum_cutoffs_match_reference_measurement(dialect, ref_frames):
    """The dialect's per-pair cutoffs ARE the first minima of the partial
    g(r) measured on the 15 reference frames (the provenance pinned to data,
    so a regenerated reference that moves its minima fails here).

    Method (X6, 2026-10-14): partial g(r) on the frame's own grid at
    dr = 0.01 sigma (partial_gr, rmax 3.0, 300 bins), moving-average width 5,
    first minimum = the lowest smoothed bin between the first peak and the
    peak + 0.9 sigma.  Measured per-frame minima: AA 1.365-1.465 (mean 1.406),
    AB 1.125-1.205 (mean 1.168), BB 1.875-2.045 (mean 1.944 -- the B-B first
    shell is the split 1.38/1.71 doublet, so its first DEEP minimum follows
    the doublet)."""
    from chaord.cv.rich import partial_gr

    def smooth(x, w=5):                       # measurement smoothing, sigma 0.05
        return np.convolve(x, np.ones(w) / w, mode="same")

    entry = dialect.threshold("lj_mixtures")["kob_andersen"]
    cuts = entry["coordination_cutoffs"]
    mins = {"AA": [], "AB": [], "BB": []}
    for f in ref_frames:
        for a, b in combinations_with_replacement(("A", "B"), 2):
            g, r = partial_gr(f, dialect, a, b, rmax=3.0, bins=300)
            gs = smooth(g)
            ipk = int(np.argmax((g * (r > 0.7))[:220]))
            imin = ipk + int(np.argmin(gs[ipk:ipk + 90]))
            mins[a + b].append(r[imin])
    for key, stated in cuts.items():
        stated = float(stated)
        lo, hi = min(mins[key]), max(mins[key])
        mean = float(np.mean(mins[key]))
        assert lo <= stated <= hi, \
            f"{key}: dialect cutoff {stated} outside the measured minima " \
            f"[{lo:.3f}, {hi:.3f}]"
        assert abs(stated - mean) <= 0.05, \
            f"{key}: dialect cutoff {stated} more than 0.05 from the " \
            f"measured mean first minimum {mean:.3f}"


# ------------------------------------------------------- the network gate --

def test_dense_glass_prints_no_ring_or_angle_asserts(dialect, ref_frames):
    """D27: ring_mode and angle_mean are NETWORK-glass statistics; the KA
    glass (mean coordination ~13.7, A-centered 13.36 / B-centered 15.2 at the
    per-pair first minima) must print neither."""
    for k in (0, 7, 14):
        text = format_program(lift_amorphous(ref_frames[k], dialect))
        for key in ("ring_mode", "angle_mean"):
            assert f"assert {key}" not in text, \
                f"frame {k}: network statistic {key} printed for a dense glass"


def _jittered_diamond(amplitude=0.20, seed=11):
    """A disordered 4-coordinated network (diamond Si + 0.20 A jitter): mean
    coordination 4 under the glass dialect's 2.85 A Si-Si first shell, so it
    IS a network glass by D27's gate (the crystal fit would refuse it as a
    glass; lift_amorphous is called directly, as the pipeline does after
    is_amorphous)."""
    from chaord.build.crystal import build_conventional
    dia = build_conventional("diamond", {"a": 5.43}, ("Si",), (2, 2, 2))
    pos = dia.pos + np.random.default_rng(seed).normal(0, amplitude,
                                                       dia.pos.shape)
    return Frame(pos=np.mod(pos, dia.cell_diag), cell=dia.cell,
                 symbols=list(dia.symbols), pbc=(True,) * 3)


def test_network_glass_still_prints_ring_and_angle(dialect):
    """The gate cuts both ways: a low-coordination network glass (mean
    coordination 4 <= 6) still prints its ring_mode and angle_mean asserts."""
    frame = _jittered_diamond()
    text = format_program(lift_amorphous(frame, load_dialect(("core", "glass"))))
    assert "assert ring_mode" in text, text
    assert "assert angle_mean" in text, text
    # ... and the tetrahedral network's angle is near 109.5 deg, not a dense
    # packing's ~97 deg (the value the KA glass used to print)
    m = re.search(r"assert angle_mean ([\d.]+)", text)
    assert m and float(m.group(1)) == pytest.approx(109.5, abs=3.0)


def test_gate_threshold_is_the_dialects(dialect):
    """The network/dense boundary is D27's stated 6 and lives in glass.yaml,
    not in the lifter."""
    assert float(dialect.threshold("network_glass_max_coordination")) == 6.0


# ------------------------------------------------------------ gr_peak -------

def test_gr_peak_is_first_peak_in_window(dialect, ref_frames):
    """D27: gr_peak states the FIRST peak of the partial g(r), searched in a
    per-pair window from the dialect table (AA [1.02, 1.16], AB [0.80, 0.94],
    BB [1.28, 1.46] sigma) -- never the global maximum.  The B-B global max
    lands on the second sub-peak (~1.71 sigma) on 5 of the 15 frames."""
    entry = dialect.threshold("lj_mixtures")["kob_andersen"]
    wins = entry["gr_peak_windows"]
    for k in range(15):
        idx = _assert_index(lift_amorphous(ref_frames[k], dialect))
        for pair, key in (("A-A", "AA"), ("A-B", "AB"), ("B-B", "BB")):
            pos = float(idx[("gr_peak", pair)][0][0].num)
            lo, hi = (float(x) for x in wins[key])
            assert lo <= pos <= hi, \
                f"frame {k} pair {pair}: gr_peak {pos} outside the window"


def test_every_printed_assert_within_tolerance_across_reference(dialect,
                                                                ref_frames):
    """X6 done-when: EVERY numeric assert the lift prints on the reference
    frames (a) carries a printed tolerance and (b) varies across the 15
    frames by LESS than that tolerance.

    Before (same measurement): gr_peak printed no tolerance at all and the
    B-B position spread over the 15 frames was 0.38 sigma (1.35..1.73); the
    cn asserts moved 0.02-0.16 inside their 0.30."""
    spreads: dict = {}
    tols_by_ident: dict = {}
    for k in range(15):
        for (key, pair), (nums, tols) in \
                _assert_index(lift_amorphous(ref_frames[k], dialect)).items():
            assert tols, \
                f"frame {k}: assert {key} pair {pair} prints no tolerance"
            assert len(tols) == len(nums), \
                f"frame {k}: assert {key} pair {pair} has " \
                f"{len(nums)} values but {len(tols)} tolerances"
            for i, (q, tol) in enumerate(zip(nums, tols)):
                ident = (key, pair, i)
                v = float(q.num)
                t = float(tol.value.num)
                assert t > 0.0, f"frame {k}: {ident} tolerance not positive"
                lo, hi = spreads.get(ident, (v, v))
                spreads[ident] = (min(lo, v), max(hi, v))
                tols_by_ident[ident] = t   # dialect constants: same every frame
    for ident, (lo, hi) in sorted(spreads.items()):
        spread = hi - lo
        tol = tols_by_ident[ident]
        assert spread < tol, \
            f"assert {ident}: 15-frame spread {spread:.3f} >= tolerance " \
            f"{tol:.3f} (range {lo:.3f}..{hi:.3f})"


# ------------------------------------------------- active re-measure fix ----

def test_pair_scoped_asserts_are_unverified_not_failed(dialect, ref_frames):
    """Collateral: uncertainty_report used to re-measure the per-pair cn
    asserts with cn_mean (an ALL-species count) at the pair's cutoff and
    record them as FAILURES (three false failures on the KA program).  A
    pair-scoped assert has no local all-species measure: it must count as
    unverified, never as failed."""
    from chaord.active import uncertainty_report
    program = lift_amorphous(ref_frames[0], dialect)
    report = uncertainty_report(program, ref_frames[0], dialect)
    assert report.asserts_failed == 0, report.failures
    failed_pairs = [f["key"] for f in report.failures if "pair" in str(f)]
    assert not failed_pairs
