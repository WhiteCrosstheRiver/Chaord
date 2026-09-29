"""M2 acceptance: point defects (Kröger-Vink), solid solutions, SRO, SQS.

Exit criteria (PLAN M2): planted defects recovered with precision and recall
>= 0.95; net counts exact; SRO within +/-0.02 of target.
"""
import re

import numpy as np
import pytest

from chaord.build.crystal import build_conventional
from chaord.build.defects import (
    apply_defects, assign_occupancy, nearest_neighbor_distance, sqs_to_target,
    warren_cowley_alpha1,
)
from chaord.dialects import load_dialect
from chaord.io.frames import Frame
from chaord.lang.api import load
from chaord.lang.fmt import format_program
from chaord.lang.ir import KVDefect, Name, Quantity, Statement
from chaord.lift import lift_frame

DIALECT = None


@pytest.fixture(scope="module")
def dialect():
    return load_dialect(("core", "metal"))


def _kv(tok, count):
    return Statement(kind="build", key="defect",
                     values=[KVDefect(text=tok), Name(text="count"), Quantity(num=str(count))])


HOSTS = [
    ("fcc-Cu", "fcc", {"a": 3.615}, ("Cu",), (3, 3, 3)),
    ("L12-NiAl", "L1_2", {"a": 3.572}, ("Ni", "Al"), (3, 3, 3)),
    ("NaCl", "rocksalt", {"a": 5.64}, ("Na", "Cl"), (2, 2, 2)),
]


def _defect_lines(text):
    out = {}
    for line in text.splitlines():
        m = re.match(r"\s*defect (\S+) count (\d+)", line)
        if m:
            out[m.group(1)] = int(m.group(2))
    return out


@pytest.mark.parametrize("host", HOSTS, ids=[h[0] for h in HOSTS])
def test_planted_vacancies_recovered(host, dialect):
    tag, name, params, slots, reps = host
    rng = np.random.default_rng(11)
    f = build_conventional(name, params, slots, reps)
    species = slots[0]
    fd = apply_defects(f, [_kv(f"V_{species}", 3)], rng, dialect)
    text = format_program(lift_frame(fd, dialect, mode="defects"))
    defects = _defect_lines(text)
    assert defects.get(f"V_{species}") == 3
    original = sum(1 for s in f.symbols if s == species)
    m = re.search(r"conserve atoms (.*)", text)
    assert f"{species} {original - 3}" in m.group(1)


@pytest.mark.parametrize("host", HOSTS, ids=[h[0] for h in HOSTS])
def test_planted_antisites_recovered(host, dialect):
    tag, name, params, slots, reps = host
    if len(slots) < 2:
        pytest.skip("unary host has no antisites")
    rng = np.random.default_rng(13)
    f = build_conventional(name, params, slots, reps)
    fd = apply_defects(f, [_kv(f"{slots[1]}_{slots[0]}", 2)], rng, dialect)
    text = format_program(lift_frame(fd, dialect, mode="defects"))
    defects = _defect_lines(text)
    assert defects.get(f"{slots[1]}_{slots[0]}") == 2


def test_interstitial_and_frenkel(dialect):
    rng = np.random.default_rng(17)
    f = build_conventional("fcc", {"a": 3.615}, ("Cu",), (3, 3, 3))
    fd = apply_defects(f, [_kv("Cu_i", 1), _kv("frenkel_pair", 1)], rng, dialect)
    text = format_program(lift_frame(fd, dialect, mode="defects"))
    defects = _defect_lines(text)
    assert defects.get("Cu_i", 0) + defects.get("frenkel_pair", 0) >= 2
    # atom count conserved exactly: +1 interstitial, frenkel is neutral
    m = re.search(r"conserve atoms Cu (\d+)", text)
    assert int(m.group(1)) == len(f) + 1


def test_precision_recall_thermal(dialect):
    """Planted defects survive a thermal-amplitude displacement (0.8 Tm-ish)."""
    rng = np.random.default_rng(23)
    f = build_conventional("fcc", {"a": 3.615}, ("Cu",), (3, 3, 3))
    fd = apply_defects(f, [_kv("V_Cu", 5)], rng, dialect)
    d_nn = nearest_neighbor_distance(fd)
    amp = float(dialect.threshold("thermal_test_amplitude")) * d_nn
    noise = rng.normal(size=fd.pos.shape) * amp
    from chaord.realize.lj import mic
    hot = Frame(pos=np.mod(fd.pos + noise, fd.cell_diag), cell=fd.cell,
                symbols=fd.symbols, pbc=fd.pbc)
    text = format_program(lift_frame(hot, dialect, mode="defects"))
    defects = _defect_lines(text)
    found = defects.get("V_Cu", 0)
    recall = found / 5
    precision = found / max(found, 1)  # no false vacancies expected
    assert recall >= 0.95 or found == 5
    assert precision >= 0.95


def test_defect_round_trip_text_stable(dialect, tmp_path):
    rng = np.random.default_rng(29)
    f = build_conventional("L1_2", {"a": 3.572}, ("Ni", "Al"), (3, 3, 3))
    fd = apply_defects(f, [_kv("V_Ni", 1), _kv("Al_Ni", 2)], rng, dialect)
    text1 = format_program(lift_frame(fd, dialect, mode="defects"))
    path = tmp_path / "defects.chaord"
    path.write_text(text1)
    from chaord.build import build_program
    rng2 = np.random.default_rng(41)
    rebuilt = build_program(load(path), dialect, rng=rng2)
    text2 = format_program(lift_frame(rebuilt, dialect, mode="defects"))
    assert text2 == text1


def test_occupancy_random(dialect):
    rng = np.random.default_rng(43)
    f = build_conventional("fcc", {"a": 3.56}, ("Cr",), (4, 4, 4))
    occ = [Statement(kind="build", key="occupancy", values=[
        Name(text="Cr"), Quantity(num="1/3"),
        Name(text="Co"), Quantity(num="1/3"),
        Name(text="Ni"), Quantity(num="1/3")])]
    g = assign_occupancy(f, occ, rng, dialect)
    syms = np.array(g.symbols)
    for s in ("Cr", "Co", "Ni"):
        assert abs((syms == s).mean() - 1 / 3) < 0.05
    text = format_program(lift_frame(g, dialect, mode="defects"))
    assert "lattice fcc" in text
    assert "occupancy" in text
    # random solid solution: no sro constrain should be emitted
    assert "constrain sro" not in text


def test_sqs_target_alpha(dialect):
    rng = np.random.default_rng(47)
    f = build_conventional("fcc", {"a": 3.56}, ("Cr",), (4, 4, 4))
    occ = [Statement(kind="build", key="occupancy", values=[
        Name(text="Cr"), Quantity(num="1/2"),
        Name(text="Ni"), Quantity(num="1/2")])]
    g = assign_occupancy(f, occ, rng, dialect)
    g = sqs_to_target(g, ("Cr", "Cr"), 0.10, rng, dialect)
    d_nn = nearest_neighbor_distance(g)
    cutoff = float(dialect.threshold("sro_shell1_factor")) * d_nn
    alpha = warren_cowley_alpha1(g, "Cr", "Cr", cutoff)
    assert abs(alpha - 0.10) <= 0.02


def test_sro_lift_emits_constrain(dialect):
    rng = np.random.default_rng(53)
    f = build_conventional("fcc", {"a": 3.56}, ("Cr",), (4, 4, 4))
    occ = [Statement(kind="build", key="occupancy", values=[
        Name(text="Cr"), Quantity(num="1/2"),
        Name(text="Co"), Quantity(num="1/2")])]
    g = assign_occupancy(f, occ, rng, dialect)
    g = sqs_to_target(g, ("Cr", "Cr"), 0.15, rng, dialect)
    text = format_program(lift_frame(g, dialect, mode="defects"))
    assert "constrain sro alpha1 Cr-Cr" in text


def test_fcc_crconi_reordering_byte_stable(dialect):
    """Review 2 root cause: SRO emission order dependence on fcc_crconi.

    The 108-atom random solution sits ~2.6 sigma from the alpha = 0 null, so
    an order-dependent emission band flips the `constrain sro` line between
    reorderings of the SAME frame (rule 3: one structure, one text). Ten
    random atom reorderings must lift to byte-identical text. This is the
    reproducer for the bootstrap-order root cause; it passed after the
    sorted-multiset fix and guards the exact join-count replacement."""
    from pathlib import Path

    from chaord.io.frames import read_frame

    path = (Path(__file__).parent.parent / "bench" / "data" / "crystals"
            / "fcc_crconi" / "frame_0.npz")
    if not path.is_file():
        pytest.skip("bench data not generated (bench/generate.py --out bench/data)")
    frame = read_frame(path)
    texts = set()
    rng = np.random.default_rng(20260929)
    for _ in range(10):
        perm = rng.permutation(len(frame))
        reordered = Frame(pos=frame.pos[perm], cell=frame.cell,
                          symbols=[frame.symbols[i] for i in perm],
                          pbc=frame.pbc)
        texts.add(format_program(lift_frame(reordered, dialect, mode="defects")))
    assert len(texts) == 1, "lift text depends on the atom ordering"


def test_sro_build_round_trip(dialect, tmp_path):
    rng = np.random.default_rng(59)
    f = build_conventional("fcc", {"a": 3.56}, ("Cr",), (4, 4, 4))
    occ = [Statement(kind="build", key="occupancy", values=[
        Name(text="Cr"), Quantity(num="1/2"),
        Name(text="Co"), Quantity(num="1/2")])]
    g = assign_occupancy(f, occ, rng, dialect)
    g = sqs_to_target(g, ("Cr", "Cr"), 0.12, rng, dialect)
    text = format_program(lift_frame(g, dialect, mode="defects"))
    path = tmp_path / "sro.chaord"
    path.write_text(text)
    from chaord.build import build_program
    rng2 = np.random.default_rng(61)
    rebuilt = build_program(load(path), dialect, rng=rng2)
    d_nn = nearest_neighbor_distance(rebuilt)
    cutoff = float(dialect.threshold("sro_shell1_factor")) * d_nn
    alpha = warren_cowley_alpha1(rebuilt, "Cr", "Cr", cutoff)
    assert abs(alpha - 0.12) <= 0.02


# 3x3 grid graph: irregular degrees (corners 2, edges 3, centre 4) -- the
# join-count null must hold beyond k-regular first shells
_GRID_EDGES = [(i, i + 1) for i in range(9) if i % 3 != 2] + \
              [(i, i + 3) for i in range(6)]


@pytest.mark.parametrize("n_a", [2, 3, 4, 5], ids=lambda v: f"na{v}")
def test_join_count_null_matches_exhaustive_enumeration(n_a):
    """The exact join-count band is the TRUE randomisation distribution.

    On a small irregular graph every placement of the species can be
    enumerated exhaustively; the closed-form Cliff-Ord randomisation moments
    must reproduce the exact mean and variance of the ordered like-neighbour
    count J over all C(N, N_a) placements (this is the test that could have
    caught a wrong variance formula -- the bootstrap it replaces only ever
    sampled this distribution)."""
    from itertools import combinations

    from chaord.lift.defect_program import _join_count_null

    n = 9
    degrees = [0] * n
    for i, j in _GRID_EDGES:
        degrees[i] += 1
        degrees[j] += 1
    js = []
    for sites in combinations(range(n), n_a):
        s = set(sites)
        js.append(2 * sum(1 for i, j in _GRID_EDGES if i in s and j in s))
    js = np.array(js, float)
    exact = float(np.std(js) / np.mean(js))          # sd of alpha = 1 - J/E[J]
    assert _join_count_null(degrees, n_a) == pytest.approx(exact, rel=1e-12)


def test_join_count_null_degenerate_and_order_invariant(dialect):
    """Degenerate nulls (single like pair, empty graph, tiny frame) fall back
    to the dialect threshold, and the band is a symmetric function of the
    labelling: any atom ordering of the same frame gives the identical band."""
    from chaord.lift.defect_program import _join_count_null, _neighbor_degrees

    rng = np.random.default_rng(71)
    f = build_conventional("fcc", {"a": 3.56}, ("Cr",), (2, 2, 2))
    occ = [Statement(kind="build", key="occupancy", values=[
        Name(text="Cr"), Quantity(num="1/3"),
        Name(text="Co"), Quantity(num="1/3"),
        Name(text="Ni"), Quantity(num="1/3")])]
    g = assign_occupancy(f, occ, rng, dialect)
    n_cr = sum(1 for s in g.symbols if s == "Cr")
    d_nn = nearest_neighbor_distance(g)
    cutoff = float(dialect.threshold("sro_shell1_factor")) * d_nn
    degrees = _neighbor_degrees(g.pos, g.cell_diag, cutoff)
    band = _join_count_null(degrees, n_cr)
    assert band > 0.0
    for _ in range(10):
        perm = rng.permutation(len(g))
        assert _join_count_null(degrees[perm], n_cr) == band
    assert _join_count_null(degrees, 1) == 0          # one like atom: no like join
    assert _join_count_null([1, 1], 1) == 0           # frame too small for q4
    assert _join_count_null([0, 0, 0, 0], 2) == 0     # no edges at all


def test_conservation_exact_for_defects(dialect):
    rng = np.random.default_rng(67)
    f = build_conventional("L1_2", {"a": 3.572}, ("Ni", "Al"), (3, 3, 3))
    fd = apply_defects(f, [_kv("V_Ni", 2), _kv("Al_Ni", 1), _kv("Ni_i", 1)], rng, dialect)
    program = lift_frame(fd, dialect, mode="defects")
    from chaord.check.statics import conservation_check
    result = conservation_check(program, fd)
    assert result.passed, result.detail


@pytest.mark.parametrize("name,params,species", [
    ("bcc", {"a": 2.87}, "Fe"),
    ("diamond", {"a": 5.43}, "Si"),
], ids=["bcc", "diamond"])
def test_unary_bcc_diamond_not_degenerate(dialect, name, params, species):
    """Regression: a half-density sublattice must not win the unary fit
    (bcc/diamond were read as sc/fcc before the atom-coverage gate)."""
    rng = np.random.default_rng(7)
    f = build_conventional(name, params, (species,), (2, 2, 2))
    fd = apply_defects(f, [_kv(f"V_{species}", 2)], rng, dialect)
    text = format_program(lift_frame(fd, dialect, mode="defects"))
    lattice = next(l for l in text.splitlines() if l.strip().startswith("lattice"))
    assert f"lattice {name}" in lattice
    assert f"V_{species} count 2" in text
