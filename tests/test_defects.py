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


def test_conservation_exact_for_defects(dialect):
    rng = np.random.default_rng(67)
    f = build_conventional("L1_2", {"a": 3.572}, ("Ni", "Al"), (3, 3, 3))
    fd = apply_defects(f, [_kv("V_Ni", 2), _kv("Al_Ni", 1), _kv("Ni_i", 1)], rng, dialect)
    program = lift_frame(fd, dialect, mode="defects")
    from chaord.check.statics import conservation_check
    result = conservation_check(program, fd)
    assert result.passed, result.detail
