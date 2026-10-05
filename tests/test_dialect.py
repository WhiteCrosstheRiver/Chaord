"""Dialect loading, threshold access, versions."""
import pytest

from chaord.dialects import DIALECT_NAMES, load_dialect
from chaord.lang.errors import ChaordError


def test_core_always_first():
    d = load_dialect(("lj",))
    assert d.names == ["core", "lj"]


def test_unknown_dialect_rejected():
    with pytest.raises(ChaordError, match="unknown dialect"):
        load_dialect(("core", "nope"))


def test_merge_later_wins():
    d = load_dialect(("core", "lj"))
    assert d.threshold("overlap_tolerance") == 0.80  # lj overrides core (D1: 0.80 sigma)
    core = load_dialect(("core",))
    assert core.threshold("overlap_tolerance") == 0.5


def test_missing_threshold_raises():
    d = load_dialect(("core",))
    # cn_cutoff stays per-dialect (metal 3.5 A, glass 2.85 A, ...); since W1
    # moved the unit-free phase-rule keys (q6_solid among them) into core,
    # this example must name a key core genuinely does not define
    with pytest.raises(ChaordError, match="no threshold"):
        d.threshold("cn_cutoff")


def test_versions_recorded():
    d = load_dialect(("core", "lj"))
    assert d.versions["core"] and d.versions["lj"]
    assert "core" in d.version_string and "lj" in d.version_string


def test_units_union():
    d = load_dialect(("core", "metal"))
    assert "A" in d.units and "K" in d.units


def test_all_eight_dialects_load():
    for name in DIALECT_NAMES:
        load_dialect(("core", name) if name != "core" else ("core",))


def test_md_section_from_lj():
    d = load_dialect(("core", "lj"))
    md = d.threshold("md")
    assert md["skin"] == 0.3
