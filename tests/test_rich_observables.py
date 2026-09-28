"""Rich held-out observables: species-pair g(r), bond-angle distribution,
density profile.

Review follow-up on the statistical round trip (cv/noise.py): the default
observable set (all-species g(r) + one coordination histogram) is dominated
by the intramolecular O-H bonds in molecular systems.  The rich observables
in chaord.cv.rich add species-resolved structure:

  * partial_gr            — g(r) of one species pair (same-molecule pairs
                            excluded under the molecular dialect, so the
                            intramolecular O-H peak at 0.96 A cannot dominate)
  * angle_distribution    — bond-angle density P(theta), compared by TV
  * density_profile       — per-bin number density along a cell axis,
                            compared by RMS (sharper than a whole-box g(r)
                            for interface systems)
  * observables(selection=...) / distance(keys=...) — a configurable
    observable set; the default (selection=None) output and the default
    distance keys are byte-for-byte the previous behaviour, which
    tools/acceptance.py A5 and cv/noise callers rely on.

Literature anchors (bench/reference/*/provenance.json):
  TIP4P O-O first peak 2.75-2.90 A (Jorgensen et al., JCP 79, 926 (1983));
  H-O-H bend 104.52 deg.
"""
from pathlib import Path

import numpy as np
import pytest

from chaord.cv.noise import distance, noise_floor, observables, within_floor
from chaord.cv.rich import angle_distribution, density_profile, partial_gr
from chaord.dialects import load_dialect
from chaord.io.frames import read_frame
from chaord.lang.errors import ChaordError

ROOT = Path(__file__).parent.parent
WATER = ROOT / "bench" / "reference" / "water_tip4p"
LJ_LIQUID = ROOT / "bench" / "reference" / "lj_liquid"
INTERFACE = ROOT / "bench" / "data" / "interface" / "lj_solid_liquid"

GR_RMAX_WATER = 5.0   # provenance sanity-measurement range for O-O
GR_BINS_WATER = 100   # dr = 0.05 A, the resolution the provenance peaks use


def water_frame(k):
    return read_frame(WATER / f"frame_{k}.npz")


def lj_frame(case, k):
    return read_frame(case / f"frame_{k}.npz")


@pytest.fixture(scope="module")
def mol():
    return load_dialect(("core", "molecular"))


@pytest.fixture(scope="module")
def lj():
    return load_dialect(("core", "lj"))


# --------------------------------------------------------------- partial g(r) --

@pytest.mark.parametrize("k", range(5))
def test_water_oo_partial_gr_first_peak(k, mol):
    """TIP4P O-O partial g(r): first peak where the literature puts it."""
    g, r = partial_gr(water_frame(k), mol, "O", "O",
                      rmax=GR_RMAX_WATER, bins=GR_BINS_WATER)
    m = r > 2.0                                   # search above the hard core
    peak = float(r[m][np.argmax(g[m])])
    assert 2.70 <= peak <= 2.95, f"frame {k}: O-O peak {peak} A"
    assert g[m].max() > 1.5, f"frame {k}: O-O peak height {g[m].max()}"


@pytest.mark.parametrize("k", range(5))
def test_lj_partial_gr_matches_existing_observables(k, lj):
    """All-species partial g(r) reproduces the default observables g(r)."""
    f = lj_frame(LJ_LIQUID, k)
    g, r = partial_gr(f, lj, "X", "X",
                      rmax=lj.threshold("gr_rmax_fluid"),
                      bins=int(lj.threshold("gr_bins_fluid")))
    o = observables(f, lj)
    assert np.allclose(g, o["gr"])
    assert np.allclose(r, o["rm"])


def test_partial_gr_intramolecular_pairs_excluded_under_molecular(mol):
    """The molecular dialect drops same-molecule pairs: no O-H signal below
    the intermolecular hard core (1.5 A), where the un-excluded g(r) has a
    delta-like intramolecular peak at the 0.9572 A O-H bond."""
    f = water_frame(0)
    g, r = partial_gr(f, mol, "O", "H", rmax=GR_RMAX_WATER, bins=GR_BINS_WATER)
    assert float(g[r < 1.5].max()) == pytest.approx(0.0, abs=1e-12)
    m = r > 1.5
    assert float(r[m][np.argmax(g[m])]) > 1.6    # first peak = H-bond shell


def test_partial_gr_atomic_dialect_keeps_all_pairs():
    """Outside the molecular dialect there is no molecule concept: the raw
    atom pair list is used, so the intramolecular O-H bond IS the first peak
    (this is the dominance the rich observables exist to avoid)."""
    core = load_dialect(("core",))
    f = water_frame(0)
    g, r = partial_gr(f, core, "O", "H", rmax=GR_RMAX_WATER, bins=GR_BINS_WATER)
    peak = float(r[np.argmax(g)])
    assert 0.90 <= peak <= 1.05                  # the 0.9572 A O-H bond
    assert float(g.max()) > 10.0                 # delta-like intramolecular peak


def test_partial_gr_cross_normalisation_water(mol):
    """O-H and H-O give the same g(r) (a-b cross normalisation is symmetric)."""
    f = water_frame(0)
    goh, r1 = partial_gr(f, mol, "O", "H", rmax=GR_RMAX_WATER, bins=GR_BINS_WATER)
    gho, r2 = partial_gr(f, mol, "H", "O", rmax=GR_RMAX_WATER, bins=GR_BINS_WATER)
    assert np.allclose(r1, r2)
    assert np.allclose(goh, gho)


# --------------------------------------------------------- bond-angle density --

@pytest.mark.parametrize("k", range(5))
def test_water_angle_distribution_peak(k, mol):
    """TIP4P H-O-H bend: main peak at 104.5 deg (100-110 window)."""
    centers, density = angle_distribution(water_frame(k), mol)
    peak = float(centers[np.argmax(density)])
    assert 100.0 <= peak <= 110.0, f"frame {k}: angle peak {peak} deg"
    assert density.sum() == pytest.approx(1.0)


def _brute_force_angles(pos, L, cutoff, bins):
    """Independent O(N^2) bond-angle list + histogram (no KD tree, no
    shared helpers): the reference the library version must reproduce."""
    n = len(pos)
    edges = np.linspace(0.0, 180.0, bins + 1)
    angles = []
    for i in range(n):
        vecs = []
        for j in range(n):
            if i == j:
                continue
            d = pos[j] - pos[i]
            d = d - L * np.round(d / L)
            if np.linalg.norm(d) <= cutoff:
                vecs.append(d)
        for a in range(len(vecs)):
            for b in range(a + 1, len(vecs)):
                ca = np.dot(vecs[a], vecs[b]) / (
                    np.linalg.norm(vecs[a]) * np.linalg.norm(vecs[b]))
                angles.append(np.degrees(np.arccos(np.clip(ca, -1.0, 1.0))))
    h, _ = np.histogram(np.asarray(angles), edges)
    return h / max(h.sum(), 1)


def test_angle_distribution_matches_brute_force_water(mol):
    f = water_frame(0)
    cutoff = float(mol.threshold("angle_cutoff_fluid"))
    bins = int(mol.threshold("angle_bins_fluid"))
    centers, density = angle_distribution(f, mol, cutoff=cutoff, bins=bins)
    brute = _brute_force_angles(f.pos, f.cell_diag, cutoff, bins)
    tv = 0.5 * float(np.abs(density - brute).sum())
    assert tv < 0.02, f"water: TV(library, brute) = {tv}"


def test_angle_distribution_matches_brute_force_dense_lj(lj):
    """Same agreement on a dense first-shell neighbourhood (many float-diverse
    angles), not just the discrete bonded water case."""
    f = lj_frame(INTERFACE, 0)
    cutoff = float(lj.threshold("cn_cutoff_fluid"))
    bins = int(lj.threshold("angle_bins_fluid"))
    centers, density = angle_distribution(f, lj, cutoff=cutoff, bins=bins)
    brute = _brute_force_angles(f.pos, f.cell_diag, cutoff, bins)
    tv = 0.5 * float(np.abs(density - brute).sum())
    assert tv < 0.02, f"lj interface: TV(library, brute) = {tv}"


# ----------------------------------------------------------- density profile --

@pytest.mark.parametrize("k", range(5))
def test_density_profile_interface_two_phases(k, lj):
    """Solid-liquid slab: the layered solid half vs the homogeneous liquid
    half gives a profile contrast well above 0.3x the mean density."""
    f = lj_frame(INTERFACE, k)
    prof = density_profile(f, lj)
    assert (prof.max() - prof.min()) > 0.3 * prof.mean()
    assert prof.mean() == pytest.approx(len(f) / float(np.prod(f.cell_diag)))


def _brute_force_profile(pos, L, axis, nbins):
    edges = np.linspace(0.0, L[axis], nbins + 1)
    x = np.mod(pos[:, axis], L[axis])
    other = [i for i in range(3) if i != axis]
    bin_vol = (L[axis] / nbins) * L[other[0]] * L[other[1]]
    out = np.empty(nbins)
    for b in range(nbins):
        out[b] = np.count_nonzero((x >= edges[b]) & (x < edges[b + 1])) / bin_vol
    return out


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_density_profile_matches_brute_force(axis, lj):
    f = lj_frame(INTERFACE, 1)
    nbins = int(lj.threshold("density_profile_bins"))
    prof = density_profile(f, lj, axis=axis, nbins=nbins)
    brute = _brute_force_profile(f.pos, f.cell_diag, axis, nbins)
    assert np.allclose(prof, brute)


# ----------------------------------------------------- configurable set/dist --

def test_observables_default_behaviour_unchanged(lj):
    """selection=None is the legacy contract: same keys, byte-identical
    arrays, and the default distance dict has exactly the two legacy keys."""
    f = lj_frame(LJ_LIQUID, 0)
    o_default = observables(f, lj)
    o_explicit = observables(f, lj, selection=None)
    o_gr_group = observables(f, lj, selection=("gr",))
    assert set(o_default) == {"gr", "rm", "cn", "cn_hist"}
    for k in o_default:
        assert np.asarray(o_default[k]).tobytes() == \
            np.asarray(o_explicit[k]).tobytes()
        assert np.asarray(o_default[k]).tobytes() == \
            np.asarray(o_gr_group[k]).tobytes()
    d = distance(o_default, o_gr_group, dialect=lj)
    assert d["gr_rms"] == 0.0 and d["cn_tv"] == 0.0
    assert set(d) == {"gr_rms", "cn_tv"}


def test_observables_selection_keys(mol):
    f = water_frame(0)
    o = observables(f, mol, selection=("gr", "partial_gr", "angle", "density"))
    expected = {"gr", "rm", "cn", "cn_hist",
                "gr_H-H", "rm_H-H", "gr_H-O", "rm_H-O",
                "gr_O-O", "rm_O-O", "angle", "angle_centers", "density"}
    assert set(o) == expected
    # the embedded partial g(r) equals the standalone call
    g, r = partial_gr(f, mol, "O", "O", rmax=mol.threshold("gr_rmax_fluid"),
                      bins=int(mol.threshold("gr_bins_fluid")))
    assert np.allclose(o["gr_O-O"], g)
    centers, density = angle_distribution(f, mol)
    assert np.allclose(o["angle"], density)
    assert np.allclose(o["density"], density_profile(f, mol))


def test_observables_selection_rejects_unknown_group(mol):
    with pytest.raises(ChaordError, match="unknown observable group"):
        observables(water_frame(0), mol, selection=("gr", "nope"))


def test_distance_key_filter(mol):
    a, b = water_frame(0), water_frame(1)
    oa = observables(a, mol, selection=("gr", "partial_gr", "angle", "density"))
    ob = observables(b, mol, selection=("gr", "partial_gr", "angle", "density"))
    d = distance(oa, ob, dialect=mol)
    assert set(d) == {"gr_rms", "gr_rms_H-H", "gr_rms_H-O", "gr_rms_O-O",
                      "cn_tv", "angle_tv", "density_rms"}
    for k in d.values():
        assert np.isfinite(k)
    # exact names filter one comparison; prefixes filter a family
    only_oo = distance(oa, ob, dialect=mol, keys=("gr_rms_O-O",))
    assert set(only_oo) == {"gr_rms_O-O"}
    gr_family = distance(oa, ob, dialect=mol, keys=("gr_rms",))
    assert set(gr_family) == {"gr_rms", "gr_rms_H-H", "gr_rms_H-O",
                              "gr_rms_O-O"}
    assert gr_family["gr_rms_O-O"] == only_oo["gr_rms_O-O"]
    with pytest.raises(ChaordError, match="matched no distance key"):
        distance(oa, ob, dialect=mol, keys=("nope",))


def test_distance_mismatched_observable_sets_raise(mol):
    rich = observables(water_frame(0), mol,
                       selection=("gr", "partial_gr", "angle", "density"))
    legacy = observables(water_frame(1), mol)
    with pytest.raises(ChaordError, match="angle"):
        distance(rich, legacy, dialect=mol)


def test_within_floor_selection_passthrough(mol):
    """The round-trip verdict composes with the rich selection: A5-style
    usage (tools/acceptance.py) can gate on any observable subset."""
    fa, fb, fc = water_frame(0), water_frame(1), water_frame(2)
    floor = noise_floor(fa, fb, mol, selection=("angle", "density"))
    assert set(floor) == {"angle_tv", "density_rms"}
    verdict = within_floor(fa, fa, fb, mol, selection=("gr", "angle"))
    assert set(verdict) == {"gr_rms", "cn_tv", "angle_tv"}
    for k, v in verdict.items():
        assert v["passed"] and v["distance"] == 0.0


# ---------------------------------------------------------------- deterministic --

@pytest.mark.parametrize("k", [0, 3])
def test_rich_observables_deterministic(k, mol):
    """Same frame twice: byte-identical arrays.  A whole-cell translation
    changes nothing for the minimum-image observables (species-pair g(r),
    bond angles); the density profile is only defined up to the bin grid,
    so it is checked for reproducibility alone."""
    f = water_frame(k)
    shifted = type(f)(pos=f.pos + 7.3, cell=f.cell, symbols=f.symbols,
                      pbc=f.pbc, info=dict(f.info))

    def as_dict(result):
        return result if isinstance(result, dict) else dict(enumerate(result))

    reproducible = [
        lambda x: partial_gr(x, mol, "O", "O"),
        lambda x: partial_gr(x, mol, "H", "O"),
        lambda x: angle_distribution(x, mol),
        lambda x: density_profile(x, mol),
        lambda x: observables(x, mol, selection=("gr", "partial_gr", "angle",
                                                 "density")),
    ]
    for fn in reproducible:
        first, second = as_dict(fn(f)), as_dict(fn(f))
        for key in first:
            assert np.asarray(first[key]).tobytes() == \
                np.asarray(second[key]).tobytes(), f"not reproducible: {key}"
    translation_invariant = [
        lambda x: partial_gr(x, mol, "O", "O"),
        lambda x: partial_gr(x, mol, "H", "O"),
        lambda x: angle_distribution(x, mol),
    ]
    for fn in translation_invariant:
        first, third = as_dict(fn(f)), as_dict(fn(shifted))
        for key in first:
            assert np.allclose(np.asarray(first[key]),
                               np.asarray(third[key])), \
                f"not translation invariant: {key}"
