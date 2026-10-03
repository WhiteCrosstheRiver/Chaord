"""O2 + O3 (Reviews 4-7, docs/reviews/open_items.md): A3's geometry check
must be translation-invariant and use PLAN's StructureMatcher; A4's planter
must place interstitials at real crystallographic sites.

Red-first (AGENTS.md: the failing test is written before the fix).  Pre-fix
state, measured 2026-10-02 on this tree:

* O2: shifting the stored thermal frame by (0.7, 0, 0) A failed
  ``_geo_fit_thermal`` on 7/7 ordered cases (matched-displacement mass
  0.56-1.00 beyond 0.25 d_NN) although the lifted text is unchanged by that
  shift -- the failure was the geometry check's, not the lifter's: it read
  absolute displacement vectors with no alignment.
* O2b: the A3 evidence string claimed "p90 matched displacement vs
  0.15 d_NN" while the code gates on the fraction beyond 0.25 d_NN <= 5%.
* O2a/D6: PLAN's StructureMatcher(ltol 0.2, stol 0.3, angle 5) with its
  default primitive-cell reduction returns no fit for any thermal frame
  against its rebuild (0/9 cases) -- but it DOES match a jittered supercell
  against itself (9/9), so the committed reason for abandoning it ("no fit
  even for the frame against itself") was wrong.  With primitive_cell=False
  it fits thermal-vs-rebuild and shifted-thermal-vs-rebuild on 8/9 cases,
  but returns NO fit for l12_ni3al (1,372 atoms, ~800 s per call) and
  accepts the displace_rebuilt mutation on bcc_fe -- so the A3 gate stays
  the translation-aligned assignment and D6 is decided with these tests as
  evidence.
* O3: 9/15 held-out seed sets failed check_a4 (k = 2, 3, 5, 6, 7, 9, 10,
  12, 15).  Over those 15 sets the planted-frame minimum pair distance was
  min 0.310, p10 0.420, median 0.555 d_NN (failing draws: an interstitial
  0.40 d_NN from a host, two interstitials 0.34 d_NN apart); real fcc
  interstices sit at 0.7071 (octahedral) / 0.6124 (tetrahedral) d_NN, the
  rocksalt tetrahedral hole at 0.8660 d_NN.  After the fix the planted
  frames sit exactly at those contacts (min pair min 0.707, median 0.866,
  max 1.000 d_NN over the same 15 sets + the acceptance seeds), the
  reproducer's 15 sets pass 15/15 and 10 further fresh seed sets pass the
  full 26-cell grid 10/10.
"""
import numpy as np
import pytest
from scipy.spatial import cKDTree

from tools import acceptance as acc

# ---------------------------------------------------------------------------
# helpers (verifier-side, independent of the planter's own machinery)
# ---------------------------------------------------------------------------

O2_SHIFT = (0.7, 0.0, 0.0)          # the open-items O2 reproducer displacement


def _shifted(f, t):
    from chaord.io.frames import Frame
    return Frame(pos=np.mod(f.pos + np.asarray(t, float), f.cell_diag),
                 cell=f.cell, symbols=f.symbols, pbc=f.pbc)


def _ordered_cases():
    return [c for c in acc._exact_roundtrip_cases(None)
            if not acc._is_random_solution(c)]


def _lift_build(frame, dialect, seed=5):
    from chaord.build import build_program
    from chaord.lang.parser import parse_text
    from chaord.lift import lift_frame
    text = acc.format_program_text(lift_frame(frame, dialect))
    return build_program(parse_text(text), dialect,
                         rng=np.random.default_rng(seed))


def _min_pair(pos, L):
    p = np.mod(np.asarray(pos, float), L)
    pr = cKDTree(p, boxsize=L).query_pairs(3.5, output_type="ndarray")
    d = p[pr[:, 1]] - p[pr[:, 0]]
    d -= L * np.round(d / L)
    return float(np.linalg.norm(d, axis=1).min())


def _nn_distances(pos, L):
    """Distance from every atom to its nearest other atom (MIC)."""
    p = np.mod(np.asarray(pos, float), L)
    dd, _ = cKDTree(p, boxsize=L).query(p, k=2)
    return dd[:, 1]


# ---------------------------------------------------------------------------
# O2: the geometry gate is translation-invariant
# ---------------------------------------------------------------------------

def test_a3_geometry_check_translation_invariant_reproducer():
    """The O2 reproducer: every ordered case, stored thermal frame rigidly
    shifted by (0.7, 0, 0) A, lifted and rebuilt -- the geometry gate must
    pass (pre-fix: 7/7 fail)."""
    from chaord.dialects import load_dialect
    bad = []
    for case in _ordered_cases():
        dl = load_dialect(case["dialect"])
        g = _shifted(acc._a2a3_frame(case), O2_SHIFT)
        ok, note = acc._geo_fit_thermal(g, _lift_build(g, dl))
        if not ok:
            bad.append(f"{case['id']}: {note}")
    assert not bad, "; ".join(bad)


def test_geo_fit_thermal_random_translations():
    """Pure unit: given one rebuild, the gate is a function of the frame's
    rigid invariants alone -- 20 random translations of fcc_cu and 3 of every
    other ordered case must pass (20-draw rule for the random element)."""
    from chaord.dialects import load_dialect
    rng = np.random.default_rng(20261002)
    for case in _ordered_cases():
        dl = load_dialect(case["dialect"])
        f = acc._a2a3_frame(case)
        rebuilt = _lift_build(f, dl)
        draws = 20 if "fcc_cu" in case["id"] else 3
        for _ in range(draws):
            g = _shifted(f, rng.uniform(0, 1, 3) * f.cell_diag)
            ok, note = acc._geo_fit_thermal(g, rebuilt)
            assert ok, f"{case['id']}: {note}"


def test_a3_evidence_string_states_the_applied_gate():
    """O2b: the evidence string must state the gate the code actually applies
    (pre-fix it claims 'p90 matched displacement vs 0.15 d_NN' while the code
    gates on the displaced-mass fraction beyond 0.25 d_NN <= 5%)."""
    r = acc.check_a3(case_filter="fcc_cu")
    assert r["passed"], r["evidence"]
    ev = r["evidence"]
    assert "0.15 d_NN" not in ev, ev
    assert ("0.25 d_NN" in ev and "5%" in ev) or "StructureMatcher" in ev, ev


# ---------------------------------------------------------------------------
# O2a / D6: PLAN's StructureMatcher, committed measurements
# ---------------------------------------------------------------------------

def _matcher_pair(case):
    """(frame_pymatgen, rebuilt_pymatgen, shifted_pymatgen, shifted_rebuilt)
    for one case; random solutions anonymised species-blind as check_a3 does."""
    from chaord.dialects import load_dialect
    dl = load_dialect(case["dialect"])
    to_pm = (acc._anonymized_pymatgen if acc._is_random_solution(case)
             else acc.frame_to_pymatgen)
    f = acc._a2a3_frame(case)
    rebuilt = _lift_build(f, dl)
    g = _shifted(f, O2_SHIFT)
    return to_pm(f), to_pm(rebuilt), to_pm(g), to_pm(_lift_build(g, dl))


# l12_ni3al is excluded from the fast matcher tests: at 1,372 atoms every
# StructureMatcher call costs 10-800 s (measured 2026-10-02: self/default
# 39 s, rebuild/default 24 s, rebuild/primitive_cell=False 809 s -> no fit;
# its own slow test below owns the no-fit assertion)
_L12 = "l12_ni3al"


def test_structure_matcher_default_primitive_reduction_is_the_blocker():
    """Committed measurement (D6 evidence): PLAN's matcher with its DEFAULT
    primitive-cell reduction never fits a stored thermal frame against its
    own rebuild (0/9 cases; this was the stated but untested reason the
    criterion abandoned it).  The stronger claim quoted with it -- 'no fit
    even for the frame against ITSELF' -- is refuted: the default matcher
    self-matches every thermal frame measured (8/8 fast cases; l12_ni3al
    self-match also fits, 39 s, measured 2026-10-02)."""
    from pymatgen.analysis.structure_matcher import StructureMatcher
    sm = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)
    for case in _ordered_cases():
        f_pm, r_pm, _, _ = _matcher_pair(case)
        if _L12 not in case["id"]:
            assert sm.fit(f_pm, f_pm), f"{case['id']}: jittered self-match"
        assert not sm.fit(f_pm, r_pm), (
            f"{case['id']}: default matcher unexpectedly fits thermal vs rebuild")


def test_structure_matcher_primitive_cell_false_fits_small_thermal_frames():
    """D6: with primitive_cell=False (still PLAN's ltol/stol/angle_tol) the
    matcher fits the thermal frame against its rebuild AND against the
    rebuild of a rigidly translated copy -- on every case small enough for
    the matcher to reduce (all 8 except l12_ni3al, 1,372 atoms; the next
    test owns that exception)."""
    from pymatgen.analysis.structure_matcher import StructureMatcher
    sm = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=False)
    for case in acc._exact_roundtrip_cases(None):
        if _L12 in case["id"]:
            continue
        f_pm, r_pm, g_pm, gr_pm = _matcher_pair(case)
        assert sm.fit(f_pm, r_pm), f"{case['id']}: no thermal-vs-rebuild fit"
        assert sm.fit(g_pm, gr_pm), f"{case['id']}: no shifted-vs-rebuild fit"


@pytest.mark.slow
def test_structure_matcher_primitive_cell_false_misses_l12_ni3al():
    """D6, the decisive exception (slow: one fit on 1,372 atoms takes
    ~10-15 min): primitive_cell=False returns NO fit for the l12_ni3al
    thermal frame against its rebuild, so it cannot serve as A3's gate.
    Measured 2026-10-02: no fit, 809 s (rebuild) and 614 s (shifted)."""
    from pymatgen.analysis.structure_matcher import StructureMatcher
    sm = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=False)
    case = next(c for c in _ordered_cases() if _L12 in c["id"])
    f_pm, r_pm, _, _ = _matcher_pair(case)
    assert not sm.fit(f_pm, r_pm), (
        "l12_ni3al now fits -- D6's answer changed, revisit the A3 gate")


def test_a3_displace_mutation_power_on_the_canary_case():
    """Power of the geometry gate against the registered displace_rebuilt
    canary (fcc_cu, the case tests/acceptance/test_mutations.py runs): the
    seeded fault displaces a quarter of the rebuilt atoms by 0.6 A; on
    fcc_cu six of the eight displaced atoms land beyond 0.25 d_NN and BOTH
    candidate gates reject it -- the translation-aligned assignment gate
    and PLAN's matcher with primitive_cell=False.

    On bcc_fe (16 atoms) the same seeded draw displaces its four atoms by
    only 0.37-0.54 A -- all below the 0.25 d_NN (0.55 A) gate -- so NO
    displacement gate can catch it there (this holds for the pre-O2 gate
    too: it is a weakness of that mutation draw on the smallest cell, not
    of the alignment; recorded so the matcher's fit=True on bcc_fe is never
    read as an assignment-gate failure)."""
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from chaord.dialects import load_dialect
    from chaord.io.frames import Frame
    sm = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=False)
    case = next(c for c in _ordered_cases() if "fcc_cu" in c["id"])
    dl = load_dialect(case["dialect"])
    f = acc._a2a3_frame(case)
    rebuilt = _lift_build(f, dl)
    rng = np.random.default_rng(1)
    idx = rng.permutation(len(rebuilt))[: len(rebuilt) // 4]
    pos = rebuilt.pos.copy()
    pos[idx] += rng.normal(size=(len(idx), 3)) * 0.6
    mutated = Frame(pos=np.mod(pos, rebuilt.cell_diag), cell=rebuilt.cell,
                    symbols=rebuilt.symbols, pbc=rebuilt.pbc)
    ok, note = acc._geo_fit_thermal(f, mutated)
    assert not ok, f"the assignment gate must keep the power: {note}"
    assert not sm.fit(acc.frame_to_pymatgen(f), acc.frame_to_pymatgen(mutated)), (
        "PLAN's matcher (primitive_cell=False) must reject the canary fault on "
        "fcc_cu too")

    rng = np.random.default_rng(1)
    n_bcc = 16
    idx = rng.permutation(n_bcc)[: n_bcc // 4]
    bcc_steps = np.linalg.norm(rng.normal(size=(len(idx), 3)) * 0.6, axis=1)
    assert bcc_steps.max() < 0.25 * 2.201, (
        "the bcc_fe seeded draw grew above the gate -- promote it to a real "
        "power assertion")


# ---------------------------------------------------------------------------
# O3: the planter plants physical interstitials
# ---------------------------------------------------------------------------

# verifier-side crystallography (International Tables): (site class,
# contact to nearest host as a fraction of the host's d_NN) per A4 host --
# fcc-family octahedral hole a/2 / (a/sqrt(2)) = 0.7071; rocksalt's
# octahedral holes are filled by the counter-ion, its empty tetrahedral
# hole (a*sqrt(3)/4) / (a/2) = 0.8660.
HOST_INTERSTITIAL_CONTACT = {"fcc-Cu": 0.7071, "L12-NiAl": 0.7071,
                             "NaCl": 0.8660}


def _planted_frames(seed_tag=None):
    """Every A4 planting (host x plan x temperature draw seed) as
    (tag, dtype, frame, d_nn, n_interstitial).  Re-derives the planting
    exactly as check_a4 does, optionally under a held-out seed override."""
    from chaord.build.crystal import build_conventional
    orig = acc._stable_seed
    try:
        if seed_tag is not None:
            acc._stable_seed = lambda s: orig(f"{s}|{seed_tag}")
        for host in acc.A4_HOSTS:
            tag, name, params, slots, reps = host
            perfect = build_conventional(name, params, slots, reps)
            d_nn = acc.median_nn_distance(perfect.pos, perfect.cell_diag)
            _kind, site_pos, site_contact = acc._interstitial_site_lattice(
                name, params, reps)
            for dtype, planting in acc.A4_PLANS[tag]:
                n_inter = sum(n for tok, n in planting.items()
                              if tok.endswith("_i") or tok == "frenkel_pair")
                for temp in ("room", "0.8Tm"):
                    seed = acc._stable_seed(f"{tag}|{dtype}|{temp}")
                    rng = np.random.default_rng(seed)
                    if dtype == "mixed":
                        frame = acc._plant_defects_mixed(
                            perfect, [(acc._kv_kind(tok), tok, n)
                                      for tok, n in planting.items()],
                            d_nn, rng, interstitial_sites=site_pos,
                            interstitial_contact=site_contact)
                    else:
                        (token, n_planted), = planting.items()
                        frame = acc._plant_defects(
                            perfect, dtype, token, n_planted, d_nn, rng,
                            interstitial_sites=site_pos,
                            interstitial_contact=site_contact)
                    yield tag, dtype, frame, d_nn, n_inter
    finally:
        acc._stable_seed = orig


def test_a4_planted_frames_pass_physical_sanity():
    """O3 (AGENTS rule: every planted input passes the sanity checks): over
    the acceptance seeds AND five held-out seed sets (the reproducer's
    failing ones), every planted frame has (a) min pair distance >= 0.8 x
    its crystallographic contact floor (0.8 x d_NN with no interstitials; a
    literal 0.8 d_NN floor rejects every true fcc interstice -- octahedral
    0.7071, tetrahedral 0.6124 d_NN -- so the floor is 0.8 x the
    interstitial-site contact for cells that plant one); (b) every
    interstitial at its host's crystallographic interstice contact
    (+/- 0.05 d_NN); (c) distinct planted interstitials >= 2 d_NN apart.
    Pre-fix: min pair reaches 0.31-0.42 d_NN on interstitial cells."""
    worst = []
    for seed_tag in (None, "heldout2", "heldout3", "heldout5", "heldout6",
                     "heldout7"):
        for tag, dtype, frame, d_nn, n_inter in _planted_frames(seed_tag):
            L = frame.cell_diag
            contact = HOST_INTERSTITIAL_CONTACT[tag]
            floor = 0.8 * (contact if n_inter else 1.0) * d_nn
            m = _min_pair(frame.pos, L)
            worst.append(m / d_nn)
            assert m >= floor, (
                f"{tag}/{dtype} [{seed_tag}]: min pair {m / d_nn:.3f} d_NN < "
                f"floor {floor / d_nn:.3f} d_NN")
            if n_inter:
                nn = _nn_distances(frame.pos, L)
                inter_nn = nn[-n_inter:]   # planters append interstitials last
                for d in inter_nn:
                    assert abs(d / d_nn - contact) <= 0.05, (
                        f"{tag}/{dtype} [{seed_tag}]: interstitial "
                        f"{d / d_nn:.3f} d_NN from its nearest neighbour, "
                        f"expected the {contact:.4f} d_NN interstice contact")
                p = np.mod(frame.pos[-n_inter:], L)
                for i in range(len(p)):
                    for j in range(i + 1, len(p)):
                        d = p[i] - p[j]
                        d -= L * np.round(d / L)
                        assert np.linalg.norm(d) >= 2.0 * d_nn, (
                            f"{tag}/{dtype} [{seed_tag}]: two interstitials "
                            f"{np.linalg.norm(d) / d_nn:.3f} d_NN apart")
    assert worst, "no planted frames enumerated"


def test_a4_held_out_seed_sets_pass():
    """The O3 reproducer: 15 held-out seed sets (the exact open-items
    monkeypatch), 0.8Tm draw -- pre-fix 9 of 15 fail (k = 2, 3, 5, 6, 7, 9,
    10, 12, 15)."""
    orig = acc._stable_seed
    fails = []
    try:
        for k in range(1, 16):
            acc._stable_seed = lambda s, k=k: orig(f"{s}|heldout{k}")
            r = acc.check_a4(temps=("0.8Tm",))
            if not r["passed"]:
                fails.append(k)
    finally:
        acc._stable_seed = orig
    assert not fails, f"held-out seed sets still failing: {fails}"


@pytest.mark.slow
def test_a4_ten_new_seed_sets_full_grid_pass():
    """O3 Done: 10 NEW seed sets (disjoint from the reproducer's heldout1-15)
    x the full 26-cell grid (3 hosts, all K-V kinds plus the mixed cells,
    both temperatures) -- pre-fix the plantings are unphysical and seed
    sets fail."""
    orig = acc._stable_seed
    fails = []
    try:
        for k in range(21, 31):
            acc._stable_seed = lambda s, k=k: orig(f"{s}|fresh{k}")
            r = acc.check_a4()
            if not r["passed"]:
                fails.append((k, r["evidence"]))
    finally:
        acc._stable_seed = orig
    assert not fails, "; ".join(f"k={k}: {e}" for k, e in fails)
