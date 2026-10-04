"""O5: every bench frame is physical and every lifted program compiles.

Reference-data-style sanity over the FROZEN bench/data tree (O5, Review 6,
2026-10-03; AGENTS.md: "every test input, planted or synthetic, passes the
reference-data sanity checks"):

  (a) minimum pair distance against the system's hard core, by family:
      * molecular frames (dialect core+molecular, plus the Cu/water interface
        whose water side is molecular): no UNBONDED pair inside the molecular
        dialect's covalent bond window -- for every atom pair of two different
        bond-graph components whose elements are both bondable (the dialect's
        ion_solvation_elements are solvation contacts and never bond, exactly
        as in the bond graph itself), d >= bond_tolerance x (r_i + r_j).
        Intramolecular bond lengths are legal by definition; the window IS the
        physical cutoff the generator packs against;
      * reduced-unit X frames: min pair >= the lj dialect's overlap_tolerance
        (sigma); intramolecular does not exist (single atoms);
      * every other frame (crystals, defects, alloy, slabs): min pair >= the
        core dialect's overlap_tolerance (0.5 A) -- thermally displaced
        crystal nearest neighbours legitimately dip toward it, never through.
  (b) density within +-10% of the density the case describes (the frozen
      ground truth records the target: stated mass density for the dense
      packed boxes, mass/box for the plain packed fluids, number density for
      reduced-unit and exact-construction cases, water-region density for the
      Cu/water interface).
  (c) the lifted program of every frame BUILDS with physics off -- O5's core
      requirement (A13 only lifted). gases/co2_dense failed this before its
      2026-10 regeneration ("cannot place 60 CO2 at this density": the frames
      were 2.54 g/cm3 while the description claimed ~1.0; the compiler's
      molecular-sphere packer jams near 0.7 g/cm3, and 2.54 was also past any
      equilibrium fluid CO2 density); it is regenerated at 0.68 g/cm3 (a
      liquid-like supercritical state point), which rebuilds on every seed.

Pinned defects (asserted to STILL fail, never skipped: fixing the underlying
defect flips the pin red and forces the case back into the strict checks):

  * solutions/lipf6_ec and surfaces/si001_2x1 lift to programs that name
    vocabulary the compiler does not implement (species C3H4O3/F6P -- the EC
    and PF6- templates exist only in bench/generate.py -- and adsorption site
    'far'); closing those gaps is src work outside this stream.

  interface/lj_solid_liquid was the third pin (liquid half uniform random,
  min pair 0.047-0.18 sigma); W6 (Review 8, 2026-10-04) regenerated the
  liquid half as RSA at >= 0.85 sigma followed by a short LJ MD relaxation
  (T* = 0.7, frozen solid half) in its own reviewed commit with
  checksums.json, and the pin was removed so the strict 0.80 sigma hard-core
  rule governs again.
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

from ase.data import atomic_masses, chemical_symbols, covalent_radii  # noqa: E402

from chaord.build.molecules import bond_graph, molecular_mass  # noqa: E402
from chaord.dialects import load_dialect  # noqa: E402
from chaord.io.frames import read_frame  # noqa: E402

DATA = ROOT / "bench" / "data"
MOLECULAR = load_dialect(("core", "molecular"))
LJ = load_dialect(("core", "lj"))
CORE = load_dialect(("core",))
BOND_TOL = float(MOLECULAR.threshold("bond_tolerance"))
ION_ELEMENTS = frozenset(MOLECULAR.threshold("ion_solvation_elements"))
LJ_OVERLAP = float(LJ.threshold("overlap_tolerance"))       # sigma
CORE_OVERLAP = float(CORE.threshold("overlap_tolerance"))   # A
DENSITY_TOL = 0.10          # +-10% against the described density (O5 wording)
AMU_A3_TO_G_CM3 = 1.66053906660  # CODATA amu/A^3 -> g/cm3

# lift_frame's own mode vocabulary (mirrors tools/acceptance.py): a ground
# truth 'lift_mode' that names a bench category (not a mode) lifts auto
_LIFT_MODES = {"pipeline", "crystal", "defects", "amorphous", "surface",
               "fluid", "slab"}

# lifted programs that name compiler vocabulary which does not exist in src
# yet; each entry is (exact static-error substring, must_always_fail). A build
# failure for any OTHER reason is always red; when must_always_fail is set, a
# successful build is red too (the gap closed -- delete the pin). si001_2x1 is
# flagged must_always_fail=False because its surface lift is not
# process-history invariant (a fresh process lifts every frame to `site
# bridge` only, which builds; after certain other lifts in the same process
# some draws classify one atom as `site far`, which the builder rejects -- a
# determinism bug of the surface lift itself, same family as O1, reported
# 2026-10-02): a draw that compiles satisfies O5 for that draw, a `far` draw
# must fail with exactly the pinned reason.
PINNED_COMPILER_GAPS = {
    "solutions/lipf6_ec":
        ("unknown species 'C3H4O3'", True),
    "surfaces/si001_2x1":
        ("unknown adsorption site 'far'", False),
}

# frozen frames that violate their family hard core and cannot be regenerated
# in this stream; the distance test asserts the violation persists (tripwire:
# a physical regeneration flips it red and the case reverts to the strict rule).
# interface/lj_solid_liquid was removed here by W6 (2026-10-04): its liquid
# half is regenerated physical (RSA >= 0.85 sigma + LJ MD relaxation), so the
# strict LJ hard-core assert governs it again.
PINNED_HARDCORE_VIOLATIONS: set = set()


def _cases():
    manifest = json.loads((DATA / "index.json").read_text(encoding="utf-8"))
    out = {}
    for entry in manifest["cases"]:
        gt = json.loads((DATA / entry["ground_truth"]).read_text("utf-8"))
        out[entry["id"]] = {"frames": [DATA / f for f in entry["frames"]],
                            "gt": gt}
    return out


CASES = _cases()


def _elements_mass(symbols) -> float:
    return float(sum(atomic_masses[chemical_symbols.index(s)] for s in symbols))


def _min_image_min_pair(frame) -> float:
    L = frame.cell_diag
    pos = np.mod(frame.pos, L)
    pos = np.minimum(pos, L * (1 - 1e-9))
    tree = cKDTree(pos, boxsize=L)
    d, _ = tree.query(pos, k=2)
    return float(d[:, 1].min())


def _components(frame):
    """Bond-graph connected components under the molecular dialect rule."""
    edges = bond_graph(frame, MOLECULAR)
    n = len(frame)
    if not edges:
        return np.arange(n)
    rows = [e[0] for e in edges] + [e[1] for e in edges]
    cols = [e[1] for e in edges] + [e[0] for e in edges]
    _, labels = connected_components(
        coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)),
        directed=False)
    return labels


def _min_unbonded_pair_over_window(frame) -> tuple[float, str]:
    """(min over unbonded bondable pairs of d - bond_window, description).

    Intramolecular pairs (same component) are legal bond lengths; pairs whose
    elements the dialect declares solvation ions never bond (their contact is
    ion solvation, not covalence) and are exempt, mirroring the bond graph.
    """
    L = frame.cell_diag
    pos = np.mod(frame.pos, L)
    pos = np.minimum(pos, L * (1 - 1e-9))
    syms = frame.symbols
    labels = _components(frame)
    rad = {s: covalent_radii[chemical_symbols.index(s)]
           for s in set(syms)}
    r_max = BOND_TOL * 2 * max(rad.values())
    tree = cKDTree(pos, boxsize=L)
    pairs = tree.query_pairs(r_max, output_type="ndarray")
    worst, desc = np.inf, "no unbonded bondable pair within the window scan"
    for i, j in pairs:
        if labels[i] == labels[j]:
            continue                          # intramolecular: legal bond
        if syms[i] in ION_ELEMENTS or syms[j] in ION_ELEMENTS:
            continue                          # solvation contact: never bonds
        d = pos[j] - pos[i]
        d -= L * np.round(d / L)
        margin = float(np.linalg.norm(d)) - BOND_TOL * (rad[syms[i]]
                                                        + rad[syms[j]])
        if margin < worst:
            worst, desc = margin, f"{syms[i]}-{syms[j]}"
    return worst, desc


def _is_molecular(case_id, gt) -> bool:
    return (gt.get("dialect") == "core+molecular"
            or case_id == "interfaces/cu_water")


def _density_target(case_id, gt):
    """(kind, target, note) -- the density the case's ground truth describes."""
    exp = gt["expected"]
    if case_id == "interfaces/cu_water":
        return ("mass", float(exp["water_density_g_cm3"]),
                "water region (upper half of the doubled box)")
    if "density_g_cm3" in gt:
        return ("mass", float(gt["density_g_cm3"]),
                "recorded achieved mass density")
    if "molecules" in gt and "box" in gt:
        mass = sum(molecular_mass(m) * n for m, n in gt["molecules"].items())
        return ("mass", mass * AMU_A3_TO_G_CM3 / float(np.prod(gt["box"])),
                "packed molecules in the stated box")
    if case_id == "glass/lj_glass_rho085":
        return ("number", float(exp["density"]), "stated number density")
    if case_id == "interface/lj_solid_liquid":
        # both half-boxes hold equal counts, so the overall number density is
        # the stated rho_solid (the ground-truth description records 1.0)
        return ("number", 1.0, "rho_solid, both halves at equal counts")
    if "cell" in exp:
        return ("number", float(exp["n_atoms"]) / float(np.prod(exp["cell"])),
                "exact-construction number density")
    if "host" in gt:
        # defect hosts keep the (cubic) host box: fcc/bcc/rocksalt/L1_2
        # conventional cells tile it exactly, vacancies and antisites do not
        # change the volume
        host = gt["host"]
        V = float(host["a"]) ** 3 * float(np.prod(host["reps"]))
        return ("number", float(exp["n_atoms"]) / V,
                "defective host in the perfect-host box")
    raise KeyError(f"{case_id}: ground truth states no density to check")


# ------------------------------------------------------------ (a) hard core --

@pytest.mark.parametrize("case_id", sorted(CASES), ids=lambda c: c.replace("/", "_"))
def test_min_pair_distance_respects_the_system_hard_core(case_id):
    entry = CASES[case_id]
    gt = entry["gt"]
    molecular = _is_molecular(case_id, gt)
    all_x = set(read_frame(entry["frames"][0]).symbols) == {"X"}
    for k, path in enumerate(entry["frames"]):
        frame = read_frame(path)
        if molecular:
            worst, pair = _min_unbonded_pair_over_window(frame)
            assert worst >= -1e-9, (
                f"{case_id} frame {k}: unbonded {pair} pair inside the "
                f"covalent bond window by {-worst:.3f} A (hard core violated)")
        elif all_x:
            floor = LJ_OVERLAP
            got = _min_image_min_pair(frame)
            if case_id in PINNED_HARDCORE_VIOLATIONS:
                # tripwire: still unphysical -> still pinned; a physical
                # regeneration fails here and the case reverts to the assert
                # above (delete the pin in the same reviewed commit)
                assert got < floor, (
                    f"{case_id} frame {k}: min pair {got:.3f} sigma now meets "
                    f"the {floor} sigma hard core -- remove the pin and let "
                    "the strict check govern")
            else:
                assert got >= floor, (
                    f"{case_id} frame {k}: min pair {got:.3f} sigma < "
                    f"{floor} sigma hard core")
        else:
            got = _min_image_min_pair(frame)
            assert got >= CORE_OVERLAP, (
                f"{case_id} frame {k}: min pair {got:.3f} A < {CORE_OVERLAP} "
                "A static-overlap floor")


# --------------------------------------------------------------- (b) density --

@pytest.mark.parametrize("case_id", sorted(CASES), ids=lambda c: c.replace("/", "_"))
def test_density_matches_the_described_target(case_id):
    entry = CASES[case_id]
    gt = entry["gt"]
    kind, target, note = _density_target(case_id, gt)
    for k, path in enumerate(entry["frames"]):
        frame = read_frame(path)
        V = float(np.prod(frame.cell_diag))
        if kind == "mass":
            if case_id == "interfaces/cu_water":
                n_water = sum(1 for s in frame.symbols if s == "O")
                measured = n_water * molecular_mass("H2O") \
                    * AMU_A3_TO_G_CM3 / (V / 2.0)
            else:
                measured = _elements_mass(frame.symbols) * AMU_A3_TO_G_CM3 / V
        else:
            measured = len(frame) / V
        assert abs(measured - target) <= DENSITY_TOL * target, (
            f"{case_id} frame {k}: {measured:.4f} vs described {target:.4f} "
            f"({note}) outside +-{DENSITY_TOL:.0%}")


# ------------------------------------------------------- (c) build the lift --

def _lift(frame, dl, recorded_mode):
    """Lift as tools/acceptance.py does: recorded mode when it is a mode,
    auto otherwise; a refusal of the recorded mode falls back to auto."""
    from chaord.lift import lift_frame
    if recorded_mode in _LIFT_MODES:
        try:
            return lift_frame(frame, dl, mode=recorded_mode)
        except Exception:
            return lift_frame(frame, dl)
    return lift_frame(frame, dl)


@pytest.mark.parametrize("case_id", sorted(CASES), ids=lambda c: c.replace("/", "_"))
def test_every_lifted_program_builds_physics_off(case_id):
    """O5: lift -> build (physics off) must succeed for every bench frame."""
    from chaord.build import build_program

    entry = CASES[case_id]
    gt = entry["gt"]
    dl = load_dialect(tuple(gt["lift_dialect"]))
    pinned = PINNED_COMPILER_GAPS.get(case_id)
    for k, path in enumerate(entry["frames"]):
        program = _lift(read_frame(path), dl, gt.get("lift_mode"))
        try:
            build_program(program, dl, rng=np.random.default_rng(k),
                          physics=False)
        except Exception as exc:                                  # noqa: BLE001
            if pinned is not None:
                reason, _ = pinned
                assert reason in str(exc), (
                    f"{case_id} frame {k}: build fails for a NEW reason "
                    f"({type(exc).__name__}: {exc}); only the pinned gap "
                    f"{reason!r} may fail -- investigate")
            else:
                pytest.fail(f"{case_id} frame {k}: lifted program does not "
                            f"build (physics off): {type(exc).__name__}: {exc}")
        else:
            if pinned is not None:
                reason, must_fail = pinned
                assert not must_fail, (
                    f"{case_id} frame {k}: builds again -- the compiler gap "
                    f"{reason!r} is closed; delete the pin so the strict "
                    "build check governs")


# ------------------------------------------------------------- co2 (the fix) --

def test_co2_dense_regenerated_at_physical_rebuildable_density():
    """O5 regression pin: the co2 frames carry a stated physical target density
    (fluid CO2 at equilibrium spans roughly 0.45-1.1 g/cm3 around the critical
    point; dry ice is 1.56; the pre-fix frames were 2.54 -- none of these) and
    the achieved density matches it. The rebuild side is pinned by the build
    scan above: the compiler's molecular-sphere packer jams near 0.7 g/cm3, so
    the target must stay at or below that."""
    gt = CASES["gases/co2_dense"]["gt"]
    assert "density_target_g_cm3" in gt, (
        "co2_dense ground truth must state the physical target density "
        "(pre-regeneration it only recorded the achieved 2.537 g/cm3 while "
        "the description claimed ~1.0)")
    target = float(gt["density_target_g_cm3"])
    assert 0.45 <= target <= 0.70, (
        f"target {target} g/cm3 is not a rebuildable fluid-CO2 density")
    achieved = float(gt["density_g_cm3"])
    assert abs(achieved - target) <= DENSITY_TOL * target
