"""Chaord-Bench generator (PLAN WP21): seeded case builders + ground truth.

Schema 2 (multi-frame): every case directory under --out receives five
independent frames (frame_0.npz .. frame_4.npz, keys r/L/symbols with symbols
stored as dtype U8) and one ground_truth.json. Frames are independent
microstates of the same macrostate:

  * perfect crystals   -> thermal Gaussian displacement, per-frame seed,
                          amplitude = metal dialect `thermal_test_amplitude`
                          (0.06) x d_NN;
  * random packing     -> re-packed from scratch with a per-frame seed;
  * point-defect hosts -> defect positions re-drawn per frame (counts fixed);
  * surfaces           -> adsorbate site selection re-drawn per frame.

Case families beyond the original 18 (same schema, same rules):

  * solutions/nacl_aq    120 H2O + 8 Na+ + 8 Cl- (pack_molecules, box grows
                         from 16.5 A until the packer succeeds; draws whose
                         ion-ion contact falls inside the census bond window
                         are rejected and re-drawn, so the census stays exact);
  * solutions/lipf6_ec   EC + Li+ + PF6- electrolyte at liquid density —
                         packed by atom-level RSA (pack_molecules' molecular
                         spheres jam far above liquid density for 10-atom
                         solvents); EC and PF6- templates are registered here;
  * solutions/cuau_random fcc Cu/Au 1/2-1/2 random solution (the fcc_crconi
                         construction with two species);
  * gases/co2_dense      60 CO2 at 0.68 g/cm3 (a liquid-like supercritical
                         state point, ~315 K / ~11 MPa on the Span-Wagner
                         EOS; O5: the pre-2026-10 frames were 2.54 g/cm3 and
                         their lifted programs never rebuilt), the same
                         atom-level RSA packing;
  * surfaces/si001_2x1   diamond Si(001) slab via the ASE builder, tiled to an
                         even surface cell, with an independently planted
                         p(2x1) missing-row top layer and a small thermal
                         displacement (the Wood net read compares two primitive
                         vectors per layer with no ensemble averaging, so the
                         crystal-case thermal amplitude would drown it);
  * surfaces/pt111_o     Pt(111) + O on top sites, the ni111_o_top
                         construction at the Pt lattice constant;
  * interfaces/cu_water  Cu fcc block in the lower half of a doubled box,
                         water RSA-packed into the upper half at 0.9 g/cm3
                         with the Cu atoms as excluded obstacles; the two-phase
                         label accuracy of every frame is measured and recorded.

ground_truth.json records per-frame invariants (atom counts per species,
census, defect counts, boundary z, ...) plus case-level expectations. An
index.json manifest at the root lists every case and every frame.

Run with the project virtualenv:

    .venv/Scripts/python.exe bench/generate.py --out bench/data

Everything is seeded, so re-running reproduces byte-identical frames.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from math import gcd
from pathlib import Path

import numpy as np

from chaord.build import build_program
from chaord.build.crystal import SLOT_COUNTS, build_conventional
from chaord.build.defects import (apply_defects, assign_occupancy,
                                  nearest_neighbor_distance,
                                  typical_neighbor_distance)
from chaord.build.molecules import (TEMPLATES, _register_mol, bond_graph,
                                    contact_floor_target, molecule_census,
                                    molecular_mass, pack_molecules,
                                    random_rotation,
                                    relax_intermolecular_contacts)
from chaord.dialects import load_dialect
from chaord.io.frames import Frame, write_frame
from chaord.lang.errors import ChaordError
from chaord.lang.ir import KVDefect, Name, Quantity, Statement
from chaord.lang.parser import parse_text
from chaord.lift.segment import label_accuracy, phase_labels

SCHEMA = "chaord-bench/2"
N_FRAMES = 5

# amu/A^3 -> g/cm^3 (CODATA: 1 amu = 1.66053906660e-24 g, 1 A = 1e-8 cm)
_AMU_PER_A3_TO_G_CM3 = 1.66053906660

# --------------------------------------------------------------------- specs --

# exact crystals; each frame is a fresh thermal displacement of the same host
# (case id, prototype, lattice params, species per slot, supercell reps,
#  orientation rows or None, base seed)
CRYSTAL_CASES = [
    ("fcc_cu", "fcc", {"a": 3.615}, ("Cu",), (2, 2, 2), None, 101),
    ("bcc_fe", "bcc", {"a": 2.87}, ("Fe",), (2, 2, 2), None, 103),
    ("rocksalt_nacl", "rocksalt", {"a": 5.64}, ("Na", "Cl"), (2, 2, 2), None, 107),
    # 7x7x7 conventional L1_2 cells = 1,372 atoms: A9 (compression) needs a
    # raw bench frame of >= 1,000 atoms
    ("l12_ni3al", "L1_2", {"a": 3.572}, ("Ni", "Al"), (7, 7, 7), None, 109),
    ("hcp_mg", "hcp", {"a": 3.21, "c": 5.21}, ("Mg",), (2, 2, 2),
     np.array([[1, 0, 0], [1, 2, 0], [0, 0, 1]], int), 113),
    ("diamond_si", "diamond", {"a": 5.43}, ("Si",), (2, 2, 2), None, 127),
    ("perovskite_srtio3", "perovskite", {"a": 3.905}, ("Sr", "Ti", "O"),
     (2, 2, 2), None, 131),
]

# fcc solid solutions: same fcc host, species assigned to sites with unit
# fractions (a fresh random assignment per frame; counts stay exact)
SOLUTION_SPECS = [
    {"id": "fcc_crconi", "category": "crystals", "prototype": "fcc",
     "params": {"a": 3.56}, "host_species": ("Cr",),
     "occupancy": (("Cr", "1/3"), ("Co", "1/3"), ("Ni", "1/3")),
     "occ_text": "Cr/Co/Ni assigned to fcc sites with unit fractions",
     "reps": (3, 3, 3), "seed": 137},
    {"id": "cuau_random", "category": "solutions", "prototype": "fcc",
     "params": {"a": 3.8}, "host_species": ("Cu",),
     "occupancy": (("Cu", "1/2"), ("Au", "1/2")),
     "occ_text": "Cu/Au assigned to fcc sites with unit fractions",
     "reps": (3, 3, 3), "seed": 199},
]

# point-defect hosts; defect sites are re-drawn per frame, counts are fixed
DEFECT_CASES = [
    {"id": "fcc_cu_vacancies", "prototype": "fcc", "params": {"a": 3.615},
     "species": ("Cu",), "reps": (3, 3, 3), "defects": [("V_Cu", 3)], "seed": 11},
    {"id": "l12_ni3al_vac_antisite", "prototype": "L1_2", "params": {"a": 3.572},
     "species": ("Ni", "Al"), "reps": (4, 4, 4),
     "defects": [("V_Ni", 3), ("Al_Ni", 4)], "seed": 139},
    {"id": "rocksalt_nacl_vna", "prototype": "rocksalt", "params": {"a": 5.64},
     "species": ("Na", "Cl"), "reps": (3, 3, 3), "defects": [("V_Na", 6)],
     "seed": 149},
]

# random-sequential packing of rigid molecules; re-packed per frame.
# "pack_then_relax": True (W8, review open_items_v2.md step 3) adds the
# rigid-body contact-opening relaxation after the RSA (pack-then-relax), so
# every intermolecular pair clears the 0.75x-Bondi contact floor; the plain
# RSA packs at the census-safe sphere contact, whose worst O-O (1.75-1.9 A)
# sits below that floor.  Only the cases W8 regenerates carry the flag:
# ar_gas_box25 and n2_box22 keep their frozen bytes.
FLUID_CASES = [
    {"id": "water_box15", "molecules": {"H2O": 60}, "box": (15.0, 15.0, 15.0),
     "seed": 13, "pack_then_relax": True},
    {"id": "ar_gas_box25", "molecules": {"Ar": 50}, "box": (25.0, 25.0, 25.0),
     "seed": 151},
    {"id": "n2_box22", "molecules": {"N2": 80}, "box": (22.0, 22.0, 22.0),
     "seed": 157},
]

REACTIVE_CASE = {"id": "water_oh_h_box20", "molecules": {"H2O": 50, "OH": 9, "H": 9},
                 "box": (20.0, 20.0, 20.0), "seed": 3,
                 "pack_then_relax": True}

# LJ glass: overlap-free RSA packing at a stated number density (physics off)
GLASS_CASE = {"id": "lj_glass_rho085", "n": 200, "rho": 0.85, "species": "X",
              "seed": 163}

# LJ solid-liquid slab (W6, Review 8, 2026-10-04): the fcc lower half is kept
# perfect; the liquid half is regenerated PHYSICALLY. The pre-W6 frames placed
# the liquid half as uniform random points (min pair 0.047-0.18 sigma), an
# unphysical input that made A7's phase-segmentation evidence trivial
# (perfect crystal vs overlapping random points). The liquid now carries the
# LJ coexistence liquid density (rho_liquid = 0.85 at T* ~ 0.7; the reference
# case melts and equilibrates at 0.845/0.65): with equal atom counts per
# equal half-boxes the liquid would sit at rho_solid = 1.0, the fcc density,
# and a T* = 0.7 relaxation freezes it epitaxially onto the perfect wall
# (measured: liquid-core q6bar crosses the 0.32 solid threshold between 4 and
# 6 tau). So the z cell is asymmetric: 4 perfect fcc cells at rho_solid below
# (256 atoms), then a liquid region of the volume that gives 256 atoms at
# rho_liquid above, and the two-phase label boundary stays the box mid-plane
# z = Lz/2 (A7's periodic second interface plane). The liquid is placed by
# RSA at a 0.85 sigma hard core (liquid-liquid and liquid-solid, minimum
# image) and relaxed by short Lennard-Jones MD at T* = 0.7 with the solid
# atoms frozen (ASE Langevin, the same engine stack as
# bench/reference/generate_reference.py). Every engine parameter is recorded
# in the case ground truth under "md".
INTERFACE_CASE = {"id": "lj_solid_liquid", "rho_solid": 1.0, "rho_liquid": 0.85,
                  "reps": (4, 4, 8), "species": "X", "seed": 167}
# RSA hard core for the liquid placement, sigma (review-approved 0.85; the lj
# dialect's bulk rsa_dmin 0.90 is feasible here but 0.85 is the reviewed
# value and leaves the same margin to the 0.80 hard-core floor after MD)
INTERFACE_RSA_DMIN = 0.85
INTERFACE_RSA_MAX_TRIES = 4_000_000
# relaxation protocol: two Langevin stages, total 12 tau. Duration measured
# time-resolved at rho_l = 0.85, T* = 0.7 (two seeds): liquid structure is
# formed by ~7 tau (liquid-core q6bar ~0.16, min pair > 0.9); the first
# transient solid-like blips in the liquid core appear at ~17 tau, so 12 tau
# relaxes the RSA packing with margin on both sides
INTERFACE_MD = {"T_star": 0.7, "dt_fast": 0.002, "steps_fast": 1000,
                "dt": 0.005, "steps": 2000, "friction_per_tau": 0.5,
                "time_tau": 1000 * 0.002 + 2000 * 0.005}

# Ni(111)/Pt(111) slabs + O adsorbates, built by the real surface builder, then
# stored in its rectangular (orthohexagonal, index-2) supercell so the
# diagonal-only npz snapshot format round-trips exactly
SURF_PROGRAM = """chaord 0.1
dialect core + metal + surface

system {{
  cell {lx:.3f} {ly:.3f} {lz:.3f}
  pbc xyz
  conserve atoms {el} {n}
}}

physics {{
  backend eam
}}

crystal slab : slab z 0 .. {ztop:.1f} {{
  lattice fcc
  a {a:.3f} A
  surface ({hkl}) top
  adsorb O count 4 site top
}}

vacuum gap : slab z {ztop:.1f} .. {lz:.1f} {{
}}
"""
SURFACE_CASES = [
    {"id": "ni111_o_top", "category": "surface", "element": "Ni", "a": 3.615,
     "hkl": "111",
     # 3x3x4 ASE fcc(111) slab at a = 3.615 with a 12 A vacuum
     "lx": 7.669, "ly": 6.641, "lz": 18.26, "ztop": 6.3, "n_slab": 36,
     "seed": 173},
    {"id": "pt111_o", "category": "surfaces", "element": "Pt", "a": 3.92,
     "hkl": "111",
     # same construction at the Pt lattice constant (slab height 3 x 3.92/sqrt(3))
     "lx": 8.315, "ly": 7.201, "lz": 18.788, "ztop": 6.8, "n_slab": 36,
     "seed": 193},
]

# ------------------------------------------------------------------ new ----
# WP21 extension cases (solutions / gases / reconstructed surfaces / solid-
# liquid interfaces). All builders keep the bench rules: seeded, exact counts,
# census verified on every frame.

# aqueous NaCl: pack_molecules from the nominal 16.5 A box; the packer jams at
# molecular-sphere RSA density, so the box grows until it succeeds and the
# ground truth records the actual box and density. Draws whose Na-Na / Na-Cl
# contact lands inside the census bond window (covalent radii are generous for
# metals) are rejected and re-drawn so the census stays exact.
NACL_CASE = {"id": "nacl_aq", "molecules": {"H2O": 120, "Na+": 8, "Cl-": 8},
             "box0": 16.5, "box_step": 0.5, "box_max": 21.0,
             "census_retries": 40, "seed": 179}

# ethylene carbonate (C3H4O3) and hexafluorophosphate templates for the
# electrolyte case. Simplified geometry (dialect-exempt per AGENTS.md): a
# regular pentagon ring with 1.43 A bonds, carbonyl C=O 1.20 A, C-H 1.09 A
# (35-degree H-C-H spread), octahedral P-F at 1.58 A.
def _ec_template() -> tuple[list, list]:
    R = 1.43 / (2.0 * np.sin(np.deg2rad(36.0)))   # pentagon circumradius
    ring = {}
    for name, deg in (("C1", 90), ("O2", 162), ("C3", 234), ("C4", 306),
                      ("O5", 18)):
        ring[name] = R * np.array([np.cos(np.deg2rad(deg)),
                                   np.sin(np.deg2rad(deg)), 0.0])
    ring["O6"] = ring["C1"] + np.array([0.0, 1.20, 0.0])

    def hydrogens(c):
        r = np.asarray(c[:2], float)
        r = r / np.linalg.norm(r)
        perp = np.array([-r[1], r[0]])
        out = []
        for side in (1.0, -1.0):
            d = (np.cos(np.deg2rad(35.0)) * r
                 + side * np.sin(np.deg2rad(35.0)) * perp)
            out.append(np.array([c[0] + 1.09 * d[0], c[1] + 1.09 * d[1], 0.0]))
        return out

    h3, h4 = hydrogens(ring["C3"]), hydrogens(ring["C4"])
    symbols = ["C", "O", "C", "C", "O", "O", "H", "H", "H", "H"]
    rel = [ring["C1"], ring["O2"], ring["C3"], ring["C4"], ring["O5"],
           ring["O6"], h3[0], h3[1], h4[0], h4[1]]
    return symbols, rel


_register_mol("EC", *_ec_template(), charge=0)
_register_mol("PF6-", ["P"] + ["F"] * 6,
              [[0.0, 0.0, 0.0], [1.58, 0.0, 0.0], [-1.58, 0.0, 0.0],
               [0.0, 1.58, 0.0], [0.0, -1.58, 0.0], [0.0, 0.0, 1.58],
               [0.0, 0.0, -1.58]], charge=-1)

LIPF6_CASE = {"id": "lipf6_ec", "molecules": {"EC": 40, "Li+": 4, "PF6-": 4},
              "box": (17.0, 17.0, 17.0), "margin": 0.05, "seed": 181}

# dense supercritical CO2 (O5, Review 6; regenerated 2026-10-02): 60 CO2 at
# 0.68 g/cm3, a liquid-like supercritical state point on the Span-Wagner EOS
# (about 315 K and 11 MPa; the critical point is 304.1 K / 7.38 MPa /
# 0.4676 g/cm3, dry ice 1.56 g/cm3). The pre-O5 frames packed 2.54 g/cm3 into
# a 12 A box (molecule centres 1.96 A apart) while the ground truth claimed
# ~1.0 g/cm3: past any equilibrium fluid density, and the lifted program
# could not be rebuilt ("cannot place 60 CO2 at this density") because the
# compiler's pack_molecules RSA-packs whole molecular spheres (CO2's
# census-safe radius is 1.985 A), which jams near 0.70 g/cm3 -- measured on
# 20 seeds: 20/20 pack at 0.68 g/cm3, 16/20 at 0.70, 0/20 at 0.80. The box
# edge is derived from the target density and rounded UP to 0.01 A so the
# achieved density never exceeds the target.
CO2_DENSITY_G_CM3 = 0.68   # dialect-exempt: published-EOS state point, not a dialect threshold
CO2_BOX_EDGE = float(np.ceil((60 * molecular_mass("CO2") * _AMU_PER_A3_TO_G_CM3
                              / CO2_DENSITY_G_CM3) ** (1.0 / 3.0) * 100.0)) / 100.0

CO2_CASE = {"id": "co2_dense", "molecules": {"CO2": 60},
            "box": (CO2_BOX_EDGE,) * 3,
            "density_target_g_cm3": CO2_DENSITY_G_CM3,
            "margin": 0.05, "seed": 191, "pack_then_relax": True}

# diamond Si(001) slab (ASE path of the surface builder) doubled along x so
# the surface cell holds whole p(2x1) cells, with the missing-row
# reconstruction planted independently of the builder's Wood machinery.
# Thermal amplitude 0.05 A: the Wood net recognition reads two primitive
# vectors per layer with no ensemble averaging (wood_commensurate_tol is 0.15
# on the superstructure matrix), so the crystal-case amplitude
# (0.06 x 2.35 A ~ 0.14 A) would drown the commensurability check.
SI001_PROGRAM = """chaord 0.1
dialect core + metal + surface

system {{
  cell {lx:.3f} {ly:.3f} {lz:.3f}
  pbc xyz
  conserve atoms Si {n}
}}

physics {{
  backend eam
}}

crystal slab : slab z 0 .. {ztop:.1f} {{
  lattice diamond
  a 5.43 A
  surface (001) top
}}

vacuum gap : slab z {ztop:.1f} .. {lz:.1f} {{
}}
"""
SI001_CASE = {"id": "si001_2x1", "a": 5.43, "tile_x": 2, "n_slab": 36,
              # 3x3x4 ASE diamond(100) slab at a = 5.43 with a 12 A vacuum
              "lx": 11.519, "ly": 11.519, "lz": 16.073, "ztop": 4.1,
              "thermal_amplitude": 0.05, "seed": 211}

# Cu(001) block in the lower half of a doubled box, water RSA-packed into the
# upper half at 0.9 g/cm3 around the Cu atoms as excluded obstacles
CUWATER_CASE = {"id": "cu_water", "a": 3.615, "reps": (4, 4, 6),
                "water_density_g_cm3": 0.9, "margin": 0.05, "seed": 197,
                "segmentation_target": 0.9}

GLASS_PROGRAM = """chaord 0.1
dialect core + glass

system {{
  pbc xyz
  conserve atoms X {n}
}}

physics {{
  backend lj
}}

amorphous glass : all {{
  state density {rho}
}}
"""


# ------------------------------------------------------------------- helpers --

def _kv_defect(token: str, count: int) -> Statement:
    return Statement(kind="build", key="defect",
                     values=[KVDefect(text=token), Name(text="count"),
                             Quantity(num=str(count))])


def _species_counts(symbols) -> dict:
    counts = Counter(symbols)
    return {s: int(counts[s]) for s in sorted(counts)}


def _formula_composition(formula: str) -> frozenset:
    """Element multiset of a canonical census formula, e.g. 'H2O' -> {H:2, O:1}."""
    c = Counter()
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        if el:
            c[el] += int(n) if n else 1
    return frozenset(c.items())


def _census_to_names(census: dict) -> dict:
    """Translate canonical census formulas back to template molecule names.

    molecule_census labels a component by its canonical formula (C first, H
    second, then alphabetical), so a hydroxide is reported as 'HO'. Ground
    truth stores both views; this maps the census back onto the molecule names
    of the packing spec so the generator can verify the frame is exact."""
    by_composition = {}
    for name in TEMPLATES:
        comp = frozenset(Counter(TEMPLATES[name]["symbols"]).items())
        by_composition.setdefault(comp, name)
    out = {}
    for formula, n in census.items():
        name = by_composition.get(_formula_composition(formula))
        if name is None:
            raise ValueError(f"census component {formula!r} matches no template")
        out[name] = out.get(name, 0) + int(n)
    return dict(sorted(out.items()))


def _sanitize(frame: Frame) -> Frame:
    """Wrap into [0, L) with a strict upper edge (all bench cells are diagonal)."""
    if float(np.abs(frame.cell - np.diag(np.diag(frame.cell))).max()) > 1e-9:
        raise ValueError("bench storage requires a diagonal cell")
    L = frame.cell_diag
    pos = np.mod(np.asarray(frame.pos, float), L)
    pos = np.minimum(pos, L * (1 - 1e-9))  # strict upper edge for KD trees
    return Frame(pos=pos, cell=frame.cell, symbols=frame.symbols, pbc=frame.pbc)


def _density_g_cm3(counts: dict, volume: float) -> float:
    """Mass density of a molecule-count spec in a given volume."""
    return sum(n * molecular_mass(name) for name, n in counts.items()) \
        * _AMU_PER_A3_TO_G_CM3 / float(volume)


def _contact_floor_worst_margin(frame: Frame, dialect) -> tuple[float, str]:
    """(worst d - floor over constrained intermolecular pairs, pair name).

    The W8 floor's one definition is chaord.build.molecules.
    contact_floor_target (0.75 x the Bondi sum on heavy-atom pairs,
    H...O/N >= 1.5 A, ion solvation pairs exempt); intermolecular = different
    bond-graph components.  Positive = clearance of the closest pair (the
    recorded per-frame provenance); negative = violation."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    edges = bond_graph(frame, dialect)
    n = len(frame)
    if edges:
        rows = [e[0] for e in edges] + [e[1] for e in edges]
        cols = [e[1] for e in edges] + [e[0] for e in edges]
        _, labels = connected_components(
            coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)),
            directed=False)
    else:
        labels = np.arange(n)
    L = frame.cell_diag
    pos = np.mod(frame.pos, L)
    pos = np.minimum(pos, L * (1 - 1e-9))
    syms = frame.symbols
    present = sorted(set(syms))
    rmax = max(contact_floor_target(a, b, dialect) or 0.0
               for a in present for b in present) + 1e-6
    tree = cKDTree(pos, boxsize=L)
    pairs = tree.query_pairs(rmax, output_type="ndarray")
    worst, pair = np.inf, "no constrained intermolecular pair"
    for i, j in pairs:
        if labels[i] == labels[j]:
            continue
        target = contact_floor_target(syms[i], syms[j], dialect)
        if target is None:
            continue
        d = pos[j] - pos[i]
        d -= L * np.round(d / L)
        margin = float(np.linalg.norm(d)) - target
        if margin < worst:
            worst, pair = margin, f"{syms[i]}-{syms[j]}"
    return worst, pair


def _pack_then_relax(frame: Frame, dialect) -> tuple[Frame, float]:
    """W8 step 3: open sub-floor intermolecular contacts by rigid-body
    relaxation (chaord.build.molecules.relax_intermolecular_contacts) and
    verify the result against the same contact floor (fail loud at generation
    time, never ship a sub-floor frame).  Returns (frame, worst margin A)."""
    out = relax_intermolecular_contacts(frame, dialect)
    worst, pair = _contact_floor_worst_margin(out, dialect)
    if worst < -1e-9:
        raise RuntimeError(
            f"pack-then-relax left an intermolecular {pair} pair "
            f"{abs(worst):.3f} A below the W8 contact floor")
    return out, worst


def _dense_pack(counts: dict, box, rng, dialect, margin=0.05,
                obstacles: Frame | None = None, z_range=None) -> Frame:
    """Random-sequential packing of rigid molecules at liquid density.

    pack_molecules excludes whole molecular bonding spheres, which jams far
    below liquid density for multi-atom solvents (EC, CO2 at supercritical
    density). This variant inserts whole rigid molecules like pack_molecules
    (random centre + rotation) but rejects on ATOM-ATOM distances: a placement
    stands only if every new atom stays `margin` beyond the bond-graph
    threshold (bond_tolerance x covalent radii sum) from every existing atom,
    so the census of the packed frame is exact by construction. `obstacles`
    (e.g. a metal slab) and `z_range` (lo, hi for every atom) restrict the
    packed region; obstacle interactions use the minimum image."""
    from ase.data import chemical_symbols, covalent_radii
    from scipy.spatial import cKDTree

    tol = float(dialect.threshold("bond_tolerance"))
    L = np.asarray(box, float)
    radii: dict[str, float] = {}

    def r_of(s: str) -> float:
        if s not in radii:
            radii[s] = tol * covalent_radii[chemical_symbols.index(s)]
        return radii[s]

    obst = (np.mod(obstacles.pos, L) if obstacles is not None
            else np.zeros((0, 3)))
    obst = np.minimum(obst, L * (1 - 1e-9))
    obst_rad = np.array([r_of(s) for s in obstacles.symbols]
                        if obstacles is not None else [])
    tree = cKDTree(obst, boxsize=L) if len(obst) else None

    total = sum(counts.values())
    budget = int(dialect.threshold("packing_max_tries")) * max(total, 1) * 40
    chunks: list[np.ndarray] = []
    chunk_syms: list[list[str]] = []
    placed_atoms: list[str] = []
    tries = 0
    for name, n in counts.items():
        t = TEMPLATES[name]
        placed = 0
        while placed < n:
            tries += 1
            if tries > budget:
                raise RuntimeError(
                    f"dense packing of {total} molecules stuck at "
                    f"{sum(len(c) for c in chunk_syms)} atoms")
            centre = rng.uniform(0, L)
            if z_range is not None:
                centre[2] = rng.uniform(z_range[0], z_range[1])
            R = random_rotation(rng)
            atoms = np.mod(t["rel"] @ R.T + centre, L)
            if z_range is not None and not (
                    (atoms[:, 2] > z_range[0])
                    & (atoms[:, 2] < z_range[1])).all():
                continue
            new_rad = np.array([r_of(s) for s in t["symbols"]])
            ok = True
            if tree is not None:
                for ia in range(len(atoms)):
                    if not ok:
                        break
                    for j in tree.query_ball_point(atoms[ia], 6.5):
                        d = atoms[ia] - obst[j]
                        d -= L * np.round(d / L)
                        if np.linalg.norm(d) < new_rad[ia] + obst_rad[j] + margin:
                            ok = False
                            break
            if ok and chunks:
                old = np.vstack(chunks)
                old_rad = np.array([r_of(s) for s in placed_atoms])
                d = atoms[:, None, :] - old[None, :, :]
                d -= L * np.round(d / L)
                if (np.linalg.norm(d, axis=2)
                        < (new_rad[:, None] + old_rad[None, :] + margin)).any():
                    ok = False
            if not ok:
                continue
            chunks.append(atoms)
            chunk_syms.append(list(t["symbols"]))
            placed_atoms.extend(t["symbols"])
            placed += 1
    pos = np.vstack(chunks)
    return Frame(pos=pos, cell=np.diag(L),
                 symbols=list(placed_atoms), pbc=(True, True, True))


def _thermally_displace(frame: Frame, amp: float, rng) -> Frame:
    """Gaussian displacement of the exact sites, wrapped through the cell."""
    noise = rng.normal(size=frame.pos.shape) * amp
    frac = (frame.pos + noise) @ np.linalg.inv(frame.cell)
    frac -= np.floor(frac)
    return Frame(pos=frac @ frame.cell, cell=frame.cell, symbols=frame.symbols,
                 pbc=frame.pbc)


def _composition_string(name: str, slots: tuple) -> str | None:
    if len(SLOT_COUNTS[name]) < 2:
        return None
    want = SLOT_COUNTS[name]
    g = 0
    for n in want:
        g = gcd(g, n)
    return "".join(f"{el}{n // g if n // g > 1 else ''}"
                   for el, n in zip(slots, want))


def _write_case(out: Path, category: str, case_id: str, frames, ground) -> dict:
    case_dir = out / category / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    rel_frames = []
    for k, frame in enumerate(frames):
        npz = case_dir / f"frame_{k}.npz"
        write_frame(npz, frame)
        # storage sanity: the benchmark snapshot format is r (N,3), L (3,), U8 symbols
        with np.load(npz) as z:
            assert set(z.files) == {"r", "L", "symbols"}, f"{npz}: keys {z.files}"
            assert z["r"].shape == frame.pos.shape
            assert z["symbols"].dtype == np.dtype("U8")
            assert z["L"].shape == (3,)
        rel_frames.append(npz.relative_to(out).as_posix())
    gt_path = case_dir / "ground_truth.json"
    gt_path.write_text(json.dumps(ground, indent=2, sort_keys=False) + "\n",
                       encoding="utf-8")
    return {"id": f"{category}/{case_id}", "category": category,
            "frames": rel_frames, "ground_truth": gt_path.relative_to(out).as_posix()}


def _gt_header(case_id: str, category: str, description: str, seed,
               dialect: str = "core+metal") -> dict:
    return {"schema": SCHEMA, "case": case_id, "category": category,
            "description": description, "seed": seed, "n_frames": N_FRAMES,
            "dialect": dialect, "frames": []}


def _frame_record(k: int, seed: int, frame: Frame, **extra) -> dict:
    rec = {"file": f"frame_{k}.npz", "seed": seed,
           "n_atoms": len(frame), "counts": _species_counts(frame.symbols)}
    rec.update(extra)
    return rec


# -------------------------------------------------------------------- cases --

def generate_crystals(out: Path, metal) -> list:
    entries = []
    amp_frac = float(metal.threshold("thermal_test_amplitude"))
    specs = list(CRYSTAL_CASES)
    # the solid solutions are built from the same exact-crystal machinery
    for sol in SOLUTION_SPECS:
        specs.append((sol["id"], sol["prototype"], sol["params"],
                      sol["host_species"], sol["reps"], None, sol["seed"]))
    for case_id, name, params, slots, reps, orient, seed in specs:
        host = _sanitize(build_conventional(name, params, slots, reps,
                                            orient_rows=orient))
        amp = amp_frac * nearest_neighbor_distance(host)
        solution_spec = next((s for s in SOLUTION_SPECS if s["id"] == case_id),
                             None)
        solution = solution_spec is not None
        category = solution_spec["category"] if solution else "crystals"
        occ_stmt = None
        if solution:
            occ_stmt = Statement(
                kind="build", key="occupancy",
                values=[v for pair in solution_spec["occupancy"]
                        for v in (Name(text=pair[0]), Quantity(num=pair[1]))])
        frames = []
        for k in range(N_FRAMES):
            rng = np.random.default_rng(seed + k)
            frame = host
            if solution:
                frame = assign_occupancy(host, [occ_stmt], rng, metal)
            frames.append(_sanitize(_thermally_displace(frame, amp, rng)))
        composition = _composition_string(name, slots)
        setting = ("orthohexagonal supercell (orient x [100], y [1 2 0], z [001]); "
                   "reps count the rectangular cell") if orient is not None else ""
        ground = _gt_header(
            case_id, category,
            f"{name} {composition or slots[0]}, {reps[0]}x{reps[1]}x{reps[2]} "
            f"supercell built exactly from the prototype table"
            + (f", {solution_spec['occ_text']}" if solution else "")
            + (f"; {setting}" if setting else ""), seed)
        ground["lift_mode"] = "crystal"
        ground["lift_dialect"] = ["core", "metal"]
        ground["thermal_amplitude_fraction"] = amp_frac
        ground["expected"] = {
            "prototype": name,
            "a": params["a"],
            "composition": composition,
            "reps": list(reps),
            "cell": [float(v) for v in frames[0].cell_diag],
            "counts": _species_counts(frames[0].symbols),
            "n_atoms": len(frames[0]),
        }
        if "c" in params:
            ground["expected"]["c"] = params["c"]
        if solution:
            ground["expected"]["occupancy"] = {s: f for s, f in
                                               solution_spec["occupancy"]}
        for k, frame in enumerate(frames):
            ground["frames"].append(_frame_record(k, seed + k, frame))
        entries.append(_write_case(out, category, case_id, frames, ground))
        print(f"[{category:<8}] {case_id}: {name} a={params['a']} "
              f"{N_FRAMES} frames x {len(frames[0])} atoms "
              f"counts={ground['expected']['counts']}")
    return entries


def generate_defects(out: Path, metal) -> list:
    entries = []
    for spec in DEFECT_CASES:
        host = _sanitize(build_conventional(spec["prototype"], spec["params"],
                                            spec["species"], spec["reps"]))
        stmts = [_kv_defect(tok, n) for tok, n in spec["defects"]]
        frames = []
        for k in range(N_FRAMES):
            rng = np.random.default_rng(spec["seed"] + k)
            frames.append(_sanitize(apply_defects(host, stmts, rng, metal)))
        defect_txt = ", ".join(f"{n} {tok}" for tok, n in spec["defects"])
        ground = _gt_header(
            spec["id"], "defects",
            f"{spec['prototype']} {spec['species']} {spec['reps'][0]}x"
            f"{spec['reps'][1]}x{spec['reps'][2]} host with random {defect_txt} "
            f"(Kröger-Vink); defect sites are re-drawn every frame, counts fixed",
            spec["seed"])
        ground["lift_mode"] = "defects"
        ground["lift_dialect"] = ["core", "metal"]
        ground["host"] = {"prototype": spec["prototype"], "a": spec["params"]["a"],
                          "reps": list(spec["reps"]),
                          "counts": _species_counts(host.symbols)}
        ground["expected"] = {
            "defects": {tok: n for tok, n in spec["defects"]},
            "counts": _species_counts(frames[0].symbols),
            "n_atoms": len(frames[0]),
        }
        for k, frame in enumerate(frames):
            ground["frames"].append(
                _frame_record(k, spec["seed"] + k, frame,
                              defects=dict(ground["expected"]["defects"])))
        entries.append(_write_case(out, "defects", spec["id"], frames, ground))
        print(f"[defects ] {spec['id']}: {dict(spec['defects'])} "
              f"{N_FRAMES} frames x {len(frames[0])} atoms "
              f"counts={ground['expected']['counts']}")
    return entries


def _packed_frames(spec, dialect, category, description, out):
    frames = []
    census_list = []
    margins = []
    for k in range(N_FRAMES):
        rng = np.random.default_rng(spec["seed"] + k)
        frame = pack_molecules(spec["molecules"], spec["box"], rng, dialect)
        census = dict(sorted(molecule_census(frame, dialect).items()))
        planted = _census_to_names(census)
        if planted != spec["molecules"]:
            raise RuntimeError(f"census {census} does not match the packing spec "
                               f"{spec['molecules']}")
        margin = None
        if spec.get("pack_then_relax"):
            # W8 step 3: pack-then-relax opens the RSA packing's sub-floor
            # intermolecular contacts; the census is re-verified after the
            # rigid-body moves (a fused or split molecule is a generation
            # error, never a shipped frame)
            frame, margin = _pack_then_relax(frame, dialect)
            census = dict(sorted(molecule_census(frame, dialect).items()))
            if _census_to_names(census) != spec["molecules"]:
                raise RuntimeError(
                    f"census {census} does not survive pack-then-relax "
                    f"(spec {spec['molecules']})")
        frames.append(_sanitize(frame))
        census_list.append(census)
        margins.append(margin)
    ground = _gt_header(spec["id"], category, description, spec["seed"],
                         dialect="core+molecular")
    ground["lift_mode"] = "fluid" if category == "fluid" else "reactive"
    ground["lift_dialect"] = ["core", "molecular"]
    ground["molecules"] = spec["molecules"]
    ground["box"] = list(spec["box"])
    if spec.get("pack_then_relax"):
        # only the regenerated cases carry the flag: ar_gas_box25 and
        # n2_box22 keep their frozen ground truths byte-identical
        ground["pack_then_relax"] = True
    ground["expected"] = {
        "molecules": spec["molecules"],
        "census": census_list[0],
        "counts": _species_counts(frames[0].symbols),
        "n_atoms": len(frames[0]),
    }
    for k, (frame, census, margin) in enumerate(
            zip(frames, census_list, margins)):
        rec = _frame_record(k, spec["seed"] + k, frame, census=census)
        if margin is not None:
            rec["contact_floor_worst_margin_A"] = round(margin, 4)
        ground["frames"].append(rec)
    entry = _write_case(out, category, spec["id"], frames, ground)
    print(f"[{category:<8}] {spec['id']}: census={census_list[0]} "
          f"{N_FRAMES} frames x {len(frames[0])} atoms")
    return [entry]


def generate_fluids(out: Path, molecular) -> list:
    entries = []
    for spec in FLUID_CASES:
        main = next(iter(spec["molecules"]))
        relax_note = ("; W8 pack-then-relax: the RSA packing's sub-floor "
                      "intermolecular contacts opened by rigid-body "
                      "relaxation until every pair clears the 0.75x-Bondi "
                      "contact floor (per-frame worst margin recorded)"
                      if spec.get("pack_then_relax") else "")
        entries += _packed_frames(
            spec, molecular, "fluid",
            f"{spec['molecules'][main]} {main} packed into a "
            f"{spec['box'][0]:.0f} A cubic box (seeded random-sequential packing, "
            f"re-packed per frame){relax_note}", out)
    return entries


def generate_reactive(out: Path, molecular) -> list:
    spec = REACTIVE_CASE
    return _packed_frames(
        spec, molecular, "reactive",
        "dissociation fragment mixture (water, hydroxide, atomic hydrogen) in "
        "a 20 A box; the species census must be recovered exactly from the bond "
        "graph of every frame; W8 pack-then-relax: the RSA packing's sub-floor "
        "intermolecular contacts opened by rigid-body relaxation until every "
        "pair clears the 0.75x-Bondi contact floor (per-frame worst margin "
        "recorded)", out)


def generate_glass(out: Path) -> list:
    spec = GLASS_CASE
    glass = load_dialect(("core", "glass"))
    program = parse_text(GLASS_PROGRAM.format(n=spec["n"], rho=spec["rho"]))
    frames = []
    for k in range(N_FRAMES):
        rng = np.random.default_rng(spec["seed"] + k)
        frame = build_program(program, glass, rng=rng, physics=False)
        frames.append(_sanitize(frame))
    ground = _gt_header(
        spec["id"], "glass",
        f"LJ amorphous packing: {spec['n']} X atoms RSA-placed at number "
        f"density {spec['rho']} /sigma^3 (physics off, no protocol)", spec["seed"],
        dialect="core+glass")
    ground["lift_mode"] = "amorphous"
    ground["lift_dialect"] = ["core", "glass"]
    ground["expected"] = {
        "density": spec["rho"],
        "species": spec["species"],
        "counts": _species_counts(frames[0].symbols),
        "n_atoms": len(frames[0]),
        "cell": [float(v) for v in frames[0].cell_diag],
    }
    for k, frame in enumerate(frames):
        rho = len(frame) / float(np.prod(frame.cell_diag))
        ground["frames"].append(_frame_record(k, spec["seed"] + k, frame,
                                              density=float(rho)))
    entry = _write_case(out, "glass", spec["id"], frames, ground)
    print(f"[glass   ] {spec['id']}: rho={spec['rho']} {N_FRAMES} frames x "
          f"{len(frames[0])} atoms")
    return [entry]


def _rsa_pack_liquid(z_lo: float, z_hi: float, n: int, frozen: np.ndarray,
                     L: np.ndarray, dmin: float, rng, max_tries: int):
    """Random-sequential placement of n liquid atoms between z_lo and z_hi at
    a dmin hard core from every other liquid atom and from the frozen solid
    (minimum image; the solid walls exclude centres across the z boundary
    too, so the packed slab never starts overlapped)."""
    from scipy.spatial import cKDTree

    hi = L * (1.0 - 1e-9)                 # strict upper edge for KD trees
    frozen_tree = cKDTree(np.minimum(np.mod(frozen, L), hi), boxsize=L)
    placed = np.zeros((0, 3))
    tree = None
    for attempt in range(1, max_tries + 1):
        if len(placed) == n:
            break
        cand = np.minimum(rng.uniform(0.0, L), hi)
        cand[2] = min(rng.uniform(z_lo, z_hi), hi[2])
        if tree is not None and float(tree.query(cand[None, :])[0][0]) < dmin:
            continue
        if float(frozen_tree.query(cand[None, :])[0][0]) < dmin:
            continue
        placed = np.vstack([placed, cand[None, :]])
        tree = cKDTree(placed, boxsize=L)
    else:
        raise RuntimeError(f"liquid RSA placed {len(placed)} of {n} atoms in "
                           f"{max_tries} tries at dmin = {dmin} sigma")
    return placed


def _lj_relax_liquid(liquid: np.ndarray, frozen: np.ndarray, L: np.ndarray,
                     rng) -> tuple[np.ndarray, dict]:
    """Short Lennard-Jones MD relaxation of the liquid atoms at T* = 0.7 with
    the solid half frozen, on the same engine stack the reference generator
    uses (ASE LennardJones + Langevin + FixAtoms; reduced units
    epsilon = sigma = mass = 1, temperatures passed as T*/k_B with
    epsilon = 1 eV). Returns (relaxed positions, diagnostics)."""
    import ase
    from ase import Atoms
    from ase.calculators.lj import LennardJones
    from ase.constraints import FixAtoms
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import Stationary, thermalize_momenta
    from ase.units import kB

    n_liq, n_frozen = len(liquid), len(frozen)
    atoms = Atoms(symbols=["X"] * (n_liq + n_frozen),
                  positions=np.vstack([liquid, frozen]),
                  cell=np.diag(L), pbc=True)
    atoms.set_masses(np.full(len(atoms), 1.0))
    frozen_mask = np.zeros(len(atoms), bool)
    frozen_mask[n_liq:] = True
    atoms.calc = LennardJones(epsilon=1.0, sigma=1.0, rc=2.5, smooth=False)
    thermalize_momenta(atoms, temperature_K=INTERFACE_MD["T_star"] / kB,
                       rng=rng)
    Stationary(atoms)
    atoms.set_constraint(FixAtoms(mask=frozen_mask))
    t_k = INTERFACE_MD["T_star"] / kB
    for dt, steps in ((INTERFACE_MD["dt_fast"], INTERFACE_MD["steps_fast"]),
                      (INTERFACE_MD["dt"], INTERFACE_MD["steps"])):
        Langevin(atoms, dt, temperature_K=t_k,
                 friction=INTERFACE_MD["friction_per_tau"], rng=rng,
                 fixcm=False).run(steps)
    pos = atoms.positions[:n_liq].copy()
    diag = {
        "T_measured_star": float(atoms.get_temperature() * kB),
        "pe_per_atom": float(atoms.get_potential_energy()
                             / (n_liq + n_frozen)),
        "engine": f"ASE {ase.__version__} LennardJones + Langevin",
    }
    return pos, diag


def generate_interface(out: Path) -> list:
    """LJ solid-liquid slab: perfect fcc at rho_solid below, a physically
    regenerated liquid at rho_liquid above (W6: RSA at a 0.85 sigma hard
    core, then short LJ MD at T* = 0.7 with the solid frozen; engine
    parameters in the ground truth). The liquid region volume gives the
    same atom count at the LJ coexistence liquid density, so the z cell is
    asymmetric and the label boundary stays the box mid-plane. The min pair
    of every frame is asserted against the lj dialect's overlap_tolerance
    before the frame is written."""
    from scipy.spatial import cKDTree

    spec = INTERFACE_CASE
    lj = load_dialect(("core", "lj"))
    floor = float(lj.threshold("overlap_tolerance"))          # sigma
    a = float((4.0 / spec["rho_solid"]) ** (1.0 / 3.0))  # fcc cube edge at rho_solid
    solid_full = build_conventional("fcc", {"a": a}, (spec["species"],),
                                    spec["reps"])
    Lx, Ly = float(solid_full.cell_diag[0]), float(solid_full.cell_diag[1])
    z_solid = (spec["reps"][2] // 2) * a          # the perfect-crystal region
    solid = np.asarray(solid_full.pos[solid_full.pos[:, 2] < z_solid], float)
    n_liq = len(solid_full) - len(solid)          # equal counts by construction
    Lz = z_solid + n_liq / (spec["rho_liquid"] * Lx * Ly)
    L = np.array([Lx, Ly, Lz])
    boundary = Lz / 2.0
    frames = []
    stats = []
    for k in range(N_FRAMES):
        rng = np.random.default_rng(spec["seed"] + k)
        liquid0 = _rsa_pack_liquid(z_solid, Lz, n_liq, solid, L,
                                   INTERFACE_RSA_DMIN, rng,
                                   INTERFACE_RSA_MAX_TRIES)
        liquid, diag = _lj_relax_liquid(liquid0, solid, L, rng)
        frame = _sanitize(Frame(pos=np.vstack([solid, liquid]),
                                cell=np.diag(L),
                                symbols=[spec["species"]] * len(solid_full)))
        p = np.mod(frame.pos, L)
        p = np.minimum(p, L * (1 - 1e-9))
        min_pair = float(cKDTree(p, boxsize=L).query(p, k=2)[0][:, 1].min())
        if min_pair < floor:
            raise RuntimeError(f"{spec['id']} frame {k}: min pair "
                               f"{min_pair:.3f} sigma < {floor} sigma hard "
                               "core after MD relaxation")
        frames.append(frame)
        stats.append(dict(min_pair_sigma=min_pair, **diag))
    ground = _gt_header(
        spec["id"], "interface",
        f"LJ solid-liquid slab at coexistence densities: perfect fcc "
        f"({spec['reps'][0]}x{spec['reps'][1]}x{spec['reps'][2] // 2} cells, "
        f"rho_solid = {spec['rho_solid']}/sigma^3) below z = {z_solid:.4f}; "
        f"{n_liq} liquid atoms at rho_liquid = {spec['rho_liquid']}/sigma^3 "
        f"above (the z cell is asymmetric so the liquid carries its "
        f"coexistence density; at rho = rho_solid a T* = "
        f"{INTERFACE_MD['T_star']} relaxation freezes epitaxially onto the "
        f"perfect wall). The liquid half is RSA-placed at a "
        f"{INTERFACE_RSA_DMIN} sigma hard core and relaxed by short "
        f"Lennard-Jones MD at T* = {INTERFACE_MD['T_star']} with the solid "
        f"atoms frozen (engine parameters under 'md'; per-frame min pair "
        f"distance, measured T* and potential energy in the frame records); "
        f"the two-phase label boundary is the mid-plane of the box",
        spec["seed"], dialect="core+lj")
    ground["lift_mode"] = "interface"
    ground["lift_dialect"] = ["core", "lj"]
    ground["md"] = {
        "engine": {"name": "ASE", "integrator": "ase.md.langevin.Langevin",
                   "constraint": "ase.constraints.FixAtoms (solid half frozen)",
                   "initial_velocities": "thermalize_momenta + Stationary"},
        "potential": {
            "name": "Lennard-Jones 12-6",
            "implementation": "ase.calculators.lj.LennardJones",
            "citation": ("Lennard-Jones potential as implemented in ASE; "
                         "original: J. E. Jones, Proc. R. Soc. Lond. A 106, "
                         "463 (1924)"),
            "parameters": {"epsilon_eV": 1.0, "sigma_A": 1.0,
                           "rc_sigma": 2.5, "smooth": False},
        },
        "protocol": {
            "description": (
                "per frame (independent RSA + MD draw): random-sequential "
                "placement of the liquid half at a 0.85 sigma hard core "
                "(liquid-liquid and liquid-solid, minimum image), then "
                "Langevin NVT at T* = 0.7 with the solid half frozen; "
                "dt* = 0.002 for 1000 steps then dt* = 0.005 for 2000 steps "
                "(12 tau total), friction 0.5/tau. Duration measured "
                "time-resolved: liquid structure forms by ~7 tau, the first "
                "transient solid-like blips in the liquid core appear at "
                "~17 tau"),
            "rsa_dmin_sigma": INTERFACE_RSA_DMIN,
            "T_star": INTERFACE_MD["T_star"],
            "dt_fast": INTERFACE_MD["dt_fast"],
            "steps_fast": INTERFACE_MD["steps_fast"],
            "dt": INTERFACE_MD["dt"],
            "steps": INTERFACE_MD["steps"],
            "time_tau": INTERFACE_MD["time_tau"],
            "friction_per_tau": INTERFACE_MD["friction_per_tau"],
        },
    }
    ground["expected"] = {
        "axis": "z",
        "boundary_z": float(boundary),
        "phases": {"solid": "below", "liquid": "above"},
        "rho_solid": spec["rho_solid"],
        "rho_liquid": spec["rho_liquid"],
        "z_solid": float(z_solid),
        "counts": _species_counts(frames[0].symbols),
        "n_atoms": len(frames[0]),
        "cell": [float(v) for v in frames[0].cell_diag],
    }
    for k, frame in enumerate(frames):
        ground["frames"].append(
            _frame_record(k, spec["seed"] + k, frame,
                          boundary_z=float(boundary),
                          min_pair_sigma=round(stats[k]["min_pair_sigma"], 4),
                          liquid_T_measured_star=round(
                              stats[k]["T_measured_star"], 4),
                          liquid_pe_per_atom_epsilon=round(
                              stats[k]["pe_per_atom"], 4)))
    entry = _write_case(out, "interface", spec["id"], frames, ground)
    print(f"[interface] {spec['id']}: boundary z={boundary:.3f} "
          f"min pair {min(s['min_pair_sigma'] for s in stats):.3f} sigma "
          f"{N_FRAMES} frames x {len(frames[0])} atoms")
    return [entry]


def _ortho_double(frame: Frame) -> Frame:
    """Rectangular (index-2) supercell of a hexagonal-xy cell.

    Rows x = a1, y = -a1 + 2*a2 make a diagonal cell for the ASE hexagonal
    slab setting (a1 along x, a2 at 60 degrees); atoms are tiled with the
    old lattice translations and re-sorted canonically."""
    T = np.array([[1, 0, 0], [-1, 2, 0], [0, 0, 1]], int)
    new_cell = T @ frame.cell
    if float(np.abs(new_cell - np.diag(np.diag(new_cell))).max()) > 1e-9:
        raise ValueError("ortho doubling did not diagonalise the cell")
    inv = np.linalg.inv(new_cell)
    a1, a2 = frame.cell[0], frame.cell[1]
    pos_list, sym_list = [], []
    for k1 in range(-3, 4):
        for k2 in range(-3, 4):
            cand = frame.pos + k1 * a1 + k2 * a2
            frac = cand @ inv
            keep = np.all((frac > -1e-9) & (frac < 1 - 1e-9), axis=1)
            pos_list.append(cand[keep])
            sym_list.extend(frame.symbols[i] for i in np.where(keep)[0])
    pos = np.vstack(pos_list)
    if len(pos) != 2 * len(frame):
        raise RuntimeError("ortho doubling lost or duplicated atoms")
    order = np.lexsort((pos[:, 2], pos[:, 1], pos[:, 0]))
    return Frame(pos=pos[order], cell=new_cell,
                 symbols=[sym_list[i] for i in order], pbc=frame.pbc)


def generate_surface(out: Path) -> list:
    entries = []
    surface = load_dialect(("core", "metal", "surface"))
    for spec in SURFACE_CASES:
        text = SURF_PROGRAM.format(lx=spec["lx"], ly=spec["ly"], lz=spec["lz"],
                                   el=spec["element"], n=spec["n_slab"],
                                   a=spec["a"], hkl=spec["hkl"],
                                   ztop=spec["ztop"])
        program = parse_text(text)
        frames = []
        for k in range(N_FRAMES):
            rng = np.random.default_rng(spec["seed"] + k)
            slab = build_program(program, surface, rng=rng)
            frames.append(_sanitize(_ortho_double(slab)))
        counts = _species_counts(frames[0].symbols)
        ground = _gt_header(
            spec["id"], spec["category"],
            f"{spec['element']}({spec['hkl']}) slab with O adsorbates on top "
            f"sites; built by the surface builder, stored in its rectangular "
            f"supercell so the diagonal npz format round-trips exactly "
            f"(adsorbate sites re-drawn per frame)",
            spec["seed"], dialect="core+metal+surface")
        ground["lift_mode"] = "surface"
        ground["lift_dialect"] = ["core", "metal", "surface"]
        ground["expected"] = {
            "surface": f"({spec['hkl']})",
            "termination": spec["element"],
            "adsorb": {"species": "O",
                       "count": counts.get("O", 0),
                       "site": "top"},
            "counts": counts,
            "n_atoms": len(frames[0]),
            "cell": [float(v) for v in frames[0].cell_diag],
        }
        for k, frame in enumerate(frames):
            ground["frames"].append(
                _frame_record(k, spec["seed"] + k, frame,
                              adsorb_o_count=_species_counts(frame.symbols)["O"]))
        entries.append(_write_case(out, spec["category"], spec["id"], frames,
                                   ground))
        print(f"[{spec['category']:<8}] {spec['id']}: ({spec['hkl']}) "
              f"O on top x{counts['O']} {N_FRAMES} frames x {len(frames[0])} "
              f"atoms counts={counts}")
    return entries


# ------------------------------------------------------- new case builders --

def generate_nacl_aq(out: Path, molecular) -> list:
    """Aqueous NaCl via pack_molecules with a box-growth and census-retry ladder.

    The nominal 16.5 A box jams the molecular-sphere RSA packer, so the box
    grows by box_step until every frame packs; the actual box and the actual
    mass density are recorded. A draw whose Na-Na / Na-Cl contact lands inside
    the census bond window (covalent radii are generous for metals) would fuse
    ions into one component, so draws are re-seeded until the census matches
    the packing spec exactly."""
    spec = NACL_CASE
    target = dict(sorted(spec["molecules"].items()))
    box = spec["box0"]
    while box <= spec["box_max"]:
        try:
            frames, censuses, seeds, attempts = [], [], [], []
            for k in range(N_FRAMES):
                for attempt in range(spec["census_retries"]):
                    seed = spec["seed"] + k + 1000 * attempt
                    rng = np.random.default_rng(seed)
                    frame = pack_molecules(spec["molecules"], (box,) * 3,
                                           rng, molecular)
                    census = dict(sorted(molecule_census(frame, molecular).items()))
                    try:
                        clean = _census_to_names(census) == target
                    except ValueError:
                        # fused ion cluster (Na-Na / Na-Cl inside the generous
                        # covalent-radius bond window): reject the draw
                        clean = False
                    if clean:
                        frames.append(_sanitize(frame))
                        censuses.append(census)
                        seeds.append(seed)
                        attempts.append(attempt)
                        break
                else:
                    raise RuntimeError(
                        f"frame {k}: no census-clean ion draw in "
                        f"{spec['census_retries']} attempts at box {box}")
            break
        except (ChaordError, RuntimeError):
            box += spec["box_step"]
    else:
        raise RuntimeError("nacl_aq: no box size up to "
                           f"{spec['box_max']} A packed cleanly")
    density = _density_g_cm3(spec["molecules"], box ** 3)
    ground = _gt_header(
        spec["id"], "solutions",
        f"aqueous NaCl: 120 H2O + 8 Na+ + 8 Cl- random-sequential packed "
        f"(re-packed per frame); the packer jammed at the nominal 16.5 A box, "
        f"so the box grew to {box:.1f} A (actual density "
        f"{density:.3f} g/cm3 recorded); draws with ion pairs inside the "
        f"census bond window are rejected and re-drawn, keeping the census "
        f"exact", spec["seed"], dialect="core+molecular")
    ground["lift_mode"] = "fluid"
    ground["lift_dialect"] = ["core", "molecular"]
    ground["molecules"] = spec["molecules"]
    ground["box"] = [box] * 3
    ground["density_g_cm3"] = density
    ground["expected"] = {
        "molecules": spec["molecules"],
        "census": censuses[0],
        "counts": _species_counts(frames[0].symbols),
        "n_atoms": len(frames[0]),
        "cell": [box] * 3,
        "density_g_cm3": density,
    }
    for k, (frame, census, seed, attempt) in enumerate(
            zip(frames, censuses, seeds, attempts)):
        ground["frames"].append(_frame_record(k, seed, frame, census=census,
                                              census_retries=attempt))
    entry = _write_case(out, "solutions", spec["id"], frames, ground)
    print(f"[solution] {spec['id']}: box={box:.1f} rho={density:.3f} g/cm3 "
          f"census={censuses[0]} {N_FRAMES} frames x {len(frames[0])} atoms "
          f"(retries {attempts})")
    return [entry]


def _dense_solution_case(spec, category, description, out, molecular) -> list:
    """Shared builder for the atom-level RSA cases (electrolyte, dense CO2).

    "pack_then_relax": True (W8, review open_items_v2.md step 3) opens the
    packing's sub-floor intermolecular contacts by rigid-body relaxation
    after the RSA (co2_dense only -- lipf6_ec keeps its frozen bytes)."""
    frames, censuses, margins = [], [], []
    for k in range(N_FRAMES):
        rng = np.random.default_rng(spec["seed"] + k)
        frame = _dense_pack(spec["molecules"], spec["box"], rng, molecular,
                            margin=spec["margin"])
        census = dict(sorted(molecule_census(frame, molecular).items()))
        if _census_to_names(census) != spec["molecules"]:
            raise RuntimeError(f"{spec['id']} frame {k}: census {census} does "
                               f"not match the packing spec {spec['molecules']}")
        margin_rec = None
        if spec.get("pack_then_relax"):
            frame, margin_rec = _pack_then_relax(frame, molecular)
            census = dict(sorted(molecule_census(frame, molecular).items()))
            if _census_to_names(census) != spec["molecules"]:
                raise RuntimeError(
                    f"{spec['id']} frame {k}: census {census} does not survive "
                    f"pack-then-relax (spec {spec['molecules']})")
        frames.append(_sanitize(frame))
        censuses.append(census)
        margins.append(margin_rec)
    density = _density_g_cm3(spec["molecules"],
                             float(np.prod(np.asarray(spec["box"], float))))
    ground = _gt_header(spec["id"], category, description, spec["seed"],
                         dialect="core+molecular")
    ground["lift_mode"] = "fluid"
    ground["lift_dialect"] = ["core", "molecular"]
    ground["molecules"] = spec["molecules"]
    ground["box"] = list(spec["box"])
    ground["density_g_cm3"] = density
    if spec.get("pack_then_relax"):
        # only the regenerated case carries the flag: lipf6_ec keeps its
        # frozen ground truth byte-identical
        ground["pack_then_relax"] = True
    if "density_target_g_cm3" in spec:
        ground["density_target_g_cm3"] = spec["density_target_g_cm3"]
    ground["expected"] = {
        "molecules": spec["molecules"],
        "census": censuses[0],
        "counts": _species_counts(frames[0].symbols),
        "n_atoms": len(frames[0]),
        "cell": [float(v) for v in frames[0].cell_diag],
        "density_g_cm3": density,
    }
    for k, (frame, census, margin_rec) in enumerate(
            zip(frames, censuses, margins)):
        rec = _frame_record(k, spec["seed"] + k, frame, census=census)
        if margin_rec is not None:
            rec["contact_floor_worst_margin_A"] = round(margin_rec, 4)
        ground["frames"].append(rec)
    entry = _write_case(out, category, spec["id"], frames, ground)
    print(f"[{category:<8}] {spec['id']}: rho={density:.3f} g/cm3 "
          f"census={censuses[0]} {N_FRAMES} frames x {len(frames[0])} atoms")
    return [entry]


def generate_electrolyte(out: Path, molecular) -> list:
    spec = LIPF6_CASE
    return _dense_solution_case(
        spec, "solutions",
        "LiPF6/EC electrolyte surrogate: 40 EC + 4 Li+ + 4 PF6- in a 17 A box; "
        "EC (C3H4O3, simplified pentagon geometry, bonds 1.09-1.43 A) and "
        "PF6- (octahedral P-F 1.58 A, charge -1) templates are registered in "
        "this generator; packed by atom-level RSA (pack_molecules' molecular "
        "spheres jam above liquid density for a 10-atom solvent), so Li+ "
        "stays outside the EC/PF6 bond graphs and the census is exact",
        out, molecular)


def generate_co2(out: Path, molecular) -> list:
    spec = CO2_CASE
    return _dense_solution_case(
        spec, "gases",
        f"dense supercritical CO2: 60 molecules at {spec['density_target_g_cm3']:.2f} "
        f"g/cm3 (a liquid-like supercritical state point on the Span-Wagner "
        f"EOS, about 315 K and 11 MPa; box {spec['box'][0]:.2f} A derived from "
        f"the target density); packed by atom-level RSA with the census bond "
        f"threshold plus a 0.05 A margin as the exclusion, so molecules stay "
        f"distinct components of the bond graph; W8 pack-then-relax then "
        f"opens the packing's sub-floor intermolecular contacts by rigid-body "
        f"relaxation until every pair clears the 0.75x-Bondi contact floor "
        f"(per-frame worst margin recorded); the density also stays under "
        f"the ~0.70 g/cm3 ceiling of the compiler's molecular-sphere packer "
        f"(O5: the pre-2026-10 frames were 2.54 g/cm3 and never rebuilt)", out,
        molecular)


def _tile_along_x(frame: Frame, times: int) -> Frame:
    """In-plane tiling of a slab along x (a diagonal cell stays diagonal)."""
    pos = np.vstack([frame.pos + k * frame.cell[0] for k in range(times)])
    cell = frame.cell.copy()
    cell[0] = cell[0] * times
    return Frame(pos=pos, cell=cell, symbols=list(frame.symbols) * times,
                 pbc=frame.pbc)


def _delete_alternate_rows(frame: Frame, dialect) -> tuple[Frame, int]:
    """Plant a p(2x1) missing-row reconstruction: drop every other row of the
    top layer (independent of the builder's `_apply_reconstruction`: the top
    layer is taken by z-height and alternate sorted x-columns are deleted).

    Returns the reconstructed frame and the surviving top-layer atom count."""
    tol = float(dialect.threshold("layer_tolerance"))
    z = frame.pos[:, 2]
    top = np.where(z > z.max() - tol)[0]
    xs = np.sort(np.unique(np.round(frame.pos[top][:, 0], 3)))
    step = xs[1] - xs[0]
    drop = {int(i) for i in top
            if int(round((frame.pos[i, 0] - xs[0]) / step)) % 2 == 1}
    keep = [i for i in range(len(frame.pos)) if i not in drop]
    if 2 * (len(top) - len(drop)) != len(top):
        raise RuntimeError("missing-row deletion did not halve the top layer")
    recon = Frame(pos=frame.pos[keep], cell=frame.cell,
                  symbols=[frame.symbols[i] for i in keep])
    return recon, len(top) - len(drop)


def generate_si001(out: Path) -> list:
    spec = SI001_CASE
    surface = load_dialect(("core", "metal", "surface"))
    text = SI001_PROGRAM.format(lx=spec["lx"], ly=spec["ly"], lz=spec["lz"],
                                n=spec["n_slab"], ztop=spec["ztop"])
    program = parse_text(text)
    # the ASE path of the surface builder cuts the diamond(100) slab
    slab0 = build_program(program, surface, rng=np.random.default_rng(spec["seed"]))
    slab = _tile_along_x(slab0, spec["tile_x"])
    recon, n_top = _delete_alternate_rows(slab, surface)
    top_z = float(slab.pos[:, 2].max())
    d_layer = spec["a"] / 4.0        # diamond(100) interlayer spacing
    amp = spec["thermal_amplitude"]
    frames = []
    for k in range(N_FRAMES):
        rng = np.random.default_rng(spec["seed"] + k)
        frames.append(_sanitize(_thermally_displace(recon, amp, rng)))
    ground = _gt_header(
        spec["id"], "surfaces",
        f"diamond Si(001) slab (ASE diamond100 path, a = {spec['a']}, 3x3x4 "
        f"then tiled {spec['tile_x']}x along x so the surface cell holds "
        f"whole p(2x1) cells) with an independently planted p(2x1) "
        f"missing-row top layer (every other top-layer row deleted); small "
        f"thermal displacement per frame (the Wood net read compares two "
        f"primitive vectors per layer with no ensemble averaging)",
        spec["seed"], dialect="core+metal+surface")
    ground["lift_mode"] = "surface"
    ground["lift_dialect"] = ["core", "metal", "surface"]
    ground["thermal_amplitude"] = amp
    ground["expected"] = {
        "surface": "(001)",
        "termination": "Si",
        "reconstruction": "p(2x1)",
        "n_top": n_top,
        "top_z": top_z,
        "d_layer": d_layer,
        "prototype": "diamond",
        "a": spec["a"],
        "counts": _species_counts(frames[0].symbols),
        "n_atoms": len(frames[0]),
        "cell": [float(v) for v in frames[0].cell_diag],
    }
    for k, frame in enumerate(frames):
        ground["frames"].append(_frame_record(k, spec["seed"] + k, frame,
                                              n_top=n_top))
    entry = _write_case(out, "surfaces", spec["id"], frames, ground)
    print(f"[surfaces] {spec['id']}: top {n_top} atoms p(2x1) "
          f"{N_FRAMES} frames x {len(frames[0])} atoms")
    return [entry]


def generate_cu_water(out: Path, molecular, metal) -> list:
    """Cu block in the lower half, water in the upper half; the two-phase
    label accuracy of every frame is measured with the core+metal q6 rule
    (interface band excluded) and recorded in the ground truth."""
    spec = CUWATER_CASE
    cu = build_conventional("fcc", {"a": spec["a"]}, ("Cu",), spec["reps"])
    L = cu.cell_diag
    box = np.array([L[0], L[1], 2.0 * L[2]])
    z_half = float(L[2])
    inset = float(TEMPLATES["H2O"]["radius"]) + spec["margin"]
    n_water = int(round(float(np.prod(L)) * spec["water_density_g_cm3"]
                        / (molecular_mass("H2O") * _AMU_PER_A3_TO_G_CM3)))
    frames, accs, core_counts, d_nns = [], [], [], []
    for k in range(N_FRAMES):
        rng = np.random.default_rng(spec["seed"] + k)
        water = _dense_pack({"H2O": n_water}, box, rng, molecular,
                            margin=spec["margin"], obstacles=cu,
                            z_range=(z_half + inset, float(box[2]) - inset))
        frame = _sanitize(Frame(pos=np.vstack([cu.pos, water.pos]),
                                cell=np.diag(box),
                                symbols=list(cu.symbols) + list(water.symbols)))
        frames.append(frame)
        # ground-truth measurement: planted labels vs the dialect's phase call
        truth = np.array([s != "Cu" for s in frame.symbols])   # water = liquid
        labels = phase_labels(frame, metal)
        d_nn = typical_neighbor_distance(frame)
        z = frame.pos[:, 2]
        band = 2.0 * d_nn
        core = ((np.abs(z - z_half) > band)
                & (z > band) & (z < float(box[2]) - band))
        accs.append(label_accuracy(labels[core], (~truth)[core]))
        core_counts.append(int(core.sum()))
        d_nns.append(float(d_nn))
    target = spec["segmentation_target"]
    min_acc = float(min(accs))
    ground = _gt_header(
        spec["id"], "interfaces",
        f"Cu/water interface: fcc Cu block ({spec['reps'][0]}x{spec['reps'][1]}"
        f"x{spec['reps'][2]} conventional cells, a = {spec['a']}) filling the "
        f"lower half of a doubled box, {n_water} H2O RSA-packed into the "
        f"upper half at {spec['water_density_g_cm3']} g/cm3 with the Cu atoms "
        f"as excluded obstacles; per-frame two-phase label accuracy measured "
        f"with the core+metal dialect (q6 rule, interface band excluded)",
        spec["seed"])
    ground["lift_mode"] = "interface"
    ground["lift_dialect"] = ["core", "metal"]
    ground["expected"] = {
        "axis": "z",
        "boundary_z": z_half,
        "phases": {"solid": "Cu below", "liquid": "H2O above"},
        "n_water": n_water,
        "water_density_g_cm3": spec["water_density_g_cm3"],
        "segmentation_target": target,
        "segmentation_min_measured": min_acc,
        "segmentation_note": (
            "target 0.9 measured on every frame; the per-atom q6 rule counts "
            "water hydrogens, whose coordination is intramolecular, so frames "
            "recording below target would lower the test threshold to "
            "measured - 0.05 (not needed here)"),
        "counts": _species_counts(frames[0].symbols),
        "n_atoms": len(frames[0]),
        "cell": [float(v) for v in frames[0].cell_diag],
    }
    for k, frame in enumerate(frames):
        ground["frames"].append(
            _frame_record(k, spec["seed"] + k, frame,
                          segmentation_accuracy=accs[k],
                          segmentation_core_atoms=core_counts[k],
                          typical_neighbor_distance=d_nns[k]))
    entry = _write_case(out, "interfaces", spec["id"], frames, ground)
    print(f"[interface] {spec['id']}: boundary z={z_half:.3f} "
          f"min segmentation acc={min_acc:.4f} {N_FRAMES} frames x "
          f"{len(frames[0])} atoms counts={ground['expected']['counts']}")
    return [entry]


def generate(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    metal = load_dialect(("core", "metal"))
    molecular = load_dialect(("core", "molecular"))

    cases = []
    cases += generate_crystals(out, metal)
    cases += generate_defects(out, metal)
    cases += generate_fluids(out, molecular)
    cases += generate_reactive(out, molecular)
    cases += generate_glass(out)
    cases += generate_interface(out)
    cases += generate_surface(out)
    cases += generate_nacl_aq(out, molecular)
    cases += generate_electrolyte(out, molecular)
    cases += generate_co2(out, molecular)
    cases += generate_si001(out)
    cases += generate_cu_water(out, molecular, metal)

    manifest = {"schema": SCHEMA, "generator": "bench/generate.py",
                "n_cases": len(cases), "n_frames_per_case": N_FRAMES,
                "cases": cases}
    (out / "index.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                    encoding="utf-8")
    print(f"wrote {len(cases)} cases x {N_FRAMES} frames to {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Chaord-Bench cases.")
    ap.add_argument("--out", default="bench/data",
                    help="output directory (default: bench/data)")
    args = ap.parse_args()
    generate(Path(args.out))


if __name__ == "__main__":
    main()
