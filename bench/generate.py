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
from chaord.build.defects import apply_defects, assign_occupancy, nearest_neighbor_distance
from chaord.build.molecules import TEMPLATES, molecule_census, pack_molecules
from chaord.dialects import load_dialect
from chaord.io.frames import Frame, write_frame
from chaord.lang.ir import KVDefect, Name, Quantity, Statement
from chaord.lang.parser import parse_text

SCHEMA = "chaord-bench/2"
N_FRAMES = 5

# --------------------------------------------------------------------- specs --

# exact crystals; each frame is a fresh thermal displacement of the same host
# (case id, prototype, lattice params, species per slot, supercell reps,
#  orientation rows or None, base seed)
CRYSTAL_CASES = [
    ("fcc_cu", "fcc", {"a": 3.615}, ("Cu",), (2, 2, 2), None, 101),
    ("bcc_fe", "bcc", {"a": 2.87}, ("Fe",), (2, 2, 2), None, 103),
    ("rocksalt_nacl", "rocksalt", {"a": 5.64}, ("Na", "Cl"), (2, 2, 2), None, 107),
    ("l12_ni3al", "L1_2", {"a": 3.572}, ("Ni", "Al"), (2, 2, 2), None, 109),
    ("hcp_mg", "hcp", {"a": 3.21, "c": 5.21}, ("Mg",), (2, 2, 2),
     np.array([[1, 0, 0], [1, 2, 0], [0, 0, 1]], int), 113),
    ("diamond_si", "diamond", {"a": 5.43}, ("Si",), (2, 2, 2), None, 127),
    ("perovskite_srtio3", "perovskite", {"a": 3.905}, ("Sr", "Ti", "O"),
     (2, 2, 2), None, 131),
]

# fcc solid solution: same fcc host, Cr/Co/Ni assigned to sites with unit
# fractions (a fresh random assignment per frame; counts stay exact)
SOLUTION_CASE = {
    "id": "fcc_crconi",
    "prototype": "fcc",
    "params": {"a": 3.56},
    "host_species": ("Cr",),
    "occupancy": (("Cr", "1/3"), ("Co", "1/3"), ("Ni", "1/3")),
    "reps": (3, 3, 3),
    "seed": 137,
}

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

# random-sequential packing of rigid molecules; re-packed per frame
FLUID_CASES = [
    {"id": "water_box15", "molecules": {"H2O": 60}, "box": (15.0, 15.0, 15.0),
     "seed": 13},
    {"id": "ar_gas_box25", "molecules": {"Ar": 50}, "box": (25.0, 25.0, 25.0),
     "seed": 151},
    {"id": "n2_box22", "molecules": {"N2": 80}, "box": (22.0, 22.0, 22.0),
     "seed": 157},
]

REACTIVE_CASE = {"id": "water_oh_h_box20", "molecules": {"H2O": 50, "OH": 9, "H": 9},
                 "box": (20.0, 20.0, 20.0), "seed": 3}

# LJ glass: overlap-free RSA packing at a stated number density (physics off)
GLASS_CASE = {"id": "lj_glass_rho085", "n": 200, "rho": 0.85, "species": "X",
              "seed": 163}

# LJ solid-liquid slab: fcc lower half kept perfect, upper half re-randomised
# per frame (same construction as tests/test_surfaces.py::
# test_segmentation_labels_planted, in reduced units)
INTERFACE_CASE = {"id": "lj_solid_liquid", "rho_solid": 1.0, "reps": (4, 4, 8),
                  "species": "X", "liquid_margin": 0.5, "seed": 167}

# Ni(111) slab + O adsorbates, built by the real surface builder, then stored
# in its rectangular (orthohexagonal, index-2) supercell so the diagonal-only
# npz snapshot format round-trips exactly
SURF_PROGRAM = """chaord 0.1
dialect core + metal + surface

system {{
  cell {lx:.3f} {ly:.3f} {lz:.3f}
  pbc xyz
  conserve atoms Ni {n}
}}

physics {{
  backend eam
}}

crystal slab : slab z 0 .. {ztop:.1f} {{
  lattice fcc
  a 3.615 A
  surface ({hkl}) top
  adsorb O count 4 site top
}}

vacuum gap : slab z {ztop:.1f} .. {lz:.1f} {{
}}
"""
SURFACE_CASE = {"id": "ni111_o_top", "hkl": "111", "n_adsorb": 4,
                # 3x3x4 ASE fcc(111) slab at a = 3.615 with a 12 A vacuum
                "lx": 7.669, "ly": 6.641, "lz": 18.26, "ztop": 6.3, "n_ni": 36,
                "seed": 173}

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
        rel_frames.append(str(npz.relative_to(out)))
    gt_path = case_dir / "ground_truth.json"
    gt_path.write_text(json.dumps(ground, indent=2, sort_keys=False) + "\n",
                       encoding="utf-8")
    return {"id": f"{category}/{case_id}", "category": category,
            "frames": rel_frames, "ground_truth": str(gt_path.relative_to(out))}


def _gt_header(case_id: str, category: str, description: str, seed) -> dict:
    return {"schema": SCHEMA, "case": case_id, "category": category,
            "description": description, "seed": seed, "n_frames": N_FRAMES,
            "frames": []}


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
    # the solid solution is built from the same exact-crystal machinery
    specs.append((SOLUTION_CASE["id"], SOLUTION_CASE["prototype"],
                  SOLUTION_CASE["params"], SOLUTION_CASE["host_species"],
                  SOLUTION_CASE["reps"], None, SOLUTION_CASE["seed"]))
    for case_id, name, params, slots, reps, orient, seed in specs:
        host = _sanitize(build_conventional(name, params, slots, reps,
                                            orient_rows=orient))
        amp = amp_frac * nearest_neighbor_distance(host)
        solution = case_id == SOLUTION_CASE["id"]
        occ_stmt = None
        if solution:
            occ_stmt = Statement(
                kind="build", key="occupancy",
                values=[v for pair in SOLUTION_CASE["occupancy"]
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
            case_id, "crystals",
            f"{name} {composition or slots[0]}, {reps[0]}x{reps[1]}x{reps[2]} "
            f"supercell built exactly from the prototype table"
            + (", Cr/Co/Ni assigned to fcc sites with unit fractions" if solution else "")
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
                                               SOLUTION_CASE["occupancy"]}
        for k, frame in enumerate(frames):
            ground["frames"].append(_frame_record(k, seed + k, frame))
        entries.append(_write_case(out, "crystals", case_id, frames, ground))
        print(f"[crystals] {case_id}: {name} a={params['a']} "
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
    for k in range(N_FRAMES):
        rng = np.random.default_rng(spec["seed"] + k)
        frame = pack_molecules(spec["molecules"], spec["box"], rng, dialect)
        census = dict(sorted(molecule_census(frame, dialect).items()))
        planted = _census_to_names(census)
        if planted != spec["molecules"]:
            raise RuntimeError(f"census {census} does not match the packing spec "
                               f"{spec['molecules']}")
        frames.append(_sanitize(frame))
        census_list.append(census)
    ground = _gt_header(spec["id"], category, description, spec["seed"])
    ground["lift_mode"] = "fluid" if category == "fluid" else "reactive"
    ground["lift_dialect"] = ["core", "molecular"]
    ground["molecules"] = spec["molecules"]
    ground["box"] = list(spec["box"])
    ground["expected"] = {
        "molecules": spec["molecules"],
        "census": census_list[0],
        "counts": _species_counts(frames[0].symbols),
        "n_atoms": len(frames[0]),
    }
    for k, (frame, census) in enumerate(zip(frames, census_list)):
        ground["frames"].append(_frame_record(k, spec["seed"] + k, frame,
                                              census=census))
    entry = _write_case(out, category, spec["id"], frames, ground)
    print(f"[{category:<8}] {spec['id']}: census={census_list[0]} "
          f"{N_FRAMES} frames x {len(frames[0])} atoms")
    return [entry]


def generate_fluids(out: Path, molecular) -> list:
    entries = []
    for spec in FLUID_CASES:
        main = next(iter(spec["molecules"]))
        entries += _packed_frames(
            spec, molecular, "fluid",
            f"{spec['molecules'][main]} {main} packed into a "
            f"{spec['box'][0]:.0f} A cubic box (seeded random-sequential packing, "
            f"re-packed per frame)", out)
    return entries


def generate_reactive(out: Path, molecular) -> list:
    spec = REACTIVE_CASE
    return _packed_frames(
        spec, molecular, "reactive",
        "dissociation fragment mixture (water, hydroxide, atomic hydrogen) in "
        "a 20 A box; the species census must be recovered exactly from the bond "
        "graph of every frame", out)


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
        f"density {spec['rho']} /sigma^3 (physics off, no protocol)", spec["seed"])
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


def generate_interface(out: Path) -> list:
    """LJ solid-liquid slab: perfect fcc below the mid-plane, random above."""
    spec = INTERFACE_CASE
    lj = load_dialect(("core", "lj"))
    a = float((4.0 / spec["rho_solid"]) ** (1.0 / 3.0))  # fcc cube edge at rho_solid
    solid = build_conventional("fcc", {"a": a}, (spec["species"],), spec["reps"])
    L = solid.cell_diag
    boundary = L[2] / 2.0
    margin = spec["liquid_margin"]
    frames = []
    for k in range(N_FRAMES):
        rng = np.random.default_rng(spec["seed"] + k)
        truth = solid.pos[:, 2] > boundary
        pos = solid.pos.copy()
        n_liq = int(truth.sum())
        upper = rng.uniform(0, L, (n_liq, 3))
        upper[:, 2] = boundary + rng.uniform(0, L[2] / 2.0 - margin, n_liq)
        pos[truth] = upper
        frames.append(_sanitize(Frame(pos=pos, cell=solid.cell,
                                      symbols=solid.symbols)))
    ground = _gt_header(
        spec["id"], "interface",
        f"LJ solid-liquid slab: perfect fcc ({spec['reps'][0]}x{spec['reps'][1]}x"
        f"{spec['reps'][2]}, rho_solid = {spec['rho_solid']}/sigma^3) below "
        f"z = {boundary:.4f}, re-randomised liquid above; the two-phase label "
        f"boundary is the mid-plane of the box", spec["seed"])
    ground["lift_mode"] = "interface"
    ground["lift_dialect"] = ["core", "lj"]
    ground["expected"] = {
        "axis": "z",
        "boundary_z": float(boundary),
        "phases": {"solid": "below", "liquid": "above"},
        "counts": _species_counts(frames[0].symbols),
        "n_atoms": len(frames[0]),
        "cell": [float(v) for v in frames[0].cell_diag],
    }
    for k, frame in enumerate(frames):
        ground["frames"].append(_frame_record(k, spec["seed"] + k, frame,
                                              boundary_z=float(boundary)))
    entry = _write_case(out, "interface", spec["id"], frames, ground)
    print(f"[interface] {spec['id']}: boundary z={boundary:.3f} {N_FRAMES} "
          f"frames x {len(frames[0])} atoms")
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
    spec = SURFACE_CASE
    surface = load_dialect(("core", "metal", "surface"))
    text = SURF_PROGRAM.format(lx=spec["lx"], ly=spec["ly"], lz=spec["lz"],
                               n=spec["n_ni"], hkl=spec["hkl"], ztop=spec["ztop"])
    program = parse_text(text)
    frames = []
    for k in range(N_FRAMES):
        rng = np.random.default_rng(spec["seed"] + k)
        slab = build_program(program, surface, rng=rng)
        frames.append(_sanitize(_ortho_double(slab)))
    counts = _species_counts(frames[0].symbols)
    ground = _gt_header(
        spec["id"], "surface",
        f"Ni({spec['hkl']}) slab with O adsorbates on top sites; built by the "
        f"surface builder, stored in its rectangular supercell so the diagonal "
        f"npz format round-trips exactly (adsorbate sites re-drawn per frame)",
        spec["seed"])
    ground["lift_mode"] = "surface"
    ground["lift_dialect"] = ["core", "metal", "surface"]
    ground["expected"] = {
        "surface": f"({spec['hkl']})",
        "termination": "Ni",
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
    entry = _write_case(out, "surface", spec["id"], frames, ground)
    print(f"[surface ] {spec['id']}: ({spec['hkl']}) O on top x{counts['O']} "
          f"{N_FRAMES} frames x {len(frames[0])} atoms counts={counts}")
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
