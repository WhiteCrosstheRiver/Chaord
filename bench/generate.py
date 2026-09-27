"""Chaord-Bench generator (PLAN WP21): seeded case builders + ground truth.

Each case directory under --out receives one frame (frame.npz, keys r/L/symbols
with symbols stored as dtype U8) and a ground_truth.json describing what the
frame contains. An index.json manifest at the root lists every case.

Run with the project virtualenv:

    .venv/Scripts/python.exe bench/generate.py --out bench/data

Everything is seeded, so re-running reproduces byte-identical frames.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from chaord.build.crystal import build_conventional
from chaord.build.defects import apply_defects
from chaord.build.molecules import TEMPLATES, molecule_census, pack_molecules
from chaord.dialects import load_dialect
from chaord.io.frames import write_frame
from chaord.lang.ir import KVDefect, Name, Quantity, Statement

SCHEMA = "chaord-bench/1"

# --------------------------------------------------------------------- specs --

# (case id, prototype, lattice params, species per slot, supercell repetitions)
CRYSTAL_CASES = [
    ("fcc_cu", "fcc", {"a": 3.615}, ("Cu",), (2, 2, 2)),
    ("bcc_fe", "bcc", {"a": 2.87}, ("Fe",), (2, 2, 2)),
    ("rocksalt_nacl", "rocksalt", {"a": 5.64}, ("Na", "Cl"), (2, 2, 2)),
    ("l12_ni3al", "L1_2", {"a": 3.572}, ("Ni", "Al"), (2, 2, 2)),
]

DEFECT_CASE = {  # fcc Cu host + vacancies
    "id": "fcc_cu_vacancies",
    "prototype": "fcc",
    "params": {"a": 3.615},
    "species": ("Cu",),
    "reps": (3, 3, 3),
    "defects": [("V_Cu", 3)],
    "seed": 11,
}

FLUID_CASE = {"id": "water_box15", "molecules": {"H2O": 60},
              "box": (15.0, 15.0, 15.0), "seed": 13}

REACTIVE_CASE = {"id": "water_oh_h_box20", "molecules": {"H2O": 50, "OH": 9, "H": 9},
                 "box": (20.0, 20.0, 20.0), "seed": 3}


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


def _write_case(out: Path, category: str, case_id: str, frame, ground: dict) -> dict:
    case_dir = out / category / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    npz = case_dir / "frame.npz"
    gt_path = case_dir / "ground_truth.json"
    write_frame(npz, frame)

    # storage sanity: the benchmark snapshot format is r (N,3), L (3,), U8 symbols
    with np.load(npz) as z:
        assert set(z.files) == {"r", "L", "symbols"}, f"{npz}: keys {z.files}"
        assert z["r"].shape == frame.pos.shape
        assert z["symbols"].dtype == np.dtype("U8")
        assert z["L"].shape == (3,)

    gt_path.write_text(json.dumps(ground, indent=2, sort_keys=False) + "\n",
                       encoding="utf-8")
    return {"id": f"{category}/{case_id}", "category": category,
            "npz": str(npz.relative_to(out)), "ground_truth": str(gt_path.relative_to(out))}


def _gt_header(case_id: str, category: str, description: str, seed) -> dict:
    return {"schema": SCHEMA, "case": case_id, "category": category,
            "description": description, "seed": seed}


# -------------------------------------------------------------------- cases --

def generate_crystals(out: Path) -> list:
    entries = []
    for case_id, name, params, slots, reps in CRYSTAL_CASES:
        frame = build_conventional(name, params, slots, reps)
        counts = _species_counts(frame.symbols)
        composition = None
        if len(slots) > 1:  # canonical composition string, e.g. 'Ni3Al'
            from math import gcd
            from chaord.build.crystal import SLOT_COUNTS
            want = SLOT_COUNTS[name]
            g = 0
            for n in want:
                g = gcd(g, n)
            composition = "".join(f"{el}{n // g if n // g > 1 else ''}"
                                  for el, n in zip(slots, want))
        ground = _gt_header(
            case_id, "crystals",
            f"{name} {composition or slots[0]}, conventional {reps[0]}x{reps[1]}x{reps[2]} "
            f"supercell built exactly from the prototype table", None)
        ground["lift_mode"] = "crystal"
        ground["expected"] = {
            "prototype": name,
            "a": params["a"],
            "composition": composition,
            "reps": list(reps),
            "cell": [float(v) for v in frame.cell_diag],
            "counts": counts,
            "n_atoms": len(frame),
        }
        entries.append(_write_case(out, "crystals", case_id, frame, ground))
        print(f"[crystals] {case_id}: {name} a={params['a']} "
              f"atoms={len(frame)} counts={counts}")
    return entries


def generate_defects(out: Path, dialect) -> list:
    spec = DEFECT_CASE
    host = build_conventional(spec["prototype"], spec["params"], spec["species"],
                              spec["reps"])
    rng = np.random.default_rng(spec["seed"])
    stmts = [_kv_defect(tok, n) for tok, n in spec["defects"]]
    frame = apply_defects(host, stmts, rng, dialect)
    ground = _gt_header(
        spec["id"], "defects",
        f"fcc Cu {spec['reps'][0]}x{spec['reps'][1]}x{spec['reps'][2]} host with "
        f"{spec['defects'][0][1]} random Cu vacancies (Kröger-Vink V_Cu)",
        spec["seed"])
    ground["lift_mode"] = "defects"
    ground["host"] = {"prototype": spec["prototype"], "a": spec["params"]["a"],
                      "reps": list(spec["reps"]),
                      "counts": _species_counts(host.symbols)}
    ground["expected"] = {
        "defects": {tok: n for tok, n in spec["defects"]},
        "counts": _species_counts(frame.symbols),
        "n_atoms": len(frame),
    }
    entries = [_write_case(out, "defects", spec["id"], frame, ground)]
    print(f"[defects ] {spec['id']}: {dict(spec['defects'])} "
          f"atoms={len(frame)} counts={ground['expected']['counts']}")
    return entries


def generate_fluid(out: Path, dialect) -> list:
    spec = FLUID_CASE
    rng = np.random.default_rng(spec["seed"])
    frame = pack_molecules(spec["molecules"], spec["box"], rng, dialect)
    census = dict(sorted(molecule_census(frame, dialect).items()))
    planted = _census_to_names(census)
    if planted != spec["molecules"]:
        raise RuntimeError(f"census {census} does not match the packing spec "
                           f"{spec['molecules']}")
    ground = _gt_header(
        spec["id"], "fluid",
        f"{spec['molecules']['H2O']} water molecules packed into a "
        f"{spec['box'][0]:.0f} A cubic box (seeded random-sequential packing)",
        spec["seed"])
    ground["lift_mode"] = "fluid"
    ground["molecules"] = spec["molecules"]
    ground["box"] = list(spec["box"])
    ground["expected"] = {
        "molecules": spec["molecules"],
        "census": census,
        "counts": _species_counts(frame.symbols),
        "n_atoms": len(frame),
    }
    entries = [_write_case(out, "fluid", spec["id"], frame, ground)]
    print(f"[fluid   ] {spec['id']}: census={census} atoms={len(frame)}")
    return entries


def generate_reactive(out: Path, dialect) -> list:
    spec = REACTIVE_CASE
    rng = np.random.default_rng(spec["seed"])
    frame = pack_molecules(spec["molecules"], spec["box"], rng, dialect)
    census = dict(sorted(molecule_census(frame, dialect).items()))
    planted = _census_to_names(census)
    if planted != spec["molecules"]:
        raise RuntimeError(f"census {census} does not match the packing spec "
                           f"{spec['molecules']}")
    ground = _gt_header(
        spec["id"], "reactive",
        "dissociation fragment mixture (water, hydroxide, atomic hydrogen) in "
        "a 20 A box; species census must be recovered exactly from the bond graph",
        spec["seed"])
    ground["molecules"] = spec["molecules"]
    ground["box"] = list(spec["box"])
    ground["expected"] = {
        "molecules": spec["molecules"],
        "census": census,
        "counts": _species_counts(frame.symbols),
        "n_atoms": len(frame),
    }
    entries = [_write_case(out, "reactive", spec["id"], frame, ground)]
    print(f"[reactive] {spec['id']}: census={census} atoms={len(frame)}")
    return entries


def generate(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    metal = load_dialect(("core", "metal"))
    molecular = load_dialect(("core", "molecular"))

    cases = []
    cases += generate_crystals(out)
    cases += generate_defects(out, metal)
    cases += generate_fluid(out, molecular)
    cases += generate_reactive(out, molecular)

    manifest = {"schema": SCHEMA, "generator": "bench/generate.py",
                "n_cases": len(cases), "cases": cases}
    (out / "index.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                    encoding="utf-8")
    print(f"wrote {len(cases)} cases to {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Chaord-Bench cases.")
    ap.add_argument("--out", default="bench/data",
                    help="output directory (default: bench/data)")
    args = ap.parse_args()
    generate(Path(args.out))


if __name__ == "__main__":
    main()
