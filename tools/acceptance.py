"""Chaord v1.0 acceptance runner: executes criteria A1-A14 exactly as PLAN.md
defines them, with verification logic implemented independently of the library
under test.  This file counts atoms, parses program text, computes observables
and does defect arithmetic with its own code and its own data tables; it never
calls chaord.check helpers to decide a verdict.

Usage: python tools/acceptance.py [-o reports/acceptance.json] [--details PATH]
                                [--only A1,A2,...] [--quick]

reports/acceptance.json keeps the {criteria:[{id,name,passed,evidence}],
passed, total} schema; reports/acceptance_details.md carries the per-case
numbers (A4 P/R matrix, A5 per-case ratios, ...).  Honest FAILs (e.g. A5 with
no measured noise floor on record, A13 on frames the lifter refuses) are real
results and are reported as FAIL, never massaged.

Mutation hooks: every check_aN(mutation=...) injects one seeded fault into the
input and must return passed=False; tests/acceptance/test_mutations.py asserts
exactly that, one canary per criterion.
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import subprocess
import sys
import tempfile
import time
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

BENCH = ROOT / "bench" / "data"
NOISE_FLOORS = ROOT / "reports" / "noise_floors.json"
REFERENCE_MD = ROOT / "docs" / "reference.md"
REFERENCE_GENERATED = ROOT / "docs" / "reference_generated.md"


def _stable_seed(text: str) -> int:
    return zlib.crc32(text.encode("utf-8"))


# ============================================================ independent =====
# Verification-side data tables, written from the crystallography / chemistry
# literature on purpose: the verifier must not import the tables of the code it
# verifies.  (tools/ is not pass code; the CI magic-number grep covers
# src/chaord/lift and src/chaord/build only.)

# prototype -> (atoms per conventional cell per slot, cell-volume factor of the
# printed cell-diagonal product).  Shapes: cubic 1.0, hexagonal sqrt(3)/2
# (120 deg between a and b), tetragonal 1.0.  Source: International Tables.
PROTO_TABLE = {
    "sc":         ((1,),      1.0),
    "bcc":        ((2,),      1.0),
    "fcc":        ((4,),      1.0),
    "diamond":    ((8,),      1.0),
    "hcp":        ((2,),      np.sqrt(3) / 2),
    "rocksalt":   ((4, 4),    1.0),
    "cscl":       ((1, 1),    1.0),
    "zincblende": ((4, 4),    1.0),
    "wurtzite":   ((2, 2),    np.sqrt(3) / 2),
    "fluorite":   ((4, 8),    1.0),
    "perovskite": ((1, 1, 3), 1.0),
    "L1_2":       ((3, 1),    1.0),
    "rutile":     ((2, 4),    1.0),
}

# canonical molecular formula -> element multiset (verifier's own table;
# census keys are C first, H second, then alphabetical, hence HO / F6P)
MOLECULE_TABLE = {
    "H2O":    {"H": 2, "O": 1},
    "HO":     {"H": 1, "O": 1},           # census canonical key for OH
    "H":      {"H": 1},
    "N2":     {"N": 2},
    "O2":     {"O": 2},
    "CO2":    {"C": 1, "O": 2},
    "Ar":     {"Ar": 1},
    "Na":     {"Na": 1},
    "Cl":     {"Cl": 1},
    "Li":     {"Li": 1},
    "C3H4O3": {"C": 3, "H": 4, "O": 3},   # ethylene carbonate
    "F6P":    {"F": 6, "P": 1},           # hexafluorophosphate
}

# monovalent formal charges for the verifier's charge bookkeeping
ION_CHARGES = {"Na": 1, "Cl": -1, "Li": 1, "K": 1, "F": -1, "Cs": 1}
# elements that appear inside polyatomic ions (PF6-): element-wise charge is
# undefined, so charge conservation is not checkable on such frames
MOLECULAR_ION_ELEMENTS = {"F", "P"}


def count_species(symbols) -> dict[str, int]:
    out: dict[str, int] = {}
    for s in symbols:
        out[s] = out.get(s, 0) + 1
    return dict(sorted(out.items()))


def frame_charge(symbols) -> int:
    """Total formal charge from the verifier's own table (0 for unlisted)."""
    return sum(ION_CHARGES.get(s, 0) for s in symbols)


def charge_checkable(symbols) -> bool:
    """Element-wise charge bookkeeping is well-defined unless a polyatomic-ion
    constituent (e.g. F/P from PF6-) is present."""
    return not (set(symbols) & MOLECULAR_ION_ELEMENTS)


# ------------------------------------------------- independent text parser ----
_RE_CONSERVE_ATOMS = re.compile(r"^\s*conserve atoms (?P<rest>.+)$")
_RE_CONSERVE_CHARGE = re.compile(
    r"^\s*conserve charge\s+(?P<num>[+-]?\d+(?:\.\d+)?)")
_RE_DEFECT = re.compile(r"^\s*defect (?P<tok>\S+) count (?P<n>\d+)")
_RE_REGION = re.compile(r"^\s*(?P<phase>crystal|amorphous|liquid|gas|fluid|"
                        r"cluster|vacuum)\s+\S+\s*:\s*(?P<geo>.+?)\s*\{")
_RE_RESIDUAL_ATOM = re.compile(
    r"^\s*atom (?P<sym>\S+) (?P<x>\S+) (?P<y>\S+) (?P<z>\S+)")
_RE_CELL = re.compile(
    r"^\s*cell (?P<v1>[-+0-9.eE]+)(?:\s+(?P<v2>[-+0-9.eE]+))?"
    r"(?:\s+(?P<v3>[-+0-9.eE]+))?")
_RE_NUM = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?:/\d+)?"
_RE_STMT = re.compile(r"^\s*(?:(?:state|constrain|assert|history|conserve)\s+)?"
                      r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)\b")


def parse_program_text(text: str) -> dict:
    """Independent parser of canonical .chaord text (regex level, no Lark).

    Extracts: conserve_atoms {sym: n}, conserve_charge, defect statements per
    region, region statements (key -> [rest-of-line, ...]; a key may appear on
    several lines, e.g. one `molecules` line per species), residual atom lines.
    """
    out = {"conserve_atoms": {}, "conserve_charge": None, "regions": [],
           "residual_none": True, "residual_atoms": {}, "cell": None}
    current = None
    in_residual = False
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if m := _RE_CONSERVE_ATOMS.match(line):
            toks = m.group("rest").split()
            for i in range(0, len(toks) - 1, 2):
                out["conserve_atoms"][toks[i]] = int(float(toks[i + 1]))
            continue
        if m := _RE_CONSERVE_CHARGE.match(line):
            out["conserve_charge"] = int(round(float(m.group("num"))))
            continue
        if m := _RE_DEFECT.match(line):
            if current is not None:
                current["defects"][m.group("tok")] = int(m.group("n"))
            continue
        if s == "residual none":
            out["residual_none"] = True
            current, in_residual = None, False
            continue
        if s == "residual {":
            out["residual_none"] = False
            current, in_residual = None, True
            continue
        if in_residual:
            if m := _RE_RESIDUAL_ATOM.match(line):
                out["residual_atoms"][m.group("sym")] = \
                    out["residual_atoms"].get(m.group("sym"), 0) + 1
            continue
        if m := _RE_REGION.match(line):
            current = {"phase": m.group("phase"), "geometry": m.group("geo"),
                       "stmts": {}, "defects": {}}
            out["regions"].append(current)
            continue
        if current is not None and s == "}":
            current = None
            continue
        if current is None:
            if m := _RE_CELL.match(line):
                out["cell"] = [float(m.group(k)) for k in ("v1", "v2", "v3")
                               if m.group(k)]
            continue
        if m := _RE_STMT.match(s):
            current["stmts"].setdefault(m.group("key"), []).append(
                s[m.end():].strip())
    return out


def _stmt(region: dict, key: str, join: bool = False) -> str | None:
    """First rest-of-line for a statement key (all lines joined if join)."""
    vals = region["stmts"].get(key)
    if not vals:
        return None
    return " ".join(vals) if join else vals[0]


def _as_float(tok: str) -> float:
    if "/" in tok:
        a, b = tok.split("/")
        return float(a) / float(b)
    return float(tok)


def _parse_formula(formula: str) -> dict[str, int]:
    """'Ni3Al' -> {'Ni': 3, 'Al': 1}; a plain element -> {E: 1}."""
    out: dict[str, int] = {}
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        if el:
            out[el] = out.get(el, 0) + (int(n) if n else 1)
    return out


def _apply_defect_net(total: dict[str, int], defects: dict[str, int]):
    """Kröger-Vink net effect on species counts (verifier arithmetic)."""
    for tok, n in defects.items():
        if tok == "frenkel_pair":
            continue                        # vacancy + interstitial: net zero
        site, sub = tok.split("^")[0].split("_", 1)
        if site == "V":                     # vacancy: one `sub` atom gone
            total[sub] = total.get(sub, 0) - n
        elif sub == "i":                    # interstitial: one `site` atom added
            total[site] = total.get(site, 0) + n
        else:                               # antisite: `site` atom on `sub` site
            total[sub] = total.get(sub, 0) - n
            total[site] = total.get(site, 0) + n


def _gcd(*vals):
    from math import gcd
    out = 0
    for v in vals:
        out = gcd(out, v)
    return out


def derive_counts(parsed: dict):
    """Atom counts implied by the region statements of a lifted program.

    Verifier arithmetic: prototype/composition or lattice/occupancy times the
    conventional-cell multiplicity implied by the stated cell, minus the defect
    net, plus residual atom lines; molecular regions expand the verifier's
    formula table.  Returns (dict, how) on success, ({'__expected_total__': n},
    how) when only the site total is pinned, or (None, reason) when the program
    shape does not pin the counts (slab/interface programs, atomic fluids,
    amorphous composition without counts).
    """
    regions = parsed["regions"]
    if not regions:
        return None, "no region"
    if any(r["phase"] not in ("crystal", "liquid", "gas", "fluid", "amorphous")
           for r in regions):
        return None, "unsupported phase"
    if any(any(w in r["geometry"] for w in ("slab", "box", "sphere", "cylinder"))
           for r in regions):
        return None, "non-trivial region geometry (counts not derivable)"
    total: dict[str, int] = {}
    how = []
    cell = parsed["cell"]
    for r in regions:
        st = r["stmts"]
        if "molecules" in st:
            for line in st["molecules"]:
                for name, n in re.findall(r"(\S+)\s+(" + _RE_NUM + r")", line):
                    if name not in MOLECULE_TABLE:
                        return None, f"unknown molecule {name}"
                    for el, k in MOLECULE_TABLE[name].items():
                        total[el] = total.get(el, 0) + k * int(round(_as_float(n)))
            how.append("molecules")
            continue
        if r["phase"] == "amorphous":
            comp_line = _stmt(r, "composition")
            if comp_line is None:
                return None, "amorphous region without composition"
            pairs = re.findall(r"([A-Z][a-z]?)\s+(" + _RE_NUM + r")", comp_line)
            if pairs:
                for el, n in pairs:
                    total[el] = total.get(el, 0) + int(round(_as_float(n)))
                how.append("amorphous composition counts")
            else:
                return None, "amorphous composition names no counts"
            continue
        if r["phase"] in ("liquid", "gas", "fluid"):
            return None, "atomic fluid region (no composition statement)"
        # crystal regions
        proto_line = _stmt(r, "prototype") or _stmt(r, "lattice")
        if proto_line is None:
            return None, "crystal region without lattice/prototype"
        name = proto_line.split()[0]
        if name not in PROTO_TABLE:
            return None, f"unknown prototype {name}"
        slots, vfac = PROTO_TABLE[name]
        params = {}
        for key in ("a", "c"):
            line = _stmt(r, key)
            if line is not None:
                params[key] = _as_float(line.split()[0])
        if not params or cell is None or len(cell) != 3:
            return None, "missing lattice parameter or system cell"
        a = params.get("a")
        c = params.get("c", a)
        v_conv = a * a * c * vfac
        n_cells = int(round(float(np.prod(cell)) * vfac / v_conv))
        comp_line = _stmt(r, "composition")
        occ_line = _stmt(r, "occupancy", join=True)
        if comp_line is not None:
            # composition gives the formula; the prototype table gives the
            # per-conventional-cell multiplicities; their reduced ratios must
            # agree, and lambda rescales formula counts to per-cell counts
            comp = _parse_formula(comp_line.split()[0])
            g_f = _gcd(*comp.values())
            g_s = _gcd(*slots)
            ratio_f = sorted(v // g_f for v in comp.values())
            ratio_s = sorted(v // g_s for v in slots)
            if ratio_f != ratio_s:
                return None, (f"composition {comp_line.split()[0]} does not "
                              f"fit {name}")
            lam = sum(slots) // sum(comp.values())
            if sum(slots) != lam * sum(comp.values()):
                return None, f"composition {comp_line.split()[0]} does not tile {name}"
            for el, f_el in comp.items():
                total[el] = total.get(el, 0) + f_el * lam * n_cells
            how.append(f"{name}x{n_cells}+composition")
        elif occ_line is not None:
            n_sites = sum(slots) * n_cells
            for el, frac in re.findall(r"(\S+)\s+(" + _RE_NUM + r")", occ_line):
                total[el] = total.get(el, 0) + int(round(_as_float(frac) * n_sites))
            how.append(f"occupancy on {n_sites} sites x{n_cells} cells")
        else:
            # unary lattice: the program names the species only in conserve,
            # but the site total is pinned by lattice x cell -- the verifier
            # checks the total (sites + defect net + residual) independently.
            n_sites = sum(slots) * n_cells
            delta = 0
            for tok, n in r["defects"].items():
                if tok == "frenkel_pair":
                    continue
                site, sub = tok.split("^")[0].split("_", 1)
                if site == "V":
                    delta -= n
                elif sub == "i":
                    delta += n
            expected_total = (n_sites + delta
                              + sum(parsed["residual_atoms"].values()))
            how.append(f"sites-only: {n_sites} {name} sites + net {delta} "
                       f"+ residual {sum(parsed['residual_atoms'].values())}")
            return {"__expected_total__": expected_total}, "+".join(how)
        _apply_defect_net(total, r["defects"])
    for sym, n in parsed["residual_atoms"].items():
        total[sym] = total.get(sym, 0) + n
    total = {k: v for k, v in total.items() if v}
    if not total:
        return None, "derivation produced no counts"
    return total, "+".join(how)


# ------------------------------------------------------- independent physics --
def median_nn_distance(pos, cell_diag) -> float:
    from scipy.spatial import cKDTree
    pos = np.mod(np.asarray(pos, float), cell_diag)
    pos = np.minimum(pos, cell_diag * (1 - 1e-9))
    d, _ = cKDTree(pos, boxsize=cell_diag).query(pos, k=2)
    return float(np.median(d[:, 1]))


def my_observables(frame, rmax: float, bins: int, rcn: float) -> dict:
    """g(r) histogram and coordination histogram, verifier's own code."""
    from scipy.spatial import cKDTree
    L = frame.cell_diag
    pos = np.mod(frame.pos, L)
    rho = len(pos) / float(np.prod(L))
    tree = cKDTree(pos, boxsize=L)
    pairs = tree.query_pairs(rmax, output_type="ndarray")
    d = pos[pairs[:, 1]] - pos[pairs[:, 0]]
    d -= L * np.round(d / L)
    r = np.linalg.norm(d, axis=1)
    h, e = np.histogram(r, np.linspace(0, rmax, bins + 1))
    shell = 4 / 3 * np.pi * (e[1:] ** 3 - e[:-1] ** 3)
    g = 2 * h / (len(pos) * rho * shell)
    counts = np.array([len(x) - 1 for x in tree.query_ball_point(pos, rcn)])
    hist = np.histogram(counts, np.arange(-0.5, 25.5))[0].astype(float)
    hist = hist / max(hist.sum(), 1.0)
    return {"gr": g, "rm": 0.5 * (e[1:] + e[:-1]), "cn_hist": hist}


def my_distance(o1: dict, o2: dict, rmin: float = 0.8) -> dict:
    """g(r) RMS distance and cn-histogram total variation (verifier's code)."""
    m = o1["rm"] > rmin
    gr_rms = float(np.sqrt(np.mean((o1["gr"][m] - o2["gr"][m]) ** 2)))
    n = min(len(o1["cn_hist"]), len(o2["cn_hist"]))
    cn_tv = float(0.5 * np.abs(o1["cn_hist"][:n] - o2["cn_hist"][:n]).sum())
    return {"gr_rms": gr_rms, "cn_tv": cn_tv}


def rigid_transform(frame, rng):
    """Rotation + translation + re-imaging + atom re-ordering (verifier's own)."""
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    t = rng.uniform(-5, 5, 3)
    pos = frame.pos @ R.T + t
    cell = frame.cell @ R.T
    pos = np.mod(pos @ np.linalg.inv(cell), 1.0) @ cell
    perm = rng.permutation(len(pos))
    from chaord.io.frames import Frame
    return Frame(pos=pos[perm], cell=cell,
                 symbols=[frame.symbols[i] for i in perm], pbc=frame.pbc)


def frame_to_pymatgen(frame):
    from pymatgen.core import Lattice, Structure
    return Structure(Lattice(frame.cell), frame.symbols, frame.pos,
                     coords_are_cartesian=True)


def format_program_text(program) -> str:
    from chaord.lang.fmt import format_program
    return format_program(program)


# ============================================================== bench access ==
_CASES: list | None = None


def bench_cases() -> list[dict]:
    """[{id, category, frames [Path], gt dict, dialect tuple, mode str}]"""
    global _CASES
    if _CASES is None:
        manifest = json.loads((BENCH / "index.json").read_text(encoding="utf-8"))
        cases = []
        for entry in manifest["cases"]:
            gt = json.loads((BENCH / entry["ground_truth"]).read_text(
                encoding="utf-8"))
            cases.append(dict(
                id=entry["id"], category=entry["category"],
                frames=[BENCH / f for f in entry["frames"]], gt=gt,
                dialect=tuple(gt.get("lift_dialect", ["core"])),
                mode=gt.get("lift_mode", "auto")))
        _CASES = cases
    return _CASES


def case_by_id(cid: str) -> dict:
    return next(c for c in bench_cases() if c["id"] == cid)


_LIFT_CACHE: dict = {}


def lift_all_bench_frames(frame_limit: int | None = None):
    """Lift every bench frame once (mode auto, dialect from ground truth).

    Cached: A6, A9 and A13 all consume the same loop."""
    if frame_limit in _LIFT_CACHE:
        return _LIFT_CACHE[frame_limit]
    from chaord.dialects import load_dialect
    from chaord.io.frames import read_frame
    from chaord.lift import lift_frame
    dialects: dict = {}
    records = []
    for case in bench_cases():
        dl = dialects.setdefault(case["dialect"], load_dialect(case["dialect"]))
        for k, path in enumerate(case["frames"]):
            if frame_limit is not None and len(records) >= frame_limit:
                _LIFT_CACHE[frame_limit] = records
                return records
            frame = read_frame(path)
            t0 = time.perf_counter()
            try:
                text = format_program_text(lift_frame(frame, dl))
                rec = dict(case=case["id"], frame=k, ok=True, text=text,
                           error=None, error_type=None,
                           seconds=time.perf_counter() - t0)
            except Exception as exc:                        # noqa: BLE001
                rec = dict(case=case["id"], frame=k, ok=False, text=None,
                           error=f"{type(exc).__name__}: {exc}",
                           error_type=type(exc).__name__,
                           seconds=time.perf_counter() - t0)
            records.append(rec)
    _LIFT_CACHE[frame_limit] = records
    return records


# ================================================================ criteria ====
def record(cid, name, passed, evidence, details=None):
    return dict(id=cid, name=name, passed=bool(passed), evidence=evidence,
                details=details or {})


# ---- A1 ----------------------------------------------------------------------
def check_a1(mutation: str | None = None, max_examples: int = 10000):
    """100% of spec examples parse; fmt idempotent on 10,000 generated programs
    (strategy reused from tests/test_fmt_property.py, never duplicated)."""
    from chaord.lang.fmt import format_program
    from chaord.lang.parser import parse_text
    from chaord.lang.ir import ir_equal
    from tests.test_fmt_property import st_program
    from hypothesis import HealthCheck, given, settings

    examples = sorted((ROOT / "spec" / "examples").glob("*.chaord"))
    items = [(p.name, p.read_text(encoding="utf-8")) for p in examples]
    if mutation == "malformed_example":
        items.insert(0, ("seeded_fault.chaord", "chaord 0.1\nsystem {\n"))
    parse_fail = []
    for name, text in items:
        try:
            parse_text(text)
        except Exception as exc:                            # noqa: BLE001
            parse_fail.append(f"{name}: {type(exc).__name__}")
    parsed_ok = len(items) - len(parse_fail)

    idem_err = None

    @settings(max_examples=max_examples, deadline=None, database=None,
              suppress_health_check=list(HealthCheck))
    @given(st_program)
    def laws(program):
        text = format_program(program)
        reparsed = parse_text(text)
        assert ir_equal(reparsed, program), text
        assert format_program(reparsed) == text

    try:
        laws()
    except Exception as exc:                                # noqa: BLE001
        idem_err = f"{type(exc).__name__}: {str(exc)[:120]}"

    ok = parsed_ok == len(items) and idem_err is None
    ev = (f"{parsed_ok}/{len(items)} spec examples parse; property laws "
          f"(parse==IR and fmt idempotent) hold on {max_examples} generated "
          f"programs")
    if parse_fail:
        ev += f"; parse failures: {'; '.join(parse_fail)}"
    if idem_err:
        ev += f"; property law failure: {idem_err}"
    return record("A1", "parse and format", ok, ev,
                  dict(examples=[p.name for p in examples],
                       parse_fail=parse_fail, max_examples=max_examples))


# ---- A2 ----------------------------------------------------------------------
CRYSTAL_CATEGORIES = {"crystals"}


def _rebuild_crystal(case):
    """Perfect frame rebuilt from the case's ground-truth parameters
    (chaord.build is allowed here: A2 measures lift invariance, not builder
    independence).  Solid solutions (several species on a unary prototype)
    are rebuilt as the host plus a seeded verifier-side relabelling."""
    from chaord.build.crystal import build_conventional
    from chaord.io.frames import Frame
    gt = case["gt"]["expected"]
    name = gt["prototype"]
    params = {"a": gt["a"]}
    if "c" in gt:
        params["c"] = gt["c"]
    counts = gt["counts"]
    relabel = None
    if len(counts) == 1:
        slots = (next(iter(counts)),)
    else:
        try:
            slots = _slots_from_counts(name, counts)
        except ValueError:
            slots = (next(iter(counts)),)
            relabel = sum(([s] * n for s, n in counts.items()), [])
    frame = build_conventional(name, params, slots,
                               tuple(gt.get("reps", (2, 2, 2))))
    if relabel is not None:
        rng = np.random.default_rng(_stable_seed(case["id"] + "|relabel"))
        rng.shuffle(relabel)
        frame = Frame(pos=frame.pos, cell=frame.cell, symbols=relabel,
                      pbc=frame.pbc)
    return frame


def _slots_from_counts(name: str, counts: dict) -> tuple:
    """Assign species to prototype slots so per-slot counts are consistent."""
    slot_counts = PROTO_TABLE[name][0]
    if len(counts) != len(slot_counts):
        raise ValueError(f"{len(counts)} species cannot sit on the "
                         f"{len(slot_counts)} slot(s) of {name}")
    for perm in itertools.permutations(list(counts)):
        if all(counts[perm[i]] % slot_counts[i] == 0 for i in range(len(perm))):
            if len({counts[perm[i]] // slot_counts[i]
                    for i in range(len(perm))}) == 1:
                return perm
    raise ValueError(f"cannot map {counts} onto {name} slots")


def check_a2(mutation: str | None = None, n_transforms: int = 2,
             case_filter: str | None = None):
    """Byte-identical text under rotation, translation, re-ordering and
    re-imaging on 100% of bench crystal cases, >= 2 transforms per case."""
    from chaord.dialects import load_dialect
    from chaord.io.frames import Frame
    from chaord.lift import lift_frame

    dialects: dict = {}
    rows, n_cases_ok = [], 0
    crystal_cases = [c for c in bench_cases()
                     if c["category"] in CRYSTAL_CATEGORIES
                     and (case_filter is None or case_filter in c["id"])]
    for case in crystal_cases:
        dl = dialects.setdefault(case["dialect"], load_dialect(case["dialect"]))
        frame = _rebuild_crystal(case)
        if mutation == "scale_lattice":
            # seeded fault: non-uniform 10% strain. The strain magnitude must
            # exceed the dialect's lift_symprec (0.25 A, sized to sit above
            # thermal jitter ~0.06*d_NN): smaller distortions are legitimately
            # absorbed by spglib idealisation, which is fit accuracy, not an
            # invariance violation.
            scale = np.array([1.0, 1.0, 1.10])
            frame = Frame(pos=frame.pos * scale,
                          cell=frame.cell * scale,
                          symbols=frame.symbols, pbc=frame.pbc)
        t_ref = format_program_text(lift_frame(frame, dl))
        if mutation:
            # a seeded fault is DETECTED when it changes the lifted value; the
            # invariance comparison itself is transform-vs-transform on one
            # frame, so fault detection must be scored against the clean lift.
            # Detection marks the case as failed (the check "caught" it).
            clean_frame = _rebuild_crystal(case)
            t_clean = format_program_text(lift_frame(clean_frame, dl))
            if t_ref != t_clean:
                rows.append((case["id"], False,
                             f" [fault detected: {next((a.strip() for a, b in zip(t_clean.splitlines(), t_ref.splitlines()) if a != b), '')[:60]}]"))
            else:
                rows.append((case["id"], True, " [fault NOT detected]"))
            continue
        rng = np.random.default_rng(_stable_seed(case["id"]))
        case_ok = True
        for _ in range(n_transforms):
            g = rigid_transform(frame, rng)
            try:
                same = format_program_text(lift_frame(g, dl)) == t_ref
                reason = ""
            except Exception as exc:                        # noqa: BLE001
                same, reason = False, f" [lift failed: {str(exc)[:70]}]"
            case_ok &= same
            rows.append((case["id"], same, reason))
        n_cases_ok += int(case_ok)
    n_transforms_total = len(rows)
    ok = all(s for _, s, _ in rows) and rows
    ev = (f"{n_cases_ok}/{len(crystal_cases)} crystal cases; "
          f"{sum(s for _, s, _ in rows)}/{n_transforms_total} rigid transforms "
          f"(rotation + translation + re-imaging + re-ordering, {n_transforms} "
          f"per case) lift to byte-identical text")
    bad = sorted({c for c, s, _ in rows if not s})
    if bad:
        ev += f"; differing: {bad}"
    return record("A2", "canonical invariance", ok, ev,
                  dict(rows=[dict(case=c, same=bool(s), reason=r)
                             for c, s, r in rows],
                       n_transforms=n_transforms))


# ---- A3 ----------------------------------------------------------------------
def check_a3(mutation: str | None = None, case_filter: str | None = None):
    """lift -> build -> lift gives identical text AND the rebuilt structure
    matches the original under pymatgen StructureMatcher(ltol 0.2, stol 0.3,
    angle_tol 5 deg) on 100% of bench crystal cases."""
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from chaord.build import build_program
    from chaord.dialects import load_dialect
    from chaord.io.frames import Frame
    from chaord.lang.parser import parse_text
    from chaord.lift import lift_frame

    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)
    dialects: dict = {}
    rows = []
    for case in bench_cases():
        if case["category"] not in CRYSTAL_CATEGORIES:
            continue
        if case_filter is not None and case_filter not in case["id"]:
            continue
        dl = dialects.setdefault(case["dialect"], load_dialect(case["dialect"]))
        frame = _rebuild_crystal(case)
        t1 = format_program_text(lift_frame(frame, dl))
        rebuilt = build_program(parse_text(t1), dl, rng=np.random.default_rng(5))
        if mutation == "displace_rebuilt":
            # seeded fault: displace a quarter of the rebuilt atoms by 0.6 A
            rng = np.random.default_rng(1)
            idx = rng.permutation(len(rebuilt))[: len(rebuilt) // 4]
            pos = rebuilt.pos.copy()
            pos[idx] += rng.normal(size=(len(idx), 3)) * 0.6
            rebuilt = Frame(pos=pos, cell=rebuilt.cell,
                            symbols=rebuilt.symbols, pbc=rebuilt.pbc)
        try:
            t2 = format_program_text(lift_frame(rebuilt, dl))
            text_ok = t2 == t1
            note = ""
            if not text_ok:
                diff = [l for l in t2.splitlines() if l not in t1.splitlines()]
                note = ("; rebuilt lift differs by: "
                        + " | ".join(diff[:3]) if diff else "")
        except Exception as exc:                            # noqa: BLE001
            text_ok, note = False, f"; rebuilt lift failed: {str(exc)[:70]}"
        geo_ok = bool(matcher.fit(frame_to_pymatgen(frame),
                                  frame_to_pymatgen(rebuilt)))
        rows.append((case["id"], text_ok, geo_ok, note))
    ok = all(t and g for _, t, g, _ in rows) and rows
    ev = (f"{sum(1 for _, t, _, _ in rows if t)}/{len(rows)} lift-build-lift texts "
          f"byte-identical; {sum(1 for _, _, g, _ in rows if g)}/{len(rows)} "
          f"StructureMatcher(ltol 0.2, stol 0.3, 5 deg) fits original vs "
          f"rebuilt")
    for c, t, g, note in rows:
        if not (t and g):
            ev += (f"; {c}: text {'ok' if t else 'DIFFERS'}, matcher "
                   f"{'ok' if g else 'no fit'}{note}")
    return record("A3", "exact round trip (ordered)", ok, ev,
                  dict(rows=[dict(case=c, text=bool(t), geometry=bool(g),
                                  note=n) for c, t, g, n in rows]))


# ---- A4 ----------------------------------------------------------------------
A4_HOSTS = [
    ("fcc-Cu", "fcc", {"a": 3.615}, ("Cu",), (3, 3, 3)),
    ("L12-NiAl", "L1_2", {"a": 3.572}, ("Ni", "Al"), (4, 4, 4)),
    ("NaCl", "rocksalt", {"a": 5.64}, ("Na", "Cl"), (3, 3, 3)),
]
A4_PLANS = {
    "fcc-Cu": [("vacancy", "V_Cu", 6), ("interstitial", "Cu_i", 4),
               ("frenkel", "frenkel_pair", 4)],
    "L12-NiAl": [("vacancy", "V_Ni", 3), ("antisite", "Al_Ni", 4),
                 ("interstitial", "Ni_i", 3), ("frenkel", "frenkel_pair", 3)],
    "NaCl": [("vacancy", "V_Na", 6), ("antisite", "Cl_Na", 4),
             ("interstitial", "Na_i", 3), ("frenkel", "frenkel_pair", 3)],
}
# 0.8 Tm anchor: the metal dialect documents thermal_test_amplitude = 0.06 d_NN
# as "0.8 Tm-ish"; room temperature follows the sqrt(T) law for a ~1358 K
# melting metal.  Test-side constants, not pass code.
ROOM_T_AMP_FACTOR = 0.06 * (300.0 / (0.8 * 1358.0)) ** 0.5


def _random_unit(rng, n):
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1)[:, None]


def _pick_separated(pos, syms, species, count, min_sep, rng, return_idx=False):
    """Indices of `count` mutually well-separated atoms of one species (raw
    numpy, no library helper)."""
    idx = [i for i, s in enumerate(syms) if s == species]
    picks: list[int] = []
    for i in rng.permutation(idx):
        if picks and np.min(np.linalg.norm(pos[picks] - pos[i], axis=1)) < min_sep:
            continue
        picks.append(int(i))
        if len(picks) == count:
            break
    if len(picks) < count:
        raise RuntimeError("could not place separated defects")
    if return_idx:
        return picks
    keep = np.zeros(len(pos), bool)
    keep[picks] = True
    return keep


def _plant_defects(frame, dtype: str, token: str, count: int, d_nn: float, rng):
    """Verifier-side planter: mutate a perfect crystal by hand (no builder)."""
    pos = frame.pos.copy()
    syms = list(frame.symbols)
    L = frame.cell_diag
    if dtype == "vacancy":
        site_sp = token[2:]
        drop = _pick_separated(pos, syms, site_sp, count, 2.0 * d_nn, rng)
        pos, syms = pos[~drop], [s for i, s in enumerate(syms) if not drop[i]]
    elif dtype == "interstitial":
        sp = token.split("_")[0]
        parents = rng.permutation(len(pos))[:count]
        offs = _random_unit(rng, count) * (0.60 * d_nn)
        pos = np.vstack([pos, np.mod(pos[parents] + offs, L)])
        syms += [sp] * count
    elif dtype == "antisite":
        on_sp, site_sp = token.split("_")
        cand = [i for i, s in enumerate(syms) if s == site_sp]
        for i in rng.permutation(cand)[:count]:
            syms[i] = on_sp
    elif dtype == "frenkel":
        sp = syms[0]
        picks = _pick_separated(pos, syms, sp, count, 2.0 * d_nn, rng,
                                return_idx=True)
        offs = _random_unit(rng, len(picks)) * (0.50 * d_nn)
        pos = np.vstack([pos, np.mod(pos[picks] + offs, L)])
        syms += [sp] * len(picks)
        keep = np.ones(len(pos), bool)
        keep[picks] = False
        pos, syms = pos[keep], [s for i, s in enumerate(syms) if keep[i]]
    else:
        raise ValueError(dtype)
    from chaord.io.frames import Frame
    return Frame(pos=pos, cell=frame.cell, symbols=syms, pbc=frame.pbc)


def _pr_for_cell(dtype: str, planted_token: str, planted_n: int,
                 detected: dict[str, int]):
    """Verifier's precision/recall over defect events for one planted cell.

    A frenkel_pair detection (species-less in the lift) pairs with any planted
    frenkel; an ungrouped V_X + X_i pair also counts as one found pair."""
    det = dict(detected)
    tp = fp = fn = 0
    if dtype == "frenkel":
        grouped = min(det.pop("frenkel_pair", 0), planted_n)
        vac = sum(n for t, n in det.items() if t.startswith("V_"))
        inter = sum(n for t, n in det.items() if t.endswith("_i"))
        pairs = grouped + min(vac, inter)
        tp = min(pairs, planted_n)
        fn = planted_n - tp
        fp = (max(pairs - tp, 0) + max(vac - inter, 0) + max(inter - vac, 0)
              + sum(n for t, n in det.items()
                    if not t.startswith("V_") and not t.endswith("_i")))
    else:
        relevant = det.pop(planted_token, 0)
        tp = min(relevant, planted_n)
        fn = planted_n - tp
        fp = relevant - tp + sum(det.values())
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    return prec, rec, dict(tp=tp, fp=fp, fn=fn)


def check_a4(mutation: str | None = None, temps=("room", "0.8Tm")):
    """Precision AND recall >= 0.95 for planted point defects of all four
    Kröger-Vink kinds (vacancy / interstitial / antisite / Frenkel) on 3 hosts,
    at room temperature and at the 0.8 Tm-equivalent thermal amplitude (metal
    dialect anchor 0.06 x d_NN).  Defects are planted by the verifier's own
    coordinate surgery; detections are parsed from the lifted text."""
    from chaord.build.crystal import build_conventional
    from chaord.dialects import load_dialect
    from chaord.io.frames import Frame
    from chaord.lift import lift_frame

    metal = load_dialect(("core", "metal"))
    hot_frac = float(metal.threshold("thermal_test_amplitude"))
    rows = []
    for host in A4_HOSTS:
        tag, name, params, slots, reps = host
        perfect = build_conventional(name, params, slots, reps)
        d_nn = median_nn_distance(perfect.pos, perfect.cell_diag)
        for dtype, token, n_planted in A4_PLANS[tag]:
            for temp in temps:
                seed = _stable_seed(f"{tag}|{dtype}|{temp}")
                rng = np.random.default_rng(seed)
                frame = _plant_defects(perfect, dtype, token, n_planted, d_nn, rng)
                amp = (hot_frac if temp == "0.8Tm" else ROOM_T_AMP_FACTOR) * d_nn
                hot = Frame(
                    pos=np.mod(frame.pos + rng.normal(size=frame.pos.shape) * amp,
                               frame.cell_diag),
                    cell=frame.cell, symbols=frame.symbols, pbc=frame.pbc)
                text = format_program_text(lift_frame(hot, metal, mode="defects"))
                detected = {t: int(c) for t, c in
                            re.findall(r"defect (\S+) count (\d+)", text)}
                if mutation == "false_defect" and dtype == "vacancy":
                    detected[token] = detected.get(token, 0) + 1  # seeded fault
                prec, rec, counts = _pr_for_cell(dtype, token, n_planted, detected)
                rows.append(dict(host=tag, type=dtype, temp=temp,
                                 planted=n_planted, detected=detected,
                                 precision=prec, recall=rec, **counts))
    ok = all(r["precision"] >= 0.95 and r["recall"] >= 0.95 for r in rows)
    n_p = sum(r["precision"] >= 0.95 for r in rows)
    n_r = sum(r["recall"] >= 0.95 for r in rows)
    worst_p = min(rows, key=lambda r: r["precision"])
    worst_r = min(rows, key=lambda r: r["recall"])
    ev = (f"{len(rows)} host x defect-type x temperature cells ({len(A4_HOSTS)} "
          f"hosts, all four K-V kinds); precision >= 0.95 in {n_p}/{len(rows)}, "
          f"recall >= 0.95 in {n_r}/{len(rows)}; worst precision "
          f"{worst_p['precision']:.2f} ({worst_p['host']}/{worst_p['type']}/"
          f"{worst_p['temp']}), worst recall {worst_r['recall']:.2f} "
          f"({worst_r['host']}/{worst_r['type']}/{worst_r['temp']})")
    return record("A4", "defect recovery", ok, ev, dict(matrix=rows))


# ---- A5 ----------------------------------------------------------------------
A5_CATEGORIES = {"fluid", "interface", "interfaces", "glass"}
A5_TARGETS = {"fluid": 0.90, "interface": 0.90, "interfaces": 0.90,
              "glass": 0.80}
A5_BUILD_TIMEOUT = 150          # seconds per rebuild; packing an
                                # over-jamming-density box can spin for many
                                # minutes and is recorded as a timeout

_REBUILD_CHILD = r"""
import json, sys, time
root = sys.argv[1]
cfg = json.loads(sys.argv[2])
sys.path.insert(0, root + "/src")
import numpy as np
from chaord.build import build_program
from chaord.lang.parser import parse_text
from chaord.dialects import load_dialect
text = sys.stdin.read()
t0 = time.perf_counter()
frame = build_program(parse_text(text), load_dialect(tuple(cfg["dialect"])),
                      rng=np.random.default_rng(7), physics=False)
np.savez(cfg["out"], pos=frame.pos, cell=frame.cell,
         symbols=np.array(frame.symbols, dtype="U8"))
print(json.dumps({"seconds": time.perf_counter() - t0, "n": len(frame)}))
"""


def _rebuild_in_subprocess(text: str, dialect_names, tmpdir: str, tag: str):
    """Run lift->build in a child process with a time budget; returns
    (frame|None, seconds, status_note)."""
    import subprocess as sp
    out = str(Path(tmpdir) / f"rebuild_{tag}.npz")
    cfg = json.dumps({"dialect": list(dialect_names), "out": out})
    try:
        r = sp.run([sys.executable, "-c", _REBUILD_CHILD, str(ROOT), cfg],
                   input=text, capture_output=True, text=True, cwd=ROOT,
                   timeout=A5_BUILD_TIMEOUT)
    except sp.TimeoutExpired:
        return None, A5_BUILD_TIMEOUT, "rebuild exceeded the time budget"
    if r.returncode != 0:
        return None, 0.0, (r.stderr.strip().splitlines() or ["build error"])[-1][:80]
    z = np.load(out)
    from chaord.io.frames import Frame
    frame = Frame(pos=z["pos"], cell=z["cell"],
                  symbols=[str(s) for s in z["symbols"]], pbc=(True, True, True))
    secs = 0.0
    try:
        secs = json.loads(r.stdout.strip().splitlines()[-1]).get("seconds", 0.0)
    except Exception:                                      # noqa: BLE001
        pass
    return frame, secs, ""


def check_a5(mutation: str | None = None, floors=None,
             case_filter: str | None = None):
    """Held-out observable distance <= 1.5x the noise floor on >= 90% of fluid
    and interface cases and >= 80% of amorphous cases, per case, against the
    noise floors on record (reports/noise_floors.json; from two frames of one
    reference MD simulation).  Synthetic packed frames have no floor: recorded
    as 'no-floor: skipped (synthetic frame)', never silently passed."""
    from chaord.dialects import load_dialect
    from chaord.io.frames import read_frame
    from chaord.lift import lift_frame

    if floors is None:
        floors = {}
        if NOISE_FLOORS.exists():
            floors = json.loads(NOISE_FLOORS.read_text(encoding="utf-8"))
    rows = []
    with tempfile.TemporaryDirectory() as td:
        for case in bench_cases():
            if case["category"] not in A5_CATEGORIES:
                continue
            if case_filter is not None and case_filter not in case["id"]:
                continue
            dl = load_dialect(case["dialect"])
            frame = read_frame(case["frames"][0])
            try:
                text = format_program_text(lift_frame(frame, dl))
            except Exception as exc:                        # noqa: BLE001
                rows.append(dict(case=case["id"], category=case["category"],
                                 status="lift-failed", note=str(exc)[:80]))
                continue
            rebuilt, secs, note = _rebuild_in_subprocess(
                text, case["dialect"], td, case["id"].replace("/", "_"))
            if rebuilt is None:
                rows.append(dict(case=case["id"], category=case["category"],
                                 status="build-failed", note=note))
                continue
            if mutation == "distort_rebuild":
                # seeded fault: expand the rebuilt box (g(r) shifts past the
                # floor)
                rebuilt = type(rebuilt)(
                    pos=rebuilt.pos * 1.10, cell=rebuilt.cell * 1.10,
                    symbols=rebuilt.symbols, pbc=rebuilt.pbc)
            rmax = float(dl.threshold("gr_rmax_fluid"))
            bins = int(dl.threshold("gr_bins_fluid"))
            rcn = float(dl.threshold("cn_cutoff_fluid"))
            dist = my_distance(my_observables(frame, rmax, bins, rcn),
                               my_observables(rebuilt, rmax, bins, rcn))
            floor = floors.get(case["id"])
            if floor is None:
                rows.append(dict(case=case["id"], category=case["category"],
                                 status="no-floor: skipped (synthetic frame)",
                                 rebuild_s=round(secs, 1), distance=dist))
                continue
            fl = floor.get("floor", floor) if isinstance(floor, dict) else floor
            ratios = {k: (dist[k] / fl[k] if fl.get(k) else None) for k in dist}
            passed = all(r is not None and r <= 1.5 for r in ratios.values())
            rows.append(dict(case=case["id"], category=case["category"],
                             status="pass" if passed else "fail",
                             rebuild_s=round(secs, 1), distance=dist,
                             floor=fl, ratios=ratios))
    with_floor = [r for r in rows if r["status"] in ("pass", "fail")]
    n_pass = sum(r["status"] == "pass" for r in with_floor)
    by_cat = {}
    for cat in A5_TARGETS:
        sub = [r for r in with_floor if r["category"] == cat]
        if sub:
            by_cat[cat] = (sum(r["status"] == "pass" for r in sub), len(sub),
                           A5_TARGETS[cat])
    skipped = [r for r in rows if r["status"].startswith("no-floor")]
    failed_other = [r for r in rows
                    if r["status"] in ("lift-failed", "build-failed")]
    ok = bool(with_floor) and all(
        n >= target * N for n, N, target in by_cat.values())
    parts = [f"pass rate {n_pass}/{len(with_floor)}-with-floor"]
    parts += [f"{cat}: {n}/{N} (target >= {t:.0%})"
              for cat, (n, N, t) in by_cat.items()]
    parts.append(f"{len(skipped)} cases without a floor on record: "
                 f"no-floor: skipped (synthetic frame)")
    if failed_other:
        parts.append(f"{len(failed_other)} could not round-trip: "
                     + "; ".join(f"{r['case']} {r['status']}"
                                 for r in failed_other))
    if not with_floor:
        parts.append("no case has a measured noise floor -> criterion not "
                     "demonstrated (populate reports/noise_floors.json from "
                     "S1 MD frames with tools/noise_floor.py)")
    return record("A5", "statistical round trip", ok, "; ".join(parts),
                  dict(rows=rows))


# ---- A6 ----------------------------------------------------------------------
def _conservation_row(case_id: str, frame_k: int, symbols, text: str) -> dict:
    frame_counts = count_species(symbols)
    parsed = parse_program_text(text)
    conserve = dict(parsed["conserve_atoms"])
    derived, how = derive_counts(parsed)
    counts_ok = conserve == frame_counts
    if derived is not None and "__expected_total__" in derived:
        expected_total = derived.pop("__expected_total__")
        deriv_ok = sum(frame_counts.values()) == expected_total
        how += (f" [total {sum(frame_counts.values())} == {expected_total}]"
                if deriv_ok else
                f" [total {sum(frame_counts.values())} != {expected_total}]")
        derived = None                # species split comes from conserve only
    elif derived is None:
        deriv_ok = True               # not derivable: two-way check only
    else:
        deriv_ok = derived == frame_counts
    if charge_checkable(symbols) and any(s in ION_CHARGES for s in symbols):
        charge_frame = frame_charge(symbols)
        charge_prog = sum(ION_CHARGES.get(s, 0) * n for s, n in conserve.items())
        charge_ok = charge_frame == charge_prog
        charge_note = (f"charge {charge_prog:+d} == frame {charge_frame:+d}"
                       if charge_ok else
                       f"charge program {charge_prog:+d} != frame {charge_frame:+d}")
    elif not charge_checkable(symbols):
        charge_ok, charge_note = True, ("charge skipped (polyatomic-ion elements "
                                        "present: element-wise charge undefined)")
    else:
        charge_ok, charge_note = True, "no ionic species"
    return dict(case=case_id, frame=frame_k, frame_counts=frame_counts,
                conserve=conserve, derived=derived, derivation=how,
                counts_ok=bool(counts_ok), derivation_ok=bool(deriv_ok),
                charge_ok=bool(charge_ok), charge_note=charge_note)


def check_a6(mutation: str | None = None, frame_limit: int | None = None):
    """Never drop an atom: for every bench frame lifted, the verifier counts
    species in the frame itself, parses the conserve line, and re-derives the
    counts from the region statements (composition / occupancy / molecules /
    defect net / residual) -- all independent counts must agree.  Total charge
    likewise whenever ionic species are present."""
    from chaord.io.frames import read_frame

    recs = lift_all_bench_frames(frame_limit)
    rows = []
    for rec in recs:
        if not rec["ok"]:
            continue
        case = case_by_id(rec["case"])
        symbols = list(read_frame(case["frames"][rec["frame"]]).symbols)
        if mutation == "drop_atom" and not rows:
            symbols = symbols[:-1]          # seeded fault: one atom vanished
        rows.append(_conservation_row(rec["case"], rec["frame"], symbols,
                                      rec["text"]))
    if mutation != "drop_atom":
        rows.append(_ionic_probe_row())
    ok = all(r["counts_ok"] and r["derivation_ok"] and r["charge_ok"]
             for r in rows) and rows
    bad = [r for r in rows
           if not (r["counts_ok"] and r["derivation_ok"] and r["charge_ok"])]
    n_derived = sum(1 for r in rows if r.get("derived") is not None
                    or "sites-only" in r["derivation"])
    ev = (f"{len(rows)} lifts checked: frame count == conserve line on "
          f"{sum(r['counts_ok'] for r in rows)}/{len(rows)}; verifier's own "
          f"region arithmetic (composition/occupancy/molecules/defect net/"
          f"residual) pins or cross-checks {n_derived} programs "
          f"(non-derivable shapes check two-way); charge consistent on "
          f"{sum(r['charge_ok'] for r in rows)}/{len(rows)}")
    if bad:
        ev += f"; violations: {[(r['case'], r['frame']) for r in bad][:6]}"
    return record("A6", "conservation", ok, ev, dict(rows=rows))


def _ionic_probe_row() -> dict:
    """Bench NaCl frame with one Cl removed by hand: counts and charge must
    stay exact (+1 on both sides)."""
    from chaord.dialects import load_dialect
    from chaord.io.frames import read_frame
    from chaord.lift import lift_frame

    nacl = case_by_id("crystals/rocksalt_nacl")
    frame = read_frame(nacl["frames"][0])
    try:
        dropped = _drop_one_cl(frame)
        text = format_program_text(lift_frame(dropped,
                                              load_dialect(("core", "metal"))))
        symbols = list(dropped.symbols)
        row = _conservation_row("probe: rocksalt_nacl minus one Cl", 0,
                                symbols, text)
        # both sides must read +1 (one Na+ without its Cl-)
        row["charge_ok"] = row["charge_ok"] and frame_charge(symbols) == 1
        row["charge_note"] += " [expected: charge +1]"
    except Exception as exc:                                # noqa: BLE001
        row = dict(case="probe: rocksalt_nacl minus one Cl", frame=0,
                   counts_ok=False, derivation_ok=False, charge_ok=False,
                   derivation=f"probe failed: {exc}", charge_note="n/a")
    return row


def _drop_one_cl(frame):
    idx = [i for i, s in enumerate(frame.symbols) if s == "Cl"][0]
    keep = [i for i in range(len(frame)) if i != idx]
    from chaord.io.frames import Frame
    return Frame(pos=frame.pos[keep], cell=frame.cell,
                 symbols=[frame.symbols[i] for i in keep], pbc=frame.pbc)


# ---- A7 ----------------------------------------------------------------------
def check_a7(mutation: str | None = None, frames=(0, 1, 2, 3, 4)):
    """Per-atom phase labels >= 95% correct against planted ground truth on the
    interface cases (chaord.lift.segment labels), outside the interface band."""
    from chaord.dialects import load_dialect
    from chaord.io.frames import read_frame
    from chaord.lift.segment import phase_labels

    plans = [
        ("interface/lj_solid_liquid", ("core", "lj")),
        ("interfaces/cu_water", ("core", "metal")),
    ]
    rows = []
    for cid, names in plans:
        case = case_by_id(cid)
        gt = case["gt"]
        boundary = next((f.get("boundary_z") for f in gt["frames"]
                         if "boundary_z" in f),
                        gt["expected"].get("boundary_z"))
        dl = load_dialect(names)
        for k in frames:
            frame = read_frame(case["frames"][k])
            L = frame.cell_diag
            if cid.endswith("cu_water"):
                truth = np.array([s != "Cu" for s in frame.symbols])
            else:
                truth = frame.pos[:, 2] > boundary
            labels = phase_labels(frame, dl)
            if mutation == "flip_labels" and k == frames[0]:
                # seeded fault: corrupt a fifth of the labels
                rng = np.random.default_rng(0)
                flip = rng.permutation(len(labels))[: len(labels) // 5]
                labels = labels.copy()
                labels[flip] = ~labels[flip]
            d_nn = median_nn_distance(frame.pos, L)
            z = frame.pos[:, 2]
            band = 2.0 * d_nn
            core = ((np.abs(z - boundary) > band) & (z > band)
                    & (z < L[2] - band))
            acc = float((labels[core] == (~truth[core])).mean())
            rows.append(dict(case=cid, frame=k, accuracy=acc,
                             core_atoms=int(core.sum())))
    ok = all(r["accuracy"] >= 0.95 for r in rows) and rows
    worst = min(rows, key=lambda r: r["accuracy"])
    ev = (f"{sum(r['accuracy'] >= 0.95 for r in rows)}/{len(rows)} interface "
          f"frames (2 cases x {len(frames)} frames; interface band of 2 x d_NN "
          f"excluded, d_NN = verifier's median nearest-neighbour distance) "
          f"labelled >= 95% correct; worst {worst['accuracy']:.3f} "
          f"({worst['case']} frame {worst['frame']})")
    return record("A7", "phase segmentation", ok, ev, dict(rows=rows))


# ---- A8 ----------------------------------------------------------------------
def check_a8(mutation: str | None = None):
    """Reactive census exact on independently constructed frames: an H2O/OH/H
    mixture built by the verifier in raw numpy (no chaord builder involved)."""
    from chaord.build.molecules import molecule_census
    from chaord.dialects import load_dialect

    mol = load_dialect(("core", "molecular"))
    # (truth counts, counts actually planted in the frame, box edge, seed)
    plans = [
        ({"H2O": 40, "OH": 9, "H": 9}, {"H2O": 40, "OH": 9, "H": 9}, 20.0, 101),
        ({"H2O": 25, "OH": 5, "H": 7}, {"H2O": 25, "OH": 5, "H": 7}, 18.0, 202),
    ]
    if mutation == "extra_oh":
        # seeded fault: one more OH sits in the frame than the truth record
        # says -- the census must disagree, or the check fails
        truth, planted, box, seed = plans[0]
        planted = dict(planted)
        planted["OH"] += 1
        plans[0] = (truth, planted, box, seed)
    rows = []
    for i, (truth, planted, box, seed) in enumerate(plans):
        frame = _build_mixture_frame(planted, box, seed)
        census = dict(molecule_census(frame, mol))
        # census keys are canonical formulas (C, H, then alphabetical):
        # OH is counted under its canonical key HO
        canon = lambda d: {("HO" if k == "OH" else k): v for k, v in d.items()}
        expected, census = canon(truth), canon(census)
        exact = census == expected
        rows.append(dict(plan=i, expected=expected, census=census,
                         exact=bool(exact), n_atoms=len(frame)))
    ok = all(r["exact"] for r in rows)
    ev = "; ".join(
        f"plan {r['plan']}: census {r['census']} == planted {r['expected']} "
        f"({r['n_atoms']} atoms, coordinates written directly in numpy)"
        for r in rows)
    return record("A8", "reactive census", ok, ev, dict(rows=rows))


def _build_mixture_frame(counts: dict, box: float, seed: int):
    """Verifier-side molecular construction: raw numpy coordinates on a
    jittered grid; own water/OH geometry (O-H 0.958 A, H-O-H 104.52 deg) and
    own random rotations."""
    rng = np.random.default_rng(seed)
    L = np.array([box] * 3)
    total = sum(counts.values())
    g = int(np.ceil(total ** (1 / 3))) + 1
    spacing = box / g
    centers = np.array(list(itertools.product(np.arange(g), repeat=3))) * spacing
    centers = centers + spacing / 2
    centers = centers[rng.permutation(len(centers))[:total]]
    centers = centers + rng.uniform(-spacing * 0.12, spacing * 0.12,
                                    centers.shape)
    pos_list, syms = [], []
    OH, ang = 0.958, np.deg2rad(104.52)     # published water geometry
    i = 0
    for name, n in counts.items():
        for _ in range(n):
            c = centers[i]
            i += 1
            R = _random_rotation(rng)
            if name == "H2O":
                rel = np.array([
                    [0.0, 0.0, 0.0],
                    [OH * np.sin(ang / 2), 0, OH * np.cos(ang / 2)],
                    [-OH * np.sin(ang / 2), 0, OH * np.cos(ang / 2)]])
                syms += ["O", "H", "H"]
            elif name in ("OH", "HO"):
                rel = np.array([[0.0, 0.0, 0.0], [OH, 0, 0]])
                syms += ["O", "H"]
            elif name == "H":
                rel = np.zeros((1, 3))
                syms += ["H"]
            else:
                raise ValueError(name)
            pos_list.append(rel @ R.T + c)
    from chaord.io.frames import Frame
    return Frame(pos=np.mod(np.vstack(pos_list), L), cell=np.diag(L),
                 symbols=syms, pbc=(True, True, True))


def _random_rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


# ---- A9 ----------------------------------------------------------------------
def check_a9(mutation: str | None = None, cases_override=None):
    """Program <= 2% of the coordinate file for systems of >= 1,000 atoms;
    every bench case measured, worst value gated on the >= 1,000-atom ones."""
    from chaord.io.frames import write_frame

    measurements = []
    if cases_override is not None:
        for m in cases_override:
            measurements.append(dict(
                case=m["case"], n_atoms=m["n_atoms"],
                prog_bytes=m["prog_bytes"], xyz_bytes=m["xyz_bytes"],
                ratio=100.0 * m["prog_bytes"] / m["xyz_bytes"],
                source=m.get("source", "override")))
    else:
        by_case: dict = {}
        for rec in lift_all_bench_frames():
            if rec["ok"]:
                by_case.setdefault(rec["case"], rec)
        with tempfile.TemporaryDirectory() as td:
            for cid, rec in sorted(by_case.items()):
                case = case_by_id(cid)
                frame = _read_case_frame(case)
                q = Path(td) / f"{cid.replace('/', '_')}.extxyz"
                write_frame(q, frame)
                xyz_bytes = q.stat().st_size
                prog_bytes = len(rec["text"].encode("utf-8"))
                measurements.append(dict(
                    case=cid, n_atoms=len(frame), prog_bytes=prog_bytes,
                    xyz_bytes=xyz_bytes,
                    ratio=100.0 * prog_bytes / xyz_bytes, source="bench"))
            supp = _tiled_supplementary(td)
            if supp is not None:
                measurements.append(supp)
    if mutation == "inflate_program":
        # seeded fault: a bloated program text (comment padding counts too)
        for m in measurements:
            m["prog_bytes"] += 20000
            m["ratio"] = 100.0 * m["prog_bytes"] / m["xyz_bytes"]
    raw = [m for m in measurements if m["source"] == "bench"]
    pool = raw or measurements
    big_pool = [m for m in pool if m["n_atoms"] >= 1000]
    worst = max(pool, key=lambda m: m["ratio"]) if pool else None
    if big_pool:
        worst_big = max(big_pool, key=lambda m: m["ratio"])
        ok = worst_big["ratio"] <= 2.0
        ev = (f"{len(big_pool)} measured systems with >= 1,000 atoms "
              f"({'bench frames' if raw else 'override cases'}); worst ratio "
              f"{worst_big['ratio']:.2f}% ({worst_big['case']}, "
              f"{worst_big['n_atoms']} atoms); per-case table in details")
    else:
        ok = False
        n_max = max((m["n_atoms"] for m in raw), default=0)
        ev = (f"0 bench frames reach 1,000 atoms"
              + (f" (largest raw bench frame: {n_max} atoms)" if raw else "")
              + f" -> criterion cannot be demonstrated on this bench; worst "
                f"measured ratio {worst['ratio']:.2f}% ({worst['case']}) "
                f"across {len(raw)} cases"
              + ("; supplementary tiled >= 1,000-atom measurement in details "
                 "(not a raw bench frame, not counted)" if raw else
                 " (override mode: synthetic measurements only)"))
    return record("A9", "compression", ok, ev, dict(measurements=measurements))


def _read_case_frame(case, k=0):
    from chaord.io.frames import read_frame
    return read_frame(case["frames"][k])


def _tiled_supplementary(td: str):
    """The l12_ni3al defect case tiled 2x2 (1,012 atoms): supplementary
    measurement clearly labelled as not a raw bench frame."""
    from chaord.dialects import load_dialect
    from chaord.io.frames import Frame, write_frame
    from chaord.lift import lift_frame

    case = case_by_id("defects/l12_ni3al_vac_antisite")
    frame = _read_case_frame(case)
    L = frame.cell_diag
    pos = np.vstack([frame.pos, frame.pos + [L[0], 0, 0],
                     frame.pos + [0, L[1], 0], frame.pos + [L[0], L[1], 0]])
    cell = frame.cell.copy()
    cell[0] = cell[0] * 2
    cell[1] = cell[1] * 2
    tiled = Frame(pos=pos, cell=cell, symbols=frame.symbols * 4,
                  pbc=frame.pbc)
    try:
        text = format_program_text(
            lift_frame(tiled, load_dialect(("core", "metal")), mode="defects"))
    except Exception:                                      # noqa: BLE001
        return None
    q = Path(td) / "tiled.extxyz"
    write_frame(q, tiled)
    xyz_bytes = q.stat().st_size
    prog_bytes = len(text.encode("utf-8"))
    return dict(case="defects/l12_ni3al_vac_antisite tiled 2x2 "
                     "(supplementary, not a raw bench frame)",
                n_atoms=len(tiled), prog_bytes=prog_bytes,
                xyz_bytes=xyz_bytes, ratio=100.0 * prog_bytes / xyz_bytes,
                source="tiled-supplementary")


# ---- A10 ---------------------------------------------------------------------
def check_a10(mutation: str | None = None):
    """Same input -> byte-identical lift text; same program + seed + backend ->
    identical coordinates."""
    from chaord.build import build_program
    from chaord.dialects import load_dialect
    from chaord.io.frames import read_frame
    from chaord.lang.parser import parse_text
    from chaord.lift import lift_frame

    metal = load_dialect(("core", "metal"))
    case = case_by_id("defects/l12_ni3al_vac_antisite")
    frame = read_frame(case["frames"][0])
    t1 = format_program_text(lift_frame(frame, metal))
    t2 = format_program_text(lift_frame(frame, metal))
    text_ok = t1 == t2

    prog = parse_text(t1)
    b1 = build_program(prog, metal, rng=np.random.default_rng(42))
    b2 = build_program(prog, metal, rng=np.random.default_rng(42))
    if mutation == "perturb_build":
        b2.pos = b2.pos + 1e-3             # seeded fault: coordinates differ
    coord_ok = bool(np.array_equal(b1.pos, b2.pos))
    ok = text_ok and coord_ok
    ev = (f"repeated lift of bench defect frame byte-identical: {text_ok}; "
          f"same-seed builds identical coordinates: {coord_ok}")
    return record("A10", "determinism", ok, ev,
                  dict(text_ok=text_ok, coord_ok=coord_ok))


# ---- A11 ---------------------------------------------------------------------
def check_a11(mutation: str | None = None, n_atoms: int = 100_000,
              elapsed_override: float | None = None):
    """Lift 100,000 atoms in <= 120 s on one CPU core.  Input, honestly
    stated: an fcc lattice at rho = 0.85 (LJ reduced units) with Gaussian
    jitter of 0.10 x a -- a thermally disordered single-phase frame, NOT an
    equilibrated MD liquid."""
    from chaord.build.crystal import build_conventional
    from chaord.dialects import load_dialect
    from chaord.io.frames import Frame
    from chaord.lift import lift_frame

    rho = 0.85
    a = (4.0 / rho) ** (1.0 / 3.0)
    rng = np.random.default_rng(20260928)
    reps = _balanced_reps(n_atoms // 4)
    base = build_conventional("fcc", {"a": a}, ("X",), reps)
    assert len(base) == n_atoms, f"{len(base)} != {n_atoms}"
    amp = 0.10 * a
    pos = np.mod(base.pos + rng.normal(size=base.pos.shape) * amp,
                 base.cell_diag)
    frame = Frame(pos=pos, cell=base.cell, symbols=list(base.symbols),
                  pbc=(True, True, True))
    dl = load_dialect(("core", "lj"))
    t0 = time.perf_counter()
    program = lift_frame(frame, dl, mode="fluid")
    elapsed = time.perf_counter() - t0
    if elapsed_override is not None:
        elapsed = elapsed_override
    text = format_program_text(program)
    stated = f"conserve atoms X {n_atoms}" in text
    ok = elapsed <= 120.0
    ev = (f"lift_frame(mode=fluid) on {len(frame):,} atoms took {elapsed:.1f} s "
          f"(target <= 120 s); input = fcc lattice at rho 0.85 with 0.10 x a "
          f"jitter (thermally disordered, not equilibrated MD); program states "
          f"the exact count: {'yes' if stated else 'NO'}")
    return record("A11", "speed", ok, ev, dict(elapsed=elapsed,
                                               n_atoms=len(frame)))


def _balanced_reps(n_cells: int) -> tuple[int, int, int]:
    """Reps (rx, ry, rz) with rx*ry*rz = n_cells, as cubic as possible."""
    best = None
    for rx in range(1, n_cells + 1):
        if n_cells % rx:
            continue
        rest = n_cells // rx
        for ry in range(1, int(rest ** 0.5) + 1):
            if rest % ry:
                continue
            rz = rest // ry
            key = (max(rx, ry, rz), rx * ry + ry * rz + rx * rz)
            if best is None or key < best[0]:
                best = (key, (rx, ry, rz))
    return best[1]


# ---- A12 ---------------------------------------------------------------------
def check_a12(mutation: str | None = None):
    """100% of four seeded static errors caught: lattice mismatch, impossible
    density, charge imbalance (conserve charge contradicting the ion census),
    and a 0.1 sigma atom overlap.  The mutation stubs one checker out and the
    criterion must then fail (the seeded error goes uncaught)."""
    from chaord.build import build_program
    from chaord.check.statics import charge_check, overlap_check
    from chaord.dialects import load_dialect
    from chaord.io.frames import Frame
    from chaord.lang.api import load
    from chaord.lang.errors import ChaordError
    from chaord.lang.ir import (
        GeoChain, Name, PhysicsBlock, Program as IRProgram, Quantity,
        RegionBlock, ShAll, SpecDef, SpeciesBlock, Statement, SystemBlock,
    )

    metal = load_dialect(("core", "metal"))
    mol = load_dialect(("core", "molecular"))
    results = {}

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "mismatch.chaord"
        p.write_text("chaord 0.1\n\nsystem {\n  cell 7.5 7.5 7.5\n  pbc xyz\n}\n\n"
                     "physics {\n  backend eam\n}\n\n"
                     "crystal bulk : all {\n  lattice fcc\n  a 3.615 A\n}\n")
        try:
            build_program(load(p), metal)
            results["lattice_mismatch"] = "NOT caught"
        except ChaordError:
            results["lattice_mismatch"] = "caught"

        p = Path(td) / "density.chaord"
        p.write_text("chaord 0.1\n\nsystem {\n  cell 10 10 10\n  pbc xyz\n}\n\n"
                     "physics {\n  backend classical\n}\n\n"
                     "liquid water : all {\n  molecules H2O 200\n"
                     "  state density 1.0 g/cm3\n}\n")
        try:
            build_program(load(p), mol)
            results["impossible_density"] = "NOT caught"
        except ChaordError:
            results["impossible_density"] = "caught"

    prog_ion = IRProgram(
        version="0.1", dialects=["core", "molecular"],
        blocks=[
            SpeciesBlock(defs=[SpecDef(k="ion", name="Na+"),
                               SpecDef(k="ion", name="Cl-")]),
            SystemBlock(statements=[
                Statement(kind="build", key="pbc", values=[Name(text="xyz")]),
                Statement(kind="conserve", key="charge",
                          values=[Quantity(num="0")])]),
            PhysicsBlock(statements=[
                Statement(kind="build", key="backend",
                          values=[Name(text="classical")])]),
            RegionBlock(phase="liquid", name="electrolyte",
                        geometry=GeoChain(parts=[ShAll()], ops=[]),
                        statements=[
                            Statement(kind="build", key="molecules",
                                      values=[Name(text="Na+"),
                                              Quantity(num="10"),
                                              Name(text="Cl-"),
                                              Quantity(num="9")])]),
        ])
    syms = ["Na"] * 10 + ["Cl"] * 9
    pos = np.array([[3.0 * (i % 5), 3.0 * (i // 5), 0.0] for i in range(19)])
    frame_ion = Frame(pos=pos, cell=np.diag([15.0] * 3), symbols=syms,
                      pbc=(True, True, True))
    assert frame_charge(syms) == 1, "verifier arithmetic: 10(+1) + 9(-1) = +1"
    if mutation == "stub_charge":
        cc = type("Stub", (), {})()          # seeded fault: checker missing
        cc.passed, cc.detail = True, "stubbed out"
    else:
        cc = charge_check(prog_ion, frame_ion, mol)
    results["charge_imbalance"] = "caught" if not cc.passed else "NOT caught"

    core = load_dialect(("core",))
    frame_ov = Frame(pos=np.array([[5.0, 5.0, 5.0], [5.1, 5.0, 5.0]]),
                     cell=np.diag([10.0] * 3), symbols=["X", "X"],
                     pbc=(True, True, True))
    if mutation == "stub_overlap":
        oc = type("Stub", (), {})()          # seeded fault: checker missing
        oc.passed, oc.detail = True, "stubbed out"
    else:
        oc = overlap_check(frame_ov, core)
    results["overlap_0.1sigma"] = "caught" if not oc.passed else "NOT caught"

    uncaught = [k for k, v in results.items() if v != "caught"]
    ok = not uncaught
    ev = "; ".join(f"{k}: {v}" for k, v in results.items())
    if uncaught:
        ev += f" -- uncaught: {uncaught}"
    return record("A12", "static checks", ok, ev, dict(results=results))


# ---- A13 ---------------------------------------------------------------------
def check_a13(mutation: str | None = None, frame_limit: int | None = None):
    """Zero unhandled exceptions lifting every frame in the benchmark;
    unexplained atoms go to residual."""
    recs = list(lift_all_bench_frames(frame_limit))
    if mutation == "poison_frame":
        recs.append(dict(case="mutation/seeded_crash", frame=0, ok=False,
                         error="ValueError: NaN positions", error_type="ValueError",
                         seconds=0.0))
    n_ok = sum(r["ok"] for r in recs)
    refused = [r for r in recs if not r["ok"] and r["error_type"] == "ChaordError"]
    crashed = [r for r in recs
               if not r["ok"] and r["error_type"] != "ChaordError"]
    ok_lifts = [r for r in recs if r["ok"]]
    n_residual_atoms = sum(len(parse_program_text(r["text"])["residual_atoms"])
                           for r in ok_lifts)
    ok = n_ok == len(recs)
    ev = (f"{n_ok}/{len(recs)} bench frames lift without any exception; "
          f"{n_residual_atoms} atoms placed in residual blocks across "
          f"successful lifts")
    if refused:
        ev += (f"; {len(refused)} frames refused with ChaordError (a lift "
               f"refusal is still a failed lift for this criterion): "
               + "; ".join(f"{r['case']}[{r['frame']}] {r['error'][:60]}"
                           for r in refused[:8])
               + ("..." if len(refused) > 8 else ""))
    if crashed:
        ev += (f"; {len(crashed)} non-ChaordError failures: "
               + "; ".join(f"{r['case']}[{r['frame']}] {r['error'][:60]}"
                           for r in crashed[:8]))
    return record("A13", "no crashes", ok, ev,
                  dict(failures=[dict(case=r["case"], frame=r["frame"],
                                      error=r["error"])
                                 for r in recs if not r["ok"]],
                       n_frames=len(recs)))


# ---- A14 ---------------------------------------------------------------------
def check_a14(mutation: str | None = None, reference_text: str | None = None):
    """Every statement key defined by a dialect YAML has a reference entry
    (text-coverage check against docs/reference.md); docs/reference_generated.md
    is (re)generated with one line + example per key."""
    import yaml

    by_key: dict[str, list[str]] = {}
    for p in sorted((ROOT / "src" / "chaord" / "dialects").glob("*.yaml")):
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        for scope, keys in (data.get("keys") or {}).items():
            for key in keys:
                by_key.setdefault(key, []).append(f"{p.stem}:{scope}")

    examples = {}
    for p in sorted((ROOT / "spec" / "examples").glob("*.chaord")):
        for line in p.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^\s*((?:(?:state|constrain|assert|history|conserve)"
                         r"\s+)?)([A-Za-z_][A-Za-z0-9_]*)\b", line)
            if not m:
                continue
            kind, key = m.group(1).strip(), m.group(2)
            # the leading kind word is itself a dialect key in some scopes
            # (e.g. glass region keys include `state` and `history`)
            for token in ((kind, key) if kind else (key,)):
                if token not in examples:
                    examples[token] = (line.strip(), p.name)

    ref = reference_text if reference_text is not None else (
        REFERENCE_MD.read_text(encoding="utf-8") if REFERENCE_MD.exists() else "")
    if mutation == "hide_key":
        ref = re.sub(r"(?<![A-Za-z_])epsilon(?![A-Za-z_])", "REDACTED", ref)

    lines = ["# Reference: generated key inventory (do not edit by hand)", "",
             "One row per statement key declared in `src/chaord/dialects/*.yaml`, "
             "with a passing example from `spec/examples/` where one exists and "
             "a coverage verdict against `docs/reference.md`.  Regenerated by "
             "`python tools/acceptance.py` (criterion A14).", "",
             "| key | defined in (dialect:scope) | example | example source | "
             "reference.md entry? |",
             "| --- | --- | --- | --- | --- |"]
    missing = []
    for key in sorted(by_key):
        srcs = ", ".join(sorted(set(by_key[key])))
        ex, src = examples.get(key, ("(no example in spec/examples/ - add one)", "-"))
        covered = bool(re.search(r"(?<![A-Za-z_])" + re.escape(key)
                                 + r"(?![A-Za-z_])", ref))
        if not covered:
            missing.append(key)
        lines.append(f"| `{key}` | {srcs} | `{ex}` | {src} | "
                     f"{'yes' if covered else 'MISSING'} |")
    REFERENCE_GENERATED.parent.mkdir(exist_ok=True)
    REFERENCE_GENERATED.write_text("\n".join(lines) + "\n", encoding="utf-8")

    sketch_ok, sketch_note = True, "skipped (tool unavailable)"
    sketch = ROOT / "tools" / "sketch_check.py"
    if sketch.exists():
        ex_paths = sorted((ROOT / "spec" / "examples").glob("*.chaord"))
        r = subprocess.run([sys.executable, str(sketch)]
                           + [str(p) for p in ex_paths],
                           capture_output=True, text=True, cwd=ROOT)
        sketch_ok = r.returncode == 0
        sketch_note = "pass" if sketch_ok else "FAIL"

    n_with_example = sum(1 for k in by_key if k in examples)
    ok = not missing and sketch_ok
    ev = (f"{len(by_key)} distinct keys across dialect YAMLs; reference.md "
          f"covers {len(by_key) - len(missing)}/{len(by_key)}"
          + (f" (missing entries: {', '.join(missing)})" if missing else "")
          + f"; example found for {n_with_example}/{len(by_key)} keys "
            f"(spec/examples; sketch_check {sketch_note}); generated "
            f"docs/reference_generated.md")
    return record("A14", "documentation", ok, ev,
                  dict(missing=missing, keys=sorted(by_key),
                       generated=str(REFERENCE_GENERATED)))


# ================================================================== runner ====
CHECKS = {
    "A1": check_a1, "A2": check_a2, "A3": check_a3, "A4": check_a4,
    "A5": check_a5, "A6": check_a6, "A7": check_a7, "A8": check_a8,
    "A9": check_a9, "A10": check_a10, "A11": check_a11, "A12": check_a12,
    "A13": check_a13, "A14": check_a14,
}


def _md_report(results) -> str:
    out = ["# Acceptance details", "",
           "Per-case numbers behind `reports/acceptance.json`.  Written by "
           "`tools/acceptance.py`; the verification logic (counting, parsing, "
           "arithmetic) is independent of the chaord check helpers.", ""]
    for r in results:
        out += [f"## {r['id']} {r['name']} — {'PASS' if r['passed'] else 'FAIL'}",
                "", f"**Evidence:** {r['evidence']}", ""]
        d = r.get("details") or {}
        if r["id"] == "A4" and d.get("matrix"):
            out += ["| host | type | temp | planted | detected | P | R | tp/fp/fn |",
                    "| --- | --- | --- | --- | --- | --- | --- | --- |"]
            for m in d["matrix"]:
                det = ",".join(f"{k}:{v}" for k, v in m["detected"].items()) or "-"
                out.append(f"| {m['host']} | {m['type']} | {m['temp']} | "
                           f"{m['planted']} | {det} | {m['precision']:.2f} | "
                           f"{m['recall']:.2f} | {m['tp']}/{m['fp']}/{m['fn']} |")
        if r["id"] == "A5" and d.get("rows"):
            out += ["| case | category | status | distances | ratios |",
                    "| --- | --- | --- | --- | --- |"]
            for m in d["rows"]:
                dist = m.get("distance") or {}
                ratios = m.get("ratios") or {}
                d_txt = ", ".join(f"{k}={v:.3f}" for k, v in dist.items())
                r_txt = ", ".join(f"{k}={v:.2f}" for k, v in ratios.items())
                note = f" ({m['note']})" if m.get("note") else ""
                out.append(f"| {m['case']} | {m['category']} | {m['status']}"
                           f"{note} | {d_txt} | {r_txt} |")
        if r["id"] == "A2" and d.get("rows"):
            seen = {}
            for m in d["rows"]:
                seen.setdefault(m["case"], []).append(m["same"])
            out += ["| case | transforms identical |", "| --- | --- |"]
            for c, v in seen.items():
                out.append(f"| {c} | {sum(v)}/{len(v)} |")
        if r["id"] == "A3" and d.get("rows"):
            out += ["| case | lift-build-lift text | StructureMatcher |",
                    "| --- | --- | --- |"]
            for m in d["rows"]:
                out.append(f"| {m['case']} | {m['text']} | {m['geometry']} |")
        if r["id"] == "A6" and d.get("rows"):
            out += ["| case | frame | frame counts | conserve | derivation | "
                    "derived | ok | charge |",
                    "| --- | --- | --- | --- | --- | --- | --- | --- |"]
            for m in d["rows"]:
                derived = ("-" if m.get("derived") is None else m["derived"])
                ok_txt = m["counts_ok"] and m["derivation_ok"]
                out.append(f"| {m['case']} | {m['frame']} | {m['frame_counts']} | "
                           f"{m['conserve']} | {m['derivation']} | {derived} | "
                           f"{ok_txt} | {m['charge_note']} |")
        if r["id"] == "A7" and d.get("rows"):
            out += ["| case | frame | core atoms | accuracy |",
                    "| --- | --- | --- | --- |"]
            for m in d["rows"]:
                out.append(f"| {m['case']} | {m['frame']} | {m['core_atoms']} | "
                           f"{m['accuracy']:.3f} |")
        if r["id"] == "A8" and d.get("rows"):
            out += ["| plan | expected | census | exact |", "| --- | --- | --- | --- |"]
            for m in d["rows"]:
                out.append(f"| {m['plan']} | {m['expected']} | {m['census']} | "
                           f"{m['exact']} |")
        if r["id"] == "A9" and d.get("measurements"):
            out += ["| case | atoms | program B | extxyz B | ratio % | source |",
                    "| --- | --- | --- | --- | --- | --- |"]
            for m in d["measurements"]:
                out.append(f"| {m['case']} | {m['n_atoms']} | {m['prog_bytes']} | "
                           f"{m['xyz_bytes']} | {m['ratio']:.2f} | {m['source']} |")
        if r["id"] == "A13" and d.get("failures"):
            out += ["| case | frame | error |", "| --- | --- | --- |"]
            for m in d["failures"]:
                out.append(f"| {m['case']} | {m['frame']} | {m['error'][:110]} |")
        out.append("")
    return "\n".join(out)


def run_all(only: list[str] | None = None, quick: bool = False) -> list[dict]:
    order = [f"A{i}" for i in range(1, 15)]
    if only:
        order = [c.strip().upper() for c in only if c.strip().upper() in CHECKS]
    results = []
    for cid in order:
        kwargs = {}
        if quick and cid == "A1":
            kwargs["max_examples"] = 200
        try:
            results.append(CHECKS[cid](**kwargs))
        except Exception as exc:                            # noqa: BLE001
            results.append(record(cid, f"criterion {cid}", False,
                                  f"check crashed: {type(exc).__name__}: {exc}"))
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default=str(ROOT / "reports" / "acceptance.json"))
    ap.add_argument("--details",
                    default=str(ROOT / "reports" / "acceptance_details.md"))
    ap.add_argument("--only", default=None, help="comma-separated criterion ids")
    ap.add_argument("--quick", action="store_true",
                    help="reduced property budget (harness debugging only)")
    args = ap.parse_args()

    results = run_all(None if not args.only else args.only.split(","),
                      quick=args.quick)
    for r in results:
        print(f"[{'PASS' if r['passed'] else 'FAIL'}] {r['id']} {r['name']}: "
              f"{r['evidence']}")

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    passed = sum(r["passed"] for r in results)
    report = dict(
        criteria=[{k: r[k] for k in ("id", "name", "passed", "evidence")}
                  for r in results],
        passed=passed, total=len(results))
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    details = Path(args.details)
    details.parent.mkdir(exist_ok=True)
    details.write_text(_md_report(results), encoding="utf-8")

    print(f"\n{passed}/{len(results)} criteria pass; report: {out}; "
          f"details: {details}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
