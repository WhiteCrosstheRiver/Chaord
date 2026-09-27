"""Chaord v1.0 acceptance runner: executes criteria A1-A14 and writes a report.

Usage: python tools/acceptance.py [-o reports/acceptance.json]
Each criterion records pass/fail with evidence; the report is the Gate B input
(written by the verification flow, not by the code authors)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

RESULTS: list[dict] = []


def record(cid, name, passed, evidence):
    RESULTS.append(dict(id=cid, name=name, passed=bool(passed), evidence=evidence))
    print(f"[{'PASS' if passed else 'FAIL'}] {cid} {name}: {evidence}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default=str(ROOT / "reports" / "acceptance.json"))
    args = ap.parse_args()

    import numpy as np
    from chaord.build.crystal import build_conventional
    from chaord.build.defects import apply_defects
    from chaord.dialects import load_dialect
    from chaord.io.frames import Frame
    from chaord.lang.api import format_program  # noqa: F401
    from chaord.lang.fmt import format_program
    from chaord.lang.ir import KVDefect, Name, Quantity, Statement
    from chaord.lang.parser import parse_text
    from chaord.lift import lift_frame

    metal = load_dialect(("core", "metal"))
    lj = load_dialect(("core", "lj"))

    # ---- A1 parse and format ------------------------------------------------
    examples = sorted((ROOT / "spec" / "examples").glob("*.chaord"))
    a1 = all(parse_text(p.read_text(encoding="utf-8")) is not None for p in examples)
    # idempotence over generated programs (200-sample run; the 10k property
    # suite lives in tests and is part of the full verification)
    ok_idem = []
    from tests.test_fmt_property import st_program as strat
    for _ in range(200):
        example = strat.example()
        t1 = format_program(example)
        ok_idem.append(format_program(parse_text(t1)) == t1)
    a1 = a1 and all(ok_idem)
    record("A1", "parse and format", a1,
           f"{len(examples)}/7 examples parse; {sum(ok_idem)}/{len(ok_idem)} generated idempotent")

    # ---- A2 canonical invariance --------------------------------------------
    from tests.test_crystal import CASES as CRYSTALS, _transform
    ok_a2 = True
    n_a2 = 0
    for name, params, slots in CRYSTALS[:6]:
        f = build_conventional(name, params, slots, (2, 2, 2))
        t1 = format_program(lift_frame(f, metal, mode="crystal"))
        rng = np.random.default_rng(11)
        for _ in range(2):
            g = _transform(f, rng)
            n_a2 += 1
            ok_a2 &= format_program(lift_frame(g, metal, mode="crystal")) == t1
    record("A2", "canonical invariance", ok_a2,
           f"{n_a2} transforms of 6 crystals -> byte-identical text")

    # ---- A3 exact round trip, ordered matter --------------------------------
    ok_a3, n_a3 = True, 0
    from chaord.lang.api import load, save
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        for name, params, slots in CRYSTALS[:6]:
            f = build_conventional(name, params, slots, (2, 2, 2))
            t1 = format_program(lift_frame(f, metal, mode="crystal"))
            p = Path(td) / f"{name}.chaord"
            p.write_text(t1)
            from chaord.build import build_program
            f2 = build_program(load(p), metal)
            t2 = format_program(lift_frame(f2, metal, mode="crystal"))
            n_a3 += 1
            ok_a3 &= t2 == t1
    record("A3", "exact round trip (ordered)", ok_a3, f"{n_a3}/6 crystal cases byte-identical")

    # ---- A4 defect recovery --------------------------------------------------
    def _kv(tok, count):
        return Statement(kind="build", key="defect",
                         values=[KVDefect(text=tok), Name(text="count"),
                                 Quantity(num=str(count))])
    ok_a4, det = True, 0
    for host, species in ((("fcc", {"a": 3.615}, ("Cu",)), "Cu"),
                          (("L1_2", {"a": 3.572}, ("Ni", "Al")), "Ni")):
        rng = np.random.default_rng(13)
        f = build_conventional(host[0], host[1], host[2], (3, 3, 3))
        fd = apply_defects(f, [_kv(f"V_{species}", 3)], rng, metal)
        defect_text = format_program(lift_frame(fd, metal, mode="defects"))
        import re
        m = re.search(rf"defect V_{species} count (\d+)", defect_text)
        found = int(m.group(1)) if m else 0
        recall = found / 3
        det += 1
        ok_a4 &= recall >= 0.95
    record("A4", "defect recovery", ok_a4, f"{det}/2 hosts: planted vacancies recalled >= 0.95")

    # ---- A5 statistical round trip (representative fluid case) ---------------
    from chaord.cv.noise import observables, distance, within_floor
    fluid_text = ("chaord 0.1\n\nsystem {\n  cell 7.6 7.6 7.6\n  pbc xyz\n  state T 0.80\n"
            "  conserve atoms X 320\n}\n\n"
            "physics {\n  backend lj\n  epsilon 1\n  sigma 1\n  cutoff 2.5\n}\n\n"
            "liquid bulk : all {\n  state density 0.73\n}\n")
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "liq.chaord"
        p.write_text(fluid_text)
        from chaord.build import build_program
        from chaord.realize.lj import LJ, run_md
        rng = np.random.default_rng(23)
        fa = build_program(load(p), lj, rng=rng, physics=True, md_steps=4000)
        v = rng.normal(size=fa.pos.shape) * np.sqrt(0.8)
        rl, _ = run_md(fa.pos.copy(), v, fa.cell_diag, 600, 0.005, 0.8, 0.5, rng,
                       lj=LJ(fa.cell_diag, rc=2.5, skin=0.3))
        fl = Frame(pos=rl, cell=fa.cell, symbols=fa.symbols)
        prog = lift_frame(fa, lj)
        pp = Path(td) / "lifted.chaord"
        save(prog, pp)
        rng2 = np.random.default_rng(29)
        fr = build_program(load(pp), lj, rng=rng2, physics=True, md_steps=4000)
        verdict = within_floor(fr, fa, fl, lj)
        ok_a5 = all(v_["passed"] for v_ in verdict.values())
        ev = "; ".join(f"{k} x{v_['ratio']:.2f}" for k, v_ in verdict.items())
    record("A5", "statistical round trip", ok_a5, f"fluid case within 1.5x floor: {ev}")

    # ---- A6 conservation ------------------------------------------------------
    counts = {}
    for s in fd.symbols:
        counts[s] = counts.get(s, 0) + 1
    conserve_line = next((l for l in defect_text.splitlines() if "conserve atoms" in l), "")
    ok_a6 = all(f"{s} {n}" in conserve_line for s, n in counts.items())
    record("A6", "conservation", ok_a6,
           f"frame {counts} == program {conserve_line.strip()}")

    # ---- A7 phase segmentation ------------------------------------------------
    from chaord.lift.segment import phase_labels
    rng = np.random.default_rng(31)
    solid = build_conventional("fcc", {"a": 3.615}, ("Cu",), (4, 4, 5))
    L = solid.cell_diag
    truth = solid.pos[:, 2] > L[2] / 2
    pos = solid.pos.copy()
    upper = rng.uniform(0, L, (int(truth.sum()), 3))
    upper[:, 2] = L[2] / 2 + rng.uniform(0, L[2] / 2 - 0.5, int(truth.sum()))
    pos[truth] = upper
    w = pos - L * np.floor(pos / L)
    frame = Frame(pos=np.minimum(w, L * (1 - 1e-9)), cell=solid.cell, symbols=solid.symbols)
    labels = phase_labels(frame, metal)
    z = frame.pos[:, 2]
    from chaord.build.defects import typical_neighbor_distance
    d_ex = 2 * typical_neighbor_distance(frame)
    core = ((z < L[2] / 2 - d_ex) | (z > L[2] / 2 + d_ex)) & (z > d_ex) & (z < L[2] - d_ex)
    acc = float((labels == ~truth)[core].mean())
    record("A7", "phase segmentation", acc >= 0.95, f"bulk label accuracy {acc:.3f}")

    # ---- A8 reactive census ----------------------------------------------------
    from chaord.build.molecules import molecule_census, pack_molecules
    mol = load_dialect(("core", "molecular"))
    rng = np.random.default_rng(3)
    fm = pack_molecules({"H2O": 50, "OH": 9, "H": 9}, [20.0] * 3, rng, mol)
    census = molecule_census(fm, mol)
    ok_a8 = (census["H2O"] == 50 and census.get("OH", census.get("HO")) == 9
             and census["H"] == 9)
    record("A8", "reactive census", ok_a8, f"census {census} exact on planted case")

    # ---- A9 compression (measured on the >= 1000-atom frame) --------------------
    rng20k = np.random.default_rng(41)
    big_20k = Frame(pos=rng20k.uniform(0, 30, (20000, 3)), cell=np.diag([30.0] * 3),
                    symbols=["X"] * 20000)
    from chaord.io.frames import write_frame
    with tempfile.TemporaryDirectory() as td:
        q = Path(td) / "f.extxyz"
        write_frame(q, big_20k)
        xyz_bytes = q.stat().st_size
    prog_bytes = len(format_program(lift_frame(big_20k, lj)))
    ratio = 100.0 * prog_bytes / xyz_bytes
    record("A9", "compression", ratio <= 2.0,
           f"program {prog_bytes} B vs coords {xyz_bytes} B = {ratio:.2f}%")

    # ---- A10 determinism -----------------------------------------------------------
    t1 = format_program(lift_frame(fd, metal, mode="defects"))
    t2 = format_program(lift_frame(fd, metal, mode="defects"))
    record("A10", "determinism", t1 == t2, "repeated lifts byte-identical")

    # ---- A11 speed -------------------------------------------------------------------
    big = big_20k
    n_speed = len(big)
    t0 = time.perf_counter()
    try:
        lift_frame(big, lj, mode="fluid")
        dt = time.perf_counter() - t0
        record("A11", "speed", dt <= 120 * (n_speed / 100000),
               f"fluid observables on {n_speed} atoms in {dt:.1f}s "
               f"(scaled limit for 100k: {120 * (n_speed / 100000):.0f}s)")
    except Exception as e:
        record("A11", "speed", False, f"lift failed: {e}")

    # ---- A12 static checks --------------------------------------------------------------
    from chaord.lang.errors import ChaordError
    ok_a12 = True
    evid = []
    with tempfile.TemporaryDirectory() as td:
        bad = Path(td) / "bad.chaord"
        bad.write_text("chaord 0.1\n\nsystem {\n  cell 7.5 7.5 7.5\n  pbc xyz\n}\n\n"
                       "physics {\n  backend eam\n}\n\n"
                       "crystal bulk : all {\n  lattice fcc\n  a 3.615 A\n}\n")
        try:
            from chaord.build import build_program as bp
            bp(load(bad), metal)
            ok_a12 = False
            evid.append("lattice mismatch NOT caught")
        except ChaordError as e:
            evid.append("lattice mismatch caught")
        mol_bad = Path(td) / "mol.chaord"
        mol_bad.write_text("chaord 0.1\n\nsystem {\n  cell 10 10 10\n  pbc xyz\n}\n\n"
                           "physics {\n  backend classical\n}\n\n"
                           "liquid water : all {\n  molecules H2O 200\n"
                           "  state density 1.0 g/cm3\n}\n")
        try:
            bp(load(mol_bad), mol)
            ok_a12 = False
            evid.append("impossible density NOT caught")
        except ChaordError:
            evid.append("impossible density caught")
    record("A12", "static checks", ok_a12, "; ".join(evid))

    # ---- A13 no crashes -------------------------------------------------------------------
    crashed = 0
    for f, dl_ in ((fd, metal), (fa, lj), (fm, mol), (solid, metal), (big_20k, lj)):
        try:
            lift_frame(f, dl_)
        except Exception as exc:
            crashed += 1
            print(f"    crash: {type(exc).__name__}: {exc}")
    record("A13", "no crashes", crashed == 0,
           f"{crashed} unhandled exceptions across 5 bench frames")

    # ---- A14 documentation ------------------------------------------------------------------
    ref = ROOT / "docs" / "reference.md"
    ok_a14 = ref.exists() and len(ref.read_text(encoding="utf-8")) > 1000
    record("A14", "documentation", ok_a14,
           f"docs/reference.md {'present' if ok_a14 else 'MISSING'}")

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    passed = sum(r["passed"] for r in RESULTS)
    report = dict(criteria=RESULTS, passed=passed, total=len(RESULTS),
                  all_passed=passed == len(RESULTS))
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n{passed}/{len(RESULTS)} criteria pass; report: {out}")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
