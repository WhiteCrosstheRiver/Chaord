"""Q3 confirmatory builds (Review 12, D49 trigger): the Z3 builder with the
RESERVOIR composition (88 EC / 55 EMC / 15 LiPF6 per 98.4 A tube — Q3 PLAN
section 3) and NEW seeds (20281101+, disjoint from every prior range).

STANDALONE: imports only build_systems/params (stdlib + numpy, never chaord);
build_systems.py and params.py are used UNMODIFIED on disk; the overrides
this wrapper applies in-process are recorded here and in PLAN.md:
  * the composition override (the Z3 same-molarity one is replaced by the
    Q2-measured reservoir filling, neutralized at 15 LiPF6);
  * new build seeds.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                               # .../cnt_electrolyte
sys.path.insert(0, ROOT)
import params as P          # noqa: E402  (standalone, no chaord)
import build_systems as B   # noqa: E402

# ---- recorded overrides (in-process; the on-disk modules stay untouched) ---
NCELL = 40                       # Lz = 98.4 A (the Z3 geometry)
Q3_COMPOSITION_16 = dict(n_EC=88, n_EMC=55, n_LiPF6=15)   # PLAN section 3
BUILD_SEEDS = {f"q3_b{k}": 20281101 + k for k in range(1, 4)}
VELOCITY_SEEDS = {f"q3_b{k}": 20281500 + k for k in range(1, 4)}

EXPECTED_ATOMS = (40 * 4 * 16) + 88 * 10 + 55 * 15 + 15 * 7 + 15
# 2560 tube + 880 EC + 825 EMC + 105 PF6 + 15 Li
EXPECTED_BONDS = 88 * 10 + 55 * 14 + 15 * 6


def main():
    names = sys.argv[1:] or sorted(BUILD_SEEDS)
    P.COMPOSITION[16] = dict(Q3_COMPOSITION_16)   # the RESERVOIR filling
    for name in names:
        if name not in BUILD_SEEDS:
            sys.exit(f"unknown build {name}")
        P.SYSTEMS[name] = dict(n=16, ncell=NCELL, field="zf")
        P.SEEDS[name] = dict(build=BUILD_SEEDS[name],
                             velocity=VELOCITY_SEEDS[name])
    B.HERE = os.path.join(HERE, "data")   # write data under q3/data/
    os.makedirs(B.HERE, exist_ok=True)

    for name in names:
        info, data_path = B.build_system(name)
        guard = B.validate_data_file(data_path)
        with open(data_path, encoding="utf-8") as f:
            head = [next(f) for _ in range(8)]
        natoms = int(next(l for l in head if "atoms" in l).split()[0])
        nbonds = int(next(l for l in head if "bonds" in l).split()[0])
        if natoms != EXPECTED_ATOMS or nbonds != EXPECTED_BONDS:
            sys.exit(f"{name}: unexpected counts {natoms} atoms / "
                     f"{nbonds} bonds (expected {EXPECTED_ATOMS}/"
                     f"{EXPECTED_BONDS})")
        entry = dict(info, guard=guard,
                     velocity_seed=VELOCITY_SEEDS[name],
                     ncell=NCELL,
                     composition_source="Q2-measured reservoir filling "
                                        "scaled to 98.4 A and neutralized "
                                        "at 15 LiPF6 (PLAN.md section 3)")
        frag = os.path.join(HERE, "data", f"build_manifest_{name}.json")
        with open(frag, "w", encoding="utf-8") as f:
            json.dump({name: entry}, f, indent=2)
        print(f"[q3-build] {name}: {natoms} atoms, "
              f"Lz={info['cnt_length_A']} A, "
              f"{info['molarity_M_vdw_volume']} M, "
              f"min_inter={guard['min_intermolecular_A']} A, "
              f"min_tube={guard['min_mol_tube_A']} A -> {data_path}",
              flush=True)


if __name__ == "__main__":
    main()
