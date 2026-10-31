# -*- coding: utf-8 -*-
"""Build the six Z3 confirmatory (16,16) systems: 98.4 A tubes (ncell = 40),
composition doubled from the pilot for the same molarity (PLAN.md section 6).

STANDALONE: imports only build_systems/params (stdlib + numpy, never chaord).
build_systems.py and params.py are used UNMODIFIED on disk; the overrides
below are in-process only, recorded here and in confirmatory/data/
build_manifest.json:

  * ncell 20 -> 40 (Lz = 40 x 2.46 A = 98.4 A, twice the pilot tube);
  * composition (16,16): 37/24/6 -> 74/48/12 (exactly x2: the vdW pore
    volume doubles, so the same 1.101 M / 1.224 g/cm3 needs twice the
    counts -- derivation in PLAN.md section 6);
  * new build seeds 20273001..20273006, disjoint from every committed seed
    family (PLAN.md section 5);
  * data files land in confirmatory/data/ (B.HERE patched), NOT in the
    shared pilot data/ directory.

Every written data file is re-validated with the f9cae3f hardening
(validate_data_file: inter-molecular and molecule-tube minimum-image floors,
molecule wholeness, bond wholeness) plus the count/charge checks, and the
guard stats go into the manifest.

Usage:
    python confirmatory_build.py                # all six builds, sequential
    python confirmatory_build.py cf16_b1 ...    # given builds, parallel-safe
                                                # (each writes its own
                                                #  manifest fragment)
    python confirmatory_build.py --merge        # assemble the fragments into
                                                # data/build_manifest.json
"""
import glob
import json
import os
import sys

import numpy as np  # noqa: F401  (build_systems needs it; keep the import here too)

HERE = os.path.dirname(os.path.abspath(__file__))          # .../confirmatory
ROOT = os.path.dirname(HERE)                               # .../cnt_electrolyte
sys.path.insert(0, ROOT)
import params as P          # noqa: E402  (standalone, no chaord)
import build_systems as B   # noqa: E402

# ---- recorded overrides (in-process; the on-disk modules stay untouched) ---
NCELL = 40                       # Lz = 98.4 A (D41: 40 periods)
CONF_COMPOSITION_16 = dict(n_EC=74, n_EMC=48, n_LiPF6=12)
BUILD_SEEDS = {f"cf16_b{k}": 20273000 + k for k in range(1, 7)}
VELOCITY_SEEDS = {f"cf16_b{k}": 20273500 + k for k in range(1, 7)}

EXPECTED_ATOMS = 4116            # 2560 tube + 740 EC + 720 EMC + 84 PF6 + 12 Li
EXPECTED_BONDS = 1484            # 740x10 EC + 48x14 EMC + 12x6 PF6


def merge_fragments():
    """Assemble the per-build fragments into data/build_manifest.json."""
    manifest = {}
    for frag in sorted(glob.glob(os.path.join(HERE, "data",
                                              "build_manifest_*.json"))):
        with open(frag, encoding="utf-8") as f:
            manifest.update(json.load(f))
    out = os.path.join(HERE, "data", "build_manifest.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[confirmatory-build] merged {len(manifest)} builds -> {out}")


def main():
    if "--merge" in sys.argv:
        merge_fragments()
        return
    names = sys.argv[1:] or sorted(BUILD_SEEDS)
    P.COMPOSITION[16] = dict(CONF_COMPOSITION_16)   # same molarity at 2x pore
    for name in names:
        if name not in BUILD_SEEDS:
            sys.exit(f"unknown build {name}")
        P.SYSTEMS[name] = dict(n=16, ncell=NCELL, field="zf")
        P.SEEDS[name] = dict(build=BUILD_SEEDS[name],
                             velocity=VELOCITY_SEEDS[name])
    B.HERE = HERE                    # write data under confirmatory/data/

    for name in names:
        info, data_path = B.build_system(name)
        # belt and braces: the f9cae3f guard re-parses the WRITTEN file
        guard = B.validate_data_file(data_path)
        # count checks against the recorded composition (fail closed)
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
                     composition_source="pilot 37/24/6 x2 for the "
                                        "same molarity at 2x pore "
                                        "volume (PLAN.md section 6)")
        frag = os.path.join(HERE, "data", f"build_manifest_{name}.json")
        with open(frag, "w", encoding="utf-8") as f:
            json.dump({name: entry}, f, indent=2)
        print(f"[confirmatory-build] {name}: {natoms} atoms, "
              f"Lz={info['cnt_length_A']} A, "
              f"{info['molarity_M_vdw_volume']} M, "
              f"min_inter={guard['min_intermolecular_A']} A, "
              f"min_tube={guard['min_mol_tube_A']} A -> {data_path}",
              flush=True)


if __name__ == "__main__":
    main()
