# -*- coding: utf-8 -*-
"""Build the six R13 reservoir-filling (16,16) systems: 98.4 A tubes
(ncell = 40) at the corrected open-tube uptake (r13/PLAN.md section 1,
D55): EC 125 / EMC 58 / 24 LiPF6 (Q2 per-A x 98.4, salt neutralized at the
rounded mean of the measured Li+ 26.3 / PF6- 20.9).

STANDALONE: imports only build_systems/params (stdlib + numpy, never chaord).
build_systems.py and params.py are used UNMODIFIED on disk; the overrides
below are in-process only, recorded here and in r13/data/build_manifest.json:

  * ncell 40 (Lz = 98.4 A);
  * composition (16,16): 125/58/24 (the reservoir filling, PLAN.md 1);
  * new build seeds 20291101..20291106 (PLAN.md section 5, disjoint);
  * data files land in r13/data/.

JAM POLICY (T1 order, verbatim): "if the packer jams at this density, pack
in stages (insert at reduced LJ radius, then relax to the floor) and record
every attempt; do not lower the counts." Implemented here as a staged floor
scale: each attempt multiplies the module-level MIN_ALLOWED threshold matrix
by a scale factor s < 1 (reduced contact floors for INSERTION only), then
after the last insertion restores the matrix to s = 1 and runs the full
push-apart relaxation + strict verify() against the TRUE floors. The counts
are never changed. Every attempt (scale, outcome) is recorded in the
manifest.

Every written data file is re-validated with the f9cae3f hardening
(validate_data_file) plus count/charge checks.

Usage:
    python r13_build.py                 # all six builds, sequential
    python r13_build.py r13_b1 ...      # given builds
    python r13_build.py --merge         # assemble manifest fragments
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                               # .../cnt_electrolyte
sys.path.insert(0, ROOT)
import params as P          # noqa: E402  (standalone, no chaord)
import build_systems as B   # noqa: E402

# ---- recorded overrides (in-process; the on-disk modules stay untouched) ---
NCELL = 40                       # Lz = 98.4 A
R13_COMPOSITION_16 = dict(n_EC=125, n_EMC=58, n_LiPF6=24)
BUILD_SEEDS = {f"r13_b{k}": 20291101 + k - 1 for k in range(1, 7)}
VELOCITY_SEEDS = {f"r13_b{k}": 20291151 + k - 1 for k in range(1, 7)}

EXPECTED_ATOMS = 4872            # 2560 tube + 1250 EC + 870 EMC + 168 PF6 + 24 Li
EXPECTED_BONDS = 2206            # 125x10 EC + 58x14 EMC + 24x6 PF6

# staged-packing ladder (PLAN.md section 5): 1.0 = the stock pack-then-relax
# packer untouched; each failed build retries at the next reduced scale
FLOOR_SCALES = (1.0, 0.90, 0.85, 0.80, 0.75, 0.70)

# The reduced floors apply to INSERTION ONLY: _ok / _candidate see the
# scaled matrix (swapped in by reference around each call and restored
# immediately), while Packer.relax / Packer.verify and the final strict
# verify() always run against the TRUE floor matrix. Reference swapping
# keeps the restore bit-exact.
_TRUE_MATRIX = None
_SCALED_MATRIX = None
_ok_orig = B.Packer._ok
_candidate_orig = B.Packer._candidate


def _ok_staged(self, types, xyz, li_sep):
    B.MIN_ALLOWED = _SCALED_MATRIX
    try:
        return _ok_orig(self, types, xyz, li_sep)
    finally:
        B.MIN_ALLOWED = _TRUE_MATRIX


def _candidate_staged(self, template, R_pack, longest_axis):
    B.MIN_ALLOWED = _SCALED_MATRIX
    try:
        return _candidate_orig(self, template, R_pack, longest_axis)
    finally:
        B.MIN_ALLOWED = _TRUE_MATRIX


def merge_fragments():
    manifest = {}
    for frag in sorted(glob.glob(os.path.join(HERE, "data",
                                              "build_manifest_*.json"))):
        with open(frag, encoding="utf-8") as f:
            manifest.update(json.load(f))
    out = os.path.join(HERE, "data", "build_manifest.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[r13-build] merged {len(manifest)} builds -> {out}")


def build_one(name):
    """One build with the staged-floor jam policy. Returns (info, path)."""
    global _TRUE_MATRIX, _SCALED_MATRIX
    attempts = []
    for scale in FLOOR_SCALES:
        _TRUE_MATRIX = B.MIN_ALLOWED
        _SCALED_MATRIX = _TRUE_MATRIX * scale if scale < 1.0 else _TRUE_MATRIX
        if scale < 1.0:
            B.Packer._ok = _ok_staged
            B.Packer._candidate = _candidate_staged
        try:
            info, data_path = B.build_system(name)
            attempts.append(dict(floor_scale=scale, outcome="packed"))
            break
        except RuntimeError as exc:
            attempts.append(dict(floor_scale=scale, outcome=f"jam: {exc}"))
            print(f"[r13-build] {name}: jam at floor scale {scale} "
                  f"({exc}); retrying at the next scale", flush=True)
        finally:
            B.MIN_ALLOWED = _TRUE_MATRIX
            B.Packer._ok = _ok_orig
            B.Packer._candidate = _candidate_orig
    else:
        raise RuntimeError(f"{name}: packing failed at every staged floor "
                           f"scale {FLOOR_SCALES}; counts NOT lowered "
                           f"(T1 policy)")
    # belt and braces: the f9cae3f guard re-parses the WRITTEN file
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
                 packing_attempts=attempts,
                 jam_policy="staged floor scale (T1): insert at reduced "
                            "MIN_ALLOWED, restore true floors, full relax + "
                            "strict verify; counts never lowered",
                 composition_source="corrected open-tube uptake (fc8bec2) "
                                    "x 98.4 A: EC 125 / EMC 58 / 24 LiPF6 "
                                    "(salt neutralized at the rounded mean "
                                    "of the measured Li+ 26.3 / PF6- 20.9; "
                                    "PLAN.md section 1)")
    frag = os.path.join(HERE, "data", f"build_manifest_{name}.json")
    with open(frag, "w", encoding="utf-8") as f:
        json.dump({name: entry}, f, indent=2)
    print(f"[r13-build] {name}: {natoms} atoms, "
          f"Lz={info['cnt_length_A']} A, "
          f"{info['molarity_M_vdw_volume']} M, "
          f"min_inter={guard['min_intermolecular_A']} A, "
          f"min_tube={guard['min_mol_tube_A']} A, "
          f"staged_scale={attempts[-1]['floor_scale']} -> {data_path}",
          flush=True)


def main():
    if "--merge" in sys.argv:
        merge_fragments()
        return
    names = sys.argv[1:] or sorted(BUILD_SEEDS)
    P.COMPOSITION[16] = dict(R13_COMPOSITION_16)
    for name in names:
        if name not in BUILD_SEEDS:
            sys.exit(f"unknown build {name}")
        P.SYSTEMS[name] = dict(n=16, ncell=NCELL, field="zf")
        P.SEEDS[name] = dict(build=BUILD_SEEDS[name],
                             velocity=VELOCITY_SEEDS[name])
    B.HERE = HERE                    # write data under r13/data/

    for name in names:
        build_one(name)


if __name__ == "__main__":
    main()
