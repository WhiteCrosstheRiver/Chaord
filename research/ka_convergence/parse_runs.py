"""Parse run logs, convert dumps to the bench npz frame convention, manifest.

Per run (runs/<id>/):
  * STAGE_END / FRAME / IS_RESULT lines from log.lammps
    (units lj: thermo pe is per atom -- thermo_modify norm yes default)
  * frame_<k>.lammpstrj -> frame_<k>.npz in the bench convention
    (keys r, L, symbols; ref_common.py save_frame format)
  * sha256 of every artifact -> sha256.txt
  * summary.json (protocol, stage temperatures, frame energies, IS energy)

Usage:  python parse_runs.py [r0] [r1_q03] ...
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

import ka_params as K

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"


def read_dump(path: Path):
    """Minimal LAMMPS custom-dump reader -> (ids, types, pos, L)."""
    lines = path.read_text().splitlines()
    n = None
    L = []
    cols = None
    rows = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("ITEM: TIMESTEP"):
            step = int(lines[i + 1])
            i += 2
        elif lines[i].startswith("ITEM: NUMBER OF ATOMS"):
            n = int(lines[i + 1])
            i += 2
        elif lines[i].startswith("ITEM: BOX BOUNDS"):
            for d in range(3):
                lo, hi = lines[i + 1 + d].split()[:2]
                L.append(float(hi) - float(lo))
            i += 4
        elif lines[i].startswith("ITEM: ATOMS"):
            cols = lines[i].split()[2:]
            i += 1
            for _ in range(n):
                rows.append(lines[i].split())
                i += 1
        else:
            i += 1
    arr = np.array(rows, float)
    ix = {c: k for k, c in enumerate(cols)}
    ids = arr[:, ix["id"]].astype(int)
    types = arr[:, ix["type"]].astype(int)
    pos = arr[:, [ix["x"], ix["y"], ix["z"]]]
    o = np.argsort(ids)
    return step, types[o], np.mod(pos[o], np.array(L)), np.array(L)


def parse_log(path: Path):
    txt = path.read_text()
    out = {"stages": {}, "frames": [], "is": None}
    for m in re.finditer(
            r'STAGE_END (\w+) T\*=\S+ PE=(\S+)', txt):
        out["stages"][m.group(1)] = {"T_star": None, "pe_per_atom": None}
    # fuller: temp and pe captured separately
    for m in re.finditer(
            r'STAGE_END (\w+) T\*=([-\d.]+) PE=([-\d.]+)', txt):
        out["stages"][m.group(1)] = {"T_star": float(m.group(2)),
                                     "pe_per_atom": float(m.group(3))}
    for m in re.finditer(r'FRAME (\d+) PE=([-\d.]+)', txt):
        out["frames"].append({"k": int(m.group(1)),
                              "pe_per_atom": float(m.group(2))})
    m = re.search(r'IS_RESULT T\*=\S+ PE=([-\d.]+) F=(\S+) step=(\d+)', txt)
    if m:
        out["is"] = {"pe_per_atom": float(m.group(1)),
                     "fnorm": float(m.group(2)), "fire_steps": int(m.group(3))}
    m = re.search(r'quench T\*=([-\d.]+)', txt)
    return out


def sha256_tree(d: Path):
    out = []
    for p in sorted(d.rglob("*")):
        if p.is_file() and p.name not in ("sha256.txt", "summary.json",
                                          "DONE", "FAILED"):
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            out.append(f"{h}  {p.relative_to(d).as_posix()}")
    (d / "sha256.txt").write_text("\n".join(out) + "\n")
    return out


def parse_one(rid: str) -> dict:
    d = RUNS / rid
    log = parse_log(d / "log.lammps")
    rate_idx = K.RATE_TAGS.index(rid.split("_")[0])
    prot = K.protocol_record(K.RATES[rate_idx])
    frames_meta = []
    for k in range(K.N_FRAMES):
        step, types, pos, L = read_dump(d / f"frame_{k}.lammpstrj")
        sym = ["B" if t == 2 else "A" for t in types]
        np.savez(d / f"frame_{k}.npz", r=pos,
                 L=np.asarray(L, dtype=np.float64),
                 symbols=np.array(sym, dtype="U8"))
        frames_meta.append({"k": k, "dump_step": step,
                            "protocol_step": prot["frame_steps"][k]})
    # IS frame converted too (provenance of the D20 minimisation)
    step, types, pos, L = read_dump(d / "is_frame.lammpstrj")
    sym = ["B" if t == 2 else "A" for t in types]
    np.savez(d / "is_frame.npz", r=pos, L=np.asarray(L, dtype=np.float64),
             symbols=np.array(sym, dtype="U8"))
    log["frames"] = sorted(
        [f for f in log["frames"] if f["k"] < K.N_FRAMES], key=lambda f: f["k"])
    summary = {
        "run_id": rid, "rate": K.RATES[rate_idx], "rate_idx": rate_idx,
        "protocol": prot, "stages": log["stages"], "frames": frames_meta,
        "frame_pe_per_atom": [f["pe_per_atom"] for f in log["frames"]],
        "inherent_structure": log["is"],
        "density": K.N_ATOMS / float(np.prod([K.L_BOX] * 3)),
    }
    (d / "summary.json").write_text(json.dumps(summary, indent=1))
    summary["sha256"] = sha256_tree(d)
    return summary


def main(argv):
    want = argv[1:] or ["all"]
    out = {}
    for r in range(len(K.RATES)):
        for q in range(K.N_SEEDS):
            rid = K.run_id(r, q)
            if (RUNS / rid / "DONE").exists() and (
                    want == ["all"] or K.RATE_TAGS[r] in want or rid in want):
                s = parse_one(rid)
                out[rid] = {k: s[k] for k in
                            ("rate", "stages", "frame_pe_per_atom",
                             "inherent_structure")}
                print(f"{rid}: quench_end T*={s['stages']['quench']['T_star']} "
                      f"PE5={s['frame_pe_per_atom'][-1]:.4f} "
                      f"IS={s['inherent_structure']['pe_per_atom']:.4f}")
    print(f"parsed {len(out)} runs")


if __name__ == "__main__":
    main(sys.argv)
