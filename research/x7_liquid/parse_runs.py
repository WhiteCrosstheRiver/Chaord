"""Parse run logs, convert dumps to the bench npz frame convention, manifest.

Per run (runs/<id>/):
  * STAGE_END / FRAME lines from log.lammps (units lj: thermo pe is per atom,
    thermo_modify norm yes default -- the ka_convergence precedent)
  * frame_<k>.lammpstrj -> frame_<k>.npz in the bench convention
    (keys r, L, symbols; ref_common.py save_frame format)
  * sha256 of every artifact -> sha256.txt
  * summary.json (protocol, equil temperature/energy, frame energies)

Usage:  python parse_runs.py [lj500] [lj2048_t01] ...
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

import liq_params as K

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"


def read_dump(path: Path):
    """Minimal LAMMPS custom-dump reader -> (step, types, pos, L)."""
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
    out = {"stages": {}, "frames": []}
    for m in re.finditer(r'STAGE_END (\w+) T\*=([-\d.]+) PE=([-\d.]+)', txt):
        out["stages"][m.group(1)] = {"T_star": float(m.group(2)),
                                     "pe_per_atom": float(m.group(3))}
    for m in re.finditer(r'FRAME (\d+) PE=([-\d.]+)', txt):
        out["frames"].append({"k": int(m.group(1)),
                              "pe_per_atom": float(m.group(2))})
    m = re.search(r'RUN_END T\*=([-\d.]+) PE=([-\d.]+)', txt)
    if m:
        out["run_end"] = {"T_star": float(m.group(1)),
                          "pe_per_atom": float(m.group(2))}
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
    case_idx = 0 if rid.startswith("lj500") else 1
    prot = K.protocol_record(case_idx)
    frames_meta = []
    for k in range(K.N_FRAMES):
        step, types, pos, L = read_dump(d / f"frame_{k}.lammpstrj")
        assert len(pos) == prot["n_atoms"], (rid, k, len(pos))
        sym = ["X" for _ in types]     # single species (committed convention)
        np.savez(d / f"frame_{k}.npz", r=pos,
                 L=np.asarray(L, dtype=np.float64),
                 symbols=np.array(sym, dtype="U8"))
        frames_meta.append({"k": k, "dump_step": int(step),
                            "protocol_step": prot["frame_steps"][k]})
    summary = {
        "run_id": rid, "case": prot["case"], "case_idx": case_idx,
        "protocol": prot,
        "seeds": json.loads(
            (RUNS / "inputs_manifest.json").read_text())["runs"][rid]["seeds"],
        "stages": log["stages"], "frames": frames_meta,
        "frame_pe_per_atom": [f["pe_per_atom"] for f in log["frames"]],
        "run_end": log.get("run_end"),
        "density": prot["n_atoms"] / prot["box_L"] ** 3,
    }
    (d / "summary.json").write_text(json.dumps(summary, indent=1))
    sha256_tree(d)
    return summary


def main(argv):
    want = [a for a in argv[1:]] or None
    man = json.loads((RUNS / "inputs_manifest.json").read_text())
    for rid in man["runs"]:
        if want and not any(rid.startswith(w) or rid == w for w in want):
            continue
        d = RUNS / rid
        if not (d / "log.lammps").exists():
            print(f"  {rid}: no log.lammps, skipped")
            continue
        s = parse_one(rid)
        n_frames = len(s["frames"])
        te = s["stages"].get("equil", {}).get("T_star")
        print(f"  {rid}: {n_frames} frames, equil T* {te}, "
              f"end PE/atom {s['run_end']['pe_per_atom'] if s['run_end'] else '?'}")


if __name__ == "__main__":
    main(sys.argv)
