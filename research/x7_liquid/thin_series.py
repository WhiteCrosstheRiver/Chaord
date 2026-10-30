"""Thin the frame series and export the observable series (chaord-free).

只算不存 discipline: the raw .lammpstrj dumps are deleted after thinning.
What comes back to the repo (small):
  * runs/<id>/frame_<k>.npz for every 3rd frame (k = 0, 3, ..., 42 -> 15 per
    trajectory, 15 tau apart >= the 20-tau... no: 15 tau apart, still >= 3x
    the measured decorrelation of ~2.5 tau) -- the same bench npz convention
    as the committed cases, so the frames remain independently reusable;
  * observables/<case>_obs.npz -- per-frame observables (gr 60, rm 60,
    cn_hist 26, cn) for ALL 135 frames of the case, from which every pair
    distance and both floors reproduce bit-exactly with liq_obs.py;
  * thinning_manifest.json -- sha256 of every artifact kept.
The dumps are deleted here (both the local copies and the cluster copies are
handled by the caller; this script deletes the local ones).

Usage:  python thin_series.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import liq_params as K
from liq_obs import frame_observables

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
OBS = HERE / "observables"
THIN_EVERY = 3


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    OBS.mkdir(exist_ok=True)
    man = {"generator": "research/x7_liquid/thin_series.py",
           "thin_every": THIN_EVERY,
           "note": "raw lammpstrj dumps deleted after thinning (只算不存); "
                   "thinned frames = every 3rd frame (15/trajectory, 15 tau "
                   "apart); observable series covers ALL frames so every "
                   "floor pair reproduces bit-exactly from liq_obs.py",
           "files": {}}
    for case_idx, case_name in ((0, "lj_liquid"), (1, "lj_liquid_large")):
        obs_all = {k: [] for k in ("gr", "rm", "cn_hist", "cn")}
        traj_idx = []
        frame_idx = []
        for t in range(K.N_TRAJ):
            rid = K.run_id(case_idx, t)
            d = RUNS / rid
            kept = []
            for k in range(K.N_FRAMES):
                src = d / f"frame_{k}.npz"
                with np.load(src) as z:
                    pos = np.asarray(z["r"], float)
                    L = np.asarray(z["L"], float)
                    sym = z["symbols"]
                o = frame_observables(pos, L)
                for key in ("gr", "rm", "cn_hist", "cn"):
                    obs_all[key].append(np.asarray(o[key]))
                traj_idx.append(t)
                frame_idx.append(k)
                if k % THIN_EVERY == 0:
                    kept.append((k, src))
            # rewrite the thinned frames in place (delete the rest)
            for k in range(K.N_FRAMES):
                p = d / f"frame_{k}.npz"
                if k % THIN_EVERY != 0:
                    p.unlink()
            for k, src in kept:
                dst = d / f"frame_{k}.npz"      # already in place
                man["files"][f"{rid}/frame_{k}.npz"] = sha256(dst)
            print(f"  {rid}: kept {len(kept)}/{K.N_FRAMES} thinned frames")
        np.savez(OBS / f"{case_name}_obs.npz",
                 gr=np.array(obs_all["gr"]),
                 rm=np.array(obs_all["rm"]),
                 cn_hist=np.array(obs_all["cn_hist"]),
                 cn=np.array(obs_all["cn"]),
                 traj=np.array(traj_idx), frame=np.array(frame_idx),
                 case=case_name)
        man["files"][f"observables/{case_name}_obs.npz"] = sha256(
            OBS / f"{case_name}_obs.npz")
        # hash the small per-run artifacts kept in the repo
        for t in range(K.N_TRAJ):
            rid = K.run_id(case_idx, t)
            d = RUNS / rid
            for name in (f"data.{rid}", f"in.{rid}", "log.lammps",
                         "summary.json", "sha256.txt"):
                p = d / name
                if p.exists():
                    man["files"][f"{rid}/{name}"] = sha256(p)
        print(f"  {case_name}: observable series "
              f"{np.array(obs_all['gr']).shape} -> observables/{case_name}_obs.npz")
    (HERE / "thinning_manifest.json").write_text(json.dumps(man, indent=1))
    # delete the raw dumps (the heavy raw artefact)
    n_dump = 0
    for dmp in RUNS.glob("*/frame_*.lammpstrj"):
        dmp.unlink()
        n_dump += 1
    print(f"deleted {n_dump} raw dump files (只算不存)")
    print(f"manifest -> thinning_manifest.json ({len(man['files'])} files)")
    return 0


if __name__ == "__main__":
    main()
