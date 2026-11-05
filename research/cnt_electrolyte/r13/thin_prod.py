# -*- coding: utf-8 -*-
"""Cluster-side production-dump thinner (只算不存 post-processing step).

Keeps EVERY 12th production frame (12 ps spacing -- the same cadence as the
ratified Stage B cluster record), verifies the expected frame count and last
timestep, writes prod_thinned.dump + prod_thinned.sha256 (bytes + sha256,
same schema as stageB's prod_series_manifest.json) and a THIN_DONE marker.
The caller (sbatch.sh) deletes prod.dump only after THIN_DONE exists.

A frame is the block of lines from an 'ITEM: TIMESTEP' line up to (not
including) the next one -- kept verbatim or dropped whole, so the thinned
file is byte-identical to the corresponding frames of prod.dump.

Run in a job directory:  python3 thin_prod.py [dir]  (default: cwd)

STANDALONE: Python 3 stdlib only (cluster anaconda python3, no numpy needed).
"""
import hashlib
import json
import os
import sys

STRIDE = 12                      # keep frame if (ts // 1000) % 12 == 0
EXPECTED_LAST_TS = 12_000_000    # 2 x 6 ns blocks at dt 1 fs
EXPECTED_FRAMES = 1001           # ts = 0, 12000, ..., 12000000


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def thin(d):
    src = os.path.join(d, "prod.dump")
    dst = os.path.join(d, "prod_thinned.dump")
    if not os.path.isfile(src):
        sys.exit(f"thin_prod: no {src}")
    kept = n_read = 0
    last_ts = None
    buf = []
    with open(src) as fi, open(dst, "w") as fo:
        for line in fi:
            if line.startswith("ITEM: TIMESTEP") and buf:
                ts = int(buf[1])
                n_read += 1
                last_ts = ts
                if (ts // 1000) % STRIDE == 0:
                    fo.writelines(buf)
                    kept += 1
                buf = []
            buf.append(line)
        if buf:
            ts = int(buf[1])
            n_read += 1
            last_ts = ts
            if (ts // 1000) % STRIDE == 0:
                fo.writelines(buf)
                kept += 1
    if kept != EXPECTED_FRAMES or last_ts != EXPECTED_LAST_TS:
        if os.path.isfile(dst):
            os.remove(dst)
        sys.exit(f"thin_prod: REFUSED -- read {n_read} frames, kept {kept} "
                 f"(expected {EXPECTED_FRAMES}), last ts {last_ts} "
                 f"(expected {EXPECTED_LAST_TS}); thinned file deleted, "
                 f"prod.dump kept for diagnosis")
    info = dict(prod_thinned_bytes=os.path.getsize(dst),
                prod_thinned_sha256=sha256(dst),
                stride=STRIDE, frames=kept, last_timestep=last_ts,
                source="prod.dump (1 ps) -> every 12th frame (12 ps)")
    with open(os.path.join(d, "prod_thinned.sha256"), "w") as f:
        json.dump(info, f, indent=2)
    with open(os.path.join(d, "THIN_DONE"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"thin_prod: kept {kept}/{n_read} frames "
          f"(last ts {last_ts}), {info['prod_thinned_bytes']} bytes, "
          f"sha256 {info['prod_thinned_sha256'][:16]}...", flush=True)


if __name__ == "__main__":
    thin(sys.argv[1] if len(sys.argv) > 1 else os.getcwd())
