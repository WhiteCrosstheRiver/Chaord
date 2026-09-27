"""Noise-floor tool: distances between two frames of one simulation.

Usage: python tools/noise_floor.py FRAME_A FRAME_B [--dialect core+lj]
Writes a small JSON report and prints it. These numbers define the
statistical round-trip tolerance (1.5x this floor), per PLAN.md.
"""
from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("frame_a")
    ap.add_argument("frame_b")
    ap.add_argument("--dialect", default="core+lj")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    from chaord.cv.noise import noise_floor
    from chaord.dialects import load_dialect
    from chaord.io.frames import read_frame

    dialect = load_dialect(tuple(args.dialect.split("+")))
    fa = read_frame(args.frame_a)
    fb = read_frame(args.frame_b)
    floor = noise_floor(fa, fb, dialect)
    report = {"frame_a": args.frame_a, "frame_b": args.frame_b,
              "dialect": dialect.version_string, "floor": floor,
              "roundtrip_tolerance": {k: 1.5 * v for k, v in floor.items()}}
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
