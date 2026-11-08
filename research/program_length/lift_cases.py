# -*- coding: utf-8 -*-
"""Design input for the program-length experiment: lift frame 0 of each A5
reference case exactly as tools/acceptance.py check_a5 does (provenance T,
the case's dialect) and write the program text to out/<case>.chaord.
No rebuild, no observables: this produces no outcome data."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import acceptance as A                      # noqa: E402
from chaord.dialects import load_dialect    # noqa: E402
from chaord.io.frames import read_frame     # noqa: E402
from chaord.lift import lift_frame          # noqa: E402

floors = json.loads(A.NOISE_FLOORS.read_text(encoding="utf-8"))
out = Path("out")
for case in A._reference_cases(ROOT / "bench" / "reference", floors):
    dl = load_dialect(case["dialect"])
    frame = read_frame(case["frames"][0])
    text = A.format_program_text(lift_frame(frame, dl, T=case.get("t_provenance")))
    name = case["id"].split("/")[-1]
    (out / f"{name}.chaord").write_text(text, encoding="utf-8")
    print(name, case["dialect"], len(frame), "atoms", len(text), "chars",
          "T", case.get("t_provenance"), flush=True)
