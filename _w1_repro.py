"""W1 reproducer (docs/reviews/open_items_v2.md, section 2 W1).

Run from the repository root: .venv/Scripts/python.exe _w1_repro.py
"""
import sys

import numpy as np

sys.path[:0] = ["src", "tools"]
import acceptance as acc  # noqa: E402
from chaord.dialects import load_dialect  # noqa: E402
from chaord.io.frames import Frame, read_frame  # noqa: E402
from chaord.lift import lift_frame  # noqa: E402


def shifted(f, t):
    """The same frame, rigidly translated by t (A) and wrapped into the box."""
    return Frame(pos=np.mod(f.pos + np.asarray(t, float), f.cell_diag),
                 cell=f.cell, symbols=f.symbols, pbc=f.pbc)


def load_ref(path):
    """A bench/reference frame (npz with r, L, symbols)."""
    z = np.load(path, allow_pickle=True)
    L = np.asarray(z["L"], float)
    return Frame(pos=np.asarray(z["r"], float),
                 cell=np.diag(L) if L.ndim == 1 else L,
                 symbols=[str(s) for s in z["symbols"]],
                 pbc=(True, True, True))


# --- snippet 1: how many bench crystals are called amorphous -----------------
from chaord.lift.amorphous import is_amorphous  # noqa: E402

gl = load_dialect(("core", "glass"))
bad = [c["id"] for c in acc._exact_roundtrip_cases(None)
       if is_amorphous(acc._a2a3_frame(c), gl)]
print(len(bad), "of 9 bench crystals called amorphous under core + glass")
print("  ids:", bad)

# --- snippet 2: does mode='fluid' accept the dilute argon gas ----------------
ar = acc.case_by_id("fluid/ar_gas_box25")
try:
    lift_frame(read_frame(ar["frames"][0]), load_dialect(ar["dialect"]),
               mode="fluid")
    print("argon gas accepted by mode='fluid'")
except Exception as e:
    print("argon gas refused by mode='fluid':", e)
