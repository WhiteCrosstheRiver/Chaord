import sys, numpy as np
sys.path[:0] = ["src", "tools"]
import acceptance as acc
from chaord.dialects import load_dialect
from chaord.io.frames import Frame, read_frame
from chaord.lift import lift_frame
from chaord.build import build_program

def load_ref(path):
    z = np.load(path, allow_pickle=True)
    L = np.asarray(z["L"], float)
    return Frame(pos=np.asarray(z["r"], float), cell=np.diag(L) if L.ndim == 1 else L,
                 symbols=[str(s) for s in z["symbols"]], pbc=(True, True, True))

f = load_ref("bench/reference/cu_solid_liquid/frame_0.npz")
print("symbols:", sorted(set(f.symbols)), "N:", len(f.symbols))
text = acc.format_program_text(lift_frame(f, load_dialect(("core", "metal"))))
print([l.strip() for l in text.splitlines()
       if l.strip().startswith(("units", "conserve", "state T", "backend", "lattice"))])
print(text.count("frenkel_pair"), "frenkel_pair lines")
