import sys, numpy as np
sys.path[:0] = ["src", "tools"]
import acceptance as acc
from chaord.dialects import load_dialect
from chaord.io.frames import Frame
from chaord.lift.fluid import lift_fluid, is_single_phase
from chaord.lift import lift_frame

def load_ref(path):
    z = np.load(path, allow_pickle=True)
    L = np.asarray(z["L"], float)
    return Frame(pos=np.asarray(z["r"], float), cell=np.diag(L) if L.ndim == 1 else L,
                 symbols=[str(s) for s in z["symbols"]], pbc=(True, True, True))

f = load_ref("bench/reference/cu_solid_liquid/frame_0.npz")
dl = load_dialect(("core", "metal"))
print("is_single_phase:", is_single_phase(f, dl))
try:
    p = lift_fluid(f, dl, T=300.0)
    t = acc.format_program_text(p)
    print("lift_fluid SUCCEEDS; first lines:")
    print("\n".join(t.splitlines()[:12]))
except Exception as e:
    print("lift_fluid RAISES:", type(e).__name__, e)
# also pipeline mode on this frame
try:
    p = lift_frame(f, dl, mode="pipeline")
    t = acc.format_program_text(p)
    print("pipeline SUCCEEDS:")
    print("\n".join(t.splitlines()[:10]))
    print("...provenance:", [l.strip() for l in t.splitlines() if "note" in l])
except Exception as e:
    print("pipeline RAISES:", type(e).__name__, e)
