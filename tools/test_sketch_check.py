"""Seed tests for WP01: the structural checker must accept the examples and reject malformed programs."""
import glob, os, sys, tempfile
sys.path.insert(0, os.path.dirname(__file__))
from sketch_check import check

HERE = os.path.dirname(os.path.abspath(__file__))
GOOD = sorted(glob.glob(os.path.join(HERE, "..", "spec", "examples", "*.chaord")))

BAD = {
    "missing header":        "system {\n  pbc xyz\n}\n",
    "unclosed block":        "chaord 0.1\nsystem {\n  pbc xyz\n",
    "pm without number":     "chaord 0.1\nliquid B : all {\n  assert cn 12.0 +- \n}\n",
    "region without colon":  "chaord 0.1\ncrystal A slab z 0 .. 10 {\n  lattice fcc\n}\n",
    "unknown shape":         "chaord 0.1\ncrystal A : blob 3 {\n  lattice fcc\n}\n",
    "unknown block":         "chaord 0.1\nwidget A {\n}\n",
    "bad range":             "chaord 0.1\ncrystal A : slab z 0 .. {\n}\n",
    "interface without bar": "chaord 0.1\ninterface A B {\n  width 1\n}\n",
}

def run():
    fails = 0
    for f in GOOD:
        try:
            check(f); print(f"ok    accepts {os.path.basename(f)}")
        except SyntaxError as e:
            fails += 1; print(f"FAIL  rejected good file {os.path.basename(f)}: {e}")
    for name, src in BAD.items():
        with tempfile.NamedTemporaryFile("w", suffix=".chaord", delete=False) as t:
            t.write(src); path = t.name
        try:
            check(path); fails += 1; print(f"FAIL  accepted bad input: {name}")
        except SyntaxError as e:
            print(f"ok    rejects {name:<22} -> {e}")
        finally:
            os.unlink(path)
    print(f"\n{len(GOOD) + len(BAD) - fails}/{len(GOOD) + len(BAD)} checks passed")
    return fails

if __name__ == "__main__":
    sys.exit(1 if run() else 0)
