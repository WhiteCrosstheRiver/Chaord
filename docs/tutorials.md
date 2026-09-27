# Tutorial 1 — your first crystal

Build a defect-bearing intermetallic, lift it back, and check the round trip.

```bash
cd Chaord
.venv/Scripts/python -m chaord.cli build spec/examples/02_crystal_defects.chaord -o ni3al.extxyz --seed 7
.venv/Scripts/python -m chaord.cli lift ni3al.extxyz -o lifted.chaord --dialect core+metal --mode defects
.venv/Scripts/python -m chaord.cli fmt lifted.chaord --check
.venv/Scripts/python -m chaord.cli diff spec/examples/02_crystal_defects.chaord lifted.chaord
```

What happened:
- `build` walked the IR down to coordinates (863 atoms, exact conserve counts);
- `lift` re-derived `prototype L1_2`, `composition Ni3Al`, `a 3.572 A` and the
  planted `defect V_Ni count 1` / `defect Al_Ni count 1` from the coordinates;
- `fmt --check` confirms the text is canonical;
- `diff` compares the two programs (provenance ignored): same macrostate.

# Tutorial 2 — a fluid round trip against the noise floor

```bash
printf 'chaord 0.1\n\nsystem {\n  cell 7.6 7.6 7.6\n  pbc xyz\n  state T 0.80\n  conserve atoms X 320\n}\n\nphysics {\n  backend lj\n  epsilon 1\n  sigma 1\n  cutoff 2.5\n}\n\nliquid bulk : all {\n  state density 0.73\n}\n' > liquid.chaord
.venv/Scripts/python -m chaord.cli build liquid.chaord -o a.npz --seed 23
.venv/Scripts/python -m chaord.cli lift a.npz -o p.chaord --dialect core+lj
.venv/Scripts/python -m chaord.cli build p.chaord -o b.npz --seed 29
.venv/Scripts/python tools/noise_floor.py a.npz b.npz --dialect core+lj
.venv/Scripts/python -m chaord.cli roundtrip a.npz --seed 3
```

The noise floor (`g(r)` RMS between two frames of one simulation) defines the
tolerance; the round trip passes when the rebuild is no further from the
original than 1.5x that floor — never a guessed number.

# Tutorial 3 — surfaces, adsorbates, and the shortest program

```bash
.venv/Scripts/python - <<'EOF'
from chaord.integrations import system, physics, crystal, program, defect
from chaord.lang.api import format_program
p = program(
    system(cell=[7.669, 6.641, 18.26], seed=3, conserve={"Ni": 36}),
    physics(backend="eam"),
    crystal("slab", lattice="fcc", a=3.615),
)
print(format_program(p))
EOF
```

The scripting layer lowers to exactly the canonical text; from Python you can
assemble programs, then hand them to `build`. Lift a slab frame with
`--mode surface` to recover `surface (111) top`, `termination Ni` and
`adsorb ... site top` with coverage; `tools/acceptance.py` runs the full
A1-A14 acceptance suite and writes `reports/acceptance.json`.
