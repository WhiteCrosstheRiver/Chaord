# Prototype (Lennard-Jones, numpy + scipy only)

A minimal end-to-end demonstration of lift and build for a solid–liquid slab. It is a reference for
behaviour and numbers, not code to extend: M0 ports the ideas into `src/chaord/` properly.

| File | Role |
| --- | --- |
| `ljmd.py` | Lennard-Jones MD (BAOAB Langevin, cKDTree neighbour lists) |
| `make_snapshot.py` | Builds the fcc crystal + liquid snapshot with 3 planted vacancies (`snap.npz`, `snap_later.npz`) |
| `passes.py` | Lechner–Dellago averaged q6, Otsu threshold |
| `decompile.py` | Lift passes: phase profile and tanh interface fit, lattice fit, site matching, defect lifting, liquid statistics; `to_text` prints Chaord v0.1 |
| `compile_prog.py` | Build: crystal + defects by depth, random-sequential liquid packing, optional MD with the physics prior |
| `roundtrip.py` | Lift → build (with and without MD) → lift; compares held-out observables with a later frame of the same run |

Known limitations (deliberate, for M0 to fix): slab geometry along z only; one species; fixed dialect
constants in code (q6 > 0.32, CN cut-off 1.5σ, margin 2.5σ); no canonical ordering beyond what the slab gives;
LJ only; `constrain` statements are printed but not yet enforced by the compiler.
