# -*- coding: utf-8 -*-
"""Starting configurations and LAMMPS inputs for the four macrostates. All four
share the potential (LJ, rc 2.5), T and density; only the structure differs."""
import numpy as np

STATES = ("liquid", "fcc", "hcp", "coex")


def lattice(state, rho):
    """Atom positions and orthogonal box lengths of the starting lattice."""
    if state == "hcp":
        a = (np.sqrt(2) / rho) ** (1 / 3)
        c = a * np.sqrt(8 / 3)
        basis = np.array([[0, 0, 0], [.5, .5, 0], [.5, 5 / 6, .5], [0, 1 / 3, .5]])
        cell, reps = np.array([a, a * np.sqrt(3), c]), (12, 7, 6)
    else:
        a = (4 / rho) ** (1 / 3)
        basis = np.array([[0, 0, 0], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]])
        cell, reps = np.array([a, a, a]), ((6, 6, 28) if state == "coex" else (8, 8, 8))
    grid = np.array([[i, j, k] for i in range(reps[0]) for j in range(reps[1])
                     for k in range(reps[2])], float)
    pos = np.vstack([(g + basis) * cell for g in grid])
    return pos, cell * np.array(reps, float)


def write_data(path, pos, box):
    with open(path, "w", newline="\n") as fh:
        fh.write(f"macrostate start\n\n{len(pos)} atoms\n1 atom types\n\n")
        for x, L in zip("xyz", box):
            fh.write(f"0 {L:.10f} {x}lo {x}hi\n")
        fh.write("\nMasses\n\n1 1.0\n\nAtoms # atomic\n\n")
        for i, p in enumerate(pos, 1):
            fh.write(f"{i} 1 {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}\n")


HEAD = """units lj
atom_style atomic
boundary p p p
read_data start.data
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0
neighbor 0.3 bin
neigh_modify every 1 delay 0 check yes
timestep 0.005
thermo 2000
"""


def reference_input(state, T, seed, box, equil, prod, every):
    """Preparation (state-specific), equilibration, then frames."""
    s = HEAD + f"velocity all create {2 * T} {seed} dist gaussian\n"
    if state == "liquid":
        s += "fix m all nvt temp 3.0 3.0 0.5\nrun 10000\nunfix m\n"
        s += f"fix r all nvt temp 3.0 {T} 0.5\nrun 10000\nunfix r\n"
    elif state == "coex":
        z = box[2] / 2
        s += f"region top block INF INF INF INF {z} INF units box\n"
        s += "group top region top\ngroup bot subtract all top\n"
        s += "velocity bot set 0 0 0\nfix f bot setforce 0 0 0\n"
        s += "fix m top nvt temp 3.0 3.0 0.5\nrun 10000\nunfix m\n"
        s += f"fix r top nvt temp 3.0 {T} 0.5\nrun 10000\nunfix r\nunfix f\n"
        s += f"velocity bot create {T} {seed + 1} dist gaussian\n"
    s += f"fix e all nvt temp {T} {T} 0.5\nrun {equil}\n"
    s += f"dump d all custom {every} prod.dump id x y z\ndump_modify d sort id\n"
    s += f"run {prod}\n"
    return s


def rescore_input(T, seed, equil, prod, every):
    """The MD every rebuild receives, whatever its program said."""
    s = HEAD + "minimize 1e-4 1e-6 200 2000\nreset_timestep 0\n"
    s += f"velocity all create {T} {seed} dist gaussian\n"
    s += f"fix e all nvt temp {T} {T} 0.5\nrun {equil}\n"
    s += f"dump d all custom {every} prod.dump id x y z\ndump_modify d sort id\n"
    s += f"run {prod}\n"
    return s


def read_dump(path):
    """List of (positions, box lengths) from a sorted `custom id x y z` dump."""
    frames, lines = [], open(path).read().split("\n")
    i = 0
    while i < len(lines) and lines[i].startswith("ITEM: TIMESTEP"):
        n = int(lines[i + 3])
        bounds = [l.split() for l in lines[i + 5:i + 8]]
        lo = np.array([float(b[0]) for b in bounds])
        box = np.array([float(b[1]) for b in bounds]) - lo
        xyz = np.array([l.split()[1:4] for l in lines[i + 9:i + 9 + n]], float)
        frames.append((xyz - lo, box))
        i += 9 + n
    return frames


def run_lammps(cwd):
    """Run in.l in `cwd`, one serial LAMMPS (PyPI wheel 2024.8.29.3.0) per call."""
    import subprocess
    import sys
    code = ("from lammps import lammps; "
            "l = lammps(cmdargs=['-log', 'log.l', '-screen', 'none']); "
            "l.file('in.l'); l.close()")
    return subprocess.run([sys.executable, "-c", code], cwd=cwd,
                          capture_output=True, text=True)
