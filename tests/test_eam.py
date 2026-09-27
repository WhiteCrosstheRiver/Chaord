"""EAM backend acceptance: symmetry, equilibrium, thermostat stability, build wiring.

Covers the analytic Finnis-Sinclair backend (realize/eam.py) wired into the fluid
builder's physics branch: (a) perfect crystals carry no force, (b) the energy has
a smooth interior minimum under uniform scaling, (c) a short thermostatted run
holds the target temperature, (d) a backend-eam program builds and conserves
atoms. Tolerances and MD settings are read from the metal dialect by name.
"""
import numpy as np
import pytest

from chaord.build import build_program
from chaord.dialects import load_dialect
from chaord.lang.api import load
from chaord.realize.eam import EAM
from chaord.realize.lj import fcc, run_md

GPA_PER_EV_A3 = 1.0 / 160.21766  # unit conversion, eV/A^3 per GPa


@pytest.fixture(scope="module")
def metal():
    return load_dialect(("core", "metal"))


@pytest.fixture(scope="module")
def pots(metal):
    return metal.threshold("eam_potentials")


def _eam(metal, pots, species, L, symbols=None):
    p = pots[species]
    return EAM(L, rc=max(p["c"], p["d"]),
               skin=float(metal.threshold("eam_md")["skin"]),
               species_params={species: p}, symbols=symbols)


def _bcc(nx, ny, nz, a):
    basis = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    grid = np.array(np.meshgrid(range(nx), range(ny), range(nz),
                                indexing="ij")).reshape(3, -1).T
    return ((grid[:, None, :] + basis[None]).reshape(-1, 3)) * a


def _crystal(species, pots):
    """Perfect periodic cell; list radius stays below half the smallest edge."""
    a = float(pots[species]["a_ref"])
    if pots[species]["lattice"] == "bcc":
        n = 4
        return _bcc(n, n, n, a), np.full(3, n * a, float)
    pos = fcc(3, 3, 3, a)
    return pos, np.full(3, 3 * a, float)


@pytest.mark.parametrize("species", ["Cu", "Fe", "Ni"])
def test_perfect_crystal_forces_vanish(metal, pots, species):
    """(a) symmetry: residual force on a perfect lattice is at the noise floor."""
    pos, L = _crystal(species, pots)
    eam = _eam(metal, pots, species, L)
    F = eam.forces(pos)
    tol = float(metal.threshold("eam_force_symmetry_tol"))
    assert np.abs(F).max() < tol
    # the calibrated set reproduces its stated cohesive energy
    e_atom = eam.energy(pos) / len(pos)
    assert e_atom == pytest.approx(-float(pots[species]["ecoh_ref"]), abs=1e-3)


@pytest.mark.parametrize("species", ["Cu", "Fe", "Ni"])
def test_energy_scan_has_smooth_interior_minimum(metal, pots, species):
    """(b) E(scale) over 0.95-1.05 is smooth and minimal at the equilibrium cell."""
    pos, L = _crystal(species, pots)
    lo = float(metal.threshold("eam_scan_lo"))
    hi = float(metal.threshold("eam_scan_hi"))
    n = int(metal.threshold("eam_scan_steps"))
    scales = np.linspace(lo, hi, n)
    # each scaled cell is a different periodic box: build the potential per point
    E = np.array([
        _eam(metal, pots, species, L * s).energy((pos * s) % (L * s)) / len(pos)
        for s in scales
    ])
    assert np.isfinite(E).all()
    step = (hi - lo) / (n - 1)
    i = int(np.argmin(E))
    # interior minimum, at the calibrated lattice constant within one scan step
    assert 0 < i < n - 1
    assert abs(scales[i] - 1.0) <= 2 * step
    # smooth: no cutoff jumps (second differences stay tiny for a parabola)
    assert np.abs(np.diff(E, 2)).max() < 0.2
    # monotone away from the minimum on both sides
    assert np.all(np.diff(E[:i + 1]) <= 0.0)
    assert np.all(np.diff(E[i:]) >= 0.0)
    # and a real well: both scan edges sit above the minimum
    assert E[0] - E[i] > 0.01 and E[-1] - E[i] > 0.01
    # curvature at the minimum reproduces the stated bulk modulus within 3x
    curv = (E[i + 1] + E[i - 1] - 2 * E[i]) / step ** 2
    a = float(pots[species]["a_ref"])
    v0 = a ** 3 / (2 if pots[species]["lattice"] == "bcc" else 4)
    b_target = 9 * float(pots[species]["bulk_modulus_ref_GPa"]) * GPA_PER_EV_A3 * v0
    assert curv == pytest.approx(b_target, rel=3.0)


def test_short_md_holds_temperature(metal, pots):
    """(c) 200-step Langevin run on 64 Cu atoms stays in the dialect band."""
    a = float(pots["Cu"]["a_ref"])
    pos = fcc(2, 2, 4, a)
    L = np.array([2 * a, 2 * a, 4 * a])
    eam = _eam(metal, pots, "Cu", L, symbols=["Cu"] * len(pos))
    md = metal.threshold("eam_md")
    T = 300.0 * float(md["kB_eV_per_K"])  # kB T in eV (run_md uses kB = 1)
    rng = np.random.default_rng(20260928)
    v = rng.normal(size=pos.shape) * np.sqrt(T)
    r = pos.copy()
    t_trace = []
    for _ in range(4):  # 4 x 50 = 200 production steps
        r, v = run_md(r, v, L, 50, float(md["relax_dt"]), T,
                      float(md["relax_gamma"]), rng, lj=eam)
        t_trace.append(np.mean(np.sum(v ** 2, 1)) / 3 / T)
    band = float(metal.threshold("eam_temperature_band"))
    assert np.isfinite(r).all() and np.isfinite(v).all()
    assert abs(np.mean(t_trace[1:]) - 1.0) < band  # steady state (last 150 steps)
    assert abs(t_trace[-1] - 1.0) < band           # final instantaneous read


def test_backend_eam_program_builds_and_conserves(metal, tmp_path):
    """(d) a backend-eam program goes through build_program with atoms conserved."""
    text = (
        "chaord 0.1\n"
        "dialect core + metal\n"
        "\n"
        "system {\n"
        "  cell 11.25 11.25 11.25 A\n"
        "  pbc xyz\n"
        "  state T 300 K\n"
        "  conserve atoms Cu 108\n"
        "}\n"
        "\n"
        "physics {\n"
        "  backend eam\n"
        "}\n"
        "\n"
        "liquid cu : all {\n"
        "  state density 8.0 g/cm3\n"
        "}\n"
    )
    path = tmp_path / "cu_eam.chaord"
    path.write_text(text)
    frame = build_program(load(path), metal, rng=np.random.default_rng(41),
                          physics=True, md_steps=200)
    assert len(frame) == 108
    assert frame.symbols.count("Cu") == 108
    assert np.all(frame.pos >= 0.0) and np.all(frame.pos < frame.cell_diag + 1e-9)
    # deterministic: same program + seed + backend -> same coordinates
    frame2 = build_program(load(path), metal, rng=np.random.default_rng(41),
                           physics=True, md_steps=200)
    assert np.allclose(frame.pos, frame2.pos, atol=1e-12)
