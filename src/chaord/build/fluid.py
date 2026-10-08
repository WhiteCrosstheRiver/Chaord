"""Fluid builder: molecules + density -> packed (and optionally relaxed) frame."""
from __future__ import annotations

import numpy as np

from ..build.molecules import TEMPLATES, pack_molecules, species_mass
from ..io.frames import Frame
from ..lang.errors import ChaordError
from ..lang.ir import Program


def _num(v) -> float:
    s = v.num if hasattr(v, "num") else str(v)
    if "/" in s:
        a, b = s.split("/")
        return float(a) / float(b)
    return float(s)


def _molecule_counts(region) -> dict[str, int]:
    counts = {}
    for s in region.statements:
        if s.key == "molecules":
            vals = s.values
            i = 0
            while i + 1 < len(vals):
                if vals[i].t == "n":
                    name = vals[i].text
                    n = int(_num(vals[i + 1])) if vals[i + 1].t == "q" else 1
                    counts[name] = counts.get(name, 0) + n
                i += 2
    return counts


def _packs_monatomic(dialect) -> bool:
    """Does the dialect pack monatomic ion templates with the census-safe
    molecular rule? (Molecular yes; reduced-unit dialects such as lj place
    them with their own rsa hard core instead.)"""
    try:
        dialect.threshold("packing_contact_margin")
    except ChaordError:
        return False
    return True


def _box_from_density(counts, density_stmt, program) -> np.ndarray:
    """Cubic box edge from mass and target g/cm3 density (auto cell)."""
    total_mass = sum(species_mass(name) * n for name, n in counts.items())
    rho = _num(density_stmt.values[0])  # g/cm3
    vol_amu = total_mass / rho / 0.6022140857  # A^3  # dialect-exempt: exact-geometry
    edge = float(vol_amu ** (1 / 3))
    return np.array([edge, edge, edge])


def _temperature_K(system) -> float | None:
    """`state T` as Kelvin (an eV statement converts through ase); None when
    the program states no temperature (the backend's reference default)."""
    from ase import units
    tstmt = system.get("T")
    if tstmt is None:
        return None
    tv = tstmt.values[0]
    T = _num(tv)
    if getattr(tv, "unit", None) == "eV":
        T = T / float(units.kB)
    return T


def build_fluid(program: Program, dialect, rng, physics=True, md_steps=None) -> Frame:
    regions = [b for b in program.blocks if b.t == "region"]
    if len(regions) != 1:
        raise ChaordError("fluid builder expects exactly one region")
    region = regions[0]
    system = {}
    for b in program.blocks:
        if b.t == "system":
            system = {s.key: s for s in b.statements if s.kind in ("build", "state", "conserve")}

    counts = _molecule_counts(region)
    if not counts:
        # atomic fluid: conserve atoms defines the species and count
        conserve = system.get("atoms")
        if conserve is None:
            raise ChaordError("fluid program needs molecules or conserve atoms")
        vals = conserve.values
        i = 0
        while i + 1 < len(vals):
            if vals[i].t == "n" and vals[i + 1].t == "q":
                counts[vals[i].text] = int(_num(vals[i + 1]))
            i += 2

    cell_vals = []
    if "cell" in system:
        cell_vals = [v for v in system["cell"].values if v.t == "q"]
        if any(v.t == "n" and v.text == "auto" for v in system["cell"].values):
            cell_vals = []  # `cell auto cubic`: derive from the density
    if len(cell_vals) == 3:
        L = np.array([_num(v) for v in cell_vals], float)
    else:
        dens = next((s for s in region.statements
                     if s.kind == "state" and s.key == "density"), None)
        if dens is None:
            raise ChaordError("auto cell needs a state density (g/cm3)")
        L = _box_from_density(counts, dens, program)

    # static density check when both cell and density are stated
    dens = next((s for s in region.statements if s.kind == "state" and s.key == "density"), None)
    if dens is not None and dens.values and dens.values[0].t == "q" and (
            dens.values[0].unit == "g/cm3"):
        total_mass = sum(species_mass(name) * n for name, n in counts.items())
        rho = total_mass / float(np.prod(L)) / 0.6022140857  # dialect-exempt: exact-geometry
        stated = _num(dens.values[0])
        tol = float(dialect.threshold("density_match_tolerance"))
        if abs(rho - stated) > tol * stated:
            raise ChaordError(
                f"stated density {stated:.3f} g/cm3 does not match the cell "
                f"({rho:.3f} g/cm3); impossible density (A12)")

    # every named species with a template packs together (monatomic ions
    # included: their template pins the element and the census-safe contact);
    # only template-less names (bare elements of hand-written programs) fall
    # through to the atomic hard-core rule. Monatomic templates under a
    # dialect without the molecular packing rule (lj: covalent radii are
    # Angstrom-based, meaningless in sigma) also use the hard-core rule with
    # that dialect's own rsa dmin.
    mono_packs = _packs_monatomic(dialect)
    templated = {k: v for k, v in counts.items()
                 if k in TEMPLATES and (len(TEMPLATES[k]["symbols"]) > 1 or mono_packs)}
    atomic = {k: v for k, v in counts.items() if k not in templated}
    frame = None
    if templated:
        from .molecules import PackingJam, grid_pack_molecules
        try:
            frame = pack_molecules(templated, L, rng, dialect)
        except PackingJam:
            # W8 (review open_items_v2.md step 2): the whole-molecule RSA
            # jammed on its census-safe spheres (CO2 at liquid density) --
            # fall back to the grid + minimise packing the molecular
            # reference path uses (grid-centred placement + rigid-body
            # contact-opening minimisation), which clears the W8 contact
            # floor or fails closed itself.
            frame = grid_pack_molecules(templated, L, rng, dialect)

    if atomic:
        # atomic species: place with the hard-core rule (M0 rsa)
        from .slab import rsa
        # the frame carries element symbols: every placed name must be one
        for s in atomic:
            species_mass(s)  # raises ChaordError for non-element names
        atomic_syms = [s for s, n in atomic.items() for _ in range(n)]
        dmin = float(dialect.threshold("fluid_rsa_dmin"))
        existing = frame.pos if frame is not None else np.zeros((0, 3))
        n_atomic = sum(atomic.values())
        pos = rsa(existing, L, float(L[2] / 2), float(L[2] / 2), n_atomic, dmin, rng)
        if frame is not None:
            frame = Frame(pos=np.vstack([frame.pos, pos]), cell=np.diag(L),
                          symbols=frame.symbols + atomic_syms, pbc=(True, True, True))
        else:
            frame = Frame(pos=pos, cell=np.diag(L), symbols=atomic_syms,
                          pbc=(True, True, True))

    if not physics:
        return frame

    # physics prior: relax with the stated backend when we have one
    physics_block = next((b for b in program.blocks if b.t == "physics"), None)
    backend = "classical"
    if physics_block:
        for s in physics_block.statements:
            if s.key == "backend":
                backend = s.values[0].text
    if backend not in ("lj", "eam", "classical"):
        # Review 2: no silent skip. A backend with no core realization is an
        # error; the packed frame is available through the explicit
        # physics=False opt-out, never by accident of the backend name.
        raise ChaordError(
            f"no realize backend for {backend!r}; available: lj, eam, "
            f"classical (build with physics=False for the packed frame alone)")

    if backend == "classical":
        # ASE realization (Review 2): TIP4P water for the molecular lifts;
        # water + Joung-Cheatham ions for solutions. Rigid water is kept
        # rigid; the calculator choice honours the dialect's ase_backend
        # mapping and the protocol its ase_md section, when defined.
        from ..realize.ase_backend import ASEBackend, wrap_molecular
        L3 = frame.cell_diag
        ase = ASEBackend(L3, frame.symbols, "classical", dialect=dialect)
        r = wrap_molecular(ase.relax(np.mod(frame.pos, L3), rng,
                                     T_K=_temperature_K(system),
                                     md_steps=md_steps),
                           frame.symbols, L3)
        return Frame(pos=r, cell=frame.cell, symbols=frame.symbols,
                     pbc=frame.pbc)

    if backend == "eam":
        # analytic Finnis-Sinclair relaxation (eV / A units, dialect parameters)
        from ..realize.eam import EAM
        from ..realize.lj import run_md
        try:
            pots = dialect.threshold("eam_potentials")
        except ChaordError:
            # no analytic parameters in this dialect: tabulated ASE EAM on
            # the shipped potential file (FIRE + Langevin, physical units)
            from ..realize.ase_backend import ASEBackend
            L3 = frame.cell_diag
            ase = ASEBackend(L3, frame.symbols, "eam", dialect=dialect)
            r = np.mod(ase.relax(np.mod(frame.pos, L3), rng,
                                 T_K=_temperature_K(system),
                                 md_steps=md_steps), L3)
            return Frame(pos=r, cell=frame.cell, symbols=frame.symbols,
                         pbc=frame.pbc)
        present = sorted(set(frame.symbols))
        unknown = [s for s in present if s not in pots]
        if unknown:
            raise ChaordError(
                f"eam backend has no potential parameters for {', '.join(unknown)}; "
                f"parameterised species: {', '.join(sorted(pots))}")
        md = dialect.threshold("eam_md")
        tstmt = system.get("T")
        if tstmt is not None:
            tv = tstmt.values[0]
            T = _num(tv)
            if getattr(tv, "unit", None) != "eV":  # Kelvin (or unitless) -> eV
                T = T * float(md["kB_eV_per_K"])
        else:
            T = float(md["reference_T_K"]) * float(md["kB_eV_per_K"])
        L3 = frame.cell_diag
        rc = max(max(float(pots[s]["c"]), float(pots[s]["d"])) for s in present)
        eam = EAM(L3, rc=rc, skin=float(md["skin"]), species_params=pots,
                  symbols=frame.symbols)
        r = np.mod(frame.pos, L3)
        v = np.zeros_like(r)
        r, v = run_md(r, v, L3, int(md["relax_steps_fast"]), float(md["relax_dt_fast"]),
                      T, float(md["relax_gamma_fast"]), rng, lj=eam,
                      fcap=float(md["fcap"]))
        steps = int(md_steps if md_steps is not None else md["relax_steps"])
        r, v = run_md(r, v, L3, steps, float(md["relax_dt"]), T,
                      float(md["relax_gamma"]), rng, lj=eam)
        return Frame(pos=np.mod(r, L3), cell=frame.cell, symbols=frame.symbols,
                     pbc=frame.pbc)

    from ..realize.lj import LJ, run_md
    T = _num(system["T"].values[0]) if "T" in system else None
    if T is None:
        T = float(dialect.threshold("md_reference_T")) if dialect.names[-1] == "lj" else None
    if T is None:
        # Review 2: no silent skip. The physics prior cannot run without a
        # temperature; returning the packed frame would hand out a sample the
        # program never described.
        raise ChaordError(
            "lj backend needs a temperature: state T in the system block "
            "or build with physics=False for the packed frame alone")
    md = dialect.threshold("md")
    L3 = frame.cell_diag
    lj = LJ(L3, rc=float(md["relax_rc"]), skin=float(md["skin"]))
    r = np.mod(frame.pos, L3)
    v = np.zeros_like(r)
    r, v = run_md(r, v, L3, int(md["relax_steps_fast"]), float(md["relax_dt_fast"]),
                  T, float(md["relax_gamma_fast"]), rng, lj=lj, fcap=float(md["fcap"]))
    steps = int(md_steps if md_steps is not None else md["relax_steps"])
    r, v = run_md(r, v, L3, steps, float(md["relax_dt"]), T, float(md["relax_gamma"]),
                  rng, lj=lj)
    return Frame(pos=np.mod(r, L3), cell=frame.cell, symbols=frame.symbols,
                 pbc=frame.pbc)
