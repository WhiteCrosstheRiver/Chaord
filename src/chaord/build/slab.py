"""Solid-liquid slab builder: the M0 compiler path.

Ported from prototype/compile_prog.py. Uses only build/state statements (cell, T,
crystal slab + lattice + strain + defects by depth, liquid density); `assert`
lines are never used to construct — they are checked afterwards. Every threshold
comes from the dialect by name.

Multi-species interface programs (crystal lattice + molecular `molecules`
liquid, e.g. Cu | H2O): the crystal block builds its sites exactly, the
molecular liquid packs by random sequential insertion of whole rigid molecules
with the crystal atoms as excluded obstacles (the bench generator's protocol:
reject below bond_tolerance x covalent radii sum + slab_packing_margin), and
physics relaxes each region under its own backend (see _relax_per_region).
"""
from __future__ import annotations

import numpy as np

from ..io.frames import Frame
from ..lang.errors import ChaordError
from ..lang.ir import Program, Quantity, RegionBlock, Statement
from ..realize.eam import EAM
from ..realize.lj import LJ, mic, run_md, wrap


def _mz(dz, Lz):
    return dz - Lz * np.round(dz / Lz)


def _num(v) -> float:
    """Numeric value of a Quantity token (fractions such as one-third supported)."""
    if isinstance(v, Quantity):
        s = v.num
    else:
        s = str(v)
    if "/" in s:
        a, b = s.split("/")
        return float(a) / float(b)
    return float(s)


def _find_stmts(program: Program):
    """Collect the statements the builder consumes, by block and key."""
    out = {"system": {}, "physics": {}, "regions": [], "residual": None}
    for b in program.blocks:
        if b.t in ("system", "physics"):
            for s in b.statements:
                if s.kind in ("build", "state", "conserve"):
                    out[b.t][s.key] = s
        elif b.t == "region":
            out["regions"].append(b)
        elif b.t == "residual" and not b.none:
            out["residual"] = b
    return out


def _slab_params(region: RegionBlock, L):
    """(axis, lo, hi, zc, H) from a region whose geometry is a single z-slab."""
    geo = region.geometry
    if len(geo.parts) != 1 or geo.parts[0].t != "slab":
        raise ChaordError(
            f"region {region.name!r}: the M0 builder supports a single slab geometry")
    slab = geo.parts[0]
    if slab.axis != "z":
        raise ChaordError(f"region {region.name!r}: the M0 builder supports slabs along z only")
    lo = _num(slab.rng.lo)
    hi = _num(slab.rng.hi)
    Lz = L[2]
    span = np.mod(hi - lo, Lz)
    H = span / 2
    zc = np.mod(lo + H, Lz)
    return lo, hi, zc, H


def _region_stmt_map(region: RegionBlock) -> dict[str, Statement]:
    return {s.key: s for s in region.statements if s.kind in ("build", "state", "constrain")}


def _conserve_species(system: dict) -> dict[str, int]:
    """{symbol: count} of `conserve atoms`. The M0 single-species form
    (`conserve atoms X 512`) reads as {'X': 512}; a bare count keeps the
    placeholder species 'X'."""
    if "atoms" not in system:
        raise ChaordError("system.conserve atoms is required (never drop an atom)")
    vals = system["atoms"].values
    out: dict[str, int] = {}
    i = 0
    while i + 1 < len(vals):
        if vals[i].t == "n":
            out[vals[i].text] = int(_num(vals[i + 1]))
        i += 2
    if not out:                        # `conserve atoms 512` without a species
        out["X"] = int(_num(vals[-1]))
    return out


def _molecule_counts(region: RegionBlock) -> dict[str, int]:
    """{species: count} of a region's `molecules` statements."""
    counts: dict[str, int] = {}
    for s in region.statements:
        if s.key != "molecules":
            continue
        vals = s.values
        i = 0
        while i + 1 < len(vals):
            if vals[i].t == "n":
                n = int(_num(vals[i + 1])) if vals[i + 1].t == "q" else 1
                counts[vals[i].text] = counts.get(vals[i].text, 0) + n
            i += 2
    return counts


def _species_composition(name: str) -> dict[str, int]:
    """Element multiset behind a species name (template symbols when the name
    is a known molecule, else the name parsed as an element formula)."""
    from ..build.molecules import TEMPLATES
    if name in TEMPLATES:
        out: dict[str, int] = {}
        for s in TEMPLATES[name]["symbols"]:
            out[s] = out.get(s, 0) + 1
        return out
    # bare element or formula: reuse the canonical parser of the check module
    from ..check.statics import _ion_composition
    comp = _ion_composition(name)
    if comp is None:
        raise ChaordError(
            f"unknown species {name!r}; known molecules: "
            f"{', '.join(sorted(TEMPLATES))}; otherwise a bare element symbol")
    out = {}
    for s in comp:
        out[s] = out.get(s, 0) + 1
    return out


def rsa(existing, L, zc, H, n, dmin, rng):
    """Random sequential addition in the slab |z - zc| < H (hard-core rule only)."""
    nc = np.maximum((L / dmin).astype(int), 1)
    cs = L / nc
    grid = {}

    def key(p):
        return (int(p[0] // cs[0]) % nc[0], int(p[1] // cs[1]) % nc[1], int(p[2] // cs[2]) % nc[2])

    for p in wrap(existing, L):
        grid.setdefault(key(p), []).append(tuple(p))
    out = []
    d2 = dmin * dmin
    Lx, Ly, Lz = L
    while len(out) < n:
        p = (rng.random() * Lx, rng.random() * Ly, (zc + (2 * rng.random() - 1) * H) % Lz)
        k = key(p)
        ok = True
        for a in (-1, 0, 1):
            for b in (-1, 0, 1):
                for c in (-1, 0, 1):
                    for q in grid.get(((k[0] + a) % nc[0], (k[1] + b) % nc[1], (k[2] + c) % nc[2]), ()):
                        dx = q[0] - p[0]
                        dx -= Lx * round(dx / Lx)
                        dy = q[1] - p[1]
                        dy -= Ly * round(dy / Ly)
                        dz = q[2] - p[2]
                        dz -= Lz * round(dz / Lz)
                        if dx * dx + dy * dy + dz * dz < d2:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            grid.setdefault(k, []).append(p)
            out.append(p)
    return np.array(out)


def _feasible_dmin(dialect, dmin: float, n: int, volume: float) -> float:
    """Placement hard core capped by RSA feasibility.

    Hard spheres above the random-sequential saturation density never finish;
    the dialect's rsa_max_packing_fraction bounds the volume fraction the
    packer may target. Above it, the hard core shrinks to the feasible value
    and the MD prior (its force cap) relaxes the closer contacts -- the
    placement is a prior, not the physics."""
    phi = float(dialect.threshold("rsa_max_packing_fraction"))
    if n <= 0 or volume <= 0:
        return dmin
    rho = n / volume
    return min(dmin, float((6 * phi / (np.pi * rho)) ** (1 / 3)))


def rsa_bounded(existing, L, zc, H, n, dmin, rng, dialect):
    """rsa() with a feasibility-capped hard core and a placement budget:
    running out of attempts is an honest ChaordError, never an unbounded
    spin (A13)."""
    max_tries = int(float(dialect.threshold("rsa_max_tries")))
    dmin_eff = _feasible_dmin(dialect, dmin, n, 2 * H * float(L[0]) * float(L[1]))
    state = {"tries": 0}

    class _Budget:
        """rng wrapper that raises when the budget is exhausted."""

        def __init__(self, inner):
            self.inner = inner

        def random(self, *a):
            state["tries"] += 1
            if state["tries"] > max_tries * max(n, 1):
                raise ChaordError(
                    f"cannot RSA-place {n} atoms at hard core {dmin_eff:.2f} "
                    f"in the region (budget {max_tries * max(n, 1)} tries)")
            return self.inner.random(*a)

    out = rsa(existing, L, zc, H, n, dmin_eff, _Budget(rng))
    return out, dmin_eff


def _pack_molecules_slab(counts: dict[str, int], L, zc, H, obstacles, obstacle_syms,
                         rng, dialect):
    """Random-sequential packing of rigid molecules into the slab |z - zc| < H,
    the crystal atoms as excluded obstacles (bench generator protocol).

    A placement stands only if every new atom stays slab_packing_margin beyond
    the bond-graph threshold (bond_tolerance x covalent radii sum) from every
    existing atom, crystal or placed molecule, and every atom of the molecule
    stays inside the slab shrunk by the template extent -- the census of the
    packed frame is exact by construction. Deterministic for a given seed."""
    from ase.data import chemical_symbols, covalent_radii
    from scipy.spatial import cKDTree

    from .molecules import TEMPLATES, random_rotation

    tol = float(dialect.threshold("bond_tolerance"))
    margin = float(dialect.threshold("slab_packing_margin"))
    max_tries = int(float(dialect.threshold("slab_packing_max_tries")))
    Lx, Ly, Lz = L

    def r_of(s: str) -> float:
        return tol * covalent_radii[chemical_symbols.index(s)]

    obst = wrap(np.asarray(obstacles, float), L)
    obst_rad = np.array([r_of(s) for s in obstacle_syms]) if len(obst) else np.zeros(0)
    tree = cKDTree(obst, boxsize=L) if len(obst) else None
    placed_pos: list[np.ndarray] = []
    placed_rad: list[np.ndarray] = []
    syms: list[str] = []
    tries = 0
    total = sum(counts.values())
    for name, n in counts.items():
        if name not in TEMPLATES:
            raise ChaordError(
                f"no template for molecule {name!r}; known: {', '.join(sorted(TEMPLATES))}")
        t = TEMPLATES[name]
        inset = float(t["radius"]) + margin       # keeps whole molecules in the slab
        band = H - inset
        if band <= 0:
            raise ChaordError(
                f"liquid slab too thin to hold {name} molecules "
                f"(half-height {H:.2f}, molecule extent {inset:.2f})")
        new_rad = np.array([r_of(s) for s in t["symbols"]])
        placed = 0
        while placed < n:
            tries += 1
            if tries > max_tries * max(total, 1):
                raise ChaordError(
                    f"cannot place {total} molecules in the liquid slab "
                    f"(budget {max_tries * max(total, 1)} tries)")
            centre = rng.uniform(0, L)
            centre[2] = zc + (2 * rng.random() - 1) * band
            R = random_rotation(rng)
            atoms = t["rel"] @ R.T + centre
            dz = _mz(atoms[:, 2] - zc, Lz)
            ok = bool((np.abs(dz) < band).all())
            if ok and tree is not None:
                for ia in range(len(atoms)):
                    reach = new_rad[ia] + float(obst_rad.max(initial=0.0)) + margin  # dialect-exempt: numerical-guard: empty-obstacle init
                    for j in tree.query_ball_point(atoms[ia], reach):
                        d = atoms[ia] - obst[j]
                        d -= L * np.round(d / L)
                        if np.linalg.norm(d) < new_rad[ia] + obst_rad[j] + margin:
                            ok = False
                            break
                    if not ok:
                        break
            if ok and placed_pos:
                old = np.vstack(placed_pos)
                old_rad = np.concatenate(placed_rad)
                d = atoms[:, None, :] - old[None, :, :]
                d -= L * np.round(d / L)
                if (np.linalg.norm(d, axis=2)
                        < (new_rad[:, None] + old_rad[None, :] + margin)).any():
                    ok = False
            if not ok:
                continue
            placed_pos.append(atoms)
            placed_rad.append(new_rad)
            syms.extend(t["symbols"])
            placed += 1
    pos = (np.vstack(placed_pos) if placed_pos else np.zeros((0, 3)))
    return pos, syms


def _temperature_K(system: dict) -> float | None:
    """`state T` as Kelvin (unitless or K statements are Kelvin already)."""
    if "T" not in system:
        return None
    return float(_num(system["T"].values[0]))


def _relax_crystal_region(pos, symbols, L, T_stmt, rng, md_steps, dialect,
                          crystal_species):
    """Relax the crystal region with the metal EAM prior.

    The dialect's analytic Finnis-Sinclair parameters (eam_potentials,
    calibrated to the experimental lattice constant) run through the same
    run_md protocol as the fluid builder's eam branch; the ASE tabulated
    fallback (bench/reference/potentials/Cu_u3.eam) covers a species the
    dialect does not parameterise. Both are metal-metal only."""
    from ..realize.lj import run_md

    try:
        pots = dialect.threshold("eam_potentials")
    except ChaordError:
        pots = None
    if pots is None or crystal_species not in pots:
        # no analytic parameters: tabulated ASE EAM on the shipped potential
        # (valid only for Cu, its own species); slow but exact
        if crystal_species != "Cu":
            raise ChaordError(
                f"no interface EAM potential for {crystal_species!r}; the "
                "dialect parameterises "
                f"{', '.join(sorted(pots)) if pots else 'no species'} and the "
                "shipped tabulated potential is Cu (build with physics=False "
                "for the packed frame alone)")
        from ..realize.ase_backend import ASEBackend
        ase = ASEBackend(L, list(symbols), "eam", dialect=dialect)
        T_K = T_stmt if T_stmt is not None else None
        return np.mod(ase.relax(np.mod(pos, L), rng, T_K=T_K, md_steps=md_steps), L)

    md = dialect.threshold("eam_md")
    if T_stmt is not None:
        T = float(T_stmt) * float(md["kB_eV_per_K"])   # Kelvin -> eV (run_md units)
    else:
        T = float(md["reference_T_K"]) * float(md["kB_eV_per_K"])
    rc = max(float(pots[crystal_species]["c"]), float(pots[crystal_species]["d"]))
    eam = EAM(L, rc=rc, skin=float(md["skin"]), species_params=pots,
              symbols=list(symbols))
    r = np.mod(pos, L)
    v = np.zeros_like(r)
    r, v = run_md(r, v, L, int(md["relax_steps_fast"]), float(md["relax_dt_fast"]),
                  T, float(md["relax_gamma_fast"]), rng, lj=eam, fcap=float(md["fcap"]))
    steps = int(md_steps if md_steps is not None else md["relax_steps"])
    r, v = run_md(r, v, L, steps, float(md["relax_dt"]), T,
                  float(md["relax_gamma"]), rng, lj=eam)
    return np.mod(r, L)


def _relax_per_region(pos, symbols, L, T_stmt, rng, md_steps, dialect, crystal_species):
    """Physics of a multi-species interface build: per-region relaxation.

    Protocol decision (review 2026-09-29): the backend set has no published
    potential for the metal-water cross terms (the EAM priors are
    metal-metal, TIP4P is water-water), so each region relaxes under its own
    backend inside the shared cell and the interface is NOT relaxed across.
    This is stated honestly here and in the reference manual; a future
    cross-term potential replaces the whole routine.
    """
    from ..realize.ase_backend import ASEBackend, wrap_molecular

    L = np.asarray(L, float)
    syms = np.array(symbols, dtype=object)
    cmask = syms == crystal_species
    out = pos.copy()

    if cmask.any():
        out[cmask] = _relax_crystal_region(
            pos[cmask], [str(s) for s in syms[cmask]], L, T_stmt, rng,
            md_steps, dialect, crystal_species)

    if (~cmask).any():
        liquid_syms = [str(s) for s in syms[~cmask]]
        ase = ASEBackend(L, liquid_syms, "classical", dialect=dialect)
        out[~cmask] = wrap_molecular(
            ase.relax(pos[~cmask], rng, T_K=T_stmt, md_steps=md_steps),
            liquid_syms, L)
    return out


def build_slab(program: Program, dialect, rng, physics: bool = True, md_steps=None) -> Frame:
    """Compile a solid-liquid program into a Frame (one microstate, seeded).

    Single-species slabs build exactly as the M0 path always did. A program
    whose liquid region carries `molecules` statements is a multi-species
    interface: the crystal block builds its sites exactly and the molecules
    RSA-pack into the liquid slab around the crystal as excluded obstacles.
    """
    parts = _find_stmts(program)
    system, phys = parts["system"], parts["physics"]
    if "cell" not in system:
        raise ChaordError("system.cell is required")
    cell_vals = [_num(v) for v in system["cell"].values if v.t == "q"]
    if len(cell_vals) != 3:
        raise ChaordError("system.cell needs three numbers")
    L = np.array(cell_vals, float)
    Lx, Ly, Lz = L
    T = _num(system["T"].values[0]) if "T" in system else None
    species = _conserve_species(system)

    crystal = next((r for r in parts["regions"] if r.phase == "crystal"), None)
    liquid = next((r for r in parts["regions"] if r.phase == "liquid"), None)
    if crystal is None or liquid is None:
        raise ChaordError("the M0 builder needs one crystal and one liquid region")

    cs = _region_stmt_map(crystal)
    ls = _region_stmt_map(liquid)
    mol_counts = _molecule_counts(liquid)

    # species attribution: molecules statements account for their elements;
    # exactly one crystal species must remain (the unary lattice rule: it is
    # named in conserve atoms)
    crystal_species = None
    crystal_expected = None
    if mol_counts:
        accounted: dict[str, int] = {}
        for name, n in mol_counts.items():
            for el, k in _species_composition(name).items():
                accounted[el] = accounted.get(el, 0) + k * n
        rest = [s for s, n in species.items() if n - accounted.get(s, 0) > 0]
        if len(rest) != 1:
            raise ChaordError(
                f"multi-species slab needs exactly one crystal species beside "
                f"the molecules; candidates: {rest or 'none'}")
        crystal_species = rest[0]
        crystal_expected = species[crystal_species] - accounted.get(crystal_species, 0)
        for s, n in species.items():
            unexplained = n - accounted.get(s, 0) - (crystal_expected if s == crystal_species else 0)
            if unexplained != 0:
                raise ChaordError(
                    f"conserve atoms {s} {n} is not explained by the regions "
                    f"(molecules account for {accounted.get(s, 0)}, crystal "
                    f"{crystal_expected if s == crystal_species else 0})")
    elif len(species) == 1:
        crystal_species, crystal_expected = next(iter(species.items()))
    else:
        symbol = system["atoms"].values[0].text if system["atoms"].values[0].t == "n" else "X"
        crystal_species, crystal_expected = symbol, sum(species.values())

    a_x = _num(cs["a"].values[0])
    strain = 0.0  # dialect-exempt: numerical-guard: neutral init, replaced when a strain statement exists
    if "strain" in cs:
        vals = [v for v in cs["strain"].values if v.t == "q"]
        strain = _num(vals[0])
    a_z = a_x * (1 + strain / 100)

    _, _, zcr, Hc = _slab_params(crystal, L)
    _, _, zliq, Hl = _slab_params(liquid, L)

    # crystal: in-plane spacing commensurate with the cell, z spacing from the strain statement
    kx = int(round(Lx / (a_x / 2)))
    dx = Lx / kx
    ky = int(round(Ly / (a_x / 2)))
    dy = Ly / ky
    dz = dx * (a_z / a_x)
    nplanes = int(round(2 * Hc / dz))
    xs = []
    for k in range(nplanes):
        z = zcr + (k - (nplanes - 1) / 2) * dz
        for i in range(kx):
            for j in range(ky):
                if (i + j + k) % 2 == 0:
                    xs.append((i * dx, j * dy, z))
    X = np.array(xs)

    # defects: remove sites at the stated depth; in-plane position is free (symmetry)
    depth = Hc - np.abs(_mz(X[:, 2] - zcr, Lz))
    keep = np.ones(len(X), bool)
    for s in crystal.statements:
        if s.key != "defect":
            continue
        vals = s.values
        count = depth_target = None
        for i, v in enumerate(vals):
            if v.t == "n" and v.text == "count" and i + 1 < len(vals):
                count = int(_num(vals[i + 1]))
            if v.t == "n" and v.text == "depth" and i + 1 < len(vals):
                depth_target = _num(vals[i + 1])
        if count is None or depth_target is None:
            raise ChaordError(f"defect statement needs count and depth: {s.key}")
        for _ in range(count):
            cand = np.where(keep & (np.abs(depth - depth_target) < dz / 2 + 1e-9))[0]  # dialect-exempt: numerical-guard: fp tolerance on the depth band
            if len(cand) == 0:
                cand = np.where(keep)[0][np.argsort(np.abs(depth[keep] - depth_target))[:20]]
            if _ == 0:
                first = X[rng.choice(cand)]
            else:  # neighbouring site for divacancy / trivacancy
                dd = np.linalg.norm(mic(X[cand] - first, L), axis=1)
                cand = cand[np.argsort(dd)[:1]]
            idx = cand[0] if _ else np.where((X == first).all(1))[0][0]
            keep[idx] = False
    X = X[keep]

    if mol_counts and len(X) != crystal_expected:
        # multi-species: both regions pin their counts exactly (molecules
        # statements + crystal sites); a mismatch is a program inconsistency.
        # The single-species M0 program pins only the total (the liquid takes
        # N - sites), so no check applies there.
        raise ChaordError(
            f"crystal block builds {len(X)} {crystal_species} sites but "
            f"conserve atoms states {crystal_expected}; the program is "
            f"inconsistent (never drop an atom)")

    if mol_counts:
        # multi-species: molecules RSA-pack into the liquid slab with the
        # crystal sites as excluded obstacles (generator protocol)
        W, water_syms = _pack_molecules_slab(
            mol_counts, L, zliq, Hl, X, [crystal_species] * len(X), rng, dialect)
        from ..realize.ase_backend import wrap_molecular
        W = wrap_molecular(W, water_syms, L)
        r = np.r_[X, W].astype(float)
        symbols = [crystal_species] * len(X) + list(water_syms)
        if physics:
            backend = (phys["backend"].values[0].text if "backend" in phys
                       else "eam")
            if backend == "lj":
                raise ChaordError(
                    "multi-species interface builds need a physical backend; "
                    "physics.backend lj cannot realize molecules")
            r = _relax_per_region(r, symbols, L, _temperature_K(system), rng,
                                  md_steps, dialect, crystal_species)
        return Frame(pos=r, cell=np.diag(L), symbols=symbols,
                     pbc=(True, True, True))

    # liquid: stated density over the liquid slab, placed with a hard-core rule only
    nliq = sum(species.values()) - len(X)  # exact conservation: counts are stated
    dmin = float(dialect.threshold("rsa_dmin"))
    shrink = float(dialect.threshold("liquid_margin_shrink"))
    Y, dmin_eff = rsa_bounded(X, L, zliq, Hl - shrink, nliq, dmin, rng, dialect)
    r = np.r_[X, Y].astype(float)
    symbol = crystal_species

    if physics and T is not None:
        backend = (phys["backend"].values[0].text if "backend" in phys
                   else "lj")
        if backend != "lj":
            # physical-unit dialect: per-region ASE relaxation (metal slabs)
            return Frame(pos=_relax_per_region(r, [symbol] * len(r), L,
                                               _temperature_K(system), rng,
                                               md_steps, dialect, symbol),
                         cell=np.diag(L), symbols=[symbol] * len(r),
                         pbc=(True, True, True))
        md = dialect.threshold("md")
        lj_rc = _num(phys["cutoff"].values[0]) if "cutoff" in phys else float(dialect.threshold("default_cutoff"))
        lj = LJ(L, rc=lj_rc, skin=float(md["skin"]))
        v = np.zeros_like(r)
        r, v = run_md(r, v, L, int(md["relax_steps_fast"]), float(md["relax_dt_fast"]),
                      T, float(md["relax_gamma_fast"]), rng, lj=lj, fcap=float(md["fcap"]))
        steps = int(md_steps if md_steps is not None else md["relax_steps"])
        r, v = run_md(r, v, L, steps, float(md["relax_dt"]), T, float(md["relax_gamma"]), rng, lj=lj)
    return Frame(pos=wrap(r, L), cell=np.diag(L), symbols=[symbol] * len(r),
                 pbc=(True, True, True))
