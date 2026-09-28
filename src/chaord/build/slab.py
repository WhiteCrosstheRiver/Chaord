"""Solid-liquid slab builder: the M0 compiler path.

Ported from prototype/compile_prog.py. Uses only build/state statements (cell, T,
crystal slab + lattice + strain + defects by depth, liquid density); `assert`
lines are never used to construct — they are checked afterwards. Every threshold
comes from the dialect by name.
"""
from __future__ import annotations

import numpy as np

from ..io.frames import Frame
from ..lang.errors import ChaordError
from ..lang.ir import Program, Quantity, RegionBlock, Statement
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


def build_slab(program: Program, dialect, rng, physics: bool = True, md_steps=None) -> Frame:
    """Compile an LJ solid-liquid program into a Frame (one microstate, seeded)."""
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
    if "atoms" not in system:
        raise ChaordError("system.conserve atoms is required (never drop an atom)")
    N = int(_num(system["atoms"].values[-1]))
    symbol = system["atoms"].values[0].text if system["atoms"].values[0].t == "n" else "X"

    crystal = next((r for r in parts["regions"] if r.phase == "crystal"), None)
    liquid = next((r for r in parts["regions"] if r.phase == "liquid"), None)
    if crystal is None or liquid is None:
        raise ChaordError("the M0 builder needs one crystal and one liquid region")

    cs = _region_stmt_map(crystal)
    ls = _region_stmt_map(liquid)
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

    # liquid: stated density over the liquid slab, placed with a hard-core rule only
    nliq = N - len(X)  # exact conservation: total atom count is stated in the program
    dmin = float(dialect.threshold("rsa_dmin"))
    shrink = float(dialect.threshold("liquid_margin_shrink"))
    Y = rsa(X, L, zliq, Hl - shrink, nliq, dmin, rng)
    r = np.r_[X, Y].astype(float)

    if physics and T is not None:
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
