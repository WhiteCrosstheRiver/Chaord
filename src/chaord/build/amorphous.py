"""Amorphous builder: density + history protocol -> quenched frame.

Single-species regions (the monatomic LJ glass) build through the exact
v1 path: RSA start + realize.protocols.run_protocol on the plain LJ engine.

Multi-species regions (W7 / D9: `conserve atoms A 1600 B 400`, physics
`model kob_andersen`) require a named LJ mixture model from the dialect's
lj_mixtures table; the history then runs on the pair-resolved engine
(realize.lj.LJMixture) with the same protocol semantics (melt / linear
quench / anneal, uncapped forces), and the placement is a species-blind
feasibility-capped RSA start that the melt erases.
"""
from __future__ import annotations

import sys

import numpy as np

from ..io.frames import Frame
from ..lang.errors import ChaordError
from ..lang.ir import Program
from ..realize.protocols import parse_history, run_protocol


def _num(v) -> float:
    s = v.num if hasattr(v, "num") else str(v)
    if "/" in s:
        a, b = s.split("/")
        return float(a) / float(b)
    return float(s)


def _parse_conserve_atoms(vals) -> dict[str, int]:
    """`conserve atoms A 1600 B 400` -> {'A': 1600, 'B': 400} (v1 read the
    single-species pair only; W7 allows any number of species pairs)."""
    counts: dict[str, int] = {}
    i = 0
    while i < len(vals):
        if vals[i].t == "n":
            name = vals[i].text
            if i + 1 < len(vals) and vals[i + 1].t == "q":
                counts[name] = counts.get(name, 0) + int(_num(vals[i + 1]))
                i += 2
                continue
        i += 1
    return counts


def _mixture_table(program: Program, dialect, counts: dict[str, int]) -> tuple:
    """The named lj_mixtures table entry the physics block states.

    Fail closed (Review 8 rule 3): a multi-species amorphous program whose
    physics block names no model -- or names one the dialect does not
    define, or whose species disagree with the conserve statement -- refuses
    with a ChaordError; it never silently builds a wrong potential."""
    model = None
    for b in program.blocks:
        if b.t == "physics":
            for s in b.statements:
                if s.key == "model":
                    model = s.values[0].text if s.values else None
    if model is None:
        raise ChaordError(
            "multi-species amorphous build needs a physics `model` from the "
            f"dialect's lj_mixtures table (species {sorted(counts)}); a "
            "potential cannot be guessed from counts alone")
    table = dialect.threshold("lj_mixtures")
    if model not in table:
        raise ChaordError(
            f"dialect defines no lj_mixtures model {model!r}; known: "
            + ", ".join(sorted(table)))
    entry = table[model]
    if sorted(entry["species"]) != sorted(counts):
        raise ChaordError(
            f"lj_mixtures model {model!r} serves species "
            f"{sorted(entry['species'])}, the program conserves "
            f"{sorted(counts)}")
    return entry


def _run_mixture_history(frame: Frame, steps, dialect, rng, entry: dict) -> Frame:
    """Execute melt/quench/anneal on the pair-resolved LJ engine.

    Mirrors realize.protocols.run_protocol exactly (same temperature
    tracking, same linear-quench segmentation, same uncapped forces: a force
    cap lets hot pairs tunnel through the repulsive core and freeze a
    stressed, wrong-peaked glass); run_protocol itself constructs the plain
    monatomic LJ engine, so the mixture runs here."""
    from ..realize.lj import LJMixture, run_md
    md = dialect.threshold("md")
    dt = float(md["relax_dt"])
    gamma = float(md["relax_gamma"])
    skin = float(md["skin"])
    L = frame.cell_diag
    r = np.mod(frame.pos, L)
    symbols = list(frame.symbols)
    T_cur = float(dialect.threshold("protocol_init_T"))
    v = rng.normal(size=r.shape) * np.sqrt(T_cur)
    mix = LJMixture(L, symbols, entry["epsilon"], entry["sigma"],
                    float(entry["rc_factor"]), skin=skin)
    for step in steps:
        kind = step[0]
        if kind == "melt":
            _, T, n = step
            r, v = run_md(r, v, L, int(n), dt, T, gamma, rng, lj=mix)
            T_cur = float(T)
        elif kind == "quench":
            _, T_hi, T_lo, rate = step
            if T_hi == T_lo:  # `quench to T`: start from the current temperature
                T_hi = T_cur
            n = int(max(abs(T_hi - T_lo) / max(rate, 1e-6), 1))  # dialect-exempt: numerical-guard: divide-by-zero guard on the rate
            n = min(n, int(md["quench_max_steps"]))
            Ts = np.linspace(T_hi, T_lo, n)
            for T in Ts[::max(n // 200, 1)]:
                r, v = run_md(r, v, L, max(n // 200, 1), dt, float(T), gamma,
                              rng, lj=mix)
            T_cur = float(T_lo)
        elif kind == "anneal":
            _, T, n = step
            r, v = run_md(r, v, L, int(n), dt, T, gamma, rng, lj=mix)
            T_cur = float(T)
        else:
            raise ChaordError(
                f"the mixture amorphous history supports melt/quench/anneal, "
                f"not {kind!r}")
    return Frame(pos=np.mod(r, L), cell=frame.cell, symbols=symbols,
                 pbc=frame.pbc)


def _lattice_start(counts: dict[str, int], L: float, rng) -> np.ndarray:
    """Overlap-free lattice prior for the mixture melt (W7).

    A simple-cubic k x k x k grid in the cubic box with n of its k^3 sites
    kept (seeded choice) and species assigned by seeded shuffle.  At the
    Kob-Andersen density rho* = 1.2 a random-sequential start sits exactly at
    its saturation ceiling (the feasible hard core is 0.83 sigma_AA ~
    eta 0.36) and completes only by luck in small boxes -- a lattice prior
    with every pair above `mixture_lattice_min_gap` exists for every N and
    is erased by the T* = 2 melt exactly like the RSA prior of the
    monatomic path (the placement is a prior, not the physics; the caller
    verifies the gap and falls back to bounded RSA when no grid clears it)."""
    n = sum(counts.values())
    k = max(1, int(np.ceil(n ** (1 / 3))))
    while k ** 3 < n:
        k += 1
    g = np.array(np.meshgrid(*[np.arange(k)] * 3, indexing="ij")).reshape(3, -1).T
    sites = (g + 0.5) * (L / k)  # dialect-exempt: exact-geometry: cell-centred grid sites
    return sites[rng.choice(len(sites), n, replace=False)]


def build_amorphous(program: Program, dialect, rng, physics=True,
                    md_steps=None) -> Frame:
    regions = [b for b in program.blocks if b.t == "region"]
    if len(regions) != 1 or regions[0].phase != "amorphous":
        raise ChaordError("amorphous builder expects one amorphous region")
    region = regions[0]
    system = {}
    for b in program.blocks:
        if b.t == "system":
            system = {s.key: s for s in b.statements
                      if s.kind in ("build", "state", "conserve")}

    conserve = system.get("atoms")
    if conserve is None:
        raise ChaordError("amorphous program needs conserve atoms")
    counts = _parse_conserve_atoms(conserve.values)
    n = sum(counts.values())
    species = next(iter(counts), "X")
    if n == 0:
        raise ChaordError("amorphous build needs state density and conserve atoms")

    dens = next((s for s in region.statements
                 if s.kind == "state" and s.key == "density"), None)
    if dens is None or n == 0:
        raise ChaordError("amorphous build needs state density and conserve atoms")

    rho = _num(dens.values[0])
    if dens.values[0].unit == "g/cm3":
        from ase.data import atomic_masses, chemical_symbols
        from ..build.defects import typical_neighbor_distance  # noqa: F401
        mass = atomic_masses[chemical_symbols.index(species)] if species in chemical_symbols else 1.0  # dialect-exempt: numerical-guard: unit-mass fallback for placeholder species
        rho_number = rho / mass * 0.6022140857  # atoms/A^3  # dialect-exempt: exact-geometry
    else:
        rho_number = rho
    L = float((n / rho_number) ** (1 / 3))

    # a history statement's key is its first command word (melt/quench/...),
    # so the statement is recognised by kind, not key.  As in v1, a
    # physics=False build returns the packing prior before any protocol is
    # read: the packing is a prior, not the physics.
    history = [s for s in region.statements if s.kind == "history"]
    assumed = False
    if not history:
        # Review 2: no silent skip to the random packing. A program without a
        # history line builds with the dialect's default melt-quench -- the
        # same glass_* thresholds the amorphous lifter states -- and the
        # assumption is recorded on the frame and printed to stderr.
        assumed = True

    def _steps():
        if history:
            return parse_history(history[0], dialect)
        return _default_protocol(dialect)

    if len(counts) == 1:
        # ---- v1 monatomic path (unchanged physics; the placement is now
        # the bounded packer -- rsa_bounded caps the hard core to RSA
        # feasibility and raises an honest ChaordError when the placement
        # budget is exhausted, where the raw rsa() of v1 could spin forever
        # on an infeasible density; same prior, same physics) ----
        from .slab import rsa_bounded
        dmin = float(dialect.threshold("fluid_rsa_dmin"))
        rng_start = np.random.default_rng(rng.integers(1 << 31))
        pos0, _dmin_eff = rsa_bounded(np.zeros((0, 3)), np.array([L] * 3),
                                      L / 2, L / 2, n, dmin, rng_start,
                                      dialect)
        frame = Frame(pos=pos0, cell=np.diag([L] * 3), symbols=[species] * n,
                      pbc=(True, True, True))
        if not physics:
            return frame
        out = run_protocol(frame, _steps(), dialect, rng)
        if assumed:
            out.info["assumed_history"] = "assumed default protocol from dialect"
            print("chaord: amorphous program states no history: assumed default "
                  "protocol from dialect", file=sys.stderr)
        return out

    # ---- W7 / D9 multi-species (mixture) path ----
    entry = _mixture_table(program, dialect, counts)
    rng_start = np.random.default_rng(rng.integers(1 << 31))
    # placement prior: a simple-cubic lattice start when its neighbour gap
    # clears the dialect's mixture_lattice_min_gap (sigma_AA), else the
    # feasibility-capped bounded RSA of the fluid path.  Species go on the
    # kept sites by seeded shuffle; the melt erases the prior.
    k = max(1, int(np.ceil(n ** (1 / 3))))
    while k ** 3 < n:
        k += 1
    if L / k >= float(dialect.threshold("mixture_lattice_min_gap")):
        pos0 = _lattice_start(counts, L, rng_start)
    else:
        from .slab import rsa_bounded
        dmin = float(dialect.threshold("fluid_rsa_dmin"))
        pos0, _dmin_eff = rsa_bounded(np.zeros((0, 3)), np.array([L] * 3),
                                      L / 2, L / 2, n, dmin, rng_start,
                                      dialect)
    symbols: list[str] = []
    for s in sorted(counts):
        symbols += [s] * counts[s]
    rng.shuffle(symbols)                     # seeded species assignment
    frame = Frame(pos=pos0, cell=np.diag([L] * 3), symbols=symbols,
                  pbc=(True, True, True))
    if not physics:
        return frame
    out = _run_mixture_history(frame, _steps(), dialect, rng, entry)
    if assumed:
        out.info["assumed_history"] = "assumed default protocol from dialect"
        print("chaord: amorphous program states no history: assumed default "
              "protocol from dialect", file=sys.stderr)
    return out


def _default_protocol(dialect) -> list:
    """The dialect's default melt -> quench -> anneal, as the amorphous lifter
    states it (glass_melt/quench/anneal thresholds, LJ reduced units; the
    quench rate is linear over glass_quench_steps)."""
    t_melt = float(dialect.threshold("glass_melt_T"))
    n_melt = int(dialect.threshold("glass_melt_steps"))
    t_q = float(dialect.threshold("glass_quench_T"))
    n_q = int(dialect.threshold("glass_quench_steps"))
    t_a = float(dialect.threshold("glass_anneal_T"))
    n_a = int(dialect.threshold("glass_anneal_steps"))
    rate = round((t_melt - t_q) / n_q, 6)  # dialect-exempt: numerical-guard: printable cooling rate, T per step
    # `quench to T at rate` parses to (T, T, rate): run_protocol quenches from
    # wherever the protocol currently is (after the melt)
    return [("melt", t_melt, n_melt), ("quench", t_q, t_q, rate),
            ("anneal", t_a, n_a)]
