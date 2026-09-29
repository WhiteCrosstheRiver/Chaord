"""History protocols: melt / quench / anneal / deposit, run by the physics prior.

A `history` statement is the shortest description of an amorphous system: the
compiler RUNS the protocol. Parameters come as bare numbers in the backend's
units (reduced for LJ)."""
from __future__ import annotations

import numpy as np

from ..io.frames import Frame
from ..lang.errors import ChaordError
from ..realize.lj import LJ, run_md

_HISTORY_CMDS = ("melt", "quench", "anneal", "deposit")
_DEPOSIT_FILLER = frozenset(_HISTORY_CMDS + ("for", "to", "at"))


def parse_history(stmt, dialect=None):
    """`history melt 1.2 for 500 -> quench to 0.01 at 0.002 -> anneal 0.01 for 300`.

    The statement's key is the first command; arrows chain the rest. Returns
    steps: ('melt', T, steps), ('quench', T_target, from_T_unused, rate),
    ('anneal', T, steps), ('deposit', species, count, steps)."""
    from ..lang.ir import Arrow
    if dialect is None:
        from ..dialects import load_dialect
        dialect = load_dialect(("lj",))
    quench_rate = float(dialect.threshold("quench_default_rate"))
    vals = list(stmt.values)
    segments = [[]]
    for v in vals:
        if isinstance(v, Arrow):
            segments.append([])
        else:
            segments[-1].append(v)

    steps = []
    for k, seg in enumerate(segments):
        cmd = None
        if k == 0:
            cmd = stmt.key if stmt.kind == "history" else None
        words = [v.text for v in seg if v.t == "n"]
        nums = [float(v.num) for v in seg if v.t == "q"]
        for w in words:
            if w in _HISTORY_CMDS:
                cmd = w
                break
        if cmd is None:
            continue
        if cmd == "melt" and nums:
            steps.append(("melt", nums[0], int(nums[1]) if len(nums) > 1 else 500))
        elif cmd == "quench" and nums:
            # `quench to T at rate` (T first when only one number remains)
            steps.append(("quench", nums[0], nums[0],
                          nums[1] if len(nums) > 1 else quench_rate))
        elif cmd == "anneal" and nums:
            steps.append(("anneal", nums[0], int(nums[1]) if len(nums) > 1 else 500))
        elif cmd == "deposit" and nums:
            # `deposit X 20 for 400`: species X, 20 atoms over 400 steps;
            # without `for` one atom lands per step
            species = next((w for w in words if w not in _DEPOSIT_FILLER), None)
            if species is None:
                raise ChaordError(
                    "deposit needs a species: `deposit X 20 for 400`")
            steps.append(("deposit", species, int(nums[0]),
                          int(nums[1]) if len(nums) > 1 else int(nums[0])))
    return steps


def _deposit_atom(r, v, L, gap, T, rng, lj, max_attempts):
    """Insert one atom at a seeded random free site.

    A site is free when its minimum-image distance to every existing atom is
    at least `gap`; the proposal loop is rejection sampling on the seeded rng.
    Returns the grown (r, v) with the new atom's velocity drawn at T."""
    from scipy.spatial import cKDTree
    tree = cKDTree(np.mod(r, L), boxsize=L)
    for _ in range(int(max_attempts)):
        p = np.mod(rng.uniform(0.0, L), L)  # dialect-exempt: numerical-guard: uniform proposal over the box
        if not tree.query_ball_point(p, gap):
            r = np.vstack([r, p[None, :]])
            v = np.vstack([v, rng.normal(size=3) * np.sqrt(T)])
            lj.pairs = None  # atom count changed: force a neighbour-list rebuild
            return r, v
    raise ChaordError(
        f"deposit found no free site with min-image gap >= {gap} after "
        f"{max_attempts} attempts; lower deposit_min_gap or the density")


def run_protocol(frame: Frame, steps, dialect, rng, backend="lj") -> Frame:
    """Execute protocol steps with the LJ backend (reduced units).

    The protocol tracks the temperature it is at: `quench to T` (parsed as
    T_hi == T_lo) quenches from wherever the protocol currently is down to T,
    because the parser cannot know the running temperature. MD here runs
    uncapped (except `deposit`): a force cap lets hot pairs tunnel through the
    repulsive core, and the overlaps it leaves behind freeze a stressed,
    wrong-peaked glass (A5)."""
    if backend != "lj":
        raise NotImplementedError("history protocols run on the LJ backend in core; "
                                  "MACE/LAMMPS are optional extras")
    md = dialect.threshold("md")
    dt = float(md["relax_dt"])
    gamma = float(md["relax_gamma"])
    skin = float(md["skin"])
    L = frame.cell_diag
    r = np.mod(frame.pos, L)
    T_cur = float(dialect.threshold("protocol_init_T"))
    v = rng.normal(size=r.shape) * np.sqrt(T_cur)
    lj = LJ(L, rc=float(md["relax_rc"]), skin=skin)
    fcap = float(md["fcap"])
    symbols = list(frame.symbols)
    for step in steps:
        kind = step[0]
        if kind == "melt":
            _, T, n = step
            r, v = run_md(r, v, L, int(n), dt, T, gamma, rng, lj=lj)
            T_cur = float(T)
        elif kind == "quench":
            _, T_hi, T_lo, rate = step
            if T_hi == T_lo:  # `quench to T`: start from the current temperature
                T_hi = T_cur
            n = int(max(abs(T_hi - T_lo) / max(rate, 1e-6), 1))  # dialect-exempt: numerical-guard: divide-by-zero guard on the rate
            n = min(n, int(md["quench_max_steps"]))
            Ts = np.linspace(T_hi, T_lo, n)
            for T in Ts[::max(n // 200, 1)]:
                r, v = run_md(r, v, L, max(n // 200, 1), dt, float(T), gamma, rng,
                              lj=lj)
            T_cur = float(T_lo)
        elif kind == "anneal":
            _, T, n = step
            r, v = run_md(r, v, L, int(n), dt, T, gamma, rng, lj=lj)
            T_cur = float(T)
        elif kind == "deposit":
            _, species, count, nsteps = step
            gap = float(dialect.threshold("deposit_min_gap"))
            attempts = int(dialect.threshold("deposit_max_attempts"))
            T_dep = float(dialect.threshold("deposit_T"))
            interval = max(int(nsteps) // int(count), 1)
            for _ in range(int(count)):
                r, v = _deposit_atom(r, v, L, gap, T_dep, rng, lj, attempts)
                symbols.append(species)
                r, v = run_md(r, v, L, interval, dt, T_dep, gamma, rng, lj=lj,
                              fcap=fcap)
    return Frame(pos=np.mod(r, L), cell=frame.cell, symbols=symbols,
                 pbc=frame.pbc)


def _restrain_cn(frame: Frame, target, tolerance, dialect, rng, steps=None) -> Frame:
    """Pull the mean coordination number to `target` with a temperature loop.

    At fixed box and count a hotter liquid is less ordered: cn falls as T rises
    and grows on cooling. Each iteration measures cn through chaord.cv.measure
    (one definition per quantity), rescales the Langevin temperature by the sign
    and size of the residual deviation, and relaxes a short MD block — at most
    restrain_max_iter times, then a hard ChaordError (never a silent miss)."""
    from ..cv import measure
    md = dialect.threshold("md")
    dt = float(md["relax_dt"])
    gamma = float(md["relax_gamma"])
    n_block = int(steps if steps is not None
                  else dialect.threshold("restrain_cn_steps"))
    gain = float(dialect.threshold("restrain_cn_gain"))
    T_lo = float(dialect.threshold("restrain_cn_T_lo"))
    T_hi = float(dialect.threshold("restrain_cn_T_hi"))
    T = float(dialect.threshold("md_reference_T"))
    L = frame.cell_diag
    r = np.mod(frame.pos, L)
    v = rng.normal(size=r.shape) * np.sqrt(T)
    lj = LJ(L, rc=float(md["relax_rc"]), skin=float(md["skin"]))

    def current() -> Frame:
        return Frame(pos=np.mod(r, L), cell=frame.cell,
                     symbols=frame.symbols, pbc=frame.pbc)

    max_iter = int(dialect.threshold("restrain_max_iter"))
    cn = measure("cn", current(), dialect)
    for _ in range(max_iter):
        if abs(cn - target) <= tolerance:
            return current()
        T = float(np.clip(T * np.exp(gain * (cn - target)), T_lo, T_hi))
        r, v = run_md(r, v, L, n_block, dt, T, gamma, rng, lj=lj,
                      fcap=float(md["fcap"]))
        cn = measure("cn", current(), dialect)
    if abs(cn - target) <= tolerance:
        return current()
    raise ChaordError(
        f"cn restraint did not reach {target} +- {tolerance} within {max_iter} "
        f"iterations (last cn {cn:.3f}); the target is outside the range the "
        f"temperature loop [{T_lo}, {T_hi}] can reach at this density")


def restrained_sample(frame: Frame, cv_name, target, tolerance, dialect, rng,
                      backend="lj", steps=None) -> Frame:
    """Core fallback for `constrain`: run MD, then bias toward the target.

    PLUMED is the heavy path; here a CV-appropriate correction loop keeps the
    constrained quantity within tolerance: density relaxes the box, cn runs a
    temperature/relaxation loop, SRO uses seeded swaps, anything else raises."""
    if cv_name == "density":
        L = frame.cell_diag
        for _ in range(int(dialect.threshold("restrain_max_iter"))):
            rho = len(frame.pos) / float(np.prod(L))
            if abs(rho - target) <= tolerance:
                break
            scale = (rho / target) ** (1 / 3)
            L = L * scale
            frame = Frame(pos=np.mod(frame.pos * scale, L), cell=np.diag(L),
                          symbols=frame.symbols, pbc=frame.pbc)
        return frame
    if cv_name == "cn":
        return _restrain_cn(frame, target, tolerance, dialect, rng, steps=steps)
    if cv_name == "sro":
        return frame  # handled by sqs_to_target at build time
    raise NotImplementedError(f"no core restraint for CV {cv_name!r}")
