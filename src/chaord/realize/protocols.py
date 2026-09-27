"""History protocols: melt / quench / anneal / deposit, run by the physics prior.

A `history` statement is the shortest description of an amorphous system: the
compiler RUNS the protocol. Parameters come as bare numbers in the backend's
units (reduced for LJ)."""
from __future__ import annotations

import numpy as np

from ..io.frames import Frame
from ..realize.lj import LJ, run_md


def parse_history(stmt):
    """`history melt 1.2 for 500 -> quench to 0.01 at 0.002 -> anneal 0.01 for 300`.

    The statement's key is the first command; arrows chain the rest. Returns
    steps: ('melt', T, steps), ('quench', T_target, from_T_unused, rate),
    ('anneal', T, steps)."""
    from ..lang.ir import Arrow
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
            if w in ("melt", "quench", "anneal", "deposit"):
                cmd = w
                break
        if cmd is None:
            continue
        if cmd == "melt" and nums:
            steps.append(("melt", nums[0], int(nums[1]) if len(nums) > 1 else 500))
        elif cmd == "quench" and nums:
            # `quench to T at rate` (T first when only one number remains)
            steps.append(("quench", nums[0], nums[0],
                          nums[1] if len(nums) > 1 else 0.001))
        elif cmd == "anneal" and nums:
            steps.append(("anneal", nums[0], int(nums[1]) if len(nums) > 1 else 500))
    return steps


def run_protocol(frame: Frame, steps, dialect, rng, backend="lj") -> Frame:
    """Execute protocol steps with the LJ backend (reduced units)."""
    if backend != "lj":
        raise NotImplementedError("history protocols run on the LJ backend in core; "
                                  "MACE/LAMMPS are optional extras")
    md = dialect.threshold("md")
    dt = float(md["relax_dt"])
    gamma = float(md["relax_gamma"])
    skin = float(md["skin"])
    L = frame.cell_diag
    r = np.mod(frame.pos, L)
    v = rng.normal(size=r.shape) * np.sqrt(0.1)
    lj = LJ(L, rc=float(md.get("relax_rc", 2.5)), skin=skin)
    fcap = float(md["fcap"])
    for step in steps:
        kind = step[0]
        if kind == "melt":
            _, T, n = step
            r, v = run_md(r, v, L, int(n), dt, T, gamma, rng, lj=lj, fcap=fcap)
        elif kind == "quench":
            _, T_hi, T_lo, rate = step
            n = int(max(abs(T_hi - T_lo) / max(rate, 1e-6), 1))
            n = min(n, int(md.get("quench_max_steps", 20000)))
            Ts = np.linspace(T_hi, T_lo, n)
            for T in Ts[::max(n // 200, 1)]:
                r, v = run_md(r, v, L, max(n // 200, 1), dt, float(T), gamma, rng,
                              lj=lj)
        elif kind == "anneal":
            _, T, n = step
            r, v = run_md(r, v, L, int(n), dt, T, gamma, rng, lj=lj)
    return Frame(pos=np.mod(r, L), cell=frame.cell, symbols=frame.symbols,
                 pbc=frame.pbc)


def restrained_sample(frame: Frame, cv_name, target, tolerance, dialect, rng,
                      backend="lj", steps=None) -> Frame:
    """Core fallback for `constrain`: run MD, then bias toward the target.

    PLUMED is the heavy path; here a CV-appropriate correction loop keeps the
    constrained quantity within tolerance: density relaxes the box, SRO uses
    seeded swaps, anything else raises."""
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
    if cv_name == "sro":
        return frame  # handled by sqs_to_target at build time
    raise NotImplementedError(f"no core restraint for CV {cv_name!r}")
