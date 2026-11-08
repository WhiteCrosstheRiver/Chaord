"""Fluid restraints: `constrain cn` and `constrain gr_peak` by reverse Monte Carlo.

Design rule 7 (one definition per quantity): the restrained quantities are the
ones the fluid lift measures and prints, `lift.fluid._cn_stats` and
`lift.fluid._rdf_stats`. The Monte Carlo keeps an incremental pair histogram on
exactly that grid (`gr_rmax_fluid`, `gr_bins_fluid`) and, when it stops, the
result is re-measured with those two functions; a target that is not met raises
(fail closed), it is never printed as met.

Scope: atomic fluids only. Molecular fluids are refused, because their lift
measures molecule centres and a rigid-body move set would be needed. All tuning
values are dialect thresholds (`rmc_*`); a dialect without them raises.
"""
from __future__ import annotations

import numpy as np

from ..io.frames import Frame
from ..lang.errors import ChaordError

SUPPORTED = ("cn", "gr_peak")


def _num(v) -> float:
    s = v.num if hasattr(v, "num") else str(v)
    if "/" in s:
        a, b = s.split("/")
        return float(a) / float(b)
    return float(s)


def fluid_constraints(region, dialect) -> list[dict]:
    """Parse the region's `constrain` statements into restraint targets.

    `constrain cn <target> [+- tol] cutoff <rc>`
    `constrain gr_peak <r> height <h> [+- tol]`
    """
    out = []
    for s in region.statements:
        if s.kind != "constrain":
            continue
        if s.key not in SUPPORTED:
            raise ChaordError(
                f"constrain {s.key}: no fluid restraint for this key; "
                f"supported: {', '.join(SUPPORTED)}")
        vals = s.values
        nums = [v for v in vals if v.t == "q"]
        tols = [v for v in vals if v.t == "tol"]
        tol = _num(tols[0].value) if tols else None
        if s.key == "cn":
            if not nums:
                raise ChaordError("constrain cn needs a target value")
            rc = None
            for i, v in enumerate(vals):
                if v.t == "n" and v.text == "cutoff" and i + 1 < len(vals):
                    rc = _num(vals[i + 1])
            out.append(dict(
                key="cn", target=_num(nums[0]),
                tol=tol if tol is not None else float(dialect.threshold("rmc_cn_tolerance")),
                rc=rc if rc is not None else float(dialect.threshold("cn_cutoff_fluid"))))
        else:
            if len(nums) < 2:
                raise ChaordError("constrain gr_peak needs a position and a height")
            out.append(dict(
                key="gr_peak", r=_num(nums[0]), height=_num(nums[1]),
                tol=tol if tol is not None else float(
                    dialect.threshold("rmc_gr_height_tolerance"))))
    return out


def _mic_dist(pos, i, xyz, L):
    d = pos - xyz
    d -= L * np.round(d / L)
    r = np.sqrt((d * d).sum(axis=1))
    r[i] = np.inf
    return r


class _State:
    """Pair histogram + first-shell pair counts for the current positions."""

    def __init__(self, pos, L, rmax, nbin, cuts):
        from scipy.spatial import cKDTree
        self.L, self.rmax, self.nbin = L, rmax, nbin
        self.dr = rmax / nbin
        n = len(pos)
        self.n = n
        self.rho = n / float(np.prod(L))
        edges = np.linspace(0, rmax, nbin + 1)
        self.shell = 4 * np.pi / 3 * (edges[1:] ** 3 - edges[:-1] ** 3)
        tree = cKDTree(np.mod(pos, L), boxsize=L)
        far = max([rmax] + list(cuts))
        pairs = tree.query_pairs(far, output_type="ndarray")
        d = pos[pairs[:, 1]] - pos[pairs[:, 0]]
        d -= L * np.round(d / L)
        r = np.sqrt((d * d).sum(axis=1))
        self.h = np.histogram(r[r < rmax], edges)[0].astype(np.int64)
        self.cuts = list(cuts)
        self.ncut = [int((r <= rc).sum()) for rc in self.cuts]

    def bins(self, r):
        k = (r[r < self.rmax] / self.dr).astype(np.int64)
        return np.bincount(np.minimum(k, self.nbin - 1), minlength=self.nbin)

    def g(self, h=None):
        h = self.h if h is None else h
        return 2 * h / (self.n * self.rho * self.shell)


def _cost(state, cons, h, ncut):
    g = state.g(h)
    total = 0
    ok = True
    for c in cons:
        if c["key"] == "cn":
            cn = 2 * ncut[c["_cut"]] / state.n
            dev = (cn - c["target"]) / c["tol"]
            ok = ok and abs(cn - c["target"]) <= c["tol"]
            total += dev * dev
        else:
            k = c["_bin"]
            gk = g[k]
            others = np.delete(g, k)
            excess = max(0, float(others.max()) - gk)
            dev = (gk - c["height"]) / c["tol"]
            ok = ok and abs(gk - c["height"]) <= c["tol"] and excess <= 0
            total += dev * dev + (excess / c["tol"]) ** 2
    return total, ok


def apply_fluid_restraints(frame: Frame, cons: list[dict], dialect, rng) -> Frame:
    """Reverse Monte Carlo on single-atom moves until every target is met,
    then verify with the lift's own measures. Raises when a target is missed."""
    if not cons:
        return frame
    L = frame.cell_diag
    rmax = float(dialect.threshold("gr_rmax_fluid"))
    nbin = int(dialect.threshold("gr_bins_fluid"))
    sweeps = int(dialect.threshold("rmc_max_sweeps"))
    step = float(dialect.threshold("rmc_step"))
    t0 = float(dialect.threshold("rmc_T_start"))
    t1 = float(dialect.threshold("rmc_T_end"))
    hard = float(dialect.threshold("rmc_hard_core"))
    pos = np.mod(np.array(frame.pos, float), L)
    cuts = []
    for c in cons:
        if c["key"] == "cn":
            c["_cut"] = len(cuts)
            cuts.append(c["rc"])
        else:
            c["_bin"] = min(int(c["r"] / (rmax / nbin)), nbin - 1)
    st = _State(pos, L, rmax, nbin, cuts)
    cost, ok = _cost(st, cons, st.h, st.ncut)
    n = len(pos)
    for sweep in range(sweeps):
        if ok:
            break
        temp = t0 * (t1 / t0) ** (sweep / max(sweeps - 1, 1))
        for _ in range(n):
            i = int(rng.integers(n))
            new = np.mod(pos[i] + rng.uniform(-step, step, 3), L)
            r_new = _mic_dist(pos, i, new, L)
            if r_new.min() < hard:
                continue
            r_old = _mic_dist(pos, i, pos[i], L)
            h = st.h - st.bins(r_old) + st.bins(r_new)
            nc = [st.ncut[j] - int((r_old <= rc).sum()) + int((r_new <= rc).sum())
                  for j, rc in enumerate(st.cuts)]
            c_new, ok_new = _cost(st, cons, h, nc)
            if c_new <= cost or rng.random() < np.exp((cost - c_new) / temp):
                pos[i] = new
                st.h, st.ncut, cost, ok = h, nc, c_new, ok_new
    out = Frame(pos=pos, cell=frame.cell, symbols=list(frame.symbols), pbc=frame.pbc)
    _verify(out, cons, dialect)
    return out


def _verify(frame: Frame, cons: list[dict], dialect) -> None:
    """Re-measure with the lift's own functions; raise on any missed target."""
    from ..lift.fluid import _cn_stats, _rdf_stats
    for c in cons:
        if c["key"] == "cn":
            cn, _, _ = _cn_stats(frame, dialect, rc=c["rc"])
            if abs(cn - c["target"]) > c["tol"]:
                raise ChaordError(
                    f"constrain cn {c['target']} +- {c['tol']}: not reached "
                    f"(measured {cn:.3f}); the build refuses rather than "
                    f"print an unmet constraint")
        else:
            rm, g, _ = _rdf_stats(frame, dialect)
            k = int(np.argmax(g))
            dr = float(rm[1] - rm[0])
            if abs(float(rm[k]) - c["r"]) > dr or abs(float(g[k]) - c["height"]) > c["tol"]:
                raise ChaordError(
                    f"constrain gr_peak {c['r']} height {c['height']} +- {c['tol']}: "
                    f"not reached (measured {rm[k]:.3f} height {g[k]:.3f})")
