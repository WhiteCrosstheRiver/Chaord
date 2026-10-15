# -*- coding: utf-8 -*-
"""
Stage A analysis (Review 10, work order Y6). numpy only, no chaord.

Measures, from the Stage A production dumps (stageA/jobs/<job>/prod.dump,
columns id mol type q x y z ix iy iz):

  * tau_rot: single-molecule dipole autocorrelation
        C1(t) = <u(0).u(t)>,  u = mu/|mu| per solvent molecule
    single-exponential fits, per species (EC 10 atoms, EMC 15 atoms).
  * P1(t) series per run (mean axial cos theta of the solvent dipoles),
    saved small to stageA/p1_series/<job>.csv.
  * tau_M three ways:
      - zero field: P1 autocorrelation C_P1(t) = <P1(t0)P1(t0+t)> - <P1>^2;
      - switch-on (wf/sf): exponential fit of the P1 rise;
      - switch-off (off): exponential fit of the P1 decay.
    Ensemble errors = sem over the independent builds (D31).
  * mirrored-pair verdict: does the zero-field P1 offset of build b1 flip
    sign in the mirrored copy b1m?
  * reversal counts (Y1's second block): molecules that never reverse
    their axial dipole (|cos| never exceeds 0.5 in both directions).
  * filling (fill_cntN): molecules/ions whose COM sits in the tube middle
    (1 nm from each mouth) and inside the tube radius; block means with
    standard errors + plateau check. The tube geometry (radius, z extent)
    is measured from the frozen tube atoms in frame 0 (the tube is never
    dilated by the barostat, so it is static).

Usage:
    python stageA_analyze.py            # every job that has a prod.dump
    python stageA_analyze.py zf_cnt10_b1 ...
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STAGEA = os.path.join(HERE, "stageA")
JOBS = os.path.join(STAGEA, "jobs")
P1DIR = os.path.join(STAGEA, "p1_series")
OUTDIR = os.path.join(STAGEA, "analysis")

EC_N, EMC_N, PF6_N = 10, 15, 7     # atoms per molecule (Li = 1)
REVERSAL_COS = 0.5                 # Y1's second-block threshold
FIT_LO, FIT_HI = 0.15, 0.85        # C1 single-exp fit window
FILL_STRIDE = 5                    # parse every 5th fill frame (5 ps)


# ----------------------------------------------------------------------------
# dump reading
# ----------------------------------------------------------------------------

def frames(path, stride=1, want_frames=None):
    """Yield (timestep, box(3,2), atoms[N,9]) from a LAMMPS custom dump
    (id mol type q x y z ix iy iz, sorted by id by dump_modify sort id).
    stride skips whole blocks without tokenizing them."""
    n_seen = 0
    with open(path) as f:
        while True:
            line = f.readline()
            if not line:
                return
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            t = int(f.readline())
            hdr = f.readline()
            assert hdr.startswith("ITEM: NUMBER OF ATOMS"), hdr
            n = int(f.readline())
            f.readline()                                   # BOX BOUNDS
            box = np.array([f.readline().split()[:2] for _ in range(3)],
                           float)
            f.readline()                                   # ATOMS header
            if stride > 1 and n_seen % stride != 0:
                for _ in range(n):
                    f.readline()
                n_seen += 1
                continue
            rows = []
            for _ in range(n):
                rows.append(f.readline().split())
            a = np.array(rows, float)
            yield t, box, a[a[:, 0].argsort()]
            n_seen += 1


def molecule_table(a):
    """mol id -> species + index arrays, from one frame (sorted by id)."""
    mol = a[:, 1].astype(int)
    umol, inv = np.unique(mol, return_inverse=True)
    species = {}
    for k, m in enumerate(umol):
        idx = np.where(inv == k)[0]
        n_at = len(idx)
        sp = {EC_N: "EC", EMC_N: "EMC", PF6_N: "PF6", 1: "Li"}.get(n_at)
        species[int(m)] = (sp, idx)
    return species


def dipoles(a, box, sol_idx):
    """Unit dipole vectors of the given molecules (min-image within the
    molecule; image flags not needed). Returns (M,3)."""
    L = box[:, 1] - box[:, 0]
    xyz = a[:, 4:7]
    q = a[:, 3]
    out = np.zeros((len(sol_idx), 3))
    for k, idx in enumerate(sol_idx):
        d = xyz[idx] - xyz[idx[0]]
        d -= L * np.round(d / L)
        mu = (q[idx, None] * d).sum(0)
        n = np.linalg.norm(mu)
        out[k] = mu / n if n > 1e-12 else 0.0
    return out


# ----------------------------------------------------------------------------
# fits and stats
# ----------------------------------------------------------------------------

def exp_fit_tau(t, y, lo=FIT_LO, hi=FIT_HI):
    """Single-exponential time constant from a decaying curve y(t):
    linear regression of ln y on t over lo<y<hi. Returns (tau, npoints)
    or (nan, 0)."""
    m = (y < hi) & (y > lo) & np.isfinite(y) & (y > 0)
    if m.sum() < 5:
        return float("nan"), 0
    slope = np.polyfit(t[m], np.log(y[m]), 1)[0]
    if slope >= -1e-12:
        return float("nan"), 0
    return -1.0 / slope, int(m.sum())


def approach_fit_tau(t, y):
    """tau for y(t) -> Pinf starting at P0: ln(Pinf - y) decay."""
    pinf = y[int(len(y) * 0.8):].mean()
    p0 = y[:5].mean()
    gap = pinf - p0
    if abs(gap) < 1e-6:
        return float("nan")
    z = (pinf - y) / gap                      # ~exp(-t/tau), starts at 1
    m = (z < 0.9) & (z > 0.1) & np.isfinite(z) & (z > 0)
    if m.sum() < 5:
        return float("nan")
    slope = np.polyfit(t[m], np.log(z[m]), 1)[0]
    return -1.0 / slope if slope < -1e-12 else float("nan")


def autocorr(x, dt, tmax):
    """<x(t0)x(t0+t)> - <x>^2 for a 1-D series x sampled every dt,
    up to tmax (all origins)."""
    x = np.asarray(x, float)
    x = x - x.mean()
    n = len(x)
    lags = np.arange(0, min(tmax, n - 1) + 1, dtype=int)
    c = np.array([ (x[:n - k] * x[k:]).mean() if k else (x * x).mean()
                   for k in lags ])
    return lags * dt, c


def block_sem(x, nblocks=8):
    x = np.asarray(x, float)
    n = (len(x) // nblocks) * nblocks
    if n == 0:
        return float("nan"), float("nan")
    b = x[:n].reshape(nblocks, -1).mean(1)
    return float(b.mean()), float(b.std(ddof=1) / math.sqrt(nblocks)
                                  if nblocks > 1 else 0.0)


# ----------------------------------------------------------------------------
# per-job analysis
# ----------------------------------------------------------------------------

def analyze_tube(job):
    """zf/wf/sf/off: dipole series, P1, tau_rot, tau_M, reversals."""
    path = os.path.join(JOBS, job["name"], "prod.dump")
    if not os.path.isfile(path):
        return None
    us, ts = [], []
    tube_R = tube_z = None
    nframes = 0
    for t, box, a in frames(path):
        if nframes == 0:
            spec = molecule_table(a)
            sol = [(m, v) for m, v in spec.items() if v[0] in ("EC", "EMC")]
            sol.sort()
            sol_idx = [v[1] for _m, v in sol]
            sol_sp = np.array([v[0] for _m, v in sol])
            tub = a[:, 2] == 1
            if tub.any():
                xyz = a[tub, 4:7]
                tube_R = float(np.mean(np.hypot(xyz[:, 0], xyz[:, 1])))
                tube_z = (float(xyz[:, 2].min()), float(xyz[:, 2].max()))
        us.append(dipoles(a, box, sol_idx))
        ts.append(t)
        nframes += 1
    if nframes < 10:
        return dict(job=job["name"], error=f"only {nframes} frames")
    us = np.array(us)                          # (T, M, 3)
    dt = (ts[1] - ts[0]) / 1000.0              # ps
    t_ax = np.arange(nframes) * dt
    uz = us[:, :, 2]
    P1 = uz.mean(1)

    res = dict(job=job["name"], kind=job["kind"], tube=job["tube"],
               build=job["build"], field=job["field"],
               n_frames=nframes, dt_ps=dt,
               tube_R_A=tube_R, tube_z_A=tube_z,
               n_solvent=int(us.shape[1]))

    # P1 series (small, committed)
    os.makedirs(P1DIR, exist_ok=True)
    with open(os.path.join(P1DIR, job["name"] + ".csv"), "w") as f:
        f.write("t_ps,P1\n")
        for tt, pp in zip(t_ax, P1):
            f.write(f"{tt:g},{pp:.6f}\n")

    # tau_rot per species + reversals (only meaningful without a field,
    # but computed everywhere for the record)
    res["tau_rot"] = {}
    for sp in ("EC", "EMC"):
        m = sol_sp == sp
        if not m.any():
            continue
        taus = []
        for j in np.where(m)[0]:
            # per-molecule C1 with all origins up to 1 ns
            x = us[:, j, :]
            kmax = min(len(t_ax) - 1, int(1000.0 / dt))
            cc = np.array([(x[:len(x) - k] * x[k:]).sum(1).mean()
                           for k in range(0, kmax + 1, max(1, int(5 / dt)))])
            tk = np.arange(len(cc)) * 5.0
            tau, _n = exp_fit_tau(tk, cc)
            taus.append(tau)
        taus = np.array(taus)
        pooled_x = us[:, m, :]
        kmax = min(len(t_ax) - 1, int(1000.0 / dt))
        cc = np.array([(pooled_x[:len(pooled_x) - k] * pooled_x[k:]).sum(-1)
                       .mean() for k in range(0, kmax + 1, max(1, int(5 / dt)))])
        tk = np.arange(len(cc)) * 5.0
        pooled_tau, _n = exp_fit_tau(tk, cc)
        res["tau_rot"][sp] = dict(
            per_molecule_tau_ps=[round(v, 1) for v in
                                 taus[np.isfinite(taus)]],
            mean_tau_ps=round(float(np.nanmean(taus)), 1) if
                            np.isfinite(taus).any() else None,
            sem_over_molecules_ps=round(float(np.nanstd(taus) /
                                              math.sqrt(np.isfinite(taus).sum())),
                                        1) if np.isfinite(taus).sum() > 1
            else None,
            pooled_tau_ps=round(pooled_tau, 1) if
                            np.isfinite(pooled_tau) else None)

    # reversal counts (Y1 second block), over solvent molecules
    never = int((~((uz > REVERSAL_COS).any(0) &
                   (uz < -REVERSAL_COS).any(0))).sum())
    res["reversals"] = dict(
        never_reversed=never, n_molecules=int(us.shape[1]),
        frac_never_reversed=round(never / us.shape[1], 3),
        per_molecule_mean_cos_sd=round(float(uz.mean(0).std()), 3))

    # P1 stats + tau_M per kind
    mean, sem = block_sem(P1)
    res["P1_mean"] = round(mean, 4)
    res["P1_block_sem"] = round(sem, 4)
    if job["kind"] == "zf":
        lag, c = autocorr(P1, dt, min(1000.0, t_ax[-1] / 2))
        tau, _n = exp_fit_tau(lag, c / c[0])
        res["tau_M_P1autocorr_ps"] = round(tau, 1) if np.isfinite(tau) else None
    if job["kind"] in ("wf", "sf"):
        res["tau_M_switch_on_ps"] = round(
            approach_fit_tau(t_ax, P1), 1)
        res["P1_final_mean"] = round(float(P1[int(len(P1) * 0.8):].mean()), 4)
    if job["kind"] == "off":
        res["tau_M_switch_off_ps"] = round(
            approach_fit_tau(t_ax, P1), 1)
    return res


def analyze_fill(job):
    path = os.path.join(JOBS, job["name"], "prod.dump")
    if not os.path.isfile(path):
        return None
    series = []
    for t, box, a in frames(path, stride=FILL_STRIDE):
        if not series:
            spec = molecule_table(a)
            mols = sorted(spec.items())
            tub = a[:, 2] == 1
            xyz = a[tub, 4:7]
            R = float(np.mean(np.hypot(xyz[:, 0], xyz[:, 1])))
            # outermost tube atoms sit a/2 = 1.23 A inside the true tube
            # ends (the last ring is half a unit cell in): extend so the
            # "1 nm from each mouth" region uses the real tube extent
            z0 = float(xyz[:, 2].min())
            z1 = float(xyz[:, 2].max()) + 1.23
        L = box[:, 1] - box[:, 0]
        xyz = a[:, 4:7]
        counts = dict(middle={}, whole={})
        for region, lo, hi in (("middle", z0 + 10.0, z1 - 10.0),
                               ("whole", z0, z1)):
            c = {"EC": 0, "EMC": 0, "Li": 0, "PF6": 0}
            for m, (sp, idx) in mols:
                if sp is None:
                    continue
                d = xyz[idx] - xyz[idx[0]]
                d -= L * np.round(d / L)
                com = xyz[idx[0]] + d.mean(0)
                if np.hypot(com[0], com[1]) <= R and lo <= com[2] <= hi:
                    c[sp] += 1
            counts[region] = c
        series.append((t, counts))
    if len(series) < 10:
        return dict(job=job["name"], error=f"only {len(series)} frames")
    ts = np.array([s[0] for s in series]) / 1000.0        # ps
    res = dict(job=job["name"], kind="fill", tube=job["tube"],
               n_frames_parsed=len(series), stride_ps=FILL_STRIDE * 1.0,
               tube_R_A=R, tube_z_A=[z0, z1],
               middle_region_A=[z0 + 10.0, z1 - 10.0])
    out = {}
    for region in ("middle", "whole"):
        out[region] = {}
        for sp in ("EC", "EMC", "Li", "PF6"):
            y = np.array([s[1][region][sp] for s in series], float)
            half = len(y) // 2
            m2, se = block_sem(y[half:])
            out[region][sp] = dict(
                mean_last_half=m2, block_sem_last_half=se,
                mean_first_half=round(float(y[:half].mean()), 2),
                plateau_delta=round(float(m2 - y[:half].mean()), 2))
    res["counts"] = out
    # series (small) for the record
    os.makedirs(P1DIR, exist_ok=True)
    with open(os.path.join(P1DIR, job["name"] + ".csv"), "w") as f:
        f.write("t_ps,EC_mid,EMC_mid,Li_mid,PF6_mid,EC_all,EMC_all,"
                "Li_all,PF6_all\n")
        for (t, c) in series:
            row = [c[r][sp] for r in ("middle", "whole")
                   for sp in ("EC", "EMC", "Li", "PF6")]
            f.write(f"{t / 1000:g}," + ",".join(str(v) for v in row) + "\n")
    return res


def main():
    which = sys.argv[1:]
    with open(os.path.join(STAGEA, "jobs_index.json")) as f:
        jobs = {j["name"]: j for j in json.load(f)}
    names = which or sorted(jobs)
    results = []
    for name in names:
        job = jobs.get(name)
        if job is None:
            print(f"[analyze] unknown job {name}", file=sys.stderr)
            continue
        try:
            r = analyze_fill(job) if job["kind"] == "fill" \
                else analyze_tube(job)
        except Exception as e:                       # noqa: BLE001
            r = dict(job=name, error=f"{type(e).__name__}: {e}")
        if r is not None:
            results.append(r)
            tail = {k: v for k, v in r.items()
                    if k in ("P1_mean", "tau_M_switch_on_ps",
                             "tau_M_switch_off_ps", "tau_M_P1autocorr_ps",
                             "error")}
            print(f"[analyze] {name}: {tail}")
    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, "per_job.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    # ---- cross-build summaries (tau_M ensembles, mirror verdict)
    summary = dict(mirror_verdict={}, tau_M_ensemble={})
    for tube in (10, 16):
        b1 = next((r for r in results if r.get("job") == f"zf_cnt{tube}_b1"),
                  None)
        bm_ = next((r for r in results if r.get("job") == f"zf_cnt{tube}_b1m"),
                   None)
        if b1 and bm_ and "P1_mean" in b1 and "P1_mean" in bm_:
            a, b = b1["P1_mean"], bm_["P1_mean"]
            sa = b1.get("P1_block_sem", 0.0) or 0.0
            sb = bm_.get("P1_block_sem", 0.0) or 0.0
            flips = (a > 0) != (b > 0)
            summary["mirror_verdict"][f"cnt{tube}"] = dict(
                b1_P1=a, b1m_P1=b, b1_sem=sa, b1m_sem=sb,
                signs_opposite=bool(flips),
                verdict=("zero-field offset flips sign in the mirrored "
                         "copy" if flips else
                         "zero-field offset does NOT flip sign in the "
                         "mirrored copy (both builds same sign)"))
        for key, kinds in (("tau_M_switch_on_wf", ("wf",)),
                           ("tau_M_switch_on_sf", ("sf",)),
                           ("tau_M_switch_off", ("off",)),
                           ("tau_M_zf_autocorr", ("zf",))):
            vals, keyfield = [], None
            for r in results:
                if r.get("tube") != tube or r.get("kind") not in kinds:
                    continue
                for kf in ("tau_M_switch_on_ps", "tau_M_switch_off_ps",
                           "tau_M_P1autocorr_ps"):
                    if kf in r and r[kf] is not None:
                        keyfield = kf
                        vals.append(r[kf])
            if vals:
                v = np.array(vals, float)
                summary["tau_M_ensemble"][f"cnt{tube}_{key}"] = dict(
                    field=key, key=keyfield,
                    mean_ps=round(float(v.mean()), 1),
                    sem_over_builds_ps=round(float(v.std(ddof=1) /
                                                    math.sqrt(len(v))), 1)
                    if len(v) > 1 else 0.0,
                    per_build=[round(float(x), 1) for x in v])
    with open(os.path.join(OUTDIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("[analyze] wrote", out, "and", os.path.join(OUTDIR, "summary.json"))


if __name__ == "__main__":
    main()
