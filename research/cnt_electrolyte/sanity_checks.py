# -*- coding: utf-8 -*-
"""
Post-MD sanity checks for the CNT/electrolyte reference systems.

STANDALONE: imports only stdlib + numpy + params.py (never `chaord`, house
rule). Mirrors the repo's molecular-dialect contact rule exactly:
  * intermolecular HEAVY-atom pairs must sit at >= 0.75 x the Bondi vdW sum
    (ase.data.vdw_radii set: C 1.70, O 1.52, Li 1.82, P 1.80, F 1.47,
    H 1.20 A; fraction 0.75 from src/chaord/dialects/molecular.yaml);
  * pairs involving H are outside the rule (reported separately,
    informationally, against the 1.5 A H-bond minimum of the dialect);
  * pairs involving a cation element (ion_solvation_elements = Li...) are
    exempt from the floor (an ion-solvation contact is not a vdW overlap)
    and reported separately.
Also checks: atom count per frame (no loss), composition per frame,
inside-tube density, max radial excursion (no overlap explosions ejecting
atoms through the wall), and the extra AGENTS.md floor "no pair closer than
0.8 sigma" on intermolecular heavy-atom LJ pairs.

Every number is recorded honestly; a failed check is recorded, never tuned.

Usage:  python sanity_checks.py <system> [...]     (default: all four)
Reads  runs/<system>/prod.dump (+ equil.dump, log.lammps when present)
Writes runs/<system>/sanity_<system>.json
"""
import json
import math
import os
import re
import sys

import numpy as np

import params as P

HERE = os.path.dirname(os.path.abspath(__file__))

# Bondi/ase radii actually used by the repo (ase.data.vdw_radii), hardcoded
# so this script stays standalone; values verified against the venv's ase.
VDW = {"H": 1.20, "C": 1.70, "O": 1.52, "Li": 1.82, "P": 1.80, "F": 1.47}
BONDI_FRAC = 0.75            # dialect contact_floor_bondi_fraction
HBOND_MIN = 1.5              # dialect contact_floor_hbond_min
ION_ELEMENTS = {"Li"}        # dialect ion_solvation_elements (cations)
LJ_SIGMA_FLOOR = 0.80        # AGENTS.md MD-sanity floor, x mixed sigma

ELEMENT_OF_TYPE = {t: P.ATOM_TYPES[t][0] for t in P.ATOM_TYPES}
LJ_SIGMA = {t: P.ATOM_TYPES[t][2] for t in P.ATOM_TYPES}
MASS_OF_TYPE = {t: P.AMU[P.ATOM_TYPES[t][0]] for t in P.ATOM_TYPES}


# ---------------------------------------------------------------- dump parser

def parse_dump(path):
    """Yield (timestep, boxlo(3), boxhi(3), cols dict-of-arrays)."""
    with open(path) as f:
        lines = f.read().splitlines()
    i, n = 0, len(lines)
    while i < n:
        if lines[i].startswith("ITEM: TIMESTEP"):
            step = int(lines[i + 1])
            natoms = int(lines[i + 3])
            boxlo, boxhi = [], []
            for k in range(3):
                lo, hi = lines[i + 5 + k].split()[:2]
                boxlo.append(float(lo))
                boxhi.append(float(hi))
            hdr = lines[i + 8].split()[2:]
            data = np.array([ln.split() for ln in
                             lines[i + 9: i + 9 + natoms]], dtype=float)
            cols = {name: data[:, j] for j, name in enumerate(hdr)}
            yield step, np.array(boxlo), np.array(boxhi), cols
            i += 9 + natoms
        else:
            i += 1


# ---------------------------------------------------------------- frame checks

def classify_composition(mols, types):
    """mols: per-atom molecule id; types: per-atom type -> counts dict."""
    counts = {"EC": 0, "EMC": 0, "PF6": 0, "Li": 0}
    uniq = np.unique(mols)
    for m in uniq:
        if m == 1:                      # tube
            continue
        tset = types[mols == m]
        if 10 in tset:
            counts["PF6"] += 1
        elif 9 in tset and len(tset) == 1:
            counts["Li"] += 1
        elif len(tset) == 10:
            counts["EC"] += 1
        elif len(tset) == 15:
            counts["EMC"] += 1
        else:
            counts.setdefault("UNKNOWN(%d atoms)" % len(tset), 0)
            counts["UNKNOWN(%d atoms)" % len(tset)] += 1
    return counts


def frame_checks(cols, boxlo, boxhi, expected_n, R, Lz):
    ids = cols["id"].astype(int)
    mols = cols["mol"].astype(int)
    types = cols["type"].astype(int)
    xyz = np.column_stack([cols["x"], cols["y"], cols["z"]])
    box = boxhi - boxlo
    out = {"n_atoms": int(len(ids))}

    # composition on first call pattern: count per mol
    out["composition"] = classify_composition(mols, types)

    mobile = mols > 1
    r = np.hypot(xyz[:, 0], xyz[:, 1])
    out["max_mobile_radius_A"] = float(r[mobile].max()) if mobile.any() else 0.0
    out["n_mobile_outside_tube_R"] = int((r[mobile] > R).sum())
    out["n_outside_box"] = int(((xyz < boxlo) | (xyz > boxhi)).any(axis=1).sum())

    # density inside the tube (mobile mass / pore volume).
    # rho[g/cm3] = m[amu]*1.66054e-24 / (V[nm3]*1e-21) = m/(602.214*V_nm3)
    mass = sum(MASS_OF_TYPE[t] for t in types[mobile])
    V_geom = math.pi * R ** 2 * Lz / 1e3       # nm^3
    out["density_in_tube_geom_g_cm3"] = round(mass / 602.214 / V_geom, 4)
    out["mobile_mass_amu"] = round(float(mass), 1)

    # ---- intermolecular minimum distances (minimum-image, all axes)
    idx = np.arange(len(ids))
    dmin_heavy = None
    worst = None            # (ratio, dist, floor, elem_i, elem_j)
    dmin_heavy_Li = None    # cation-exempt pairs (solvation contacts)
    dmin_any = None
    dmin_H_X = None
    worst_LJ = None         # (dist / 0.8 sigma_mix)
    # process in blocks to bound memory; pairs across blocks too
    block = 512
    elems = [ELEMENT_OF_TYPE[t] for t in types]
    for a0 in range(0, len(ids), block):
        a1 = min(a0 + block, len(ids))
        for b0 in range(a0, len(ids), block):
            b1 = min(b0 + block, len(ids))
            if b0 == a0:
                iu, ju = np.triu_indices(a1 - a0, k=1)
                ii = idx[a0:a1][iu]
                jj = idx[a0:a1][ju]
            else:
                ii = np.repeat(idx[a0:a1], b1 - b0)
                jj = np.tile(idx[b0:b1], a1 - a0)
            keep = mols[ii] != mols[jj]
            ii, jj = ii[keep], jj[keep]
            if not len(ii):
                continue
            d = xyz[ii] - xyz[jj]
            d -= box * np.round(d / box)
            dist = np.sqrt((d * d).sum(axis=1))
            if dmin_any is None:
                dmin_any = dist.min()
            else:
                dmin_any = min(dmin_any, float(dist.min()))
            ei = np.array(elems)[ii]
            ej = np.array(elems)[jj]
            heavy = (ei != "H") & (ej != "H")
            dh = dist[heavy]
            eh_i, eh_j = ei[heavy], ej[heavy]
            # 0.75 x Bondi floor, cation pairs exempt
            li_pair = (eh_i == "Li") | (eh_j == "Li")
            floors = BONDI_FRAC * (np.vectorize(VDW.get)(eh_i, 0.0)
                                   + np.vectorize(VDW.get)(eh_j, 0.0))
            if (~li_pair).any():
                ratio = dh[~li_pair] / floors[~li_pair]
                k = int(np.argmin(ratio))
                dmin_heavy = (float(dh[~li_pair].min()) if dmin_heavy is None
                              else min(dmin_heavy, float(dh[~li_pair].min())))
                cand = (float(ratio[k]), float(dh[~li_pair][k]),
                        float(floors[~li_pair][k]), str(eh_i[k]), str(eh_j[k]))
                if worst is None or cand[0] < worst[0]:
                    worst = cand
            if li_pair.any():
                dmin_heavy_Li = (float(dh[li_pair].min())
                                 if dmin_heavy_Li is None
                                 else min(dmin_heavy_Li, float(dh[li_pair].min())))
            # H-involving, informational vs hbond min
            hp = (~heavy)
            if hp.any():
                dH = float(dist[hp].min())
                dmin_H_X = dH if dmin_H_X is None else min(dmin_H_X, dH)
            # 0.8 x mixed-sigma LJ floor (AGENTS.md), heavy non-cation pairs
            si = np.array([LJ_SIGMA[t] for t in types[ii]])[heavy]
            sj = np.array([LJ_SIGMA[t] for t in types[jj]])[heavy]
            sig = 0.5 * (si + sj)
            lj_floor = LJ_SIGMA_FLOOR * sig
            nc = ~li_pair
            if nc.any():
                r2 = dh[nc] / lj_floor[nc]
                k = int(np.argmin(r2))
                wl = (float(r2[k]), float(dh[nc][k]), float(lj_floor[nc][k]))
                if worst_LJ is None or wl[0] < worst_LJ[0]:
                    worst_LJ = wl
    out["min_intermol_heavy_A"] = round(dmin_heavy, 4) if dmin_heavy else None
    out["bondi_worst"] = ({"ratio": round(worst[0], 4), "dist_A": round(worst[1], 4),
                           "floor_A": round(worst[2], 4), "pair": worst[3] + "-" + worst[4]}
                          if worst else None)
    out["min_intermol_heavy_cationpair_A"] = (round(dmin_heavy_Li, 4)
                                              if dmin_heavy_Li else None)
    out["min_intermol_any_A"] = round(dmin_any, 4) if dmin_any else None
    out["min_intermol_H_involving_A"] = (round(dmin_H_X, 4)
                                         if dmin_H_X else None)
    out["lj08_worst"] = ({"ratio": round(worst_LJ[0], 4), "dist_A": round(worst_LJ[1], 4),
                          "floor_A": round(worst_LJ[2], 4)} if worst_LJ else None)
    return out


def expected_counts(n_cnt):
    c = P.COMPOSITION[n_cnt]
    return {"EC": c["n_EC"], "EMC": c["n_EMC"], "PF6": c["n_LiPF6"],
            "Li": c["n_LiPF6"]}


def check_system(name):
    mdir = os.path.join(HERE, "runs", name)
    prod = os.path.join(mdir, "prod.dump")
    manifest = json.load(open(os.path.join(HERE, "data",
                                           "build_manifest.json")))[name]
    n_cnt = P.SYSTEMS[name]["n"]
    R = manifest["cnt_diameter_A"] / 2.0
    Lz = manifest["cnt_length_A"]
    expected_n = manifest["n_atoms_total"]
    result = {"system": name, "expected_n_atoms": expected_n,
              "expected_composition": expected_counts(n_cnt),
              "frames": [], "equil_density_vs_t": [], "errors": []}

    # production frames
    times, keys_seen = [], set()
    for step, blo, bhi, cols in parse_dump(prod):
        fc = frame_checks(cols, blo, bhi, expected_n, R, Lz)
        fc["t_ps"] = step * P.TIMESTEP_FS / 1000.0
        result["frames"].append(fc)
        if fc["n_atoms"] != expected_n:
            result["errors"].append(f"frame t={fc['t_ps']}ps: "
                                    f"{fc['n_atoms']} atoms != {expected_n}")
    result["n_prod_frames"] = len(result["frames"])
    result["frame_times_ps"] = [f["t_ps"] for f in result["frames"]]
    dt = (result["frame_times_ps"][1] - result["frame_times_ps"][0]
          if len(result["frame_times_ps"]) > 1 else None)
    result["frame_spacing_ps"] = dt

    # aggregate: worst over production frames
    def agg(key, how):
        vals = [f[key] for f in result["frames"] if f[key] is not None]
        return how(vals) if vals else None
    result["prod_min_intermol_heavy_A"] = agg("min_intermol_heavy_A", min)
    result["prod_min_intermol_any_A"] = agg("min_intermol_any_A", min)
    result["prod_min_intermol_H_involving_A"] = agg(
        "min_intermol_H_involving_A", min)
    result["prod_min_intermol_heavy_cationpair_A"] = agg(
        "min_intermol_heavy_cationpair_A", min)
    bw = min((f["bondi_worst"] for f in result["frames"]
              if f["bondi_worst"]), key=lambda w: w["ratio"], default=None)
    result["prod_bondi_worst"] = bw
    result["prod_bondi_pass"] = bool(bw and bw["ratio"] >= 1.0)
    lw = min((f["lj08_worst"] for f in result["frames"] if f["lj08_worst"]),
             key=lambda w: w["ratio"], default=None)
    result["prod_lj08_worst"] = lw
    result["prod_lj08_pass"] = bool(lw and lw["ratio"] >= 1.0)
    result["prod_density_in_tube_geom_g_cm3"] = [
        f["density_in_tube_geom_g_cm3"] for f in result["frames"]]
    V_vdw = manifest["V_pore_vdw_nm3"]
    mm = result["frames"][0]["mobile_mass_amu"]
    result["prod_density_in_tube_vdw_g_cm3"] = round(mm / 602.214 / V_vdw, 4)
    result["prod_max_mobile_radius_A"] = agg("max_mobile_radius_A", max)
    result["prod_n_mobile_outside_tube"] = sum(
        f["n_mobile_outside_tube_R"] for f in result["frames"])
    # composition stable?
    comps = {json.dumps(f["composition"], sort_keys=True)
             for f in result["frames"]}
    result["composition_stable"] = len(comps) == 1
    result["composition_observed"] = json.loads(sorted(comps)[0])

    # equilibration density trajectory (light parse)
    eq = os.path.join(mdir, "equil.dump")
    if os.path.exists(eq):
        for step, blo, bhi, cols in parse_dump(eq):
            mols = cols["mol"].astype(int)
            types = cols["type"].astype(int)
            mass = sum(MASS_OF_TYPE[t] for t in types[mols > 1])
            result["equil_density_vs_t"].append(
                {"t_ps": round(step * P.TIMESTEP_FS / 1000.0, 1),
                 "density_g_cm3": round(mass / 602.214
                                        / (math.pi * R ** 2 * Lz / 1e3), 4)})

    outp = os.path.join(mdir, f"sanity_{name}.json")
    with open(outp, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[sanity] {name}: wrote {outp}")
    print(f"   prod frames {result['n_prod_frames']} spacing {dt} ps; "
          f"N {result['frames'][0]['n_atoms']}; "
          f"comp {result['composition_observed']} stable="
          f"{result['composition_stable']}")
    print(f"   min intermol heavy {result['prod_min_intermol_heavy_A']} A "
          f"(Bondi worst ratio "
          f"{bw['ratio'] if bw else None} pair {bw['pair'] if bw else None} "
          f"pass={result['prod_bondi_pass']}); "
          f"LJ0.8 pass={result['prod_lj08_pass']}")
    dens = result["prod_density_in_tube_geom_g_cm3"]
    print(f"   in-tube density {min(dens)}..{max(dens)} g/cm3 "
          f"(geom pore); equil n={len(result['equil_density_vs_t'])}")
    if result["errors"]:
        print("   ERRORS:", *result["errors"], sep="\n    ")
    return result


def main():
    systems = sys.argv[1:] or list(P.SYSTEMS)
    summary = {}
    for name in systems:
        summary[name] = check_system(name)
    with open(os.path.join(HERE, "runs", "sanity_summary.json"), "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "frames"}
                   for k, v in summary.items()}, f, indent=2)
    print("[sanity] wrote runs/sanity_summary.json")


if __name__ == "__main__":
    main()
