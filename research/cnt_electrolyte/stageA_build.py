# -*- coding: utf-8 -*-
"""
Stage A builds (Review 10, work order Y6: relaxation times and filling).

Builds, per tube (10,10) and (16,16), axial length 49.2 A (as the pilot):

  1. ZERO-FIELD builds: 3 independent builds (new seeds -> new random
     orientations) reusing build_systems.build_system verbatim (same
     geometry, composition and pack-then-relax packer as the pilot four),
     plus one MIRRORED copy of build 1 (z -> Lz - z for every atom;
     positions only - velocities are drawn fresh by the run input).
  2. FILLING baths: an open-ended tube centered in a bulk 1 M LiPF6 in
     1:1 (by volume) EC:EMC bath, >= 3 nm of reservoir on each tube mouth
     and a 15 A liquid shell around the tube. The bath is built by packing
     ONE small tri-periodic liquid block at the target bulk density
     (pack-then-relax), tiling it 2x2x4, then carving out (deleting) every
     molecule that clashes with the tube atoms - so the tube starts EMPTY
     and the 5 ns NPT run is a genuine imbibition/filling experiment.

STANDALONE: imports only stdlib + numpy + this directory's own modules
(params, build_systems); never `chaord` (house rule).

Usage:
    python stageA_build.py            # everything
    python stageA_build.py zf fill ...
"""
import json
import math
import os
import sys
import time

import numpy as np

import params as P
import build_systems as B

HERE = os.path.dirname(os.path.abspath(__file__))

TUBES = (10, 16)


# ----------------------------------------------------------------------------
# mirrored copy of a built data file (z -> Lz - z for every atom)
# ----------------------------------------------------------------------------

def mirror_data(src, dst):
    """z -> Lz - z for every atom of a LAMMPS 'full' data file written by
    build_systems.write_data (id mol type q x y z rows; no velocities)."""
    with open(src) as f:
        lines = f.read().splitlines()
    # box z bounds
    zlo = zhi = None
    for l in lines:
        if l.endswith("zlo zhi"):
            zlo, zhi = (float(v) for v in l.split()[:2])
            break
    assert zlo is not None
    Lz = zhi - zlo
    i = next(k for k, l in enumerate(lines) if l.startswith("Atoms"))
    k = i + 2
    n_mirrored = 0
    while k < len(lines) and lines[k].strip():
        p = lines[k].split()
        p[6] = f"{Lz - float(p[6]):.6f}"      # row: id mol type q x y z
        lines[k] = " ".join(p)
        n_mirrored += 1
        k += 1
    with open(dst, "w") as f:
        f.write("\n".join(lines) + "\n")
    return n_mirrored, Lz


# ----------------------------------------------------------------------------
# tri-periodic bulk-liquid block packer (port of build_systems.Packer with
# full 3D PBC and no wall; same strict floors, same tolerant+relax scheme)
# ----------------------------------------------------------------------------

class PBCPacker:
    RELAX_RATE = 0.35
    RELAX_SWEEPS_PER_INSERT = 150
    RELAX_SWEEPS_FINAL = 4000
    TOLERANT_RATIO = 0.70
    strict_attempts = 5000          # fast skip-on-fail budget for the tail

    def __init__(self, box, rng):
        self.box = np.asarray(box, float).reshape(3, 2)
        self.L = self.box[:, 1] - self.box[:, 0]
        self.rng = rng
        self.pos = np.zeros((0, 3))
        self.typ = np.array([], dtype=int)
        self.bodies = []                 # (types, local, rot_matrix, com)
        self.body_tmpl = []              # template objects, parallel
        self.n_tolerant = 0

    def _flat(self):
        self.pos = np.vstack([loc @ m.T + com
                              for _t, loc, m, com in self.bodies])
        self.typ = np.concatenate([t for t, _l, _m, _c in self.bodies])

    def _own(self):
        return np.concatenate([np.full(len(t), k)
                               for k, (t, _l, _m, _c)
                               in enumerate(self.bodies)])

    def _d2(self, xyz):
        """Min-image squared distance matrix of xyz (n,3) vs placed atoms."""
        d = xyz[:, None, :] - self.pos[None, :, :]
        d -= self.L * np.round(d / self.L)
        return (d ** 2).sum(-1)

    def _ok(self, types, xyz):
        tt = np.asarray(types)
        if len(self.typ):
            d2 = self._d2(xyz)
            if np.any(d2 < B.MIN_ALLOWED[tt][:, self.typ] ** 2):
                return False
            if tt[0] == B.TYPE_OF["Li"] and B.TYPE_OF["P"] in set(self.typ.tolist()):
                pmask = self.typ == B.TYPE_OF["P"]
                if np.any(d2[:, pmask] < 4.2 ** 2):
                    return False
        return True

    def _candidate(self, template):
        types = np.asarray([t for _, t, _, _ in template])
        local = np.array([a[3] for a in template])
        best = None
        for _ in range(20000):
            com = self.box[:, 0] + self.rng.random(3) * self.L
            m = B.quat_to_mat(B.rand_quaternion(self.rng))
            xyz = local @ m.T + com
            ratio = 1.0
            if len(self.typ):
                ratios = np.sqrt(self._d2(xyz) /
                                 B.MIN_ALLOWED[types][:, self.typ] ** 2)
                ratio = float(ratios.min())
            if ratio >= 1.0:
                return xyz, m, 1.0
            if best is None or ratio > best[0]:
                best = (ratio, xyz, m)
        return (None, None, best[0]) if best is None else (best[1], best[2], best[0])

    def relax(self, sweeps):
        own = self._own()
        converged_at = None
        for sweep in range(sweeps):
            self._flat()
            pos, typ = self.pos, self.typ
            d = pos[:, None, :] - pos[None, :, :]
            d -= self.L * np.round(d / self.L)
            d2 = (d ** 2).sum(-1)
            thr = B.MIN_ALLOWED[typ][:, typ].copy() * 1.002
            np.fill_diagonal(thr, np.inf)
            viol = (d2 < thr ** 2) & (own[:, None] != own[None, :])
            ii, jj = np.nonzero(viol)
            ii, jj = ii[ii < jj], jj[ii < jj]
            if len(ii) == 0:
                converged_at = sweep
                break
            push = np.zeros((len(self.bodies), 3))
            for a, b in zip(ii, jj):
                dvec = pos[a] - pos[b]
                dvec -= self.L * np.round(dvec / self.L)
                dd = math.sqrt((dvec ** 2).sum())
                if dd < 1e-9:
                    dvec, dd = np.array([1.0, 0.0, 0.0]), 1.0
                deficit = thr[a, b] - dd
                step = 0.5 * self.RELAX_RATE * deficit * dvec / dd
                push[own[a]] += step
                push[own[b]] -= step
            for k, (_t, _l, _m, com) in enumerate(self.bodies):
                com += push[k]
                com -= self.L * np.floor((com - self.box[:, 0]) / self.L)
        self._flat()
        return converged_at

    def verify(self):
        own = self._own()
        d = self.pos[:, None, :] - self.pos[None, :, :]
        d -= self.L * np.round(d / self.L)
        d2 = (d ** 2).sum(-1)
        thr = B.MIN_ALLOWED[self.typ][:, self.typ] * 0.999
        np.fill_diagonal(thr, np.inf)
        if np.any((d2 < thr ** 2) & (own[:, None] != own[None, :])):
            return False
        return True

    def try_insert(self, template, attempts):
        types = [t for _, t, _, _ in template]
        local = np.array([a[3] for a in template])
        for _ in range(attempts):
            com = self.box[:, 0] + self.rng.random(3) * self.L
            m = B.quat_to_mat(B.rand_quaternion(self.rng))
            xyz = local @ m.T + com
            if self._ok(types, xyz):
                self.bodies.append((np.asarray(types), local, m,
                                    xyz[0] - (local @ m.T)[0]))
                self.body_tmpl.append(template)
                self._flat()
                return xyz
        return None

    def try_insert_relaxed(self, template):
        strict = self.try_insert(template, self.strict_attempts)
        if strict is not None:
            return strict
        xyz, m, ratio = self._candidate(template)
        if xyz is None or ratio < self.TOLERANT_RATIO:
            return None
        types = np.asarray([t for _, t, _, _ in template])
        local = np.array([a[3] for a in template])
        self.bodies.append((types, local, m, xyz[0] - (local @ m.T)[0]))
        self.body_tmpl.append(template)
        self.n_tolerant += 1
        self.relax(self.RELAX_SWEEPS_PER_INSERT)
        return xyz

    def drop_body(self, species):
        """Remove the first body of the given species ('PF6' or 'Li')."""
        tmap = {B.TYPE_OF["P"]: "PF6", B.TYPE_OF["Li"]: "Li"}
        for k, (t, _l, _m, _c) in enumerate(self.bodies):
            if tmap.get(t[0]) == species:
                self.bodies.pop(k)
                self.body_tmpl.pop(k)
                return True
        return False


# ----------------------------------------------------------------------------
# filling-bath construction
# ----------------------------------------------------------------------------

def tube_positions(n, ncell, z0):
    """Armchair (n,n) tube, ncell unit cells, radius R, z in [z0, z0+ncell*a]."""
    a = P.GRAPHENE_A
    R = a * n * math.sqrt(3.0) / (2.0 * math.pi)
    xyz = []
    for cell in range(ncell):
        zc = z0 + cell * a
        for k in (0, 1):
            for c in range(n):
                phiA = 2.0 * math.pi * (c + k / 2.0) / n
                phiB = phiA + 2.0 * math.pi / (3.0 * n)
                z = zc + k * a / 2.0
                xyz.append([R * math.cos(phiA), R * math.sin(phiA), z])
                xyz.append([R * math.cos(phiB), R * math.sin(phiB), z])
    return np.array(xyz), R, ncell * a


def build_block(bx, by, bz, seed):
    """Pack one tri-periodic liquid block toward block_pack_frac x the
    target bulk density, skipping (and counting) molecules that cannot be
    placed: bulk-density sequential insertion jams in a small PBC box
    (unlike the confined tube there is no free volume to relax into;
    found on execution 2026-10-15: best-of-20000 tolerant candidates fall
    below the 0.70 insertion floor near ~95% fill). Skipped salt members
    are equalized so the block stays neutral; the achieved density is
    recorded and the run's pressure anneal compresses the remainder."""
    dens = P.STAGEA["bulk_n_dens"]
    frac = P.STAGEA["block_pack_frac"]
    n_ec = int(round(dens["EC"] * bx * by * bz * frac))
    n_emc = int(round(dens["EMC"] * bx * by * bz * frac))
    n_salt = int(round(dens["LiPF6"] * bx * by * bz * frac))
    rng = np.random.default_rng(seed)
    pk = PBCPacker((0.0, bx, 0.0, by, 0.0, bz), rng)
    ec, _, _ = B.build_ec()
    emc, _ = B.build_emc()
    pf6, _ = B.build_pf6()
    li, _ = B.build_li()
    order = ([pf6] * n_salt + [li] * n_salt + [ec] * n_ec + [emc] * n_emc)
    skipped = {"EC": 0, "EMC": 0, "PF6": 0, "Li": 0}
    t0 = time.time()
    for tmpl in order:
        placed = pk.try_insert(tmpl, 400000) if tmpl[0][0] in ("Li", "P") \
            else pk.try_insert_relaxed(tmpl)
        if placed is None:
            skipped[_species(tmpl)] += 1
    # equalize skipped salt so the block stays neutral
    extra = skipped["Li"] - skipped["PF6"]
    while extra != 0:
        if extra > 0:                                # more Li skipped
            assert pk.drop_body("Li")
            extra -= 1
        else:
            assert pk.drop_body("PF6")
            extra += 1
    pk._flat()
    conv = pk.relax(pk.RELAX_SWEEPS_FINAL)
    if not pk.verify():
        raise RuntimeError("block relaxation failed to clear contact floors")
    V = bx * by * bz
    target_atoms = (dens["EC"] * 10 + dens["EMC"] * 15 +
                    dens["LiPF6"] * 8) * V           # frac = 1.0 reference
    n_of = {"EC": 0, "EMC": 0, "PF6": 0, "Li": 0}
    for tmpl in pk.body_tmpl:
        n_of[_species(tmpl)] += 1
    info = dict(block_box_A=[round(v, 4) for v in (bx, by, bz)],
                pack_frac_target=frac,
                n_EC=n_of["EC"], n_EMC=n_of["EMC"],
                n_PF6=n_of["PF6"], n_Li=n_of["Li"],
                skipped=skipped,
                achieved_frac_of_bulk=round(len(pk.pos) / target_atoms, 4),
                n_atoms=len(pk.pos), n_tolerant_inserts=pk.n_tolerant,
                final_relax_converged_sweep=conv,
                build_seed=seed, build_seconds=round(time.time() - t0, 1))
    return pk, order, info


def _species(tmpl):
    """Species of a molecule template from its ATOM COUNT (both EC and EMC
    templates start with the CO carbonyl carbon, so the first label cannot
    distinguish them; found on execution 2026-10-15)."""
    return {10: "EC", 15: "EMC", 7: "PF6", 1: "Li"}[len(tmpl)]


def build_fill(name, n, seed):
    S = P.STAGEA
    Lz = 2.0 * S["fill_reservoir_A"] + 49.2          # 109.2 A
    side = 2.0 * (P.GRAPHENE_A * n * math.sqrt(3.0) / (2.0 * math.pi)
                  + S["fill_lat_liquid_A"])
    z0 = S["fill_reservoir_A"]                       # tube spans [30, 79.2]
    tube_xyz, R, Ltube = tube_positions(n, 20, z0)

    # --- pack one block and tile it 2x2x4 to the full bath box
    bx = by = side / 2.0
    bz = Lz / 4.0
    pk, _order, block_info = build_block(bx, by, bz, seed)

    molecules = []                                   # [(template, xyz(N,3))]
    for tmpl, (_t, loc, m, com) in zip(pk.body_tmpl, pk.bodies):
        mol = loc @ m.T + com
        for i in (0, 1):
            for j in (0, 1):
                for kk in (0, 1, 2, 3):
                    off = np.array([i * bx - side / 2.0, j * by - side / 2.0,
                                    kk * bz])
                    molecules.append((tmpl, mol + off))
    n_tiled = len(molecules)

    # --- carve: delete every molecule with an atom closer than the packing
    #     floor to any tube atom (min image on the full bath box); the tube
    #     starts empty. Li/PF6 deletions are equalized in pairs so the bath
    #     stays neutral.
    L = np.array([side, side, Lz])
    floor_t = B.MIN_ALLOWED[:, B.TYPE_OF["C_tube"]]  # per atom type
    keep, deleted = [], {"EC": 0, "EMC": 0, "PF6": 0, "Li": 0}
    for tmpl, xyz in molecules:
        clash = False
        for c0 in range(0, len(tube_xyz), 200):
            tb = tube_xyz[c0:c0 + 200]
            d = xyz[:, None, :] - tb[None, :, :]
            d -= L * np.round(d / L)
            d2 = (d ** 2).sum(-1)
            if np.any(d2 < floor_t[[t for _, t, _, _ in tmpl]][:, None] ** 2):
                clash = True
                break
        if clash:
            deleted[_species(tmpl)] += 1
        else:
            keep.append((tmpl, xyz))
    extra = deleted["Li"] - deleted["PF6"]
    while extra != 0:
        if extra > 0:                                # deleted more Li than PF6
            idx = next(i for i, (t, _x) in enumerate(keep) if t[0][0] == "P")
            keep.pop(idx)
            deleted["PF6"] += 1
            extra -= 1
        else:
            idx = next(i for i, (t, _x) in enumerate(keep) if t[0][0] == "Li")
            keep.pop(idx)
            deleted["Li"] += 1
            extra += 1

    # --- assemble and write the data file (tube = molecule 1, frozen)
    counts = {"EC": 0, "EMC": 0, "PF6": 0, "Li": 0}
    mol_list = []
    mid = 1
    for tmpl, xyz in keep:
        mid += 1
        counts[_species(tmpl)] += 1
        atoms = [(lab, t, q, xyz[i]) for i, (lab, t, q, _) in enumerate(tmpl)]
        mol_list.append((mid, atoms, _bonds_of(tmpl)))
    tot_q = sum(q for _, m, _ in mol_list for _, _, q, _ in m)
    assert abs(tot_q) < 1e-6, f"{name}: bath not neutral: {tot_q}"
    box = (-side / 2.0, side / 2.0, -side / 2.0, side / 2.0, 0.0, Lz)
    natoms, nbonds = B.write_data(
        os.path.join(HERE, "data", f"{name}.data"), box, tube_xyz, mol_list,
        f"Stage A filling: open ({n},{n}) CNT in 1M LiPF6/EC:EMC bath; "
        f"block seed {seed}")

    n_liq_atoms = sum(len(m[1]) for m in mol_list)
    V = side * side * Lz
    V_nm3 = V / 1e3
    frac = P.STAGEA["block_pack_frac"]
    info = dict(
        name=name, cnt_chirality=[n, n], cnt_diameter_A=round(2 * R, 4),
        cnt_length_A=round(Ltube, 4), cnt_atoms=len(tube_xyz),
        tube_z_span_A=[z0, round(z0 + Ltube, 4)],
        box=[round(v, 4) for v in box],
        reservoir_each_side_A=S["fill_reservoir_A"],
        lateral_liquid_A=S["fill_lat_liquid_A"],
        block=block_info, tiles="2x2x4", molecules_tiled=n_tiled,
        molecules_deleted_by_carving=dict(deleted),
        composition=dict(n_EC=counts["EC"], n_EMC=counts["EMC"],
                         n_LiPF6=counts["Li"]),
        note="bath packed at "
             f"{frac:.0%} of the target bulk density (see block.pack_frac); "
             "the run's pressure anneal (fix nph target -1000 atm -> 1 atm "
             "over the 300 ps ramp stage) compresses it to 1 atm before "
             "the 5 ns production",
        bath_molarity_M_at_pack=round(counts["Li"] / (0.60221408 * V_nm3), 4),
        bath_molarity_M_after_anneal=round(
            counts["Li"] / (0.60221408 * V_nm3 * frac), 4),
        bath_atom_density_per_A3=round(n_liq_atoms / V, 5),
        n_atoms_total=natoms, n_bonds=nbonds,
        build_seed=seed,
        packer="tri-periodic block (pack-then-relax at "
               f"{frac:.0%} bulk density) tiled 2x2x4, carved "
               "around the tube; tube starts empty (imbibition start)",
    )
    return info


# bonds lookup for molecule templates (rebuilt here; build_systems returns
# bonds alongside templates only in its own calls)
_pf6_b = None
_li_b = None
_ec_b = None
_emc_b = None


def _init_bonds():
    global _pf6_b, _li_b, _ec_b, _emc_b
    _ec_b = B.build_ec()[1]
    _emc_b = B.build_emc()[1]
    _pf6_b = B.build_pf6()[1]
    _li_b = B.build_li()[1]


def _bonds_of(tmpl):
    return {10: _ec_b, 15: _emc_b, 7: _pf6_b, 1: _li_b}[len(tmpl)]


# ----------------------------------------------------------------------------

def main():
    _init_bonds()
    what = sys.argv[1:] or ["zf", "fill"]
    out = os.path.join(HERE, "data", "stageA_build_manifest.json")
    manifest = {}
    if os.path.isfile(out):                    # incremental: keep earlier
        with open(out) as f:                   # entries on restarts
            try:
                manifest = json.load(f)
            except ValueError:
                manifest = {}

    def save():
        with open(out, "w") as f:
            json.dump(manifest, f, indent=2)

    if "zf" in what:                                 # all tube builds first
        for n in TUBES:
            for b in ("b1", "b2", "b3"):
                name = f"cnt{n}_{b}"
                if name in manifest and os.path.isfile(
                        os.path.join(HERE, "data", f"{name}.data")):
                    print(f"[stageA-build] {name}: already built, skipped",
                          flush=True)
                    continue
                t0 = time.time()
                info, _ = B.build_system(name)
                info["stageA_role"] = "zero-field independent build"
                info["build_seconds"] = round(time.time() - t0, 1)
                manifest[name] = info
                save()
                print(f"[stageA-build] {name}: {info['n_atoms_total']} atoms "
                      f"({info['build_seconds']} s)", flush=True)
            name = f"cnt{n}_b1m"
            if name not in manifest or not os.path.isfile(
                    os.path.join(HERE, "data", f"{name}.data")):
                t0 = time.time()
                nmir, Lz = mirror_data(
                    os.path.join(HERE, "data", f"cnt{n}_b1.data"),
                    os.path.join(HERE, "data", f"{name}.data"))
                manifest[name] = dict(
                    name=name, stageA_role="zero-field MIRRORED copy of b1 "
                    "(z -> Lz - z, every atom, positions only; fresh "
                    "velocities in the run input)",
                    source=f"cnt{n}_b1.data", atoms_mirrored=nmir,
                    n_atoms_total=manifest[f"cnt{n}_b1"]["n_atoms_total"],
                    cnt_length_A=round(Lz, 4), build_seed=None,
                    velocity_seed=P.SEEDS[name]["velocity"],
                    build_seconds=round(time.time() - t0, 1))
                save()
            print(f"[stageA-build] {name}: mirrored "
                  f"{manifest[name]['atoms_mirrored']} atoms", flush=True)
    if "fill" in what:
        for n in TUBES:
            name = f"fill_cnt{n}"
            if name in manifest and os.path.isfile(
                    os.path.join(HERE, "data", f"{name}.data")):
                print(f"[stageA-build] {name}: already built, skipped",
                      flush=True)
                continue
            t0 = time.time()
            info = build_fill(name, n, 20264100 + n // 16)
            info["build_seconds"] = round(time.time() - t0, 1)
            manifest[name] = info
            save()
            print(f"[stageA-build] {name}: {info['n_atoms_total']} atoms, "
                  f"{info['bath_molarity_M_after_anneal']} M bath after "
                  f"anneal (packed at {info['bath_molarity_M_at_pack']} M), "
                  f"deleted {info['molecules_deleted_by_carving']} "
                  f"({info['build_seconds']} s)", flush=True)
    print("[stageA-build] wrote", out)


if __name__ == "__main__":
    main()
