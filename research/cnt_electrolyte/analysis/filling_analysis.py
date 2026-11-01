# -*- coding: utf-8 -*-
"""Q2 (Review 12, open_items_v6.md) -- the Y6 reservoir-filling analysis.

THE CONTRACT (Q2, verbatim): "From the fill series: molecules and ions
per Angstrom in the middle of each tube (more than 1 nm from either
mouth), block means and errors. Compare per species with the pilot
compositions per Angstrom ((10,10): EC 8, EMC 5, LiPF6 2 per 49.2 A;
(16,16): EC 37, EMC 24, LiPF6 6 per 49.2 A). Report in
reports/research/filling.md."

Inputs (evidence committed by Q1: 7e5fe1d / b994426):
  research/cnt_electrolyte/fills_cluster/fill_cnt{10,16}/final.data
  research/cnt_electrolyte/fills_cluster/fill_cnt{10,16}/log.lammps
  research/cnt_electrolyte/fills_cluster/fill_cnt{10,16}/prod_thinned.dump
  research/cnt_electrolyte/fills_cluster/fill_series_manifest.json (sha256)
  research/cnt_electrolyte/data/stageA_build_manifest.json (builder record)
  research/cnt_electrolyte/params.py (pilot composition)

Definitions (read from the records, then verified -- never assumed):
- Mouths: the builder's declared tube z-span tube_z_span_A = [30.0, 79.2]
  (both fills; cnt_length_A = 49.2 = the pilot tube length, Y6: "axial
  length 49.2 A (as the pilot, so the numbers compare)").  NOTE (recorded
  as found, Q2 step 1): the brief's "mouths at the z-boundaries / tube
  continuous through the box" does NOT describe these cells -- the tube
  is a finite open segment centred in the box with a 30 A reservoir each
  side (build_fill in stageA_build.py; the log's read_data box z = 0 to
  109.2).  The tube is molecule 1, frozen by `fix setforce` and never
  dilated (`fix nph ... dilate mobile`), so the span is constant through
  the run -- verified against every frame (V5).  The outermost tube
  ATOMS sit at z = [30.0, 77.97] (the top ring is 1.23 A inside the
  declared top mouth plane); the declared planes are used, and the
  counts under the atom-extent window are reported as a sensitivity.
- "Middle of the tube": axial window [z0+10, z0+49.2-10] = [40, 69.2] A,
  length 29.2 A (more than 1 nm from either mouth, Q2).
- Counted iff the molecule's MASS-WEIGHTED centre of mass is in the
  window AND radially inside the geometric tube cylinder (r_com < R, R
  measured from the frozen tube atoms).  The pilot counts (all molecules
  packed inside the periodic tube) satisfy the same criterion trivially,
  so the comparison is like-for-like.  Molecules are unwrapped with the
  dump image flags (orthogonal box, per-frame lengths) before the COM;
  x/y COMs are brought to the image nearest the box centre, which the
  tube axis coincides with (verified per frame).  Without the radial
  criterion the axial window alone counts BATH molecules (a first draft
  of this analysis did exactly that and reported +2686% -- the bath
  shares the tube's z range).
- Species: Li+ = the single 6.94 amu atom (type 9, q +1); PF6- = P type
  10 + 6 F type 11 (q +1.70/-0.45); EC and EMC by the per-molecule TYPE
  CENSUS of the build templates (build_systems.py TYPE_OF):
  EC  = 10 atoms  {CO:1, OS:2, CT:2, OC:1, HC:4} -> types {2:1,3:2,4:2,5:1,6:4}
  EMC = 15 atoms  {CO:1, OS:2, CT:2, OC:1, C_term:1, HC:5, HM:3}
                                          -> types {2:1,3:2,4:2,5:1,6:5,7:3,8:1}
- Blocks (Q2 brief: "2 blocks of the production half"): production =
  the 5 ns NPT at 1 atm that follows the 650 ps staged decompression
  (log.lammps: reset_timestep 0; run 5000000; dt 1 fs from the dump
  cadence "every 1000 steps = 1 ps").  Block A = t in [2500, 3750) ps,
  block B = t in [3750, 5000] ps (250/251 frames of the 5 ps series).
  Central value = per-frame mean of the half; error = between-block SEM
  = std([A,B], ddof=1)/sqrt(2).  HONEST CAVEAT (goes in the report):
  Y8 asks for blocks of >= 5 tau_M; stage_a.md measures tau_M ~ 2.7-3.0
  ns at the working field for these tubes, so a 1.25 ns block is
  ~0.4-0.5 tau_M and the between-block SEM is a LOWER bound on the true
  uncertainty; with n_blocks = 2 it equals |A-B|/2 exactly.
- D49 gate (Q3 trigger): ANY species in (16,16) whose count per
  Angstrom differs from the pilot's by more than 5%.

Validation (every check must pass; all recorded in the output JSON):
  V0 sha256 of final.data / log.lammps / prod_thinned.dump ==
     fill_series_manifest.json;
  V1 the data file's bond-graph connected components match its mol-id
     partition for every MOBILE molecule (the tube body is written
     without bonds -- solvent bond counts match the headers exactly -- so
     it is asserted separately: one body, all type 1);
  V2 data-file species totals == the builder's recorded composition
     (stageA_build_manifest.json);
  V3 per-molecule charge sums: EC/EMC 0, Li+ +1, PF6- -1, tube 0; whole
     cell neutral;
  V4 EVERY dump frame: atom ids 1..N sorted, per-atom type and mol id
     identical to the data file (so the per-molecule species map is the
     data file's by construction); per-frame cell charge ~ 0;
  V5 the tube is frozen: tube-atom coordinates identical in every frame
     (max deviation recorded); tube axis on the box centre;
  V6 log.lammps: read_data box line == the build manifest box, "atoms in
     group tube" == the build's tube atom count, "atoms in group li" ==
     the build's ion-pair count, production dump cadence 1000 steps;
     final.data written at the last dump timestep;
  V7 the series is exactly the 1/5-thinned 1 ps production cadence:
     timesteps 0..5,000,000 at exactly 5000-step (5 ps) spacing, 1001
     frames (b994426: "1/5-thinned (5 ps, 1001-frame)").

STANDALONE: imports only stdlib + numpy (+ the repo's own params.py for
the pilot composition), never `chaord` (house rule).
"""
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE.parent
sys.path.insert(0, str(RES))
import params as P  # noqa: E402  (repo config only; no side effects)

FILLS = RES / "fills_cluster"
SERIES_MANIFEST = FILLS / "fill_series_manifest.json"
BUILD_MANIFEST = RES / "data" / "stageA_build_manifest.json"
RESULTS = HERE / "filling_results.json"

MARGIN_A = 10.0            # > 1 nm from either mouth (Q2 contract)
PILOT_TUBE_A = 49.2        # fill tubes = pilot tubes (Y6), both 49.2 A
SPECIES = ("EC", "EMC", "Li+", "PF6-")

# per-molecule type multisets (module docstring; build_systems.py TYPE_OF)
SIGNATURES = {
    "EC":   {2: 1, 3: 2, 4: 2, 5: 1, 6: 4},              # 10 atoms
    "EMC":  {2: 1, 3: 2, 4: 2, 5: 1, 6: 5, 7: 3, 8: 1},  # 15 atoms
    "Li+":  {9: 1},                                      # 1 atom
    "PF6-": {10: 1, 11: 6},                              # 7 atoms
}
TUBE_TYPE = 1
EXPECTED_CHARGE = {"EC": 0.0, "EMC": 0.0, "Li+": 1.0, "PF6-": -1.0}


# ---------------------------------------------------------------------------
# provenance: sha256 of the local series vs the committed manifest (V0)
# ---------------------------------------------------------------------------

def check_sha256(name):
    man = json.load(open(SERIES_MANIFEST))[name]
    got = {}
    for fname, rec in man.items():
        h = hashlib.sha256()
        with open(FILLS / name / fname, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        got[fname] = h.hexdigest()
        assert got[fname] == rec["sha256"], \
            f"V0 FAIL: {name}/{fname} sha256 {got[fname]} != manifest"
    return got


# ---------------------------------------------------------------------------
# data-file parsing (final.data) and the molecule census
# ---------------------------------------------------------------------------

def read_data_file(path):
    """Parse a LAMMPS 'full' data file: box, Masses, Atoms, Bonds."""
    box, masses, atoms, bonds = {}, {}, [], []
    sect = None
    for line in Path(path).read_text(errors="replace").splitlines():
        s = line.strip()
        if not s:
            continue
        low = s.lower()
        if low.endswith(("xlo xhi", "ylo yhi", "zlo zhi")):
            p = s.split()
            axis = p[2][0]          # token "xlo" -> axis letter
            box[axis] = (float(p[0]), float(p[1]))
            continue
        if low.startswith("masses"):
            sect = "masses"
            continue
        if low.startswith("atoms"):
            sect = "atoms"
            continue
        if low.startswith("bonds"):
            sect = "bonds"
            continue
        if low.startswith("velocities"):
            sect = None
            continue
        if sect == "masses" and re.fullmatch(r"\d+ [\d.]+", s):
            t, m = s.split()
            masses[int(t)] = float(m)
        elif sect == "atoms":
            p = s.split()
            # full style: id mol type q x y z [ix iy iz]
            atoms.append((int(p[0]), int(p[1]), int(p[2]), float(p[3]),
                          float(p[4]), float(p[5]), float(p[6])))
        elif sect == "bonds":
            p = s.split()
            bonds.append((int(p[2]), int(p[3])))
    assert len(box) == 3 and masses and atoms and bonds, f"{path}: parse failed"
    return dict(box=box, masses=masses, atoms=np.array(atoms), bonds=bonds)


def bond_graph_components(n_atoms, bonds):
    """Connected components over atom ids 1..n_atoms (union-find)."""
    parent = np.arange(n_atoms + 1)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in bonds:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    comps = {}
    for a in range(1, n_atoms + 1):
        comps.setdefault(int(find(a)), []).append(a)
    return list(comps.values())


def species_of_types(types):
    """Species from a molecule's type multiset; raises on no match."""
    u, c = np.unique(np.asarray(types), return_counts=True)
    sig = {int(k): int(v) for k, v in zip(u, c)}
    if set(sig) == {TUBE_TYPE}:
        return "tube"       # the frozen carbon body (800/1280 x type 1)
    for sp, want in SIGNATURES.items():
        if sig == want:
            return sp
    raise RuntimeError(f"type census {sig} matches no species "
                       f"(expected one of {SIGNATURES})")


def census_from_data_file(d):
    """mol id -> species via mol-id grouping AND bond graph (V1), species
    totals (V2 in the caller), per-molecule charge sums (V3)."""
    A = d["atoms"]
    order = np.argsort(A[:, 0], kind="stable")
    ids, mols, types, qs = (A[order][:, 0].astype(int),
                            A[order][:, 1].astype(int),
                            A[order][:, 2].astype(int), A[order][:, 3])
    assert np.array_equal(ids, np.arange(1, len(ids) + 1)), \
        "data file atom ids not 1..N"

    rows_of_mol = {}
    for row, m in enumerate(mols):
        rows_of_mol.setdefault(int(m), []).append(row)

    # V1: bond-graph components vs mol-id partition (both directions)
    comps = bond_graph_components(len(ids), d["bonds"])
    comp_of = np.zeros(len(ids) + 1, dtype=int)
    for k, comp in enumerate(comps):
        for a in comp:
            comp_of[a] = k + 1
    # The tube body is written WITHOUT bonds (the data file carries solvent
    # bonds only: 814*10 + 420*14 + 100*6 = 14,620 for fill_cnt10; 1132*10 +
    # 551*14 + 143*6 = 19,892 for fill_cnt16 -- exactly the header counts),
    # so its atoms are 800/1280 singleton components of one mol id.  V1's
    # single-component rule therefore applies to the MOBILE molecules; the
    # unbonded tube body is asserted separately (exactly one mol with > 50
    # atoms, all type 1 -- checked by the species census below and by the
    # caller's tube-atom count).
    big = [m for m, rows in rows_of_mol.items() if len(rows) > 50]
    assert big == [1], f"V1 scope: expected only the unbonded tube body, got {big}"
    for m, rows in rows_of_mol.items():
        if m in big:
            continue
        ks = {int(comp_of[i + 1]) for i in rows}
        assert len(ks) == 1, f"V1 FAIL: mol {m} split over bond components {ks}"
    for k, comp in enumerate(comps):
        ms = {int(mols[i - 1]) for i in comp}
        assert len(ms) == 1, f"V1 FAIL: component {k} spans mol ids {ms}"

    mol_species, mol_charge = {}, {}
    for m, rows in sorted(rows_of_mol.items()):
        sp = species_of_types(types[rows])
        mol_species[m] = sp
        mol_charge[m] = float(qs[rows].sum())
        if sp != "tube":
            assert abs(mol_charge[m] - EXPECTED_CHARGE[sp]) < 1e-6, \
                f"V3 FAIL: mol {m} ({sp}) charge {mol_charge[m]}"
    assert abs(float(qs.sum())) < 1e-6, f"V3 FAIL: cell charge {qs.sum()}"
    totals = {}
    for sp in mol_species.values():
        totals[sp] = totals.get(sp, 0) + 1
    tube_mols = [m for m, sp in mol_species.items() if sp == "tube"]
    assert len(tube_mols) == 1, f"expected exactly one tube body, got {tube_mols}"
    return dict(mol_species=mol_species, totals=totals,
                cell_charge=float(qs.sum()), n_atoms=len(ids),
                types_by_id=types, mols_by_id=mols)


# ---------------------------------------------------------------------------
# dump streaming
# ---------------------------------------------------------------------------

def iter_dump_frames(path):
    """Yield (timestep, box dict, cols list, ndarray(N, ncols))."""
    with open(path) as f:
        while True:
            line = f.readline()
            if not line:
                return
            assert line.startswith("ITEM: TIMESTEP"), f"bad dump: {line!r}"
            ts = int(f.readline())
            assert f.readline().startswith("ITEM: NUMBER OF ATOMS")
            n = int(f.readline())
            assert f.readline().startswith("ITEM: BOX BOUNDS")
            box = {ax: tuple(float(v) for v in f.readline().split()[:2])
                   for ax in "xyz"}
            cols = f.readline().split()[2:]
            rows = [f.readline() for _ in range(n)]
            arr = np.fromstring("".join(rows), sep=" ")
            assert arr.size == n * len(cols), f"frame {ts}: short atom block"
            yield ts, box, cols, arr.reshape(n, len(cols))


def analyse_fill(name):
    print(f"\n===== {name} =====")
    ddir = FILLS / name

    # ---- V0 provenance
    sha = check_sha256(name)
    print("V0 OK: sha256 of final.data / log.lammps / prod_thinned.dump "
          "== fill_series_manifest.json")

    # ---- builder record
    build = json.load(open(BUILD_MANIFEST))[name]
    z_span = build["tube_z_span_A"]
    Ltube = build["cnt_length_A"]
    n_tube = build["cnt_atoms"]
    assert abs(Ltube - PILOT_TUBE_A) < 1e-9, \
        f"{name}: tube length {Ltube} != pilot {PILOT_TUBE_A}"
    win = (z_span[0] + MARGIN_A, z_span[1] - MARGIN_A)
    win_len = win[1] - win[0]
    assert abs(win_len - (PILOT_TUBE_A - 2 * MARGIN_A)) < 1e-9
    # sensitivity window: the tube ATOMS' extent (top ring 1.23 A inside)
    win_s = (z_span[0] + MARGIN_A, z_span[0] + Ltube - 1.23 - MARGIN_A)
    print(f"declared tube z-span {z_span} A, length {Ltube} A; middle window "
          f"[{win[0]:.1f}, {win[1]:.1f}] A ({win_len:.1f} A); sensitivity "
          f"window (atom extent) [{win_s[0]:.1f}, {win_s[1]:.2f}] A")

    # ---- data file census + geometry
    d = read_data_file(ddir / "final.data")
    cen = census_from_data_file(d)
    print(f"data file: {cen['n_atoms']} atoms; census {cen['totals']}; "
          f"cell charge {cen['cell_charge']:+.1e}")
    man = {"EC": build["composition"]["n_EC"],
           "EMC": build["composition"]["n_EMC"],
           "Li+": build["composition"]["n_LiPF6"],
           "PF6-": build["composition"]["n_LiPF6"],
           "tube": 1}
    assert cen["totals"] == man, \
        f"V2 FAIL: census {cen['totals']} != build manifest {man}"
    print(f"V2 OK: census == stageA_build_manifest.json composition {man}")

    types_by_id = cen["types_by_id"]
    mols_by_id = cen["mols_by_id"]
    tube_rows = np.flatnonzero(types_by_id == TUBE_TYPE)
    assert len(tube_rows) == n_tube, \
        f"tube atoms {len(tube_rows)} != manifest {n_tube}"
    tube_xyz = d["atoms"][np.argsort(d["atoms"][:, 0], kind="stable")][tube_rows][:, 4:7]
    r_tube = np.hypot(tube_xyz[:, 0], tube_xyz[:, 1])
    R = float(r_tube.mean())
    assert np.ptp(r_tube) < 0.02, f"tube radius spread {np.ptp(r_tube)}"
    assert abs(2 * R - build["cnt_diameter_A"]) < 0.05, \
        f"measured diameter {2*R:.4f} vs manifest {build['cnt_diameter_A']}"
    assert abs(tube_xyz[:, 2].min() - z_span[0]) < 1e-3
    print(f"tube: R = {R:.4f} A measured (spread {np.ptp(r_tube):.1e}); "
          f"outermost atoms z [{tube_xyz[:, 2].min():.2f}, "
          f"{tube_xyz[:, 2].max():.2f}]")

    # molecule bookkeeping for the dump (rows id-sorted; V4 enforces it)
    perm = np.argsort(mols_by_id, kind="stable")
    mol_sorted = mols_by_id[perm]
    mol_ids = np.array(sorted(cen["mol_species"]))
    starts = np.searchsorted(mol_sorted, mol_ids)
    assert np.all(np.diff(np.append(starts, len(mol_sorted))) > 0)
    sp_arr = np.array([cen["mol_species"][m] for m in mol_ids])
    is_sp = {sp: sp_arr == sp for sp in SPECIES}
    mass_of_type = np.zeros(max(d["masses"]) + 1)
    for t, m in d["masses"].items():
        mass_of_type[t] = m
    w_by_mol = mass_of_type[types_by_id[perm]]

    # ---- V6 log cross-checks
    log = (ddir / "log.lammps").read_text(errors="replace")
    m = re.search(r"orthogonal box = \((\S+) (\S+) (\S+)\) to \((\S+) (\S+) (\S+)\)",
                  log)
    assert m, "V6 FAIL: no read_data box line in log"
    got = [round(float(v), 4) for v in m.groups()]
    b = build["box"]  # (xlo, xhi, ylo, yhi, zlo, zhi)
    want = [round(b[0], 4), round(b[2], 4), round(b[4], 4),
            round(b[1], 4), round(b[3], 4), round(b[5], 4)]
    assert got == want, f"V6 FAIL: log box {got} != manifest {want}"
    n_tube_log = int(re.search(r"(\d+) atoms in group tube", log).group(1))
    n_li_log = int(re.search(r"(\d+) atoms in group li", log).group(1))
    assert n_tube_log == n_tube, f"V6 FAIL: group tube {n_tube_log}"
    assert n_li_log == build["composition"]["n_LiPF6"], \
        f"V6 FAIL: group li {n_li_log} != manifest ion pairs"
    ts_final = int(re.search(r"timestep = (\d+)",
                             (ddir / "final.data").read_text(errors="replace")
                             .splitlines()[0]).group(1))
    dump_line = re.search(r"dump\s+pd all custom (\d+) prod.dump", log)
    assert dump_line, "V6 FAIL: no production dump line"
    assert int(dump_line.group(1)) == 1000, \
        f"V6 FAIL: production dump cadence {dump_line.group(1)}"
    print(f"V6 OK: log box == manifest box; group tube {n_tube_log}, group li "
          f"{n_li_log}; prod.dump every 1000 steps; final.data step {ts_final}")

    # ---- V7 + stream the thinned series
    n_frames = 1001
    series = {sp: np.zeros(n_frames, dtype=np.int64) for sp in SPECIES}
    series_win_atom = {sp: np.zeros(n_frames, dtype=np.int64) for sp in SPECIES}
    series_tube_all = {sp: np.zeros(n_frames, dtype=np.int64) for sp in SPECIES}
    tss = np.zeros(n_frames, dtype=np.int64)
    tube_max_dev = axis_max_dev = charge_max = 0.0
    i = -1
    for ts, box, cols, arr in iter_dump_frames(ddir / "prod_thinned.dump"):
        i += 1
        assert i < n_frames, "more than 1001 frames"
        ci = {c: k for k, c in enumerate(cols)}
        for c in ("id", "mol", "type", "q", "x", "y", "z", "ix", "iy", "iz"):
            assert c in ci, f"dump missing column {c}"
        ids = arr[:, ci["id"]].astype(int)
        assert np.array_equal(ids, np.arange(1, len(ids) + 1)), \
            f"V4 FAIL: frame {ts}: atom ids not 1..N sorted"
        typ = arr[:, ci["type"]].astype(int)
        mols = arr[:, ci["mol"]].astype(int)
        assert np.array_equal(typ, types_by_id), \
            f"V4 FAIL: frame {ts}: per-atom types differ from the data file"
        assert np.array_equal(mols, mols_by_id), \
            f"V4 FAIL: frame {ts}: per-atom mol ids differ from the data file"
        q = arr[:, ci["q"]]
        charge_max = max(charge_max, abs(float(q.sum())))

        Lx = box["x"][1] - box["x"][0]
        Ly = box["y"][1] - box["y"][0]
        Lz = box["z"][1] - box["z"][0]
        cx = 0.5 * (box["x"][0] + box["x"][1])
        cy = 0.5 * (box["y"][0] + box["y"][1])

        dev = float(np.abs(np.stack([arr[tube_rows, ci["x"]] - tube_xyz[:, 0],
                                     arr[tube_rows, ci["y"]] - tube_xyz[:, 1],
                                     arr[tube_rows, ci["z"]] - tube_xyz[:, 2]])
                           ).max())
        tube_max_dev = max(tube_max_dev, dev)
        axis_max_dev = max(axis_max_dev,
                           float(np.hypot(arr[tube_rows, ci["x"]].mean() - cx,
                                          arr[tube_rows, ci["y"]].mean() - cy)))

        xu = arr[:, ci["x"]] + arr[:, ci["ix"]] * Lx
        yu = arr[:, ci["y"]] + arr[:, ci["iy"]] * Ly
        zu = arr[:, ci["z"]] + arr[:, ci["iz"]] * Lz
        wsum = np.add.reduceat(w_by_mol, starts)
        comx = np.add.reduceat(w_by_mol * xu[perm], starts) / wsum
        comy = np.add.reduceat(w_by_mol * yu[perm], starts) / wsum
        comz = np.add.reduceat(w_by_mol * zu[perm], starts) / wsum
        comx -= Lx * np.round((comx - cx) / Lx)   # image nearest box centre
        comy -= Ly * np.round((comy - cy) / Ly)
        rcom = np.hypot(comx, comy)
        in_r = rcom < R
        in_win = in_r & (comz >= win[0]) & (comz <= win[1])
        in_win_s = in_r & (comz >= win_s[0]) & (comz <= win_s[1])
        in_tube = in_r & (comz >= z_span[0]) & (comz <= z_span[1])
        tss[i] = ts
        for sp in SPECIES:
            m_sp = is_sp[sp]
            series[sp][i] = np.count_nonzero(in_win & m_sp)
            series_win_atom[sp][i] = np.count_nonzero(in_win_s & m_sp)
            series_tube_all[sp][i] = np.count_nonzero(in_tube & m_sp)
    assert i + 1 == n_frames, f"{i + 1} frames, expected {n_frames}"
    assert np.array_equal(tss, np.arange(0, 5_000_001, 5000)), \
        "V7 FAIL: timesteps not 0..5,000,000 at 5000-step spacing"
    assert ts_final == int(tss[-1]), "V6 FAIL: final.data step != last dump step"
    assert tube_max_dev < 1e-4, f"V5 FAIL: tube moved by {tube_max_dev} A"
    assert axis_max_dev < 0.05, f"tube axis off box centre by {axis_max_dev} A"
    assert charge_max < 1e-6, f"V4 FAIL: frame charge {charge_max}"
    print(f"V4 OK: every frame's per-atom types and mol ids identical to the "
          f"data file ({n_frames} frames); max |cell charge| {charge_max:.1e}")
    print(f"V5 OK: tube frozen (max deviation {tube_max_dev:.1e} A), axis on "
          f"box centre (max {axis_max_dev:.1e} A)")
    print(f"V7 OK: timesteps 0..{tss[-1]} at exactly 5000-step (5 ps) spacing, "
          f"{n_frames} frames (1/5 of the 1 ps production cadence)")
    t_ps = tss / 1000.0  # dump every 1000 steps = 1 ps => dt = 1 fs

    # ---- blocks: 2 blocks of the production half (Q2 brief)
    half = t_ps >= 2500.0
    blkA = half & (t_ps < 3750.0)
    blkB = half & (t_ps >= 3750.0)
    stats = {}
    for sp in SPECIES:
        c = series[sp].astype(float)
        a, bb = c[blkA], c[blkB]
        mean_half = float(c[half].mean())
        sem_blocks = float(np.std([a.mean(), bb.mean()], ddof=1) / np.sqrt(2.0))
        sem_frames = float(c[half].std(ddof=1) / np.sqrt(int(half.sum())))
        slope = float(np.polyfit(t_ps[half], c[half], 1)[0]) * 1000.0
        # trailing 250 ps (50-frame) rolling mean; plateau = first t within
        # 10% of the half mean
        roll = np.convolve(series[sp].astype(float), np.ones(50) / 50.0,
                           mode="valid")
        roll_t = t_ps[49:]
        above = np.flatnonzero(roll >= 0.9 * mean_half)
        plateau = float(roll_t[above[0]]) if above.size else None
        stats[sp] = dict(
            frame0=int(series[sp][0]),
            block_A_mean=float(a.mean()), block_A_n=int(blkA.sum()),
            block_B_mean=float(bb.mean()), block_B_n=int(blkB.sum()),
            half_mean=mean_half, half_frames=int(half.sum()),
            between_block_sem=sem_blocks, per_frame_sem=sem_frames,
            slope_per_ns=slope, plateau_ps=plateau,
            per_A=mean_half / win_len,
            per_A_sem_blocks=sem_blocks / win_len,
            # sensitivity: atom-extent window
            win_atom_mean=float(series_win_atom[sp][half].mean()),
            win_atom_per_A=float(series_win_atom[sp][half].mean()) /
            (win_s[1] - win_s[0]),
            # whole declared tube (mouths included), for the dynamics note
            tube_all_mean=float(series_tube_all[sp][half].mean()),
            tube_all_frame0=int(series_tube_all[sp][0]),
        )
    pilot = P.COMPOSITION[10 if name.endswith("10") else 16]
    comp, gate_fail = {}, []
    for sp, key in (("EC", "n_EC"), ("EMC", "n_EMC"),
                    ("Li+", "n_LiPF6"), ("PF6-", "n_LiPF6")):
        pilot_per_A = pilot[key] / PILOT_TUBE_A
        fill_per_A = stats[sp]["per_A"]
        pct = 100.0 * (fill_per_A - pilot_per_A) / pilot_per_A
        pct_s = 100.0 * (stats[sp]["win_atom_per_A"] - pilot_per_A) / pilot_per_A
        comp[sp] = dict(pilot_count_per_49p2A=pilot[key],
                        pilot_per_A=pilot_per_A,
                        fill_per_A=fill_per_A, pct_diff=pct,
                        pct_diff_atom_extent_window=pct_s)
        if abs(pct) > 5.0:
            gate_fail.append(sp)
    q3 = bool(gate_fail) and name.endswith("16")
    print(f"blocks (production half t in [2500, 5000] ps): A [2500, 3750) ps "
          f"({stats['EC']['block_A_n']} frames), B [3750, 5000] ps "
          f"({stats['EC']['block_B_n']} frames)")
    for sp in SPECIES:
        s, c = stats[sp], comp[sp]
        print(f"  {sp:5s}: half mean {s['half_mean']:6.2f} "
              f"(A {s['block_A_mean']:6.2f} / B {s['block_B_mean']:6.2f}) "
              f"sem(blocks) {s['between_block_sem']:.3f} "
              f"sem(frames) {s['per_frame_sem']:.3f} slope "
              f"{s['slope_per_ns']:+.3f}/ns | {s['per_A']:.4f}/A vs pilot "
              f"{c['pilot_per_A']:.4f}/A -> {c['pct_diff']:+.1f}% "
              f"(atom-extent window {c['pct_diff_atom_extent_window']:+.1f}%)")
    print(f"D49 gate ({name}): species beyond 5% of the pilot: "
          f"{gate_fail if gate_fail else 'none'} -> "
          f"{'Q3 TRIGGERS' if q3 else 'no Q3 trigger from this tube'}")

    return dict(
        name=name,
        provenance=dict(
            sha256=sha,
            build=dict(cnt_chirality=build["cnt_chirality"],
                       cnt_length_A=build["cnt_length_A"],
                       cnt_diameter_A=build["cnt_diameter_A"],
                       tube_z_span_A=build["tube_z_span_A"],
                       box=build["box"],
                       reservoir_each_side_A=build["reservoir_each_side_A"],
                       composition=build["composition"]),
            pilot_composition={k: pilot[k] for k in ("n_EC", "n_EMC", "n_LiPF6")},
            pilot_source="params.py COMPOSITION (the pilot builds' "
                         "composition per 49.2 A tube)",
        ),
        geometry=dict(window=[win[0], win[1]], window_length_A=win_len,
                      window_atom_extent=[win_s[0], win_s[1]],
                      R_measured_A=R, margin_A=MARGIN_A,
                      tube_atom_z_range=[float(tube_xyz[:, 2].min()),
                                         float(tube_xyz[:, 2].max())]),
        census_totals=cen["totals"],
        validation=dict(
            v0_sha256=True, v1_bond_graph_matches_mol_partition_mobile=True,
            v2_census_equals_build_manifest=True,
            v3_molecule_charges=True, cell_charge=cen["cell_charge"],
            v4_per_atom_types_and_mol_ids_stable_all_frames=True,
            max_abs_cell_charge=charge_max,
            v5_tube_max_deviation_A=tube_max_dev,
            v5_axis_max_deviation_A=axis_max_dev,
            v6_log_box_groups_dump_cadence_final_step=True,
            v7_thinned_series=f"{n_frames} frames, 0..{int(tss[-1])} step 5000",
        ),
        timesteps_ps=[float(v) for v in t_ps],
        series={sp: [int(v) for v in series[sp]] for sp in SPECIES},
        series_atom_extent_window={sp: [int(v) for v in series_win_atom[sp]]
                                   for sp in SPECIES},
        series_whole_tube={sp: [int(v) for v in series_tube_all[sp]]
                           for sp in SPECIES},
        block_convention=dict(
            definition="production = the 5 ns NPT after the 650 ps staged "
                       "decompression (log: reset_timestep 0, run 5000000, "
                       "dt 1 fs); half = t in [2500, 5000] ps; block A = "
                       "[2500, 3750) ps, block B = [3750, 5000] ps; central "
                       "value = per-frame mean of the half; error = "
                       "between-block SEM = std([A,B], ddof=1)/sqrt(2)",
            caveat="Y8 wants blocks of >= 5 tau_M; stage_a.md measures "
                   "tau_M ~ 2.7-3.0 ns at the working field (zero-field "
                   "collective autocorr 0.34-0.95 ns), so a 1.25 ns block "
                   "is ~0.4-0.5 tau_M: the between-block SEM is a LOWER "
                   "bound, and with n_blocks=2 it equals |A-B|/2 exactly",
        ),
        stats=stats, comparison=comp,
        d49_gate=dict(species_over_5pct=gate_fail, q3_triggers=q3),
    )


def main():
    results = {name: analyse_fill(name)
               for name in ("fill_cnt10", "fill_cnt16")}
    with open(RESULTS, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nwrote {RESULTS}")
    print("\n===== D49 gate summary =====")
    for name, r in results.items():
        g = r["d49_gate"]
        print(f"{name}: species over 5%: {g['species_over_5pct'] or 'none'}; "
              f"Q3 {'TRIGGERS' if g['q3_triggers'] else 'not triggered'}")


if __name__ == "__main__":
    main()
