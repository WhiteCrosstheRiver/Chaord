# -*- coding: utf-8 -*-
"""
Consolidate provenance JSONs for each system from params.py, the build
manifest, the LAMMPS log/screen files and the sanity/equil analyses.

STANDALONE (no chaord import). Run AFTER the runs and analyses:
    python sanity_checks.py ... ; python analyze_equil.py ...
    python make_provenance.py

Writes runs/<system>/provenance_<system>.json and prints a summary.
"""
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone

import params as P

HERE = os.path.dirname(os.path.abspath(__file__))

KB_KCAL = 0.0019872041

CITATIONS = [
    {"key": "Nutakki2023",
     "ref": "J. S. C. Nutakki, 'Molecular Modelling of Lithium Salts in the "
            "Binary Mixtures of Ethylene and Ethyl Methyl Carbonates', M.S. "
            "thesis, University of Illinois Chicago (2023); Tables II-III.",
     "url": "https://indigo.uic.edu (file 44990929)",
     "local_file": "_lit/nutakki2023.pdf",
     "used_for": "All LiPF6 (Li, P, F) and EC/EMC (CO, OS, CT, OC, HC, HM) "
                 "LJ parameters and charges; equilibrium internal "
                 "coordinates (verified against the PDF tables on "
                 "2026-10-14)"},
    {"key": "Jorgensen96",
     "ref": "W. L. Jorgensen, D. S. Maxwell, J. Tirado-Rives, J. Am. Chem. "
            "Soc. 118, 11225 (1996).",
     "local_file": None,
     "used_for": "OPLS-AA framework; terminal alkyl C (opls_135) LJ and "
                 "uncharged alkyl sites for the EMC methyl tail"},
    {"key": "Joung08",
     "ref": "I. S. Joung, T. E. Cheatham III, J. Phys. Chem. B 112, 9020 "
            "(2008); Table 5.",
     "local_file": "_lit/jc2008.xml",
     "used_for": "ALTERNATIVE ONLY, NOT USED in the runs: Li+/SPC/E "
                 "(Rmin/2 = 0.791 A, eps = 0.3367344 kcal/mol), "
                 "re-verified from the local XML on 2026-10-14"},
    {"key": "Werder03",
     "ref": "T. Werder, J. H. Walther, R. L. Jaffe, T. Halicioglu, "
            "P. Koumoutsakos, J. Phys. Chem. B 107, 1345 (2003).",
     "local_file": None,
     "used_for": "CNT sp2-carbon LJ (sigma = 3.40 A, eps = 0.0853 kcal/mol)"},
    {"key": "Saito98",
     "ref": "R. Saito, G. Dresselhaus, M. S. Dresselhaus, 'Physical "
            "Properties of Carbon Nanotubes', Imperial College Press (1998).",
     "local_file": None,
     "used_for": "graphene lattice constant a = 2.46 A; armchair (n,n) tube "
                 "geometry (d = a*n*sqrt(3)/pi; unit cell a with 4n atoms)"},
    {"key": "Xie2023",
     "ref": "Y. Xie et al., 'A computational study of the effect of external "
            "heat flux and electric field on the water flow inside CNTs' "
            "(2023), doi:10.1016/j.c2tec.2023.100157.",
     "local_file": "_lit/esi_field2023.pdf (ESI only)",
     "used_for": "context for the 0.2 V/A field strength (0.1 V/A used "
                 "there; 0.1-0.5 V/A is the common MD range for field-driven "
                 "transport in nanopores)"},
]

DEVIATIONS = [
    "Molecules rigid (fixed literature geometry; LJ + point charges); no "
    "intra-molecular bonded dynamics - the documented first-result "
    "simplification allowed by the task.",
    "EMC charges: the thesis EMC column does not sum to zero (+0.68 e with "
    "its own H typing); charges transplanted from the chemically equivalent "
    "EC sites + OPLS-AA uncharged alkyl tail, terminal methyl C absorbs the "
    "residual (exact neutrality). Verified against the local PDF on "
    "2026-10-14.",
    "Combining rule Lorentz-Berthelot (LAMMPS default; arithmetic sigma) - "
    "recorded deviation from strict OPLS-AA geometric sigma.",
    "EC ring angle closure: thesis angles at the carbonate carbon "
    "(2x123.4 + 118.18 = 364.98 deg) cannot coexist; OC-CO-OS = 123.4 "
    "enforced on both sides, fixing OS-CO-OS = 113.2 deg.",
    "EMC template REBUILT 2026-10-14: the original z-matrix builder placed "
    "the ethoxy OS 0.68 A from the carbonyl OC (enforced OC-CO-OS = "
    "143.7/30.6 instead of 123.4 deg). The rebuild is an explicit planar "
    "construction asserting every bond/angle/dihedral of Table III "
    "(dihedrals: OC-CO-OS-CT = 0 deg ester s-cis; CO-OS-CT-CT = 180 deg "
    "anti ethyl, both documented conformer choices).",
    "EMC H-bond wiring fixed (original bonded CT_me to the terminal-methyl "
    "H's; harmless because bonds are zero-force and molecule-intra "
    "neighbour exclusions supersede 1-2/1-3 terms, but wrong).",
    "(16,16) composition trimmed 37/24 -> 36/23 (EC/EMC) after the EMC "
    "geometry fix: with the correct extended EMC, 37/24 exhausted 1M "
    "insertion attempts per molecule on every seed tried (20261003-38 "
    "range). 36/23 keeps EC:EMC 1:1 by volume (2401 vs 2394 volume units), "
    "the same 6 LiPF6 (same molarity), and a vdW-pore density closer to "
    "the bulk mixture.",
    "cnt16 build seeds re-chosen by scan (see per-system seeds); the "
    "original seeds 20261003/20261004 were chosen against the broken EMC "
    "template.",
    "Li+ integration: fix rigid/nvt/small (and fix rigid/nvt) reject "
    "single-particle bodies, so Li+ atoms carry their own Nose-Hoover "
    "fix nvt (same Tstart/Tstop/Tdamp); multi-atom molecules use "
    "fix rigid/nvt/small molecule. Found on first execution; the "
    "previous agent's input plan had never been run.",
    "Velocities created at 10 K (the plan's 300 K create + 10 K ramp "
    "start was an internal inconsistency - quench-then-reheat).",
    "Equilibration dump every 25 ps and thermo 500 added (equilibration "
    "evidence); thermo displays v_Tmob = 2KE/(DOF*kB) of the mobile "
    "subsystem because the default temperature is diluted by the frozen "
    "zero-velocity tube atoms.",
    "Real-space intra-molecular nonbonded excluded (zero-K bonds + "
    "special_bonds 0 0 0 + neigh_modify exclude molecule/intra); the "
    "Ewald reciprocal intra-body term remains as a constant energy "
    "offset (zero net force/torque on rigid bodies).",
    "Joung-Cheatham Li+ recorded as alternative only - the run uses the "
    "thesis Li+ so the whole electrolyte comes from one parameterization.",
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def engine_version(screen_log):
    with open(screen_log) as f:
        first = f.readline().strip()
    return first  # e.g. "LAMMPS (22 Jul 2025 - Update 4)"


def total_wall(screen_log):
    txt = open(screen_log).read()
    m = re.search(r"Total wall time: (\d+):(\d+):(\d+)", txt)
    if m:
        return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
    return None


def build(name):
    mdir = os.path.join(HERE, "runs", name)
    manifest = json.load(open(os.path.join(HERE, "data",
                                           "build_manifest.json")))[name]
    equil = json.load(open(os.path.join(mdir, f"equil_{name}.json")))
    sanity = json.load(open(os.path.join(mdir, f"sanity_{name}.json")))
    spec = P.SYSTEMS[name]
    n_cnt = spec["n"]
    comp = P.COMPOSITION[n_cnt]

    data_rel = f"data/{name}.data"
    prov = {
        "system": name,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "engine": {
            "name": "LAMMPS",
            "version": engine_version(os.path.join(mdir, "screen.log")),
            "executable": "C:/ProgramData/anaconda3/Scripts/lmp.exe "
                          "(serial, 1 MPI rank, Windows)",
        },
        "potential": {
            "name": "OPLS-AA-based rigid LiPF6 / EC:EMC model + Werder sp2-C "
                    "for the CNT; LJ + point charges; Lorentz-Berthelot "
                    "mixing (recorded deviation from OPLS geometric sigma)",
            "cutoff_A": P.CUTOFF_A,
            "long_range": f"PPPM rel-acc {P.PPPM_REL_ACC:g}",
            "citations": CITATIONS,
        },
        "system_definition": {
            "cnt_chirality": [n_cnt, n_cnt],
            "cnt_diameter_A": manifest["cnt_diameter_A"],
            "cnt_length_A": manifest["cnt_length_A"],
            "cnt_atoms": manifest["cnt_atoms"],
            "box_A": manifest["box"],
            "composition": {"n_EC": comp["n_EC"], "n_EMC": comp["n_EMC"],
                            "n_LiPF6": comp["n_LiPF6"]},
            "n_atoms_total": manifest["n_atoms_total"],
            "n_bonds_topology_only": manifest["n_bonds"],
            "pore_volume_geom_nm3": manifest["V_pore_geom_nm3"],
            "pore_volume_vdw_nm3": manifest["V_pore_vdw_nm3"],
            "molarity_M_vdw_volume": manifest["molarity_M_vdw_volume"],
            "external_field_V_per_A": (P.FIELD_ON_V_PER_A
                                       if spec["field"] == "on" else 0.0),
        },
        "protocol": {
            "ensemble": "NVT (tube atoms frozen)",
            "temperature_K": P.TEMPERATURE_K,
            "timestep_fs": P.TIMESTEP_FS,
            "thermostat": "Nose-Hoover, damping %.0f fs: fix "
                          "rigid/nvt/small molecule on multi-atom bodies, "
                          "fix nvt on Li+ atoms (identical targets/damping)"
                          % P.THERMO_DAMP_FS,
            "stage1_ramp_ps": P.RAMP_PS,
            "stage2_equil_ps": P.EQUIL_PS,
            "stage3_production_ps": P.PROD_PS,
            "field_present_during_all_stages": spec["field"] == "on",
        },
        "seeds": {
            "build": P.SEEDS[name]["build"],
            "velocity": P.SEEDS[name]["velocity"],
            "note": "original planned seed set (20261001-04). During the "
                    "resume, intermediate builds scanned other seeds after "
                    "the EMC geometry fix; the pack-then-relax packer "
                    "restored the original seeds. Data files were built "
                    "with exactly these seeds.",
        },
        "outputs": {
            "data_file": data_rel,
            "data_sha256": sha256(os.path.join(HERE, data_rel)),
            "production_dump": f"runs/{name}/prod.dump",
            "prod_sha256": sha256(os.path.join(mdir, "prod.dump")),
            "dump_columns": "id mol type q x y z (unscaled) + box bounds",
            "n_frames": sanity["n_prod_frames"],
            "frame_times_ps": sanity["frame_times_ps"],
            "frame_spacing_ps": sanity["frame_spacing_ps"],
            "final_config": f"runs/{name}/final.data",
        },
        "wall_time_s": equil["total_wall_s"],
        "lammps_total_wall_hms": None,  # filled from screen.log below
        "equilibration_evidence": {
            "Tmob_mean_K": equil.get("equil", {}).get("Tmob_mean_K"),
            "Tmob_std_K": equil.get("equil", {}).get("Tmob_std_K"),
            "Tmob_mean_last_half_K":
                equil.get("equil", {}).get("Tmob_mean_last_half_K"),
            "PE_quarter_means_kcal_mol":
                equil.get("equil", {}).get("PE_quarter_means_kcal_mol"),
            "PE_q4_minus_q1_kcal_mol":
                equil.get("equil", {}).get("PE_q4_minus_q1_kcal_mol"),
            "PE_q4_minus_q1_per_molecule_kT":
                equil.get("equil", {}).get("PE_q4_minus_q1_per_molecule_kT"),
            "PE_slope_last_half_kcal_per_ps":
                equil.get("equil", {}).get("PE_slope_last_half_kcal_per_ps"),
            "density_vs_t_equil": sanity["equil_density_vs_t"],
            "prod_PE_mean_kcal_mol": equil.get("prod", {}).get("PE_mean_kcal_mol"),
        },
        "sanity": {
            "n_atoms_all_frames": [f["n_atoms"] for f in sanity["frames"]],
            "composition_stable": sanity["composition_stable"],
            "composition_observed": sanity["composition_observed"],
            "min_intermol_heavy_A": sanity["prod_min_intermol_heavy_A"],
            "bondi_floor": {
                "rule": "intermolecular heavy-atom pairs >= 0.75 x Bondi "
                        "vdW sum (ase set: C 1.70, O 1.52, Li 1.82, "
                        "P 1.80, F 1.47 A); cation pairs exempt "
                        "(ion solvation); H-involving pairs outside rule",
                "worst": sanity["prod_bondi_worst"],
                "pass": sanity["prod_bondi_pass"],
            },
            "lj08_floor": {"worst": sanity["prod_lj08_worst"],
                           "pass": sanity["prod_lj08_pass"]},
            "min_intermol_any_A": sanity["prod_min_intermol_any_A"],
            "min_intermol_H_involving_A":
                sanity["prod_min_intermol_H_involving_A"],
            "min_intermol_heavy_cationpair_A":
                sanity["prod_min_intermol_heavy_cationpair_A"],
            "density_in_tube_geom_g_cm3":
                sanity["prod_density_in_tube_geom_g_cm3"][0],
            "density_in_tube_vdw_g_cm3":
                sanity["prod_density_in_tube_vdw_g_cm3"],
            "bulk_reference_g_cm3": 1.17,
            "n_mobile_outside_tube_all_frames":
                sanity["prod_n_mobile_outside_tube"],
            "max_mobile_radius_A": sanity["prod_max_mobile_radius_A"],
            "cnt_radius_A": manifest["cnt_diameter_A"] / 2.0,
        },
        "type_map": {str(t): {"label": P.ATOM_TYPES[t][1],
                              "element": P.ATOM_TYPES[t][0],
                              "sigma_A": P.ATOM_TYPES[t][2],
                              "eps_kcal_mol": round(P.ATOM_TYPES[t][3], 6),
                              "source": P.ATOM_TYPES[t][5]}
                     for t in sorted(P.ATOM_TYPES)},
        "masses_amu": {str(t): P.AMU[P.ATOM_TYPES[t][0]]
                       for t in sorted(P.ATOM_TYPES)},
        "deviations_and_notes": DEVIATIONS,
        "analysis_files": {
            "sanity": f"runs/{name}/sanity_{name}.json",
            "equil": f"runs/{name}/equil_{name}.json",
            "screen_log": f"runs/{name}/screen.log",
            "lammps_log": f"runs/{name}/log.lammps",
        },
    }
    tw = total_wall(os.path.join(mdir, "screen.log"))
    prov["lammps_total_wall_hms"] = tw
    outp = os.path.join(mdir, f"provenance_{name}.json")
    with open(outp, "w") as f:
        json.dump(prov, f, indent=2)
    print(f"[prov] {name}: wall {tw} s; N {manifest['n_atoms_total']}; "
          f"{comp}; field "
          f"{prov['system_definition']['external_field_V_per_A']} V/A; "
          f"sanity bondi pass {prov['sanity']['bondi_floor']['pass']}")
    return prov


def main():
    for name in (sys.argv[1:] or list(P.SYSTEMS)):
        build(name)


if __name__ == "__main__":
    main()
