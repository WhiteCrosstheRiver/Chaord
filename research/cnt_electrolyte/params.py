# -*- coding: utf-8 -*-
"""
All physical parameters for the CNT / LiPF6-in-carbonate MD generation.

STANDALONE BY DESIGN: this module must NOT import anything from `chaord`
(house rule, AGENTS.md: reference-data generators are independent evidence).

Units: LAMMPS "real" units -> Angstrom, kcal/mol, elementary charge, g/cm^3.

PARAMETER PROVENANCE (primary sources, see research/cnt_electrolyte/_lit/):
  [Nutakki2023]  J. S. C. Nutakki, "Molecular Modelling of Lithium Salts in the
                 Binary Mixtures of Ethylene and Ethyl Methyl Carbonates",
                 M.S. thesis, University of Illinois Chicago (2023),
                 Table II (LiPF6) and Table III (EC/EMC).
                 Charges: CHELPG fit to the MP2/aug-cc-pVTZ electrostatic
                 potential (Gaussian 16); LJ from OPLS-AA atom types.
                 Local copy: _lit/nutakki2023.pdf (downloaded from
                 https://indigo.uic.edu, file 44990929).
  [Jorgensen96]  W. L. Jorgensen, D. S. Maxwell, J. Tirado-Rives,
                 J. Am. Chem. Soc. 118, 11225 (1996). OPLS-AA (LJ types,
                 alkyl sites carry zero charge in OPLS-AA alkanes).
  [Joung08]      I. S. Joung, T. E. Cheatham III, J. Phys. Chem. B 112,
                 9020 (2008). Li+/SPC/E values (Table 5, primary table
                 re-extracted locally: _lit/jc2008.xml):
                 Rmin/2 = 0.791 A, eps = 0.3367344 kcal/mol
                 -> sigma = 2*0.791/2^(1/6) = 1.40928 A.
                 RECORDED AS ALTERNATIVE ONLY; the run uses [Nutakki2023] Li+
                 so that the whole electrolyte comes from one consistent
                 LiPF6/EC/EMC parameterization.
  [Werder03]     T. Werder, J. H. Walther, R. L. Jaffe, T. Halicioglu,
                 P. Koumoutsakos, J. Phys. Chem. B 107, 1345 (2003).
                 sp2-carbon (graphite/CNT) LJ: sigma = 3.40 A,
                 eps = 0.0853 kcal/mol (0.357 kJ/mol).
  [Saito98]      R. Saito, G. Dresselhaus, M. S. Dresselhaus, "Physical
                 Properties of Carbon Nanotubes", Imperial College Press
                 (1998). Graphene lattice constant a = 2.46 A; armchair (n,n)
                 diameter d = (a/pi) * n * sqrt(3); unit cell length along the
                 tube axis T = a containing 4n atoms.
  [Xie2023]      Y. Xie et al., "A computational study of the effect of
                 external heat flux and electric field on the water flow
                 inside CNTs" (2023), used 0.1 V/A on a CNT in MD.
                 doi:10.1016/j.c2tec.2023.100157-ish (ScienceDirect
                 S2238785423012498). MD of field-driven transport in
                 nanopores commonly uses 0.1-0.5 V/A; we use 0.2 V/A
                 (see FIELD below for the reasoning).

DEVIATIONS / SIMPLIFICATIONS (honest record):
  * Molecules are RIGID (fixed literature geometry, LJ + point charges).
    This is the documented first-result simplification explicitly allowed
    by the task: no intra-molecular bonded dynamics.
  * EMC charges: the EMC column printed in [Nutakki2023] Table III does not
    sum to zero (about +0.7 e to +1.2 e depending on H typing) and cannot be
    used as printed. We therefore assign EMC charges by transplanting the
    chemically equivalent EC sites (OC/CO/OS/CT/HC from the EC column) and
    standard OPLS-AA uncharged alkyl sites for the terminal methyl; the
    residual (+0.02 e) is absorbed by the terminal methyl carbon to enforce
    exact neutrality. Recorded in provenance.
  * Combining rules: Lorentz-Berthelot (arithmetic sigma, geometric epsilon),
    LAMMPS default, consistent with [Werder03] parameterization style.
    Strict OPLS-AA would use geometric sigma; this is a deliberate,
    recorded deviation (cross sigmas shift by a few percent).
  * Intra-molecular 1-4+ nonbonded terms are constants of the motion for
    rigid bodies (internal forces cancel exactly in rigid-body integration);
    real-space 1-2/1-3 pairs are excluded via zero-force bonds +
    special_bonds 0 0 0, and molecule-intra real-space pairs are additionally
    excluded via neigh_modify. The Ewald reciprocal-space intra-body term
    remains as a constant energy offset (does not affect dynamics).
"""

import math

# ---------------------------------------------------------------- constants
AMU = {  # atomic masses (IUPAC 2021, standard atomic weights), amu
    "C": 12.011, "H": 1.008, "O": 15.999, "Li": 6.94, "P": 30.974, "F": 18.998,
}

KJ_TO_KCAL = 1.0 / 4.184

# ------------------------------------------------------------- LJ by type
# type id -> (element, label, sigma [A], eps [kcal/mol], charge, source)
# Charges here are the DEFAULT per-type charge; per-atom charges are written
# explicitly in the data file (they differ only where noted).
#
# sigma/eps sources:
#   CO/OS/OC/CT  : [Nutakki2023] Table III, converted kJ/mol -> kcal/mol
#   HC (opls_777): [Nutakki2023] Table III
#   HM (opls_140): [Nutakki2023] Table III
#   C_term       : OPLS-AA alkyl carbon (opls_135) sigma/eps, q=0 [Jorgensen96]
#   C_tube       : [Werder03]
#   Li/P/F       : [Nutakki2023] Table II, converted kJ/mol -> kcal/mol
ATOM_TYPES = {
    1:  ("C", "C_tube", 3.40, 0.0853, 0.00, "Werder03"),
    2:  ("C", "CO",     3.75, 0.4393 * KJ_TO_KCAL, None, "Nutakki2023 T3"),  # carbonyl C
    3:  ("O", "OS",     3.00, 0.7113 * KJ_TO_KCAL, None, "Nutakki2023 T3"),  # carbonate ester O
    4:  ("C", "CT",     3.50, 0.2761 * KJ_TO_KCAL, None, "Nutakki2023 T3"),  # O-alkyl sp3 C
    5:  ("O", "OC",     2.96, 0.8786 * KJ_TO_KCAL, None, "Nutakki2023 T3"),  # carbonyl O
    6:  ("H", "HC",     2.42, 0.0628 * KJ_TO_KCAL, None, "Nutakki2023 T3"),  # H on O-adjacent C
    7:  ("H", "HM",     2.50, 0.1255 * KJ_TO_KCAL, 0.00, "Nutakki2023 T3 / Jorgensen96"),  # alkyl H
    8:  ("C", "C_term", 3.50, 0.066,              None, "Jorgensen96 opls_135"),  # terminal CH3 C (EMC)
    9:  ("Li", "Li",    1.51, 0.6906 * KJ_TO_KCAL, 1.00, "Nutakki2023 T2"),
    10: ("P", "P",      3.74, 0.8368 * KJ_TO_KCAL, 1.70, "Nutakki2023 T2"),
    11: ("F", "F",      3.118, 0.2552 * KJ_TO_KCAL, -0.45, "Nutakki2023 T2"),
}

# ------------------------------------------------ molecule charge models
# EC (ethylene carbonate, C3H4O3, 10 atoms) -- [Nutakki2023] Table III EC
# column, verbatim. Sum = 0 exactly.
EC_CHARGES = {"CO": 0.99, "OS": -0.44, "OC": -0.63, "CT": 0.22, "HC": 0.02}

# EMC (ethyl methyl carbonate, C4H8O3, 15 atoms) -- EC sites transplanted,
# OPLS-AA uncharged alkyl tail, terminal C absorbs the +0.02 e residual.
EMC_CHARGES = {
    "OC": -0.63, "CO": 0.99,
    "OS_me": -0.44, "CT_me": 0.22, "HC_me": 0.02,       # methoxy side
    "OS_et": -0.44, "CT_et": 0.22, "HC_et": 0.02,       # ethoxy CH2
    "C_term": -0.02, "HM": 0.00,                        # terminal methyl
}

# PF6- -- [Nutakki2023] Table II: q_P = +1.70, q_F = -0.45 (x6) -> -1.00
PF6_CHARGES = {"P": 1.70, "F": -0.45}
LI_CHARGE = 1.00

# ------------------------------------------------ molecule geometries
# Equilibrium internal coordinates from [Nutakki2023] Tables II-III (A / deg).
GEOM = {
    "bonds": {
        ("CO", "OC"): 1.229,   # C=O
        ("CO", "OS"): 1.327,   # C-O (ester)
        ("OS", "CT"): 1.410,   # O-CH2 / O-CH3
        ("CT", "HC"): 1.090,   # C-H
        ("CT", "CT"): 1.529,   # C-C (ethyl)
        ("CT", "HM"): 1.090,
        ("P", "F"): 1.606,     # P-F
    },
    "angles": {
        ("OC", "CO", "OS"): 123.4,
        ("CO", "OS", "CT"): 116.9,
        ("OS", "CO", "OS"): 118.18,
        ("OS", "CT", "HC"): 109.5,
        ("OS", "CT", "CT"): 109.5,
        ("HC", "CT", "HC"): 107.8,
        ("OS", "CT", "CT"): 109.5,
        ("CO", "OS", "CT"): 116.9,
    },
    "dihedrals": {
        "EMC_OC_CO_OS_CT": 0.0,     # planar carbonate (s-cis ester)
        "EMC_CO_OS_CT_CT": 180.0,   # anti ethyl conformer (lowest energy)
    },
}

# --------------------------------------------------------------- CNT
GRAPHENE_A = 2.46  # A [Saito98]
# armchair (n,n): d = a*n*sqrt(3)/pi ; unit cell length along axis = a ;
# 4n atoms per unit cell.

# --------------------------------------------------------------- field
# Field strength for the "on" state, V/Angstrom (LAMMPS real units).
# 0.2 V/A: within the 0.1-0.5 V/A range used in MD of field-driven
# transport in CNTs/nanopores ([Xie2023] used 0.1 V/A). Chosen at 2x the
# low end so that PF6- (the slower ion) shows a resolvable drift inside a
# viscous confined carbonate electrolyte within the 100 ps production
# window, while staying far below the 0.5 V/A upper end where carbonate
# chemistry would be field-distorted.
FIELD_ON_V_PER_A = 0.2

# --------------------------------------------------------------- protocol
TIMESTEP_FS = 1.0
TEMPERATURE_K = 300.0
THERMO_DAMP_FS = 100.0        # Nose-Hoover damping (fix rigid/nvt/small)
RAMP_PS = 50.0                # temperature ramp 10 -> 300 K
EQUIL_PS = 400.0              # NVT equilibration at 300 K
PROD_PS = 100.0               # production
DUMP_EVERY_PS = 5.0           # >= 1 ps decorrelation; 100/5 = 20 frames
CUTOFF_A = 12.0               # real-space Coulomb + LJ cutoff
PPPM_REL_ACC = 1.0e-4

# --------------------------------------------------------------- systems
SYSTEMS = {
    # name: dict(CNT chiral, unit cells along z, field, composition, seeds)
    "cnt10_field0": dict(n=10, ncell=20, field="off"),
    "cnt10_field1": dict(n=10, ncell=20, field="on"),
    "cnt16_field0": dict(n=16, ncell=20, field="off"),
    "cnt16_field1": dict(n=16, ncell=20, field="on"),
}
# velocities seed, build seed (deterministic packing)
SEEDS = {
    "cnt10_field0": dict(build=20261001, velocity=730101),
    "cnt10_field1": dict(build=20261002, velocity=730102),
    "cnt16_field0": dict(build=20261003, velocity=730103),
    "cnt16_field1": dict(build=20261004, velocity=730104),
}
# (original seed set restored: with the pack-then-relax packer the original
# cnt16 seeds are expected to work again; if a seed fails it is re-chosen
# by scan and recorded both here and in provenance)

# Composition targets (see build_systems.py for the packing rules):
# EC:EMC = 1:1 by volume (n_EC * MW/rho = n_EMC * MW/rho using
# rho_EC = 1.321 g/cm3, rho_EMC = 1.00 g/cm3 [CRC values], i.e.
# n_EC/n_EMC = 1.56); LiPF6 count set to give ~1 M in the vdW-corrected
# pore volume V = pi*(R-3.2A)^2 * Lz (small tube: 2 pairs -> 1.6-1.7 M,
# kept for minimal ion statistics - recorded in provenance).
COMPOSITION = {
    # (10,10): the previous agent's original 9 EC + 6 EMC target was 23%
    # over the bulk mixture density in the vdW pore (1.44 g/cm3 vs 1.17)
    # AND un-packable; their trim to 8/5 sits at 1.28 g/cm3 (+9.5% over
    # bulk, reasonable for a strongly confined pore) and packs - kept.
    # (16,16): 37/24/6 = 1.22 g/cm3 (+4.5% over bulk). With the corrected
    # extended EMC template, pure rejection sampling jams at ~55-60% of
    # bulk density, so 37/24 was unreachable by rejection (the previous
    # agent's successes used the internally collapsed EMC). Restored
    # 2026-10-14 by upgrading the packer to pack-then-relax (the repo's
    # W8 approach): tolerant best-of-N insertion + rigid-body push-apart
    # relaxation, final config verified against the same strict floors.
    10: dict(n_EC=8, n_EMC=5, n_LiPF6=2),
    16: dict(n_EC=37, n_EMC=24, n_LiPF6=6),
}

# solvent number density used only for reporting (bulk EC/EMC ~0.088 atoms/A^3)
BULK_ATOM_DENSITY = 0.088  # atoms/A^3
