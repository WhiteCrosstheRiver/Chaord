# Acceptance details

Per-case numbers behind `reports/acceptance.json`.  Written by `tools/acceptance.py`; the verification logic (counting, parsing, arithmetic) is independent of the chaord check helpers.

## A1 parse and format — PASS

**Evidence:** 9/9 spec examples parse; property laws (parse==IR and fmt idempotent) hold on 10000 generated programs


## A2 canonical invariance — PASS

**Evidence:** 9/9 crystal + random-solution cases; 18/18 rigid transforms (rotation + translation + re-imaging + re-ordering, 2 per case) lift to byte-identical text

| case | transforms identical |
| --- | --- |
| crystals/fcc_cu | 2/2 |
| crystals/bcc_fe | 2/2 |
| crystals/rocksalt_nacl | 2/2 |
| crystals/l12_ni3al | 2/2 |
| crystals/hcp_mg | 2/2 |
| crystals/diamond_si | 2/2 |
| crystals/perovskite_srtio3 | 2/2 |
| crystals/fcc_crconi | 2/2 |
| solutions/cuau_random | 2/2 |

## A3 exact round trip (ordered) — PASS

**Evidence:** 9/9 lift-build-lift texts byte-identical; 9/9 structure fits original vs rebuilt (species-aware for ordered cases; species-blind lattice/positions + Warren-Cowley alpha vs relabel noise floor for random solutions)

| case | lift-build-lift text | StructureMatcher |
| --- | --- | --- |
| crystals/fcc_cu | True | True |
| crystals/bcc_fe | True | True |
| crystals/rocksalt_nacl | True | True |
| crystals/l12_ni3al | True | True |
| crystals/hcp_mg | True | True |
| crystals/diamond_si | True | True |
| crystals/perovskite_srtio3 | True | True |
| crystals/fcc_crconi | True | True |
| solutions/cuau_random | True | True |

## A4 defect recovery — FAIL

**Evidence:** 22 host x defect-type x temperature cells (3 hosts, all four K-V kinds); precision >= 0.95 in 18/22, recall >= 0.95 in 22/22; worst precision 0.09 (L12-NiAl/interstitial/0.8Tm), worst recall 1.00 (fcc-Cu/vacancy/room)

| host | type | temp | planted | detected | P | R | tp/fp/fn |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fcc-Cu | vacancy | room | 6 | V_Cu:6 | 1.00 | 1.00 | 6/0/0 |
| fcc-Cu | vacancy | 0.8Tm | 6 | V_Cu:6 | 1.00 | 1.00 | 6/0/0 |
| fcc-Cu | interstitial | room | 4 | Cu_i:4 | 1.00 | 1.00 | 4/0/0 |
| fcc-Cu | interstitial | 0.8Tm | 4 | Cu_i:4,frenkel_pair:13 | 0.24 | 1.00 | 4/13/0 |
| fcc-Cu | frenkel | room | 4 | frenkel_pair:4 | 1.00 | 1.00 | 4/0/0 |
| fcc-Cu | frenkel | 0.8Tm | 4 | frenkel_pair:10 | 1.00 | 1.00 | 4/0/0 |
| L12-NiAl | vacancy | room | 3 | V_Ni:3 | 1.00 | 1.00 | 3/0/0 |
| L12-NiAl | vacancy | 0.8Tm | 3 | V_Ni:3,frenkel_pair:3 | 0.50 | 1.00 | 3/3/0 |
| L12-NiAl | antisite | room | 4 | Al_Ni:4 | 1.00 | 1.00 | 4/0/0 |
| L12-NiAl | antisite | 0.8Tm | 4 | Al_Ni:4 | 1.00 | 1.00 | 4/0/0 |
| L12-NiAl | interstitial | room | 3 | Ni_i:3 | 1.00 | 1.00 | 3/0/0 |
| L12-NiAl | interstitial | 0.8Tm | 3 | Ni_i:3,frenkel_pair:30 | 0.09 | 1.00 | 3/30/0 |
| L12-NiAl | frenkel | room | 3 | frenkel_pair:3 | 1.00 | 1.00 | 3/0/0 |
| L12-NiAl | frenkel | 0.8Tm | 3 | frenkel_pair:4 | 1.00 | 1.00 | 3/0/0 |
| NaCl | vacancy | room | 6 | V_Na:6 | 1.00 | 1.00 | 6/0/0 |
| NaCl | vacancy | 0.8Tm | 6 | V_Na:6 | 1.00 | 1.00 | 6/0/0 |
| NaCl | antisite | room | 4 | Cl_Na:4 | 1.00 | 1.00 | 4/0/0 |
| NaCl | antisite | 0.8Tm | 4 | Cl_Na:4 | 1.00 | 1.00 | 4/0/0 |
| NaCl | interstitial | room | 3 | Na_i:3 | 1.00 | 1.00 | 3/0/0 |
| NaCl | interstitial | 0.8Tm | 3 | Na_i:3,frenkel_pair:5 | 0.38 | 1.00 | 3/5/0 |
| NaCl | frenkel | room | 3 | frenkel_pair:3 | 1.00 | 1.00 | 3/0/0 |
| NaCl | frenkel | 0.8Tm | 3 | frenkel_pair:7 | 1.00 | 1.00 | 3/0/0 |

## A5 statistical round trip — FAIL

**Evidence:** pass rate 2/5-with-floor; fluid: 1/3 (target >= 90%); interface: 1/1 (target >= 90%); glass: 0/1 (target >= 80%); 4 cases without a floor on record: no-floor: skipped (synthetic frame); 2 could not round-trip: interface/lj_solid_liquid build-failed; interfaces/cu_water build-failed

| case | category | status | distances | ratios |
| --- | --- | --- | --- | --- |
| reference/lj_glass | glass | fail (cn_tv 0.180 vs floor 0.053 (x3.4); gr_rms 0.835 vs floor 0.100 (x8.4)) |  |  |
| reference/lj_liquid | fluid | pass (cn_tv 0.066 vs floor 0.060 (x1.1); gr_rms 0.090 vs floor 0.092 (x1.0)) |  |  |
| reference/lj_solid_liquid | interface | pass (cn_tv 0.064 vs floor 0.059 (x1.1); gr_rms 0.078 vs floor 0.068 (x1.2)) |  |  |
| reference/nacl_aq | fluid | fail (cn_tv 0.165 vs floor 0.042 (x3.9); gr_rms 1.993 vs floor 0.056 (x35.7)) |  |  |
| reference/water_tip4p | fluid | fail (cn_tv 0.174 vs floor 0.073 (x2.4); gr_rms 0.105 vs floor 0.048 (x2.2)) |  |  |
| fluid/water_box15 | fluid | no-floor: skipped (synthetic frame) | gr_rms=0.138, cn_tv=0.100 |  |
| fluid/ar_gas_box25 | fluid | no-floor: skipped (synthetic frame) | gr_rms=0.648, cn_tv=0.000 |  |
| fluid/n2_box22 | fluid | no-floor: skipped (synthetic frame) | gr_rms=0.582, cn_tv=0.019 |  |
| glass/lj_glass_rho085 | glass | no-floor: skipped (synthetic frame) | gr_rms=1.285, cn_tv=0.265 |  |
| interface/lj_solid_liquid | interface | build-failed (rebuild exceeded the time budget) |  |  |
| interfaces/cu_water | interfaces | build-failed (chaord.lang.errors.ChaordError: unknown species 'Cu384'; known molecules: Ar, CO) |  |  |

## A6 conservation — PASS

**Evidence:** 126 lifts checked: frame count == conserve line on 126/126; verifier's own region arithmetic (composition/occupancy/molecules/defect net/residual) pins or cross-checks 71 programs (non-derivable shapes check two-way); charge consistent on 126/126

| case | frame | frame counts | conserve | derivation | derived | ok | charge |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crystals/fcc_cu | 0 | {'Cu': 32} | {'Cu': 32} | sites-only: 32 fcc sites + net 0 + residual 0 [total 32 == 32] | - | True | no ionic species |
| crystals/fcc_cu | 1 | {'Cu': 32} | {'Cu': 32} | sites-only: 32 fcc sites + net 0 + residual 0 [total 32 == 32] | - | True | no ionic species |
| crystals/fcc_cu | 2 | {'Cu': 32} | {'Cu': 32} | sites-only: 32 fcc sites + net 0 + residual 0 [total 32 == 32] | - | True | no ionic species |
| crystals/fcc_cu | 3 | {'Cu': 32} | {'Cu': 32} | sites-only: 32 fcc sites + net 0 + residual 0 [total 32 == 32] | - | True | no ionic species |
| crystals/fcc_cu | 4 | {'Cu': 32} | {'Cu': 32} | sites-only: 32 fcc sites + net 0 + residual 0 [total 32 == 32] | - | True | no ionic species |
| crystals/bcc_fe | 0 | {'Fe': 16} | {'Fe': 16} | sites-only: 16 bcc sites + net 0 + residual 0 [total 16 == 16] | - | True | no ionic species |
| crystals/bcc_fe | 1 | {'Fe': 16} | {'Fe': 16} | sites-only: 16 bcc sites + net 0 + residual 0 [total 16 == 16] | - | True | no ionic species |
| crystals/bcc_fe | 2 | {'Fe': 16} | {'Fe': 16} | sites-only: 16 bcc sites + net 0 + residual 0 [total 16 == 16] | - | True | no ionic species |
| crystals/bcc_fe | 3 | {'Fe': 16} | {'Fe': 16} | sites-only: 16 bcc sites + net 0 + residual 0 [total 16 == 16] | - | True | no ionic species |
| crystals/bcc_fe | 4 | {'Fe': 16} | {'Fe': 16} | sites-only: 16 bcc sites + net 0 + residual 0 [total 16 == 16] | - | True | no ionic species |
| crystals/rocksalt_nacl | 0 | {'Cl': 32, 'Na': 32} | {'Cl': 32, 'Na': 32} | rocksaltx8+composition | {'Na': 32, 'Cl': 32} | True | charge +0 == frame +0 |
| crystals/rocksalt_nacl | 1 | {'Cl': 32, 'Na': 32} | {'Cl': 32, 'Na': 32} | rocksaltx8+composition | {'Na': 32, 'Cl': 32} | True | charge +0 == frame +0 |
| crystals/rocksalt_nacl | 2 | {'Cl': 32, 'Na': 32} | {'Cl': 32, 'Na': 32} | rocksaltx8+composition | {'Na': 32, 'Cl': 32} | True | charge +0 == frame +0 |
| crystals/rocksalt_nacl | 3 | {'Cl': 32, 'Na': 32} | {'Cl': 32, 'Na': 32} | rocksaltx8+composition | {'Na': 32, 'Cl': 32} | True | charge +0 == frame +0 |
| crystals/rocksalt_nacl | 4 | {'Cl': 32, 'Na': 32} | {'Cl': 32, 'Na': 32} | rocksaltx8+composition | {'Na': 32, 'Cl': 32} | True | charge +0 == frame +0 |
| crystals/l12_ni3al | 0 | {'Al': 343, 'Ni': 1029} | {'Al': 343, 'Ni': 1029} | L1_2x343+composition | {'Ni': 1029, 'Al': 343} | True | no ionic species |
| crystals/l12_ni3al | 1 | {'Al': 343, 'Ni': 1029} | {'Al': 343, 'Ni': 1029} | L1_2x343+composition | {'Ni': 1029, 'Al': 343} | True | no ionic species |
| crystals/l12_ni3al | 2 | {'Al': 343, 'Ni': 1029} | {'Al': 343, 'Ni': 1029} | L1_2x343+composition | {'Ni': 1029, 'Al': 343} | True | no ionic species |
| crystals/l12_ni3al | 3 | {'Al': 343, 'Ni': 1029} | {'Al': 343, 'Ni': 1029} | L1_2x343+composition | {'Ni': 1029, 'Al': 343} | True | no ionic species |
| crystals/l12_ni3al | 4 | {'Al': 343, 'Ni': 1029} | {'Al': 343, 'Ni': 1029} | L1_2x343+composition | {'Ni': 1029, 'Al': 343} | True | no ionic species |
| crystals/hcp_mg | 0 | {'Mg': 32} | {'Mg': 32} | unknown molecule Mg32 | - | True | no ionic species |
| crystals/hcp_mg | 1 | {'Mg': 32} | {'Mg': 32} | unknown molecule Mg32 | - | True | no ionic species |
| crystals/hcp_mg | 2 | {'Mg': 32} | {'Mg': 32} | unknown molecule Mg32 | - | True | no ionic species |
| crystals/hcp_mg | 3 | {'Mg': 32} | {'Mg': 32} | unknown molecule Mg32 | - | True | no ionic species |
| crystals/hcp_mg | 4 | {'Mg': 32} | {'Mg': 32} | unknown molecule Mg32 | - | True | no ionic species |
| crystals/diamond_si | 0 | {'Si': 64} | {'Si': 64} | sites-only: 64 diamond sites + net 0 + residual 0 [total 64 == 64] | - | True | no ionic species |
| crystals/diamond_si | 1 | {'Si': 64} | {'Si': 64} | sites-only: 64 diamond sites + net 0 + residual 0 [total 64 == 64] | - | True | no ionic species |
| crystals/diamond_si | 2 | {'Si': 64} | {'Si': 64} | sites-only: 64 diamond sites + net 0 + residual 0 [total 64 == 64] | - | True | no ionic species |
| crystals/diamond_si | 3 | {'Si': 64} | {'Si': 64} | sites-only: 64 diamond sites + net 0 + residual 0 [total 64 == 64] | - | True | no ionic species |
| crystals/diamond_si | 4 | {'Si': 64} | {'Si': 64} | sites-only: 64 diamond sites + net 0 + residual 0 [total 64 == 64] | - | True | no ionic species |
| crystals/perovskite_srtio3 | 0 | {'O': 24, 'Sr': 8, 'Ti': 8} | {'O': 24, 'Sr': 8, 'Ti': 8} | perovskitex8+composition | {'Sr': 8, 'Ti': 8, 'O': 24} | True | no ionic species |
| crystals/perovskite_srtio3 | 1 | {'O': 24, 'Sr': 8, 'Ti': 8} | {'O': 24, 'Sr': 8, 'Ti': 8} | perovskitex8+composition | {'Sr': 8, 'Ti': 8, 'O': 24} | True | no ionic species |
| crystals/perovskite_srtio3 | 2 | {'O': 24, 'Sr': 8, 'Ti': 8} | {'O': 24, 'Sr': 8, 'Ti': 8} | perovskitex8+composition | {'Sr': 8, 'Ti': 8, 'O': 24} | True | no ionic species |
| crystals/perovskite_srtio3 | 3 | {'O': 24, 'Sr': 8, 'Ti': 8} | {'O': 24, 'Sr': 8, 'Ti': 8} | perovskitex8+composition | {'Sr': 8, 'Ti': 8, 'O': 24} | True | no ionic species |
| crystals/perovskite_srtio3 | 4 | {'O': 24, 'Sr': 8, 'Ti': 8} | {'O': 24, 'Sr': 8, 'Ti': 8} | perovskitex8+composition | {'Sr': 8, 'Ti': 8, 'O': 24} | True | no ionic species |
| crystals/fcc_crconi | 0 | {'Co': 36, 'Cr': 36, 'Ni': 36} | {'Co': 36, 'Cr': 36, 'Ni': 36} | occupancy on 108 sites x27 cells | {'Co': 36, 'Cr': 36, 'Ni': 36} | True | no ionic species |
| crystals/fcc_crconi | 1 | {'Co': 36, 'Cr': 36, 'Ni': 36} | {'Co': 36, 'Cr': 36, 'Ni': 36} | occupancy on 108 sites x27 cells | {'Co': 36, 'Cr': 36, 'Ni': 36} | True | no ionic species |
| crystals/fcc_crconi | 2 | {'Co': 36, 'Cr': 36, 'Ni': 36} | {'Co': 36, 'Cr': 36, 'Ni': 36} | occupancy on 108 sites x27 cells | {'Co': 36, 'Cr': 36, 'Ni': 36} | True | no ionic species |
| crystals/fcc_crconi | 3 | {'Co': 36, 'Cr': 36, 'Ni': 36} | {'Co': 36, 'Cr': 36, 'Ni': 36} | occupancy on 108 sites x27 cells | {'Co': 36, 'Cr': 36, 'Ni': 36} | True | no ionic species |
| crystals/fcc_crconi | 4 | {'Co': 36, 'Cr': 36, 'Ni': 36} | {'Co': 36, 'Cr': 36, 'Ni': 36} | occupancy on 108 sites x27 cells | {'Co': 36, 'Cr': 36, 'Ni': 36} | True | no ionic species |
| solutions/cuau_random | 0 | {'Au': 54, 'Cu': 54} | {'Au': 54, 'Cu': 54} | occupancy on 108 sites x27 cells | {'Au': 54, 'Cu': 54} | True | no ionic species |
| solutions/cuau_random | 1 | {'Au': 54, 'Cu': 54} | {'Au': 54, 'Cu': 54} | occupancy on 108 sites x27 cells | {'Au': 54, 'Cu': 54} | True | no ionic species |
| solutions/cuau_random | 2 | {'Au': 54, 'Cu': 54} | {'Au': 54, 'Cu': 54} | occupancy on 108 sites x27 cells | {'Au': 54, 'Cu': 54} | True | no ionic species |
| solutions/cuau_random | 3 | {'Au': 54, 'Cu': 54} | {'Au': 54, 'Cu': 54} | occupancy on 108 sites x27 cells | {'Au': 54, 'Cu': 54} | True | no ionic species |
| solutions/cuau_random | 4 | {'Au': 54, 'Cu': 54} | {'Au': 54, 'Cu': 54} | occupancy on 108 sites x27 cells | {'Au': 54, 'Cu': 54} | True | no ionic species |
| defects/fcc_cu_vacancies | 0 | {'Cu': 105} | {'Cu': 105} | sites-only: 108 fcc sites + net -3 + residual 0 [total 105 == 105] | - | True | no ionic species |
| defects/fcc_cu_vacancies | 1 | {'Cu': 105} | {'Cu': 105} | sites-only: 108 fcc sites + net -3 + residual 0 [total 105 == 105] | - | True | no ionic species |
| defects/fcc_cu_vacancies | 2 | {'Cu': 105} | {'Cu': 105} | sites-only: 108 fcc sites + net -3 + residual 0 [total 105 == 105] | - | True | no ionic species |
| defects/fcc_cu_vacancies | 3 | {'Cu': 105} | {'Cu': 105} | sites-only: 108 fcc sites + net -3 + residual 0 [total 105 == 105] | - | True | no ionic species |
| defects/fcc_cu_vacancies | 4 | {'Cu': 105} | {'Cu': 105} | sites-only: 108 fcc sites + net -3 + residual 0 [total 105 == 105] | - | True | no ionic species |
| defects/l12_ni3al_vac_antisite | 0 | {'Al': 68, 'Ni': 185} | {'Al': 68, 'Ni': 185} | L1_2x64+composition | {'Ni': 185, 'Al': 68} | True | no ionic species |
| defects/l12_ni3al_vac_antisite | 1 | {'Al': 68, 'Ni': 185} | {'Al': 68, 'Ni': 185} | L1_2x64+composition | {'Ni': 185, 'Al': 68} | True | no ionic species |
| defects/l12_ni3al_vac_antisite | 2 | {'Al': 68, 'Ni': 185} | {'Al': 68, 'Ni': 185} | L1_2x64+composition | {'Ni': 185, 'Al': 68} | True | no ionic species |
| defects/l12_ni3al_vac_antisite | 3 | {'Al': 68, 'Ni': 185} | {'Al': 68, 'Ni': 185} | L1_2x64+composition | {'Ni': 185, 'Al': 68} | True | no ionic species |
| defects/l12_ni3al_vac_antisite | 4 | {'Al': 68, 'Ni': 185} | {'Al': 68, 'Ni': 185} | L1_2x64+composition | {'Ni': 185, 'Al': 68} | True | no ionic species |
| defects/rocksalt_nacl_vna | 0 | {'Cl': 108, 'Na': 102} | {'Cl': 108, 'Na': 102} | rocksaltx27+composition | {'Na': 102, 'Cl': 108} | True | charge -6 == frame -6 |
| defects/rocksalt_nacl_vna | 1 | {'Cl': 108, 'Na': 102} | {'Cl': 108, 'Na': 102} | rocksaltx27+composition | {'Na': 102, 'Cl': 108} | True | charge -6 == frame -6 |
| defects/rocksalt_nacl_vna | 2 | {'Cl': 108, 'Na': 102} | {'Cl': 108, 'Na': 102} | rocksaltx27+composition | {'Na': 102, 'Cl': 108} | True | charge -6 == frame -6 |
| defects/rocksalt_nacl_vna | 3 | {'Cl': 108, 'Na': 102} | {'Cl': 108, 'Na': 102} | rocksaltx27+composition | {'Na': 102, 'Cl': 108} | True | charge -6 == frame -6 |
| defects/rocksalt_nacl_vna | 4 | {'Cl': 108, 'Na': 102} | {'Cl': 108, 'Na': 102} | rocksaltx27+composition | {'Cl': 108, 'Na': 102} | True | charge -6 == frame -6 |
| fluid/water_box15 | 0 | {'H': 120, 'O': 60} | {'H': 120, 'O': 60} | molecules | {'H': 120, 'O': 60} | True | no ionic species |
| fluid/water_box15 | 1 | {'H': 120, 'O': 60} | {'H': 120, 'O': 60} | molecules | {'H': 120, 'O': 60} | True | no ionic species |
| fluid/water_box15 | 2 | {'H': 120, 'O': 60} | {'H': 120, 'O': 60} | molecules | {'H': 120, 'O': 60} | True | no ionic species |
| fluid/water_box15 | 3 | {'H': 120, 'O': 60} | {'H': 120, 'O': 60} | molecules | {'H': 120, 'O': 60} | True | no ionic species |
| fluid/water_box15 | 4 | {'H': 120, 'O': 60} | {'H': 120, 'O': 60} | molecules | {'H': 120, 'O': 60} | True | no ionic species |
| fluid/ar_gas_box25 | 0 | {'Ar': 50} | {'Ar': 50} | atomic fluid region (no composition statement) | - | True | no ionic species |
| fluid/ar_gas_box25 | 1 | {'Ar': 50} | {'Ar': 50} | atomic fluid region (no composition statement) | - | True | no ionic species |
| fluid/ar_gas_box25 | 2 | {'Ar': 50} | {'Ar': 50} | atomic fluid region (no composition statement) | - | True | no ionic species |
| fluid/ar_gas_box25 | 3 | {'Ar': 50} | {'Ar': 50} | atomic fluid region (no composition statement) | - | True | no ionic species |
| fluid/ar_gas_box25 | 4 | {'Ar': 50} | {'Ar': 50} | atomic fluid region (no composition statement) | - | True | no ionic species |
| fluid/n2_box22 | 0 | {'N': 160} | {'N': 160} | molecules | {'N': 160} | True | no ionic species |
| fluid/n2_box22 | 1 | {'N': 160} | {'N': 160} | molecules | {'N': 160} | True | no ionic species |
| fluid/n2_box22 | 2 | {'N': 160} | {'N': 160} | molecules | {'N': 160} | True | no ionic species |
| fluid/n2_box22 | 3 | {'N': 160} | {'N': 160} | molecules | {'N': 160} | True | no ionic species |
| fluid/n2_box22 | 4 | {'N': 160} | {'N': 160} | molecules | {'N': 160} | True | no ionic species |
| reactive/water_oh_h_box20 | 0 | {'H': 118, 'O': 59} | {'H': 118, 'O': 59} | unknown molecule OH | - | True | no ionic species |
| reactive/water_oh_h_box20 | 1 | {'H': 118, 'O': 59} | {'H': 118, 'O': 59} | unknown molecule OH | - | True | no ionic species |
| reactive/water_oh_h_box20 | 2 | {'H': 118, 'O': 59} | {'H': 118, 'O': 59} | unknown molecule OH | - | True | no ionic species |
| reactive/water_oh_h_box20 | 3 | {'H': 118, 'O': 59} | {'H': 118, 'O': 59} | unknown molecule OH | - | True | no ionic species |
| reactive/water_oh_h_box20 | 4 | {'H': 118, 'O': 59} | {'H': 118, 'O': 59} | unknown molecule OH | - | True | no ionic species |
| glass/lj_glass_rho085 | 0 | {'X': 200} | {'X': 200} | amorphous composition names no counts | - | True | no ionic species |
| glass/lj_glass_rho085 | 1 | {'X': 200} | {'X': 200} | amorphous composition names no counts | - | True | no ionic species |
| glass/lj_glass_rho085 | 2 | {'X': 200} | {'X': 200} | amorphous composition names no counts | - | True | no ionic species |
| glass/lj_glass_rho085 | 3 | {'X': 200} | {'X': 200} | amorphous composition names no counts | - | True | no ionic species |
| glass/lj_glass_rho085 | 4 | {'X': 200} | {'X': 200} | amorphous composition names no counts | - | True | no ionic species |
| interface/lj_solid_liquid | 0 | {'X': 512} | {'X': 512} | atomic fluid region (no composition statement) | - | True | no ionic species |
| interface/lj_solid_liquid | 1 | {'X': 512} | {'X': 512} | atomic fluid region (no composition statement) | - | True | no ionic species |
| interface/lj_solid_liquid | 2 | {'X': 512} | {'X': 512} | atomic fluid region (no composition statement) | - | True | no ionic species |
| interface/lj_solid_liquid | 3 | {'X': 512} | {'X': 512} | atomic fluid region (no composition statement) | - | True | no ionic species |
| interface/lj_solid_liquid | 4 | {'X': 512} | {'X': 512} | atomic fluid region (no composition statement) | - | True | no ionic species |
| surface/ni111_o_top | 0 | {'Ni': 72, 'O': 8} | {'Ni': 72, 'O': 8} | unsupported phase | - | True | no ionic species |
| surface/ni111_o_top | 1 | {'Ni': 72, 'O': 8} | {'Ni': 72, 'O': 8} | unsupported phase | - | True | no ionic species |
| surface/ni111_o_top | 2 | {'Ni': 72, 'O': 8} | {'Ni': 72, 'O': 8} | unsupported phase | - | True | no ionic species |
| surface/ni111_o_top | 3 | {'Ni': 72, 'O': 8} | {'Ni': 72, 'O': 8} | unsupported phase | - | True | no ionic species |
| surface/ni111_o_top | 4 | {'Ni': 72, 'O': 8} | {'Ni': 72, 'O': 8} | unsupported phase | - | True | no ionic species |
| surfaces/pt111_o | 0 | {'O': 8, 'Pt': 72} | {'O': 8, 'Pt': 72} | unsupported phase | - | True | no ionic species |
| surfaces/pt111_o | 1 | {'O': 8, 'Pt': 72} | {'O': 8, 'Pt': 72} | unsupported phase | - | True | no ionic species |
| surfaces/pt111_o | 2 | {'O': 8, 'Pt': 72} | {'O': 8, 'Pt': 72} | unsupported phase | - | True | no ionic species |
| surfaces/pt111_o | 3 | {'O': 8, 'Pt': 72} | {'O': 8, 'Pt': 72} | unsupported phase | - | True | no ionic species |
| surfaces/pt111_o | 4 | {'O': 8, 'Pt': 72} | {'O': 8, 'Pt': 72} | unsupported phase | - | True | no ionic species |
| solutions/nacl_aq | 0 | {'Cl': 8, 'H': 240, 'Na': 8, 'O': 120} | {'Cl': 8, 'H': 240, 'Na': 8, 'O': 120} | unknown molecule Cl- | - | True | charge +0 == frame +0 |
| solutions/nacl_aq | 1 | {'Cl': 8, 'H': 240, 'Na': 8, 'O': 120} | {'Cl': 8, 'H': 240, 'Na': 8, 'O': 120} | unknown molecule Cl- | - | True | charge +0 == frame +0 |
| solutions/nacl_aq | 2 | {'Cl': 8, 'H': 240, 'Na': 8, 'O': 120} | {'Cl': 8, 'H': 240, 'Na': 8, 'O': 120} | unknown molecule Cl- | - | True | charge +0 == frame +0 |
| solutions/nacl_aq | 3 | {'Cl': 8, 'H': 240, 'Na': 8, 'O': 120} | {'Cl': 8, 'H': 240, 'Na': 8, 'O': 120} | unknown molecule Cl- | - | True | charge +0 == frame +0 |
| solutions/nacl_aq | 4 | {'Cl': 8, 'H': 240, 'Na': 8, 'O': 120} | {'Cl': 8, 'H': 240, 'Na': 8, 'O': 120} | unknown molecule Cl- | - | True | charge +0 == frame +0 |
| solutions/lipf6_ec | 0 | {'C': 120, 'F': 24, 'H': 160, 'Li': 4, 'O': 120, 'P': 4} | {'C': 120, 'F': 24, 'H': 160, 'Li': 4, 'O': 120, 'P': 4} | unknown molecule Li+ | - | True | charge skipped (polyatomic-ion elements present: element-wise charge undefined) |
| solutions/lipf6_ec | 1 | {'C': 120, 'F': 24, 'H': 160, 'Li': 4, 'O': 120, 'P': 4} | {'C': 120, 'F': 24, 'H': 160, 'Li': 4, 'O': 120, 'P': 4} | unknown molecule Li+ | - | True | charge skipped (polyatomic-ion elements present: element-wise charge undefined) |
| solutions/lipf6_ec | 2 | {'C': 120, 'F': 24, 'H': 160, 'Li': 4, 'O': 120, 'P': 4} | {'C': 120, 'F': 24, 'H': 160, 'Li': 4, 'O': 120, 'P': 4} | unknown molecule Li+ | - | True | charge skipped (polyatomic-ion elements present: element-wise charge undefined) |
| solutions/lipf6_ec | 3 | {'C': 120, 'F': 24, 'H': 160, 'Li': 4, 'O': 120, 'P': 4} | {'C': 120, 'F': 24, 'H': 160, 'Li': 4, 'O': 120, 'P': 4} | unknown molecule Li+ | - | True | charge skipped (polyatomic-ion elements present: element-wise charge undefined) |
| solutions/lipf6_ec | 4 | {'C': 120, 'F': 24, 'H': 160, 'Li': 4, 'O': 120, 'P': 4} | {'C': 120, 'F': 24, 'H': 160, 'Li': 4, 'O': 120, 'P': 4} | unknown molecule Li+ | - | True | charge skipped (polyatomic-ion elements present: element-wise charge undefined) |
| gases/co2_dense | 0 | {'C': 60, 'O': 120} | {'C': 60, 'O': 120} | molecules | {'C': 60, 'O': 120} | True | no ionic species |
| gases/co2_dense | 1 | {'C': 60, 'O': 120} | {'C': 60, 'O': 120} | molecules | {'C': 60, 'O': 120} | True | no ionic species |
| gases/co2_dense | 2 | {'C': 60, 'O': 120} | {'C': 60, 'O': 120} | molecules | {'C': 60, 'O': 120} | True | no ionic species |
| gases/co2_dense | 3 | {'C': 60, 'O': 120} | {'C': 60, 'O': 120} | molecules | {'C': 60, 'O': 120} | True | no ionic species |
| gases/co2_dense | 4 | {'C': 60, 'O': 120} | {'C': 60, 'O': 120} | molecules | {'C': 60, 'O': 120} | True | no ionic species |
| surfaces/si001_2x1 | 0 | {'Si': 63} | {'Si': 63} | unsupported phase | - | True | no ionic species |
| surfaces/si001_2x1 | 1 | {'Si': 63} | {'Si': 63} | unsupported phase | - | True | no ionic species |
| surfaces/si001_2x1 | 2 | {'Si': 63} | {'Si': 63} | unsupported phase | - | True | no ionic species |
| surfaces/si001_2x1 | 3 | {'Si': 63} | {'Si': 63} | unsupported phase | - | True | no ionic species |
| surfaces/si001_2x1 | 4 | {'Si': 63} | {'Si': 63} | unsupported phase | - | True | no ionic species |
| interfaces/cu_water | 0 | {'Cu': 384, 'H': 272, 'O': 136} | {'Cu': 384, 'H': 272, 'O': 136} | unknown molecule Cu384 | - | True | no ionic species |
| interfaces/cu_water | 1 | {'Cu': 384, 'H': 272, 'O': 136} | {'Cu': 384, 'H': 272, 'O': 136} | unknown molecule Cu384 | - | True | no ionic species |
| interfaces/cu_water | 2 | {'Cu': 384, 'H': 272, 'O': 136} | {'Cu': 384, 'H': 272, 'O': 136} | unknown molecule Cu384 | - | True | no ionic species |
| interfaces/cu_water | 3 | {'Cu': 384, 'H': 272, 'O': 136} | {'Cu': 384, 'H': 272, 'O': 136} | unknown molecule Cu384 | - | True | no ionic species |
| interfaces/cu_water | 4 | {'Cu': 384, 'H': 272, 'O': 136} | {'Cu': 384, 'H': 272, 'O': 136} | unknown molecule Cu384 | - | True | no ionic species |
| probe: rocksalt_nacl minus one Cl | 0 | {'Cl': 31, 'Na': 32} | {'Cl': 31, 'Na': 32} | rocksaltx8+composition | {'Na': 32, 'Cl': 31} | True | charge +1 == frame +1 [expected: charge +1] |

## A7 phase segmentation — PASS

**Evidence:** 10/10 interface frames (2 cases x 5 frames; interface band of 2 x d_NN excluded, d_NN = verifier's median nearest-neighbour distance) labelled >= 95% correct; worst 0.999 (interfaces/cu_water frame 3)

| case | frame | core atoms | accuracy |
| --- | --- | --- | --- |
| interface/lj_solid_liquid | 0 | 182 | 1.000 |
| interface/lj_solid_liquid | 1 | 186 | 1.000 |
| interface/lj_solid_liquid | 2 | 192 | 1.000 |
| interface/lj_solid_liquid | 3 | 202 | 1.000 |
| interface/lj_solid_liquid | 4 | 187 | 1.000 |
| interfaces/cu_water | 0 | 692 | 1.000 |
| interfaces/cu_water | 1 | 676 | 1.000 |
| interfaces/cu_water | 2 | 686 | 1.000 |
| interfaces/cu_water | 3 | 686 | 0.999 |
| interfaces/cu_water | 4 | 677 | 1.000 |

## A8 reactive census — PASS

**Evidence:** plan 0: census {'H2O': 40, 'HO': 9, 'H': 9} == planted {'H2O': 40, 'HO': 9, 'H': 9} (147 atoms, coordinates written directly in numpy); plan 1: census {'H2O': 25, 'HO': 5, 'H': 7} == planted {'H2O': 25, 'HO': 5, 'H': 7} (92 atoms, coordinates written directly in numpy)

| plan | expected | census | exact |
| --- | --- | --- | --- |
| 0 | {'H2O': 40, 'HO': 9, 'H': 9} | {'H2O': 40, 'HO': 9, 'H': 9} | True |
| 1 | {'H2O': 25, 'HO': 5, 'H': 7} | {'H2O': 25, 'HO': 5, 'H': 7} | True |

## A9 compression — PASS

**Evidence:** 1 measured systems with >= 1,000 atoms (bench frames); worst ratio 0.53% (crystals/l12_ni3al, 1372 atoms); per-case table in details

| case | atoms | program B | extxyz B | ratio % | source |
| --- | --- | --- | --- | --- | --- |
| crystals/bcc_fe | 16 | 335 | 977 | 34.29 | bench |
| crystals/diamond_si | 64 | 342 | 3620 | 9.45 | bench |
| crystals/fcc_crconi | 108 | 412 | 6041 | 6.82 | bench |
| crystals/fcc_cu | 32 | 335 | 1857 | 18.04 | bench |
| crystals/hcp_mg | 32 | 337 | 1871 | 18.01 | bench |
| crystals/l12_ni3al | 1372 | 399 | 75565 | 0.53 | bench |
| crystals/perovskite_srtio3 | 40 | 374 | 2297 | 16.28 | bench |
| crystals/rocksalt_nacl | 64 | 370 | 3620 | 10.22 | bench |
| defects/fcc_cu_vacancies | 105 | 360 | 5879 | 6.12 | bench |
| defects/l12_ni3al_vac_antisite | 253 | 412 | 14019 | 2.94 | bench |
| defects/rocksalt_nacl_vna | 210 | 393 | 11690 | 3.36 | bench |
| fluid/ar_gas_box25 | 50 | 379 | 2847 | 13.31 | bench |
| fluid/n2_box22 | 160 | 381 | 8898 | 4.28 | bench |
| fluid/water_box15 | 180 | 387 | 9998 | 3.87 | bench |
| gases/co2_dense | 180 | 387 | 9998 | 3.87 | bench |
| glass/lj_glass_rho085 | 200 | 460 | 11137 | 4.13 | bench |
| interface/lj_solid_liquid | 512 | 373 | 28300 | 1.32 | bench |
| interfaces/cu_water | 792 | 408 | 43661 | 0.93 | bench |
| reactive/water_oh_h_box20 | 177 | 420 | 9833 | 4.27 | bench |
| solutions/cuau_random | 108 | 370 | 6080 | 6.09 | bench |
| solutions/lipf6_ec | 432 | 447 | 23858 | 1.87 | bench |
| solutions/nacl_aq | 376 | 435 | 20778 | 2.09 | bench |
| surface/ni111_o_top | 80 | 485 | 4538 | 10.69 | bench |
| surfaces/pt111_o | 80 | 485 | 4532 | 10.70 | bench |
| surfaces/si001_2x1 | 63 | 560 | 3604 | 15.54 | bench |
| defects/l12_ni3al_vac_antisite tiled 2x2 (supplementary, not a raw bench frame) | 1012 | 415 | 55765 | 0.74 | tiled-supplementary |

## A10 determinism — PASS

**Evidence:** repeated lift of bench defect frame byte-identical: True; same-seed builds identical coordinates: True


## A11 speed — PASS

**Evidence:** lift_frame(mode=fluid) on 100,000 atoms took 4.0 s (target <= 120 s); input = fcc lattice at rho 0.85 with 0.10 x a jitter (thermally disordered, not equilibrated MD); program states the exact count: yes


## A12 static checks — PASS

**Evidence:** lattice_mismatch: caught; impossible_density: caught; charge_imbalance: caught; overlap_0.1sigma: caught


## A13 no crashes — PASS

**Evidence:** 125/125 bench frames lift without any exception; 0 atoms placed in residual blocks across successful lifts


## A14 documentation — PASS

**Evidence:** 35 distinct keys across dialect YAMLs; reference.md covers 35/35; example found for 26/35 keys (spec/examples; sketch_check pass); generated docs/reference_generated.md

