# Acceptance details

Per-case numbers behind `reports/acceptance.json`.  Written by `tools/acceptance.py`; the verification logic (counting, parsing, arithmetic) is independent of the chaord check helpers.

## A1 parse and format — PASS

**Evidence:** 10/10 spec examples parse; property laws (parse==IR and fmt idempotent) hold on 10000 generated programs


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

## A4 defect recovery — PASS

**Evidence:** 22 host x defect-type x temperature cells (3 hosts, all four K-V kinds); precision >= 0.95 in 22/22, recall >= 0.95 in 22/22; worst precision 1.00 (fcc-Cu/vacancy/room), worst recall 1.00 (fcc-Cu/vacancy/room)

| host | type | temp | planted | detected | P | R | tp/fp/fn |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fcc-Cu | vacancy | room | 6 | V_Cu:6 | 1.00 | 1.00 | 6/0/0 |
| fcc-Cu | vacancy | 0.8Tm | 6 | V_Cu:6 | 1.00 | 1.00 | 6/0/0 |
| fcc-Cu | interstitial | room | 4 | Cu_i:4 | 1.00 | 1.00 | 4/0/0 |
| fcc-Cu | interstitial | 0.8Tm | 4 | Cu_i:4 | 1.00 | 1.00 | 4/0/0 |
| fcc-Cu | frenkel | room | 4 | frenkel_pair:4 | 1.00 | 1.00 | 4/0/0 |
| fcc-Cu | frenkel | 0.8Tm | 4 | frenkel_pair:4 | 1.00 | 1.00 | 4/0/0 |
| L12-NiAl | vacancy | room | 3 | V_Ni:3 | 1.00 | 1.00 | 3/0/0 |
| L12-NiAl | vacancy | 0.8Tm | 3 | V_Ni:3 | 1.00 | 1.00 | 3/0/0 |
| L12-NiAl | antisite | room | 4 | Al_Ni:4 | 1.00 | 1.00 | 4/0/0 |
| L12-NiAl | antisite | 0.8Tm | 4 | Al_Ni:4 | 1.00 | 1.00 | 4/0/0 |
| L12-NiAl | interstitial | room | 3 | Ni_i:3 | 1.00 | 1.00 | 3/0/0 |
| L12-NiAl | interstitial | 0.8Tm | 3 | Ni_i:3 | 1.00 | 1.00 | 3/0/0 |
| L12-NiAl | frenkel | room | 3 | frenkel_pair:3 | 1.00 | 1.00 | 3/0/0 |
| L12-NiAl | frenkel | 0.8Tm | 3 | frenkel_pair:3 | 1.00 | 1.00 | 3/0/0 |
| NaCl | vacancy | room | 6 | V_Na:6 | 1.00 | 1.00 | 6/0/0 |
| NaCl | vacancy | 0.8Tm | 6 | V_Na:6 | 1.00 | 1.00 | 6/0/0 |
| NaCl | antisite | room | 4 | Cl_Na:4 | 1.00 | 1.00 | 4/0/0 |
| NaCl | antisite | 0.8Tm | 4 | Cl_Na:4 | 1.00 | 1.00 | 4/0/0 |
| NaCl | interstitial | room | 3 | Na_i:3 | 1.00 | 1.00 | 3/0/0 |
| NaCl | interstitial | 0.8Tm | 3 | Na_i:3 | 1.00 | 1.00 | 3/0/0 |
| NaCl | frenkel | room | 3 | frenkel_pair:3 | 1.00 | 1.00 | 3/0/0 |
| NaCl | frenkel | 0.8Tm | 3 | frenkel_pair:3 | 1.00 | 1.00 | 3/0/0 |

## A5 statistical round trip — PASS

**Evidence:** pass rate 6/6-with-floor; fluid: 4/4 (target >= 90%); interface: 1/1 (target >= 90%); glass: 1/1 (target >= 80%); 6 cases without a floor on record: no-floor: skipped (synthetic frame); floor provenance: floor = mean over the cross-quench frame pairs (last two, most-annealed frames of each of the 3 independent quenches, identical protocol, distinct seeds).; floor = mean over frame pairs at lag >= 2 (the decorrelated half; 6 of 10 pairs, residual short-lag correlation must not shrink the floor).; floor = pooled decorrelated frame pairs of 2 independent trajectories (within-trajectory lag >= 2 plus cross pairs of the decorrelated halves; cross sits only +1%/+10% above within -- an equilibrated liquid carries no preparation memory, unlike the glass quenches; the second trajectory supplies the independent pairs the floor's upper quantile needs).; floor = mean over frame pairs at lag >= 2 (the decorrelated half; 6 of 10 pairs, residual short-lag correlation must not shrink the floor); frames spaced 5 ps (beyond the water structural relaxation time, Review 2): the floor is not shrunk by residual inter-frame correlation. (full notes: details rows / noise_floors.json)

| case | category | status | T | backend | md_steps | distances | ratios |
| --- | --- | --- | --- | --- | --- | --- | --- |
| reference/lj_glass | glass | pass (cn_tv 0.061 vs floor 0.092 (x0.7; floor = max(mean 0.073, P90 0.092); draws 0.042/0.080); gr_rms 0.114 vs floor 0.229 (x0.5; floor = max(mean 0.165, P90 0.229); draws 0.122/0.106); floor: floor = mean over the cross-quench frame pairs (last two, most-annealed frames of each of the 3 independent quenches, identical protocol, distinct seeds). Within-one-quench pairs of the same frames (intra_quench below) share the anneal basin and sit closer; a floor built from them is too tight for a perfect independent rebuild (Review 2).) | - | lj | 10993 |  |  |
| reference/lj_liquid | fluid | pass (cn_tv 0.045 vs floor 0.076 (x0.6; floor = max(mean 0.056, P90 0.076); draws 0.034/0.056); gr_rms 0.077 vs floor 0.107 (x0.7; floor = max(mean 0.094, P90 0.107); draws 0.088/0.066); floor: floor = mean over frame pairs at lag >= 2 (the decorrelated half; 6 of 10 pairs, residual short-lag correlation must not shrink the floor)) | 0.65 | lj | 7400 |  |  |
| reference/lj_liquid_large | fluid | pass (cn_tv 0.011 vs floor 0.029 (x0.4; floor = max(mean 0.021, P90 0.029); draws 0.010/0.012); gr_rms 0.038 vs floor 0.053 (x0.7; floor = max(mean 0.042, P90 0.053); draws 0.042/0.033); floor: floor = pooled decorrelated frame pairs of 2 independent trajectories (within-trajectory lag >= 2 plus cross pairs of the decorrelated halves; cross sits only +1%/+10% above within -- an equilibrated liquid carries no preparation memory, unlike the glass quenches; the second trajectory supplies the independent pairs the floor's upper quantile needs)) | 0.65 | lj | 7400 |  |  |
| reference/lj_solid_liquid | interface | pass (cn_tv 0.087 vs floor 0.091 (x1.0; floor = max(mean 0.067, P90 0.091); draws 0.085/0.089); gr_rms 0.085 vs floor 0.087 (x1.0; floor = max(mean 0.071, P90 0.087); draws 0.086/0.084); floor: floor = mean over frame pairs at lag >= 2 (the decorrelated half; 6 of 10 pairs, residual short-lag correlation must not shrink the floor)) | 0.65 | lj | 7400 |  |  |
| reference/nacl_aq | fluid | pass (cn_tv 0.045 vs floor 0.041 (x1.1; floor = max(mean 0.033, P90 0.041); draws 0.046/0.045); gr_rms 0.034 vs floor 0.044 (x0.8; floor = max(mean 0.037, P90 0.044); draws 0.036/0.032); floor: floor = mean over frame pairs at lag >= 2 (the decorrelated half; 6 of 10 pairs, residual short-lag correlation must not shrink the floor)) | - | classical | 0 |  |  |
| reference/water_tip4p | fluid | pass (cn_tv 0.077 vs floor 0.086 (x0.9; floor = max(mean 0.073, P90 0.086); draws 0.069/0.086); gr_rms 0.056 vs floor 0.061 (x0.9; floor = max(mean 0.054, P90 0.061); draws 0.052/0.060); floor: floor = mean over frame pairs at lag >= 2 (the decorrelated half; 6 of 10 pairs, residual short-lag correlation must not shrink the floor); frames spaced 5 ps (beyond the water structural relaxation time, Review 2): the floor is not shrunk by residual inter-frame correlation) | - | classical | 0 |  |  |
| fluid/water_box15 | fluid | no-floor: skipped (synthetic frame) | - | classical | 0 | gr_rms=0.319, cn_tv=0.450 |  |
| fluid/ar_gas_box25 | fluid | no-floor: skipped (synthetic frame) | - | classical | 0 | gr_rms=0.769, cn_tv=0.160 |  |
| fluid/n2_box22 | fluid | no-floor: skipped (synthetic frame) | - | classical | 0 | gr_rms=1.126, cn_tv=0.319 |  |
| glass/lj_glass_rho085 | glass | no-floor: skipped (synthetic frame) | - | lj | 10993 | gr_rms=1.285, cn_tv=0.265 |  |
| interface/lj_solid_liquid | interface | no-floor: skipped (synthetic frame) | 0.65 | lj | 7400 | gr_rms=1.553, cn_tv=0.506 |  |
| interfaces/cu_water | interfaces | no-floor: skipped (synthetic frame) | 300 | eam | 1000 | gr_rms=1.222, cn_tv=0.226 |  |

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
| interface/lj_solid_liquid | 0 | {'X': 512} | {'X': 512} | non-trivial region geometry (counts not derivable) | - | True | no ionic species |
| interface/lj_solid_liquid | 1 | {'X': 512} | {'X': 512} | non-trivial region geometry (counts not derivable) | - | True | no ionic species |
| interface/lj_solid_liquid | 2 | {'X': 512} | {'X': 512} | non-trivial region geometry (counts not derivable) | - | True | no ionic species |
| interface/lj_solid_liquid | 3 | {'X': 512} | {'X': 512} | non-trivial region geometry (counts not derivable) | - | True | no ionic species |
| interface/lj_solid_liquid | 4 | {'X': 512} | {'X': 512} | non-trivial region geometry (counts not derivable) | - | True | no ionic species |
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
| interfaces/cu_water | 0 | {'Cu': 384, 'H': 272, 'O': 136} | {'Cu': 384, 'H': 272, 'O': 136} | non-trivial region geometry (counts not derivable) | - | True | no ionic species |
| interfaces/cu_water | 1 | {'Cu': 384, 'H': 272, 'O': 136} | {'Cu': 384, 'H': 272, 'O': 136} | non-trivial region geometry (counts not derivable) | - | True | no ionic species |
| interfaces/cu_water | 2 | {'Cu': 384, 'H': 272, 'O': 136} | {'Cu': 384, 'H': 272, 'O': 136} | non-trivial region geometry (counts not derivable) | - | True | no ionic species |
| interfaces/cu_water | 3 | {'Cu': 384, 'H': 272, 'O': 136} | {'Cu': 384, 'H': 272, 'O': 136} | non-trivial region geometry (counts not derivable) | - | True | no ionic species |
| interfaces/cu_water | 4 | {'Cu': 384, 'H': 272, 'O': 136} | {'Cu': 384, 'H': 272, 'O': 136} | non-trivial region geometry (counts not derivable) | - | True | no ionic species |
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

**Evidence:** 1 measured systems with >= 1,000 atoms (bench frames); worst ratio 0.49% (crystals/l12_ni3al, 1372 atoms); per-case table in details

| case | atoms | program B | extxyz B | ratio % | source |
| --- | --- | --- | --- | --- | --- |
| crystals/bcc_fe | 16 | 335 | 977 | 34.29 | bench |
| crystals/diamond_si | 64 | 342 | 3620 | 9.45 | bench |
| crystals/fcc_crconi | 108 | 383 | 6041 | 6.34 | bench |
| crystals/fcc_cu | 32 | 335 | 1857 | 18.04 | bench |
| crystals/hcp_mg | 32 | 354 | 1871 | 18.92 | bench |
| crystals/l12_ni3al | 1372 | 370 | 75565 | 0.49 | bench |
| crystals/perovskite_srtio3 | 40 | 374 | 2297 | 16.28 | bench |
| crystals/rocksalt_nacl | 64 | 370 | 3620 | 10.22 | bench |
| defects/fcc_cu_vacancies | 105 | 360 | 5879 | 6.12 | bench |
| defects/l12_ni3al_vac_antisite | 253 | 412 | 14019 | 2.94 | bench |
| defects/rocksalt_nacl_vna | 210 | 393 | 11690 | 3.36 | bench |
| fluid/ar_gas_box25 | 50 | 379 | 2847 | 13.31 | bench |
| fluid/n2_box22 | 160 | 381 | 8898 | 4.28 | bench |
| fluid/water_box15 | 180 | 401 | 9998 | 4.01 | bench |
| gases/co2_dense | 180 | 387 | 9998 | 3.87 | bench |
| glass/lj_glass_rho085 | 200 | 507 | 11137 | 4.55 | bench |
| interface/lj_solid_liquid | 512 | 766 | 28300 | 2.71 | bench |
| interfaces/cu_water | 792 | 755 | 43661 | 1.73 | bench |
| reactive/water_oh_h_box20 | 177 | 434 | 9833 | 4.41 | bench |
| solutions/cuau_random | 108 | 370 | 6080 | 6.09 | bench |
| solutions/lipf6_ec | 432 | 447 | 23858 | 1.87 | bench |
| solutions/nacl_aq | 376 | 449 | 20778 | 2.16 | bench |
| surface/ni111_o_top | 80 | 485 | 4538 | 10.69 | bench |
| surfaces/pt111_o | 80 | 485 | 4532 | 10.70 | bench |
| surfaces/si001_2x1 | 63 | 560 | 3604 | 15.54 | bench |
| defects/l12_ni3al_vac_antisite tiled 2x2 (supplementary, not a raw bench frame) | 1012 | 415 | 55765 | 0.74 | tiled-supplementary |

## A10 determinism — PASS

**Evidence:** repeated lift of bench defect frame byte-identical: True; same-seed builds identical coordinates: True


## A11 speed — PASS

**Evidence:** lift_frame(mode=fluid) on 100,000 atoms took 6.6 s (target <= 120 s); input = fcc lattice at rho 0.85 with 0.10 x a jitter (thermally disordered, not equilibrated MD); program states the exact count: yes


## A12 static checks — PASS

**Evidence:** lattice_mismatch: caught; impossible_density: caught; charge_imbalance: caught; overlap_0.1sigma: caught


## A13 no crashes — PASS

**Evidence:** 125/125 bench frames lift without any exception; 0 atoms placed in residual blocks across successful lifts


## A14 documentation — PASS

**Evidence:** 35 distinct keys across dialect YAMLs; reference.md covers 35/35; example found for 26/35 keys (spec/examples; sketch_check pass); generated docs/reference_generated.md

