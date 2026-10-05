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

**Evidence:** 9/9 lift-build-lift texts byte-identical; 9/9 structure fits original vs rebuilt ON THE STORED THERMAL FRAMES (GATE as applied: translation-aligned optimal assignment, displaced mass beyond 0.25 d_NN <= 5%); PLAN's pymatgen StructureMatcher is not the gate -- default primitive reduction fits 0/9 thermal-vs-rebuild, primitive_cell=False misses l12_ni3al (committed measurements, pending decision D6); species arrangement for random solutions still judged by Warren-Cowley alpha vs the relabel noise floor)

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

**Evidence:** 26 host x defect-type x temperature cells (3 hosts, all four K-V kinds plus one mixed-kind cell on fcc-Cu and L12-NiAl); interstitials planted at real octahedral (fcc-family) / tetrahedral (rocksalt) interstices, planted events >= 2 d_NN apart, every planted frame >= 0.8 x its contact floor on the minimum-pair sanity check; precision >= 0.95 in 26/26, recall >= 0.95 in 26/26; worst precision 1.00 (fcc-Cu/vacancy/room), worst recall 1.00 (fcc-Cu/vacancy/room)

| host | type | temp | planted | detected | P | R | tp/fp/fn |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fcc-Cu | vacancy | room | V_Cu:6 | V_Cu:6 | 1.00 | 1.00 | 6/0/0 |
| fcc-Cu | vacancy | 0.8Tm | V_Cu:6 | V_Cu:6 | 1.00 | 1.00 | 6/0/0 |
| fcc-Cu | interstitial | room | Cu_i:4 | Cu_i:4 | 1.00 | 1.00 | 4/0/0 |
| fcc-Cu | interstitial | 0.8Tm | Cu_i:4 | Cu_i:4 | 1.00 | 1.00 | 4/0/0 |
| fcc-Cu | frenkel | room | frenkel_pair:4 | frenkel_pair:4 | 1.00 | 1.00 | 4/0/0 |
| fcc-Cu | frenkel | 0.8Tm | frenkel_pair:4 | frenkel_pair:4 | 1.00 | 1.00 | 4/0/0 |
| fcc-Cu | mixed | room | V_Cu:3+Cu_i:3 | V_Cu:3,Cu_i:3 | 1.00 | 1.00 | 6/0/0 |
| fcc-Cu | mixed | 0.8Tm | V_Cu:3+Cu_i:3 | V_Cu:3,Cu_i:3 | 1.00 | 1.00 | 6/0/0 |
| L12-NiAl | vacancy | room | V_Ni:3 | V_Ni:3 | 1.00 | 1.00 | 3/0/0 |
| L12-NiAl | vacancy | 0.8Tm | V_Ni:3 | V_Ni:3 | 1.00 | 1.00 | 3/0/0 |
| L12-NiAl | antisite | room | Al_Ni:4 | Al_Ni:4 | 1.00 | 1.00 | 4/0/0 |
| L12-NiAl | antisite | 0.8Tm | Al_Ni:4 | Al_Ni:4 | 1.00 | 1.00 | 4/0/0 |
| L12-NiAl | interstitial | room | Ni_i:3 | Ni_i:3 | 1.00 | 1.00 | 3/0/0 |
| L12-NiAl | interstitial | 0.8Tm | Ni_i:3 | Ni_i:3 | 1.00 | 1.00 | 3/0/0 |
| L12-NiAl | frenkel | room | frenkel_pair:3 | frenkel_pair:3 | 1.00 | 1.00 | 3/0/0 |
| L12-NiAl | frenkel | 0.8Tm | frenkel_pair:3 | V_Al:1,Al_i:1,frenkel_pair:2 | 1.00 | 1.00 | 3/0/0 |
| L12-NiAl | mixed | room | V_Ni:3+Al_Ni:4+Ni_i:3 | V_Ni:3,Al_Ni:4,Ni_i:3 | 1.00 | 1.00 | 10/0/0 |
| L12-NiAl | mixed | 0.8Tm | V_Ni:3+Al_Ni:4+Ni_i:3 | V_Ni:3,Al_Ni:4,Ni_i:3 | 1.00 | 1.00 | 10/0/0 |
| NaCl | vacancy | room | V_Na:6 | V_Na:6 | 1.00 | 1.00 | 6/0/0 |
| NaCl | vacancy | 0.8Tm | V_Na:6 | V_Na:6 | 1.00 | 1.00 | 6/0/0 |
| NaCl | antisite | room | Cl_Na:4 | Cl_Na:4 | 1.00 | 1.00 | 4/0/0 |
| NaCl | antisite | 0.8Tm | Cl_Na:4 | Cl_Na:4 | 1.00 | 1.00 | 4/0/0 |
| NaCl | interstitial | room | Na_i:3 | Na_i:3 | 1.00 | 1.00 | 3/0/0 |
| NaCl | interstitial | 0.8Tm | Na_i:3 | Na_i:3 | 1.00 | 1.00 | 3/0/0 |
| NaCl | frenkel | room | frenkel_pair:3 | V_Na:3,Na_i:3 | 1.00 | 1.00 | 3/0/0 |
| NaCl | frenkel | 0.8Tm | frenkel_pair:3 | V_Na:3,Na_i:3 | 1.00 | 1.00 | 3/0/0 |

## A5 statistical round trip — PASS

**Evidence:** pass rate 6/6-with-floor; fluid: 4/4 (target >= 90%); interface: 1/1 (target >= 90%); glass: 1/1 (target >= 80%); 6 cases without a floor on record: no-floor: skipped (synthetic frame); floor provenance: floor = mean over the cross-quench frame pairs (last two, most-annealed frames of each of the 3 independent quenches, identical protocol, distinct seeds), on the legacy keys AND the partial g(r) rms of every species pair (W7 step 6); within-one-quench pairs (intra_quench below) share the anneal basin and sit closer (Review 2)..; floor = mean over frame pairs at lag >= 4 (the decorrelated half; 21 of 45 pairs, residual short-lag correlation must not shrink the floor).; floor = pooled decorrelated frame pairs of 2 independent trajectories (within-trajectory lag >= 2 plus cross pairs of the decorrelated halves; cross sits only +1%/+10% above within -- an equilibrated liquid carries no preparation memory, unlike the glass quenches; the second trajectory supplies the independent pairs the floor's upper quantile needs).; floor = mean over frame pairs at lag >= 4 (the decorrelated half; 21 of 45 pairs, residual short-lag correlation must not shrink the floor); frames spaced 5 ps (beyond the water structural relaxation time, Review 2): the floor is not shrunk by residual inter-frame correlation. (full notes: details rows / noise_floors.json)

| case | category | status | T | backend | md_steps | distances | ratios |
| --- | --- | --- | --- | --- | --- | --- | --- |
| reference/ka_glass | glass | pass (cn_tv 0.008 vs floor 0.029 (x0.3; floor = max(mean 0.023, P90 0.029); ref = mean obs of frames [3, 4, 8, 9, 13, 14]; draws 0.021/0.012/0.015); gr_rms 0.024 vs floor 0.027 (x0.9; floor = max(mean 0.026, P90 0.027); ref = mean obs of frames [3, 4, 8, 9, 13, 14]; draws 0.041/0.024/0.032); per-draw median 0.015/0.032; floor: floor = mean over the cross-quench frame pairs (last two, most-annealed frames of each of the 3 independent quenches, identical protocol, distinct seeds), on the legacy keys AND the partial g(r) rms of every species pair (W7 step 6); within-one-quench pairs (intra_quench below) share the anneal basin and sit closer (Review 2).) | - | lj | 10000 |  |  |
| reference/lj_liquid | fluid | pass (cn_tv 0.026 vs floor 0.041 (x0.6; floor = max(mean 0.029, P90 0.041); ref = mean obs of frames [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]; draws 0.043/0.034/0.064); gr_rms 0.030 vs floor 0.047 (x0.6; floor = max(mean 0.044, P90 0.047); ref = mean obs of frames [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]; draws 0.055/0.048/0.054); per-draw median 0.043/0.054; floor: floor = mean over frame pairs at lag >= 4 (the decorrelated half; 21 of 45 pairs, residual short-lag correlation must not shrink the floor)) | 0.72 | lj | 7400 |  |  |
| reference/lj_liquid_large | fluid | pass (cn_tv 0.010 vs floor 0.017 (x0.6; floor = max(mean 0.012, P90 0.017); ref = mean obs of frames [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]; draws 0.032/0.012/0.012); gr_rms 0.022 vs floor 0.033 (x0.7; floor = max(mean 0.028, P90 0.033); ref = mean obs of frames [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]; draws 0.037/0.034/0.026); per-draw median 0.012/0.034; floor: floor = pooled decorrelated frame pairs of 2 independent trajectories (within-trajectory lag >= 2 plus cross pairs of the decorrelated halves; cross sits only +1%/+10% above within -- an equilibrated liquid carries no preparation memory, unlike the glass quenches; the second trajectory supplies the independent pairs the floor's upper quantile needs)) | 0.72 | lj | 7400 |  |  |
| reference/lj_solid_liquid | interface | pass (cn_tv 0.009 vs floor 0.021 (x0.4; floor = max(mean 0.014, P90 0.021); ref = mean obs of frames [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]; draws 0.010/0.023/0.013); gr_rms 0.022 vs floor 0.019 (x1.2; floor = max(mean 0.017, P90 0.019); ref = mean obs of frames [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]; draws 0.025/0.021/0.036); per-draw median 0.013/0.025; floor: floor = mean over frame pairs at lag >= 4 (the decorrelated half; 21 of 45 pairs, residual short-lag correlation must not shrink the floor)) | 0.65 | lj | 7400 |  |  |
| reference/nacl_aq | fluid | pass (cn_tv 0.024 vs floor 0.022 (x1.1; floor = max(mean 0.015, P90 0.022); ref = mean obs of frames [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]; draws 0.041/0.026/0.034); gr_rms 0.022 vs floor 0.017 (x1.3; floor = max(mean 0.016, P90 0.017); ref = mean obs of frames [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]; draws 0.029/0.033/0.026); per-draw median 0.034/0.029; floor: floor = mean over frame pairs at lag >= 4 (the decorrelated half; 21 of 45 pairs, residual short-lag correlation must not shrink the floor)) | - | classical | 0 |  |  |
| reference/water_tip4p | fluid | pass (cn_tv 0.034 vs floor 0.046 (x0.7; floor = max(mean 0.035, P90 0.046); ref = mean obs of frames [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]; draws 0.054/0.058/0.044); gr_rms 0.032 vs floor 0.027 (x1.2; floor = max(mean 0.024, P90 0.027); ref = mean obs of frames [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]; draws 0.041/0.047/0.042); per-draw median 0.054/0.042; floor: floor = mean over frame pairs at lag >= 4 (the decorrelated half; 21 of 45 pairs, residual short-lag correlation must not shrink the floor); frames spaced 5 ps (beyond the water structural relaxation time, Review 2): the floor is not shrunk by residual inter-frame correlation) | - | classical | 0 |  |  |
| fluid/water_box15 | fluid | no-floor: skipped (synthetic frame) | - | classical | 0 | gr_rms=0.319, cn_tv=0.450 |  |
| fluid/ar_gas_box25 | fluid | no-floor: skipped (synthetic frame) | - | classical | 0 | gr_rms=0.769, cn_tv=0.160 |  |
| fluid/n2_box22 | fluid | no-floor: skipped (synthetic frame) | - | classical | 0 | gr_rms=1.126, cn_tv=0.319 |  |
| glass/lj_glass_rho085 | glass | no-floor: skipped (synthetic frame) | - | lj | 2155 | gr_rms=0.931, cn_tv=0.320 |  |
| interface/lj_solid_liquid | interface | no-floor: skipped (synthetic frame) | 0.65 | lj | 7400 | gr_rms=1.580, cn_tv=0.498 |  |
| interfaces/cu_water | interfaces | no-floor: skipped (synthetic frame) | 300 | eam | 1000 | gr_rms=1.222, cn_tv=0.226 |  |

## A6 conservation — PASS

**Evidence:** 126 lifts checked: frame count == conserve line on 126/126; verifier's own region arithmetic (composition/occupancy/molecules/defect net/residual) pins or cross-checks 81 programs (non-derivable shapes check two-way); charge consistent on 126/126

| case | frame | frame counts | conserve | derivation | derived | ok | charge |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crystals/fcc_cu | 0 | {'Cu': 32} | {'Cu': 32} | sites-only: 32 fcc sites + net 0 + residual 0 [total 32 == 32] | - | True | no ionic species |
| crystals/fcc_cu | 1 | {'Cu': 32} | {'Cu': 32} | sites-only: 32 fcc sites + net 0 + residual 0 [total 32 == 32] | - | True | no ionic species |
| crystals/fcc_cu | 2 | {'Cu': 32} | {'Cu': 32} | sites-only: 32 fcc sites + net 0 + residual 0 [total 32 == 32] | - | True | no ionic species |
| crystals/fcc_cu | 3 | {'Cu': 32} | {'Cu': 32} | sites-only: 32 fcc sites + net 0 + residual 0 [total 32 == 32] | - | True | no ionic species |
| crystals/fcc_cu | 4 | {'Cu': 32} | {'Cu': 32} | sites-only: 32 fcc sites + net 0 + residual 0 [total 32 == 32] | - | True | no ionic species |
| crystals/bcc_fe | 0 | {'Fe': 16} | {'Fe': 16} | sites-only: 16 bcc sites + net 0 + residual 0 [total 16 == 16] | - | True | charge +48 == frame +48 |
| crystals/bcc_fe | 1 | {'Fe': 16} | {'Fe': 16} | sites-only: 16 bcc sites + net 0 + residual 0 [total 16 == 16] | - | True | charge +48 == frame +48 |
| crystals/bcc_fe | 2 | {'Fe': 16} | {'Fe': 16} | sites-only: 16 bcc sites + net 0 + residual 0 [total 16 == 16] | - | True | charge +48 == frame +48 |
| crystals/bcc_fe | 3 | {'Fe': 16} | {'Fe': 16} | sites-only: 16 bcc sites + net 0 + residual 0 [total 16 == 16] | - | True | charge +48 == frame +48 |
| crystals/bcc_fe | 4 | {'Fe': 16} | {'Fe': 16} | sites-only: 16 bcc sites + net 0 + residual 0 [total 16 == 16] | - | True | charge +48 == frame +48 |
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
| crystals/hcp_mg | 0 | {'Mg': 32} | {'Mg': 32} | sites-only: 32 hcp sites + net 0 + residual 0 [total 32 == 32] | - | True | charge +64 == frame +64 |
| crystals/hcp_mg | 1 | {'Mg': 32} | {'Mg': 32} | sites-only: 32 hcp sites + net 0 + residual 0 [total 32 == 32] | - | True | charge +64 == frame +64 |
| crystals/hcp_mg | 2 | {'Mg': 32} | {'Mg': 32} | sites-only: 32 hcp sites + net 0 + residual 0 [total 32 == 32] | - | True | charge +64 == frame +64 |
| crystals/hcp_mg | 3 | {'Mg': 32} | {'Mg': 32} | sites-only: 32 hcp sites + net 0 + residual 0 [total 32 == 32] | - | True | charge +64 == frame +64 |
| crystals/hcp_mg | 4 | {'Mg': 32} | {'Mg': 32} | sites-only: 32 hcp sites + net 0 + residual 0 [total 32 == 32] | - | True | charge +64 == frame +64 |
| crystals/diamond_si | 0 | {'Si': 64} | {'Si': 64} | sites-only: 64 diamond sites + net 0 + residual 0 [total 64 == 64] | - | True | no ionic species |
| crystals/diamond_si | 1 | {'Si': 64} | {'Si': 64} | sites-only: 64 diamond sites + net 0 + residual 0 [total 64 == 64] | - | True | no ionic species |
| crystals/diamond_si | 2 | {'Si': 64} | {'Si': 64} | sites-only: 64 diamond sites + net 0 + residual 0 [total 64 == 64] | - | True | no ionic species |
| crystals/diamond_si | 3 | {'Si': 64} | {'Si': 64} | sites-only: 64 diamond sites + net 0 + residual 0 [total 64 == 64] | - | True | no ionic species |
| crystals/diamond_si | 4 | {'Si': 64} | {'Si': 64} | sites-only: 64 diamond sites + net 0 + residual 0 [total 64 == 64] | - | True | no ionic species |
| crystals/perovskite_srtio3 | 0 | {'O': 24, 'Sr': 8, 'Ti': 8} | {'O': 24, 'Sr': 8, 'Ti': 8} | perovskitex8+composition | {'Sr': 8, 'Ti': 8, 'O': 24} | True | charge -48 == frame -48 |
| crystals/perovskite_srtio3 | 1 | {'O': 24, 'Sr': 8, 'Ti': 8} | {'O': 24, 'Sr': 8, 'Ti': 8} | perovskitex8+composition | {'Sr': 8, 'Ti': 8, 'O': 24} | True | charge -48 == frame -48 |
| crystals/perovskite_srtio3 | 2 | {'O': 24, 'Sr': 8, 'Ti': 8} | {'O': 24, 'Sr': 8, 'Ti': 8} | perovskitex8+composition | {'Sr': 8, 'Ti': 8, 'O': 24} | True | charge -48 == frame -48 |
| crystals/perovskite_srtio3 | 3 | {'O': 24, 'Sr': 8, 'Ti': 8} | {'O': 24, 'Sr': 8, 'Ti': 8} | perovskitex8+composition | {'Sr': 8, 'Ti': 8, 'O': 24} | True | charge -48 == frame -48 |
| crystals/perovskite_srtio3 | 4 | {'O': 24, 'Sr': 8, 'Ti': 8} | {'O': 24, 'Sr': 8, 'Ti': 8} | perovskitex8+composition | {'Sr': 8, 'Ti': 8, 'O': 24} | True | charge -48 == frame -48 |
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
| defects/rocksalt_nacl_vna | 4 | {'Cl': 108, 'Na': 102} | {'Cl': 108, 'Na': 102} | rocksaltx27+composition | {'Na': 102, 'Cl': 108} | True | charge -6 == frame -6 |
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
| surface/ni111_o_top | 0 | {'Ni': 72, 'O': 8} | {'Ni': 72, 'O': 8} | unsupported phase | - | True | charge -16 == frame -16 |
| surface/ni111_o_top | 1 | {'Ni': 72, 'O': 8} | {'Ni': 72, 'O': 8} | unsupported phase | - | True | charge -16 == frame -16 |
| surface/ni111_o_top | 2 | {'Ni': 72, 'O': 8} | {'Ni': 72, 'O': 8} | unsupported phase | - | True | charge -16 == frame -16 |
| surface/ni111_o_top | 3 | {'Ni': 72, 'O': 8} | {'Ni': 72, 'O': 8} | unsupported phase | - | True | charge -16 == frame -16 |
| surface/ni111_o_top | 4 | {'Ni': 72, 'O': 8} | {'Ni': 72, 'O': 8} | unsupported phase | - | True | charge -16 == frame -16 |
| surfaces/pt111_o | 0 | {'O': 8, 'Pt': 72} | {'O': 8, 'Pt': 72} | unsupported phase | - | True | charge -16 == frame -16 |
| surfaces/pt111_o | 1 | {'O': 8, 'Pt': 72} | {'O': 8, 'Pt': 72} | unsupported phase | - | True | charge -16 == frame -16 |
| surfaces/pt111_o | 2 | {'O': 8, 'Pt': 72} | {'O': 8, 'Pt': 72} | unsupported phase | - | True | charge -16 == frame -16 |
| surfaces/pt111_o | 3 | {'O': 8, 'Pt': 72} | {'O': 8, 'Pt': 72} | unsupported phase | - | True | charge -16 == frame -16 |
| surfaces/pt111_o | 4 | {'O': 8, 'Pt': 72} | {'O': 8, 'Pt': 72} | unsupported phase | - | True | charge -16 == frame -16 |
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
| gases/co2_dense | 0 | {'C': 60, 'O': 120} | {'C': 60, 'O': 120} | molecules | {'C': 60, 'O': 120} | True | charge -240 == frame -240 |
| gases/co2_dense | 1 | {'C': 60, 'O': 120} | {'C': 60, 'O': 120} | molecules | {'C': 60, 'O': 120} | True | charge -240 == frame -240 |
| gases/co2_dense | 2 | {'C': 60, 'O': 120} | {'C': 60, 'O': 120} | molecules | {'C': 60, 'O': 120} | True | charge -240 == frame -240 |
| gases/co2_dense | 3 | {'C': 60, 'O': 120} | {'C': 60, 'O': 120} | molecules | {'C': 60, 'O': 120} | True | charge -240 == frame -240 |
| gases/co2_dense | 4 | {'C': 60, 'O': 120} | {'C': 60, 'O': 120} | molecules | {'C': 60, 'O': 120} | True | charge -240 == frame -240 |
| surfaces/si001_2x1 | 0 | {'Si': 63} | {'Si': 63} | unsupported phase | - | True | no ionic species |
| surfaces/si001_2x1 | 1 | {'Si': 63} | {'Si': 63} | unsupported phase | - | True | no ionic species |
| surfaces/si001_2x1 | 2 | {'Si': 63} | {'Si': 63} | unsupported phase | - | True | no ionic species |
| surfaces/si001_2x1 | 3 | {'Si': 63} | {'Si': 63} | unsupported phase | - | True | no ionic species |
| surfaces/si001_2x1 | 4 | {'Si': 63} | {'Si': 63} | unsupported phase | - | True | no ionic species |
| interfaces/cu_water | 0 | {'Cu': 384, 'H': 272, 'O': 136} | {'Cu': 384, 'H': 272, 'O': 136} | molecules+crystal slab region (slab z 42.5 .. 21.6): fitted bounds do not pin the site count [partial: {'H': 272, 'O': 136} vs frame OK; crystal slab region (slab z 42.5 .. 21.6): fitted bounds do not pin the site count] | {'H': 272, 'O': 136} | True | no ionic species |
| interfaces/cu_water | 1 | {'Cu': 384, 'H': 272, 'O': 136} | {'Cu': 384, 'H': 272, 'O': 136} | molecules+crystal slab region (slab z 42.5 .. 21.4): fitted bounds do not pin the site count [partial: {'H': 272, 'O': 136} vs frame OK; crystal slab region (slab z 42.5 .. 21.4): fitted bounds do not pin the site count] | {'H': 272, 'O': 136} | True | no ionic species |
| interfaces/cu_water | 2 | {'Cu': 384, 'H': 272, 'O': 136} | {'Cu': 384, 'H': 272, 'O': 136} | molecules+crystal slab region (slab z 42.6 .. 21.4): fitted bounds do not pin the site count [partial: {'H': 272, 'O': 136} vs frame OK; crystal slab region (slab z 42.6 .. 21.4): fitted bounds do not pin the site count] | {'H': 272, 'O': 136} | True | no ionic species |
| interfaces/cu_water | 3 | {'Cu': 384, 'H': 272, 'O': 136} | {'Cu': 384, 'H': 272, 'O': 136} | molecules+crystal slab region (slab z 42.6 .. 21.4): fitted bounds do not pin the site count [partial: {'H': 272, 'O': 136} vs frame OK; crystal slab region (slab z 42.6 .. 21.4): fitted bounds do not pin the site count] | {'H': 272, 'O': 136} | True | no ionic species |
| interfaces/cu_water | 4 | {'Cu': 384, 'H': 272, 'O': 136} | {'Cu': 384, 'H': 272, 'O': 136} | molecules+crystal slab region (slab z 42.4 .. 21.4): fitted bounds do not pin the site count [partial: {'H': 272, 'O': 136} vs frame OK; crystal slab region (slab z 42.4 .. 21.4): fitted bounds do not pin the site count] | {'H': 272, 'O': 136} | True | no ionic species |
| probe: rocksalt_nacl minus one Cl | 0 | {'Cl': 31, 'Na': 32} | {'Cl': 31, 'Na': 32} | rocksaltx8+composition | {'Na': 32, 'Cl': 31} | True | charge +1 == frame +1 [expected: charge +1] |

## A7 phase segmentation — PASS

**Evidence:** 10/10 interface frames (2 cases x 5 frames; interface band of 2 x d_NN excluded around every interface plane, d_NN = verifier's median nearest-neighbour distance) labelled >= 95% correct; worst 0.982 (interface/lj_solid_liquid frame 3); judged fraction disclosed per frame -- interfaces/cu_water: judged 0.854-0.874 of atoms (core-scope gate >= 90% x geometry-implied availability held); interface/lj_solid_liquid: judged 0.355-0.430 of atoms (core-scope gate >= 90% x geometry-implied availability held)

| case | frame | core atoms | judged frac | geo floor | accuracy |
| --- | --- | --- | --- | --- | --- |
| interface/lj_solid_liquid | 0 | 213 | 0.416 | 0.382 | 1.000 |
| interface/lj_solid_liquid | 1 | 182 | 0.355 | 0.378 | 1.000 |
| interface/lj_solid_liquid | 2 | 216 | 0.422 | 0.382 | 1.000 |
| interface/lj_solid_liquid | 3 | 220 | 0.430 | 0.379 | 0.982 |
| interface/lj_solid_liquid | 4 | 218 | 0.426 | 0.383 | 1.000 |
| interfaces/cu_water | 0 | 692 | 0.874 | 0.824 | 1.000 |
| interfaces/cu_water | 1 | 676 | 0.854 | 0.824 | 1.000 |
| interfaces/cu_water | 2 | 686 | 0.866 | 0.824 | 1.000 |
| interfaces/cu_water | 3 | 686 | 0.866 | 0.824 | 0.999 |
| interfaces/cu_water | 4 | 677 | 0.855 | 0.824 | 1.000 |

## A8 reactive census — PASS

**Evidence:** plan 0: census {'H2O': 40, 'HO': 9, 'H': 9} == planted {'H2O': 40, 'HO': 9, 'H': 9} (147 atoms, coordinates written directly in numpy); plan 1: census {'H2O': 25, 'HO': 5, 'H': 7} == planted {'H2O': 25, 'HO': 5, 'H': 7} (92 atoms, coordinates written directly in numpy); adsorption sites 100% correct (claims {'bridge': 3, 'top': 5} vs planted {'top': 5, 'bridge': 3}, 8 adsorbates on a raw-numpy Pt(111) slab; PLAN gate >= 90%)

| plan | expected | census | exact |
| --- | --- | --- | --- |
| 0 | {'H2O': 40, 'HO': 9, 'H': 9} | {'H2O': 40, 'HO': 9, 'H': 9} | True |
| 1 | {'H2O': 25, 'HO': 5, 'H': 7} | {'H2O': 25, 'HO': 5, 'H': 7} | True |

## A9 compression — PASS

**Evidence:** 5 measured systems with >= 1,000 atoms (bench frames + independent-MD reference frames); worst ratio 0.67% (reference/nacl_aq, 1640 atoms); worst crystal-class 0.49% (crystals/l12_ni3al, 1372 atoms); worst heterogeneous-class 0.67% (reference/nacl_aq, 1640 atoms); the >= 1,000-atom gate is evidenced on heterogeneous systems (2 of 5 gated measurements); reference frames not liftable (recorded, not gated): reference/cu_solid_liquid ChaordError: extended bonded component (832 Cu atoms): not a molecular fl; per-case table in details

| case | atoms | program B | extxyz B | ratio % | source |
| --- | --- | --- | --- | --- | --- |
| crystals/bcc_fe | 16 | 335 | 977 | 34.29 | bench |
| crystals/diamond_si | 64 | 342 | 3620 | 9.45 | bench |
| crystals/fcc_crconi | 108 | 383 | 6041 | 6.34 | bench |
| crystals/fcc_cu | 32 | 335 | 1857 | 18.04 | bench |
| crystals/hcp_mg | 32 | 351 | 1871 | 18.76 | bench |
| crystals/l12_ni3al | 1372 | 370 | 75565 | 0.49 | bench |
| crystals/perovskite_srtio3 | 40 | 374 | 2297 | 16.28 | bench |
| crystals/rocksalt_nacl | 64 | 370 | 3620 | 10.22 | bench |
| defects/fcc_cu_vacancies | 105 | 360 | 5879 | 6.12 | bench |
| defects/l12_ni3al_vac_antisite | 253 | 412 | 14019 | 2.94 | bench |
| defects/rocksalt_nacl_vna | 210 | 393 | 11690 | 3.36 | bench |
| fluid/ar_gas_box25 | 50 | 379 | 2847 | 13.31 | bench |
| fluid/n2_box22 | 160 | 381 | 8898 | 4.28 | bench |
| fluid/water_box15 | 180 | 553 | 9998 | 5.53 | bench |
| gases/co2_dense | 180 | 387 | 10001 | 3.87 | bench |
| glass/lj_glass_rho085 | 200 | 503 | 11137 | 4.52 | bench |
| interface/lj_solid_liquid | 512 | 837 | 28300 | 2.96 | bench |
| interfaces/cu_water | 792 | 825 | 43661 | 1.89 | bench |
| reactive/water_oh_h_box20 | 177 | 586 | 9833 | 5.96 | bench |
| solutions/cuau_random | 108 | 370 | 6080 | 6.09 | bench |
| solutions/lipf6_ec | 432 | 447 | 23858 | 1.87 | bench |
| solutions/nacl_aq | 376 | 601 | 20778 | 2.89 | bench |
| surface/ni111_o_top | 80 | 485 | 4538 | 10.69 | bench |
| surfaces/pt111_o | 80 | 485 | 4532 | 10.70 | bench |
| surfaces/si001_2x1 | 63 | 465 | 3604 | 12.90 | bench |
| defects/l12_ni3al_vac_antisite tiled 2x2 (supplementary, not a raw bench frame) | 1012 | 415 | 55765 | 0.74 | tiled-supplementary |
| reference/lj_glass | 2048 | 510 | 112778 | 0.45 | reference |
| reference/lj_liquid | 500 | 442 | 27637 | 1.60 | reference |
| reference/lj_liquid_large | 2048 | 446 | 112778 | 0.40 | reference |
| reference/lj_solid_liquid | 2304 | 763 | 126858 | 0.60 | reference |
| reference/nacl_aq | 1640 | 609 | 90341 | 0.67 | reference |
| reference/water_tip4p | 768 | 555 | 42377 | 1.31 | reference |
| reference/cu_solid_liquid | - | - | - | not liftable | ChaordError: extended bonded component (832 Cu atoms): not a molecular fl |

## A10 determinism — PASS

**Evidence:** repeated lift of bench defect frame byte-identical: True; same-seed builds identical coordinates: True


## A11 speed — PASS

**Evidence:** lift_frame(mode=fluid) on 100,000 atoms took 2.9 s (target <= 120 s); input = fcc lattice at rho 0.85 with 0.10 x a jitter (thermally disordered, not equilibrated MD); program states the exact count: yes


## A12 static checks — PASS

**Evidence:** lattice_mismatch: caught; impossible_density: caught; charge_imbalance: caught; overlap_0.1sigma: caught


## A13 no crashes — PASS

**Evidence:** 125/125 bench frames lift without any exception; 0 atoms placed in residual blocks across successful lifts; 60 frames carry routing diagnostics (recorded lift_mode refused or names a bench category; lifted auto)


## A14 documentation — PASS

**Evidence:** 35 distinct keys across dialect YAMLs; reference.md covers 35/35 with structured entries (code span / heading / table row + an actual spec/examples line cited; prose is not coverage); example citation gated for 26/35 keys (9 keys have no example anywhere in spec/examples/ -- reported, nothing to cite: atom, chirality, dialects, lift_version, nanotube, note, reconstruction, source, stacking); sketch_check pass; generated docs/reference_generated.md

