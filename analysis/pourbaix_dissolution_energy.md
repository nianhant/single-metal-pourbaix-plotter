# Pourbaix Dissolution Energy

Definition: Delta G_pbx(U, pH) = G_aq,most stable(U, pH) - G_solid,ref(U, pH).
The solid reference is the most stable solid species for that metal at the same point.

Condition: pH 12.00, 2.00 V vs RHE (1.291 V vs SHE), activity 1e-04 M.

| Case | Metal | NH3 (M) | Gly (M) | CN (M) | Best aq species | G_aq best (eV/metal) | Best solid ref | G_solid ref (eV/metal) | Delta G_pbx (eV/metal) |
|---|---|---:|---:|---:|---|---:|---|---:|---:|
| nh3_gly_low | Au | 0.02 | 0.005 | 0 | H2AuO3[1-] | -2.911 | Au4O6 | -2.985 | 0.074 |
| nh3_gly_low | Pd | 0.02 | 0.005 | 0 | Pd(NH3)4[2+] | -2.527 | Pd2O4 | -4.636 | 2.109 |
| nh3_gly_low | Ni | 0.02 | 0.005 | 0 | Ni(OH)4[2-] | -4.833 | NiO2 | -4.516 | -0.317 |
| nh3_gly_low | Cu | 0.02 | 0.005 | 0 | CuHO2[1-] | -2.890 | Cu16O24 | -3.923 | 1.033 |
| nh3_gly_low | Ti | 0.02 | 0.005 | 0 | TiHO3[1-] | -12.058 | Ti4O8 | -12.958 | 0.900 |
| nh3_gly_cn_low | Au | 0.02 | 0.005 | 0.0001 | H2AuO3[1-] | -2.911 | Au4O6 | -2.985 | 0.074 |
| nh3_gly_cn_low | Pd | 0.02 | 0.005 | 0.0001 | Pd(CN)4[2+] | -2.551 | Pd2O4 | -4.636 | 2.085 |
| nh3_gly_cn_low | Ni | 0.02 | 0.005 | 0.0001 | Ni(OH)4[2-] | -4.833 | NiO2 | -4.516 | -0.317 |
| nh3_gly_cn_low | Cu | 0.02 | 0.005 | 0.0001 | CuHO2[1-] | -2.890 | Cu16O24 | -3.923 | 1.033 |
| nh3_gly_cn_low | Ti | 0.02 | 0.005 | 0.0001 | TiHO3[1-] | -12.058 | Ti4O8 | -12.958 | 0.900 |
| nh3_gly_low | AuPd | 0.02 | 0.005 | 0 | weighted(Au:H2AuO3[1-]; Pd:Pd(NH3)4[2+]) | -2.719 | weighted(Au:Au4O6; Pd:Pd2O4) | -3.810 | 1.091 |
| nh3_gly_low | TiCu | 0.02 | 0.005 | 0 | weighted(Ti:TiHO3[1-]; Cu:CuHO2[1-]) | -7.474 | weighted(Ti:Ti4O8; Cu:Cu16O24) | -8.441 | 0.966 |
| nh3_gly_low | TiNi | 0.02 | 0.005 | 0 | weighted(Ti:TiHO3[1-]; Ni:Ni(OH)4[2-]) | -8.445 | weighted(Ti:Ti4O8; Ni:NiO2) | -8.737 | 0.292 |
| nh3_gly_cn_low | AuPd | 0.02 | 0.005 | 0.0001 | weighted(Au:H2AuO3[1-]; Pd:Pd(CN)4[2+]) | -2.731 | weighted(Au:Au4O6; Pd:Pd2O4) | -3.810 | 1.079 |
| nh3_gly_cn_low | TiCu | 0.02 | 0.005 | 0.0001 | weighted(Ti:TiHO3[1-]; Cu:CuHO2[1-]) | -7.474 | weighted(Ti:Ti4O8; Cu:Cu16O24) | -8.441 | 0.966 |
| nh3_gly_cn_low | TiNi | 0.02 | 0.005 | 0.0001 | weighted(Ti:TiHO3[1-]; Ni:Ni(OH)4[2-]) | -8.445 | weighted(Ti:Ti4O8; Ni:NiO2) | -8.737 | 0.292 |
