"""A/B analysis of Ghosh et al. 2023 NAR 51, 4101 (DOI 10.1093/nar/gkad020), SI Table S8:
RNA duplex GAUUACGCCUG/complement (11 bp, non-self-complementary), UV melting, 37 C,
buffer NaCl + 10 mM Na2HPO4 (pH 7.0) + 1 mM Na2EDTA; total Na+ 0.122 M (authors) and 1.022 M.
dG37 (kcal/mol) +/- err as printed.
"""
import numpy as np
RT = 1.98720425864083e-3 * 310.15
dlnc = np.log(1.022 / 0.122)
lo = {0: (-14.1, 0.1), 10: (-13.9, 0.2), 20: (-13.4, 0.1), 30: (-13.0, 0.2), 40: (-12.0, 0.2), 50: (-11.0, 0.1)}
hi = {0: (-16.5, 0.1), 10: (-16.1, 0.1), 20: (-15.4, 0.2), 40: (-13.4, 0.7)}
aw_lo = {0: 0.996, 10: 0.985, 20: 0.971, 30: 0.956, 40: 0.947, 50: 0.931}
aw_hi = {0: 0.967, 10: 0.953, 20: 0.939, 40: 0.911}
print(f'RT = {RT:.4f} kcal/mol, ln(c_hi/c_lo) = {dlnc:.3f}')
S0 = (lo[0][0] - hi[0][0]) / RT / dlnc
print(f'S_sec(0) = {S0:.2f} +/- {np.hypot(lo[0][1], hi[0][1])/RT/dlnc:.2f}')
print(' wt%  dlnK(0.122)     dlnK(1.022)     hi-lo           S_sec(x)       phi_A=-dlnK/S0 (lo, hi)   dlnK/dln aw (lo, hi)')
for w in sorted(hi):
    dl = -(lo[w][0] - lo[0][0]) / RT; edl = np.hypot(lo[w][1], lo[0][1]) / RT
    dh = -(hi[w][0] - hi[0][0]) / RT; edh = np.hypot(hi[w][1], hi[0][1]) / RT
    S = (lo[w][0] - hi[w][0]) / RT / dlnc; eS = np.hypot(lo[w][1], hi[w][1]) / RT / dlnc
    na = (dl / np.log(aw_lo[w] / aw_lo[0]), dh / np.log(aw_hi[w] / aw_hi[0])) if w else (np.nan, np.nan)
    print(f' {w:3d}  {dl:+.2f}+/-{edl:.2f}   {dh:+.2f}+/-{edh:.2f}   {dh-dl:+.2f}+/-{np.hypot(edl,edh):.2f}   '
          f'{S:.2f}+/-{eS:.2f}    {-dl/S0:+.2f}, {-dh/S0:+.2f}            {na[0]:.0f}, {na[1]:.0f}')
# EG: 100 mM has 10 and 15 wt%, 1 M has 20 wt%; linear extrapolation of the 100 mM slope (flagged)
s_EG = np.polyfit([0, 10, 15], [-14.1, -13.6, -13.4], 1)[0]
print(f'EG: ddG/wt% at 0.122 M = {s_EG:.4f} kcal/mol/wt% -> extrapolated 20 wt%: {20*s_EG:+.2f}; measured at 1.022 M, 20 wt%: {-15.0+16.5:+.2f} kcal/mol')
