"""A/B (horizontal vs vertical) analysis of Nakano et al. 2004 JACS SI Table S1/S2 (25 C) and
Ghosh et al. 2020 PNAS SI Table S5 (37 C, same raw data re-evaluated): d(ATGCGCAT)2, 8-bp
self-complementary DNA duplex, UV melting, buffer NaCl + 10 mM Na2HPO4 (pH 7.0) + 1 mM Na2EDTA,
PEG 200 0-50 wt% at 100 mM and 1 M NaCl.
Total Na+ (authors' convention in Ghosh 2023: 122 mM for 100 mM NaCl) -> 0.122 and 1.022 M.

Hypotheses (x = cosolute):
 A: ln K(c,x) = F(ln c - phi(x))  ->  Delta lnK(c,x) ~= -phi(x) * S_K(c)
 B: ln K(c,x) = F(ln c) + H(x)    ->  Delta lnK independent of c ; S_K independent of x
Only two salt points -> F known only by its secant slope S_sec between 0.122 and 1.022 M.
"""
import numpy as np

R = 1.98720425864083e-3  # kcal/mol/K
c_lo, c_hi = 0.122, 1.022
dlnc = np.log(c_hi / c_lo)

# Nakano 2004 SI Table S1, dG25 (kcal/mol) +/- err ; Table S2 ln aw (x1e-2)
wt = np.array([0, 10, 20, 30, 40, 50])
G25_hi = np.array([-11.2, -9.8, -8.7, -7.6, -6.8, -5.9]); e25_hi = np.array([0.4, 0.2, 0.3, 0.2, 0.3, 0.4])
G25_lo = np.array([-10.2, -9.6, -8.8, -7.8, -7.2, -5.3]); e25_lo = np.array([0.6, 0.2, 0.3, 0.3, 0.5, 0.4])
lnaw_hi = np.array([-3.37, -4.85, -6.34, -8.03, -9.29, -10.8]) * 1e-2
lnaw_lo = np.array([-0.37, -1.57, -3.18, -4.58, -5.47, -7.46]) * 1e-2
# Ghosh 2020 SI Table S5, dG37 (kcal/mol); PEG 200 50 wt% at 100 mM not listed there
wt37 = np.array([0, 10, 20, 30, 40])
G37_hi = np.array([-9.2, -8.1, -7.0, -5.9, -5.0]); e37_hi = np.array([0.4, 0.2, 0.3, 0.2, 0.3])
G37_lo = np.array([-8.9, -7.8, -6.8, -6.0, -5.3]); e37_lo = np.array([0.6, 0.2, 0.3, 0.3, 0.5])
aw37_hi = np.array([0.967, 0.953, 0.939, 0.923, 0.911]); aw37_lo = np.array([0.996, 0.985, 0.965, 0.956, 0.947])


def analyse(label, w, Ghi, ehi, Glo, elo, T, lnaw_hi=None, lnaw_lo=None):
    RT = R * T
    lnKhi, lnKlo = -Ghi / RT, -Glo / RT
    s_hi, s_lo = ehi / RT, elo / RT
    Ssec = (lnKhi - lnKlo) / dlnc
    eS = np.hypot(s_hi, s_lo) / dlnc
    dK_hi, dK_lo = lnKhi - lnKhi[0], lnKlo - lnKlo[0]
    edK_hi = np.hypot(s_hi, s_hi[0]); edK_lo = np.hypot(s_lo, s_lo[0])
    print(f'\n=== {label}  (T = {T} K, RT = {RT:.4f} kcal/mol)')
    print(' wt%  dlnK(0.122M)    dlnK(1.022M)    diff(hi-lo)      S_sec(x)        '
          'phi_A(lo)  phi_A(hi)  [phi = -dlnK/S_sec(0)]')
    for i in range(len(w)):
        d = dK_hi[i] - dK_lo[i]; ed = np.hypot(edK_hi[i], edK_lo[i]) if i else 0
        phlo = -dK_lo[i] / Ssec[0]; phhi = -dK_hi[i] / Ssec[0]
        print(f' {w[i]:3d}  {dK_lo[i]:+6.2f}+/-{edK_lo[i]:.2f}  {dK_hi[i]:+6.2f}+/-{edK_hi[i]:.2f}  '
              f'{d:+6.2f}+/-{ed:.2f}   {Ssec[i]:+5.2f}+/-{eS[i]:.2f}   {phlo:+6.2f}    {phhi:+6.2f}')
    # weighted linear fit S_sec vs wt% (mixed derivative)
    wts = 1 / eS ** 2
    A = np.vstack([np.ones_like(w, dtype=float), w]).T
    cov = np.linalg.inv(A.T @ (A * wts[:, None]))
    p = cov @ (A.T @ (wts * Ssec))
    print(f' d S_sec / d wt% = {p[1]:+.4f} +/- {np.sqrt(cov[1,1]):.4f}  (B predicts 0)')
    if lnaw_hi is not None:
        daw_hi, daw_lo = lnaw_hi - lnaw_hi[0], lnaw_lo - lnaw_lo[0]
        print(' per water activity: dlnK/dln aw  (= -Delta n_w^app)  at 0.122 M | 1.022 M; and dln aw hi/lo')
        for i in range(1, len(w)):
            print(f'  {w[i]:3d}%: {dK_lo[i]/daw_lo[i]:6.1f} | {dK_hi[i]/daw_hi[i]:6.1f}   '
                  f'dln aw: {daw_lo[i]:+.4f} | {daw_hi[i]:+.4f}  ratio hi/lo {daw_hi[i]/daw_lo[i]:.2f}; '
                  f'ratio dlnK hi/lo {dK_hi[i]/dK_lo[i]:.2f}')
    return Ssec


print(f'ln(c_hi/c_lo) = {dlnc:.3f}')
S25 = analyse('Nakano 2004 SI Tab. S1, 25 C, PEG 200', wt, G25_hi, e25_hi, G25_lo, e25_lo, 298.15,
              lnaw_hi, lnaw_lo)
S37 = analyse('Ghosh 2020 SI Tab. S5, 37 C, PEG 200', wt37, G37_hi, e37_hi, G37_lo, e37_lo, 310.15,
              np.log(aw37_hi), np.log(aw37_lo))

# If A with our 13-bp curvature were imposed: phi needed to turn S from S0 to S(x),
# phi = -(S(x)-S0)/(dS/dln c). Use dS/dln c of the 8-mer unknown; report for a range.
print('\nIf the S_sec change were entirely an A-type shift: phi = -(S(x)-S(0))/(dS/dlnc)')
for dSdl in (-0.5, -1.0, -1.7):
    print(f'  dS/dln c = {dSdl}: phi(40 wt%, 25 C) = {-(S25[4]-S25[0])/dSdl:+.2f}; '
          f'phi(40 wt%, 37 C) = {-(S37[4]-S37[0])/dSdl:+.2f}')
print('  Vertical part left after removing such a shift (25 C, 40 wt%, at 0.122 M):',
      'H = dlnK - (-phi*S(0))')
