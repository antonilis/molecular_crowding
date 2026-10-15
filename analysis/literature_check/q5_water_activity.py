"""Q5: water activity of the crowder solutions and 'apparent hydration' implied by our K data.

Sources of a_w (all at/near 25 C):
 (A) ideal-solution estimate, ln a_w = -M_w * m (osmotic coefficient phi = 1), binary water+cosolute;
 (B) Nakano et al. JACS 2004 (10.1021/ja0463029) SI Table S2, VPO / freezing-point osmometry,
     in 1 M NaCl (+10 mM Na2HPO4, 1 mM EDTA) -> we use d ln a_w = ln a_w(cosolute) - ln a_w(no cosolute);
     PEG 200 also in 100 mM NaCl;
 (C) PEG osmotic EOS of Cohen, Podgornik, Hansen, Parsegian JPCB 2009 (10.1021/jp806893a):
     Pi = RT C/(Mm N) + alpha RT (C v)^(9/4) / (Mm v), alpha = 0.49, v = 0.825 mL/g, Mm = 44 g/mol,
     N = (Mp-18)/44; ln a_w = -Pi V_w/(RT). Fitted to binary PEG/water data; questionable for N < ~10.
Our K: source_data/MacromoleculeEquilibria_Crowding/OriginalData_*.xlsx (25 C, 20 mM PB).
"""
import numpy as np, pandas as pd, os
R, T = 8.314462618, 298.15
RT = R * T
Vw = 18.07e-6     # m3/mol
Mw = 18.015e-3    # kg/mol
root = os.path.dirname(os.path.abspath(__file__))
src = os.path.join(root, '..', '..', 'source_data', 'MacromoleculeEquilibria_Crowding')

def rho(w):  # same density formula as reacting_probes_equillibrium_constants.py (g/cm3)
    return 0.99707 + 0.0017441 * 100 * w

def lnaw_ideal(w, M):
    m = w / (M * 1e-3 * (1 - w))   # mol/kg water
    return -Mw * m

def lnaw_cohen(w, Mp):
    N = (Mp - 18) / 44.0
    C = w * rho(w) * 1e3          # kg/m3
    v = 0.825e-3                  # m3/kg
    Mm = 0.044
    Pi = RT * C / (Mm * N) + 0.49 * RT * (C * v) ** 2.25 / (Mm * v)
    return -Pi * Vw / RT, Pi

# Nakano 2004 Table S2 (ln a_w x 1e-2); baseline 1 M NaCl: -3.37 ; 100 mM NaCl: -0.37
nak_1M = {('EGly', 20): -9.68, ('EGly', 30): -14.4, ('EGly', 40): -17.5,
          ('PEG200', 20): -6.34, ('PEG200', 30): -8.03, ('PEG200', 40): -9.29,
          ('PEG1000', 20): -5.31, ('PEG1000', 40): -6.66, ('PEG6000', 20): -5.06}
nak_100mM = {('PEG200', 10): -1.57, ('PEG200', 20): -3.18, ('PEG200', 30): -4.58, ('PEG200', 40): -5.47}

MW = {'EGly': 62.07, 'PEG200': 200, 'PEG400': 400, 'PEG600': 600, 'PEG1000': 1000, 'PEG1500': 1500,
      'PEG3000': 3000, 'PEG6000': 6000, 'PEG12000': 12000, 'PEG20000': 20000, 'PEG35000': 35000}

def K_of(crowder, wt):
    d = pd.read_excel(os.path.join(src, f'OriginalData_{crowder}.xlsx'))
    K0 = d.loc[d['crowder wt. [%]'] == 0, 'K [M]'].iloc[0]
    row = d.loc[d['crowder wt. [%]'] == wt]
    return (np.nan, np.nan) if row.empty else (row['K [M]'].iloc[0], K0)

print('wt%=40 unless stated. dlnK = ln(K/K0) from our data; n_app = dlnK/dln a_w (waters taken up per duplex);')
print('ddG_pred(15/bp) = predicted destabilization if 13 bp x 15 waters (Nakano 2004, PEG200) acted alone.\n')
print(f"{'crowder':9s} {'w':>4s} {'lnaw_ideal':>10s} {'lnaw_Cohen':>10s} {'Pi_Cohen/MPa':>12s} {'Nakano_1M':>9s} {'Nak_100mM':>9s} "
      f"{'dlnK':>6s} {'ddG_obs':>7s} {'n_app(range)':>14s} {'n/bp':>10s} {'ddG_pred15':>10s}")
for cr, M in MW.items():
    for wt in ([20, 40] if cr in ('EGly', 'PEG200', 'PEG400', 'PEG6000', 'PEG35000') else [40]):
        w = wt / 100
        li = lnaw_ideal(w, M)
        lc, Pi = lnaw_cohen(w, M) if M >= 400 else (np.nan, np.nan)
        n1 = (nak_1M.get((cr, wt), np.nan) + 3.37) * 1e-2
        n2 = (nak_100mM.get((cr, wt), np.nan) + 0.37) * 1e-2
        K, K0 = K_of(cr, wt)
        dlnK = np.log(K / K0)
        cands = [x for x in (li, lc, n1, n2) if np.isfinite(x)]
        # for EG and PEG<=400 the ideal value is a reasonable bracket; for big PEG ideal is far too small
        if M >= 1000:
            cands = [x for x in (lc, n1) if np.isfinite(x)]
        lo, hi = min(cands), max(cands)
        n_app = sorted([dlnK / lo, dlnK / hi])
        pred = -13 * 15 * np.array([lo, hi]) * RT / 1e3
        print(f"{cr:9s} {wt:4d} {li:10.4f} {lc:10.4f} {Pi/1e6 if np.isfinite(Pi) else np.nan:12.2f} {n1:9.4f} {n2:9.4f} "
              f"{dlnK:6.2f} {-RT*dlnK/1e3:7.1f} {n_app[0]:6.0f}-{n_app[1]:<6.0f} {n_app[0]/13:4.1f}-{n_app[1]/13:<4.1f} "
              f"{pred.min():4.1f}-{pred.max():<4.1f}")
