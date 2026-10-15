"""Q9: nearest-neighbour (NN) model in crowding (Ghosh, ..., Sugimoto, PNAS 117, 14194 (2020)) applied to our 13-mer,
plus the water balance that reconciles our DOSY 'Na+ binding' with the VPO result (net Na+ exclusion).

(1) Ghosh 2020, Table 3: dG_NN = dG_NN,[cation] + m_cs * da_w, da_w = a_w(no cosolute) - a_w(cosolute).
    m_cs per NN pair for class 'PEG / 1,2-DME' and class 'EG / glycerol'. No fitted parameter for our data.
    a_w: EG, PEG 200 ideal (phi = 1); PEG >= 400 osmotic EOS of Cohen et al. JPCB 2009 (as in q5_water_activity.py);
    Nakano 2004 VPO values (in 1 M NaCl) where available, as a second estimate.
    Caveats: m_cs at 37 C and 100 mM NaCl; we are at 25 C and ~35 mM Na+; a_w of long PEG uncertain ~3x.
(2) SantaLucia 1998 unified NN + SantaLucia salt correction: K0 of our duplex at 35 mM Na+, 25 C (sanity check).
(3) Water balance: net cation term mu_+ = RT (n_w/55.51 - kappa) (dilute-salt limit), n_w = water per EO
    inaccessible to Na+, kappa = DOSY binding per EO. VPO (Knowles 2015) mu_+ = +4.66 cal -> required n_w.
Writes nn_numbers.tex.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', '..', 'source_data', 'MacromoleculeEquilibria_Crowding')
R, T = 8.314462618, 298.15
RT = R * T
RTcal = 1.98720 * T  # cal/mol

SEQ = 'ATCGTGTAGGCAT'  # donor strand (manuscript, Materials)
COMP = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
CANON = ['AA', 'AT', 'TA', 'CA', 'GT', 'CT', 'GA', 'CG', 'GC', 'GG']


def nn_key(s):
    if s in CANON:
        return s
    rc = ''.join(COMP[c] for c in reversed(s))
    assert rc in CANON, s
    return rc


def nn_sum(table, init):
    s = sum(table[nn_key(SEQ[i:i + 2])] for i in range(len(SEQ) - 1))
    return s + sum(init['GC' if e in 'GC' else 'AT'] for e in (SEQ[0], SEQ[-1]))


# ---------- (1) Ghosh 2020 Table 3, kcal/mol per unit da_w ----------
m_peg = dict(zip(CANON, [2.0, 6.4, 4.0, 4.6, 6.2, 4.0, 1.2, 9.4, 14.4, 8.8]))
m_eg = dict(zip(CANON, [0.7, 2.2, 1.4, 1.6, 2.2, 1.4, 0.4, 3.3, 5.0, 3.0]))
M_PEG = nn_sum(m_peg, {'GC': -4.4, 'AT': -0.6})
M_EG = nn_sum(m_eg, {'GC': -1.5, 'AT': -0.2})


def rho(w):  # g/cm3, same formula as reacting_probes_equillibrium_constants.py
    return 0.99707 + 0.17441 * w


def lnaw_ideal(w, M):
    return -18.015e-3 * w / (M * 1e-3 * (1 - w))


def lnaw_cohen(w, Mp):
    N = (Mp - 18) / 44.0
    C = w * rho(w) * 1e3
    v, Mm = 0.825e-3, 0.044
    Pi = RT * C / (Mm * N) + 0.49 * RT * (C * v) ** 2.25 / (Mm * v)
    return -Pi * 18.07e-6 / RT


# Nakano 2004 SI Table S2, ln a_w x 1e-2 in 1 M NaCl; baseline -3.37
NAK = {('EGly', 40): -17.5, ('PEG200', 40): -9.29, ('PEG1000', 40): -6.66}
MW = {'EGly': 62.07, 'PEG200': 200, 'PEG400': 400, 'PEG600': 600, 'PEG1000': 1000, 'PEG1500': 1500,
      'PEG3000': 3000, 'PEG6000': 6000, 'PEG12000': 12000, 'PEG20000': 20000, 'PEG35000': 35000}

rows = []
for cr, M in MW.items():
    d = pd.read_excel(os.path.join(SRC, f'OriginalData_{cr}.xlsx'))
    K0 = d.loc[d['crowder wt. [%]'] == 0, 'K [M]'].mean()
    for wt, g in d.groupby('crowder wt. [%]'):
        if wt == 0:
            continue
        w = wt / 100
        obs = RT * np.log(K0 / g['K [M]'].mean()) / 1e3  # kJ/mol, > 0 destabilisation
        lna = lnaw_ideal(w, M) if M <= 200 else lnaw_cohen(w, M)
        m = M_EG if cr == 'EGly' else M_PEG
        pred = m * (1 - np.exp(lna)) * 4.184
        key = (cr, int(wt))
        pred_nak = m * (1 - np.exp((NAK[key] + 3.37) * 1e-2)) * 4.184 if key in NAK else np.nan
        rows.append(dict(crowder=cr, M=M, wt=wt, obs=obs, pred=pred, pred_nak=pred_nak, resid=obs - pred))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(HERE, 'q9_nn_table.csv'), index=False)

long_ = df[(df.M >= 3000) & (df.wt >= 10)]

# ---------- (2) SantaLucia 1998 unified NN ----------
SL = {'AA': (-7.9, -22.2), 'AT': (-7.2, -20.4), 'TA': (-7.2, -21.3), 'CA': (-8.5, -22.7), 'GT': (-8.4, -22.4),
      'CT': (-7.8, -21.0), 'GA': (-8.2, -22.2), 'CG': (-10.6, -27.2), 'GC': (-9.8, -24.4), 'GG': (-8.0, -19.9)}
H = sum(SL[nn_key(SEQ[i:i + 2])][0] for i in range(len(SEQ) - 1))
S = sum(SL[nn_key(SEQ[i:i + 2])][1] for i in range(len(SEQ) - 1))
for e in (SEQ[0], SEQ[-1]):
    h, s = (0.1, -2.8) if e in 'GC' else (2.3, 4.1)
    H += h
    S += s
S_na = S + 0.368 * (len(SEQ) - 1) * np.log(0.035)
G_sl = H - T * S_na / 1000
K_sl = np.exp(-G_sl * 1000 / RTcal)
K0_obs = pd.read_excel(os.path.join(SRC, 'OriginalData_PEG6000.xlsx')).query('`crowder wt. [%]` == 0')['K [M]'].mean()
G_obs = -RTcal * np.log(K0_obs) / 1000

# ---------- (3) water balance ----------
MU_NA_VPO = 4.66  # cal/(mol molal EO), Knowles 2015 Table 2B
nw = {k: 55.51 * (MU_NA_VPO / RTcal + k) for k in (0.055, 0.168)}

out = {
    'NNmPEG': f'{M_PEG:.0f}', 'NNmEG': f'{M_EG:.0f}',
    'NNKsl': f'{K_sl / 1e9:.0f}', 'NNKobs': f'{K0_obs / 1e9:.1f}', 'NNdGdiff': f'{(G_obs - G_sl) * 4.184:.1f}',
    'NNresLo': f'{long_.resid.max():.0f}', 'NNresHi': f'{long_.resid.min():.0f}',
    'NNnwLo': f'{nw[0.055]:.1f}', 'NNnwHi': f'{nw[0.168]:.1f}',
}
pick = {'EGly': 'EG', 'PEG200': 'PtwoH', 'PEG600': 'PsixH', 'PEG1000': 'Ponek', 'PEG6000': 'Psixk', 'PEG35000': 'Pthirtyfivek'}
for cr, tag in pick.items():
    r = df[(df.crowder == cr) & (df.wt == 40)].iloc[0]
    out[f'NN{tag}obs'] = f'{r.obs:+.1f}'
    out[f'NN{tag}pred'] = f'{r.pred:+.1f}'
    out[f'NN{tag}res'] = f'{r.resid:+.1f}'
    out[f'NN{tag}nak'] = '--' if np.isnan(r.pred_nak) else f'{r.pred_nak:+.1f}'
with open(os.path.join(HERE, 'nn_numbers.tex'), 'w') as f:
    for k, v in out.items():
        f.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')

print(f'sum m_cs: PEG {M_PEG:.1f}, EG {M_EG:.1f} kcal/mol per unit da_w')
print(f'SantaLucia K0(35 mM, 25 C) = {K_sl:.2e} M^-1, measured {K0_obs:.2e}; dG(obs-NN) = {(G_obs - G_sl) * 4.184:.1f} kJ/mol')
print(f'residual PEG>=3000, 10-40 wt%: {long_.resid.min():.1f} .. {long_.resid.max():.1f} kJ/mol')
print(f'water per EO needed to cancel DOSY binding: kappa 0.055 -> {nw[0.055]:.1f}, 0.168 -> {nw[0.168]:.1f}')
print(df[df.wt == 40].round(2).to_string(index=False))
