"""Final table + A/B/C analysis for Nakano et al. 2014 FEBS Open Bio Fig. 3B (11-mer RNA duplex, 37 C,
20 wt% cosolute, NaCl + 10 mM Na2HPO4 + 1 mM Na2EDTA as written in the paper's CD method; the melting
buffer is called 'the phosphate buffer').
Values from na_digitize_nakano2014_fig3B.py (colour segmentation). x snapped to the nominal NaCl
concentrations 0.02, 0.05, 0.1, 0.2, 0.5, 0.7, 1.0 M (read x within 0.025 of log of these values).
Unreliable/overlapping series (EtOH, FA at low salt) are omitted. Uncertainty of log K reading ~ +/-0.04.
Note: log K is decadic; ln K = 2.303 log K. Slopes d logK/d log c = d lnK/d ln c (= S_K).
"""
import os, csv
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
lc = np.log10([0.02, 0.05, 0.1, 0.2, 0.5, 0.7, 1.0])
D = {  # log K read
    'none': [9.787, 10.370, 10.740, 11.364, 12.159, 12.595, 12.780],
    'PEG200 20wt%': [9.281, 9.783, 10.210, 10.636, 11.194, 11.424, 11.635],
    'PEG8000 20wt%': [10.553, 10.923, 11.301, 11.684, None, None, None],
    'DMF 20wt%': [7.742, 8.190, 8.820, 9.298, 9.764, 10.080, 10.234],
    'FA 20wt%': [None, None, None, None, 10.281, 10.574, 10.731],
}
rows = []
for s, v in D.items():
    for x, y in zip(lc, v):
        if y is not None:
            rows.append(dict(series=s, NaCl_M=round(10 ** x, 3), log_NaCl=round(x, 3), logK=y, logK_unc=0.04))
with open(os.path.join(HERE, 'na_nakano2014_fig3B.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

none = np.array(D['none'])
p0 = np.polyfit(lc, none, 1); q0 = np.polyfit(lc, none, 2)
print(f'none: slope {p0[0]:.2f} (authors 1.8); quadratic coeff {q0[0]:+.2f} -> dS/dlog c = {2*q0[0]:+.2f} '
      f'(local S at log c=-1.7: {2*q0[0]*-1.7+q0[1]:.2f}, at 0: {q0[1]:.2f})')
for s, v in D.items():
    if s == 'none':
        continue
    m = np.array([y is not None for y in v]); y = np.array([np.nan if t is None else t for t in v])
    if m.sum() >= 3:
        p = np.polyfit(lc[m], y[m], 1)
        p0m = np.polyfit(lc[m], none[m], 1)
        print(f'{s}: slope {p[0]:.2f} (none over same range: {p0m[0]:.2f})')
    d = y - none
    print('   Delta logK vs none at', ', '.join(f'{10**x:.2f} M: {dd:+.2f}' for x, dd in zip(lc, d) if not np.isnan(dd)))
# PEG8000: extrapolated crossing with buffer line
p8 = np.polyfit(lc[:4], np.array(D['PEG8000 20wt%'][:4]), 1); pn = np.polyfit(lc[:4], none[:4], 1)
xc = (pn[1] - p8[1]) / (p8[0] - pn[0])
print(f'PEG8000 vs none (0.02-0.2 M fits): slopes {p8[0]:.2f} vs {pn[0]:.2f}; extrapolated crossing at log c = {xc:.2f} ({10**xc:.2f} M)')
# A-model test for PEG8000: Delta lnK = -phi * S ; S nearly constant over 0.02-0.2 M in buffer
d8 = (np.array(D['PEG8000 20wt%'][:4]) - none[:4])
S_loc = pn[0]
print('A-model phi for PEG8000 (phi = -Delta logK / S_none):', ', '.join(f'{-dd/S_loc:+.2f}' for dd in d8),
      '(log10 units of c; x2.303 for ln)')
print('A cannot change the slope where the buffer line is straight: observed slope change '
      f'{p8[0]-pn[0]:+.2f} over 0.02-0.2 M while buffer curvature over that range is small.')
