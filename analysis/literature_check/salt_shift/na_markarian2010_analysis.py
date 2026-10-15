"""Markarian & Schlenoff 2010 (J. Phys. Chem. B 114, 10620; DOI 10.1021/jp103213w; PMC2925268 author ms.)
Final digitized data (Fig. 6 and Fig. 7A) + A/B analysis.

Fig. 6 pixel positions: template matching (na_digitize_markarian2010_fig6.py -> *_raw.csv) checked
and corrected by eye on 6x/8x gridded zooms (false hits removed, overlapped markers added; flag='overlap').
Calibration: x px 94..485 <-> ln a_Na+ -4.0..-2.0 ; y px 37..411 <-> ln Kobs 18..10.
Fig. 7A: frame x 95..365 px <-> 0..35 wt%, y 28..284 px <-> 0.40..0.00; marker centres from
connected components after 4x4 binary opening (na_digitize_markarian2010.py), series assigned by
marker shape/rank at each nominal PEG wt% (0,1,5,10,20,30); 1 px = 0.0016 units.
Writes na_markarian2010_fig6.csv, na_markarian2010_fig7A.csv, prints fits.
"""
import os, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
lna = lambda px: -4.0 + (px - 94) / (485 - 94) * 2.0
lnK = lambda py: 18.0 - (py - 37) / (411 - 37) * 8.0

# (PEG wt%, x_px, y_px, flag)
F6 = [
    (0, 156, 302.5, ''), (0, 192, 267, ''), (0, 210.6, 236.3, ''), (0, 241.9, 220.6, ''),
    (0, 270, 203.5, 'overlap(under 1% square)'), (0, 345, 161.3, ''), (0, 391.3, 134.7, ''),
    (0, 436, 93.3, 'overlap(on 1% square)'), (0, 474, 72, ''),
    (1, 121.5, 336.5, ''), (1, 159.5, 283.5, ''), (1, 191.9, 248, ''), (1, 210, 220, ''),
    (1, 246, 207.5, 'overlap(with 5% triangle)'), (1, 269, 192.5, ''), (1, 343, 149.2, ''),
    (1, 387.5, 123, ''), (1, 436, 96.7, 'overlap(with 0% diamond)'), (1, 468, 62.5, ''),
    (5, 109, 349.5, ''), (5, 142, 294.5, ''), (5, 193, 257.5, 'overlap(under 1% square)'),
    (5, 221.3, 228.5, ''), (5, 251, 206, 'overlap(with 1% square)'),
    (5, 290, 183.5, 'overlap(asterisk/circle)'), (5, 353.7, 146.7, ''), (5, 415.3, 125.3, ''),
    (5, 458.3, 105.8, ''),
    (10, 121, 306, ''), (10, 159, 256, ''), (10, 204.4, 225, ''), (10, 226, 206, 'overlap(asterisk)'),
    (10, 258, 191, 'overlap(asterisk/circle)'), (10, 275, 182.5, ''), (10, 306, 172.7, ''),
    (10, 384, 137, ''), (10, 442.5, 112.5, ''),
    (20, 121, 264, ''), (20, 179, 224, ''), (20, 223.8, 202.5, 'overlap(x)'),
    (20, 257.5, 189.4, 'overlap(x/circle)'), (20, 290, 177, 'overlap(triangle)'), (20, 315, 171.7, ''),
    (20, 337, 166, ''), (20, 358.7, 160.5, ''), (20, 434.7, 139.7, ''),
    (30, 157.5, 231.5, ''), (30, 216.9, 211.3, ''), (30, 260, 194.4, 'overlap(hidden, uncertain)'),
    (30, 296, 188, ''), (30, 327.5, 178.5, ''), (30, 353, 171.7, ''), (30, 376.3, 165.8, ''),
    (30, 396, 159.7, ''), (30, 473.7, 140.3, ''),
]

# Fig 7A, Delta psi per bp (author definition = 2 x slope / N_bp, see text), read values
F7A = {  # series: [(wt, dpsi)]
    '35-mer (67 C)': [(0, 0.169), (1, 0.169), (5, 0.158), (10, 0.138), (20, 0.0875), (30, 0.077)],
    '25-mer (55 C)': [(0, 0.340), (1, 0.341), (5, 0.320), (10, 0.250), (20, 0.2305), (30, 0.191)],
    '15-mer (35 C)': [(0, 0.350), (5, 0.291), (10, 0.309), (20, 0.261), (30, 0.239)],
}
NBP = {'35-mer (67 C)': 35, '25-mer (55 C)': 25, '15-mer (35 C)': 15}

if __name__ == '__main__':
    rows = [dict(peg_wt=w, x_px=x, y_px=y, ln_aNa=round(lna(x), 3), lnKobs=round(lnK(y), 3),
                 flag=f) for w, x, y, f in F6]
    with open(os.path.join(HERE, 'na_markarian2010_fig6.csv'), 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    print('Fig 6: linear fits ln K = a + S*ln a_Na (35-mer, 67 C); unc. read +/-0.01 (x), +/-0.03 (y)')
    fits = {}
    for w in sorted(set(r['peg_wt'] for r in rows)):
        d = np.array([(r['ln_aNa'], r['lnKobs']) for r in rows if r['peg_wt'] == w])
        p, cov = np.polyfit(d[:, 0], d[:, 1], 1, cov=True)
        fits[w] = p
        # quadratic curvature test
        q = np.polyfit(d[:, 0], d[:, 1], 2) if len(d) > 4 else [np.nan]
        print(f'  {w:2d} wt%: n={len(d):2d}, range ln a {d[:,0].min():.2f}..{d[:,0].max():.2f}, '
              f'S = {p[0]:.2f} +/- {np.sqrt(cov[0,0]):.2f}, lnK(-3.5) = {np.polyval(p,-3.5):.2f}, '
              f'lnK(-2.2) = {np.polyval(p,-2.2):.2f}, quad coeff = {q[0]:.2f}')
    p0 = fits[0]
    print('  Delta lnK (fit PEG - fit 0%) at ln a = -3.7, -3.0, -2.2 and crossover ln a*:')
    for w, p in fits.items():
        if w == 0: continue
        dd = [np.polyval(p, x) - np.polyval(p0, x) for x in (-3.7, -3.0, -2.2)]
        xs = (p0[1] - p[1]) / (p[0] - p0[0]) if abs(p[0] - p0[0]) > 1e-6 else np.nan
        print(f'   {w:2d}%: {dd[0]:+.2f} {dd[1]:+.2f} {dd[2]:+.2f}   ln a* = {xs:.2f} (a* = {np.exp(xs):.3f})')
    rows7 = []
    print('Fig 7A: S_K = dpsi*N/2 ; dS_K/dwt from linear fit')
    for s, pts in F7A.items():
        a = np.array(pts)
        p = np.polyfit(a[:, 0], a[:, 1], 1)
        print(f'  {s}: S_K(0%) = {a[0,1]*NBP[s]/2:.2f}, S_K(30%) = {a[-1,1]*NBP[s]/2:.2f}, '
              f'dS_K/d(wt%) = {p[0]*NBP[s]/2:.4f} per wt%')
        for w, v in pts:
            rows7.append(dict(series=s, peg_wt=w, dpsi_per_bp=v, S_K=round(v * NBP[s] / 2, 3),
                              read_unc=0.005))
    with open(os.path.join(HERE, 'na_markarian2010_fig7A.csv'), 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows7[0])); wr.writeheader(); wr.writerows(rows7)
