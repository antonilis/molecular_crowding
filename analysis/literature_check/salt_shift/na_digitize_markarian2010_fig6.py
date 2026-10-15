"""Template-matching digitization of Markarian & Schlenoff 2010, Fig. 6
(ln Kobs vs ln a_Na+, 35-mer duplex, 67 C, 1 mM phosphate pH 7, PEG3 = 'PEG 4000' Mw~3000).
Legend (caption): 0 wt% diamond, 1 wt% square, 5 wt% triangle, 10 wt% x, 20 wt% asterisk, 30 wt% circle.

Input: nihms-224880-f0006.jpg from PMC2925268 (601x516 px, 150 dpi), path as argv[1].
Templates are cut from isolated markers in the same image; every marker type is matched by
normalized correlation on the binary (dark) mask; detections are kept greedily (best score first)
with a minimum centre distance of 5 px between detections of any type.
Axis calibration: box frame x = 94..485 px <-> ln a = -4.0..-2.0; y = 37..411 px <-> ln K = 18..10
(frame lines found from long dark pixel runs; tick labels -4.0/-3.0/-2.0, 10..18 coincide with them).
1 px = 0.0051 ln a units (x) and 0.0214 ln K units (y). Reading uncertainty (+/-1.5 px):
+/-0.008 in ln a, +/-0.03 in ln K, plus marker-overlap errors noted in output 'flag'.
Output: na_markarian2010_fig6.csv next to this script.
"""
import sys, os, csv
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.signal import fftconvolve

HERE = os.path.dirname(os.path.abspath(__file__))
X0, X1, Y0, Y1 = 94, 485, 37, 411
lna = lambda px: -4.0 + (px - X0) / (X1 - X0) * 2.0
lnK = lambda py: 18.0 - (py - Y0) / (Y1 - Y0) * 8.0

# (label, PEG wt%, centre x, centre y) of isolated instances used as templates (half-size 6 px)
TEMPL = [('diamond', 0, 156, 302), ('square', 1, 121, 336), ('triangle', 5, 109, 349),
         ('x', 10, 121, 306), ('asterisk', 20, 121, 264), ('circle', 30, 157, 231)]
R = 6


def main(path):
    img = np.array(Image.open(path).convert('L')).astype(float)
    mask = (img < 140).astype(float)
    inner = np.zeros_like(mask); inner[Y0 + 3:Y1 - 2, X0 + 3:X1 - 2] = 1
    mask *= inner
    W = 2 * mask - 1                      # +1 dark, -1 light
    cands = []
    for name, wt, cx, cy in TEMPL:
        t = mask[cy - R:cy + R + 1, cx - R:cx + R + 1]
        # refine template centre to its centre of mass
        t = 2 * t - 1
        tpos = (t > 0).sum()
        score = fftconvolve(W, t[::-1, ::-1], mode='same') / t.size
        # local maxima
        mx = ndi.maximum_filter(score, size=5)
        ys, xs = np.where((score == mx) & (score > 0.55))
        for y, x in zip(ys, xs):
            cands.append((score[y, x], name, wt, x, y))
    cands.sort(reverse=True)
    kept = []
    for s, name, wt, x, y in cands:
        if all((x - k[3]) ** 2 + (y - k[4]) ** 2 >= 25 for k in kept):
            kept.append((s, name, wt, x, y))
    # template centre offset: centroid of template mask relative to its window centre
    rows = []
    for s, name, wt, x, y in kept:
        rows.append(dict(peg_wt=wt, marker=name, x_px=int(x), y_px=int(y),
                         ln_aNa=round(lna(x), 3), lnKobs=round(lnK(y), 3), score=round(float(s), 3)))
    rows.sort(key=lambda r: (r['peg_wt'], r['ln_aNa']))
    return rows


if __name__ == '__main__':
    rows = main(sys.argv[1])
    for r in rows:
        print(r)
    out = os.path.join(HERE, 'na_markarian2010_fig6_raw.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print('wrote', out, len(rows))
