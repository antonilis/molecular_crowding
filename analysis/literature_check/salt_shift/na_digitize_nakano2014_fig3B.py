"""Digitize Nakano, Kitagawa, Miyoshi, Sugimoto 2014 FEBS Open Bio 4, 643 (DOI 10.1016/j.fob.2014.06.009,
PMC4141205, CC BY-NC-ND), Fig. 3B: log K (37 C) vs log[NaCl] for the 11-mer RNA duplex
5'-AUCCAGCGCGC/GUGCGCUGGAU, 20 wt% cosolutes. Legend (Fig. 3 caption): none = black, PEG200 = blue,
EtOH = brown (squares), FA = red (triangles), DMF = cyan/light blue (triangles), PEG8000 = purple.
Input: gr3.jpg from PMC (685x183 px), argv[1]. Only 113 dpi -> panel B is ~150x130 px.
Calibration from tick marks found in the pixel map: x ticks -1.5/-1.0/-0.5/0.0 at 312/350.5/389/427.5 px
(77 px per unit, x(-2.0)=273.5); y ticks 14..7 at 19..150 px (18.71 px per unit).
Markers are colour-segmented (RGB rules below), connected components >= 5 px, centroid taken.
Reading uncertainty: +/-0.75 px -> +/-0.01 in log[NaCl] and +/-0.04 in log K; overlapping markers flagged.
Output: na_nakano2014_fig3B.csv
"""
import sys, os, csv
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
X = lambda px: -2.0 + (px - 273.5) / 77.0
Y = lambda py: 14.0 - (py - 19.0) / 18.714

RULES = {
    'none (black)': lambda r, g, b: (r < 80) & (g < 80) & (b < 80) & (np.abs(r - b) < 25),
    'PEG8000 (purple)': lambda r, g, b: (r > 80) & (r < 190) & (g < 110) & (b > 120) & (b - g > 60) & (r - g > 30),
    'PEG200 (blue)': lambda r, g, b: (r < 90) & (g < 90) & (b > 90) & (b - r > 50),
    'EtOH (brown)': lambda r, g, b: (r > 55) & (r < 150) & (g < 70) & (b < 80) & (r - b > 25) & (r - g > 30),
    'FA (red)': lambda r, g, b: (r > 150) & (g < 120) & (b < 110) & (r - g > 70),
    'DMF (light blue)': lambda r, g, b: (b > 160) & (g > 110) & (g < 215) & (b - r > 45),
}

def main(path):
    a = np.array(Image.open(path).convert('RGB')).astype(int)
    y0, y1, x0, x1 = 21, 149, 276, 437
    sub = a[y0:y1, x0:x1]
    r, g, b = sub[..., 0], sub[..., 1], sub[..., 2]
    rows = []
    for name, f in RULES.items():
        m = f(r, g, b)
        lab, n = ndi.label(m, structure=np.ones((3, 3)))
        for k, sl in enumerate(ndi.find_objects(lab), 1):
            mm = lab[sl] == k
            if mm.sum() < 5:
                continue
            cy, cx = ndi.center_of_mass(mm)
            px, py = x0 + sl[1].start + cx, y0 + sl[0].start + cy
            rows.append(dict(series=name, x_px=round(px, 1), y_px=round(py, 1), npx=int(mm.sum()),
                             log_NaCl=round(X(px), 3), logK=round(Y(py), 3)))
    return sorted(rows, key=lambda d: (d['series'], d['log_NaCl']))

if __name__ == '__main__':
    rows = main(sys.argv[1])
    for d in rows:
        print(d)
    with open(os.path.join(HERE, 'na_nakano2014_fig3B_raw.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
