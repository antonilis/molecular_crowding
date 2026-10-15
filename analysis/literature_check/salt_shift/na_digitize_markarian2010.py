"""Digitize Markarian & Schlenoff 2010 (JPCB 114, 10620; PMC2925268, NIH author manuscript)
Fig. 6 (ln Kobs vs ln a_Na+, 35-mer, 67 C, PEG3 0-30 wt%) and Fig. 7A (Delta psi per bp vs PEG wt%).

Input: figure JPEGs downloaded from PMC (nihms-224880-f0006.jpg, -f0007.jpg), passed as arguments.
Method: axis frame located from long dark pixel runs (box frame); markers found as connected
components of the dark mask; shape classified from bbox fill ratio and centroid offset.
Output: CSV files next to this script (na_markarian2010_fig6.csv, na_markarian2010_fig7A.csv).
Usage: python -I na_digitize_markarian2010.py f0006.jpg f0007.jpg
"""
import sys, os, csv
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))


def frame(mask, thr):
    cols = [i for i, c in enumerate(mask.sum(0)) if c > thr]
    rows = [i for i, c in enumerate(mask.sum(1)) if c > thr]
    return min(cols), max(cols), min(rows), max(rows)


def components(mask, x0, x1, y0, y1, min_px=12):
    sub = mask.copy()
    # remove the frame lines themselves
    sub[:, max(x0 - 1, 0):x0 + 2] = False
    sub[:, x1 - 1:x1 + 2] = False
    sub[y0 - 1:y0 + 2, :] = False
    sub[y1 - 1:y1 + 2, :] = False
    sub[:, :x0] = False; sub[:, x1:] = False; sub[:y0, :] = False; sub[y1:, :] = False
    lab, n = ndi.label(sub, structure=np.ones((3, 3)))
    out = []
    for k, sl in enumerate(ndi.find_objects(lab), start=1):
        m = lab[sl] == k
        npx = m.sum()
        if npx < min_px:
            continue
        h, w = m.shape
        cy, cx = ndi.center_of_mass(m)
        out.append(dict(x=sl[1].start + cx, y=sl[0].start + cy, w=w, h=h, npx=int(npx),
                        fill=npx / (w * h), dyrel=(cy - (h - 1) / 2) / h,
                        bx=sl[1].start + (w - 1) / 2, by=sl[0].start + (h - 1) / 2))
    return out


def fig7A(path):
    img = np.array(Image.open(path).convert('L'))[:330]
    mask = img < 128
    x0, x1, y0, y1 = frame(mask, 150)          # x: 0..35 wt%, y: 0.40 (top) .. 0.00 (bottom)
    X = lambda px: (px - x0) / (x1 - x0) * 35.0
    Y = lambda py: (y1 - py) / (y1 - y0) * 0.40
    # Opening removes 1-2 px fit lines and error bars, keeps filled markers.
    opened = ndi.binary_opening(mask, structure=np.ones((4, 4)))
    comps = components(opened, x0, x1, y0, y1, min_px=10)
    rows = []
    for c in comps:
        xv = X(c['bx'])
        # shape: square fill ~1, triangle centroid below bbox centre, diamond centred
        if c['fill'] > 0.8:
            s = 'square(25-mer)'
        elif c['dyrel'] > 0.06:
            s = 'triangle(35-mer)'
        else:
            s = 'diamond(15-mer)'
        # marker centre: bbox centre for square/diamond, centroid->bbox centre for triangle
        rows.append(dict(series=s, x_px=round(c['bx'], 1), y_px=round(c['by'], 1),
                         peg_wt=round(xv, 2), dpsi_per_bp=round(Y(c['by']), 4),
                         w=c['w'], h=c['h'], fill=round(c['fill'], 2)))
    return (x0, x1, y0, y1), sorted(rows, key=lambda r: (r['series'], r['peg_wt']))


def fig6(path):
    img = np.array(Image.open(path).convert('L'))
    mask = img < 140
    x0, x1, y0, y1 = frame(mask, 200)          # x: -4.0..-2.0, y: 18 (top) .. 10 (bottom)
    X = lambda px: -4.0 + (px - x0) / (x1 - x0) * 2.0
    Y = lambda py: 10.0 + (y1 - py) / (y1 - y0) * 8.0
    comps = components(mask, x0 + 2, x1 - 2, y0 + 2, y1 - 2, min_px=12)
    rows = []
    for c in comps:
        rows.append(dict(x_px=round(c['bx'], 1), y_px=round(c['by'], 1), lnaNa=round(X(c['bx']), 3),
                         lnK=round(Y(c['by']), 3), w=c['w'], h=c['h'], npx=c['npx'], fill=round(c['fill'], 2),
                         dyrel=round(c['dyrel'], 3)))
    return (x0, x1, y0, y1), sorted(rows, key=lambda r: r['lnaNa'])


if __name__ == '__main__':
    f6, f7 = sys.argv[1], sys.argv[2]
    fr7, r7 = fig7A(f7)
    print('Fig7A frame px', fr7)
    for r in r7:
        print(r)
    fr6, r6 = fig6(f6)
    print('Fig6 frame px', fr6)
    for r in r6:
        print(r)
