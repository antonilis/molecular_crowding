"""Digitize Bielec et al. JPCL 2022 SI Fig. S3 (K vs Na+ at 40 wt% crowders).
Input: page-5 raster extracted with `pdfimages -f 5 -l 5 -png literature/pdf/jz1c03596_si_001.pdf bielec_s3`.
Axis calibration from detected tick marks: x: 0..120 mM -> px 331..1526; y: log10 K 11..5 -> px 100..1006.
"""
import numpy as np, pandas as pd
from PIL import Image
from scipy import ndimage as ndi
im = np.array(Image.open("bielec_s3-000.png").convert("RGB")).astype(int)
X0, XS = 331.0, (1526 - 331) / 120.0
Y0, YS = 100.0, (1006 - 100) / 6.0
COL = {"Glycerol": (240, 192, 48), "Ficoll 400": (240, 160, 32), "Dextran 40k": (208, 128, 224),
       "Dextran 70k": (160, 48, 160), "Ethylene Glycol": (128, 240, 192), "PEG 400": (48, 192, 112)}
rows = []
for name, c in COL.items():
    q = (im // 16) * 16
    m = np.all(q == np.array(c), axis=2)
    m[:, 1640:] = False          # legend
    m[:100, :] = False; m[1004:, :] = False
    m = ndi.binary_opening(m, structure=np.ones((9, 9)))   # removes error bars and dashed line
    lab, n = ndi.label(m)
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        if len(xs) < 120:
            continue
        cx, cy = xs.mean(), ys.mean()
        c_mM = (cx - X0) / XS
        logK = 11 - (cy - Y0) / YS
        rows.append(dict(crowder=name, Na_mM=round(c_mM, 1), log10K=round(logK, 3), area_px=len(xs),
                         bbox_w=xs.max() - xs.min() + 1, bbox_h=ys.max() - ys.min() + 1))
df = pd.DataFrame(rows).sort_values(["crowder", "Na_mM"])
# buffer power-law line (dashed, pink): sample its pixels for a check
q = (im // 16) * 16
pk = np.all(q == np.array(COL["Dextran 40k"]), axis=2); pk[:, 1640:] = False
thin = pk & ~ndi.binary_dilation(ndi.binary_opening(pk, structure=np.ones((9, 9))), iterations=4)
ys, xs = np.where(thin)
line = pd.DataFrame({"Na_mM": (xs - X0) / XS, "log10K": 11 - (ys - Y0) / YS})
line = line[(line.Na_mM > 1)].groupby(line.Na_mM.round(0)).log10K.median().reset_index()
df.to_csv("bielec_s3_points.csv", index=False)
line.to_csv("bielec_s3_bufferline.csv", index=False)
print(df.to_string(index=False))
print(line.iloc[::10].to_string(index=False))
