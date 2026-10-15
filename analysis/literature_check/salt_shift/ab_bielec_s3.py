"""A/B analysis of Bielec et al. JPCL 2022 SI Fig. S3 (digitized by digitize_bielec_s3.py).

A (salt activity): lnK_x(c) = F(ln c - phi)      B (additive): lnK_x(c) = F(ln c) + H
Their buffer reference F is a power law (dashed line, K = X c^Y), so on their own reference A and B
both predict the SAME local slope as the buffer; a different slope in a crowder contradicts both and
means the crowder changes SK itself (d SK/dx != 0). Under a simple cation-binding model
(free Na = c/(1 + kappa [EO]), independent of c at low salt) phi is constant in c.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pts = pd.read_csv("bielec_s3_points.csv")
# keep main-marker detections only (bbox >= 16 px in both directions) and add occluded points read by hand
pts = pts[(pts.bbox_w >= 16) & (pts.bbox_h >= 16) | (pts.crowder == "Glycerol")].copy()
pts = pts[~((pts.crowder == "Ethylene Glycol") & (pts.area_px < 300))]
pts["src"] = "auto"
manual = pd.DataFrame([
    ("Ficoll 400", 10.0, 7.60, "occluded by Dextran 70k; manual"),
    ("Ficoll 400", 15.0, 8.03, "partially occluded; partial blob"),
    ("Ficoll 400", 50.0, 8.85, "occluded; manual"),
    ("Dextran 40k", 50.0, 8.86, "occluded by Dextran 70k; manual"),
    ("Ethylene Glycol", 30.0, 7.42, "partially occluded; partial blob"),
    ("Ethylene Glycol", 50.0, 7.69, "partially occluded; partial blob"),
], columns=["crowder", "Na_mM", "log10K", "src"])
pts = pd.concat([pts[["crowder", "Na_mM", "log10K", "src"]], manual], ignore_index=True)
pts["Na_mM"] = pts.Na_mM.round(0)
pts["u_read"] = np.where(pts.src == "auto", 0.02, np.where(pts.src.str.contains("manual"), 0.10, 0.05))
pts = pts.sort_values(["crowder", "Na_mM"]).drop_duplicates(["crowder", "Na_mM"])
pts.to_csv("bielec_s3_points_final.csv", index=False)

line = pd.read_csv("bielec_s3_bufferline.csv")
ln_ = line[(line.Na_mM >= 5) & (line.Na_mM <= 125)]
b, a = np.polyfit(np.log10(ln_.Na_mM), ln_.log10K, 1)   # buffer reference: log10K = a + b log10 c
res = {"buffer_slope_Y": b, "buffer_intercept": a}


def Fbuf(c):
    return a + b * np.log10(c)


out = []
for cr, g in pts.groupby("crowder"):
    x = np.log10(g.Na_mM.values); y = g.log10K.values; u = g.u_read.values
    # free slope fit with scatter-based uncertainty
    p, cov = np.polyfit(x, y, 1, cov=True) if len(x) > 3 else (np.polyfit(x, y, 1), np.full((2, 2), np.nan))
    resid = y - np.polyval(p, x)
    # B fit: slope fixed to buffer slope
    H = np.mean(y - Fbuf(g.Na_mM.values))
    resB = y - (Fbuf(g.Na_mM.values) + H)
    # local slopes low/high
    lo = g[g.Na_mM <= 50]; hi = g[g.Na_mM >= 50]
    s_lo = np.polyfit(np.log10(lo.Na_mM), lo.log10K, 1)[0] if len(lo) >= 2 else np.nan
    s_hi = np.polyfit(np.log10(hi.Na_mM), hi.log10K, 1)[0] if len(hi) >= 2 else np.nan
    # implied A-shift per point on the power-law reference: c_eff = c * 10**((y - F)/b)
    frac_bound = 1 - 10 ** ((y - Fbuf(g.Na_mM.values)) / b)
    out.append(dict(crowder=cr, n=len(x), c_min=g.Na_mM.min(), c_max=g.Na_mM.max(),
                    slope=p[0], slope_err=float(np.sqrt(cov[0, 0])) if np.isfinite(cov[0, 0]) else np.nan,
                    slope_lo=s_lo, slope_hi=s_hi, H_log10=H, rmsB=float(np.sqrt(np.mean(resB ** 2))),
                    rmsFree=float(np.sqrt(np.mean(resid ** 2))),
                    fbound_cmin=float(frac_bound[0]), fbound_cmax=float(frac_bound[-1])))
T = pd.DataFrame(out)
T.to_csv("ab_bielec_s3_summary.csv", index=False)
print(f"buffer reference slope Y = {b:.2f}")
print(T.round(3).to_string(index=False))
res["table"] = T.round(4).to_dict(orient="records")
json.dump(res, open("ab_bielec_s3_results.json", "w"), indent=1)

# figure
cols = {"Glycerol": "#e8b100", "Ficoll 400": "#f08c00", "Dextran 40k": "#d07de0", "Dextran 70k": "#9c2fa0",
        "Ethylene Glycol": "#59d6a6", "PEG 400": "#1f9e5a"}
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0))
cg = np.linspace(8, 130, 100)
ax = axs[0]
ax.plot(cg, Fbuf(cg), "--", color="#d07de0", label=f"bufor (dopasowanie potęgowe, Y={b:.2f})")
for cr, g in pts.groupby("crowder"):
    ax.errorbar(g.Na_mM, g.log10K, g.u_read, fmt="o", ms=4, color=cols[cr], label=cr)
    r = T[T.crowder == cr].iloc[0]
    ax.plot(g.Na_mM, Fbuf(g.Na_mM) + r.H_log10, ":", color=cols[cr], lw=1)
ax.set_xscale("log"); ax.set_xlabel("Na$^+$ [mM]"); ax.set_ylabel("log$_{10}$K")
ax.set_title("kropki: hipoteza B (nachylenie buforu)", fontsize=8.5)
ax.legend(frameon=False, fontsize=5.8, loc="lower right")
ax = axs[1]
for i, r in T.iterrows():
    ax.errorbar(i, r.slope, r.slope_err, fmt="o", color=cols[r.crowder])
    ax.plot([i - .2, i + .2], [r.slope_lo, r.slope_hi], "-", color=cols[r.crowder], lw=1, alpha=.6)
ax.axhline(b, color="#d07de0", ls="--", label="nachylenie buforu (ich referencja)")
ax.set_xticks(range(len(T)), [c.replace(" ", "\n") for c in T.crowder], fontsize=6.5)
ax.set_ylabel("d log K / d log c (w crowderze)")
ax.set_title("A i B (przy potęgowym F) wymagają\nnachylenia buforu; linia: ≤50 → ≥50 mM", fontsize=8.5)
ax.legend(frameon=False, fontsize=6.5)
fig.tight_layout()
fig.savefig("ab_bielec_s3.pdf", bbox_inches="tight")
