"""Re-plot digitized literature data (na_markarian2010_fig6.csv, na_nakano2014_fig3B.csv) in the A/B/C frame."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
m = pd.read_csv("na_markarian2010_fig6.csv"); n = pd.read_csv("na_nakano2014_fig3B.csv")
fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.9))
ax = axs[0]; cm = plt.get_cmap("viridis")
for i, (w, g) in enumerate(m.groupby("peg_wt")):
    p = np.polyfit(g.ln_aNa, g.lnKobs, 1); col = cm(i / 5)
    ax.plot(g.ln_aNa, g.lnKobs, "o", ms=3, color=col, label=f"PEG {w:g}% (S={p[0]:.2f})")
    xx = np.linspace(g.ln_aNa.min(), g.ln_aNa.max(), 20); ax.plot(xx, np.polyval(p, xx), "-", color=col, lw=.8)
ax.set_xlabel("ln a$_\\mathrm{Na^+}$ (z VPO)"); ax.set_ylabel("ln K$_\\mathrm{obs}$")
ax.set_title("Markarian & Schlenoff 2010 (35-mer): przecięcie, S↓", fontsize=8.5); ax.legend(frameon=False, fontsize=6)
ax = axs[1]
for s, g in n.groupby("series"):
    p = np.polyfit(g.log_NaCl, g.logK, 1)
    ax.errorbar(g.log_NaCl, g.logK, g.logK_unc, fmt="o-", ms=3, lw=.8, label=f"{s} (S={p[0]:.2f})")
ax.set_xlabel("log [NaCl] (M)"); ax.set_ylabel("log K"); ax.legend(frameon=False, fontsize=6)
ax.set_title("Nakano 2014 (RNA 11-mer, 37 °C)", fontsize=8.5)
fig.tight_layout(); fig.savefig("lit_replot.pdf", bbox_inches="tight")
