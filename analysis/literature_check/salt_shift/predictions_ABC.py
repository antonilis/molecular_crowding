"""Predictions of hypotheses A (horizontal), B (vertical), C (slope scaling) on OUR buffer curve F(ln c).
F: smoothing spline of ln K vs ln c_Na from source_data/IonStrengthDependency/Na_FRET.csv (<=175 mM)
plus the buffer points > 175 mM listed in manuscript/SI/main.tex (Table of K vs Na+), read-only.
phi ranges: from Q2 (VPO, salt exclusion: dln a_pm ~ +0.2..+0.5 at 40 wt%, i.e. phi < 0) and from
DOSY binding (consistent version phi = 0.5 ln(1 + kappa*c_EO), kappa 0.055..0.168 M^-1, c_EO ~ 9.7 M).
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

REPO = Path(__file__).resolve().parents[3]
s = pd.read_csv(REPO / "source_data/IonStrengthDependency/Na_FRET.csv")
c = s["C_Na+_[mM]"].values.astype(float); K = s["K_Na+_[1/M]"].values.astype(float); sK = s["K_err_Na+_[1/M]"].values.astype(float)
extra = []
for line in open(REPO / "manuscript/SI/main.tex", encoding="utf8"):
    p = [q.strip() for q in line.split("&")]
    if len(p) == 6 and p[3].replace(".", "").isdigit():
        try:
            cc = float(p[3]); kk = float(p[4].replace("$", "").replace("\\mathrm{e}^{", "e").replace("}", "").replace(" ", ""))
            ee = float(p[5].replace("$", "").replace("\\mathrm{e}^{", "e").replace("}", "").replace("\\", "").replace(" ", ""))
        except ValueError:
            continue
        if cc > 176:
            extra.append((cc, kk, ee))
extra = np.array(extra)
cc = np.r_[c, extra[:, 0]]; KK = np.r_[K, extra[:, 1]]; ss = np.r_[sK, extra[:, 2]]
o = np.argsort(cc); cc, KK, ss = cc[o], KK[o], ss[o]
lc, lk, w = np.log(cc), np.log(KK), KK / ss
F = UnivariateSpline(lc, lk, w=w, k=3, s=2.5 * len(lc))
g = np.linspace(np.log(5.0), np.log(1754), 400)
cmax = float(np.exp(g[np.argmax(F(g))]))
c0 = 35.0
SK0 = float(F.derivative()(np.log(c0)))
res = {"c_max_buffer_mM": cmax, "SK_35": SK0, "n_points": int(len(cc))}

scen = {
    "A, φ = −0.4 (wypychanie soli, VPO)": lambda x: F(x + 0.4),
    "A, φ = +0.4 (wiązanie Na⁺, DOSY)": lambda x: F(x - 0.4),
    "B, H = −1": lambda x: F(x) - 1.0,
    "C, SK × 0.7": lambda x: F(np.log(c0)) + 0.7 * (F(x) - F(np.log(c0))),
}
for k, f in scen.items():
    v = f(g)
    d = (f(np.log(c0) + 1e-3) - f(np.log(c0) - 1e-3)) / 2e-3
    res[k] = {"c_max_mM": float(np.exp(g[np.argmax(v)])), "lnK_max_minus_buffer": float(v.max() - F(g).max()),
              "SK_35": float(d)}
json.dump(res, open(Path(__file__).with_name("predictions_ABC.json"), "w"), indent=1, ensure_ascii=False)

fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.9))
ax = axs[0]
ax.errorbar(cc, lk, ss / KK, fmt="ko", ms=3.5, label="bufor (Na_FRET.csv + SI)")
ax.plot(np.exp(g), F(g), "k-", lw=1.2, label="F (splajn)")
cols = ["C0", "C3", "C2", "C1"]
for (k, f), col in zip(scen.items(), cols):
    ax.plot(np.exp(g), f(g), color=col, lw=1.1, label=k)
    ax.axvline(res[k]["c_max_mM"], color=col, ls=":", lw=.8)
ax.axvline(cmax, color="k", ls=":", lw=.8)
ax.set_xscale("log"); ax.set_xlabel("c$_\\mathrm{Na}$ [mM]"); ax.set_ylabel("ln K")
ax.legend(frameon=False, fontsize=6); ax.set_ylim(12, 26)
ax.set_title("tylko A przesuwa położenie maksimum", fontsize=9)
ax = axs[1]
for (k, f), col in zip(scen.items(), cols):
    d = np.gradient(f(g), g)
    ax.plot(np.exp(g), d, color=col, lw=1.1)
ax.plot(np.exp(g), F.derivative()(g), "k-", lw=1.2)
ax.axvline(c0, color="grey", ls=":")
ax.axhline(0, color="grey", lw=.5)
ax.set_xscale("log"); ax.set_xlabel("c$_\\mathrm{Na}$ [mM]"); ax.set_ylabel("SK = d ln K / d ln c")
ax.set_title("SK: B bez zmian, C skaluje, A przesuwa krzywą SK(c)", fontsize=9)
fig.tight_layout()
fig.savefig(Path(__file__).with_name("predictions_ABC.pdf"), bbox_inches="tight")
print(json.dumps(res, indent=1, ensure_ascii=False))
