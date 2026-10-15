"""Compare our DDG(x) with terms taken directly from the literature (no fitting to our data).

Run: .venv/bin/python analysis/literature_check/lit_budget.py
Sources of constants (see notes/*.md for DOIs and access status):
  Knowles 2011 PNAS (local PDF): monomer m-values, melting sign -> association sign flipped:
      EG..triEG duplex -0.340 kcal/molal -> +0.340 destabilisation per molal monomer
      high-M plateau +0.156 (linear) / +0.25 (quadratic initial) -> stabilisation
      reversal centred at 6 monomolal; above it m/N comparable to PEG 200 (-0.21 -> +0.21)
  Q3 (notes/Q3_depletion.md, q3_table.csv): protein-limit depletion, linear in mass concentration
  Q2 (notes/Q2_peg_salt.md, q2_mux_estimate_output.txt): d lnK / d m_EO = +0.123 (NaCl proxy)
      to +0.48 (Na2HPO4 proxy), i.e. stabilisation through raised a_pm
  Q5 (notes/Q5_Q6_dielectric_dyes.md): dielectric term |DDG| <= ~2 kJ/mol at 40 wt%, sign unknown
"""
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
os.chdir(REPO)
sys.path.insert(0, str(REPO))
from main_data_processor import Data  # noqa: E402

RT = 8.314462618e-3 * 298.15
KCAL = 4.184
num = {}

d = Data("source_data/MacromoleculeEquilibria_Crowding", "source_data/IonCrowderComplexation/raw_data",
         "source_data/IonStrengthDependency/Na_FRET.csv").data
d["wt"] = d["crowder wt. [%]"].astype(float)
d = d[d.wt <= 40]
d["K"] = d["K_uf [M]"].apply(lambda u: u.nominal_value)
d["sK"] = d["K_uf [M]"].apply(lambda u: u.std_dev)
K0 = float(d[d.wt == 0].K.iloc[0])
d["ddG"] = RT * np.log(K0 / d.K)
d["sddG"] = RT * d.sK / d.K


def x_of(wt, cr):
    w = np.asarray(wt, float) / 100
    return 1000 * w / ((1 - w) * (62.07 if cr == "EGly" else 44.05))


def conc(wt):
    wt = np.asarray(wt, float)
    return wt * (0.997 + 0.0017441 * wt) / 100


# protein-limit depletion (Q3), linear in mass concentration: use 40 wt% value and scale by C
q3 = pd.read_csv(HERE / "q3_table.csv")
P = q3[q3.model == "P"].groupby("ssDNA").first()
dep40 = {s: float(P.loc[s, "dG_40wt"]) for s in ["rod", "sphere"]}


def dep(wt, s):
    return dep40[s] * conc(wt) / conc(40.0)


def knowles_emp(x, s1=0.156, s2=0.21, xr=6.0):
    """association sign: stabilisation slope -s1 up to xr, then destabilisation +s2 per molal"""
    x = np.asarray(x, float)
    return KCAL * np.where(x <= xr, -s1 * x, -s1 * xr + s2 * (x - xr))


SALT = (0.123, 0.48)  # d lnK / d m_EO, Q2 section 3

wg = np.linspace(0, 40, 200)
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
for ax, cr in zip(axs, ["PEG6000", "PEG35000"]):
    o = d[d.crowder == cr]
    xg = x_of(wg, cr)
    ax.fill_between(wg, -RT * SALT[0] * xg, -RT * SALT[1] * xg, color="C2", alpha=.25, label="kanał solny z μ$_x$ (Q2)")
    ax.fill_between(wg, -2 * wg / 40, 2 * wg / 40, color="0.7", alpha=.4, label="dielektryczny (Q5)")
    ax.plot(wg, dep(wg, "rod"), "C0-", label="deplecja, gr. białkowa, ssDNA=pręt (Q3)")
    ax.plot(wg, dep(wg, "sphere"), "C0--", label="deplecja, ssDNA=kula (Q3)")
    ax.plot(wg, knowles_emp(xg), "C3-", lw=1.6, label="Knowles 2011: nachylenia empiryczne")
    ax.plot(wg, knowles_emp(xg, s1=0.25), "C3:", lw=1.4, label="Knowles 2011: nachylenie kwadratowe")
    ax.errorbar(o.wt, o.ddG, o.sddG, fmt="ko", ms=3.5, label="nasze dane")
    ax.axhline(0, color="k", lw=.5)
    ax.set_ylim(-25, 12)
    ax.set_xlabel("PEG [wt%]")
    ax.set_title({"PEG6000": "PEG 6k", "PEG35000": "PEG 35k"}[cr])
axs[0].set_ylabel("ΔΔG asocjacji [kJ/mol]")
axs[1].legend(frameon=False, fontsize=6, loc="lower left")
fig.savefig(HERE / "lit_budget.pdf", bbox_inches="tight")
plt.close(fig)

# numbers for the report
for cr, tag in [("PEG6000", "Six"), ("PEG35000", "ThirtyFive")]:
    o = d[d.crowder == cr].set_index("wt")
    for wt in [5.0, 20.0, 40.0]:
        x = float(x_of(wt, cr))
        t = f"{int(wt)}"
        num[f"obs{tag}{t}"] = o.loc[wt, "ddG"]
        num[f"kn{tag}{t}"] = float(knowles_emp(x))
        num[f"knq{tag}{t}"] = float(knowles_emp(x, s1=0.25))
    num[f"min{tag}"] = o.ddG.min()
num["depRod5"], num["depRod40"] = float(dep(5, "rod")), float(dep(40, "rod"))
num["depSph5"], num["depSph40"] = float(dep(5, "sphere")), float(dep(40, "sphere"))
x40 = float(x_of(40, "PEG35000"))
num["saltLo40"], num["saltHi40"] = -RT * SALT[0] * x40, -RT * SALT[1] * x40
x5 = float(x_of(5, "PEG35000"))
num["saltLo5"], num["saltHi5"] = -RT * SALT[0] * x5, -RT * SALT[1] * x5
eg = d[(d.crowder == "EGly") & (d.wt == 40)].iloc[0]
num["egObs40"] = eg.ddG
num["egKnowles40"] = 0.340 * KCAL * float(x_of(40, "EGly"))
pg = d[(d.crowder == "PEG200") & (d.wt == 40)].iloc[0]
num["pegTwoObs40"] = pg.ddG
num["pegTwoKnowles40"] = 0.21 * KCAL * float(x_of(40, "PEG200"))
num["xFortyPEG"] = x40
num["xFivePEG"] = x5

words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
with open(HERE / "lit_numbers.tex", "w") as fh:
    for k, v in sorted(num.items()):
        safe = "".join(ch if ch.isalpha() else words[int(ch)] for ch in k)
        fh.write(f"\\newcommand{{\\L{safe}}}{{{v:+.1f}}}\n")
for k, v in sorted(num.items()):
    print(k, f"{v:+.2f}")
