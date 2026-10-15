"""Critical check of four claims from the internal review (recenzja_manuskryptu.md, M2-M9).

Run from anywhere:  .venv/bin/python analysis/knowles_reframe/analysis.py
Reads only repository data and code; writes figures/, numbers.tex and results.json
next to this script. Every number quoted in report.tex comes from numbers.tex.
"""
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq, curve_fit

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
os.chdir(REPO)
sys.path.insert(0, str(REPO))

from main_data_processor import Data  # noqa: E402

FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
RNG = np.random.default_rng(20261009)
N_MC = 4000

R = 8.314462618e-3  # kJ mol^-1 K^-1
T = 298.15
RT = R * T  # kJ/mol
KCAL = 4.184

numbers = {}  # macro name -> formatted string
results = {}  # machine-readable record


def put(name, value, fmt="{:.2f}"):
    numbers[name] = fmt.format(value) if not isinstance(value, str) else value
    results[name] = value if isinstance(value, (str, int, float)) else float(value)


def pm(name, val, err, fmt="{:.2f}"):
    put(name, val, fmt)
    put(name + "Err", abs(err), fmt.replace("+", ""))


CROWDERS = ["EGly", "PEG200", "PEG400", "PEG600", "PEG1000", "PEG1500",
            "PEG3000", "PEG6000", "PEG12000", "PEG20000", "PEG35000"]
LABEL = {"EGly": "EG", "PEG200": "PEG 200", "PEG400": "PEG 400", "PEG600": "PEG 600",
         "PEG1000": "PEG 1k", "PEG1500": "PEG 1.5k", "PEG3000": "PEG 3k", "PEG6000": "PEG 6k",
         "PEG12000": "PEG 12k", "PEG20000": "PEG 20k", "PEG35000": "PEG 35k"}
LONG = ["PEG1000", "PEG1500", "PEG3000", "PEG6000", "PEG12000", "PEG20000", "PEG35000"]
CMAP = plt.get_cmap("viridis")
COLOR = {c: CMAP(i / (len(CROWDERS) - 1)) for i, c in enumerate(CROWDERS)}
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
                     "legend.fontsize": 7.5, "figure.dpi": 150, "savefig.bbox": "tight"})

# --------------------------------------------------------------------------------------
# Literature constants. Knowles et al. PNAS 108, 12699 (2011) + SI
# (literature/theoretical_models/Knowles2011_*). Sign convention there: melting.
# --------------------------------------------------------------------------------------
KN = {
    "m_EG_melt": (-0.340, 0.01),        # kcal/molal per monomer, EG-triEG duplex (p.12701, item 2)
    "m_PEG200_melt": (-0.21, 0.01),     # PEG 200 duplex monomer m-value (p.12701)
    "m_high_melt": (0.156, 0.020),      # high-MW plateau, linear fits (Table 1, p.12702)
    "m_high_quad_melt": (0.25, 0.05),   # quadratic initial slope (p.12704)
    "x_reversal": 6.0,                  # monomolal, Fig. 4 text (p.12703)
    "alpha_I": -2.21e-6,                # kcal molal^-1 A^-4 (p.12701)
    "ASA_I": 82.5,                      # A^2 (p.12700)
    "chi_inf": (0.45, 0.10),            # p.12703
    "ev_mon_melt": (0.309, 0.03),       # kcal/molal mon, duplex EV per monomer (Table 1)
    "dASA_12bp": 2006.0,                # A^2 duplex melting (Table S1)
    "T": 313.15,
    "mu34_PEG1000": (0.0378, 0.001),    # kcal/molal mon (Table S3)
    "mu34_PEG2000": (0.0324, 0.001),
    "mu34_sorbitol": (0.0082, 0.013), "mu34_mannitol": (-0.0015, 0.014),
    "mu34_xylitol": (0.0019, 0.007),
}
MU_X_KCL = 0.5 * (KN["mu34_PEG1000"][0] + KN["mu34_PEG2000"][0])  # kcal/molal mon
put("knEG", -KN["m_EG_melt"][0], "{:.2f}")
put("knPEGtwo", -KN["m_PEG200_melt"][0], "{:.2f}")
put("knHigh", -KN["m_high_melt"][0], "{:+.3f}")
put("knHighErr", KN["m_high_melt"][1], "{:.3f}")
put("knHighQuad", -KN["m_high_quad_melt"][0], "{:+.2f}")
put("knXrev", KN["x_reversal"], "{:.0f}")
put("knChiInf", KN["chi_inf"][0], "{:.2f}")
put("knAlphaI", KN["alpha_I"] * 1e6, "{:.2f}")
put("knASAI", KN["ASA_I"], "{:.1f}")
put("knDASA", KN["dASA_12bp"], "{:.0f}")
put("knEV", KN["ev_mon_melt"][0], "{:.3f}")
put("knMuOne", KN["mu34_PEG1000"][0], "{:.4f}")
put("knMuTwo", KN["mu34_PEG2000"][0], "{:.4f}")
put("knMuSorb", KN["mu34_sorbitol"][0], "{:.4f}")
put("knMuSorbErr", KN["mu34_sorbitol"][1], "{:.3f}")
put("knMuMann", KN["mu34_mannitol"][0], "{:.4f}")
put("knMuMannErr", KN["mu34_mannitol"][1], "{:.3f}")
put("knMuXyl", KN["mu34_xylitol"][0], "{:.4f}")
put("knMuXylErr", KN["mu34_xylitol"][1], "{:.3f}")
put("iseSlopeEG", 53.28, "{:.1f}")
put("iseSlopeBuf", 52.85, "{:.1f}")
ONE_PLUS_EPS = 0.92  # (1+eps)=1+dln(gamma)/dln(m), Davies estimate for 1:1 salt at 0.035 m
# Bielec 2022 SI (literature/pdf/jz1c03596_si_001.pdf), section 7 / Table S1
ISE = {"glycerol_shift_mV": 6.0, "slope_buffer_mV": 52.85, "slope_EG_mV": 53.28}

# --------------------------------------------------------------------------------------
# Repository data
# --------------------------------------------------------------------------------------
D = Data("source_data/MacromoleculeEquilibria_Crowding",
         "source_data/IonCrowderComplexation/raw_data",
         "source_data/IonStrengthDependency/Na_FRET.csv")
data = D.data.copy()
data["wt"] = data["crowder wt. [%]"].astype(float)
data = data[(data["wt"] <= 40) & data["crowder"].isin(CROWDERS)].copy()
data["K"] = data["K_uf [M]"].apply(lambda u: u.nominal_value)
data["sK"] = data["K_uf [M]"].apply(lambda u: u.std_dev)
k0row = data[data["wt"] == 0].iloc[0]
K0, sK0 = k0row["K"], k0row["sK"]
C0 = float(k0row["Na conc. [mM]"])  # mM, value used in the code
put("Kzero", K0 / 1e9, "{:.2f}")
put("Czero", C0, "{:.1f}")
data["ddG"] = RT * np.log(K0 / data["K"])  # kJ/mol, association, + = destabilisation
data["sddG"] = RT * data["sK"] / data["K"]  # point error only
sddG0 = RT * sK0 / K0
put("sddGzero", sddG0, "{:.2f}")

M_W = {c: float(data[data.crowder == c]["MW_[g/mol]"].iloc[0]) for c in CROWDERS}
N_MON = {c: (1.0 if c == "EGly" else float(data[data.crowder == c]["No_mono"].iloc[0])) for c in CROWDERS}


def monomolal(wt, crowder):
    w = np.asarray(wt, float) / 100.0
    mmon = 62.07 if crowder == "EGly" else 44.05
    return 1000.0 * w / ((1.0 - w) * mmon)


def wt_from_monomolal(x, crowder):
    mmon = 62.07 if crowder == "EGly" else 44.05
    w = x * mmon / (1000.0 + x * mmon)
    return 100 * w


def density(wt):  # code: source_data_processors/*: 0.997 + 0.0017441*wt
    return 0.997 + 0.0017441 * np.asarray(wt, float)


def molar_chain(wt, crowder):  # code definition, mol/L
    return np.asarray(wt, float) * density(wt) / 100 * 1000 / M_W[crowder]


data["x"] = [monomolal(w, c) for w, c in zip(data["wt"], data["crowder"])]

# ---------------- salt calibration (refit with full covariance) -----------------------
salt = D.SodiumDependencyObj.raw_data.copy()
cs = salt["C_Na+_[mM]"].values.astype(float)
Ks = salt["K_Na+_[1/M]"].values.astype(float)
sKs = salt["K_err_Na+_[1/M]"].values.astype(float)
Gs = -RT * np.log(Ks)
sGs = RT * sKs / Ks


def gcal(c, A, B, k):
    return A + B * np.exp(-k * np.sqrt(c))


popt, pcov = curve_fit(gcal, cs, Gs, sigma=sGs, absolute_sigma=True, p0=[-50, 50, 0.2])
code_popt = D.SodiumDependencyObj.popt_eff
assert np.allclose(popt, code_popt, rtol=1e-4), (popt, code_popt)
G = lambda c: gcal(c, *popt)  # noqa: E731


def SK_model(c, p=popt):
    A, B, k = p
    return B * k * np.sqrt(c) * np.exp(-k * np.sqrt(c)) / (2 * RT)


pars_mc = RNG.multivariate_normal(popt, pcov, N_MC)
SK0 = SK_model(C0)
SK0_err = np.std([SK_model(C0, p) for p in pars_mc])
pm("SKmodel", SK0, SK0_err)
sel = np.isin(np.round(cs, 2), [17.54, 35.08, 52.62])
lc, lk, slk = np.log(cs[sel]), np.log(Ks[sel]), sKs[sel] / Ks[sel]
pl, cl = np.polyfit(lc, lk, 1, w=1 / slk, cov="unscaled")
pm("SKlocal", pl[0], np.sqrt(cl[0, 0]))
SK_MODEL = SK0
SK0 = pl[0]  # data-driven local slope used in all linearised formulas (model value reported alongside)
put("SKhighc", SK_model(150.0))
put("GcalAtCzero", G(C0))
put("GfromKzero", -RT * np.log(K0))
results["calibration_popt"] = popt.tolist()

# ======================================================================================
# TEZA 1 - EG / PEG 200 do not bind Na+
# ======================================================================================
dosy = D.SodiumCrowderComplexationObj.raw_data.copy()
ana = D.SodiumCrowderComplexationObj.analyzed_data.copy()
# buffer row (wt% = 0) is identical in every raw file; the processor drops it
row0 = pd.read_csv("source_data/IonCrowderComplexation/raw_data/EGly.csv").query("`wt_%` == 0").iloc[0]
DNA0, sDNA0 = float(row0["D_Na_[um2/s]"]), float(row0["D_Na_err_[um2/s]"])
pm("DNazero", DNA0, sDNA0, "{:.0f}")


def exchange(c, bn, Dfree, Dcr):
    return (Dfree + bn * c * Dcr) / (1 + bn * c)


def fit_betan_selfdiff(df, DNa0, quad=False, perturb=False):
    w = df["wt_%"].values
    Dcr = df["D_crowder_[um2/s]"].values.copy()
    sDcr = df["D_crowder_err_[um2/s]"].values
    DNa = df["D_Na_[um2/s]"].values.copy()
    sDNa = df["D_Na_err_[um2/s]"].values
    if perturb:
        Dcr = Dcr + RNG.normal(0, sDcr)
        DNa = DNa + RNG.normal(0, sDNa)
    pcr = np.polyfit(w, Dcr, 2 if quad else 1, w=1 / sDcr)
    Dcr0 = np.polyval(pcr, 0.0)
    c = df["concentration [M]"].values
    Dfree = DNa0 * Dcr / Dcr0
    f = lambda cc, bn: exchange(cc, bn, Dfree, Dcr)  # noqa: E731
    p, cv = curve_fit(f, c, DNa, p0=[0.1], sigma=sDNa, absolute_sigma=True)
    return p[0], np.sqrt(cv[0, 0]), Dcr0


t1 = {}
for cr in ["EGly", "PEG200"]:
    df = ana[(ana.crowder == cr) & (ana["wt_%"] > 0)].sort_values("wt_%").copy()
    bn, bn_fit_err, Dcr0 = fit_betan_selfdiff(df, DNA0)
    mc = []
    for _ in range(N_MC):
        mc.append(fit_betan_selfdiff(df, RNG.normal(DNA0, sDNA0), perturb=True)[0])
    mc = np.array(mc)
    bn_q, _, _ = fit_betan_selfdiff(df, DNA0, quad=True)
    rNa = df["D_Na_[um2/s]"].values / DNA0
    rcr = df["D_crowder_[um2/s]"].values / Dcr0
    srNa = rNa * np.hypot(df["D_Na_err_[um2/s]"].values / df["D_Na_[um2/s]"].values, sDNA0 / DNA0)
    srcr = rcr * df["D_crowder_err_[um2/s]"].values / df["D_crowder_[um2/s]"].values
    ratio = rNa / rcr
    sratio = ratio * np.hypot(srNa / rNa, srcr / rcr)
    t1[cr] = dict(wt=df["wt_%"].values, rNa=rNa, srNa=srNa, rcr=rcr, srcr=srcr, ratio=ratio,
                  sratio=sratio, bn=bn, bn_mc=mc.std(), bn_q=bn_q,
                  bn_code=float(df["n Beta [1/M]"].iloc[0].nominal_value),
                  bn_code_err=float(df["n Beta [1/M]"].iloc[0].std_dev))
    tag = "EG" if cr == "EGly" else "PEGtwo"
    pm("bnSelf" + tag, bn, mc.std(), "{:.3f}")
    put("bnSelfQuad" + tag, bn_q, "{:.3f}")
    pm("bnCode" + tag, t1[cr]["bn_code"], t1[cr]["bn_code_err"], "{:.2f}")
    wmean = np.average(ratio, weights=1 / sratio ** 2)
    pm("ratioMean" + tag, wmean, 1 / np.sqrt(np.sum(1 / sratio ** 2)), "{:.3f}")
    put("ratioMin" + tag, ratio.min(), "{:.2f}")
    put("ratioMax" + tag, ratio.max(), "{:.2f}")
    put("bnUpper" + tag, bn + 2 * mc.std(), "{:.2f}")
    results["t1_" + cr] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in t1[cr].items()}

# required binding to explain EG at 40 wt% with the salt channel alone
eg40 = data[(data.crowder == "EGly") & (data.wt == 40)].iloc[0]
ddG_eg40, s_eg40 = eg40["ddG"], np.hypot(eg40["sddG"], sddG0)
pm("ddGEGforty", ddG_eg40, s_eg40, "{:.1f}")
c_eg40 = float(molar_chain(40, "EGly"))
put("cEGforty", c_eg40, "{:.2f}")
L = ddG_eg40 / RT
req = {}
# linearised SK
req["lin_cons"] = np.exp(2 * L / SK0)
req["lin_man"] = np.exp(L / SK0)
# full calibration curve
c_eq = brentq(lambda c: G(c) - (G(C0) + ddG_eg40), 0.05, C0)
put("cEqEG", c_eq, "{:.1f}")
req["full_cons"] = (C0 / c_eq) ** 2
req["full_man"] = C0 / c_eq
for key, val in req.items():
    tag = {"lin_cons": "LinCons", "lin_man": "LinMan", "full_cons": "FullCons", "full_man": "FullMan"}[key]
    put("req" + tag + "Factor", val, "{:.1f}")
    put("req" + tag + "Frac", 100 * (1 - 1 / val), "{:.0f}")
    put("req" + tag + "Bn", (val - 1) / c_eg40, "{:.2f}")
    put("req" + tag + "mV", 59.16 * np.log10(val), "{:.0f}")
# how much of EG destabilisation the DOSY values would explain (consistent version, full curve)


def ddg_salt_cons(bn, c):
    return G(C0 / np.sqrt(1 + bn * c)) - G(C0)


for tag, bnv in [("Code", t1["EGly"]["bn_code"]), ("Upper", t1["EGly"]["bn"] + 2 * t1["EGly"]["bn_mc"]),
                 ("Kappa", float(D.SodiumCrowderComplexationObj.n_complexation.nominal_value))]:
    v = ddg_salt_cons(max(bnv, 0.0), c_eg40)
    put("egExpl" + tag, v, "{:.1f}")
    put("egExplFrac" + tag, 100 * v / ddG_eg40, "{:.0f}")
put("kappaCode", float(D.SodiumCrowderComplexationObj.n_complexation.nominal_value), "{:.3f}")
put("ISEglycerol", ISE["glycerol_shift_mV"], "{:.0f}")

# Figure 1
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1.1, 1]})
ax = axs[0]
for cr, mk, col in [("EGly", "o", "C0"), ("PEG200", "s", "C1")]:
    d = t1[cr]
    ax.errorbar(d["wt"], d["rNa"], d["srNa"], fmt=mk, ms=4, color=col, label=f"{LABEL[cr]}: $D_\\mathrm{{Na}}/D_\\mathrm{{Na,0}}$")
    ax.errorbar(d["wt"], d["rcr"], d["srcr"], fmt=mk, ms=4, mfc="none", color=col, alpha=.8,
                label=f"{LABEL[cr]}: $D_\\mathrm{{cr}}/D_\\mathrm{{cr,0}}$")
ax.set_xlabel("crowder [wt%]")
ax.set_ylabel("względny współczynnik dyfuzji")
ax.set_title("Na$^+$ zwalnia tak samo jak sam crowder")
ax.legend(frameon=False, ncol=1)
ax = axs[1]
wts = sorted(set(np.round(t1["EGly"]["wt"], 1)) | set(np.round(t1["PEG200"]["wt"], 1)))
M = np.full((2, len(wts)), np.nan)
for i, cr in enumerate(["EGly", "PEG200"]):
    for w, r in zip(np.round(t1[cr]["wt"], 1), t1[cr]["ratio"]):
        M[i, wts.index(w)] = r
im = ax.imshow(M, cmap="RdBu_r", vmin=0.8, vmax=1.2, aspect="auto")
for i in range(2):
    for j in range(len(wts)):
        if np.isfinite(M[i, j]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7)
ax.set_xticks(range(len(wts)), [f"{w:g}" for w in wts], fontsize=7)
ax.set_yticks([0, 1], ["EG", "PEG 200"])
ax.set_xlabel("crowder [wt%]")
ax.set_title(r"$(D_\mathrm{Na}/D_\mathrm{Na,0})\,/\,(D_\mathrm{cr}/D_\mathrm{cr,0})$")
fig.colorbar(im, ax=ax, fraction=0.04)
fig.savefig(FIG / "t1_dosy_ratios.pdf")
plt.close(fig)

# Figure 2: required vs measured
fig, ax = plt.subplots(figsize=(7.0, 2.4))
items = [
    ("wymagane, wersja manuskryptu\n(pełna krzywa)", req["full_man"], "#c44e52"),
    ("wymagane, wersja spójna\n(pełna krzywa)", req["full_cons"], "#8c1c13"),
    ("DOSY, poprawka nanolepkości\n(Tabela S5)", 1 + t1["EGly"]["bn_code"] * c_eg40, "#4c72b0"),
    ("DOSY, sonda: samodyfuzja EG\n(górna granica 2$\\sigma$)", 1 + max(0, t1["EGly"]["bn"] + 2 * t1["EGly"]["bn_mc"]) * c_eg40, "#55a868"),
]
for i, (lab, val, col) in enumerate(items):
    ax.barh(i, np.log10(val), color=col)
    ax.text(np.log10(val) + 0.02, i, f"1+βn·c = {val:.2g}  ({100 * (1 - 1 / val):.0f}% Na$^+$ związane)", va="center", fontsize=7)
ax.set_yticks(range(len(items)), [i[0] for i in items], fontsize=7)
ax.invert_yaxis()
ax.set_xlabel(r"$\log_{10}(1+\beta n\,c)$ dla 40 wt% EG")
ax.set_xlim(0, max(np.log10(req["full_cons"]) * 1.45, 1))
ax.set_title("Wiązanie Na$^+$ potrzebne do wyjaśnienia ΔΔG(EG, 40 wt%) kanałem solnym vs zmierzone")
fig.savefig(FIG / "t1_required_binding.pdf")
plt.close(fig)

# ======================================================================================
# TEZA 3 part 1 (needed by TEZA 2): monomer m-values and smooth fits
# ======================================================================================


def zero_quad_fit(x, y, s, x_max=None):
    m = np.ones_like(x, bool) if x_max is None else (x <= x_max)
    X = np.vstack([x[m], x[m] ** 2]).T
    W = 1 / s[m] ** 2
    A = X.T @ (X * W[:, None])
    b = X.T @ (W * y[m])
    cov = np.linalg.inv(A)
    p = cov @ b
    chi2 = np.sum(W * (y[m] - X @ p) ** 2)
    dof = max(m.sum() - 2, 1)
    scale = max(1.0, chi2 / dof)
    return p, cov * scale, chi2 / dof


def zero_lin_fit(x, y, s, x_max):
    m = (x > 0) & (x <= x_max)
    W = 1 / s[m] ** 2
    a = np.sum(W * x[m] * y[m]) / np.sum(W * x[m] ** 2)
    var = 1 / np.sum(W * x[m] ** 2)
    chi2 = np.sum(W * (y[m] - a * x[m]) ** 2)
    dof = max(m.sum() - 1, 1)
    var *= max(1.0, chi2 / dof)
    # common K0 offset (correlated): slope shift = delta*sum(W x)/sum(W x^2)
    sys = sddG0 * np.sum(W * x[m]) / np.sum(W * x[m] ** 2)
    return a, np.sqrt(var + sys ** 2), m.sum(), chi2 / dof


mval = {}
for cr in CROWDERS:
    d = data[(data.crowder == cr)].sort_values("x")
    x, y, s = d["x"].values, d["ddG"].values / KCAL, d["sddG"].values / KCAL
    s = np.where(s > 0, s, sddG0 / KCAL)
    a, sa, npts, rchi = zero_lin_fit(x, y, s, 5.5)
    pq, cq, rq = zero_quad_fit(x[x > 0], y[x > 0], s[x > 0])
    mc = RNG.multivariate_normal(pq, cq, N_MC)
    xmin_mc = np.where(mc[:, 1] > 0, -mc[:, 0] / (2 * mc[:, 1]), np.nan)
    xmin = -pq[0] / (2 * pq[1]) if (pq[1] > 0 and pq[0] < 0) else np.nan
    mval[cr] = dict(m=a, sm=sa, n=npts, rchi=rchi, b1=pq[0], b2=pq[1], sb1=np.sqrt(cq[0, 0]),
                    sb2=np.sqrt(cq[1, 1]), cov=cq, xmin=xmin,
                    sxmin=np.nanstd(xmin_mc[(xmin_mc > 0) & (xmin_mc < 40)]) if np.isfinite(xmin) else np.nan,
                    x=x, y=y, s=s, N=N_MON[cr])
    tag = LABEL[cr].replace(" ", "").replace(".", "p")
    pm("m" + tag, a, sa, "{:+.3f}")
    if np.isfinite(xmin):
        pm("xmin" + tag, xmin, mval[cr]["sxmin"], "{:.1f}")
        put("wtmin" + tag, wt_from_monomolal(xmin, cr), "{:.0f}")
results["mvalues"] = {c: {k: (float(v) if np.isscalar(v) else None) for k, v in mval[c].items() if np.isscalar(v)} for c in CROWDERS}


def smooth_obs(cr, x):  # kJ/mol
    p = np.array([mval[cr]["b1"], mval[cr]["b2"]]) * KCAL
    return p[0] * x + p[1] * x ** 2


# ======================================================================================
# TEZA 2 - variants of the electrostatic term
# ======================================================================================
kappa_code = float(D.SodiumCrowderComplexationObj.n_complexation.nominal_value)


def elec_variants(cr, wt):
    wt = np.asarray(wt, float)
    c = molar_chain(wt, cr)
    N = N_MON[cr]
    bc = lambda k: k * N * c  # noqa: E731
    cdil = C0 * (1 - wt / 100) * density(wt) / 0.99707
    x = monomolal(wt, cr)
    out = {
        "a": G(cdil / (1 + bc(kappa_code))) - G(C0),
        "b": G(C0 / (1 + bc(kappa_code))) - G(C0),
        "c": G(C0 / np.sqrt(1 + bc(kappa_code))) - G(C0),
        "d1": G(C0 / np.sqrt(1 + bc(0.088))) - G(C0),
        "d2": G(C0 / np.sqrt(1 + bc(0.055))) - G(C0),
        "e": -(SK0 / (2 * ONE_PLUS_EPS)) * MU_X_KCL * KCAL * x,
    }
    return out


VLAB = {"a": "(a) obecny kod", "b": "(b) bez rozcieńczenia, $c_\\mathrm{eff}$",
        "c": "(c) $c_\\mathrm{eq}$, κ=0.168", "d1": "(d) $c_\\mathrm{eq}$, κ=0.088",
        "d2": "(d) $c_\\mathrm{eq}$, κ=0.055", "e": "(e) netto jak PEG–KCl"}
VCOL = {"a": "#c44e52", "b": "#dd8452", "c": "#8172b3", "d1": "#937860", "d2": "#da8bc3", "e": "#55a868"}

# check variant (a) reproduces the code
chk = data[data.crowder == "PEG35000"]
code_a = (chk["dG_Na_[kJ/mol]"].apply(lambda u: u.nominal_value) - G(C0)).values
mine_a = elec_variants("PEG35000", chk["wt"].values)["a"]
assert np.allclose(code_a, mine_a, atol=2e-3), (code_a, mine_a)

frac = {}
swing = {}
at40 = {}
for cr in LONG:
    x40 = monomolal(40, cr)
    xm = mval[cr]["xmin"]
    if not np.isfinite(xm):
        continue
    sw = smooth_obs(cr, x40) - smooth_obs(cr, xm)
    mc = RNG.multivariate_normal([mval[cr]["b1"], mval[cr]["b2"]], mval[cr]["cov"], N_MC) * KCAL
    xm_mc = -mc[:, 0] / (2 * mc[:, 1])
    ok = (mc[:, 1] > 0) & (xm_mc > 0) & (xm_mc < x40)
    sw_mc = (mc[ok, 0] * x40 + mc[ok, 1] * x40 ** 2) - (mc[ok, 0] * xm_mc[ok] + mc[ok, 1] * xm_mc[ok] ** 2)
    swing[cr] = (sw, sw_mc.std())
    wm = wt_from_monomolal(xm, cr)
    v40 = elec_variants(cr, 40)
    vm = elec_variants(cr, wm)
    frac[cr] = {k: (v40[k] - vm[k]) / sw for k in v40}
    at40[cr] = {k: float(v40[k]) for k in v40}
    tag = LABEL[cr].replace(" ", "").replace(".", "p")
    pm("swing" + tag, sw, sw_mc.std(), "{:.1f}")
for k in ["a", "b", "c", "d1", "d2", "e"]:
    put("at40" + k, at40["PEG35000"][k], "{:+.1f}")
    vals = [frac[c][k] for c in frac]
    put("fracMean" + k, 100 * np.mean(vals), "{:.0f}")
    put("fracMin" + k, 100 * np.min(vals), "{:.0f}")
    put("fracMax" + k, 100 * np.max(vals), "{:.0f}")
put("ddGobsFortyPEGthirtyfive", float(data[(data.crowder == "PEG35000") & (data.wt == 40)]["ddG"].iloc[0]), "{:+.1f}")
put("MUXKCL", MU_X_KCL, "{:.3f}")
put("onePlusEps", ONE_PLUS_EPS, "{:.2f}")
results["t2_fraction"] = frac
results["t2_at40"] = at40

# curvature: zero-intercept quadratic b2 for observed vs variants on the same x grid (PEG 35k + mean)
curv = {}
for cr in LONG:
    d = data[(data.crowder == cr) & (data.wt > 0)]
    x = d["x"].values
    vv = elec_variants(cr, d["wt"].values)
    cv = {"obs": mval[cr]["b2"] * KCAL}
    for k in ["a", "b", "c", "d1", "d2"]:
        X = np.vstack([x, x ** 2]).T
        cv[k] = np.linalg.lstsq(X, vv[k], rcond=None)[0][1]
    curv[cr] = cv
for k in ["obs", "a", "b", "c"]:
    put("curv" + k, curv["PEG35000"][k] * 1000, "{:+.1f}")  # J/mol per (monomolal)^2
put("curvObsSigma", mval["PEG35000"]["sb2"] * KCAL * 1000, "{:.1f}")
results["t2_curvature"] = curv

# Figure 3: variants vs wt% for PEG 6k and PEG 35k
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
wgrid = np.linspace(0, 40, 200)
for ax, cr in zip(axs, ["PEG6000", "PEG35000"]):
    vv = elec_variants(cr, wgrid)
    for k in ["a", "b", "c", "d1", "d2", "e"]:
        ax.plot(wgrid, vv[k], color=VCOL[k], lw=1.4, ls="--" if k == "e" else "-", label=VLAB[k])
    d = data[data.crowder == cr]
    ax.errorbar(d["wt"], d["ddG"], d["sddG"], fmt="ko", ms=3.5, label="ΔΔG obserwowane")
    ax.plot(wgrid, smooth_obs(cr, monomolal(wgrid, cr)), "k:", lw=1)
    ax.axhline(0, color="grey", lw=.6)
    ax.set_title(LABEL[cr])
    ax.set_xlabel("PEG [wt%]")
axs[0].set_ylabel(r"$\Delta\Delta G$ [kJ mol$^{-1}$] (asocjacja)")
axs[1].legend(frameon=False, fontsize=6.5, loc="upper left")
fig.suptitle("Człon solny w wariantach vs obserwowane ΔΔG", y=1.01)
fig.savefig(FIG / "t2_variants.pdf")
plt.close(fig)

# Figure 4: heatmap fraction of swing explained
crs = [c for c in LONG if c in frac]
keys = ["a", "b", "c", "d1", "d2", "e"]
Mh = np.array([[100 * frac[c][k] for c in crs] for k in keys])
fig, ax = plt.subplots(figsize=(7.0, 2.8))
im = ax.imshow(Mh, cmap="RdBu_r", vmin=-150, vmax=150, aspect="auto")
for i in range(len(keys)):
    for j in range(len(crs)):
        ax.text(j, i, f"{Mh[i, j]:.0f}%", ha="center", va="center", fontsize=7.5)
ax.set_xticks(range(len(crs)), [f"{LABEL[c]}\n(Δ={swing[c][0]:.1f}±{swing[c][1]:.1f})" for c in crs], fontsize=7)
ax.set_yticks(range(len(keys)), [VLAB[k] for k in keys], fontsize=7.5)
ax.set_title("Ułamek wahnięcia odwrócenia [ΔΔG(40 wt%) − ΔΔG(min), kJ/mol] wyjaśniany przez człon solny")
fig.colorbar(im, ax=ax, fraction=0.03, label="%")
fig.savefig(FIG / "t2_fraction_heatmap.pdf")
plt.close(fig)

# salt-series SK(c) figure for degeneracy argument
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.5))
cg = np.logspace(np.log10(3), np.log10(2000), 200)
ax = axs[0]
ax.errorbar(cs, np.log(Ks), sKs / Ks, fmt="ko", ms=3.5, label="seria Na$^+$ (Na_FRET.csv, kalibracja)")
si_rows = []
for line in open(REPO / "manuscript/SI/main.tex", encoding="utf8"):
    parts = [q.strip() for q in line.split("&")]
    if len(parts) == 6 and parts[3].replace(".", "").isdigit():
        mant = parts[4].replace("$", "").replace("\\mathrm{e}^{", "e").replace("}", "").replace(" ", "")
        try:
            si_rows.append((float(parts[3]), float(mant)))
        except ValueError:
            pass
si_rows = np.array([r for r in si_rows if r[0] > 176])
results["SI_salt_beyond"] = si_rows.tolist()
if len(si_rows):
    ax.plot(si_rows[:, 0], np.log(si_rows[:, 1]), "o", mfc="none", mec="C3", ms=4,
            label="SI Tab. S3, >175 mM (poza kalibracją)")
    put("nSIbeyond", len(si_rows), "{:d}")
ax.plot(cg, -G(cg) / RT, "C0-", lw=1.2, label="kalibracja $A+Be^{-k\\sqrt{c}}$")
ax.set_xscale("log")
ax.set_xlabel("$c_\\mathrm{Na}$ [mM]")
ax.set_ylabel("ln K")
ax.axvline(C0, color="grey", lw=.6, ls=":")
ax.legend(frameon=False)
ax = axs[1]
sk_band = np.array([SK_model(cg, p) for p in pars_mc[:600]])
ax.fill_between(cg, np.percentile(sk_band, 16, 0), np.percentile(sk_band, 84, 0), color="C0", alpha=.3)
ax.plot(cg, SK_model(cg), "C0-", lw=1.2)
ax.axvline(C0, color="grey", lw=.6, ls=":")
ax.set_xscale("log")
ax.set_xlabel("$c_\\mathrm{Na}$ [mM]")
ax.set_ylabel("SK = d ln K / d ln c")
ax.set_title("SK(c) z kalibracji: maleje powyżej ~20 mM", fontsize=8)
fig.savefig(FIG / "t2_SK.pdf")
plt.close(fig)

# ======================================================================================
# TEZA 3 - Knowles framework
# ======================================================================================
# Figure 5: ddG vs monomolal, all crowders
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
for ax, group in zip(axs, [["EGly", "PEG200", "PEG400", "PEG600"], LONG]):
    for cr in group:
        d = data[data.crowder == cr]
        ax.errorbar(d["x"], d["ddG"], d["sddG"], fmt="o-", ms=3, lw=.8, color=COLOR[cr], label=LABEL[cr])
    ax.axhline(0, color="grey", lw=.6)
    ax.axvline(KN["x_reversal"], color="k", lw=.7, ls=":")
    ax.set_xlabel("x [mol EO / kg wody]")
axs[0].set_ylabel(r"$\Delta\Delta G$ [kJ mol$^{-1}$]")
axs[0].set_title("krótkie: destabilizacja ∝ x")
axs[1].set_title("długie: wspólna krzywa, odwrócenie przy x≈6 (linia: Knowles)")
for ax in axs:
    ax.legend(frameon=False, ncol=2)
fig.savefig(FIG / "t3_monomolal.pdf")
plt.close(fig)

# null model
dASA13 = KN["dASA_12bp"] * 13 / 12
P_K = KN["alpha_I"] * KN["ASA_I"] * dASA13  # kcal/molal mon, melting, chi=1
E_K = KN["ev_mon_melt"][0] * (T / KN["T"]) * 13 / 12
put("PKnowles", -P_K, "{:+.3f}")
put("EKnowles", -E_K, "{:+.3f}")
put("plateauKnowlesPred", -(KN["chi_inf"][0] * P_K + E_K), "{:+.3f}")


def null_model(N, P, E, Nh, chi=KN["chi_inf"][0], h=2.0):
    s = 1 / (1 + (N / Nh) ** h)
    return -((chi + (1 - chi) * s) * P + E * (1 - s))  # association sign


Ns = np.array([mval[c]["N"] for c in CROWDERS])
ms = np.array([mval[c]["m"] for c in CROWDERS])
sms = np.array([mval[c]["sm"] for c in CROWDERS])
pn, cn = curve_fit(lambda N, P, E, Nh: null_model(N, P, E, Nh), Ns, ms, p0=[P_K, E_K, 13.0],
                   sigma=sms, absolute_sigma=True, maxfev=20000)
res_null = (ms - null_model(Ns, *pn)) / sms
chi2n = np.sum(res_null ** 2) / (len(ms) - 3)
pm("nullP", -pn[0], np.sqrt(cn[0, 0]), "{:+.3f}")
pm("nullE", -pn[1], np.sqrt(cn[1, 1]), "{:+.3f}")
pm("nullNh", pn[2], np.sqrt(cn[2, 2]), "{:.0f}")
put("nullChi", chi2n, "{:.1f}")
put("nullMaxRes", np.max(np.abs(res_null)), "{:.1f}")
pred_res = (ms - null_model(Ns, P_K, E_K, 13.4)) / sms
put("predChi", np.sum(pred_res ** 2) / len(ms), "{:.1f}")
results["null_fit"] = {"popt": pn.tolist(), "res": res_null.tolist()}

# salt term with EV fixed to Knowles value
S_dosy = {k: (SK0 / (2 * ONE_PLUS_EPS)) * RT / KCAL * k for k in [kappa_code, 0.088, 0.055]}  # kcal/molal mon destab


def null_salt(N, P, S, Nh, h=2.0, chi=KN["chi_inf"][0]):
    s = 1 / (1 + (N / Nh) ** h)
    return -((chi + (1 - chi) * s) * P + E_K * (1 - s)) + S * (1 - s)


ps, cs_ = curve_fit(null_salt, Ns, ms, p0=[P_K, 0.0, 13.0], sigma=sms, absolute_sigma=True, maxfev=20000)
chi2s = np.sum(((ms - null_salt(Ns, *ps)) / sms) ** 2) / (len(ms) - 3)
pm("saltS", ps[1], np.sqrt(cs_[1, 1]), "{:+.3f}")
put("saltChi", chi2s, "{:.1f}")
for k, v in S_dosy.items():
    put("Sdosy" + {kappa_code: "A", 0.088: "B", 0.055: "C"}[k], v, "{:+.3f}")

# Figure 6: m/N vs N
fig, ax = plt.subplots(figsize=(6.0, 3.0))
ax.errorbar(Ns, ms, sms, fmt="o", color="k", ms=4, label="nasze dane (0–5.5 monomolal)")
for c, N, m in zip(CROWDERS, Ns, ms):
    ax.annotate(LABEL[c], (N, m), textcoords="offset points", xytext=(4, 3), fontsize=6)
ax.errorbar([1, 2, 3], [-KN["m_EG_melt"][0]] * 3, [KN["m_EG_melt"][1]] * 3, fmt="s", mfc="none", color="C3",
            label="Knowles: EG, diEG, triEG")
ax.errorbar([4.0], [-KN["m_PEG200_melt"][0]], [KN["m_PEG200_melt"][1]], fmt="s", mfc="none", color="C3")
ax.fill_between([45, 454], -KN["m_high_melt"][0] - KN["m_high_melt"][1], -KN["m_high_melt"][0] + KN["m_high_melt"][1],
                color="C3", alpha=.25, label="Knowles: plateau (liniowo)")
ax.hlines(-KN["m_high_quad_melt"][0], 45, 454, color="C3", ls="--", lw=1, label="Knowles: plateau (kwadratowo)")
Ng = np.logspace(0, 3, 200)
ax.plot(Ng, null_model(Ng, *pn), "C0-", label="model zerowy, dopasowany")
ax.plot(Ng, null_model(Ng, P_K, E_K, 13.4), "C0:", label="model zerowy, parametry Knowlesa")
ax.axhline(0, color="grey", lw=.6)
ax.set_xscale("log")
ax.set_xlabel("N (liczba jednostek EO)")
ax.set_ylabel("m/N [kcal mol$^{-1}$ (molal mon.)$^{-1}$]\n(+ = destabilizacja asocjacji)")
ax.legend(frameon=False, fontsize=6.5, ncol=2)
fig.savefig(FIG / "t3_mN.pdf")
plt.close(fig)

# broken-slope test of the Knowles reversal picture for long PEGs


def broken(x, s1, s2, xs, w=1.0):
    return s1 * x + (s2 - s1) * w * np.log1p(np.exp((x - xs) / w))


bro = {}
for cr in ["PEG3000", "PEG6000", "PEG12000", "PEG20000", "PEG35000"]:
    d = data[data.crowder == cr]
    x, y, s = d["x"].values, d["ddG"].values / KCAL, np.where(d["sddG"].values > 0, d["sddG"].values, sddG0) / KCAL
    y = y - broken(0, 0, 0, 6)  # zero at x=0 by construction (broken(0)=w*log(1+e^-xs/w)*(s2-s1))
    p, cv = curve_fit(lambda xx, s1, s2, xs: broken(xx, s1, s2, xs) - broken(0, s1, s2, xs), x, y,
                      p0=[-0.25, 0.2, 6.0], sigma=s, absolute_sigma=True, maxfev=20000)
    rchi = np.sum(((y - (broken(x, *p) - broken(0, *p))) / s) ** 2) / max(len(x) - 3, 1)
    cv = cv * max(1, rchi)
    bro[cr] = dict(s1=p[0], s2=p[1], xs=p[2], ss1=np.sqrt(cv[0, 0]), ss2=np.sqrt(cv[1, 1]), sxs=np.sqrt(cv[2, 2]))
bro6 = {}
for cr in bro:
    d = data[data.crowder == cr]
    x, y, s = d["x"].values, d["ddG"].values / KCAL, np.where(d["sddG"].values > 0, d["sddG"].values, sddG0) / KCAL
    f6 = lambda xx, s1, s2: broken(xx, s1, s2, KN["x_reversal"]) - broken(0, s1, s2, KN["x_reversal"])  # noqa: E731
    p6, c6 = curve_fit(f6, x, y, p0=[-0.25, 0.2], sigma=s, absolute_sigma=True)
    chi6 = np.sum(((y - f6(x, *p6)) / s) ** 2)
    chifree = np.sum(((y - (broken(x, bro[cr]["s1"], bro[cr]["s2"], bro[cr]["xs"]) - broken(0, bro[cr]["s1"], bro[cr]["s2"], bro[cr]["xs"]))) / s) ** 2)
    xm_ = x > 0
    chiq = np.sum(((y[xm_] - (mval[cr]["b1"] * x[xm_] + mval[cr]["b2"] * x[xm_] ** 2)) / s[xm_]) ** 2)
    r6 = chi6 / max(len(x) - 2, 1)
    bro6[cr] = dict(s1=p6[0], s2=p6[1], ss1=np.sqrt(c6[0, 0] * max(1, r6)), ss2=np.sqrt(c6[1, 1] * max(1, r6)),
                    chi6=chi6, chifree=chifree, chiq=chiq, n=len(x))
results["broken6"] = bro6
s26 = np.array([bro6[c]["s2"] for c in bro6]); ss26 = np.array([bro6[c]["ss2"] for c in bro6])
s16 = np.array([bro6[c]["s1"] for c in bro6]); ss16 = np.array([bro6[c]["ss1"] for c in bro6])
put("chiSixTot", sum(bro6[c]["chi6"] for c in bro6), "{:.0f}")
put("chiFreeTot", sum(bro6[c]["chifree"] for c in bro6), "{:.0f}")
put("chiQuadTot", sum(bro6[c]["chiq"] for c in bro6), "{:.0f}")
put("nBroTot", sum(bro6[c]["n"] for c in bro6), "{:.0f}")
w_s2 = np.array([bro[c]["s2"] for c in bro])
w_ss2 = np.array([bro[c]["ss2"] for c in bro])
w_xs = np.array([bro[c]["xs"] for c in bro])
w_sxs = np.array([bro[c]["sxs"] for c in bro])
w_s1 = np.array([bro[c]["s1"] for c in bro])
w_ss1 = np.array([bro[c]["ss1"] for c in bro])
wm = lambda v, s: (np.sum(v / s ** 2) / np.sum(1 / s ** 2), 1 / np.sqrt(np.sum(1 / s ** 2)))  # noqa: E731
pm("broS1", *wm(w_s1, w_ss1), "{:+.3f}")
pm("broS2", *wm(w_s2, w_ss2), "{:+.3f}")
put("broXsMin", w_xs.min(), "{:.1f}")
put("broXsMax", w_xs.max(), "{:.1f}")
pm("broSixSone", *wm(s16, ss16), "{:+.3f}")
pm("broSixStwo", *wm(s26, ss26), "{:+.3f}")
# PEG 200 slope over full range for comparison
d = data[data.crowder == "PEG200"]
xx, yy, ss = d["x"].values, d["ddG"].values / KCAL, np.where(d["sddG"].values > 0, d["sddG"].values, sddG0) / KCAL
a200, sa200, _, _ = zero_lin_fit(xx, yy, ss, 100)
pm("mPEGtwoFull", a200, sa200, "{:+.3f}")
results["broken"] = bro

# M-independence above ~600 Da
grp = ["PEG600", "PEG1000", "PEG1500", "PEG3000", "PEG6000", "PEG12000", "PEG20000", "PEG35000"]
b1 = np.array([mval[c]["b1"] for c in grp])
sb1 = np.array([mval[c]["sb1"] for c in grp])
b2 = np.array([mval[c]["b2"] for c in grp])
sb2 = np.array([mval[c]["sb2"] for c in grp])
mb1, smb1 = wm(b1, sb1)
mb2, smb2 = wm(b2, sb2)
chi_b1 = np.sum(((b1 - mb1) / sb1) ** 2) / (len(grp) - 1)
chi_b2 = np.sum(((b2 - mb2) / sb2) ** 2) / (len(grp) - 1)
pm("bOneMean", mb1, smb1, "{:+.3f}")
pm("bTwoMean", mb2 * 1000, smb2 * 1000, "{:+.1f}")
put("bOneChi", chi_b1, "{:.1f}")
put("bTwoChi", chi_b2, "{:.1f}")
put("bOneCV", 100 * np.std(b1) / abs(np.mean(b1)), "{:.0f}")
put("bTwoCV", 100 * np.std(b2) / abs(np.mean(b2)), "{:.0f}")
# joint vs separate (F-test)
from scipy.stats import f as fdist  # noqa: E402

Xall, Yall, Sall, gid = [], [], [], []
for i, c in enumerate(grp):
    xm_ = mval[c]["x"] > 0
    Xall.append(mval[c]["x"][xm_]); Yall.append(mval[c]["y"][xm_]); Sall.append(mval[c]["s"][xm_]); gid += [i] * xm_.sum()
Xall, Yall, Sall = map(np.concatenate, (Xall, Yall, Sall))
Wj = 1 / Sall ** 2
Xj = np.vstack([Xall, Xall ** 2]).T
pj = np.linalg.solve(Xj.T @ (Xj * Wj[:, None]), Xj.T @ (Wj * Yall))
chi_joint = np.sum(Wj * (Yall - Xj @ pj) ** 2)
chi_sep = 0.0
for c in grp:
    xm_ = mval[c]["x"] > 0
    chi_sep += np.sum(((mval[c]["y"][xm_] - (mval[c]["b1"] * mval[c]["x"][xm_] + mval[c]["b2"] * mval[c]["x"][xm_] ** 2)) / mval[c]["s"][xm_]) ** 2)
n_tot, k_sep, k_joint = len(Yall), 2 * len(grp), 2
Fst = ((chi_joint - chi_sep) / (k_sep - k_joint)) / (chi_sep / (n_tot - k_sep))
pval = 1 - fdist.cdf(Fst, k_sep - k_joint, n_tot - k_sep)
put("Fstat", Fst, "{:.2f}")
put("Fpval", pval, "{:.2f}")
put("chiSepRed", chi_sep / (n_tot - k_sep), "{:.1f}")
put("chiJointRed", chi_joint / (n_tot - k_joint), "{:.1f}")

def ftest(group):
    X_, Y_, S_ = [], [], []
    chs = 0.0
    for c in group:
        xm_ = mval[c]["x"] > 0
        X_.append(mval[c]["x"][xm_]); Y_.append(mval[c]["y"][xm_]); S_.append(mval[c]["s"][xm_])
        chs += np.sum(((mval[c]["y"][xm_] - (mval[c]["b1"] * mval[c]["x"][xm_] + mval[c]["b2"] * mval[c]["x"][xm_] ** 2)) / mval[c]["s"][xm_]) ** 2)
    X_, Y_, S_ = map(np.concatenate, (X_, Y_, S_))
    W_ = 1 / S_ ** 2
    XX = np.vstack([X_, X_ ** 2]).T
    pp = np.linalg.solve(XX.T @ (XX * W_[:, None]), XX.T @ (W_ * Y_))
    chj = np.sum(W_ * (Y_ - XX @ pp) ** 2)
    n_, ks = len(Y_), 2 * len(group)
    F_ = ((chj - chs) / (ks - 2)) / (chs / (n_ - ks))
    return F_, 1 - fdist.cdf(F_, ks - 2, n_ - ks), pp


F1k, p1k, pj1k = ftest(grp[1:])
put("FstatOnek", F1k, "{:.2f}")
put("FpvalOnek", p1k, "{:.3f}")
b1_1k = np.array([mval[c]["b1"] for c in grp[1:]])
put("bOneCVOnek", 100 * np.std(b1_1k) / abs(np.mean(b1_1k)), "{:.0f}")
# typical spread in kJ/mol at 20 wt% between long PEGs (from smooth fits)
spread20 = [smooth_obs(c, monomolal(20, c)) for c in grp[1:]]
put("spreadTwenty", np.max(spread20) - np.min(spread20), "{:.1f}")
put("meanTwenty", np.mean(spread20), "{:.1f}")

# Figure 7: coefficients vs N + broken-slope summary
fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.5))
Nall = np.array([mval[c]["N"] for c in CROWDERS])
for ax, key, skey, lab, sc in [(axs[0], "b1", "sb1", "$b_1$ [kcal/(mol·molal)]", 1),
                               (axs[1], "b2", "sb2", "$b_2$ [cal/(mol·molal²)]", 1000)]:
    v = np.array([mval[c][key] for c in CROWDERS]) * sc
    sv = np.array([mval[c][skey] for c in CROWDERS]) * sc
    ax.errorbar(Nall, v, sv, fmt="o", color="k", ms=3.5)
    mm, sm_ = (mb1, smb1) if key == "b1" else (mb2 * 1000, smb2 * 1000)
    ax.axhspan(mm - sm_, mm + sm_, color="C0", alpha=.3)
    ax.axhline(0, color="grey", lw=.6)
    ax.axvline(N_MON["PEG600"], color="grey", ls=":", lw=.7)
    ax.set_xscale("log"); ax.set_xlabel("N"); ax.set_ylabel(lab)
axs[0].set_title("ΔΔG = $b_1x+b_2x^2$: stałe dla N ≥ 13")
ax = axs[2]
names = list(bro6)
yy_ = np.arange(len(names))
ax.errorbar(s16, yy_ + .15, xerr=ss16, fmt="o", color="C2", ms=3.5, label="$s_1$ (x*=6)")
ax.errorbar(s26, yy_, xerr=ss26, fmt="s", color="C3", ms=3.5, label="$s_2$ (x*=6)")
ax.errorbar(w_s2, yy_ - .15, xerr=w_ss2, fmt="^", color="C1", ms=3.5, label="$s_2$ (x* swobodne)")
ax.axvspan(a200 - sa200, a200 + sa200, color="C3", alpha=.25, label="PEG 200 (nasze)")
ax.axvline(0, color="grey", lw=.6)
ax.set_xlim(-0.45, 0.4)
ax.set_yticks(yy_, [LABEL[c] for c in names], fontsize=7)
ax.set_xlabel("nachylenie [kcal/(mol·molal)]")
ax.set_title("po odwróceniu ≈ PEG 200?")
ax.legend(frameon=False, fontsize=5.5, loc="lower left")
fig.tight_layout()
fig.savefig(FIG / "t3_coeffs_broken.pdf")
plt.close(fig)

# ======================================================================================
# TEZA 4 - V_eff collapse with constant residual (reproduces figures_publication.ipynb, cell 8)
# ======================================================================================
V_, alpha_ = 0.825, 0.49
pdat = D.data.copy()
pdat = pdat[(pdat["crowder wt. [%]"] < 50) & (pdat["molar mass [g/mol]"] >= 3000) & (pdat["concentration [M]"] > 0)].copy()
cstarN = pdat["No_mono"].astype(float) ** (-4 / 5) / V_
xr = pdat["mass_concentration_[g/mL]"].astype(float) / cstarN
pi = 8.314 * T / (44 * V_) * (xr + alpha_ * xr ** (9 / 4)) / pdat["No_mono"].astype(float) ** (9 / 5) * 1e6
rest = (pdat["dG_exp_[kJ/mol]"].apply(lambda u: u.nominal_value) - pdat["dG_Na_[kJ/mol]"].apply(lambda u: u.nominal_value))
xi = pdat["Rg_[nm]"] * pdat["c/c*"] ** (-0.77)
ssRg = D.ssDNA_dimensions.ssDNA_parameters["Rg_nm"]


def collapse_y(rest_vals):
    veff = -rest_vals * 1000 / pi * 1e27 / 6.02214076e23
    return veff / (pdat["Rg_[nm]"] * xi * ssRg)


coll = {}
xc = pdat["c/c*"].values.astype(float)
for tag, rv in [("real", rest),
                ("constPEG", pdat.assign(r=rest).groupby("crowder")["r"].transform("mean")),
                ("constAll", pd.Series(rest.mean(), index=rest.index))]:
    yv = collapse_y(rv).values.astype(float)
    m = (xc > 0) & (yv > 0) & np.isfinite(yv)
    pfit = np.polyfit(np.log10(xc[m]), np.log10(yv[m]), 1)
    resid = np.log10(yv[m]) - np.polyval(pfit, np.log10(xc[m]))
    r2 = 1 - np.sum(resid ** 2) / np.sum((np.log10(yv[m]) - np.log10(yv[m]).mean()) ** 2)
    coll[tag] = dict(alpha=pfit[0], A=10 ** pfit[1], r2=r2, x=xc[m], y=yv[m], crowder=pdat["crowder"].values[m])
    put("coll" + tag + "Alpha", pfit[0], "{:.2f}")
    put("coll" + tag + "Rtwo", r2, "{:.2f}")
put("restMean", rest.mean(), "{:.1f}")
put("restSD", rest.std(), "{:.1f}")

fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
for ax, tag, title in [(axs[0], "real", "ΔG$_\\mathrm{rest}$ z danych (Fig. 4f)"),
                       (axs[1], "constPEG", "ΔG$_\\mathrm{rest}$ = stała (średnia dla każdego PEG)")]:
    cc = coll[tag]
    for cr in ["PEG3000", "PEG6000", "PEG12000", "PEG20000", "PEG35000"]:
        mm = cc["crowder"] == cr
        ax.loglog(cc["x"][mm], cc["y"][mm], "o", ms=3.5, color=COLOR[cr], label=LABEL[cr])
    xl = np.logspace(np.log10(cc["x"].min()), np.log10(cc["x"].max()), 50)
    ax.loglog(xl, cc["A"] * xl ** cc["alpha"], "k--", lw=1, label=f"nachylenie {cc['alpha']:.2f}, R²={cc['r2']:.2f}")
    ax.set_title(title, fontsize=8.5)
    ax.set_xlabel("c/c*")
axs[0].set_ylabel(r"$V_\mathrm{eff}/(R_g^\mathrm{PEG}\,\xi\,R_g^\mathrm{ssDNA})$")
axs[1].legend(frameon=False, fontsize=6.5)
fig.savefig(FIG / "t4_collapse.pdf")
plt.close(fig)

# --------------------------------------------------------------------------------------
with open(HERE / "numbers.tex", "w") as fh:
    fh.write("% generated by analysis.py - do not edit\n")
    for k, v in sorted(numbers.items()):
        words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        safe = "".join(ch if ch.isalpha() else (words[int(ch)] if ch.isdigit() else "") for ch in k)
        fh.write(f"\\newcommand{{\\N{safe}}}{{{v}}}\n")


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


with open(HERE / "results.json", "w") as fh:
    json.dump(_clean(results), fh, indent=1, default=str)
print("OK", len(numbers), "numbers")
for k in sorted(numbers):
    print(k, numbers[k])
