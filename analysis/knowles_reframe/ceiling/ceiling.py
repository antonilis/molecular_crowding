"""Is the stabilisation branch of long PEGs at the ceiling of K determinability?

Run:  .venv/bin/python analysis/knowles_reframe/ceiling/ceiling.py
Reads repository data and the original fitting script
(source_data/MacromoleculeEquilibria_Crowding/our_raw_results/ssDNA_hybdrization_equillibrium_constant.py)
without modifying them. Writes figures/, numbers.tex and results.json next to this file.
"""
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lmfit import Model
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import norm, spearmanr

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO))
from main_data_processor import Data  # noqa: E402

FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
RNG = np.random.default_rng(20261010)
R, T = 8.314462618e-3, 298.15
RT = R * T
KCAL = 4.184
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "legend.fontsize": 7.5,
                     "figure.dpi": 150, "savefig.bbox": "tight"})

numbers, results = {}, {}


def put(name, value, fmt="{:.2f}"):
    numbers[name] = value if isinstance(value, str) else fmt.format(value)
    results[name] = value if isinstance(value, (str, int)) else float(value)


def pm(name, v, e, fmt="{:.2f}"):
    put(name, v, fmt)
    put(name + "Err", abs(e), fmt.replace("+", ""))


# ---------------- original fitting script --------------------------------------------
SCRIPT = REPO / "source_data/MacromoleculeEquilibria_Crowding/our_raw_results/ssDNA_hybdrization_equillibrium_constant.py"
spec = importlib.util.spec_from_file_location("kfit", SCRIPT)
kfit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kfit)
TDIR = "DNA-DNA_countrate_vs_crowder_concentration"
df_RI = kfit.calculate_RI_slope_with_error()


def lm_fit(x, y, a, g, v0):
    """Identical set-up to kfit.fit_single_dataset, but returns the lmfit result."""
    model = Model(lambda x, K, D: kfit.K_binding_model(x, K, D, a, g, v0))
    model.set_param_hint("K", value=1e6, min=1e2, max=5e13)
    model.set_param_hint("D", value=1e-8, min=1.1e-9, max=2e-8)
    return model.fit(y, x=x)


def model_vec(x, K, D, a, g, v0):
    term = x + D + 1.0 / K
    delta = np.clip(term ** 2 - 4 * x * D, 0, None)
    b = 0.5 * (term - np.sqrt(delta))
    return a * v0 * (D - b) * (1 + (g / a) * K * (x - b))


titr = []
for base, name, val in kfit.extract_metadata(TDIR):
    d = kfit.load_dataset(os.path.join(TDIR, f"{base}.csv"))
    val = float(val)
    RI = kfit.compute_refractive_index(val, name, df_RI)
    a, g, v0, C = kfit.compute_physical_parameters(d, val, RI)
    x = (d["ratio"] * C).values
    y = d["value"].values.astype(float)
    r = lm_fit(x, y, a, g, v0)
    ref = kfit.fit_single_dataset(d, a, g, v0, C)
    assert np.isclose(r.params["K"].value, ref["K"], rtol=1e-6)
    n = len(y)
    ssr = float(np.sum(r.residual ** 2))
    titr.append(dict(base=base, crowder=name, wt=val, a=a, g=g, v0=v0, C=C, x=x, y=y, df=d, n=n,
                     K=r.params["K"].value, Kerr=r.params["K"].stderr or np.nan, D=r.params["D"].value,
                     ssr=ssr, sigma=np.sqrt(ssr / (n - 2))))
titr.sort(key=lambda t: (t["crowder"], t["wt"]))
print(len(titr), "titrations refitted with the original script")
put("nTitr", len(titr), "{:d}")

D = Data("source_data/MacromoleculeEquilibria_Crowding", "source_data/IonCrowderComplexation/raw_data",
         "source_data/IonStrengthDependency/Na_FRET.csv")
pipe = D.data.copy()
K0 = 1.76e9
put("Kzero", K0 / 1e9)

LABEL = {"EGly": "EG", "PEG200": "PEG 200", "PEG400": "PEG 400", "PEG600": "PEG 600", "PEG1000": "PEG 1k",
         "PEG1500": "PEG 1.5k", "PEG3000": "PEG 3k", "PEG6000": "PEG 6k", "PEG12000": "PEG 12k",
         "PEG20000": "PEG 20k", "PEG35000": "PEG 35k"}
ORDER = list(LABEL)
NMON = {c: (1.0 if c == "EGly" else float(pipe[pipe.crowder == c]["No_mono"].iloc[0])) for c in ORDER}
LONG = ["PEG1000", "PEG1500", "PEG3000", "PEG6000", "PEG12000", "PEG20000", "PEG35000"]
N13 = ["PEG600"] + LONG


def monomolal(wt, cr):
    w = np.asarray(wt, float) / 100
    return 1000 * w / ((1 - w) * (62.07 if cr == "EGly" else 44.05))


# ======================================================================================
# 1. distribution of K and K*C_donor
# ======================================================================================
tab = pd.DataFrame([{k: t[k] for k in ("crowder", "wt", "K", "Kerr", "C", "D", "sigma", "n")} for t in titr])
tab["KC"] = tab["K"] * tab["C"]
tab = tab[tab.wt <= 40].copy()
tab["stab"] = tab["K"] > K0
put("nKCtwenty", int((tab.KC > 20).sum()), "{:d}")
put("nAll", len(tab), "{:d}")
put("KCmax", tab.KC.max(), "{:.0f}")
put("Kmax", tab.K.max() / 1e10, "{:.2f}")
lng = tab[tab.crowder.isin(LONG) & tab.stab]
put("nStabLong", len(lng), "{:d}")
put("nStabLongKC", int((lng.KC > 20).sum()), "{:d}")
lk = np.log10(lng.K)
put("logKStabMean", lk.mean(), "{:.2f}")
put("logKStabSD", lk.std(), "{:.2f}")
put("logKStabErrMed", np.median(lng.Kerr / lng.K / np.log(10)), "{:.2f}")
rho, prho = spearmanr(lng.wt, lk)
put("rhoWt", rho, "{:+.2f}")
put("prhoWt", prho, "{:.2f}")
rhoN, prhoN = spearmanr([NMON[c] for c in lng.crowder], lk)
put("rhoN", rhoN, "{:+.2f}")
put("prhoN", prhoN, "{:.2f}")
put("KCStabMedian", lng.KC.median(), "{:.0f}")

fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6))
bins = np.arange(6.4, 10.6, 0.15)
ax = axs[0]
ax.hist(np.log10(tab[~tab.stab].K), bins, color="0.6", label="K ≤ K$_0$")
ax.hist(np.log10(tab[tab.stab & ~tab.crowder.isin(LONG)].K), bins, color="C1", label="K > K$_0$, PEG < 1k")
ax.hist(np.log10(lng.K), bins, color="C3", alpha=.85, label="K > K$_0$, PEG ≥ 1k")
ax.axvline(np.log10(K0), color="k", ls=":", lw=1)
ax.set_xlabel("log$_{10}$ K [M$^{-1}$]"); ax.set_ylabel("liczba miareczkowań"); ax.legend(frameon=False)
ax.set_title("gałąź stabilizacji: wąski pik przy 10$^{10}$")
ax = axs[1]
kb = np.logspace(-1.5, 2.5, 30)
ax.hist(tab[~tab.stab].KC, kb, color="0.6")
ax.hist(tab[tab.stab & ~tab.crowder.isin(LONG)].KC, kb, color="C1")
ax.hist(lng.KC, kb, color="C3", alpha=.85)
ax.axvline(20, color="k", ls="--", lw=1)
ax.text(22, ax.get_ylim()[1] * .85, "K·C = 20", fontsize=7)
ax.set_xscale("log"); ax.set_xlabel("K · C$_\\mathrm{donor}$"); ax.set_title(f"{int((tab.KC > 20).sum())} z {len(tab)} miareczkowań ma K·C > 20")
fig.savefig(FIG / "c1_distribution.pdf"); plt.close(fig)

# ======================================================================================
# 3. profile likelihood (done before the simulation; cheap, vectorised in D)
# ======================================================================================
KG = 10 ** np.linspace(5, 12, 141)
DG = np.linspace(1.1e-9, 2e-8, 400)


def _ssr(x, y, f, free_scale):
    if free_scale:  # optimal amplitude (V0 x brightness) for each curve, analytic
        sc = np.sum(f * y, axis=-1, keepdims=True) / np.sum(f * f, axis=-1, keepdims=True)
        f = f * sc
    return np.sum((f - y) ** 2, axis=-1)


def ssr_profile(t, free_scale=False):
    x, y = t["x"], t["y"]
    out = np.empty(len(KG))
    for i, K in enumerate(KG):
        Y = model_vec(x[None, :], K, DG[:, None], t["a"], t["g"], t["v0"])
        s = _ssr(x, y[None, :], Y, free_scale)
        j = int(np.argmin(s))
        lo, hi = DG[max(j - 1, 0)], DG[min(j + 1, len(DG) - 1)]
        r = minimize_scalar(lambda Dv: _ssr(x, y, model_vec(x, K, Dv, t["a"], t["g"], t["v0"]), free_scale),
                            bounds=(lo, hi), method="bounded", options={"xatol": 1e-13})
        out[i] = min(r.fun, s[j])
    return out


prof = []
for t in titr:
    s = ssr_profile(t)
    smin = min(s.min(), t["ssr"])
    s2 = smin / (t["n"] - 2)
    dchi = (s - smin) / s2
    ok95 = dchi <= 3.84
    ok68 = dchi <= 1.0
    lo95 = KG[ok95].min(); hi95 = KG[ok95].max()
    lo68 = KG[ok68].min() if ok68.any() else t["K"]; hi68 = KG[ok68].max() if ok68.any() else t["K"]
    t.update(dchi=dchi, lo95=lo95, hi95=hi95, lo68=lo68, hi68=hi68,
             up_unb=bool(dchi[-1] <= 3.84), lo_unb=bool(dchi[0] <= 3.84),
             notglobal=bool(s.min() < 0.999 * t["ssr"]))
    # robust variant: amplitude (V0) free, i.e. equivalent to fitting chi/chi(0)
    s_r = ssr_profile(t, free_scale=True)
    sminr = s_r.min()
    dchr = (s_r - sminr) / (sminr / (t["n"] - 3))
    okr95, okr68 = dchr <= 3.84, dchr <= 1.0
    t.update(dchi_r=dchr, Kbest_r=KG[np.argmin(s_r)], lo95_r=KG[okr95].min(), hi95_r=KG[okr95].max(),
             lo68_r=KG[okr68].min(), hi68_r=KG[okr68].max(), up_unb_r=bool(dchr[-1] <= 3.84),
             lo_unb_r=bool(dchr[0] <= 3.84))
    prof.append(t)
tb = pd.DataFrame([{k: t[k] for k in ("crowder", "wt", "K", "C", "lo95", "hi95", "lo68", "hi68", "up_unb", "lo_unb", "notglobal",
                                       "Kbest_r", "lo95_r", "hi95_r", "up_unb_r", "lo_unb_r")}
                   for t in titr])
tb = tb[tb.wt <= 40].copy()
tb["KC"] = tb.K * tb.C
put("nUpUnb", int(tb.up_unb.sum()), "{:d}")
put("nLoUnb", int(tb.lo_unb.sum()), "{:d}")
put("nNotGlobal", int(tb.notglobal.sum()), "{:d}")
stab_long = tb[tb.crowder.isin(LONG) & (tb.K > K0)]
put("nUpUnbStabLong", int(stab_long.up_unb.sum()), "{:d}")
put("nStabLongP", len(stab_long), "{:d}")
put("nUpUnbKCtwenty", int(tb[tb.KC > 20].up_unb.sum()), "{:d}")
put("nKCtwentyP", int((tb.KC > 20).sum()), "{:d}")
put("loNinetyFiveMedStab", np.median(np.log10(stab_long.lo95)), "{:.2f}")
bnd = tb[~tb.up_unb & (tb.K > K0)]
put("hiOverKMedBounded", np.median(np.log10(bnd.hi95 / bnd.K)) if len(bnd) else np.nan, "{:.2f}")
put("nBoundedStab", len(bnd), "{:d}")
put("nUpUnbR", int(tb.up_unb_r.sum()), "{:d}")
put("nUpUnbRStabLong", int(tb[tb.crowder.isin(LONG) & (tb.K > K0)].up_unb_r.sum()), "{:d}")
put("nUpUnbRKCtwenty", int(tb[tb.KC > 20].up_unb_r.sum()), "{:d}")
put("nUpUnbRStab", int(tb[(tb.K > K0)].up_unb_r.sum()), "{:d}")
put("nStabAll", int((tb.K > K0).sum()), "{:d}")
put("nUpUnbRDestab", int(tb[(tb.K <= K0)].up_unb_r.sum()), "{:d}")
put("KCminUnbR", tb[tb.up_unb_r].KC.min() if tb.up_unb_r.any() else np.nan, "{:.0f}")
br = tb[~tb.up_unb_r]
put("widthRMed", np.median(np.log10(br.hi95_r / br.lo95_r)), "{:.2f}")
put("widthMed", np.median(np.log10(tb.hi95 / tb.lo95)), "{:.2f}")
put("loRMedStabLong", np.median(np.log10(tb[tb.crowder.isin(LONG) & (tb.K > K0)].lo95_r)), "{:.2f}")
put("logKbestRMedStabLong", np.median(np.log10(tb[tb.crowder.isin(LONG) & (tb.K > K0)].Kbest_r)), "{:.2f}")
put("KCminUnb", tb[tb.up_unb].KC.min() if tb.up_unb.any() else np.nan, "{:.0f}")
vef = {(t["crowder"], t["wt"]): float(t["df"]["Vef_[fl]"].iloc[0]) for t in titr}
tb["Vef"] = [vef[(c, w)] for c, w in zip(tb.crowder, tb.wt)]
tb["dlogK_r"] = np.log10(tb.Kbest_r / tb.K)
put("shiftRMedStab", np.median(tb[tb.K > K0].dlogK_r), "{:+.2f}")
put("shiftRMedAll", np.median(tb.dlogK_r), "{:+.2f}")
put("shiftRSD", tb.dlogK_r.std(), "{:.2f}")
rv, pv = spearmanr(np.log(tb.Vef), tb.dlogK_r)
put("rhoShiftVef", rv, "{:+.2f}")
put("prhoShiftVef", pv, "{:.3f}")
# correlation of residual log K (vs per-crowder quadratic in wt) with V_ef: fixed vs free amplitude
def _resid(col):
    out = []
    for cr, g in tb.groupby("crowder"):
        p_ = np.polyfit(g.wt, np.log10(g[col]), 2)
        out.append(pd.Series(np.log10(g[col]) - np.polyval(p_, g.wt), index=g.index))
    return pd.concat(out).reindex(tb.index)
lv = np.log10(tb.Vef / tb.groupby("crowder").Vef.transform("mean"))
put("corrResVefFixed", np.corrcoef(_resid("K"), lv)[0, 1], "{:+.2f}")
put("corrResVefFree", np.corrcoef(_resid("Kbest_r"), lv)[0, 1], "{:+.2f}")
put("KCmaxBounded", tb[~tb.up_unb].KC.max(), "{:.0f}")

# heat map of status
wts = sorted(tb.wt.unique())
M = np.full((len(ORDER), len(wts)), np.nan)
L = np.full_like(M, np.nan)
for _, r in tb.iterrows():
    i, j = ORDER.index(r.crowder), wts.index(r.wt)
    M[i, j] = 1.0 if r.up_unb_r else 0.0
    L[i, j] = np.log10(r.lo95_r if r.up_unb_r else r.K)
fig, ax = plt.subplots(figsize=(7.2, 3.3))
cm = matplotlib.colors.ListedColormap(["#d9e6f2", "#e8a29a"])
ax.imshow(np.ma.masked_invalid(M), cmap=cm, vmin=0, vmax=1, aspect="auto")
for i in range(len(ORDER)):
    for j in range(len(wts)):
        if np.isfinite(L[i, j]):
            ax.text(j, i, ("≥" if M[i, j] == 1 else "") + f"{L[i, j]:.1f}", ha="center", va="center", fontsize=6.3)
ax.set_xticks(range(len(wts)), [f"{w:g}" for w in wts]); ax.set_yticks(range(len(ORDER)), [LABEL[c] for c in ORDER])
ax.set_xlabel("crowder [wt%]")
ax.set_title("log$_{10}$K; czerwone: brak górnej granicy 95% przy wolnej amplitudzie (liczba = dolna granica)", fontsize=8.5)
fig.savefig(FIG / "c3_profile_status.pdf"); plt.close(fig)

# example profiles
ex = [t for t in titr if (t["crowder"], t["wt"]) in [("EGly", 25.0), ("PEG35000", 25.0), ("PEG35000", 10.0), ("PEG6000", 12.5)]]
fig, ax = plt.subplots(figsize=(4.6, 2.6))
for t, c in zip(ex, ["C0", "C2", "C3", "C1"]):
    ax.plot(np.log10(KG), t["dchi"], color=c, label=f"{LABEL[t['crowder']]} {t['wt']:g}%")
    ax.plot(np.log10(KG), t["dchi_r"], color=c, ls="--", lw=1)
ax.axhline(3.84, color="k", ls="--", lw=.8); ax.set_ylim(0, 30)
ax.set_xlabel("log$_{10}$ K"); ax.set_ylabel("Δχ² (przeskalowane)"); ax.legend(frameon=False)
ax.set_title("ciągłe: V$_0$ ustalone (skrypt); przerywane: amplituda wolna", fontsize=8)
fig.savefig(FIG / "c3_profiles.pdf"); plt.close(fig)

# ======================================================================================
# 2. simulation with the original fitting function
# ======================================================================================
KT = 10 ** np.arange(8.0, 12.01, 0.25)
NREP = 8
t0 = time.time()
sim = []
for t in titr:
    if t["wt"] > 40:
        continue
    for Kt in KT:
        ytrue = model_vec(t["x"], Kt, t["D"], t["a"], t["g"], t["v0"])
        for _ in range(NREP):
            d2 = t["df"].copy()
            d2["value"] = ytrue + RNG.normal(0, t["sigma"], len(ytrue))
            res = kfit.fit_single_dataset(d2, t["a"], t["g"], t["v0"], t["C"])
            sim.append((t["crowder"], t["wt"], Kt, res["K"]))
sim = pd.DataFrame(sim, columns=["crowder", "wt", "Ktrue", "Krec"])
print(f"simulation: {len(sim)} fits in {time.time() - t0:.0f} s")
put("nSim", len(sim), "{:d}")
put("nRep", NREP, "{:d}")
sim["dev"] = np.log10(sim.Krec / sim.Ktrue)

# per-titration ceiling: largest Ktrue with |median dev| < 0.2 and robust SD < 0.3 for all Ktrue below it
ceil = {}
for (cr, wt), g in sim.groupby(["crowder", "wt"]):
    good = []
    for Kt, gg in g.groupby("Ktrue"):
        med = np.median(gg.dev)
        rsd = (np.percentile(gg.dev, 84) - np.percentile(gg.dev, 16)) / 2
        good.append((Kt, abs(med) < 0.2 and rsd < 0.3))
    kc = np.nan
    for Kt, ok in good:
        if ok:
            kc = Kt
        else:
            break
    ceil[(cr, wt)] = kc
tb["Kceil"] = [ceil.get((c, w), np.nan) for c, w in zip(tb.crowder, tb.wt)]
tb["above"] = tb.K > tb.Kceil
cl = np.log10(tb[tb.crowder.isin(LONG)].Kceil)
put("ceilMedLong", np.median(cl), "{:.2f}")
put("ceilMinLong", cl.min(), "{:.2f}")
put("ceilMaxLong", cl.max(), "{:.2f}")
put("ceilMedAll", np.median(np.log10(tb.Kceil)), "{:.2f}")
put("ceilKCmed", np.median(tb.Kceil * tb.C), "{:.0f}")
put("nAboveCeil", int(tb.above.sum()), "{:d}")
put("nAboveCeilStabLong", int(tb[tb.crowder.isin(LONG) & (tb.K > K0)].above.sum()), "{:d}")
# pooled behaviour for long-PEG titrations
pool = sim[sim.crowder.isin(LONG)]
qs = pool.groupby("Ktrue").dev.quantile([0.16, 0.5, 0.84]).unstack()
for Kt in [1e9, 1e10, 3.16e10, 1e11]:
    k = qs.index[np.argmin(abs(np.log10(qs.index) - np.log10(Kt)))]
    tag = {1e9: "Nine", 1e10: "Ten", 3.16e10: "TenHalf", 1e11: "Eleven"}[Kt]
    put("simMed" + tag, qs.loc[k, 0.5], "{:+.2f}")
    put("simWid" + tag, (qs.loc[k, 0.84] - qs.loc[k, 0.16]) / 2, "{:.2f}")
recmax = pool.groupby("Ktrue").Krec.median()
put("simRecMedAtTwelve", np.log10(recmax.iloc[-1]), "{:.2f}")

fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7))
ax = axs[0]
ax.fill_between(np.log10(qs.index), np.log10(qs.index) + qs[0.16], np.log10(qs.index) + qs[0.84], color="C3", alpha=.3,
                label="16–84% (PEG ≥ 1k)")
ax.plot(np.log10(qs.index), np.log10(qs.index) + qs[0.5], "C3-", label="mediana")
ax.plot([8, 12], [8, 12], "k:", lw=1, label="K odzyskane = K prawdziwe")
ax.axhspan(np.log10(lng.K).min(), np.log10(lng.K).max(), color="0.85", zorder=0, label="zakres K gałęzi stab. (dane)")
ax.set_xlabel("log$_{10}$ K prawdziwe"); ax.set_ylabel("log$_{10}$ K odzyskane"); ax.legend(frameon=False, fontsize=6.5)
ax.set_title("symulacja tym samym skryptem")
ax = axs[1]
ax.hist(np.log10(tb.Kceil.dropna()), bins=np.arange(8.875, 12.2, 0.25), color="0.6", label="wszystkie")
ax.hist(np.log10(tb[tb.crowder.isin(LONG)].Kceil.dropna()), bins=np.arange(8.875, 12.2, 0.25), color="C3", alpha=.8, label="PEG ≥ 1k")
ax.axvline(np.log10(lng.K.median()), color="k", ls="--", lw=1, label="mediana K gałęzi stab.")
ax.set_xlabel("log$_{10}$ K$_\\mathrm{sufit}$ (na miareczkowanie)"); ax.legend(frameon=False, fontsize=6.5)
ax.set_title("sufit = |bias| < 0.2 i rozrzut < 0.3 dex")
fig.savefig(FIG / "c2_simulation.pdf"); plt.close(fig)

# ======================================================================================
# 4. censored re-analysis (Tobit likelihood)
# ======================================================================================
pts = []
for t in titr:
    if t["wt"] > 40:
        continue
    dd = RT * np.log(K0 / t["K"])
    Kc = t["Kbest_r"]
    hi68 = KG[t["dchi_r"] <= 1.0].max(); lo68 = KG[t["dchi_r"] <= 1.0].min()
    s_lo = RT * np.log(hi68 / Kc) if hi68 > Kc else np.nan  # towards stabilisation
    s_hi = RT * np.log(Kc / lo68) if lo68 < Kc else np.nan
    sig = np.nanmean([s_lo, s_hi]) if np.isfinite([s_lo, s_hi]).any() else np.nan
    bound = RT * np.log(K0 / lo68)  # censored: true ddG <= bound
    pts.append(dict(crowder=t["crowder"], wt=t["wt"], x=float(monomolal(t["wt"], t["crowder"])), y=dd / KCAL,
                    s=sig / KCAL if np.isfinite(sig) else np.nan, cens=t["up_unb_r"], b=bound / KCAL,
                    s_fit=(RT * t["Kerr"] / t["K"] / KCAL) if np.isfinite(t["Kerr"]) else np.nan))
P = pd.DataFrame(pts)
for cr in ORDER:  # censored points: sigma = median sigma of the bounded points of that crowder
    m = P.crowder == cr
    med = np.nanmedian(P.loc[m & ~P.cens, "s"])
    P.loc[m & (P.cens | P.s.isna()), "s"] = med
P.loc[P.s <= 0, "s"] = np.nanmedian(P.s)


def nll(pred, y, s, cens, b):
    o = ~cens
    l = 0.5 * np.sum(((y[o] - pred[o]) / s[o]) ** 2)
    if cens.any():
        l -= np.sum(norm.logcdf((b[cens] - pred[cens]) / s[cens]))
    return l


def fit_ml(fun, p0, sub, mode):
    x = sub.x.values; y = sub.y.values; s = sub.s.values; c = sub.cens.values; b = sub.b.values
    if mode == "drop":
        x, y, s, b = x[~c], y[~c], s[~c], b[~c]; c = c[~c]
    if mode == "orig":
        s = np.where(np.isfinite(sub.s_fit.values) & (sub.s_fit.values > 0), sub.s_fit.values, np.nanmedian(sub.s_fit))
        c = np.zeros_like(c)
    r = minimize(lambda p: nll(fun(x, p), y, s, c, b), p0, method="Nelder-Mead",
                 options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 20000})
    # numerical Hessian
    h = 1e-4 * np.maximum(np.abs(r.x), 1e-2)
    H = np.zeros((len(r.x), len(r.x)))
    f0 = lambda p: nll(fun(x, p), y, s, c, b)  # noqa: E731
    for i in range(len(r.x)):
        for j in range(len(r.x)):
            ei = np.eye(len(r.x))[i] * h[i]; ej = np.eye(len(r.x))[j] * h[j]
            H[i, j] = (f0(r.x + ei + ej) - f0(r.x + ei - ej) - f0(r.x - ei + ej) + f0(r.x - ei - ej)) / (4 * h[i] * h[j])
    try:
        cov = np.linalg.inv(H)
        err = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        err = np.full(len(r.x), np.nan)
    npts = len(y)
    red = 2 * r.fun / max(npts - len(r.x), 1)
    return r.x, err * np.sqrt(max(red, 1.0)), 2 * r.fun


MODES = ["orig", "drop", "tobit"]
MLAB = {"orig": "jak w raporcie (bez cenzury)", "drop": "bez punktów cenzurowanych", "tobit": "cenzura (Tobit)"}
lin = lambda x, p: p[0] * x  # noqa: E731
quad = lambda x, p: p[0] * x + p[1] * x ** 2  # noqa: E731


def broken(x, p, xs=None, w=1.0):
    s1, s2 = p[0], p[1]
    xs = p[2] if xs is None else xs
    f = lambda xx: s1 * xx + (s2 - s1) * w * np.logaddexp(0, (xx - xs) / w)  # noqa: E731
    return f(x) - f(0.0)


summ = {m: {} for m in MODES}
P.to_csv(HERE / "points.csv", index=False)
put("nCensPts", int(P.cens.sum()), "{:d}")
for mode in MODES:
    # m-values N>=13 on 0<x<=5.5
    mv = []
    for cr in N13:
        sub = P[(P.crowder == cr) & (P.x > 0) & (P.x <= 5.5)]
        p, e, _ = fit_ml(lin, [-0.3], sub, mode)
        mv.append((p[0], e[0]))
    mv = np.array(mv)
    w = 1 / mv[:, 1] ** 2
    summ[mode]["m"] = (np.sum(mv[:, 0] * w) / np.sum(w), 1 / np.sqrt(np.sum(w)))
    summ[mode]["m_each"] = mv.tolist()
    # quadratic minimum, PEG >= 1k
    xm = []
    for cr in LONG:
        sub = P[(P.crowder == cr) & (P.x > 0)]
        p, e, _ = fit_ml(quad, [-0.3, 0.02], sub, mode)
        xm.append(-p[0] / (2 * p[1]) if p[1] > 0 else np.nan)
    summ[mode]["xmin"] = (np.nanmedian(xm), np.nanmin(xm), np.nanmax(xm))
    # broken fits PEG 3k-35k
    s2f, xsf, s16, s26, l6, lf = [], [], [], [], 0.0, 0.0
    for cr in LONG[2:]:
        sub = P[(P.crowder == cr)]
        best = None
        for xs0 in [0.5, 1.5, 3.0, 6.0]:
            pf, ef, lf_ = fit_ml(lambda x, p: broken(x, p), [-0.5, 0.15, xs0], sub, mode)
            if best is None or lf_ < best[2]:
                best = (pf, ef, lf_)
        pf, ef, lf_ = best
        p6, e6, l6_ = fit_ml(lambda x, p: broken(x, p, xs=6.0), [-0.25, 0.18], sub, mode)
        s2f.append((pf[1], ef[1])); xsf.append(pf[2]); s16.append((p6[0], e6[0])); s26.append((p6[1], e6[1]))
        l6 += l6_; lf += lf_
    def wmean(a):
        a = np.array(a); ww = 1 / np.maximum(a[:, 1], 1e-4) ** 2
        return np.sum(a[:, 0] * ww) / np.sum(ww), 1 / np.sqrt(np.sum(ww))
    summ[mode].update(s2free=wmean(s2f), xsfree=(np.median(xsf), np.min(xsf), np.max(xsf)), s1six=wmean(s16),
                      s2six=wmean(s26), m2l_six=l6, m2l_free=lf)
    tag = {"orig": "O", "drop": "D", "tobit": "T"}[mode]
    pm("mN" + tag, *summ[mode]["m"], "{:+.3f}")
    put("xminMed" + tag, summ[mode]["xmin"][0], "{:.1f}")
    put("xminLo" + tag, summ[mode]["xmin"][1], "{:.1f}")
    put("xminHi" + tag, summ[mode]["xmin"][2], "{:.1f}")
    pm("sTwoFree" + tag, *summ[mode]["s2free"], "{:+.3f}")
    put("xsMed" + tag, summ[mode]["xsfree"][0], "{:.1f}")
    put("xsLo" + tag, summ[mode]["xsfree"][1], "{:.1f}")
    put("xsHi" + tag, summ[mode]["xsfree"][2], "{:.1f}")
    pm("sOneSix" + tag, *summ[mode]["s1six"], "{:+.3f}")
    pm("sTwoSix" + tag, *summ[mode]["s2six"], "{:+.3f}")
    put("dChiSixFree" + tag, l6 - lf, "{:.0f}")
results["censored_summary"] = summ

# Delta G_rest, PEG >= 3k (manuscript Fig. 4d-f set)
pr = pipe[(pipe["crowder wt. [%]"] < 50) & (pipe["molar mass [g/mol]"] >= 3000) & (pipe["crowder wt. [%]"] > 0)].copy()
pr["rest"] = pr["dG_exp_[kJ/mol]"].apply(lambda u: u.nominal_value) - pr["dG_Na_[kJ/mol]"].apply(lambda u: u.nominal_value)
pr = pr.merge(tb[["crowder", "wt", "up_unb_r"]].rename(columns={"up_unb_r": "up_unb"}), left_on=["crowder", "crowder wt. [%]"], right_on=["crowder", "wt"], how="left")
put("restMeanAll", pr.rest.mean(), "{:.1f}")
put("restMeanBounded", pr[~pr.up_unb].rest.mean(), "{:.1f}")
put("restMeanCens", pr[pr.up_unb].rest.mean(), "{:.1f}")
put("restFracCens", 100 * pr.up_unb.mean(), "{:.0f}")
put("restSDBounded", pr[~pr.up_unb].rest.std(), "{:.1f}")

# forest plot of the key quantities
fig, axs = plt.subplots(1, 4, figsize=(7.4, 2.2), sharey=True)
qlist = [("m", "m/N, N ≥ 13 [kcal/(mol·molal)]"), ("s1six", "$s_1$ (x*=6)"), ("s2six", "$s_2$ (x*=6)"), ("s2free", "$s_2$ (x* swobodne)")]
for ax, (q, lab) in zip(axs, qlist):
    for i, mode in enumerate(MODES):
        v, e = summ[mode][q]
        ax.errorbar(v, i, xerr=e, fmt="o", color=["0.3", "C0", "C3"][i], ms=4)
    ax.set_xlabel(lab, fontsize=7.5)
    ax.axvline(0, color="grey", lw=.5)
axs[0].set_yticks(range(3), [MLAB[m] for m in MODES], fontsize=7.5)
axs[0].invert_yaxis()
fig.suptitle("Kluczowe wielkości z raportu: wpływ cenzurowania", y=1.04, fontsize=9)
fig.savefig(FIG / "c4_censored.pdf"); plt.close(fig)

# ======================================================================================
# 5. spline salt calibration
# ======================================================================================
salt = D.SodiumDependencyObj.raw_data.copy().sort_values("C_Na+_[mM]")
cs = salt["C_Na+_[mM]"].values.astype(float)
Ks = salt["K_Na+_[1/M]"].values.astype(float)
sKs = salt["K_err_Na+_[1/M]"].values.astype(float)
lc, lK, slK = np.log(cs), np.log(Ks), sKs / Ks
C0 = 35.0


def spl(lk):
    return UnivariateSpline(lc, lk, w=1 / slK, k=3, s=len(lc))


s0 = spl(lK)
NB = 2000
bs = [spl(lK + RNG.normal(0, slK)) for _ in range(NB)]
SKs = np.array([b.derivative()(np.log(C0)) for b in bs])
pm("SKspline", float(s0.derivative()(np.log(C0))), SKs.std())
put("SKsplineLo", np.percentile(SKs, 16)); put("SKsplineHi", np.percentile(SKs, 84))
A, B, k = D.SodiumDependencyObj.popt_eff
Gexp = lambda c: A + B * np.exp(-k * np.sqrt(c))  # noqa: E731
put("SKexp", B * k * np.sqrt(C0) * np.exp(-k * np.sqrt(C0)) / (2 * RT))

par = json.load(open(HERE.parent / "results.json"))
kappa = 0.168
MUX = 0.5 * (0.0378 + 0.0324) * KCAL  # kJ / molal mon (Knowles SI Table S3)
ONEEPS = 0.92


def density(wt):
    return 0.997 + 0.0017441 * wt


def molar_chain(wt, cr):
    return wt * density(wt) / 100 * 1000 / float(pipe[pipe.crowder == cr]["MW_[g/mol]"].iloc[0])


def var_c(cr, wt, G, kap=kappa):
    bc = kap * NMON[cr] * molar_chain(wt, cr)
    return G(C0 / np.sqrt(1 + bc)) - G(C0)


Gspl = lambda c, s=s0: -RT * s(np.log(c))  # noqa: E731
mval = par["mvalues"]
fr = {"c_exp": [], "c_spl": [], "e_exp": [], "e_spl": []}
fr_unc = []
SKexp = B * k * np.sqrt(C0) * np.exp(-k * np.sqrt(C0)) / (2 * RT)
SKloc = float(par["SKlocal"])
for cr in LONG:
    b1, b2 = mval[cr]["b1"] * KCAL, mval[cr]["b2"] * KCAL
    xm = -b1 / (2 * b2)
    x40 = monomolal(40, cr)
    sw = (b1 * x40 + b2 * x40 ** 2) - (b1 * xm + b2 * xm ** 2)
    wm = 100 * xm * 44.05 / (1000 + xm * 44.05)
    fr["c_exp"].append((var_c(cr, 40, Gexp) - var_c(cr, wm, Gexp)) / sw)
    fr["c_spl"].append((var_c(cr, 40, Gspl) - var_c(cr, wm, Gspl)) / sw)
    sk = float(s0.derivative()(np.log(C0)))
    fr["e_spl"].append(-(sk / (2 * ONEEPS)) * MUX * (x40 - xm) / sw)
    fr["e_exp"].append(-(SKloc / (2 * ONEEPS)) * MUX * (x40 - xm) / sw)
    fr_unc.append(np.std([((var_c(cr, 40, lambda c, s=b: -RT * s(np.log(c))) -
                            var_c(cr, wm, lambda c, s=b: -RT * s(np.log(c)))) / sw) for b in bs[:400]]))
for kk, v in fr.items():
    put("frac" + kk.replace("_", ""), 100 * np.mean(v), "{:.0f}")
put("fraccsplUnc", 100 * np.mean(fr_unc), "{:.0f}")
put("cFortySpl", var_c("PEG35000", 40, Gspl), "{:+.1f}")
put("cFortyExp", var_c("PEG35000", 40, Gexp), "{:+.1f}")
put("SKlocalReport", SKloc)

fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.5))
cg = np.exp(np.linspace(lc.min(), lc.max(), 200))
band = np.array([b(np.log(cg)) for b in bs])
ax = axs[0]
ax.errorbar(cs, lK, slK, fmt="ko", ms=3.5, label="seria Na$^+$")
ax.fill_between(cg, np.percentile(band, 16, 0), np.percentile(band, 84, 0), color="C2", alpha=.3)
ax.plot(cg, s0(np.log(cg)), "C2-", label="splajn ln K(ln c)")
ax.plot(cg, -Gexp(cg) / RT, "C0--", label="$A+Be^{-k\\sqrt{c}}$")
ax.set_xscale("log"); ax.set_xlabel("c$_\\mathrm{Na}$ [mM]"); ax.set_ylabel("ln K"); ax.legend(frameon=False)
ax = axs[1]
dband = np.array([b.derivative()(np.log(cg)) for b in bs])
ax.fill_between(cg, np.percentile(dband, 16, 0), np.percentile(dband, 84, 0), color="C2", alpha=.3)
ax.plot(cg, s0.derivative()(np.log(cg)), "C2-", label="splajn")
ax.plot(cg, B * k * np.sqrt(cg) * np.exp(-k * np.sqrt(cg)) / (2 * RT), "C0--", label="model wykładniczy")
ax.axvline(C0, color="grey", ls=":")
ax.set_xscale("log"); ax.set_xlabel("c$_\\mathrm{Na}$ [mM]"); ax.set_ylabel("SK"); ax.legend(frameon=False)
ax.set_title("SK(c$_0$) zależy od formy kalibracji")
fig.savefig(FIG / "c5_spline.pdf"); plt.close(fig)

# ---------------------------------------------------------------------------------------
# reference numbers from the parent report (read, not recomputed)
for key in ["broSixStwo", "broSixSone", "broStwo", "restMean", "plateauKnowlesPred", "nullE", "EKnowles", "fracMeanc", "fracMeane"]:
    put("rep" + key[0].upper() + key[1:], float(par[key]) if key in par else np.nan, "{:+.3f}" if key.startswith(("bro", "plat", "null", "EK")) else "{:.1f}")
with open(HERE / "numbers.tex", "w") as fh:
    fh.write("% generated by ceiling.py\n")
    words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    for kk, v in sorted(numbers.items()):
        safe = "".join(ch if ch.isalpha() else (words[int(ch)] if ch.isdigit() else "") for ch in kk)
        fh.write(f"\\newcommand{{\\C{safe}}}{{{v}}}\n")


def _c(o):
    if isinstance(o, dict):
        return {str(k): _c(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_c(v) for v in o]
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return o


json.dump(_c(results), open(HERE / "results.json", "w"), indent=1, default=str)
tb.to_csv(HERE / "titrations.csv", index=False)
print("OK", len(numbers))
for kk in sorted(numbers):
    print(kk, numbers[kk])
