"""Q3: polymer-depletion contribution to the free energy of 13-bp DNA hybridization in PEG.

dG_dep(x) = mu_ins(dsDNA) - 2 mu_ins(ssDNA)   (negative = depletion stabilises the duplex)

Three families of insertion free energies (sources/eq. numbers in notes/Q3_depletion.md):
  P  protein limit (particle << xi, R_x): small-radius expansion, Hanke-Eisenriegler-Dietrich
     PRE 59, 6853 (1999) Eqs. (1.5), (1.8), (2.54), (2.55); semidilute: Odijk Macromolecules
     29, 1842 (1996) Eq. (5).  mu = A_{d,3} * R^{d-1/nu} * l^{3-d} * <rho>,  <rho> ~ c_monomer.
  G  generalised free-volume theory (all size ratios, interpolation): mu = Pi * V_excl(R + delta),
     delta from Fleer-Tuinier (Lekkerkerker & Tuinier 2011, Eqs. 4.21, 4.25/4.33, 6.48-6.50).
  C  colloid-limit asymptote (particle >> xi): mu = Pi*V + gamma*A with gamma = Pi*delta_flat
     (first-order expansion of G; Louis et al. 2002 Eq. 6 gives gamma = 1.5 Pi xi).
Pi from the Cohen-Podgornik-Hansen-Parsegian EOS (JPCB 113, 3709 (2009) Eq. 4, alpha = 0.49).

Run:  .venv/bin/python analysis/literature_check/q3_depletion_exponents.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
NA = 6.02214076e23
T = 298.15
RT = 8.314462618e-3 * T  # kJ/mol

# ---------------- PEG solution properties (project conventions) ----------------
NU = 0.583                 # project Rg law exponent (utils.crowders_properties)
M_MON = 44.05              # g/mol per EO unit
V_BAR, ALPHA, M_M_COHEN = 0.825, 0.49, 44.0  # Cohen et al. 2009 (mL/g, -, Da)
RE2_RG2 = 6.25             # <R_E^2>/<R_g^2> for a self-avoiding chain (assumption, ~6.25)


def rg_nm(M):
    return 0.0215 * M ** NU


def c_star(M):  # g/mL, same definition as utils.py: M / (N_A * 4/3 pi Rg^3)
    return M / (NA * 4 / 3 * np.pi * (rg_nm(M) * 1e-7) ** 3)


def density(wt):  # g/mL, processor convention
    return 0.99707 + 0.0017441 * wt


def mass_conc(wt):  # g/mL
    return wt * density(wt) / 100


def monomolal(wt):  # mol EO / kg water
    w = wt / 100
    return 1000 * w / ((1 - w) * M_MON)


def osmotic_kT_nm3(wt, M):
    """Cohen EOS, Eq. 4 rewritten: Pi/RT = C/M + alpha (C Vbar)^{5/4} C / M_m  [mol/mL]."""
    C = mass_conc(wt)
    mol_per_ml = C / M + ALPHA * (C * V_BAR) ** 1.25 * C / M_M_COHEN
    return mol_per_ml * NA * 1e-21  # number density in nm^-3  ->  Pi/kT


def osmotic_dC_kT_nm3(wt):
    C = mass_conc(wt)
    return ALPHA * (C * V_BAR) ** 1.25 * C / M_M_COHEN * NA * 1e-21


def delta_flat(wt, M):
    """Fleer-Skvortsov-Tuinier flat-wall depletion thickness, good solvent (L&T Eq. 4.21/6.48)."""
    phi = mass_conc(wt) / c_star(M)
    return 1.071 * rg_nm(M) / np.sqrt(1 + 3.95 * phi ** 1.54)


def xi_scaling(wt, M):
    phi = mass_conc(wt) / c_star(M)
    return np.where(phi > 1, rg_nm(M) * phi ** (-0.77), rg_nm(M))


# ---------------- insertion free energies (units of kT) ----------------
A33, A23 = 9.82, 1.23  # Hanke et al. 1999 Eqs. (2.54), (2.55)


def rho_mod(wt):
    """<rho> = n_mon * R_x^{1/nu}/N  (Hanke Eq. 1.13), M-independent; nm^{-(3-1/nu)}."""
    n_mon = mass_conc(wt) * NA / M_MON * 1e-21                # nm^-3
    rx_per_mon = (np.sqrt(RE2_RG2 / 3) * 0.0215) ** (1 / NU) * M_MON  # R_x^{1/nu}/N, nm^{1/nu}
    return n_mon * rx_per_mon


def mu_protein(shape, wt, M):
    if shape[0] == "sphere":
        R = shape[1]
        return A33 * R ** (3 - 1 / NU) * rho_mod(wt)
    r, L = shape[1], shape[2]
    return A23 * L * r ** (2 - 1 / NU) * rho_mod(wt)


def mu_gfvt(shape, wt, M):
    Pi = osmotic_kT_nm3(wt, M)
    phi = mass_conc(wt) / c_star(M)
    if shape[0] == "sphere":
        R = shape[1]
        q = rg_nm(M) / R
        ds = R * 0.865 * (q / np.sqrt(1 + 3.95 * phi ** 1.54)) ** 0.88  # Eq. 4.33
        return Pi * 4 / 3 * np.pi * (R + ds) ** 3
    r, L = shape[1], shape[2]
    y = 1.071 * (rg_nm(M) / r) / np.sqrt(1 + 3.95 * phi ** 1.54)  # Eq. 6.50 argument
    dc = r * (np.sqrt(1 + 2 * y - 0.51215 * y ** 1.6 + 1.214 * y ** 1.77) - 1)
    return Pi * np.pi * (r + dc) ** 2 * (L + 2 * dc)


def mu_colloid(shape, wt, M):
    Pi, d = osmotic_kT_nm3(wt, M), delta_flat(wt, M)
    if shape[0] == "sphere":
        R = shape[1]
        V, A = 4 / 3 * np.pi * R ** 3, 4 * np.pi * R ** 2
    else:
        r, L = shape[1], shape[2]
        V, A = np.pi * r ** 2 * L, 2 * np.pi * r * L + 2 * np.pi * r ** 2
    return Pi * (V + A * d)


MODELS = {"P": ("protein limit (SRE)", mu_protein),
          "G": ("GFVT  Pi*V(R+delta)", mu_gfvt),
          "C": ("colloid asymptote Pi*V+Pi*delta*A", mu_colloid)}
DS = ("rod", 1.0, 4.4)
SS = {"rod": ("rod", 0.5, 8.0), "sphere": ("sphere", 1.5)}


def dG_dep(model, ss, wt, M):  # kJ/mol
    f = MODELS[model][1]
    return RT * (f(DS, wt, M) - 2 * f(SS[ss], wt, M))


# ---------------- table ----------------
wt = np.linspace(1, 40, 400)
x = monomolal(wt)
rows = []
for M in (600, 6000, 35000):
    for model in MODELS:
        for ss in SS:
            g = dG_dep(model, ss, wt, M)
            lg = np.log(np.abs(g))
            p = np.gradient(lg, np.log(x))
            m = (wt >= 5) & (wt <= 40)
            p_fit = np.polyfit(np.log(x[m]), lg[m], 1)[0]
            r = dict(PEG=M, model=model, ssDNA=ss, p_fit_5_40=p_fit)
            for w0 in (5, 10, 20, 40):
                i = np.argmin(np.abs(wt - w0))
                r[f"dG_{w0}wt"] = g[i]
                r[f"p_{w0}wt"] = p[i]
            rows.append(r)
tab = pd.DataFrame(rows)
tab.to_csv(OUT / "q3_table.csv", index=False, float_format="%.3f")

lens = []
for M in (600, 6000, 35000):
    for w0 in (5, 10, 20, 40):
        lens.append(dict(PEG=M, wt=w0, x_molal=monomolal(w0), c_over_cstar=mass_conc(w0) / c_star(M),
                         Rg=rg_nm(M), xi_scaling=float(xi_scaling(w0, M)), delta_flat=delta_flat(w0, M),
                         xi_Pi=(1 / osmotic_dC_kT_nm3(w0)) ** (1 / 3),
                         Pi_kT_nm3=osmotic_kT_nm3(w0, M), Pi_bar=osmotic_kT_nm3(w0, M) * 1.380649e-23 * T * 1e27 / 1e5,
                         rho_mod=rho_mod(w0)))
lens = pd.DataFrame(lens)
lens.to_csv(OUT / "q3_lengths.csv", index=False, float_format="%.4g")

pd.set_option("display.width", 250)
print(f"c* [g/mL]: 600 {c_star(600):.3f}, 6k {c_star(6000):.4f}, 35k {c_star(35000):.4f}")
print(lens.round(3).to_string(index=False))
print()
print(tab.round(2).to_string(index=False))

# Odijk check: effective prefactor of (R/xi)^{4/3} implied by the SRE amplitude
for w0 in (10, 40):
    xs = float(xi_scaling(w0, 35000))
    print(f"wt {w0}%: SRE sphere R=1.5 mu={mu_protein(SS['sphere'], w0, 35000):.2f} kT; "
          f"(R/xi)^(4/3)={(1.5 / xs) ** (4 / 3):.2f} with xi={xs:.2f} nm")

# ---------------- plots ----------------
COL = {"P": "#2a78d6", "G": "#eb6834", "C": "#1baf7a"}
LS = {"rod": "-", "sphere": "--"}
x_obs = np.linspace(monomolal(5), monomolal(40), 50)
fig, axs = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
for ax, M in zip(axs, (600, 6000, 35000)):
    ax.axhspan(-6, -4, color="0.85", lw=0)
    ax.plot(x_obs, -5 + 0.10 * 4.184 * (x_obs - x_obs[0]), color="0.3", lw=1.2,
            label="obs.: plateau −4…−6, +0.42 kJ/mol per m")
    for model in MODELS:
        for ss in SS:
            ax.plot(x, dG_dep(model, ss, wt, M), color=COL[model], ls=LS[ss], lw=1.6,
                    label=f"{MODELS[model][0]}; ss={ss}")
    ax.axhline(0, color="0.5", lw=0.6)
    ax.set_title(f"PEG {M}")
    ax.set_xlabel("x [mol EO / kg H$_2$O]")
    ax.set_ylim(-60, 5)
    ax.grid(alpha=0.25, lw=0.5)
axs[0].set_ylabel(r"$\Delta G_\mathrm{dep}$ [kJ/mol]")
axs[2].legend(fontsize=6, frameon=False, loc="lower left")
fig.tight_layout()
fig.savefig(OUT / "q3_dG_dep.pdf")
plt.close(fig)

fig, axs = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
for ax, M in zip(axs, (600, 6000, 35000)):
    for model in MODELS:
        for ss in SS:
            g = dG_dep(model, ss, wt, M)
            ax.plot(x, np.gradient(np.log(np.abs(g)), np.log(x)), color=COL[model], ls=LS[ss], lw=1.6,
                    label=f"{MODELS[model][0]}; ss={ss}")
    ax.axvspan(monomolal(5), monomolal(40), color="0.93", lw=0)
    ax.axhline(1, color="0.5", lw=0.6)
    ax.set_xscale("log")
    ax.set_ylim(-0.5, 3)
    ax.set_title(f"PEG {M}")
    ax.set_xlabel("x [mol EO / kg H$_2$O]")
    ax.grid(alpha=0.25, lw=0.5)
axs[0].set_ylabel(r"$p = d\ln|\Delta G_\mathrm{dep}|/d\ln x$")
axs[0].legend(fontsize=6, frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "q3_local_slope.pdf")
plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(8, 3.4))
for M, c in zip((600, 6000, 35000), ("#2a78d6", "#eb6834", "#1baf7a")):
    axs[0].plot(x, xi_scaling(wt, M), color=c, label=f"ξ=Rg(c/c*)^-0.77, PEG {M}")
    axs[0].plot(x, delta_flat(wt, M), color=c, ls="--", label=f"δ flat (Fleer), PEG {M}")
    axs[1].plot(x, osmotic_kT_nm3(wt, M) * 1.380649e-23 * T * 1e27 / 1e5, color=c, label=f"PEG {M}")
axs[0].plot(x, (1 / osmotic_dC_kT_nm3(wt)) ** (1 / 3), color="0.3", ls=":", label="ξ_Π=(kT/Π_dC)^{1/3}")
for a, w in ((1.0, "r ds"), (0.5, "r ss rod"), (1.5, "R ss coil")):
    axs[0].axhline(a, color="0.6", lw=0.6)
    axs[0].text(x[-1], a, w, fontsize=6, ha="right", va="bottom")
axs[0].set_yscale("log"); axs[0].set_xscale("log")
axs[0].set_ylabel("length [nm]"); axs[0].legend(fontsize=5.5, frameon=False)
axs[1].set_yscale("log"); axs[1].set_xscale("log"); axs[1].set_ylabel("Π [bar] (Cohen EOS)")
axs[1].legend(fontsize=6, frameon=False)
for a in axs:
    a.set_xlabel("x [mol EO / kg H$_2$O]"); a.grid(alpha=0.25, lw=0.5)
fig.tight_layout()
fig.savefig(OUT / "q3_lengths.pdf")
plt.close(fig)
