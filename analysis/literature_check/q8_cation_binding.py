"""Q8: evaluation of claims of Na+/K+ binding by PEG on one axis.

Axis: mu_+ = cation contribution to (d mu_salt / d m_EO), cal mol^-1 (molal EO)^-1, Knowles 2015 convention
(> 0 exclusion, < 0 accumulation). Weak 1:1 binding with constant kappa per EO: mu_+ ~ -RT*kappa
(linear limit; kappa in M^-1 treated as molal^-1, error < 10% for our solutions).
Sources: notes/Q2_peg_salt.md, notes/Q4_Q7_duplex_experiments.md, salt_shift/na_notes.md,
knowles_reframe/numbers.tex, recenzja_manuskryptu.md (Bielec 2022 SI sec. 7).
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RT = 592.5  # cal/mol, 298 K
HERE = Path(__file__).resolve().parent


def mu(kappa):
    return -RT * kappa


# ISE bound for EG 40 wt% (Bielec 2022 SI 7): total calibration offset ~6 mV, also seen without Na+.
# 59.16 log10(1 + bn c) <= 6 mV, c_EG(40 wt%) ~ 420 g/L / 62.07 g/mol
c_EG40 = 420 / 62.07
bn_ise = (10 ** (6 / 59.16) - 1) / c_EG40
# DOSY EG and PEG 200 (self-consistent correction, knowles_reframe/numbers.tex): bn per molecule
bn_EG, bn_EG_err = -0.034, 0.154
bn_P2, bn_P2_err = 0.034, 0.121
DP_P2 = (200 - 18) / 44.05
# nanopore (Rodrigues 2008, PT): PEG-pore interaction energy per monomer 0.03 kT (1 M KCl) .. 0.27 kT (4 M)
# ASSUMPTION: whole energy from linear Langmuir K+ binding, E/kT = kappa*c
kap_pore = (0.03 / 1.0, 0.27 / 4.0)
# crown ethers in water (log K; 18C6-Na+ 0.8, 18C6-K+ 2.03; Can. J. Chem. 64, 808 (1986) table seen only as
# search snippet; Na+ 0.82 also from an NMR study, snippet). Per-EO upper bound: K/6.
crown = {"Na+": 10 ** 0.8 / 6, "K+": 10 ** 2.03 / 6}

# env: dil = dilute water, conc = concentrated PEG/EG (>=10 wt%), pore, iface, org, solid
# kind: thermo = bulk thermodynamic, local = local/structural/transport, infer = inferred from another reaction
P = [
    # label, env, kind, value(s) [lo, hi] or None
    ("Knowles 2015 VPO: Na$^+$", "dil", "thermo", (4.66 - 2, 4.66 + 2)),
    ("Knowles 2015 VPO: K$^+$", "dil", "thermo", (3.22 - 3.4, 3.22 + 3.4)),
    ("Knowles 2015 VPO: NaCl (sól)", "dil", "thermo", (42.3 - 3.1, 42.3 + 3.1)),
    ("Knowles 2011 VPO: KCl (sól)", "dil", "thermo", (32.4, 37.8)),
    ("Annunziata: NaCl (sól), dyfuzja", "dil", "thermo", (47.0, 49.9)),
    ("Morales 2020 ISE: NaClO$_4$ (sól)", "conc", "thermo", (-65.4, -60.2)),
    ("Bielec 2022 ISE: EG 40%", "conc", "thermo", (mu(bn_ise), 0.0)),
    ("nasze DOSY: EG", "conc", "local", (mu(bn_EG + bn_EG_err), mu(bn_EG - bn_EG_err))),
    ("nasze DOSY: PEG 200", "conc", "local", (mu((bn_P2 + bn_P2_err) / DP_P2), mu((bn_P2 - bn_P2_err) / DP_P2))),
    ("nasze DOSY: PEG $\\geq$1000", "conc", "local", (mu(0.168), mu(0.055))),
    ("Rodrigues 2008: por, 1--4 M KCl", "pore", "local", (mu(kap_pore[1]), mu(kap_pore[0]))),
    ("Bielec 2022: $\\kappa$=0.49 z K(c)", "conc", "infer", (mu(0.49), mu(0.49))),
]
NONQ = [  # not convertible
    ("Henning-Knechtel 2026 MD", "conc", "local"),
    ("Ohki 2007 XAFS/woltamperometria", "iface", "local"),
    ("Breton 2013 nanopor", "pore", "local"),
    ("Reiner 2010 nanopor (teoria)", "pore", "infer"),
    ("Hakem 2004 SANS", "org", "local"),
    ("Okada 1991 żywica", "org", "thermo"),
    ("Fenton 1973 PEO--sól", "solid", "local"),
    ("Karimata 2007, Spink 1999", "dil", "infer"),
    ("Silva \\& Weber 2026", "conc", "infer"),
]

col = {"dil": "#1f77b4", "conc": "#2ca02c", "pore": "#d62728", "iface": "#ff7f0e", "org": "#9467bd",
       "solid": "#8c564b"}
envname = {"dil": "woda, rozcieńczony PEG", "conc": "woda, stężony PEG/EG ($\\geq$10 wt%)", "pore": "por",
           "iface": "granica faz", "org": "rozp. organiczny / żywica", "solid": "ciało stałe"}
mk = {"thermo": "o", "local": "s", "infer": "^"}
kindname = {"thermo": "termodynamiczny, objętościowy", "local": "lokalny / strukturalny / transportowy",
            "infer": "wnioskowany z innej reakcji"}

out = {"RT_cal": RT, "bn_ise_EG_upper": bn_ise, "crown_kappa_per_EO": crown,
       "crown_mu_per_EO": {k: mu(v) for k, v in crown.items()},
       "points": [{"label": l, "env": e, "kind": k, "lo": float(min(v)), "hi": float(max(v))} for l, e, k, v in P]}

# reconciliation: net Na+ = coordination - exclusion. Exclusion of a structureless ion from nu_W water per EO
# gives mu_+(excl) = RT*nu_W/55.51 (per ion); NaCl net nu_W = 2.4 (Annunziata).
nuW = 2.4
mu_excl_ion = RT * nuW / 55.51
kap_coord_allowed = (mu_excl_ion - 4.66) / RT
out["reconcile"] = {"mu_excl_per_ion_cal": mu_excl_ion, "kappa_coord_allowed_by_VPO_split": kap_coord_allowed,
                    "nuW_gross_needed_for_DOSY_kappa": {str(k): (4.66 + RT * k) * 55.51 / RT for k in (0.055, 0.168)}}
json.dump(out, open(HERE / "q8_cation_binding.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(out, indent=1, ensure_ascii=False))

fig, ax = plt.subplots(figsize=(7.6, 5.4))
order = sorted(range(len(P)), key=lambda i: (list(mk).index(P[i][2]), -np.mean(P[i][3])))
for y, i in enumerate(order):
    l, e, k, v = P[i]
    lo, hi = min(v), max(v)
    ax.plot([lo, hi], [y, y], "-", color=col[e], lw=2.2, alpha=.7)
    ax.plot([(lo + hi) / 2], [y], mk[k], color=col[e], ms=7, mec="k", mew=.5)
ax.set_yticks(range(len(P)), [P[i][0] for i in order], fontsize=7.5)
ax.set_xscale("symlog", linthresh=10)
ax.axvline(0, color="grey", lw=.8)
for (ion, kap), ls in zip(crown.items(), ["--", ":"]):
    ax.axvline(mu(kap), color="k", ls=ls, lw=1)
    ax.text(mu(kap) * 1.15, -0.6, f"18C6–{ion}", fontsize=6.5, ha="right", va="bottom")
ax.set_xlim(-2e4, 150)
ax.set_xlabel("$\\mu_+$ albo $\\mu_x$ (sól) [cal mol$^{-1}$ (molal EO)$^{-1}$];  < 0 akumulacja, > 0 wykluczenie")
from matplotlib.lines import Line2D
h = [Line2D([], [], color=col[e], lw=3, label=envname[e]) for e in ["dil", "conc", "pore"]]
h += [Line2D([], [], marker=mk[k], ls="", color="w", mec="k", label=kindname[k]) for k in mk]
h += [Line2D([], [], color="k", ls="--", label="górna granica: 18-korona-6 w wodzie, K/6 na EO")]
ax.set_ylim(len(P) - 0.5, -1.0)
ax.legend(handles=h, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.35, -0.13), ncol=2, frameon=False)
fig.tight_layout()
fig.savefig(HERE / "q8_cation_binding.pdf", bbox_inches="tight")
