"""Q2: estimate of the PEG-salt coupling mu_x = (d mu_salt / d m_PEG)_{m_salt} for our system.

All literature inputs are hard-coded below with their source; see notes/Q2_peg_salt.md.
Conventions (Record/Knowles): mu_x = mu_23 in cal mol^-1 (of salt) per molal of PEG group
(EO = interior -CH2OCH2- group; 2E = the pair of -CH2OH end groups), at fixed salt molality.
For a salt with nu ions: (d ln a_pm / d m_group)_{m_salt} = mu_x / (nu R T)   (exact, by definition).
Then for the duplex: (d ln K / d m_group)_{salt-coupling part} = S_K * mu_x / (nu R T).

Everything here is a LIMITING-SLOPE (low-concentration, <0.6 molal) estimate extrapolated
linearly to 40 wt%; the extrapolation is not reliable at high PEG.
Run: .venv/bin/python analysis/literature_check/q2_mux_estimate.py
"""
import math

R = 1.98720  # cal/(mol K)
T = 298.15
RT = R * T  # cal/mol
M_EO, M_W = 44.053, 18.015
SK = 3.5  # our d lnK / d ln a_pm at ~35 mM Na+

# ---------------------------------------------------------------- literature inputs
# Knowles et al. 2015 Biochemistry 54:3528, Table 2B (VPO, 25 C, <0.6 m), cal mol^-1 molal^-1 per ion.
beta_I = {'Na+': (4.66, 2.0), 'K+': (3.22, 3.4), 'Cl-': (37.6, 2.4), 'HPO4--': (234.0, 5.5)}
beta_2E = {'Na+': (4.51, 2.7), 'K+': (-3.47, 2.6), 'Cl-': (19.3, 2.5), 'HPO4--': (79.3, 5.3)}
# Knowles et al. 2011 PNAS SI Table S3 (VPO, 25 C): KCl, kcal/(mol molal EO)
knowles2011 = {'PEG1000': (37.8, 1.0), 'PEG2000': (32.4, 1.0)}  # in cal
# Annunziata group (McAfee, Zhang, Annunziata 2014, as quoted in Annunziata 2024 Molecules 29:3618):
# preferential hydration of PEG 20k in NaCl: 2.4 excess water molecules per EO group (25 C).
nuW_NaCl = 2.4
# Annunziata 2022 IJMS 23:13710: tyloxapol (PEG-based micelle) ~7 water/EO in Na2SO4, ~8-9 in MgSO4.
nuW_Na2SO4_tylox = 7.0

salts = {'NaCl': {'Na+': 1, 'Cl-': 1}, 'KCl': {'K+': 1, 'Cl-': 1},
         'Na2HPO4': {'Na+': 2, 'HPO4--': 1}}


def mux(beta, salt):
    v = sum(n * beta[i][0] for i, n in salt.items())
    e = math.sqrt(sum((n * beta[i][1]) ** 2 for i, n in salt.items()))
    return v, e


def one_plus_eps_NaCl(m):
    """1 + d ln gamma_pm / d ln m for NaCl, Pitzer (Pitzer & Mayorga 1973 parameters, 25 C)."""
    Aphi, b, a = 0.3915, 1.2, 2.0
    b0, b1, Cg = 0.0765, 0.2664, 0.00127 * 1.5

    def lng(mm):
        I = mm; s = math.sqrt(I)
        f = -Aphi * (s / (1 + b * s) + 2 / b * math.log(1 + b * s))
        x = a * s
        B = 2 * b0 + 2 * b1 / x ** 2 * (1 - (1 + x - x ** 2 / 2) * math.exp(-x))
        return f + mm * B + mm ** 2 * Cg
    h = 1e-5
    return 1 + (lng(m * (1 + h)) - lng(m * (1 - h))) / (math.log(1 + h) - math.log(1 - h))


print(f"RT = {RT:.1f} cal/mol\n")
print("1) mu_x per molal EO (interior group), cal/(mol salt * molal EO)")
for s, comp in salts.items():
    v, e = mux(beta_I, comp)
    nu = sum(comp.values())
    print(f"   {s:8s} Knowles2015 beta_I sum: {v:6.1f} +- {e:4.1f}   -> d ln a_pm/d m_EO = {v/(nu*RT):.4f}")
for k, (v, e) in knowles2011.items():
    print(f"   KCl      Knowles2011 VPO {k}: {v:6.1f} +- {e:4.1f}   -> d ln a_pm/d m_EO = {v/(2*RT):.4f}")
for m in (0.1, 0.25, 0.5, 1.0):
    ope = one_plus_eps_NaCl(m)
    v = nuW_NaCl * 2 * RT * ope / (1000 / M_W)
    print(f"   NaCl     Annunziata nu_W=2.4/EO, m_salt={m:4.2f} (1+eps={ope:.3f}): mu_x = {v:5.1f}")
v = nuW_Na2SO4_tylox * 3 * RT * 0.85 / (1000 / M_W)
print(f"   Na2SO4   tyloxapol nu_W~7/EO, (1+eps)~0.85: mu_x ~ {v:5.0f}  (cf. Na2HPO4 from beta_I: {mux(beta_I, salts['Na2HPO4'])[0]:.0f})")

print("\n2) DOSY / Bielec 'Na+ binding' expressed as an equivalent mu_x contribution, -RT*K_b (cal per molal EO)")
for kb in (0.05, 0.17, 0.49):
    print(f"   K_b = {kb:4.2f} M^-1 per EO -> mu_x(binding) = {-RT*kb:7.1f}")
print(f"   compare net measured beta_I(Na+) = {beta_I['Na+'][0]} +- {beta_I['Na+'][1]}, NaCl total {mux(beta_I, salts['NaCl'])[0]:.1f}")

print("\n3) Crowder-level estimate: Delta ln a_pm and Delta ln K = SK*Delta ln a_pm (linear, limiting slope)")
crowders = [('EG', 62.07), ('PEG200', 200), ('PEG400', 400), ('PEG1000', 1000), ('PEG6000', 6000), ('PEG35000', 35000)]
print("   electrolyte proxy: NaCl (lower bound-ish) and Na2HPO4 (upper bound-ish); mu_x per molecule = beta_2E + (n-1) beta_I")
print(f"   {'crowder':9s} {'wt%':>4s} {'m_PEG':>8s} {'m_EOeq':>7s} | {'dln a NaCl':>10s} {'dlnK':>6s} | {'dln a Na2HPO4':>13s} {'dlnK':>6s}")
for name, M in crowders:
    if name == 'EG':
        nI = 0.0
    else:
        nI = (M - M_W) / M_EO - 1
    for w in (0.10, 0.20, 0.40):
        mP = w / (1 - w) * 1000 / M
        out = []
        for s in ('NaCl', 'Na2HPO4'):
            comp = salts[s]; nu = sum(comp.values())
            mu_mol = mux(beta_2E, comp)[0] + nI * mux(beta_I, comp)[0]
            dln = mu_mol * mP / (nu * RT)
            out += [dln, SK * dln]
        print(f"   {name:9s} {w*100:4.0f} {mP:8.3f} {mP*(nI+1):7.2f} | {out[0]:10.3f} {out[1]:6.2f} | {out[2]:13.3f} {out[3]:6.2f}")

print("\n4) Our buffer (20 mM NaPi pH 7.4, ~80% HPO4(2-), ~20% H2PO4(-), Na+ ~36 mM):")
print("   H2PO4- beta not available; if the DNA responds to the salt as a whole the relevant mu_x lies")
print("   between NaCl-like (~42) and Na2HPO4-like (~243) cal/(mol molal EO); per ion pair, the anion term dominates.")
print("   Na+-only (extrathermodynamic split of Knowles 2015): d ln a_Na/d m_EO = "
      f"{beta_I['Na+'][0]/RT:.4f} +- {beta_I['Na+'][1]/RT:.4f}")

print("\n5) Morales et al. 2020 (IJCCE 39:145), NaClO4 in PEG4000+water, 298.15 K, Table 7:")
print("   DeltaG_t0 is on the mixed-solvent molality scale (mol/kg of PEG+water). Convert to water-molality")
print("   scale: DeltaG_t0(water scale) = DeltaG_t0(mixed) + nu*RT*ln(1-Y). Slope vs m_EO ~ infinite-dilution mu_x.")
for Y, dG in ((0.05, -0.06), (0.10, -0.15), (0.15, -0.29), (0.20, -0.43), (0.25, -0.48)):
    dGw = dG * 1000 / 4.184 + 2 * RT * math.log(1 - Y)  # cal/mol
    mEO = Y / (1 - Y) * 1000 / M_EO
    print(f"   Y={Y:.2f}: dG_mixed={dG:+.2f} kJ/mol -> dG_water-scale={dGw/1000:+.3f} kcal/mol; m_EO={mEO:5.2f}; dG/m_EO={dGw/mEO:+6.1f} cal/(mol molal EO)")
print("   (ClO4- is a salting-in anion; this shows the sign is anion-controlled, not a Na+ effect.)")
