"""mg_: przeliczenia 'pionowe' <-> 'poziome' dla danych literaturowych (liczby z pełnych tekstów, zob. mg_notes.md).
Na odcinku o stałym nachyleniu n = dlnK/dln c:  ΔlnK (pionowo) = n * Δln c_m (poziomo)."""
import math
# Kilburn 2013 JACS (PT): 18% PEG, Azoarcus, 37 C; ΔΔG/RT i n z tekstu
for ion, ddG, ns in [("MgCl2", 4.3, [5.0]), ("NaCl", 1.4, [1.0, 1.5])]:
    for n in ns:
        d = ddG / n
        print(f"Kilburn2013 {ion}: -ΔΔG/RT={ddG}, n={n} -> Δln c_m={d:.2f} (czynnik {math.exp(d):.2f})")
# Kilburn 2010 JACS (PT): c_m 0.7 -> 0.21 mM (20% PEG1000); przewidywanie z aktywności 0.7 -> 0.675 mM
print("Kilburn2010 obs Δln c_m =", round(math.log(0.7/0.21), 2), "; z aktywności MgCl2:", round(math.log(0.7/0.675), 3))
# Desai 2014 JBC (PT): 1.08 -> 0.79 mM (16% PEG1000); po przeliczeniu na molalność 0.91
print("Desai2014 Δln c_m =", round(math.log(1.08/0.79), 2), "; po korekcie objętości:", round(math.log(1.08/0.91), 2))
# Paudel 2014/2018 (PT)
for lab, a, b in [("Paudel2014 docking", 11, 1.6), ("Paudel2014 activity", 8, 2), ("Paudel2018 activity", 51, 20), ("Paudel2018 N-state", 42, 25)]:
    print(f"{lab}: Δln K_Mg = {math.log(a/b):.2f} (x{a/b:.1f})")
# Hong 2005 (PT): udział członu 'aktywności soli' w efekcie mocznika/GB przy stałej molalności KCl
for s, obs, corr in [("urea", -1.7, -0.34), ("GB", 2.1, -0.37)]:
    print(f"Hong2005 {s}: (dlnK/dm)_m={obs}, korekta SK*dln a/dm={corr} -> (dlnK/dm)_a={obs+corr:.1f}; |korekta|/|obs|={abs(corr/obs):.0%}")
# Lambert 2010 (PT) Tabela 1: SK i człon interakcji sól-TMAO (dSK/dm_TMAO)
for r, sk, inter in [("A-riboswitch", -4.03, 0.90), ("tar-tar*", -2.13, 0.38), ("TLR", -3.99, 1.00)]:
    print(f"Lambert2010 {r}: SK={sk}, dSK/dm={inter} -> wzgl. zmiana SK na 1 m TMAO = {inter/abs(sk):.0%}")
