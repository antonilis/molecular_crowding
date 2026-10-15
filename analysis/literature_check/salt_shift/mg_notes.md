# Przesunięcie krzywej solnej przez kosolut: RNA/Mg²⁺, białko–DNA, osmolity (zadania 3 i 4)

Stan na 2026-10-09. Wszystkie DOI sprawdzone przez Crossref API (`api.crossref.org/works/<DOI>`), wynik w
`mg_crossref_check.txt` (40/40 OK). **[PT]** = przeczytany pełny tekst (PMC: XML z NCBI efetch albo HTML,
tekst w `mg_ft/`), **[ABS]** = tylko abstrakt (Europe PMC / OpenAlex). Liczby podaję tylko z [PT], chyba że
oznaczono „(abstrakt)”. Przeliczenia „pion ↔ poziom” są nasze i są w `mg_shift_calc.py` →
`mg_shift_calc_output.txt`.

PDF-ów **nie udało się zapisać** do `literature/review_2026-10/salt_shift/`: PMC, Europe PMC render,
academic.oup.com, rnajournal, pnas.org i jbc.org zwracały stronę Cloudflare / proof-of-work. Pełne teksty
czytałem z XML/HTML PMC. Kopie tekstowe są w `analysis/literature_check/salt_shift/mg_ft/*.txt`.

Oznaczenia (z zadania): F(ln c) = ln K w buforze; **A** (aktywność soli): ln K(c,x) = F(ln c − φ(x));
**B** (addytywny): ln K(c,x) = F(ln c) + H(x). Test: ∂S_K/∂x = ∂²lnK/∂ln c∂x; w B jest zero, w A wynosi
−φ′(x)·F″(ln c). Dodaję trzecią możliwość **C**, bo pojawia się w literaturze: kosolut zmienia samą
stechiometrię jonową reakcji (ΔΓ±, n Hilla). Wtedy ∂S_K/∂x ≠ 0, chociaż aktywność soli się nie zmienia.

---

## 1. Tabela zbiorcza

| układ | kosolut | jon i zakres | # punktów soli | obserwabla | efekt vs sól | interpretacja autorów (poziomo/pionowo) | nasze odczytanie A/B | źródło | PT/ABS |
|---|---|---|---|---|---|---|---|---|---|
| Rybozym *Azoarcus*, 37 °C, 20 mM Tris | PEG 1000, 0–20–25% w/v | MgCl₂ 0–3.13 mM | miareczkowanie, kilkanaście | R_g (SAXS) → Hill: c_m, n | c_m 0.7 → 0.21 mM (20% PEG); Δln c_m = 1.2 | Pyta wprost, czy spadek c_m to wzrost aktywności MgCl₂ (poziomo). VPO + Gibbs–Duhem daje tylko 3.5% (0.7 → 0.675 mM). Wniosek: objętość wykluczona RNA (pionowo, przeliczona na c_m przez c_m = c_m,0·exp(Δlnγ/n)) | Formalnie A i B są tu nierozróżnialne (jedna sól, n stałe). Część poziomą „aktywnościową” oszacowano niezależnie na ~3% efektu | Kilburn 2010 JACS | PT |
| ten sam rybozym | PEG 1000, 3350, 8000 (0–18%), EG | MgCl₂ i **NaCl** | miareczkowania (tab. S1/S2 nieczytane) | R_g → c_m, n | n stałe względem PEG i M_w: **n ≈ 5.0 ± 0.1 (Mg), 1–1.5 (Na)**. ΔΔG(18% PEG) = −4.3 ± 0.3 RT (Mg) i −1.4 ± 0.2 RT (Na) | ΔΔG = −nRT ln(c_m,0/c_m,C), czyli przesunięcie poziome zamienione na pionowe. Różnica Mg vs Na: według autorów stan rozwinięty zależy od soli. Objętość wykluczona | **∂n/∂x ≈ 0**, więc na tym odcinku A ≡ B. Ale: przesunięcie poziome jest **prawie takie samo dla obu soli**, Δln c_m = 0.86 (Mg) vs 0.93–1.40 (Na). Stosunek ΔΔG_Mg/ΔΔG_Na ≈ 3 ≈ n_Mg/n_Na. Tego oczekuje A (wspólny czynnik na osi c), nie czyste B (ten sam ΔΔG). Nasza interpretacja, nie autorów | Kilburn 2013 JACS | PT |
| ten sam rybozym (aktywność) | PEG 1000 16%, PEG 8000, Ficoll; sukroza | MgCl₂ ~0.1–15 mM | miareczkowanie | frakcja aktywna → c_m | c_m 1.08 → 0.79 mM (16% PEG1000); po przeliczeniu na molalność 0.91 mM | „One possibility is that PEG simply increases the activity of the Mg²⁺ ions” — odrzucone z odwołaniem do Kilburn 2010. Objętość wykluczona | Część efektu (Δln c 0.31 → 0.17) to sama zmiana skali stężeń (objętość PEG). Kolejny przykład korekty „poziomej” | Desai 2014 JBC | PT |
| ten sam rybozym (SAXS, T) | PEG 1000 10, 18% | MgCl₂ 0–5 mM, 5–55 °C | ~6 | R_g → K(T) | PEG przesuwa przejście do niższego Mg i czyni je bardziej kooperatywnym w wysokiej T; T_m w 0.4 mM bez PEG = T_m w 0.3 mM + 10% PEG | entropia, objętość wykluczona | Wzrost kooperatywności = ∂n/∂x ≠ 0 w wysokiej T (A lub C) | Kilburn 2016 NAR | PT |
| mutanty *Azoarcus* | PEG 1000 18% | MgCl₂ | miareczkowanie | SAXS, aktywność → C_m, n | PEG kompensuje mutacje; ΔG = −nRT ln(C_ref/C_m) | objętość wykluczona | jak wyżej (jedna sól) | Lee 2015 NAR | PT (przegląd) |
| rybozym CPEB3, 25 mM HEPES, 100 mM KCl | PEG 200, PEG 8000, dekstran 10 (do 30%); Dextran 40/70, Ficoll 70 | Mg²⁺ 0.1, 0.5, 10 mM | **3** | k_obs cięcia | 0.1 mM: k 0.0021 → 0.011 (PEG200), 0.012 (PEG8000) min⁻¹; 0.5 mM: wzrost do wartości z 10 mM; **10 mM: brak efektu** (0.040 vs 0.036–0.044 min⁻¹) | „reduce the Mg²⁺ requirement” (poziomo, opisowo). VPO: dodatki i Mg²⁺ zmieniają a_w niezależnie, więc „additives are not significantly influencing local Mg²⁺ concentration” | Zanik efektu przy wysokim Mg jest zgodny z A na nasyconej krzywej, ale obserwabla (szybkość) też się nasyca, więc nie rozstrzyga | Strulson 2013 Biochemistry | PT |
| tRNA^Phe, 10 mM kakodylan, 140 mM KCl | 20% PEG 1–20k, dekstrany, Ficoll; 2 m osmolity, 20% PEG 200 | Mg²⁺ 0, 0.5, 2, 10 mM | **4** | T_m, dA/dT_max, ΔH_vH | Crowdery podnoszą T_m średnio o 2.7 °C (maks. 6.5 °C) przy 0–2 mM; **przy 10 mM brak efektu na T_m**. Przy 0 mM duże crowdery obniżają ΔH_vH, przy 0.5 mM podnoszą (×1.4), przy 2 mM ×1.1 | „biological Mg²⁺ allows cooperativity to be enhanced”; brak formalnego modelu | Efekt zależny od c, malejący do zera ⇒ nie czyste B. A możliwe, jeśli dT_m/dlog Mg maleje przy 10 mM (T_m przy 10 mM nie odczytane) | Strulson 2014 RNA | PT |
| tRNA^Phe i fragmenty spinek | PEG 200–20000, 0–40% | Mg²⁺ 0, 0.5, 2 mM; 140 mM K⁺ | 3 | T_m (UV, ILP) | 40% PEG200: T_m sumy spinek −14 °C (0.5 mM) i −12 °C (2 mM), a T_m całego tRNA prawie bez zmian (65.1 vs 64.8 °C przy 2 mM, 20% PEG200) | destabilizacja II-rzędowej + stabilizacja III-rzędowej | Dwa przeciwne człony pionowe; zależność od Mg słaba | Leamy 2017 Biochemistry | PT |
| rybozym spinkowy, 50 mM Tris | PEG 8000 20% (PEG 3350 słabiej) | Mg²⁺ (miareczkowanie) | miareczkowanie | smFRET (frakcja zadokowana), cięcie | K_Mg dokowania 11 → 1.6 mM (×6.9); aktywność 8 → 2 mM (×4) | „reduce the magnesium requirement”, objętość wykluczona | jedna sól, n nie podane w tekście głównym → test niemożliwy | Paudel & Rueda 2014 JACS | PT |
| intron grupy II D135-L14, 80 mM MOPS, 500 mM KCl, 42 °C | PEG 1000/8000/35000, 10% | Mg²⁺ do 100 mM | miareczkowanie | k_obs, frakcja stanu N (smFRET) | K_Mg aktywności 51 → 20 mM; stanu N 42 → 25 mM | „crowding reduces Mg²⁺ requirement” | jak wyżej | Paudel 2018 PNAS | PT |
| rybozym hammerhead | PEG i inne kosoluty 20 wt% | MgCl₂ (też Na⁺) | ≥2 | szybkość | ×2.0–6.6 przy 10 mM MgCl₂, „much more at lower MgCl₂”; PEG obniża potrzebne Mg²⁺ i Na⁺ (abstrakt) | osmolit: stabilizuje III-rzędową, destabilizuje parowanie WC | brak danych do testu | Nakano 2009 JACS | ABS |
| 5 RNA (spinki, tar–tar*, BWYV, A-riboswitch, 58-mer rRNA), MOPS/K⁺ | 9 osmolitów 0–2 m (w tym **EG**, glicerol, TMAO, betaina, prolina) | K⁺ vs K⁺+Mg²⁺; **szerokie zakresy Mg²⁺** z/bez 2 m EG lub 1.84 m TMAO | wiele (Fig. 7) | 1/T_m → m-value | „The stabilizing influence of the osmolyte is strongly attenuated with increasing Mg²⁺”. Przy wysokim Mg²⁺ osmolit jest neutralny albo **destabilizuje** (A-riboswitch + EG). Dla 58-mer + TMAO efekt stały (addytywny). m(EG, BWYV, K⁺+Mg²⁺) = −0.86 kcal/mol/m | Wprost: „If such indirect effects [aktywność jonów] are negligible, then Mg²⁺-induced stabilization and osmolyte effects should be additive.” Wynik: **nieaddytywne**. Przyczyna: oddziaływanie osmolit–Mg²⁺ przy powierzchni RNA | Jawny test addytywności (∂S_K/∂x = 0 ⇔ B). **Odrzucony** dla większości par. Zmiana znaku efektu ze wzrostem Mg²⁺ na monotonicznej F wyklucza czyste A; 58-mer+TMAO zgodny z B | Lambert & Draper 2007 JMB | PT |
| A-riboswitch, tar–tar*, TLR, spinka; 58-mer | TMAO 0–~3 m | KCl 0.025–0.8 m (**macierz KCl × TMAO**) | wiele, globalny fit | 1/T_m → ln K | S_K: spinka −0.59, A-rs −4.03, tar–tar* −2.13, TLR −3.99. **Człon interakcji sól–TMAO (∂S_K/∂m_TMAO): 0.90 ± 0.13, 0.38 ± 0.06, 1.00 ± 0.11 m⁻¹** (spinka: przyjęte 0). Oddziaływanie TMAO–KCl z VPO znikome (korekta ~0.001 kcal/mol/m) | Rozkład linkage: (∂lnK/∂m₃)_a = (∂lnK/∂m₃)_m + S_K·μ₂₃/(2RT(1+ε±)). Wzrost aktywności soli wykluczony („ruling out the possibility that TMAO stabilizes RNA indirectly by increasing salt activity”). ∂S_K/∂x ≠ 0 tłumaczą zmianą gęstości ładunku stanu rozwiniętego | **To jest dokładnie nasz test mieszanej pochodnej, już wykonany.** ∂S_K/∂x ≠ 0 przy φ ≈ 0 (VPO) ⇒ mechanizm **C**, nie A | Lambert, Leipply & Draper 2010 JMB | PT |
| lac represor–operator SymL | mocznik 0–3.2 m, betaina (GB) 0–3 m | KCl: 0.212 i 0.427 m K⁺ (mocznik); 0.427 m (GB) | **2** (mocznik) | K_TO (filtr) | Mocznik: nachylenie −1.7 ± 0.2 m⁻¹ **wspólne dla obu soli** (dane z 0.212 i 0.427 m dopasowane jedną krzywą). S_K = −7 ± 1 (−6.3 przy 0.4 M) | Jawny rozkład na człon „stałej aktywności soli” i człon aktywności (VPO): korekta −0.34 ± 0.09 (mocznik), −0.37 ± 0.10 (GB) ⇒ (∂lnK/∂m)_a = −2.1 i +1.8 | Na stałym S_K: A i B nierozróżnialne. Poziomy (aktywnościowy) człon policzony z VPO = **~20%** efektu kosolutu. ∂S_K/∂mocznik ≈ 0 w granicach błędu | Hong 2005 Biochemistry | PT |
| RNAP–λP_R (otwarty kompleks) | mocznik, GB | KCl, MgCl₂ (stałe) | 1 | k, K kroków | (∂lnK/∂m_urea)_a = −2.1 dla LacI (cyt.), korekty dla KCl i MgCl₂ z danych osmotycznych | „addition of either solute will reduce the thermodynamic activities of KCl and MgCl₂, and thereby favors steps … driven by a reduction in salt activity” | jak Hong 2005 (korekta liczona, nie mierzona przez biopolimer) | Kontur 2006 Biochemistry | PT |
| IHF–H′ DNA | GB | KCl, KF, KGlu (S_K −4.2 do −9.1) | kilka zakresów | ITC, FRET | dlnK/d[GB] = 2.6 → **2.7 ± 0.4 przy stałej aktywności KCl** | mała korekta aktywności soli | jak wyżej | Vander Meulen 2008 JMB | PT |
| lac represor–Osym / DNA niespecyficzne | anion (Cl⁻, Ac⁻, Glu⁻), kation | KX 0.125–0.36 M | wiele | K_obs | S_K(RD) = −9.8 ± 1.0 (KCl), −9.3 ± 0.7 (KGlu); S_K(RO) niezależne od [MCl] 0.125–0.3 M (abstrakt) | rozkład na kationy/aniony/wodę | (sól jako „kosolut” — odwrotny kierunek) | Ha 1992 JMB | ABS |
| lac represor, CAP, gal represor, EcoRI | osmolity (glicerol, EG, TEG, sukroza, betaina …) | stała sól | 1 | K | uwalnianie wody: 260 ± 32 (LacI–O1), 79 ± 11 (CAP–C1) (abstrakty) | stres osmotyczny (pionowo) | brak testu solnego | Fried 2002 JBC; Vossen 1997 Biochemistry | ABS |
| EcoRI | osmolity | sól, pH, a_w | kilka | k_d, K_nsp/sp | „weak salt-sensitivity … consistent with an osmotic, rather than electrostatic, action” (abstrakt) | rozdziela sól i osmolalność | sól częściowo działa jak osmolit (człon pionowy w S_K) | Sidorova & Rau 2001 JMB | ABS |
| DNA pol I (nick translation) | PEG, dekstran, Ficoll (wysokie stężenia) | siła jonowa (rosnąca) | wiele | aktywność | crowding „extends the range of ionic conditions supporting high … rates” (abstrakt) | wzrost wiązania polimerazy do DNA (objętość wykluczona) | jakościowo „poziomo” (tolerancja na sól), bez K | Zimmerman & Harrison 1987 PNAS; Zimmerman & Trach 1988 BBA | ABS |
| *E. coli in vivo* (lac represor, RNAP) | cytoplazma (osmolalność 0.1–1.1 Osm) | K⁺ 0.23 → 0.93 m, Glu⁻ 0.03 → 0.26 m | 1 seria | ekspresja β-gal | in vivo wiązanie „only weakly (if at all) dependent on intracellular ion concentration”, in vitro silnie (abstrakt) | kompensacja (glutaminian, później crowding: Cayley & Record 2004) | **białko–DNA jako „miernik” środowiska jonowego in vivo.** Wniosek: S_K in vivo ≈ 0, czyli ∂S_K/∂x ≠ 0 | Richey 1987 JBC; Cayley & Record 2004 JMR | ABS |
| dupleksy DNA (zob. Q4_Q7) | EG, PEG 200–8000 | Na⁺ | wiele | T_m | spadek −Δn_Na⁺ (S_K) z kosolutem: Spink & Chaires 1999 (EG, „modest decrease”), Karimata 2007 (PEG; zmiana znaku efektu PEG 8000 z [NaCl]), Nakano 2008, Nordstrom 2006 (abstrakty); Markarian 2010 [PT, Q4_Q7]: dwa reżimy, przejście przy a(Na⁺) ≈ 0.05 | woda / osmolit / „PEG buffers sodium ion binding” | ∂S_K/∂x ≠ 0 i zmiana znaku przy monotonicznej F ⇒ nie czyste A | Q4_Q7 i Q5_Q6 | ABS/PT |
| aptamer RNA | PEG | Mg²⁺ | ? | T_m, SHAPE, FCS | „crowding lowered the requirement of Mg²⁺” (abstrakt) | aktywność wody | brak danych | Kaur 2025 Biochemistry | ABS |
| transkrypcja/translacja in vitro | Ficoll 70/400, 20% | wolny Mg²⁺ 0.1–>2 mM | kilka | wydajność | wzmacnia poniżej ~2 mM, **hamuje powyżej 2 mM** (abstrakt) | „regulated by free Mg²⁺” | zmiana znaku z c ⇒ nie czyste A (jeśli F monotoniczna) | Ge & Xu 2020 Biotechnol Appl Biochem | ABS |

---

## 2. Odpowiedź na zadanie 4 (nowość)

**Pytanie:** czy ktoś (a) użył kwasu nukleinowego lub innego reportera jako miernika aktywności soli w roztworze
kosolutu, albo (b) jawnie rozdzielił przesunięcie poziome i pionowe krzywej solnej (lub sprawdził
∂S_K/∂x = 0)?

**(b): TAK, zrobiono to, i to wielokrotnie.** Trzeba to w pracy powiedzieć wprost i zacytować.

1. **Formalny rozkład „poziomy + pionowy” to standard szkoły Recorda i Drapera.** Pochodna po kosolucie
   przy stałej molalności soli = człon przy stałej **aktywności** soli (u nas H′, pionowo) + S_K·(∂ln a±/∂m₃)
   (u nas −S_K·φ′, poziomo). Hong 2005 [PT], eq. 5 i Appendix 1:
   (∂lnK/∂m₃)_{a4} = (∂lnK/∂m₃)_{m4} + S_K·ΔOsm₃₄/(2m₃m₄(1+ε±)). Stosują to Kontur 2006 [PT],
   Vander Meulen 2008 [PT], Knowles 2011 (PEG–KCl; zob. Q2) i Lambert 2010 [PT]. **Różnica wobec nas:**
   człon poziomy liczą z **VPO** (μ₂₃ kosolut–sól), a nie odczytują go z biopolimeru. Zakładają też S_K
   niezależne od kosolutu w granicy m₃ → 0.
2. **Mieszaną pochodną ∂S_K/∂x zmierzono jawnie.** Lambert, Leipply & Draper 2010 [PT] mierzyli macierz
   KCl (0.025–0.8 m) × TMAO dla czterech RNA. W globalnym fit jest osobny parametr „salt–osmolyte interaction”
   = ∂S_K/∂m_TMAO: +0.38 do +1.00 m⁻¹, czyli |S_K| maleje o 18–25% na 1 m TMAO. Równolegle VPO pokazało, że
   TMAO prawie nie zmienia aktywności KCl. Autorzy wnioskują więc mechanizm **C** (zmiana ΔΓ± przez
   kompaktowanie stanu rozwiniętego), a nie A. Lambert & Draper 2007 [PT] testowali to samo dla Mg²⁺ × osmolit
   (w tym EG) jako „addytywność”. Wynik: nieaddytywne.
3. **Pytanie „czy crowder obniża potrzebne [Mg²⁺] przez wzrost aktywności jonów?” postawiono wprost** dla PEG:
   Kilburn 2010 [PT] (VPO + Gibbs–Duhem: tylko 3.5%, przewidywane c_m 0.675 vs obserwowane 0.21 mM) i
   Desai 2014 [PT]. Odpowiedź autorów: nie, objętość wykluczona. Kilburn 2013 [PT] ma nawet projekt
   „dwóch osi solnych” (MgCl₂ i NaCl) przy stałym n.
4. Spadek S_K (ion uptake) przez EG/PEG dla dupleksów DNA jest znany (Spink & Chaires 1999, Nordstrom 2006,
   Karimata 2007, Nakano 2008 [ABS]; Markarian 2010 [PT]). Interpretuje się go jako efekt wody/osmolitu albo
   „buforowanie wiązania Na⁺”. Nikt nie testował go jako przesunięcia F wzdłuż ln c.

**(a): NIE znalazłem pracy, która używa kwasu nukleinowego (ani białka/koloidu) jako *miernika* φ(x),
czyli zmiany aktywności soli w roztworze kosolutu, odczytywanego z kształtu nieliniowej krzywej solnej.**
Najbliższe są:
- Record/Draper: aktywność soli mierzona VPO, biopolimer tylko „korygowany”.
- Richey 1987 [ABS] i Cayley & Record 2004 [ABS]: wiązanie białko–DNA in vivo jako sonda środowiska
  jonowego. Wniosek to kompensacja (glutaminian, crowding „buffers … against changes in K⁺”), bez pomiaru
  aktywności.
- Kilburn 2013 [PT]: dwa kationy dają podobne Δln c_m (nasze przeliczenie). Autorzy tego tak nie czytają.
- Własna wcześniejsza praca grupy (Bielec 2022 JPCL, DOI 10.1021/acs.jpclett.1c03596; omówiona w Q2/Q4_Q7):
  interpretuje efekt crowdera jako sekwestrację Na⁺ (φ > 0), ale bez testu poziomy vs pionowy.

Wyszukiwania (Europe PMC, OpenAlex, Crossref) z frazami „horizontal shift / salt axis”, „effective/apparent
ionic strength” + crowder, „probe/indicator of salt activity” + DNA, „salt–osmolyte interaction” nie dały
trafień poza wymienionymi.

**Jak uczciwie sformułować nowość.** Nie „pierwszy test ∂S_K/∂x” (Lambert 2010 to zrobił), ani nie
„pierwsza separacja efektu aktywności soli” (Hong 2005 i dalsze). Bronią się:
(i) użycie **krzywizny / maksimum F** (u nas maksimum przy ~150–175 mM, S_K od ~3.5 do ~1) do rozróżnienia A/B.
W A przesunięcie poziome przy maksimum daje przewidywalną **zmianę znaku** efektu kosolutu i punkt
przecięcia krzywych blisko maksimum. Na monotonicznej F (wszystkie prace wyżej) taki test jest niemożliwy.
(ii) Odczyt φ(x) z samego DNA dla szeregu EG → PEG 35k i porównanie z niezależnym DOSY.
(iii) A ma jedną funkcję φ(x), która musi jednocześnie wyjaśnić ΔlnK przy każdym c i
∂S_K/∂x = −φ′F″. To test nadokreślony. Lambert 2010 pokazuje, że ∂S_K/∂x ≠ 0 bywa mechanizmem C, więc
**sama niezerowa mieszana pochodna nie dowodzi A**. Trzeba sprawdzić ilościową zgodność z F″.

Recenzent prawie na pewno przywoła Record/Draper (korekta na stałą aktywność soli z VPO) i Knowles 2011
(μ₂₃ PEG–KCl > 0, czyli φ < 0, zob. Q2). Trzeba to omówić.

---

## 3. Szczegóły prac

### 3.1 Woodson i współpracownicy (PEG, Mg²⁺/Na⁺, *Azoarcus*)
- **Kilburn 2010 [PT].** 0.4 mg/ml RNA, 20 mM Tris pH 7.5, 37 °C, PEG 1000 0–25% w/v, MgCl₂ 0–3.13 mM.
  Fit Hilla (eq. 2) do R_g. c_m = 0.7 mM (0% PEG) → 0.21 mM (20%), mały wzrost między 0 a 5%. VPO: a_w
  w MgCl₂ maleje tak samo w 10 i 18% PEG. „The molality of 1 mM MgCl₂ in 20% PEG is 20% higher …
  corresponds to a difference in chemical potential of 3.5% … would shift the transition midpoint from 0.7 mM
  to 0.675 mM.” Modele objętości wykluczonej (kule 60/31 Å) przewidują 0.25 mM. Przeliczenie efektu
  pionowego na poziomy: c_m = c_m,0·exp(Δln γ/n) („equivalent to that used by Minton … urea”). Ta sama
  algebra co nasze A ↔ B na odcinku stałego S_K. Parametry Hilla (tab. S1) nieczytane.
- **Kilburn 2013 [PT].** PEG 1000, 3350, 8000; EG. Miareczkowania MgCl₂ i NaCl (eq. 2 z log c dla NaCl).
  „The steepness of the transitions, n, appears constant with respect to both PEG concentration and M_W …
  n ≈ 5.0 ± 0.1 for MgCl₂ and 1–1.5 for NaCl.” „A constant value of n indicates that the number of Mg²⁺
  ions required … is independent of the crowder content.” 20 mM Tris daje 27-krotny nadmiar Cl⁻ przy 0.6 mM
  MgCl₂, co jest warunkiem Leipply & Draper 2010 na stały współczynnik aktywności. ΔΔG(18%) = −4.3 RT (Mg)
  i −1.4 RT (Na). Autorzy: „the stabilization is additive with that from MgCl₂ and NaCl, at least within the
  transition regions”, w odróżnieniu od Lambert & Draper 2007. EG przesuwa c_m mniej niż PEG.
  **Nasze odczytanie:** Δln c_m = 0.86 (Mg), 0.93–1.40 (Na). Wspólne przesunięcie na osi ln c dla dwóch
  różnych soli to sygnatura A (lub objętości wykluczonej działającej jak czynnik na stężenie jonów). Autorzy
  wybierają objętość wykluczoną RNA, a różnicę Mg/Na tłumaczą stanem rozwiniętym.
- **Desai 2014 [PT].** Aktywność: c_m 1.08 ± 0.03 → 0.79 ± 0.01 mM (16% PEG 1000); w skali molalnej
  0.91 mM. PEG 1000 i 8000 dają ten sam c_m (0.938 vs 0.91 mM).
- **Kilburn 2016 [PT].** 0–5 mM MgCl₂, 0/10/18% PEG 1000, 5–55 °C. PEG zwiększa kooperatywność w wysokiej T.
  ΔH° −27.6 (0%), −27.9 (10%), −22 ± 3 (18%) kcal/mol.
- **Lee 2015 [PT, przegląd].** 18% PEG 1000; ΔG = −nRT ln(C_ref/C_m); nie wnosi nic do testu A/B.

### 3.2 Bevilacqua i współpracownicy
- **Strulson 2013 [PT].** Rybozym CPEB3, 25 mM HEPES pH 8, 100 mM KCl. Liczby w tabeli. VPO (0–10 mmol/kg
  Mg²⁺ × 10/20% dodatków): „the additives and magnesium affect the activity of water independently,
  supporting the notion that the additives are not significantly influencing local Mg²⁺ concentration.”
  Uwaga: to argument o braku oddziaływania Mg–PEG, ale VPO przy ≤10 mmol/kg ma małą czułość.
- **Strulson 2014 [PT].** tRNA^Phe, 140 mM KCl. T_m bez dodatków: 53.5 (0 mM), 57.5 (0.5 mM), 65.0 °C (2 mM).
  Pozostałe liczby w tabeli. Jedyna praca z ≥4 punktami Mg i wieloma crowderami. Wartości T_m przy 10 mM
  (Suppl.) nieodczytane, więc nie da się policzyć φ.
- **Leamy 2017 [PT].** Liczby w tabeli. Przeciwne znaki efektów na II- i III-rzędową strukturę.

### 3.3 Rueda
- **Paudel & Rueda 2014 [PT]:** PEG 8000 (MW 8000), „does not interact directly with RNA”; 20% PEG.
  K_Mg dokowania 11 → 1.6 mM, aktywność 8 → 2 mM. **Paudel 2018 [PT]:** 10% PEG 8000, 500 mM KCl, 42 °C;
  K_Mg 51 → 20 mM (aktywność), 42 → 25 mM (stan N). Interpretacja: objętość wykluczona. Brak dyskusji
  aktywności jonów.

### 3.4 Draper
- **Lambert & Draper 2007 [PT].** Osmolity „could … affect RNA stability indirectly, by modulating the
  thermodynamic activity of the ions”. Test addytywności (Fig. 6–7), wynik nieaddytywny. Cytują, że zależność
  od soli jednowartościowej dla DNA w EG/betainie/moczniku zmienia się „to an extent that is at the edge of
  statistical significance” (Spink & Chaires; Hong 2004). EG + BWYV przy K⁺/Mg²⁺: −0.86 kcal/mol/m,
  saturacja.
- **Lambert, Leipply & Draper 2010 [PT].** Zob. tabela i pkt 2. Kluczowe równania: 2a, 2b (linkage przy
  stałych aktywnościach), 3 (2ΔΓ± = S_K/(1+ε±)), 5a/5b (korekta na stałą aktywność KCl przez μ_KCl–TMAO).
  Spinka: S_K −0.59, m-value −0.05 ± 0.04, czyli TMAO prawie neutralny. 58-mer: m 1.1 (400 mM K⁺) i
  1.4 kcal/mol/m (600 mM K⁺), więc efekt TMAO rośnie z [K⁺]; przy 1 m K⁺ potrzeba tylko ~3 m TMAO.
- **Leipply & Draper 2010 [ABS + cytat w Kilburn 2013]:** n Hilla ≈ Γ₂₊ tylko przy stałym współczynniku
  aktywności Mg²⁺ (~30-krotny nadmiar soli 1:1) i stałym Γ₂₊. **Grilley 2006 [ABS]:** fluorescencyjny
  barwnik HQS jako miernik aktywności/wolnego Mg²⁺ i ΔG oddziaływania Mg–RNA. To miernik aktywności jonu,
  ale bez kosolutu. **Draper 2013 [PT, przegląd]:** linkage jonów, osmolitów i wody; TMAO „mimics the effect
  of Mg²⁺”.

### 3.5 Record (białko–DNA, osmolity, sól)
- **Hong 2005 [PT].** Liczby w tabeli. Filtr, 15 punktów na izotermę, K⁺ 0.212 i 0.427 m. Bufor zawiera
  0.752 m glicerolu (BB), co jest kolejnym kosolutem. Tab. A1: S_K·ΔOsm₃₄/(2m₃m₄(1+ε±)) = −0.35/−0.33
  (mocznik, 0.212/0.427 m), −0.43/−0.34/−0.42 (GB). Krzywizna przy wysokim GB (maksimum ~3.5 m GB).
- **Kontur 2006, Vander Meulen 2008 [PT].** Ten sam formalizm. Vander Meulen podaje też, że S_K kulombowskie
  powinno być stałe względem [salt], a odchylenia przypisuje członowi „Hofmeister-osmotic” soli.
- **Ha 1992 [ABS].** S_K lac represora, wpływ anionów (Glu⁻ jako anion „obojętny”).
- **Richey 1987, Cayley & Record 2004 [ABS].** In vivo, zob. tabela.
- **Knowles 2011 [PT, w Q2/Q4_Q7].** PEG–KCl μ₂₃ > 0 z VPO i korekta na stałą aktywność KCl dla dupleksu
  DNA. To jest bezpośredni poprzednik dla PEG (φ < 0, czyli „salt exclusion”).

### 3.6 Inne
- **Zimmerman & Harrison 1987, Zimmerman & Trach 1988 [ABS]:** crowding poszerza zakres siły jonowej
  aktywności polimerazy. Klasyczny opis „poziomy”, interpretowany jako wzrost wiązania.
- **Goobes 2003 [ABS]:** „metabolic buffering”: efekt crowdingu rośnie, gdy warunki pogarszają się
  (m.in. niska sól). Czyli efekt zależy od c (∂S_K/∂x ≠ 0).
- **Jiao 2010 [ABS], Mikles 2014 [ABS]:** białko–białko i białko–DNA w PEG. Brak serii solnych w abstraktach;
  nieistotne dla testu.
- **Sidorova & Rau 2001 [ABS]**, **Fried 2002, Vossen 1997, Garner & Rau 1995 [ABS]:** stres osmotyczny przy
  stałej soli. Sidorova: słaba zależność od soli „osmotic, rather than electrostatic”.

### 3.7 Wnioski dla naszego układu
1. W literaturze RNA/Mg²⁺ „crowder obniża potrzebne Mg²⁺” jest niemal zawsze interpretowane jako
   **objętość wykluczona** (pionowo, przeliczona na oś c przez n Hilla). Hipotezę aktywności jonów
   rozważano i odrzucano na podstawie VPO (Kilburn 2010, Desai 2014, Strulson 2013, Lambert 2010).
   Nikt nie proponował sekwestracji Mg²⁺/Na⁺ przez PEG.
2. Gdzie zmierzono S_K w obecności kosolutu, jest ono albo stałe (Kilburn 2013: n; Hong 2005: mocznik), albo
   zmienia się bez zmiany aktywności soli (Lambert 2010: C). Oba wyniki trzeba uwzględnić przy
   interpretacji naszego ∂S_K/∂x.
3. Znak: wszystkie VPO dla PEG/TMAO/GB/mocznik + sól 1:1 dają |φ| małe. Dla PEG–KCl φ < 0 (Knowles 2011),
   czyli przeciwnie niż sekwestracja. Zgodne z Q2.
4. Zmiana znaku efektu kosolutu ze stężeniem soli (Lambert & Draper 2007 przy Mg²⁺; Karimata 2007,
   Markarian 2010 przy Na⁺; Ge & Xu 2020) przy monotonicznej F wyklucza czyste A. Przy naszej F z maksimum
   zmiana znaku jest natomiast **przewidywaniem** A. To najmocniejszy dyskryminator, jaki mamy.

---

## Bibliografia

Format: autorzy (rok). Tytuł oryginalny. Czasopismo tom, strony. DOI. [weryfikacja; dostęp]

1. Kilburn D., Roh J.H., Guo L., Briber R.M., Woodson S.A. (2010). Molecular Crowding Stabilizes Folded RNA Structure by the Excluded Volume Effect. J. Am. Chem. Soc. 132, 8690–8696. DOI 10.1021/ja101500g. [Crossref OK; PT, PMC2906142]
2. Kilburn D., Roh J.H., Behrouzi R., Briber R.M., Woodson S.A. (2013). Crowders Perturb the Entropy of RNA Energy Landscapes to Favor Folding. J. Am. Chem. Soc. 135, 10055–10063. DOI 10.1021/ja4030098. [Crossref OK; PT, PMC3773054; SI nieczytane]
3. Desai R., Kilburn D., Lee H.-T., Woodson S.A. (2014). Increased Ribozyme Activity in Crowded Solutions. J. Biol. Chem. 289, 2972–2977. DOI 10.1074/jbc.M113.527861. [Crossref OK; PT, PMC3908428]
4. Lee H.-T., Kilburn D., Behrouzi R., Briber R.M., Woodson S.A. (2015; Crossref: 2014 online). Molecular crowding overcomes the destabilizing effects of mutations in a bacterial ribozyme. Nucleic Acids Res. 43, 1170–1176. DOI 10.1093/nar/gku1335. [Crossref OK; PT, PMC4333387]
5. Kilburn D., Behrouzi R., Lee H.-T., Sarkar K., Briber R.M., Woodson S.A. (2016). Entropic stabilization of folded RNA in crowded solutions measured by SAXS. Nucleic Acids Res. 44(19), 9452–? (Crossref podaje tylko „gkw597”; tom i strona początkowa z URL PDF wydawcy, strona końcowa niezweryfikowana). DOI 10.1093/nar/gkw597. [Crossref OK; PT, PMC5100557]
6. Strulson C.A., Yennawar N.H., Rambo R.P., Bevilacqua P.C. (2013). Molecular Crowding Favors Reactivity of a Human Ribozyme Under Physiological Ionic Conditions. Biochemistry 52, 8187–8197. DOI 10.1021/bi400816s. [Crossref OK; PT, PMC3882164]
7. Strulson C.A., Boyer J.A., Whitman E.E., Bevilacqua P.C. (2014). Molecular crowders and cosolutes promote folding cooperativity of RNA under physiological ionic conditions. RNA 20, 331–347. DOI 10.1261/rna.042747.113. [Crossref OK; PT, PMC3923128]
8. Leamy K.A., Yennawar N.H., Bevilacqua P.C. (2017). Cooperative RNA Folding under Cellular Conditions Arises From Both Tertiary Structure Stabilization and Secondary Structure Destabilization. Biochemistry 56, 3422–3433. DOI 10.1021/acs.biochem.7b00325. [Crossref OK (56(27), 3422–3433); PT, PMC5542450]
9. Paudel B.P., Rueda D. (2014). Molecular Crowding Accelerates Ribozyme Docking and Catalysis. J. Am. Chem. Soc. 136, 16700–16703. DOI 10.1021/ja5073146. [Crossref OK; PT, PMC4277754]
10. Paudel B.P., Fiorini E., Börner R., Sigel R.K.O., Rueda D.S. (2018). Optimal molecular crowding accelerates group II intron folding and maximizes catalysis. Proc. Natl. Acad. Sci. USA 115, 11917–11922. DOI 10.1073/pnas.1806685115. [Crossref OK (115(47), 11917–11922); PT, PMC6255197]
11. Nakano S., Karimata H.T., Kitagawa Y., Sugimoto N. (2009). Facilitation of RNA Enzyme Activity in the Molecular Crowding Media of Cosolutes. J. Am. Chem. Soc. 131, 16881–16888. DOI 10.1021/ja9066628. [Crossref OK; ABS]
12. Lambert D., Draper D.E. (2007). Effects of Osmolytes on RNA Secondary and Tertiary Structure Stabilities and RNA-Mg²⁺ Interactions. J. Mol. Biol. 370, 993–1005. DOI 10.1016/j.jmb.2007.03.080. [Crossref OK; PT, PMC1995082 (HTML)]
13. Lambert D., Leipply D., Draper D.E. (2010). The Osmolyte TMAO Stabilizes Native RNA Tertiary Structures in the Absence of Mg²⁺: Evidence for a Large Barrier to Folding from Phosphate Dehydration. J. Mol. Biol. 404, 138–157. DOI 10.1016/j.jmb.2010.09.043. [Crossref OK; PT, PMC3001104]
14. Leipply D., Draper D.E. (2010). Dependence of RNA Tertiary Structural Stability on Mg²⁺ Concentration: Interpretation of the Hill Equation and Coefficient. Biochemistry 49, 1843–1853. DOI 10.1021/bi902036j. [Crossref OK; ABS (pełny tekst pobrany do mg_ft/, nieanalizowany poza cytatem w Kilburn 2013)]
15. Grilley D., Soto A.M., Draper D.E. (2006). Mg²⁺–RNA interaction free energies and their relationship to the folding of RNA tertiary structures. Proc. Natl. Acad. Sci. USA 103, 14003–14008. DOI 10.1073/pnas.0606409103. [Crossref OK; ABS]
16. Draper D.E. (2013). Folding of RNA tertiary structure: Linkages between backbone phosphates, ions, and water. Biopolymers 99, 1105–1113. DOI 10.1002/bip.22249. [Crossref OK; PT, PMC3845237]
17. Hong J., Capp M.W., Saecker R.M., Record M.T. Jr. (2005). Use of Urea and Glycine Betaine To Quantify Coupled Folding and Probe the Burial of DNA Phosphates in Lac Repressor−Lac Operator Binding. Biochemistry 44, 16896–16911. DOI 10.1021/bi0515218. [Crossref OK; PT, PMC2546872]
18. Kontur W.S., Saecker R.M., Davis C.A., Capp M.W., Record M.T. Jr. (2006). Solute Probes of Conformational Changes in Open Complex (RPo) Formation by Escherichia coli RNA Polymerase at the λPR Promoter: Evidence for Unmasking of the Active Site in the Isomerization Step and for Large-Scale Coupled Folding in the Subsequent Conversion to RPo. Biochemistry 45, 2161–2177. DOI 10.1021/bi051835v. [Crossref OK; PT, PMC2631401]
19. Vander Meulen K.A., Saecker R.M., Record M.T. Jr. (2008). Formation of a Wrapped DNA–Protein Interface: Experimental Characterization and Analysis of the Large Contributions of Ions and Water to the Thermodynamics of Binding IHF to H′ DNA. J. Mol. Biol. 377, 9–27. DOI 10.1016/j.jmb.2007.11.104. [Crossref OK; PT, PMC2336898]
20. Ha J.-H., Capp M.W., Hohenwalter M.D., Baskerville M., Record M.T. Jr. (1992). Thermodynamic stoichiometries of participation of water, cations and anions in specific and non-specific binding of lac repressor to DNA. Possible thermodynamic origins of the "glutamate effect" on protein-DNA interactions. J. Mol. Biol. 228, 252–264. DOI 10.1016/0022-2836(92)90504-D. [Crossref OK; ABS]
21. Hong J., Capp M.W., Anderson C.F., Saecker R.M., Felitsky D.J., Anderson M.W., Record M.T. Jr. (2004). Preferential Interactions of Glycine Betaine and of Urea with DNA: Implications for DNA Hydration and for Effects of These Solutes on DNA Stability. Biochemistry 43, 14744–14758. DOI 10.1021/bi049096q. [Crossref OK; ABS]
22. Knowles D.B., LaCroix A.S., Deines N.F., Shkel I., Record M.T. Jr. (2011). Separation of preferential interaction and excluded volume effects on DNA duplex and hairpin stability. Proc. Natl. Acad. Sci. USA 108, 12699–12704. DOI 10.1073/pnas.1103382108. [Crossref OK; PT (lokalnie, zob. Q2)]
23. Richey B., Cayley D.S., Mossing M.C., Kolka C., Anderson C.F., Farrar T.C., Record M.T. Jr. (1987). Variability of the intracellular ionic environment of Escherichia coli. Differences between in vitro and in vivo effects of ion concentrations on protein-DNA interactions and gene expression. J. Biol. Chem. 262, 7157–7164. DOI 10.1016/S0021-9258(18)48218-X. [Crossref OK; ABS]
24. Cayley S., Record M.T. Jr. (2004). Large changes in cytoplasmic biopolymer concentration with osmolality indicate that macromolecular crowding may regulate protein–DNA interactions and growth rate in osmotically stressed Escherichia coli K-12. J. Mol. Recognit. 17, 488–496. DOI 10.1002/jmr.695. [Crossref OK; ABS]
25. Sidorova N.Y., Rau D.C. (2001). Linkage of EcoRI dissociation from its specific DNA recognition site to water activity, salt concentration, and pH: separating their roles in specific and non-specific binding. J. Mol. Biol. 310, 801–816. DOI 10.1006/jmbi.2001.4781. [Crossref OK; ABS]
26. Fried M.G., Stickle D.F., Smirnakis K.V., Adams C., MacDonald D., Lu P. (2002). Role of Hydration in the Binding of lac Repressor to DNA. J. Biol. Chem. 277, 50676–50682. DOI 10.1074/jbc.M208540200. [Crossref OK; ABS (OA wg Unpaywall, pobranie zablokowane)]
27. Vossen K.M., Wolz R., Daugherty M.A., Fried M.G. (1997). Role of Macromolecular Hydration in the Binding of the Escherichia coli Cyclic AMP Receptor to DNA. Biochemistry 36, 11640–11647. DOI 10.1021/bi971193e. [Crossref OK; ABS]
28. Spink C.H., Chaires J.B. (1999; Crossref: 1998 online). Effects of Hydration, Ion Release, and Excluded Volume on the Melting of Triplex and Duplex DNA. Biochemistry 38, 496–508. DOI 10.1021/bi9820154. [Crossref OK; ABS]
29. Nordstrom L.J., Clark C.A., Andersen B., Champlin S.M., Schwinefus J.J. (2006). Effect of Ethylene Glycol, Urea, and N-Methylated Glycines on DNA Thermal Stability: The Role of DNA Base Pair Composition and Hydration. Biochemistry 45, 9604–9614. DOI 10.1021/bi052469i. [Crossref OK; ABS]
30. Karimata H., Nakano S., Sugimoto N. (2007). Effects of Polyethylene Glycol on DNA Duplex Stability at Different NaCl Concentrations. Bull. Chem. Soc. Jpn. 80, 1987–1994. DOI 10.1246/bcsj.80.1987. [Crossref OK; ABS]
31. Nakano S., Wu L., Oka H., Karimata H.T., Kirihata T., Sato Y., Fujii S., Sakai H., Kuwahara M., Sawai H., Sugimoto N. (2008). Conformation and the sodium ion condensation on DNA and RNA structures in the presence of a neutral cosolute as a mimic of the intracellular media. Mol. BioSyst. 4, 579–588. DOI 10.1039/B718806D. [Crossref OK; ABS]
32. Markarian M.Z., Schlenoff J.B. (2010). Effect of Molecular Crowding and Ionic Strength on the Isothermal Hybridization of Oligonucleotides. J. Phys. Chem. B 114, 10620–10627. DOI 10.1021/jp103213w. [Crossref OK; PT (zob. Q4_Q7)]
33. Goobes R., Kahana N., Cohen O., Minsky A. (2003). Metabolic Buffering Exerted by Macromolecular Crowding on DNA−DNA Interactions: Origin and Physiological Significance. Biochemistry 42, 2431–2440. DOI 10.1021/bi026775x. [Crossref OK; ABS]
34. Zimmerman S.B., Harrison B. (1987). Macromolecular crowding increases binding of DNA polymerase to DNA: an adaptive effect. Proc. Natl. Acad. Sci. USA 84, 1871–1875. DOI 10.1073/pnas.84.7.1871. [Crossref OK; ABS (PMC304543 tylko skan)]
35. Zimmerman S.B., Trach S.O. (1988). Macromolecular crowding extends the range of conditions under which DNA polymerase is functional. Biochim. Biophys. Acta 949, 297–304. DOI 10.1016/0167-4781(88)90155-8. [Crossref OK; ABS]
36. Kaur J., Jain R., Roychowdhury S., Roy R., Chattopadhyay K., Roy I. (2025). Influence of Magnesium Ions and Crowding Agents on Structure and Stability of RNA Aptamers. Biochemistry 64, 1233–1243. DOI 10.1021/acs.biochem.4c00468. [Crossref OK; ABS]
37. Ge X., Xu J. (2020; Crossref: 2019 online). Macromolecular crowding effects on transcription and translation are regulated by free magnesium ion. Biotechnol. Appl. Biochem. 67, 117–122. DOI 10.1002/bab.1827. [Crossref OK; ABS]
38. Jiao M., Li H.-T., Chen J., Minton A.P., Liang Y. (2010). Attractive Protein-Polymer Interactions Markedly Alter the Effect of Macromolecular Crowding on Protein Association Equilibria. Biophys. J. 99, 914–923. DOI 10.1016/j.bpj.2010.05.013. [Crossref OK; ABS]
39. Mikles D.C., Bhat V., Schuchardt B.J., McDonald C.B., Farooq A. (2014). Effect of osmolytes on the binding of EGR1 transcription factor to DNA. Biopolymers 103, 74–87. DOI 10.1002/bip.22556. [Crossref OK; ABS]
40. Bielec K., Kowalski A., Bubak G., Witkowska Nery E., Hołyst R. (2022). Ion Complexation Explains Orders of Magnitude Changes in the Equilibrium Constant of Biochemical Reactions in Buffers Crowded by Nonionic Compounds. J. Phys. Chem. Lett. 13, 112–117. DOI 10.1021/acs.jpclett.1c03596. [Crossref OK; PT (zob. Q2/Q4_Q7)]
41. Garner M.M., Rau D.C. (1995). Water release associated with specific binding of gal repressor. EMBO J. 14, 1257–1263. DOI 10.1002/j.1460-2075.1995.tb07109.x. [Crossref OK (14(6), 1257–1263); ABS]

Tomy i strony pozycji 2–5, 8, 10, 13–16, 18, 19, 36, 37 i 41 dodatkowo sprawdzone w pełnych metadanych
Crossref (dopisane na końcu `mg_crossref_check.txt`). Jedyny brak: strona końcowa Kilburn 2016 (poz. 5).
