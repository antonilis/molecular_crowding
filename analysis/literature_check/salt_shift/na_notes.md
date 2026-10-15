# Sól × kosolut: przesunięcie poziome (A) czy pionowe (B)? Przegląd prac z więcej niż jednym stężeniem soli

Stan na 2026-10-09. Wszystkie DOI sprawdzono przez Crossref API; wynik zapisano w `na_crossref_check.txt`.
- **[PT]** oznacza przeczytany pełny tekst (źródło podane), **[ABS]** tylko abstrakt.
- Liczby pochodzą tylko z [PT] albo są jawnie oznaczone jako wzięte z abstraktu.
- Przeliczenia oznaczone **[nasze]** są moje, nie autorów.
- Skrypty, CSV i wyniki leżą w `analysis/literature_check/salt_shift/na_*`. Pełne teksty XML są w `salt_shift/na_disc_ft/`.
- Rysunki źródłowe do digitalizacji to `na_*_source.jpg`. PDF/SI w wolnym dostępie są w `literature/review_2026-10/salt_shift/`.
- Uwaga techniczna: `Nakano2014_FEBSOpenBio_ribozyme_duplex_cosolutes_NaCl.pdf` i `na_tmp2.pdf` w tym katalogu to w rzeczywistości strony HTML, a nie PDF-y. Zostawiłem je nietknięte.

## 0. Rama interpretacyjna i co z niej wynika w praktyce

**Model A** (zmiana aktywności soli):
- ln K(c,x) = F(ln c − φ(x)), więc ΔlnK(c,x) ≈ −φ(x)·S_K(c).
- φ > 0 oznacza wiązanie kationu (spadek a±), φ < 0 oznacza wykluczenie soli (wzrost a±).

**Model B** (addytywny kosolut niejonowy):
- ln K = F(ln c) + H(x).
- Efekt kosolutu nie zależy od c, a S_K nie zależy od x.

**Model C** (spoza ramy A/B, ale potrzebny do opisu literatury): multiplikatywna zmiana S_K.
- Źródłem może być ε (Manning: S_K ∝ Δ(1/ξ) ∝ ε, gdy obie formy są w reżimie kondensacji) albo zmiana geometrii lub kondensacji na ssDNA.
- C zmienia nachylenie także na odcinku, na którym w samym buforze S_K ≈ const. A tego zrobić nie może.

**Trzy testy, którymi posługuję się niżej:**
1. **Kierunek efektu w funkcji soli.** W A efekt skaluje się z lokalnym S_K. Dla zwykłej wklęsłej F (S_K maleje z c) czysty A daje |efekt| malejący z solą. W B efekt jest stały.
2. **Przecięcie krzywych.** Dotyczy zmiany znaku efektu kosolutu przy zmianie soli na monotonicznym odcinku F.
   - Czysty A nie daje przecięcia: dla rosnącej F znak F(ln c − φ) − F(ln c) jest wszędzie równy znakowi −φ.
   - Czysty B też go nie daje, bo przesunięcie jest stałe.
   - Przecięcie wymaga A+B o przeciwnych znakach (np. H < 0 i φ < 0 przy malejącym S_K) albo C.
3. **Zmiana S_K na odcinku prawa potęgowego** (F liniowa w ln c) wyklucza A jako jej źródło. Zostaje C.

## 1. Tabela zbiorcza

Oznaczenia:
- „efekt” to ΔlnK (albo ΔΔG, ΔTm) kosolutu przy stałym x;
- S = d lnK/d ln c (= d logK/d log c);
- S_sec to sieczna między dwoma stężeniami soli;
- φ podaję w jednostkach ln c, chyba że zaznaczono inaczej.

| układ | kosolut | zakres soli | # pkt soli | obserwabla | efekt vs sól | φ (A) / H (B) / kierunek | rozstrzyga A/B? | źródło | dostęp |
|---|---|---|---|---|---|---|---|---|---|
| DNA 8-mer d(ATGCGCAT)₂, UV, 25 i 37 °C | PEG 200, 10–50 wt% | Na⁺ 0.122 i 1.022 M | 2 | ΔG°, ln K | Przy 25 °C destabilizacja większa przy 1 M (40 wt%: ΔlnK −7.4 vs −5.1; różnica −2.4 ± 1.6). Przy 37 °C (te same dane, inna ewaluacja) różnica −1.0 ± 1.5. W skali Δa_w obie sole leżą na jednej prostej. | Czysty A wymagałby φ = +6 … +30 (absurd), więc efekt jest w przeważającej części B: H(40%) ≈ −5 … −7. S_sec spada z +0.79 ± 0.57 do −0.32 ± 0.46 (25 °C), ale w granicach błędu. | **Tak co do dominacji B.** Domieszki A (\|φ\| ≲ 1–2) nie da się rozstrzygnąć. | Nakano 2004 SI Tab. S1–S2; Ghosh 2020 SI Tab. S5 | PT (lokalnie) |
| RNA 11-mer GAUUACGCCUG, UV, 37 °C | PEG 200 10/20/40 wt%; EG | Na⁺ 0.122 i 1.022 M | 2 | ΔG°37 | Destabilizacja przez PEG 200 rośnie z solą: 40%: ΔlnK −3.4 ± 0.4 (0.12 M) vs −5.0 ± 1.2 (1 M); 20%: −1.1 vs −1.8. EG 20 wt%: +1.5 kcal/mol przy 1 M vs ≈ +0.9 przy 0.12 M (ekstrapolowane z 10/15 wt%). | S_sec: 1.83 → 1.68 → 1.53 → 1.07 (0/10/20/40 wt%), czyli ∂S/∂x < 0. Kierunek przeciwny do czystego A z φ > 0. Zgodny z B + A(φ < 0) albo z B + C. | **Odrzuca czysty A (φ > 0) i czysty B** (trend monotoniczny, ok. 1.5σ na punkt). Nie rozróżnia A(φ < 0) od C. | Ghosh 2023 SI Tab. S8 | PT (SI z Europe PMC) |
| RNA 11-mer AUCCAGCGCGC, UV, 37 °C | 20 wt%: PEG 200/600/2000/8000/20000, EG, glicerol, alkohole, amidy, DMSO, THF, dioksan, dekstran 10k | NaCl 0.02–1 M | 7 | log K | PEG 8000: +0.77 (0.02 M) → +0.32 (0.2 M) jednostek log K; brak punktów > 0.2 M (rozdział faz). PEG 200: −0.5 (0.02 M) → −1.1 (1 M). DMF: −2.0 → −2.5. | Nachylenie w buforze 1.8, w PEG 8000 1.1 (autorzy); [nasze] 1.14 vs 1.54 na tym samym zakresie 0.02–0.2 M. FA i mocznik nie obniżają nachylenia. Nachylenie koreluje z 1/ε (r = 0.906), nie z a_w ani z lepkością. | **Tak: wyklucza czysty A i czysty B dla PEG 8000.** Efekt maleje z solą przy prawie prostym odcinku F (C albo B + A(φ < 0) o niewiarygodnej wielkości). Autorzy jawnie proponują człon ε·log[sól] (C). | Nakano 2014 FEBS Open Bio, Fig. 3B (zdigitalizowana), Fig. 4, Tab. 1 | PT (PMC4141205) |
| RNA 9-mer CAACGCAAG (+ dupleks GC, poli(A)/poli(U), tRNA), UV | ok. 20 kosolutów, 20 wt%, w tym PEG 8000, PEG 200 | NaCl, NH₄Cl, kationy tetraalkiloamoniowe, 0.02–1 M; MgCl₂ | kilka | log K, Tm | „The degree to which stability was dependent on salt concentration was correlated with reciprocals of the dielectric constants … regardless of interactions between the cosolutes and RNA”. Bez kosolutu: nachylenie 1.66 ± 0.05 (Na⁺). W PEG 8000 lub DME nachylenie dla TBACl staje się bardziej ujemne. | kierunek: \|S\| maleje dla Na⁺ (C) | **Jak wyżej (C).** Liczbowych nachyleń per kosolut dla NaCl nie ma w tekście (tylko na rysunkach). | Nakano 2016 BPJ | PT (PMC HTML) |
| DNA 15-, 25-, 35-mer, hipochromizm izotermiczny (35/55/67 °C) | PEG „4000” (Mw ≈ 3000) 0–30 wt% | a_Na⁺ ≈ 0.02–0.13 (Na⁺ 2–225 mM) | 8–10 na każdy % PEG | ln K_obs vs ln a_Na (a_Na z VPO) | **Przecięcie.** Przy ln a ≈ −3.7 PEG 30% daje ΔlnK ≈ +1.3, przy ln a ≈ −2.2 daje ≈ −1.2. Przecięcie przy a ≈ 0.05–0.08 (10–30 wt%). | S (35-mer, Fig. 7A): 2.96 → 1.35 (0 → 30 wt%), dS/dwt% = −0.060. 25-mer: 4.25 → 2.39. 15-mer: 2.62 → 1.79. W buforze F prawie liniowa (krzywizna −0.34). | **Tak: ani A, ani B.** Oś x to już aktywność z VPO (A „zdjęte” przez autorów), a nachylenie spada o 50% na prawie prostym odcinku F. To C, albo A o wielkości φ ≈ −2.4, sprzecznej z VPO. | Markarian & Schlenoff 2010, Fig. 6 i 7A (zdigitalizowane) | PT (manuskrypt autorski PMC2925268) |
| DNA dupleks i spinka, UV | PEG 200, PEG 8000 | NaCl, „różne” (zakres nieznany) | ? | Tm, ΔG | Według abstraktu: oba PEG destabilizują przy wysokim NaCl, a PEG 8000 „substantially increased the stability at low NaCl”. PEG obniża −ΔnNa⁺ (nachylenie) i nie zmienia −Δnw. | ∂S/∂x < 0, zmiana znaku efektu (przecięcie) | **Według abstraktu tak (jak Markarian):** nie czysty A, nie czysty B. Liczb brak. | Karimata 2007 BCSJ | ABS (pełny tekst zablokowany) |
| DNA 8-mer | PEG 200, PEG 8000 20 wt% | NaCl 1 M i 0.4 M | 2 | Tm | Abstrakt: ΔTm −11.8 °C (PEG 200), −1.5 °C (PEG 8000) przy 1 M. „when the NaCl concentration decreased from 1 M to 400 mM, the Tm increased in the presence of PEGs”: dwuznaczne. | nieokreślone | Nie | Karimata 2004 NASS | ABS |
| DNA, poli(dA)·poli(dT), tripleks; DSC/UV | EG (a_w), PEG | Na⁺, kilka stężeń (z abstraktu) | ? | Tm | Według abstraktu: „modest decrease in the extent of counterion binding … as water activity decreases” (EG) | ∂S/∂x < 0, mała | Słabo, kierunek C / A(φ < 0) | Spink & Chaires 1999 | ABS |
| DNA genomowe o różnym %GC | EG, mocznik, glicyna, sarkozyna, betaina | „increasing ionic strength” | ? | Tm | Według abstraktu: wzrost siły jonowej osłabia akumulację mocznika i glicyn (zależność od soli). Dla EG nie wymieniono. | dla mocznika/glicyn ∂S/∂x ≠ 0 | częściowo | Nordstrom 2006 | ABS |
| RNA: spinka (6 bp) + 3 struktury trzeciorzędowe | TMAO do kilku m | KCl 0.025–0.8 m, siatka KCl × TMAO | wiele | 1/Tm, ln K | Spinka: „the salt dependence of the hairpin stability was virtually unaffected by TMAO”. Struktury trzeciorzędowe mają niezerowy człon krzyżowy. | spinka: B. VPO: μ(TMAO–KCl) ≈ 0, więc φ ≈ 0. | **Tak, i autorzy zrobili to jawnie:** wykluczyli A pomiarem VPO, dla spinki wyszło B. | Lambert, Leipply, Draper 2010 | PT (PMC3001104, cytaty sprawdzone) |
| RNA: ta sama spinka + struktury trzeciorzędowe | mocznik 0–2.5 m | KCl, seria ± 2 m mocznika | kilka | ln K | „For each RNA, the salt-dependence is steeper (more negative) in the presence of urea by an average of ~20%. The m-value itself is therefore salt dependent” | \|S\| rośnie z mocznikiem (C, kierunek przeciwny do PEG) | **Tak:** autorzy skorygowali na aktywność mocznik–KCl, czyli „wyjęli” A. Zostaje człon krzyżowy (C), który przypisują stanowi rozwiniętemu. | Lambert & Draper 2012 | PT (PMC3505219) |
| DNA dodekamery | mocznik | stała molalność soli; seria w SI | ? | Tm | „urea has no effect on the salt dependence of DNA melting (supplemental)” | B | Według tekstu B. Liczb nie sprawdziłem (SI zablokowane). | Guinn 2013 JACS | PT (tekst główny); liczby NIEZWERYFIKOWANE |
| RNA, osmolity, Mg²⁺ (struktury trzeciorzędowe) | EG, TMAO, betaina, metanol, prolina, mocznik | Mg²⁺ szeroki zakres (stały KCl) | wiele | ΔG | „the stabilizing effects of Mg²⁺ and most of the osmolytes studied here are not additive”. Tylko TMAO jest addytywny. Dla soli jednowartościowej autorzy cytują efekt EG/betainy/mocznika „at the edge of statistical significance”. | nie-B dla Mg²⁺; dla K⁺ ≈ B | Tak (test addytywności), ale dla Mg²⁺ i struktur trzeciorzędowych | Lambert & Draper 2007 | PT (PMC HTML) |
| DNA genomowe | formamid 0–50%, mocznik 0–8 M | NaCl 0.035–0.88 M (i wyżej) | kilka | Tm | Według abstraktu: −0.60 °C/% formamidu, stałe w zakresie 0.035–0.88 M; powyżej „slightly greater” | B | Według abstraktu B | Hutton 1977 | ABS |
| DNA, betaina | betaina > 5 M | ? | ? | Tm | „does not greatly change the behavior of DNA as a polyelectrolyte” | ≈ B | słabo | Rees 1993 | ABS |
| spinka hTR i pseudowęzeł RNA | betaina, mocznik | NaCl 40 i 135 mM | 2 | Tm | Betaina stabilizuje strukturę trzeciorzędową przy 135 mM, a lekko destabilizuje przy 40 mM | zmiana znaku z solą | częściowo (dotyczy pseudowęzła) | Schwinefus 2007 | ABS |
| RNA (Azoarcus, Mg²⁺) | PEG 1000 0–20% w/v | Mg²⁺ (miareczkowanie) | wiele | punkt środkowy zwijania | Punkt środkowy Mg²⁺ spada z 0.7 do 0.21 mM, czyli przesunięcie wzdłuż osi soli | formalnie wygląda jak A, autorzy przypisują to objętości wyłączonej | nie (struktura trzeciorzędowa) | Kilburn 2010 | PT (w Q4_Q7) |
| tRNA^Phe, fragmenty | PEG, dekstran, Ficoll, osmolity | Mg²⁺ 0 / 0.5 / 2 mM (w 140 mM KCl) | 2–3 | Tm | Liczby są w tabelach-obrazach lub SI; struktura drugorzędowa „largely unaffected by Mg²⁺” | — | nie | Strulson 2014; Leamy 2017 | PT (bez liczb) |
| RNA, PEG 200 | PEG 200 20% | NaCl 0.01/0.1/1 M bez PEG; PEG tylko przy 1 M | 1 (dla PEG) | ΔG | **Nie jest to projekt 2D** | — | — | Gu 2013 | PT |
| RNA, PEG 200 | PEG 200 20 vol% | tylko 1 M NaCl | 1 | ΔG | nie 2D | — | — | Adams & Znosko 2019 | PT |
| A7U7p | EG, alkohole, DMSO itd. 10 mol% | dTm/dlog[Na⁺] = 17.4 °C (0.06–0.55 M) w wodzie; rozpuszczalniki przy 1 M | ? | Tm | z abstraktu nie wiadomo, czy nachylenie mierzono w mieszaninach | — | nie | Hickey & Turner 1985 | ABS |
| spinki CAG/CTG, smFRET | crowdery (rodzaj niepodany w abstrakcie) | NaCl, kilka stężeń | ? | K_fold | „Molecular crowding shifts the conformational equilibria … away from the hairpin state … with increasing NaCl”, czyli efekt crowdera maleje lub zmienia znak z solą | kierunek jak Markarian i Nakano 2014 | jakościowo nie-B | Martini 2025 | ABS |
| spinka DNA, smFRET | PEG 400, PEG 8000 10 wt% | „lower salt” (Fig. S9) | ? | K | „closed state … more favored at lower salt concentrations” w PEG 8000 | jak wyżej | jakościowo | Baltierra-Jasso 2015 | PT (bez SI) |
| DNA–DNA (ITC, UV) | crowding (PEG) | warunki „sprzyjające” vs „niesprzyjające” asocjacji | ? | K | Według abstraktu: efekt crowdingu jest marginalny w warunkach sprzyjających (np. wysoka sól) i rośnie, gdy warunki się pogarszają („buffering”) | efekt maleje z solą | jakościowo nie-B | Goobes 2003 | ABS |
| dupleks i spinka DNA/RNA, Z-DNA | PEG i małe kosolutanty | NaCl (z abstraktu) | ? | Tm | Według abstraktu: kondensacja Na⁺ (nachylenie) przy tworzeniu dupleksu i spinki maleje w PEG; dla przejścia B → Z się nie zmienia | ∂S/∂x < 0 | nie (bez liczb) | Nakano 2008 Mol BioSyst | ABS |

## 2. Szczegóły prac kluczowych

### 2.1 Nakano et al. 2004 JACS (SI Tab. S1/S2) i Ghosh et al. 2020 PNAS (SI Tab. S5): DNA 8-mer, PEG 200 przy 0.1 i 1 M NaCl [PT]

**Warunki.** d(ATGCGCAT)₂, NaCl + 10 mM Na₂HPO₄ (pH 7.0) + 1 mM Na₂EDTA. Na⁺ całkowity wynosi 0.122 lub 1.022 M (konwencja autorów, Ghosh 2023).
- Nakano 2004 podaje ΔG°25.
- Ghosh 2020 Tab. S5 podaje te same pomiary przeliczone na ΔG°37.

**Obliczenia.** Skrypt `na_nakano2004_AB.py`, wynik w `na_nakano2004_AB_output.txt`.

| wt% PEG 200 | ΔlnK 0.122 M (25 °C) | ΔlnK 1.022 M (25 °C) | różnica | S_sec(x) 25 °C | S_sec(x) 37 °C |
|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | +0.79 ± 0.57 | +0.23 ± 0.55 |
| 10 | −1.0 ± 1.1 | −2.4 ± 0.8 | −1.4 ± 1.3 | +0.16 | +0.23 |
| 20 | −2.4 ± 1.1 | −4.2 ± 0.8 | −1.9 ± 1.4 | −0.08 | +0.15 |
| 30 | −4.1 ± 1.1 | −6.1 ± 0.8 | −2.0 ± 1.4 | −0.16 | −0.08 |
| 40 | −5.1 ± 1.3 | −7.4 ± 0.8 | −2.4 ± 1.6 | −0.32 ± 0.46 | −0.23 ± 0.45 |

**Interpretacja A/B.**
1. Bez kosolutu S_sec dla 8-meru jest mały (0.2–0.8). Między 0.12 a 1 M ten dupleks leży więc blisko maksimum lub plateau F.
2. Żeby wyjaśnić ΔlnK ≈ −5 … −7 samym przesunięciem A, potrzeba φ = +6 … +30, czyli zmiany aktywności Na⁺ o czynnik e⁶–e³⁰. To fizycznie niemożliwe. **Destabilizacja przez PEG 200 jest w przeważającej części pionowa (B).**
3. Mieszana pochodna (zmiana S_sec z x): dS_sec/dwt% = −0.006 ± 0.010 (25 °C) i −0.014 ± 0.012 (37 °C). Kierunek jest ujemny, ale nieistotny. Dwie sole i błędy ±0.3–0.6 kcal/mol nie pozwalają wykryć A o |φ| ≲ 1–2.
4. W skali a_w efekty przy obu solach się pokrywają.
   - [nasze] dlnK/dln a_w wynosi 99 vs 125 (25 °C, 40%) oraz 116 vs 114 (37 °C).
   - Większy efekt przy 1 M (w skali wt%) idzie w parze z większym spadkiem a_w przy 1 M: Δln a_w −0.059 vs −0.051.
   - To nieaddytywność PEG–NaCl, czyli sygnatura μ_x > 0 (sól podnosi aktywność PEG, a PEG aktywność soli; relacja krzyżowa). W ramie A odpowiada to φ < 0.
5. **Autorzy jawnie zakładają addytywność (B):**
   - Ghosh 2020: „ΔG°37 NN = ΔG°37 NN,[cation] + ΔG°37 NN,[crowder]” oraz „destabilization with a change in water activity is independent of the ionic concentration of the solution”.
   - Testują to wyłącznie w skali Δa_w, nie w skali stężenia kosolutu.

### 2.2 Ghosh et al. 2023 NAR (SI Tab. S8): RNA 11-mer [PT]

**Warunki.** GAUUACGCCUG/komplement, 37 °C, ten sam bufor; 0.122 i 1.022 M Na⁺. Autorzy podają błędy ±0.1–0.2 kcal/mol (±0.7 dla 40% PEG 200 przy 1 M).

**Obliczenia.** Skrypt `na_ghosh2023_AB.py`, wynik w `na_ghosh2023_AB_output.txt`.
- S_sec: 1.83 ± 0.11 (bez kosolutu) → 1.68 ± 0.17 (10%) → 1.53 ± 0.17 (20%) → 1.07 ± 0.56 (40% PEG 200).
- Różnica efektów (1 M − 0.12 M) w ΔlnK: −0.32 ± 0.43 (10%), −0.65 ± 0.43 (20%), −1.6 ± 1.2 (40%).

**Ocena.**
- Kierunek jest monotoniczny i zgodny z Nakano 2004 przy 25 °C: destabilizacja rośnie z solą, czyli ∂S/∂x < 0.
- Czysty A z φ > 0 (wiązanie Na⁺) przewiduje odwrotnie, bo przy wklęsłej F lokalne S przy 1 M < S przy 0.12 M, więc efekt powinien maleć z solą.
- Zgodne z danymi są B + A(φ < 0) i B + C.

**Rząd wielkości dla A(φ < 0), z μ_x z notatki Q2 (NaCl, +42 cal/(mol·molal EO))** [nasze]:
- Dla 40 wt% PEG 200 (≈ 15 molal EO): Δln a± ≈ +0.55, czyli φ ≈ −0.55. To górna granica, bo ekstrapolacja liniowa.
- Przewidywana różnica efektów to φ·(S_lo − S_hi). Przy S_lo − S_hi ≈ 1–2 daje to −0.5 … −1.1, zgodnie z obserwowanym −1.6 ± 1.2.
- Dane nie odróżniają tego od C.

**EG.** 20 wt% przy 1 M: +1.5 kcal/mol; ekstrapolacja liniowa danych dla 0.12 M (10/15 wt%) daje +0.9. Kierunek ten sam, ale to ekstrapolacja.

**Dielektryk.** Autorzy podają ε_r = 39.4 (40 wt% PEG 200) i 62.0 (40 wt% EG) przy 76.8 bez kosolutu, w 37 °C (SI Tab. S10).
- Wartość 39.4 wygląda na zaniżoną. Dla PEG 400 przy ułamku objętościowym 0.4 Yurquina podaje 55–60 (notatka Q5).
- **Traktować ostrożnie.**

**Addytywność jawnie założona:** „suggesting that ΔG°37,wa,dup is independent of the ionic concentration of the solution, thus validating the additive nature of Equation 7”. Znów tylko w skali Δa_w.

### 2.3 Nakano, Kitagawa, Miyoshi, Sugimoto 2014 FEBS Open Bio [PT, PMC4141205; Fig. 3B zdigitalizowana]

**Warunki.**
- RNA 11-mer 5′-AUCCAGCGCGC/GUGCGCUGGAU, 2 µM, bufor fosforanowy, ΔG° i K w 37 °C z krzywych topnienia.
- NaCl 0.02, 0.05, 0.1, 0.2, 0.5, 0.7, 1 M.
- Kosolutanty 20 wt%; część także 5 i 10 wt% (SI Fig. S5, lokalny PDF).

**Cytaty i liczby autorów:**
- „The slope value 1.8 in the absence of cosolutes … the slope value decreased to 1.1 in the PEG8000-containing solution.”
- „In contrast, the slope value did not decrease in the FA- or urea-containing solutions.”
- Nachylenie i punkt przecięcia z osią x korelują z ε_r⁻¹ (Fig. 4E,F), a nie z a_w ani z lepkością.
- Autorzy proponują wprost: −ΔG° = (−ΔG° w warunkach standardowych) + (kara za oddziaływania specyficzne i obniżoną a_w) + **(człon zależny od ε) × log[sól]**. To dokładnie model **B + C**, nie A.
- O mechanizmie A piszą tylko marginalnie: „Some organic compounds may cause microphase separation or decrease the solubility of NaCl, which could increase the effective concentration of NaCl”.
- Rybozym (Tab. 1, nachylenie log k): bez kosolutu 4.9, PEG 200 3.0, PEG 8000 2.1, EG 4.5, glicerol 4.5, EtOH 2.2, mocznik 6.1, FA 5.6.

**[nasze] z digitalizacji (`na_nakano2014_analysis.py`, `na_nakano2014_analysis_output.txt`):**
- Bufor: nachylenie 1.82, zgodne z 1.8 autorów.
- Nachylenie w PEG 8000 na 0.02–0.2 M: 1.14 (bufor na tym zakresie 1.54); w PEG 200: 1.40 vs 1.82; w DMF: 1.51 vs 1.82.
- PEG 8000, Δlog K względem buforu: +0.77 (0.02 M), +0.55, +0.56, +0.32 (0.2 M).
  - To ΔG ≈ −4.6 kJ/mol przy 20 mM, czyli wielkość jak nasze −4 … −6 kJ/mol przy 35 mM.
  - Efekt maleje z solą; liniowa ekstrapolacja daje przecięcie z buforem przy ≈ 1.5 M (nie zmierzono, bo powyżej 0.2 M następuje rozdział faz).
- PEG 200: −0.51 (0.02 M) → −1.14 (1 M). Destabilizacja rośnie z solą, jak u Ghosha i Nakano 2004.

**Test A dla PEG 8000.**
- Stabilizacja wymaga φ < 0: φ_log = −0.50, −0.36, −0.36, −0.21 dekady, czyli **niestałe**.
- Na odcinku 0.02–0.2 M krzywa buforu jest prawie prosta, a nawet lekko wypukła: dopasowanie kwadratowe daje dS/dlog c = +0.68 na całym zakresie. Zmiana nachylenia o −0.40 nie może więc pochodzić z A; przy φ < 0 i wypukłej F A dałoby wręcz wzrost nachylenia.
- **Wniosek: PEG 8000 = B (stabilizacja) + C (spadek S_K).**

**Test A dla PEG 200.** Formalnie da się dopasować A z φ > 0 (φ ≈ +0.3 … +0.5 dekady) dzięki wypukłości F, ale:
- (i) znak φ jest sprzeczny z VPO (μ_x > 0, notatka Q2);
- (ii) ten sam spadek nachylenia daje DMF, aprotyczny rozpuszczalnik bez eterowych tlenów;
- (iii) korelacja z 1/ε obejmuje ok. 20 chemicznie różnych kosolutów.

**Wypukłość F w buforze jest niepewna:**
- przyrosty wynoszą 1.36 na dekadę (0.02–0.1 M) i 2.0 (0.1–1 M);
- może to być artefakt ekstrapolacji ΔG do 37 °C;
- **nie należy jej nadinterpretowywać.**

### 2.4 Nakano, Tanino, Hirayama, Sugimoto 2016 Biophys J [PT, PMC HTML]

- Układy: 9-mer RNA CAACGCAAG/CUUGCGUUG, dupleks GC, poli(A)/poli(U), tRNA.
- Nachylenie log K vs log[NaCl] bez kosolutu: 1.66 ± 0.05 dla 9-meru i 1.92 ± 0.12 dla dupleksu GC (Tab. 2).
- „The degree to which stability was dependent on salt concentration was correlated with reciprocals of the dielectric constants of mixed solutions, regardless of interactions between the cosolutes and RNA.”
- „Most mixed solutions decreased the slope of Tm versus log [NaCl] plots, whereas solutions containing FA increased the slope.”
- Nachyleń dla NaCl per kosolut nie podano w tekście (tylko Fig. 4A/B); nie digitalizowałem, bo rysunek jest za Cloudflare.
- Ten sam wniosek co w 2.3: **C**.

### 2.5 Markarian & Schlenoff 2010 JPCB [PT, manuskrypt autorski PMC2925268; Fig. 6 i 7A zdigitalizowane]

**Warunki.**
- 15-, 25- i 35-mer w 1 mM fosforanie, pH 7, 1 µM każdej nici.
- Izotermy: 35 °C (15-mer), 55 °C (25-mer), 67 °C (35-mer).
- NaCl dodawany do 2–225 mM; PEG „4000” (Mw ≈ 3000) 0–30 wt%.
- **Oś x to ln a_Na⁺ wyliczone z VPO** przy założeniu braku wiązania Na⁺–PEG. Autorzy już więc „zdjęli” mechanizm A w wersji wynikającej z VPO.

**Cytaty.**
- „From Figure 5 it is clear that the pairing of the oligonucleotides requires a lower [NaCl] with increasing PEG3 concentration.”
- „The effect of molecular crowding was observed through the decrease of ion binding onto the strands as the PEG3 content … was increased from 0% to 30%.”

**[nasze] z Fig. 6 (35-mer, `na_markarian2010_analysis.py`):**

| PEG wt% | n | nachylenie S | ΔlnK vs 0% przy ln a = −3.7 | przy −3.0 | przy −2.2 | ln a* przecięcia |
|---|---|---|---|---|---|---|
| 0 | 9 | 2.85 ± 0.12 | — | — | — | — |
| 1 | 10 | 2.95 ± 0.18 | +0.12 | +0.19 | +0.26 | (−5.0) |
| 5 | 9 | 2.75 ± 0.23 | +0.03 | −0.04 | −0.13 | −3.39 |
| 10 | 9 | 2.33 ± 0.21 | +0.62 | +0.25 | −0.16 | −2.51 |
| 20 | 9 | 1.59 ± 0.13 | +1.13 | +0.25 | −0.76 | −2.80 |
| 30 | 9 | 1.19 ± 0.04 | +1.29 | +0.13 | −1.20 | −2.92 |

- Z Fig. 7A (Δψ na bp, przy czym autorzy definiują Δψ = 2S/N_bp; [nasze] sprawdzenie: 0.17 × 35/2 = 2.96, a zmierzone na Fig. 6 nachylenie to 2.85):
  - 35-mer: S 2.96 → 1.35 (30 wt%);
  - 25-mer: 4.25 → 2.39;
  - 15-mer: 2.62 → 1.79;
  - dS/dwt% = −0.060, −0.065, −0.025.
- Krzywizna F w buforze (35-mer) jest mała: współczynnik kwadratowy −0.34, czyli dS/dln a ≈ −0.7.
- Żeby A dało ΔS = −1.6, potrzeba φ ≈ −2.4 (aktywność ×11). To sprzeczne z VPO, które autorzy już uwzględnili.

**Wniosek.** Najczystszy w literaturze dowód, że przy niskiej soli i długim PEG działają **B (stabilizacja) + C (spadek S)**, a krzywe się przecinają. Ani A, ani B samodzielnie tego nie opisują.

Zastrzeżenia:
- aktywności liczone z VPO w 20 °C, pomiary w 35–67 °C;
- krótka statystyka;
- K w 67 °C dla 35-meru.

### 2.6 Lambert, Leipply, Draper 2010 JMB i Lambert & Draper 2012 Biochemistry [PT]

**Jedyne znalezione prace, które jawnie rozdzieliły „efekt przez aktywność soli” od efektu bezpośredniego.**

**2010 (TMAO, siatka KCl × TMAO):**
- „TMAO interacts weakly, if at all, with KCl, ruling out the possibility that TMAO stabilizes RNA indirectly by increasing salt activity.”
- Dla spinki: „the salt dependence of the hairpin stability was virtually unaffected by TMAO”. To czyste **B**.

**2012 (mocznik):**
- Po korekcie na aktywność mocznika i KCl zostaje człon krzyżowy: „the salt-dependence is steeper (more negative) in the presence of urea by an average of ~20%. The m-value itself is therefore salt dependent”.
- Mocznik ma ε wyższą niż woda (ε rośnie), a |S| rośnie. To ten sam znak korelacji z 1/ε co u Nakano (2014: mocznik i FA nie obniżają nachylenia).

**Sprzeczność:** Guinn 2013 dla dodekamerów DNA pisze, że mocznik „has no effect on the salt dependence of DNA melting”. Dane są w SI, którego nie zweryfikowałem.

### 2.7 Lambert & Draper 2007 JMB [PT]

- Tylko Mg²⁺ i struktury trzeciorzędowe RNA. Wniosek: „the stabilizing effects of Mg²⁺ and most of the osmolytes studied here are not additive”.
- TMAO jest wyjątkiem i działa addytywnie.
- Dla soli jednowartościowej autorzy podsumowują literaturę: wpływ EG, betainy i mocznika na zależność od soli przy denaturacji DNA jest „at the edge of statistical significance” (cytują Nordstrom 2006 i Spink & Chaires 1999).

### 2.8 Pozostałe (tylko abstrakty)

- **Karimata 2007.** Projekt 2D (PEG 200, PEG 8000 × NaCl) ze zmianą znaku efektu PEG 8000 i spadkiem −ΔnNa⁺. Jakościowo to samo co Markarian i Nakano 2014. **Najważniejsza brakująca praca:** pełny tekst przez bibliotekę IChF.
- **Spink & Chaires 1999.** Zależność od Na⁺ w EG: „modest decrease in the extent of counterion binding”.
- **Nordstrom 2006.** Wzrost soli osłabia akumulację mocznika i glicyn; EG nie jest wymieniony.
- **Hutton 1977.** Formamid: −0.60 °C/% przy NaCl 0.035–0.88 M, czyli B; „slightly greater” przy wyższej soli.
- **Rees 1993.** Betaina „does not greatly change the behavior of DNA as a polyelectrolyte”.
- **Schwinefus 2007.** Betaina przy 40 vs 135 mM NaCl zmienia znak efektu na strukturze trzeciorzędowej.
- **Goobes 2003.** Efekt crowdingu „marginal” w warunkach sprzyjających asocjacji, rośnie w niesprzyjających.
- **Martini 2025.** Efekt crowdera przesuwa się od spinki z rosnącym NaCl.
- **Votavová 1986, Gao 2017, Nakano 2012 JPCB.** Projekty sól × kosolut potwierdzone tytułem lub abstraktem, bez liczb. Nakano 2012: efektywność dimeryzacji spinek indukowanej jonami koreluje z ε.

### 2.9 Teoria: S_K w mieszanych rozpuszczalnikach

- **Record 1975; Record, Anderson, Lohman 1978; Manning 1972/1978; Anderson & Record 1995** (DOI zweryfikowane; pełnych tekstów nie czytałem w tym zadaniu).
- Standardowo: S_K = −ΔΓ±·ν, a w ramie kondensacji ∝ Δ(1/ξ), gdzie ξ = l_B/b ∝ 1/ε.
- [nasze, wyprowadzenie, nie cytat] Gdy obie formy (ss i ds) są powyżej progu kondensacji, S_K ∝ (1/ξ_ss − 1/ξ_ds) ∝ ε. Spadek ε o x% obniża więc S_K mniej więcej o x%, czyli multiplikatywnie (C).
  - Dla Ghosh 2023 (ε 76.8 → 39.4, jeśli wierzyć tej liczbie): przewidywane S × 0.51, obserwowane 1.07/1.83 = 0.58 ± 0.31.
  - Dla Nakano 2014 PEG 8000: obserwowane 1.14/1.54 = 0.74.
  - Zgodność rzędu wielkości, ale model jest graniczny (krótkie oligo, ss słabo naładowane).
- **Opublikowanego pomiaru dTm/dlog[Na⁺] dla oligonukleotydów w mieszaninach alkohol–woda nie znalazłem.** Jedyne eksperymentalne zestawienia „S_K vs 1/ε” to prace Nakano/Sugimoto (2012, 2014, 2016).

## 3. Pytanie 2: czy któraś praca przeszła przez maksimum K(c) / Tm(c) w obecności kosolutu?

**Nie znalazłem żadnej pracy, w której krzywa K(c) lub Tm(c) w obecności kosolutu, zmierzona w ≥ 3 punktach, przechodziłaby przez maksimum.**

| praca | co pokazuje | dlaczego nie odpowiada na pytanie |
|---|---|---|
| Nakano 2004 / Ghosh 2020 (8-mer DNA) | Jedyny przypadek „ocierający się” o maksimum. Sieczna w buforze 0.2–0.8 (blisko plateau). W 40 wt% PEG 200 sieczna ujemna (−0.2 … −0.3 ± 0.45), co sugeruje przesunięcie maksimum poniżej 1 M albo spłaszczenie. | Dwa punkty soli, różnica nieistotna statystycznie. |
| Nakano 2014 / 2016 | Bufor do 1 M nadal rośnie (bez maksimum). | PEG 8000 ma dane tylko do 0.2 M (rozdział faz). |
| Markarian 2010 | — | a_Na ≤ 0.13. |
| Lambert 2010 | — | KCl ≤ 0.8 m. |
| Hutton 1977 | Abstrakt wspomina „higher salt concentrations” (genomowe DNA, maksimum Tm leży wyżej). | Liczb brak. |
| Karimata 2007 | — | Pełny tekst nieznany; abstrakt mówi tylko „high NaCl”. |

**Odpowiedź:** test A vs B przez przejście przez maksimum (lub zakręt) F nie został w literaturze wykonany. Jedynym dostępnym w literaturze testem są (i) dwupunktowe porównania 0.1 vs 1 M (Sugimoto i in.) oraz (ii) siatki na odcinku niskiej soli (Markarian, Nakano 2014, Lambert 2010/2012).

## 4. Dane zdigitalizowane

### 4.1 Markarian & Schlenoff 2010, Fig. 6

**Źródło:** ln K_obs vs ln a_Na⁺, 35-mer, 67 °C, PEG 0/1/5/10/20/30 wt%. JPEG z PMC (601×516 px, 150 dpi) zapisany jako `na_markarian2010_fig6_source.jpg`.

**Metoda.**
1. Ramka osi znaleziona z długich ciemnych serii pikseli: x 94–485 px odpowiada ln a −4.0 … −2.0, y 37–411 px odpowiada ln K 18 … 10. Etykiety podziałek pokrywają się z ramką.
2. Markery wykryto dopasowaniem wzorców (6 kształtów wyciętych z izolowanych markerów) w `na_digitize_markarian2010_fig6.py`; surowy wynik w `na_markarian2010_fig6_raw.csv`.
3. Punkty sprawdziłem i poprawiłem ręcznie na powiększeniach 6× i 8× z siatką pikseli. Usunąłem fałszywe trafienia i dodałem nakładające się markery z flagą `overlap`.

**Wynik:** `na_markarian2010_fig6.csv`, 55 punktów.

**Niepewność.**
- ±1.5 px, czyli ±0.008 w ln a i ±0.03 w ln K.
- Punkty `overlap` (ok. 12) mają niepewność do ±0.1 w ln K, a jeden (30%, ukryty) jest niepewny co do istnienia.

**Kontrola:** nachylenie przy 0% = 2.85 ± 0.12, wobec 2.96 z Fig. 7A autorów.

### 4.2 Markarian & Schlenoff 2010, Fig. 7A

**Źródło:** Δψ na bp vs PEG wt% dla 15-, 25- i 35-meru (`na_markarian2010_fig7_source.jpg`).

**Metoda.**
1. Ramka x 95–365 px odpowiada 0–35 wt%, y 28–284 px odpowiada 0.40–0.00.
2. Środki markerów wyznaczone po otwarciu morfologicznym 4×4 (`na_digitize_markarian2010.py`).
3. Serie przypisane ręcznie według kształtu i kolejności przy nominalnych 0/1/5/10/20/30 wt%.

**Wynik:** `na_markarian2010_fig7A.csv`, z kolumną S_K = Δψ·N/2.

**Niepewność:** ±0.005 (1 px = 0.0016). Dla 15-meru przy 1% marker jest zasłonięty i go pominięto.

### 4.3 Nakano et al. 2014, Fig. 3B

**Źródło:** log K vs log[NaCl], 11-mer RNA, 37 °C. JPEG z PMC (685×183 px, 113 dpi; wyższej rozdzielczości nie udało się pobrać, bo Wiley/Elsevier zwraca 403). Plik `na_nakano2014_fig3_source.jpg`.

**Metoda.**
1. Kalibracja z podziałek: x −1.5/−1.0/−0.5/0.0 przy 312/350.5/389/427.5 px; y 14 … 7 przy 19 … 150 px.
2. Segmentacja markerów po kolorze (`na_digitize_nakano2014_fig3B.py`), wynik w `na_nakano2014_fig3B_raw.csv`.
3. Wartości x przyciągnięte do nominalnych stężeń (0.02–1 M, odczyt w granicach 0.025 log).

**Wynik:** `na_nakano2014_fig3B.csv` dla serii: bufor, PEG 200, PEG 8000, DMF, FA (tylko ≥ 0.5 M).
- EtOH pominięto: markery nakładają się z FA i PEG 200, a kolory są niejednoznaczne.

**Niepewność:** ±0.04 log K.

**Kontrola:** nachylenie buforu 1.82, wobec 1.8 podanego przez autorów.

## 5. Wnioski dla naszej pracy

1. **Czy literatura zrobiła jawny rozdział A/B?**
   - W sensie „poziomo vs pionowo w osi ln c” nikt tego nie zrobił.
   - Najbliżej są Lambert et al. 2010/2012: VPO plus siatka sól × kosolut. Wynik: TMAO na spince RNA to B (A wykluczone pomiarem μ₂₃ ≈ 0); mocznik to B + człon krzyżowy (C, S rośnie o ~20%).
   - Markarian 2010 „zdjął” A przez aktywność z VPO, a nachylenie i tak spada (C).
   - Sugimoto i in. (Ghosh 2020/2023) jawnie *zakładają* B i testują go tylko w skali Δa_w.
   - Nakano 2014/2016 jawnie proponują B + C (człon ε × log[sól]).
2. **Długi PEG przy niskiej soli** (Markarian 2010, Nakano 2014, Karimata 2007 [ABS], Baltierra-Jasso 2015, Martini 2025 [ABS], Goobes 2003 [ABS]):
   - stabilizacja jest największa przy najniższej soli i maleje z solą;
   - S_K spada (o 26% dla 20 wt% PEG 8000, o 54% dla 30 wt% PEG ~3000);
   - przecięcie (zmiana znaku) wypada przy ~50–80 mM aktywności Na⁺ (Markarian, 67 °C) albo ekstrapolowanie przy ~1.5 M (Nakano 2014).
   - **Czysty A tego nie odtworzy** (patrz test 2 w §0), czysty B też nie.
3. **Krótkie PEG i EG przy 0.1–1 M:**
   - destabilizacja jest dominująco pionowa (B);
   - jej wielkość rośnie z solą o ~1.5× (Nakano 2004 przy 25 °C, Ghosh 2023, Nakano 2014 PEG 200);
   - to kierunek **przeciwny** do A z wiązaniem Na⁺ (φ > 0), a zgodny z A(φ < 0, wykluczenie soli; μ_x > 0 z VPO, notatka Q2) albo z C.
4. **Implikacja dla hipotezy kompleksowania Na⁺ (φ > 0).** Żadna praca z siatką sól × PEG nie pokazuje sygnatury φ > 0:
   - przy φ > 0 efekt PEG malałby z solą przy destabilizacji, a przy stabilizacji długimi PEG wymagałby φ < 0;
   - wszędzie, gdzie jest to mierzalne, S_K *maleje* z PEG;
   - przy F wklęsłej oznacza to φ < 0 (gdyby to był A), a przy prostym odcinku F oznacza C.
5. **Przewidywanie testowalne dla nas.** Przy 35 mM Na⁺ i 20–40 wt% długiego PEG literatura (C) sugeruje spadek S_K z 3.5 do ~2–2.6 (o −25 … −50%).
   - Nasz model PB z samą zmianą ε (notatka Q5) dawał tylko 3.0–3.3.
   - Model kompleksowania Na⁺ (A, φ > 0) przy F z maksimum przewiduje natomiast **wzrost** S_K przy 35 mM, bo przesunięcie w lewo oddala od maksimum. Wynika to z ∂S/∂x = −φ'·dS/dln c > 0 dla φ' > 0.
   - **Pomiar d lnK/d ln[Na⁺] w 20–40 wt% PEG ≥ 3k i w EG rozstrzyga więc A(φ > 0) od C znakiem zmiany nachylenia, a nie tylko jej wielkością.**

## Bibliografia

Format: autorzy, tytuł, czasopismo, tom, strony, rok, DOI, [weryfikacja; dostęp]. „Crossref OK” oznacza sprawdzenie przez api.crossref.org 2026-10-09.

1. Nakano S., Karimata H., Ohmichi T., Kawakami J., Sugimoto N. „The Effect of Molecular Crowding with Nucleotide Length and Cosolute Structure on DNA Duplex Stability”. J. Am. Chem. Soc. 126, 14330–14331 (2004). DOI 10.1021/ja0463029. [Crossref OK; PT + SI, lokalnie `literature/pdf/`]
2. Ghosh S., Takahashi S., Ohyama T., Endoh T., Tateishi-Karimata H., Sugimoto N. „Nearest-neighbor parameters for predicting DNA duplex stability in diverse molecular crowding conditions”. Proc. Natl. Acad. Sci. USA 117, 14194–14201 (2020). DOI 10.1073/pnas.1920886117. [Crossref OK; PT + SI, lokalnie]
3. Ghosh S., Takahashi S., Banerjee D., Ohyama T., Endoh T., Tateishi-Karimata H., Sugimoto N. „Nearest-neighbor parameters for the prediction of RNA duplex stability in diverse in vitro and cellular-like crowding conditions”. Nucleic Acids Res. 51, 4101–4111 (2023). DOI 10.1093/nar/gkad020. [Crossref OK; PT + SI (Europe PMC, PMC10201447)]
4. Nakano S., Kitagawa Y., Miyoshi D., Sugimoto N. „Hammerhead ribozyme activity and oligonucleotide duplex stability in mixed solutions of water and organic compounds”. FEBS Open Bio 4, 643–650 (2014). DOI 10.1016/j.fob.2014.06.009. [Crossref OK; PT (PMC4141205), SI lokalnie `literature/review_2026-10/salt_shift/Nakano2014_FEBSOpenBio_SI.pdf`]
5. Nakano S., Tanino Y., Hirayama H., Sugimoto N. „Thermal Stability of RNA Structures with Bulky Cations in Mixed Aqueous Solutions”. Biophys. J. 111, 1350–1360 (2016). DOI 10.1016/j.bpj.2016.08.031. [Crossref OK; PT (PMC5052467, HTML)]
6. Markarian M.Z., Schlenoff J.B. „Effect of Molecular Crowding and Ionic Strength on the Isothermal Hybridization of Oligonucleotides”. J. Phys. Chem. B 114, 10620–10627 (2010). DOI 10.1021/jp103213w. [Crossref OK; PT (manuskrypt autorski PMC2925268); SI nieczytane]
7. Karimata H., Nakano S., Sugimoto N. „Effects of Polyethylene Glycol on DNA Duplex Stability at Different NaCl Concentrations”. Bull. Chem. Soc. Jpn. 80, 1987–1994 (2007). DOI 10.1246/bcsj.80.1987. [Crossref OK; ABS (OpenAlex), pełny tekst zablokowany (OUP 403)]
8. Karimata H., Nakano S., Ohmichi T., Kawakami J., Sugimoto N. „Stabilization of a DNA duplex under molecular crowding conditions of PEG”. Nucleic Acids Symp. Ser. 48, 107–108 (2004). DOI 10.1093/nass/48.1.107. [Crossref OK; ABS]
9. Nakano S., Wu L., Oka H., Karimata H.T., Kirihata T., Sato Y., Fujii S., Sakai H., Kuwahara M., Sawai H., Sugimoto N. „Conformation and the sodium ion condensation on DNA and RNA structures in the presence of a neutral cosolute as a mimic of the intracellular media”. Mol. BioSyst. 4, 579–588 (2008). DOI 10.1039/B718806D. [Crossref OK; ABS]
10. Spink C.H., Chaires J.B. „Effects of Hydration, Ion Release, and Excluded Volume on the Melting of Triplex and Duplex DNA”. Biochemistry 38, 496–508 (1999; Crossref „issued” 1998). DOI 10.1021/bi9820154. [Crossref OK; ABS]
11. Spink C.H., Garbett N., Chaires J.B. „Enthalpies of DNA melting in the presence of osmolytes”. Biophys. Chem. 126, 176–185 (2007). DOI 10.1016/j.bpc.2006.07.013. [Crossref OK; ABS; jedno stężenie soli według abstraktu]
12. Nordstrom L.J., Clark C.A., Andersen B., Champlin S.M., Schwinefus J.J. „Effect of Ethylene Glycol, Urea, and N-Methylated Glycines on DNA Thermal Stability: The Role of DNA Base Pair Composition and Hydration”. Biochemistry 45, 9604–9614 (2006). DOI 10.1021/bi052469i. [Crossref OK; ABS]
13. Hickey D.R., Turner D.H. „Solvent effects on the stability of A7U7p”. Biochemistry 24, 2086–2094 (1985). DOI 10.1021/bi00329a042. [Crossref OK; ABS]
14. Gu X., Nguyen M.-T., Overacre A., Seaton S., Schroeder S.J. „Effects of Salt, Polyethylene Glycol, and Locked Nucleic Acids on the Thermodynamic Stabilities of Consecutive Terminal Adenosine Mismatches in RNA Duplexes”. J. Phys. Chem. B 117, 3531–3540 (2013). DOI 10.1021/jp312154d. [Crossref OK; PT, lokalnie; PEG tylko przy 1 M NaCl]
15. Adams M.S., Znosko B.M. „Thermodynamic characterization and nearest neighbor parameters for RNA duplexes under molecular crowding conditions”. Nucleic Acids Res. 47, 3658–3666 (2019). DOI 10.1093/nar/gkz019. [Crossref OK; PT (PMC6468164); tylko 1 M NaCl]
16. Rozners E. „Hydration of short DNA, RNA and 2′-OMe oligonucleotides determined by osmotic stressing”. Nucleic Acids Res. 32, 248–254 (2004). DOI 10.1093/nar/gkh175. [Crossref OK (Crossref podaje jako autora tylko „Rozners E.”); ABS tutaj (PT w notatce Q5); jedno stężenie soli]
17. Lambert D., Draper D.E. „Effects of Osmolytes on RNA Secondary and Tertiary Structure Stabilities and RNA-Mg2+ Interactions”. J. Mol. Biol. 370, 993–1005 (2007). DOI 10.1016/j.jmb.2007.03.080. [Crossref OK; PT (PMC1995082, HTML)]
18. Lambert D., Leipply D., Draper D.E. „The Osmolyte TMAO Stabilizes Native RNA Tertiary Structures in the Absence of Mg2+: Evidence for a Large Barrier to Folding from Phosphate Dehydration”. J. Mol. Biol. 404, 138–157 (2010). DOI 10.1016/j.jmb.2010.09.043. [Crossref OK; PT (PMC3001104)]
19. Lambert D., Draper D.E. „Denaturation of RNA Secondary and Tertiary Structure by Urea: Simple Unfolded State Models and Free Energy Parameters Account for Measured m-Values”. Biochemistry 51, 9014–9026 (2012). DOI 10.1021/bi301103j. [Crossref OK; PT (PMC3505219)]
20. Guinn E.J., Schwinefus J.J., Cha H.K., McDevitt J.L., Merker W.E., Ritzer R., Muth G.W., Engelsgjerd S.W., Mangold K.E., Thompson P.J., Kerins M.J., Record M.T. Jr. „Quantifying Functional Group Interactions That Determine Urea Effects on Nucleic Acid Helix Formation”. J. Am. Chem. Soc. 135, 5828–5838 (2013). DOI 10.1021/ja400965n. [Crossref OK; PT tekst główny (PMC3655208); dane SI NIEZWERYFIKOWANE]
21. Goobes R., Kahana N., Cohen O., Minsky A. „Metabolic Buffering Exerted by Macromolecular Crowding on DNA–DNA Interactions: Origin and Physiological Significance”. Biochemistry 42, 2431–2440 (2003). DOI 10.1021/bi026775x. [Crossref OK; ABS]
22. Martini D., Anderson J., Garcia G., Cannon B. „Contrasting Effects of Molecular Crowding on Trinucleotide-Repeat DNA Hairpins”. J. Phys. Chem. B 129, 11490–11499 (2025). DOI 10.1021/acs.jpcb.5c06024. [Crossref OK; ABS]
23. Baltierra-Jasso L.E., Morten M.J., Laflör L., Quinn S.D., Magennis S.W. „Crowding-Induced Hybridization of Single DNA Hairpins”. J. Am. Chem. Soc. 137, 16020–16023 (2015). DOI 10.1021/jacs.5b11829. [Crossref OK; PT (eprint Glasgow, CC-BY); SI (Fig. S9, sól) niedostępne]
24. Kilburn D., Roh J.H., Guo L., Briber R.M., Woodson S.A. „Molecular Crowding Stabilizes Folded RNA Structure by the Excluded Volume Effect”. J. Am. Chem. Soc. 132, 8690–8696 (2010). DOI 10.1021/ja101500g. [Crossref OK; PT (notatka Q4_Q7)]
25. Strulson C.A., Boyer J.A., Whitman E.E., Bevilacqua P.C. „Molecular crowders and cosolutes promote folding cooperativity of RNA under physiological ionic conditions”. RNA 20, 331–347 (2014). DOI 10.1261/rna.042747.113. [Crossref OK; PT według subagenta, liczby w tabeli-obrazie nieodczytane]
26. Leamy K.A., Yennawar N.H., Bevilacqua P.C. „Cooperative RNA Folding under Cellular Conditions Arises From Both Tertiary Structure Stabilization and Secondary Structure Destabilization”. Biochemistry 56, 3422–3433 (2017). DOI 10.1021/acs.biochem.7b00325. [Crossref OK; PT według subagenta, tabele Tm w SI nieczytane]
27. Schwinefus J.J., Kuprian M.J., Lamppa J.W., Merker W.E., Dorn K.N., Muth G.W. „Human Telomerase RNA Pseudoknot and Hairpin Thermal Stability with Glycine Betaine and Urea: Preferential Interactions with RNA Secondary and Tertiary Structures”. Biochemistry 46, 9068–9079 (2007). DOI 10.1021/bi602637v. [Crossref OK; ABS]
28. Hutton J.R. „Renaturation Kinetics and thermal stability of DNA in aqueous solutions of formamide and urea”. Nucleic Acids Res. 4, 3537–3555 (1977). DOI 10.1093/nar/4.10.3537. [Crossref OK; ABS]
29. Votavová H., Kucerová D., Felsberg J., Sponar J. „Changes in Conformation, Stability and Condensation of DNA by Univalent and Divalent Cations in Methanol-Water Mixtures”. J. Biomol. Struct. Dyn. 4, 477–489 (1986). DOI 10.1080/07391102.1986.10506363. [Crossref OK; ABS]
30. Rees W.A., Yager T.D., Korte J., von Hippel P.H. „Betaine can eliminate the base pair composition dependence of DNA melting”. Biochemistry 32, 137–144 (1993). DOI 10.1021/bi00052a019. [Crossref OK; ABS]
31. Gao M., Arns L., Winter R. „Modulation of the Thermodynamic Signatures of an RNA Thermometer by Osmolytes and Salts”. Angew. Chem. Int. Ed. 56, 2302–2306 (2017). DOI 10.1002/anie.201611843. [Crossref OK; ABS; siatka soli NIEZWERYFIKOWANA]
32. Nakano S., Hirayama H., Miyoshi D., Sugimoto N. „Dimerization of Nucleic Acid Hairpins in the Conditions Caused by Neutral Cosolutes”. J. Phys. Chem. B 116, 7406–7415 (2012). DOI 10.1021/jp302170f. [Crossref OK; ABS]
33. Nakano S., Kitagawa Y., Karimata H., Sugimoto N. „Molecular crowding effect on metal ion binding properties of the hammerhead ribozyme”. Nucleic Acids Symp. Ser. 52, 519–520 (2008). DOI 10.1093/nass/nrn263. [Crossref OK; ABS]
34. Knowles D.B., LaCroix A.S., Deines N.F., Shkel I., Record M.T. Jr. „Separation of preferential interaction and excluded volume effects on DNA duplex and hairpin stability”. Proc. Natl. Acad. Sci. USA 108, 12699–12704 (2011). DOI 10.1073/pnas.1103382108. [Crossref OK; PT lokalnie; jedno stężenie soli, VPO PEG–KCl w notatce Q2]
35. Knowles D.B., Shkel I.A., Phan N.M., Sternke M., Lingeman E., Cheng X., Cheng L., O'Connor K., Record M.T. Jr. „Chemical Interactions of Polyethylene Glycols (PEGs) and Glycerol with Protein Functional Groups: Applications to Effects of PEG and Glycerol on Protein Processes”. Biochemistry 54, 3528–3542 (2015). DOI 10.1021/acs.biochem.5b00246. [Crossref OK; PT w notatce Q2]
36. Manning G.S. „On the application of polyelectrolyte ‘limiting laws’ to the helix-coil transition of DNA. I. Excess univalent cations”. Biopolymers 11, 937–949 (1972). DOI 10.1002/bip.1972.360110502. [Crossref OK; nie czytano]
37. Manning G.S. „The molecular theory of polyelectrolyte solutions with applications to the electrostatic properties of polynucleotides”. Q. Rev. Biophys. 11, 179–246 (1978). DOI 10.1017/s0033583500002031. [Crossref OK; nie czytano]
38. Record M.T. Jr. „Effects of Na+ and Mg++ ions on the helix–coil transition of DNA”. Biopolymers 14, 2137–2158 (1975). DOI 10.1002/bip.1975.360141012. [Crossref OK; nie czytano]
39. Record M.T. Jr., Anderson C.F., Lohman T.M. „Thermodynamic analysis of ion effects on the binding and conformational equilibria of proteins and nucleic acids: the roles of ion association or release, screening, and ion effects on water activity”. Q. Rev. Biophys. 11, 103–178 (1978). DOI 10.1017/s003358350000202x. [Crossref OK; nie czytano]
40. Anderson C.F., Record M.T. Jr. „Salt-Nucleic Acid Interactions”. Annu. Rev. Phys. Chem. 46, 657–700 (1995). DOI 10.1146/annurev.pc.46.100195.003301. [Crossref OK; nie czytano]
41. Frank-Kamenetskii M.D. „Simplification of the empirical relationship between melting temperature of DNA, its GC content and concentration of sodium ions in solution”. Biopolymers 10, 2623–2624 (1971). DOI 10.1002/bip.360101223. [Crossref OK; nie czytano; dotyczy wody]
