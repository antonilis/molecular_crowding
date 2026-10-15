# Q2: termodynamika PEG–sól w wodzie a znak i wielkość μ_x

Data: 2026-10-09. Obliczenia: `analysis/literature_check/q2_mux_estimate.py`, wynik w
`analysis/literature_check/q2_mux_estimate_output.txt` (uruchamiane przez `.venv/bin/python`).
Pobrane pełne teksty: `literature/review_2026-10/q2/`. Każdy DOI zweryfikowano w Crossref API
(IJCCE 2020: przez doi.org i stronę wydawcy, bo nie ma go w Crossref).

**Konwencja.** μ_x ≡ μ₂₃ = (∂μ_sól/∂m_grupa)_{m_sól}, w cal/kcal na mol soli, na molal grup PEG
(EO = wewnętrzna grupa –CH₂OCH₂–; 2E = para końcowych –CH₂OH). Z definicji
(∂ ln a±/∂m_EO)_{m_sól} = μ_x/(νRT), więc wkład sprzężenia z solą do K dupleksu wynosi
(∂ ln K/∂m_EO) = S_K·μ_x/(νRT). RT = 592.5 cal/mol (25 °C). μ_x > 0 oznacza, że sól jest
wykluczana (PEG jest preferencyjnie hydratowany), a_± rośnie i dupleks się stabilizuje. μ_x < 0 oznacza
akumulację lub wiązanie soli, a_± maleje i dupleks się destabilizuje.

**Wniosek w jednym zdaniu.** Wszystkie bezpośrednie pomiary termodynamiczne, jakie znalazłem
(VPO, dyfuzja wieloskładnikowa z walidacją izopiestyczną, potencjometria ISE), dają **μ_x > 0** dla NaCl i
KCl z PEG (od tetraEG do PEG 20k). Dla HPO₄²⁻ μ_x jest kilka razy większe. Wyraz kationowy (Na⁺, K⁺)
wychodzi netto ≈ 0 lub lekko dodatni. Literatura **nie wspiera** więc obrazu, w którym PEG obniża aktywność
Na⁺ w roztworze, i przewiduje efekt solny o przeciwnym znaku: PEG podnosi a± i stabilizuje dupleks.

---

## (a) Pomiary bezpośrednie

### A1. Knowles et al. 2011, PNAS, SI Tabela S3 (pełny tekst, lokalny PDF)
VPO, 25 °C, PEG–KCl–woda (Wescor Vapro 5520). μ₃,₄ wyznaczono z nachylenia Δ₃,₄Osm względem m₃m₄ (równ. S1–S2):
- PEG 1000: **+0.0378 ± 0.001** kcal/(mol·molal monomeru PEG); PEG 2000: **+0.0324 ± 0.001**.
- Sorbitol +0.0082 ± 0.013, mannitol −0.0015 ± 0.014, ksylitol +0.0019 ± 0.007 kcal/molal (na cząsteczkę). Grupy końcowe uznano za ≈ 0.
- α_I,KCl = 4.41×10⁻⁴ ± 0.089×10⁻⁴ kcal/(molal PEG monomer·Å²) (równ. S4).
- Przeliczenie: ∂ln a±(KCl)/∂m_EO = μ/(2RT) = **0.032 (PEG 1000), 0.027 (PEG 2000)** na molal EO.
- Autorzy wprowadzili poprawkę „na stałą aktywność KCl” (równ. S5–S7). Piszą, że skorygowane m-values
  „differ quantitatively … but the qualitative features … are not affected”.

### A2. Knowles et al. 2015, Biochemistry 54, 3528 (pełny tekst główny przez PMC HTML/WebFetch; SI nieczytane)
VPO, < 0.6 m każdego składnika. Dekompozycja μ₂₃ = Σα_i·ASA_i + Σβ_k·ν_k, gdzie β_k jest dopasowywane
niezależnie („β-values for all inorganic ion are individually floated”). Tabela 2B, wartości w cal·mol⁻¹·molal⁻¹ na jon:

| jon | β_tetraEG | β_glicerol | β_2E (końce PEG) | **β_I (wewn. EO)** |
|---|---|---|---|---|
| HPO₄²⁻ | 780 ± 16 | 93.9 ± 6.3 | 79.3 ± 5.3 | **234 ± 5.5** |
| Cl⁻ | 132 ± 6.8 | 22.8 ± 2.9 | 19.3 ± 2.5 | **37.6 ± 2.4** |
| K⁺ | 6.19 ± 10 | −4.11 ± 3.1 | −3.47 ± 2.6 | **3.22 ± 3.4** |
| Na⁺ | 18.5 ± 5.2 | 5.34 ± 3.2 | 4.51 ± 2.7 | **4.66 ± 2** |

(β_tetraEG = β_2E + 3β_I. β_2E to wartości glicerolu przeskalowane przez 214/253.)
Abstrakt: tetraEG oddziałuje „unfavorably with … salt ions”, a oddziaływania z anionami są „strongly unfavorable”.
**Przeliczenie (skrypt, sekcja 1)**, μ_x na molal EO i na mol soli:
- NaCl: 4.66 + 37.6 = **+42.3 ± 3.1 cal**, czyli ∂ln a±/∂m_EO = **0.036**;
- KCl: **+40.8 ± 4.2 cal**. Zgadza się z niezależnym VPO z 2011 (+32 do +38) w granicach ~20%;
- Na₂HPO₄: 2·4.66 + 234 = **+243 ± 7 cal**, czyli ∂ln a±/∂m_EO = **0.137**;
- sam Na⁺ (podział na jony jest ekstratermodynamiczny i zależy od modelu): **+4.7 ± 2 cal**, czyli
  ∂ln a_Na/∂m_EO = 0.008 ± 0.003. W tych danych **nie ma netto akumulacji Na⁺** przy grupach EO.
- H₂PO₄⁻: brak danych.

### A3. Grupa Annunziaty: dyfuzja wieloskładnikowa (Rayleigh), PEG 20 kg/mol, 25 °C
- Tan, Albright, Annunziata 2008 JPCB (tylko abstrakt). Układ PEG 20k–KCl. Preferencyjna **hydratacja** PEG
  w KCl jest o rząd wielkości większa niż dla lizozymu i maleje ze wzrostem stężenia soli. Wynik
  potwierdzono metodą izopiestyczną.
- McAfee, Zhang, Annunziata 2014 Langmuir (tylko abstrakt). Badane sole: NaCl, KCl, NH₄Cl, CaCl₂ i Na₂SO₄, przy 0.5% w/w PEG.
  Zmiana kationu w chlorkach ma mały wpływ na N_w. Zamiana NaCl → Na₂SO₄ zwiększa N_w ok. 3-krotnie (Hofmeister).
- Annunziata 2024 Molecules (pełny tekst, OA) cytuje wynik z 2014: **„the value of ν̄ per ethoxy group is 2.4 water
  molecules … in the presence of NaCl”**, h̄ ≈ 0.14.
- Cruz Barrios et al. 2022 IJMS (pełny tekst, OA): tyloxapol (≈50 grup EO) ma ≈7 H₂O/EO w Na₂SO₄ i ≈8–9 w MgSO₄
  (ν_W = 450 ± 30 na cząsteczkę).
- **Przeliczenie (skrypt)**: Γ_sól,EO = −ν_W·m_s/55.51, stąd μ_x = ν_W·ν·RT·(1+ε)/55.51.
  Dla NaCl, z (1+ε) z Pitzera dla 0.1–1 m: **μ_x = +47 do +50 cal/(mol·molal EO)**. Zgadza się z A1/A2.
  Dla Na₂SO₄ (tyloxapol, ν_W ≈ 7, (1+ε) ≈ 0.85): ≈ **+190 cal**, podobny rząd co Na₂HPO₄ z A2 (243).
- McAfee & Annunziata 2015 Langmuir (tylko abstrakt): w solach SCN⁻ zachodzi salting-in przez **wiązanie anionu**,
  a łańcuch PEG zyskuje „extrinsic negative charge”. Specyficzne wiązanie z PEG w wodzie jest więc
  udokumentowane dla chaotropowych **anionów**, nie dla Na⁺.

### A4. Potencjometria ISE (grupa Morales / Galleguillos / Hernández-Luis), PEG 4000 0–25 wt%
- **NaCl**: Morales et al. 2009, Fluid Phase Equilib. 281, 120. Ogniwo Na-ISE|NaCl, PEG 4000 (Y)|Cl-ISE,
  288–308 K, m = 0.003–5 mol/kg (opis z metadanych/wyszukiwarki; abstraktu ani pełnego tekstu nie udało się
  pobrać, bo Elsevier blokuje dostęp). **Znaku ΔG_t(NaCl) nie znam.** To jest bezpośredni pomiar, którego tu brakuje.
- **NaClO₄**: Morales et al. 2020, IJCCE 39, 145 (pełny tekst, OA). Tabela 7, 298.15 K: ΔG_t° = −0.06, −0.15,
  −0.29, −0.43, −0.48 kJ/mol przy 5–25 wt% PEG 4000 (u = 0.05 kJ/mol), w skali molalności na kg *mieszanego*
  rozpuszczalnika. Po przeliczeniu na skalę wodną (+νRT ln(1−Y)): **μ_x(NaClO₄) ≈ −60 do −65 cal/(mol·molal EO)**,
  liniowo w całym zakresie (skrypt, sekcja 5). Ujemny znak dotyczy chaotropowego anionu (ClO₄⁻, salting-in) i jest
  zgodny z A3 (SCN⁻). Autorzy sami zaznaczają, że ogniwo „is not exactly thermodynamic” (potencjał asymetrii).
- NaNO₃ w PEG 4000 (Hernández-Luis et al. 2010 JCED; tylko abstrakt): te same warunki, w abstrakcie brak liczb.

### A5. Aktywność wody PEG + NaCl / fosforany (izopiestyka)
- Sadeghi & Ziamajidi 2006, J. Chem. Thermodyn. 38, 1335: PEG 400 + NaCl, 293–313 K, ulepszona metoda
  izopiestyczna. Abstraktu nie mam z wydawcy. Według opisu z rekordu repozytorium (przez wyszukiwarkę, niezweryfikowane
  co do treści) obniżenie prężności pary w układzie trójskładnikowym jest **większe niż suma układów
  dwuskładnikowych**. To oznacza ΔOsm > 0, czyli μ_x > 0, jakościowo zgodnie z A1–A3.
- Lin et al. 1996 JCED 41, 1040 (tylko abstrakt): izopiestyka przy 25 °C dla PEG + MgSO₄, (NH₄)₂SO₄, **KH₂PO₄, K₂HPO₄**.
  Dane dopasowano tylko równaniem empirycznym, liczb nie mam. **Pełny tekst pozwoliłby policzyć μ_x(PEG, fosforan potasu)
  wprost z równ. S1 Knowlesa.**
- Jahani et al. 2014 JCT 69, 125 (PEG 2000 + siarczany, 298 K): plik ThermoML z NIST zawiera tylko układy
  dwuskładnikowe sól–woda (PEG nie jest w nim reprezentowany), więc obliczenie było niemożliwe.

### A6. Glikol etylenowy (EG)
- Li et al. 2011 Chin. J. Chem. 29, 2007 (tylko abstrakt): γ±(NaCl) w EG–woda, w = 0.10–0.40 EG, 288–308 K,
  potencjometria. W abstrakcie brak liczb.
- Ceccattini et al. 1997 J. Solution Chem. 26, 1169 (tylko metadane): NaCl w EG–woda, EMF, 25 °C.
- Szacunek z A2: dla EG μ_x ≈ β_2E. Dla NaCl daje to **+23.8 cal/(mol·molal EG)**, dla Na₂HPO₄ **+88 cal**.
  Uwaga: β_2E to przeskalowany glicerol, a Knowles 2011 podaje dla polioli z KCl ≈ 0 ± 0.013 kcal.
  Dla EG znak jest więc niepewny, a wartość mała.

### A7. Salting-out / punkt zmętnienia (Hofmeister)
- Florin, Kjellander, Eriksson 1984 (tylko abstrakt): punkt zmętnienia PEO zmierzono dla wszystkich halogenków K,
  chlorków i wodorotlenków metali alkalicznych. Model zakłada **strefę zubożoną w sól wokół łańcucha**
  („a zone with decreased salt concentration”), czyli wykluczenie soli, zgodnie z μ_x > 0.
- Ataman & Boucher 1982 (tylko abstrakt): szereg Hofmeistera dla **anionów** w salting-out PEO, w tym fosforanu potasu.
  Kolejność zależy od tego, czy liczy się na mol anionu, na ładunek, czy na siłę jonową.
- Boucher & Hines 1978 (tylko abstrakt): θ(PEO, woda) = 369 ± 3 K. Liniowa zależność δθ/δI dla soli Na i K.
  Autorzy przypisują efekty „binding of salt to the polymer and to water structure breaking” (to interpretacja, bez stałych).
- Hey, Jackson, Yan 2005 Polymer (tylko metadane).
- **Liczbowych współczynników Setschenowa dla PEG + NaCl/NaPi w pełnym tekście nie znalazłem.**

---

## (b) Układy dwufazowe PEG–fosforan (ATPS)

- Literatura: Ananthapadmanabhan & Goddard 1987 (PEO + sole nieorganiczne), Snyder, Cole, Szlag 1992 (PEG + różne sole,
  25 °C), Zafarani-Moattar & Sadeghi 2001 (PEG + NaH₂PO₄ / Na₂HPO₄, LLE + korelacja), Huddleston, Willauer, Rogers 2003
  (binodale PEG + sole, 25 °C). Wszystkie DOI zweryfikowano, **ale do żadnego nie mam pełnego tekstu ani abstraktu**
  (ACS i Elsevier za paywallem lub Cloudflare). Nie podaję stąd żadnych liczb.
- **Ekstrakcja b_PEG,sól: nie wykonana.** Linie wiązania leżą przy ≳0.5 m soli i ≳10 wt% PEG, czyli daleko od naszych
  35 mM. Krzyżowy współczynnik wirialny dopasowany do binodali zależy od modelu (Edmond–Ogston/Pitzer/Wilson)
  i jest skorelowany z wyrazami własnymi. Bez danych liczbowych nie da się tego zrobić rzetelnie.
- **Test spójności (szacunek, skrypt):** przy μ_x(Na₂HPO₄) = 243 cal/(mol·molal EO) mamy ∂(μ_PEG/RT)/∂m_sól ≈
  89 EO × 0.41 ≈ **37 na molal soli** dla PEG 4000. Przy 0.5 m to ok. 18 RT na łańcuch (liniowo), więc
  separacja faz przy ~0.5–1 m fosforanu jest jakościowo oczekiwana. Dla NaCl (42 cal) efekt jest ~6× słabszy,
  zgodnie ze znanym faktem, że NaCl sam nie tworzy ATPS z PEG w zwykłych warunkach (to ostatnie z wiedzy ogólnej,
  bez cytatu, więc NIEZWERYFIKOWANE).
- Lepszym źródłem parametru niskostężeniowego niż ATPS jest A2 (β z VPO < 0.6 m).

---

## (c) Wiązanie kationów przez PEG w wodzie

| praca | metoda / warunki | wynik | dostęp |
|---|---|---|---|
| Okada 1991 JCS Faraday Trans 87, 3027 | **faza żywicy jonowymiennej**, nie roztwór wodny | selektywność kationów z ΔS desolwatacji; brak K w wodzie w abstrakcie | abstrakt |
| Okada 1990 Macromolecules 23, 4216 | „efficient evaluation” kompleksowania POE–M⁺ | — | tylko metadane |
| Ohki, Harada, Okada 2007 JPCB | woltamperometria transferu jonów woda/nitrobenzen + XAFS, PEG 200 | sugestia kompleksowania M⁺ „even in relatively diluted PEG solutions”; XAFS: kompleks K⁺ przy **~33% PEG200**. Jednocześnie: wzrost potencjału przy niskim PEG wskazuje na kompleksy tworzone **„at the interface rather than in bulk solution”** | abstrakt |
| Hakem, Lal, Bockstaller 2004 Macromolecules | SANS + RPA | w rozpuszczalnikach o wysokiej ε lub silnych wiązaniach H „solvation of ions counteracts the complex formation … PEO effectively remains a neutral polymer” | abstrakt |
| Lundberg, Bailey, Callard 1966 | KI–PEO, **metanol**, stan stały | ~1 sól na 9 EO; postulują asocjację **anionu** | abstrakt |
| Robertson et al. 2007 PNAS | nanopor α-HL, PEG | spektrometria mas w roztworze; abstrakt nie wspomina o kationach | abstrakt |
| Rodrigues et al. 2008 Biophys J | nanopor α-HL, **1–4 M KCl**, 5 mM Tris-cytrynian pH 7.5, 25 °C, PEG 600–4000 | **brak stałych wiązania i brak pomiaru Na⁺**. Kompleksowanie kationów cytowane za Okadą jako hipoteza. Energia oddziaływania PEG–por na monomer przy 40 mV: 0.27 ± 0.03 kT (4 M), 0.19 (3 M), 0.13 (2 M), 0.03 ± 0.01 kT (1 M) | pełny tekst (PMC przez WebFetch) |
| Reiner et al. 2010 PNAS | teoria + nanopor | ΔG tworzenia K⁺–PEG **wewnątrz poru** ≈ **−49.7 meV** (≈ −1.9 kT); partycjonowanie ≈ 0.76 meV/monomer | abstrakt (PubMed) |
| Breton et al. 2013 JPCL | nanopor, Li⁺/Na⁺/K⁺/Rb⁺/Cs⁺, „high salt concentration” | PEG zachowuje się jak naładowany dla wszystkich kationów poza Li⁺; liczby skompleksowanych kationów tylko w pełnym tekście | abstrakt |

Ocena krytyczna:
1. Wszystkie dowody „wiązania Na⁺/K⁺” w wodzie pochodzą z **porów (nanopor α-HL przy 1–4 M soli), granic faz
   (woda/nitrobenzen) albo faz niewodnych (żywica, metanol, ciało stałe)**. Żaden nie jest pomiarem objętościowym
   (bulk) w rozcieńczonej soli. Ohki et al. sami zauważają, że przy niskim PEG kompleks tworzy się na granicy faz.
2. Bezpośrednia termodynamika objętościowa (A1–A3) daje dla K⁺ i Na⁺ netto β_I ≈ +3 i +5 cal (lekkie wykluczenie),
   a NaCl/KCl jako całość silnie dodatnio. Nanopory przypisują K⁺ wiązanie silniejsze niż Na⁺, a mimo to KCl
   ma wyraźnie dodatnie μ_x.
3. **Przeliczenie naszych stałych na μ_x (skrypt, sekcja 2):** wiązanie z K_b na EO daje wkład μ_x ≈ −RT·K_b
   (słabe wiązanie, z elektroobojętnością). DOSY 0.05–0.17 M⁻¹ odpowiada **−30 do −100 cal/(mol·molal EO)**.
   Bielec 2022 (0.49 M⁻¹) odpowiada **−290 cal**. Zmierzona wartość netto dla NaCl to **+42 ± 3** (VPO),
   a dla samego Na⁺ +4.7 ± 2. Wartość 0.49 M⁻¹ jest więc niezgodna z VPO co do znaku i o rząd wielkości.
   Wartości DOSY musiałyby być w całości kompensowane wykluczeniem, a wtedy są dla K (który zależy tylko od a±)
   termodynamicznie niewidoczne. DOSY mierzy spowolnienie dyfuzji Na⁺ (przeszkody, mikrolepkość,
   przejściowe kontakty), a nie współczynnik preferencyjnego oddziaływania.
4. Hofmeister: dla PEO decydują **aniony** (Ataman & Boucher 1982; McAfee 2014: kation zmienia N_w niewiele,
   SO₄²⁻ ~3× bardziej niż Cl⁻). Fosforan (HPO₄²⁻) jest silnym anionem wysalającym (β_I = 234 cal, Knowles 2015).

---

## (d) Czy PEG obniża czy podnosi aktywność Na-fosforanu / NaCl przy 1–40 wt%?

- **NaCl / KCl: podnosi.** Trzy niezależne metody (VPO dla PEG 1000/2000 i tetraEG; dyfuzja wieloskładnikowa
  z walidacją izopiestyczną dla PEG 20k; izopiestyka PEG 400 według opisu abstraktu) dają μ_x ≈ +32 do +50 cal/(mol·molal EO).
- **Na-fosforan: podnosi, prawdopodobnie znacznie silniej.** Jedyna liczba to β_I(HPO₄²⁻) = 234 ± 5.5 cal (Knowles 2015).
  Jest spójna z szeregiem Hofmeistera, z ATPS PEG–fosforan i z ν_W ≈ 7/EO dla siarczanu (Annunziata).
  Dla H₂PO₄⁻ (~20% w naszym buforze) nie ma danych.
- Sól o ujemnym μ_x w PEG znalazłem tylko dla anionów chaotropowych (NaClO₄: −60 cal, Morales 2020; SCN⁻: wiązanie anionu).
- Nie znalazłem pracy, która dla rozcieńczonego NaCl lub Na-fosforanu w PEG pokazywałaby **obniżenie** a± przez wiązanie Na⁺.

### Najlepsze oszacowanie dla naszego układu (SZACUNEK, nachylenie graniczne)

| wielkość | NaCl jako proxy | Na₂HPO₄ jako proxy (górna granica) |
|---|---|---|
| μ_x [kcal/(mol·molal EO)] | **+0.042 ± 0.005** (zakres lit. 0.032–0.050) | **+0.24** (bez H₂PO₄⁻; niepewność systematyczna ±50%) |
| ∂ln a±/∂m_EO | 0.036 | 0.137 (średnia jonowa soli 2:1) |
| ∂ln K/∂m_EO (S_K = 3.5) | +0.12 | +0.48 |
| Δ ln K, 10 wt% PEG ≥1000 | +0.3 | +1.2 |
| Δ ln K, 40 wt% PEG ≥1000 (liniowo) | +1.9 | +7 (na pewno przeszacowane) |
| EG 40 wt% (β_2E, niepewne) | +0.8 | +1.9 |

Pełna tabela dla EG, PEG 200–35 000 i 10/20/40 wt% jest w `q2_mux_estimate_output.txt`, sekcja 3.
**Moja ocena:** znak μ_x > 0 jest pewny dla NaCl/KCl i bardzo prawdopodobny dla NaPi. Rząd wielkości ∂ln a±/∂m_EO to
0.04–0.14, wartość centralna ~0.08 ± 0.05.

Zastrzeżenia:
- μ_x to nachylenie graniczne (< 0.6 m). Preferencyjna hydratacja maleje ze wzrostem stężenia soli i PEG (Tan 2008),
  więc liniowa ekstrapolacja do 40 wt% przeszacowuje efekt.
- S_K zdefiniowane względem a± soli 1:1 przenosi się niejednoznacznie na mieszany bufor fosforanowy.
- Podział na jony (Na⁺ vs anion) jest ekstratermodynamiczny i pochodzi z dopasowania Knowlesa.

**Konsekwencja dla pracy:** jeśli w naszych danych PEG obniża K, to wyjaśnienie „wiązanie Na⁺ obniża a_Na” stoi
w sprzeczności z termodynamiką objętościową. Efekt solny powinien mieć znak przeciwny (stabilizacja), więc
obserwowana destabilizacja musiałaby pochodzić z innego źródła (np. korzystne oddziaływania PEG z eksponowanymi
zasadami ssDNA; Knowles 2011 i 2015). Dodatkowo efekt solny trzeba by od niej odjąć, a nie do niej dodać.

### Pojedynczy pomiar, który by to rozstrzygnął
**VPO (lub izopiestyka) dokładnie w naszym buforze**, według protokołu Knowles 2011 (równ. S1–S2):
ΔOsm = Osm(NaPi + PEG) − Osm(NaPi) − Osm(PEG) przy np. 0.1–0.3 m NaPi pH 7.4 oraz PEG 1000 (lub PEG 400) i EG
do ~0.6 m EO. Nachylenie ΔOsm względem m_s·m_EO daje μ_x/RT. Oczekiwany sygnał przy m_s = 0.2 i m_EO = 2.5:
**36 mOsm (gdy NaCl-podobnie) do 205 mOsm (gdy Na₂HPO₄-podobnie)**, czyli znacznie powyżej precyzji VPO
(~1–2 mOsm). Przy samych 20 mM sygnał wynosi tylko 4–20 mOsm. Kontrola: ten sam pomiar z NaCl, żeby porównać
z wartością +42 cal. Pomiar jednocześnie rozstrzyga znak, a więc czy Na⁺ jest „wiązany” netto.

---

## Luki
- Brak pełnego tekstu Morales 2009 (γ±, NaCl w PEG 4000). To najbardziej bezpośrednia praca dla NaCl.
- Brak danych β dla H₂PO₄⁻. Brak izopiestyki PEG + Na-fosforan w rozcieńczeniu (Lin 1996 ma sole K-fosforanowe).
- Nie odczytałem SI Knowles 2015 (lista soli i stężeń dla β, temperatura VPO).
- Liczby z ATPS (b) i z nanoporów Breton 2013 są niedostępne (paywall).

---

## Bibliografia

Status: „Crossref OK” oznacza, że DOI zweryfikowano przez api.crossref.org/works/<DOI> 2026-10-09.

1. Knowles D.B., LaCroix A.S., Deines N.F., Shkel I., Record M.T. Jr. „Separation of preferential interaction and excluded volume effects on DNA duplex and hairpin stability”. *PNAS* 108, 12699–12704 (2011). DOI 10.1073/pnas.1103382108. Crossref OK. **Pełny tekst + SI** (lokalnie, literature/theoretical_models/).
2. Knowles D.B., Shkel I.A., Phan N.M., Sternke M., Lingeman E., Cheng X., Cheng L., O'Connor K., Record M.T. Jr. „Chemical Interactions of Polyethylene Glycols (PEGs) and Glycerol with Protein Functional Groups: Applications to Effects of PEG and Glycerol on Protein Processes”. *Biochemistry* 54, 3528–3542 (2015). DOI 10.1021/acs.biochem.5b00246. Crossref OK. **Pełny tekst główny** (PMC4464906, HTML przez WebFetch); SI nieczytane.
3. Tan C., Albright J.G., Annunziata O. „Determination of Preferential Interaction Parameters by Multicomponent Diffusion. Applications to Poly(ethylene glycol)−Salt−Water Ternary Mixtures”. *J. Phys. Chem. B* 112, 4967–4974 (2008). DOI 10.1021/jp711725e. Crossref OK. Tylko abstrakt.
4. McAfee M.S., Zhang H., Annunziata O. „Amplification of Salt-Induced Polymer Diffusiophoresis by Increasing Salting-Out Strength”. *Langmuir* 30, 12210–12219 (2014). DOI 10.1021/la503214b. Crossref OK. Tylko abstrakt (wartość 2.4 H₂O/EO z poz. 6).
5. McAfee M.S., Annunziata O. „Effects of Salting-In Interactions on Macromolecule Diffusiophoresis and Salt Osmotic Diffusion”. *Langmuir* 31, 1353–1361 (2015). DOI 10.1021/la5046223. Crossref OK. Tylko abstrakt.
6. Annunziata O. „The Salt-Induced Diffusiophoresis of Nonionic Micelles—Does the Salt-Induced Growth of Micelles Influence Diffusiophoresis?”. *Molecules* 29, 3618 (2024). DOI 10.3390/molecules29153618. Crossref OK. **Pełny tekst** (OA, PMC11314613).
7. Cruz Barrios E., Penino K.V., Annunziata O. „Diffusiophoresis of a Nonionic Micelle in Salt Gradients; Roles of Preferential Hydration and Salt-Induced Surfactant Aggregation”. *Int. J. Mol. Sci.* 23, 13710 (2022). DOI 10.3390/ijms232213710. Crossref OK. **Pełny tekst** (OA, PMC9696892).
8. McAfee M.S., Annunziata O. „Effect of Particle Size on Salt-Induced Diffusiophoresis Compared to Brownian Mobility”. *Langmuir* 30, 4916–4923 (2014). DOI 10.1021/la500982u. Crossref OK. Tylko abstrakt.
9. Morales J.W., Galleguillos H.R., Taboada M.E., Hernández-Luis F. „Activity coefficients of NaCl in PEG 4000+water mixtures at 288.15, 298.15 and 308.15 K”. *Fluid Phase Equilib.* 281, 120–126 (2009). DOI 10.1016/j.fluid.2009.03.013. Crossref OK. Tylko metadane (abstrakt niedostępny).
10. Morales J.W., Galleguillos H.R., Hernández-Luis F., Rodríguez-Raposo R. „Activity Coefficients of NaClO4 in (PEG 4000 + H2O) Mixtures at (288.15, 298.15 and 308.15) K”. *Iran. J. Chem. Chem. Eng.* 39(2), 145–157 (2020). DOI 10.30492/ijcce.2020.33284. Zweryfikowano przez doi.org i stronę wydawcy (brak w Crossref). **Pełny tekst** (OA).
11. Hernández-Luis F., Morales J.W., Graber T.A., Galleguillos H.R. „Activity Coefficients of NaNO3 in Poly(ethylene glycol) 4000 + Water Mixtures at (288.15, 298.15, and 308.15) K”. *J. Chem. Eng. Data* 55, 4082–4087 (2010). DOI 10.1021/je1002835. Crossref OK. Tylko abstrakt.
12. Sadeghi R., Ziamajidi F. „Water activities of ternary mixtures of poly(ethylene glycol), NaCl and water over the temperature range of 293.15 K to 313.15 K”. *J. Chem. Thermodyn.* 38, 1335–1343 (2006). DOI 10.1016/j.jct.2006.02.003. Crossref OK. Tylko opis z repozytorium/wyszukiwarki (abstrakt wydawcy niedostępny).
13. Lin D.-Q., Zhu Z.-Q., Mei L.-H., Yang L.-R. „Isopiestic Determination of the Water Activities of Poly(ethylene glycol) + Salt + Water Systems at 25 °C”. *J. Chem. Eng. Data* 41, 1040–1042 (1996). DOI 10.1021/je960065a. Crossref OK. Tylko abstrakt.
14. Jahani F., Abdollahifar M., Haghnazari N. „Thermodynamic equilibrium of the polyethylene glycol 2000 and sulphate salts solutions”. *J. Chem. Thermodyn.* 69, 125–131 (2014). DOI 10.1016/j.jct.2013.10.015. Crossref OK. Tylko metadane i plik ThermoML (same układy dwuskładnikowe).
15. Vergara A., Paduano L., Vitagliano V., Sartorio R. „Multicomponent Diffusion in Crowded Solutions. 1. Mutual Diffusion in the Ternary System Poly(ethylene glycol) 400−NaCl−Water”. *Macromolecules* 34, 991–1000 (2001). DOI 10.1021/ma0011800. Crossref OK. Tylko abstrakt.
16. Li S.-N., Tang J., Ma Y., Zhai Q., Jiang Y., Hu M. „Activity Coefficients of NaCl in Ethylene Glycol‐Water Mixtures Using Potentiometric Measurements at 288.15, 298.15 and 308.15 K”. *Chin. J. Chem.* 29, 2007–2013 (2011). DOI 10.1002/cjoc.201180349. Crossref OK. Tylko abstrakt.
17. Ceccattini P., Mussini P.R., Mussini T. „Thermodynamics of NaCl in Aqueous Ethylene Glycol, Acetonitrile, and 1,4-Dioxane Mixtures from Emf Measurements at 25°C”. *J. Solution Chem.* 26, 1169–1186 (1997). DOI 10.1023/A:1022981107236. Crossref OK. Tylko metadane.
18. Kalidas C., Hefter G., Marcus Y. „Gibbs Energies of Transfer of Cations from Water to Mixed Aqueous Organic Solvents”. *Chem. Rev.* 100, 819–852 (2000). DOI 10.1021/cr980144k. Crossref OK. Tylko metadane (źródło dla EG–woda, nieczytane).
19. Ananthapadmanabhan K.P., Goddard E.D. „Aqueous biphase formation in polyethylene oxide-inorganic salt systems”. *Langmuir* 3, 25–31 (1987). DOI 10.1021/la00073a005. Crossref OK. Tylko metadane.
20. Snyder S.M., Cole K.D., Szlag D.C. „Phase compositions, viscosities, and densities for aqueous two-phase systems composed of polyethylene glycol and various salts at 25 °C”. *J. Chem. Eng. Data* 37, 268–274 (1992). DOI 10.1021/je00006a036. Crossref OK. Tylko metadane.
21. Zafarani-Moattar M.T., Sadeghi R. „Liquid–liquid equilibria of aqueous two-phase systems containing polyethylene glycol and sodium dihydrogen phosphate or disodium hydrogen phosphate”. *Fluid Phase Equilib.* 181, 95–112 (2001). DOI 10.1016/S0378-3812(01)00373-9. Crossref OK. Tylko metadane.
22. Huddleston J.G., Willauer H.D., Rogers R.D. „Phase Diagram Data for Several PEG + Salt Aqueous Biphasic Systems at 25 °C”. *J. Chem. Eng. Data* 48, 1230–1236 (2003). DOI 10.1021/je034042p. Crossref OK. Tylko metadane.
23. Hey M.J., Jackson D.P., Yan H. „The salting-out effect and phase separation in aqueous solutions of electrolytes and poly(ethylene glycol)”. *Polymer* 46, 2567–2572 (2005). DOI 10.1016/j.polymer.2005.02.019. Crossref OK. Tylko metadane.
24. Florin E., Kjellander R., Eriksson J.C. „Salt effects on the cloud point of the poly(ethylene oxide)+water system”. *J. Chem. Soc., Faraday Trans. 1* 80, 2889 (1984). DOI 10.1039/F19848002889. Crossref OK. Tylko abstrakt.
25. Ataman M., Boucher E.A. „Properties of aqueous salt solutions of poly(ethylene oxide)”. *J. Polym. Sci. Polym. Phys. Ed.* 20, 1585–1592 (1982). DOI 10.1002/pol.1982.180200907. Crossref OK. Tylko abstrakt.
26. Boucher E.A., Hines P.M. „Properties of aqueous salt solutions of poly(ethylene oxide): Thermodynamic quantities based on viscosity and other measurements”. *J. Polym. Sci. Polym. Phys. Ed.* 16, 501–511 (1978). DOI 10.1002/pol.1978.180160312. Crossref OK. Tylko abstrakt.
27. Lundberg R.D., Bailey F.E., Callard R.W. „Interactions of inorganic salts with poly(ethylene oxide)”. *J. Polym. Sci. A-1* 4, 1563–1577 (1966). DOI 10.1002/pol.1966.150040620. Crossref OK. Tylko abstrakt.
28. Okada T. „Characterization of poly(oxyethylene) complex formation with alkali-metal cations in a cation-exchange resin phase”. *J. Chem. Soc., Faraday Trans.* 87, 3027 (1991). DOI 10.1039/FT9918703027. Crossref OK. Tylko abstrakt.
29. Okada T. „Efficient evaluation of poly(oxyethylene) complex formation with alkali-metal cations”. *Macromolecules* 23, 4216–4219 (1990). DOI 10.1021/ma00221a006. Crossref OK. Tylko metadane.
30. Ohki T., Harada M., Okada T. „Structural and Thermodynamic Aspects of Ionic Solvation in Concentrated Aqueous Poly(ethylene glycol)”. *J. Phys. Chem. B* 111, 7245–7252 (2007). DOI 10.1021/jp071666j. Crossref OK. Tylko abstrakt.
31. Hakem I.F., Lal J., Bockstaller M.R. „Binding of Monovalent Ions to PEO in Solution: Relevant Parameters and Structural Transitions”. *Macromolecules* 37, 8431–8440 (2004). DOI 10.1021/ma0495801. Crossref OK. Tylko abstrakt.
32. Robertson J.W.F., Rodrigues C.G., Stanford V.M., Rubinson K.A., Krasilnikov O.V., Kasianowicz J.J. „Single-molecule mass spectrometry in solution using a solitary nanopore”. *PNAS* 104, 8207–8211 (2007). DOI 10.1073/pnas.0611085104. Crossref OK. Tylko abstrakt.
33. Rodrigues C.G., Machado D.C., Chevtchenko S.F., Krasilnikov O.V. „Mechanism of KCl Enhancement in Detection of Nonionic Polymers by Nanopore Sensors”. *Biophys. J.* 95, 5186–5192 (2008). DOI 10.1529/biophysj.108.140814. Crossref OK. **Pełny tekst** (PMC2586564, HTML przez WebFetch).
34. Reiner J.E., Kasianowicz J.J., Nablo B.J., Robertson J.W.F. „Theory for polymer analysis using nanopore-based single-molecule mass spectrometry”. *PNAS* 107, 12080–12085 (2010). DOI 10.1073/pnas.1002194107. Crossref OK. Tylko abstrakt (PubMed).
35. Breton M.F., Discala F., Bacri L., Foster D., Pelta J., Oukhaled A. „Exploration of Neutral Versus Polyelectrolyte Behavior of Poly(ethylene glycol)s in Alkali Ion Solutions using Single-Nanopore Recording”. *J. Phys. Chem. Lett.* 4, 2202–2208 (2013). DOI 10.1021/jz400938q. Crossref OK. Tylko abstrakt.
36. Bielec K., Kowalski A., Bubak G., Witkowska Nery E., Hołyst R. „Ion Complexation Explains Orders of Magnitude Changes in the Equilibrium Constant of Biochemical Reactions in Buffers Crowded by Nonionic Compounds”. *J. Phys. Chem. Lett.* 13, 112–117 (2022; online 2021). DOI 10.1021/acs.jpclett.1c03596. Crossref OK. **Pełny tekst + SI** (lokalnie).

Uwaga o dostępie: pubs.acs.org, pubs.rsc.org i sciencedirect.com zwracały stronę Cloudflare. Pliki PDF z PMC były
chronione proof-of-work, więc pełne teksty z PMC czytałem jako HTML przez WebFetch (ekstrakcja z modelem
pomocniczym, cytaty dosłowne). Plik IJCCE i XML-e z Europe PMC są w `literature/review_2026-10/q2/`.
