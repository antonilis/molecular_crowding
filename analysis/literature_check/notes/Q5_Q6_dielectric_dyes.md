# Q5 i Q6: stała dielektryczna, aktywność wody, oddziaływania preferencyjne oraz fotofizyka barwników

Przegląd z 2026-10-09. Każdy DOI sprawdzono w Crossref API (`api.crossref.org/works/<DOI>`).
Oznaczenia: **[PT]** = przeczytany pełny tekst, **[ABS]** = tylko abstrakt (Europe PMC/Crossref),
**[META]** = tylko metadane Crossref, bez treści. Pobrane PDF-y są w `literature/review_2026-10/q5_q6/`.
Pozostałe teksty pełne (PMC/Europe PMC) czytałem jako HTML/XML, bez zapisywania PDF-u.

Skrypty (uruchamiane przez `.venv/bin/python`):
- `analysis/literature_check/q5_dielectric_PB.py`: wkład czysto dielektryczny (Poisson–Boltzmann, model cylindrów).
- `analysis/literature_check/q5_water_activity.py`: a_w roztworów oraz „pozorne Δn_w” wynikające z naszych K.

Dane wejściowe z repozytorium (tylko odczyt): `source_data/MacromoleculeEquilibria_Crowding/OriginalData_*.xlsx`
(K przy 25 °C, 20 mM PB, K0 = 1.76·10⁹ M⁻¹) oraz `source_data/IonStrengthDependency/Na_FRET.csv`.

---

## Q5. Stała dielektryczna, aktywność wody i oddziaływania preferencyjne

### 5.1 Co wiadomo (liczby z warunkami)

**(a) ε mieszanin w 25 °C**

| układ | ε | źródło i warunki | uwagi |
|---|---|---|---|
| woda | 78.4 | wartość standardowa, 25 °C | |
| EG 27.8 wt% / 46.4 wt% / 100% | 72.71 / 67.18 / 41.78 | Zagrebin i in. 2010 (pomiar szerokopasmowy 0.1–89 GHz), cytowane w Kaiser i in. 2016, Tabela 1 **[PT, źródło wtórne]** | Woda ma tam ε = 80.08, czyli wartość z ok. 20 °C. Temperatura serii NIEZWERYFIKOWANA (oryginał płatny). |
| **EG 40 wt%** | **≈ 69 (seria „20 °C”) → ≈ 67.5 w 25 °C** | interpolacja liniowa w wt%, następnie przeskalowanie o 78.4/80.08 | założenie: ten sam względny spadek ε w 20 i 25 °C; zakres 65–69 |
| PEG 400, ułamek objętościowy 0.3 / 0.4 | 64.2 / 55.3 (1 MHz); 64.4 / 60.0 (2 MHz) | Yurquina i in. 2007, Tabela 2, 25.0 °C **[PT]** | Pomiar mostkiem przy 1–2 MHz, nie ekstrapolacja statyczna. Dla wody wyszło 77.3/78.5, więc dokładność ok. ±1–2. |
| **PEG 400 40 wt%** (ułamek objętościowy ≈ 0.37) | **≈ 58–61** | interpolacja w tabeli Yurquiny | |
| „PEG 400 40 wt%” = 56.74 (Bielec 2022 SI, sekcja 6) | — | źródło: Panraksa 2020 **[PT]** | **To nie jest pomiar.** Panraksa *liczy* ε z liniowego mieszania ułamków objętościowych dla 40 % **v/v**, z wartością dla PEG z literatury. Wartość jest przypadkiem bliska pomiarowi Yurquiny, ale nie należy jej cytować jako danej eksperymentalnej. |
| PEG 6000 itp., 40 wt% | brak zmierzonej wartości | Kaatze i in. 1978 **[META]** (płatny), Shikata 2006 **[ABS]**: 3.7 cząsteczki H₂O na jednostkę EO dla M > 1500, 25 °C | Robocze założenie NIEZWERYFIKOWANE: decrement na jednostkę masy jak dla PEG 400, czyli ε ≈ 58–62. **Luka.** |

Długość Bjerruma: 0.715 nm (woda) → ok. 0.83 nm (EG 40 wt%) → ok. 0.92–0.97 nm (PEG 400 40 wt%).

**(b) Aktywność wody / osmolalność**

Nakano i in. 2004, SI, Tabela S2 **[PT, lokalny PDF]**. Pomiar VPO, a dla EG metodą obniżenia temperatury krzepnięcia (FPD), więc dla EG wartość nie odnosi się ściśle do 25 °C. Tło: **1 M NaCl** + 10 mM Na₂HPO₄ + 1 mM EDTA. Poniżej Δln a_w = ln a_w(kosolut) − ln a_w(bez kosolutu):

| kosolut | 20 wt% | 30 wt% | 40 wt% |
|---|---|---|---|
| EG | −0.063 | −0.110 | −0.141 |
| PEG 200 (1 M NaCl) | −0.030 | −0.047 | −0.059 |
| PEG 200 (100 mM NaCl) | −0.028 | −0.042 | −0.051 |
| PEG 1000 | −0.019 | — | −0.033 |
| PEG 6000 / PEG 8000 | −0.017 / −0.014 | — | — |

Równanie stanu PEG (Cohen, Podgornik, Hansen, Parsegian 2009 **[PT, lokalny PDF]**):
Π = RT·C/(M_m N) + α·RT·(C v̄)^{9/4}/(M_m v̄), z α = 0.49, v̄ = 0.825 mL/g, M_m = 44 g/mol.
Dla układu binarnego PEG/woda daje to: PEG 6000 20 wt% Π ≈ 0.71 MPa (ln a_w ≈ −0.005), 40 wt% Π ≈ 3.4 MPa (ln a_w ≈ −0.025).
PEG ≥ 3k przy 40 wt%: ln a_w ≈ −0.024 do −0.026.

**Krytycznie:** dla PEG 6000 20 wt% Nakano podaje −0.017, a równanie Cohena daje −0.005, czyli 3× mniej. To niedodatniość PEG–NaCl przy 1 M soli. Wartości a_w Nakano dla PEG nie przenoszą się na nasz bufor (35 mM Na⁺). Dla EG 40 wt% rozwiązanie idealne (m = 10.7 mol/kg) daje −0.19, a Nakano −0.14. Przyjmuję zakres −0.14 do −0.19.

**(c) Δn_w dla krótkich dupleksów (osmotic stress)**

- Nakano 2004 **[PT]**: 8-mer ATGCGCAT, wykres ln K vs ln a_w przy 25 °C. PEG 200–8000 w 1 M NaCl daje **15.0 ± 0.6 H₂O/bp**, a PEG 200 w 100 mM NaCl **14.4 ± 0.5**. Glicerol daje 4.7 ± 0.3, EG leży blisko glicerolu („2–4 w literaturze”). Sami autorzy wiążą różnicę ze strukturą kosolutu (wicynalne OH) i wymieniają ε jako możliwy, niezbadany czynnik.
- Nakano i in. 2012 **[PT]**: spinki i dupleksy, 37 °C, 1 M NaCl, PEG 200. Wynik: **6–14 H₂O/bp**, średnio 9–14 dla sekwencji losowych. Dupleks PNA ma Δn_w ≈ 0, hybrydy PNA/DNA około połowy wartości DNA. Wyniki w KCl i CsCl pokrywają się z NaCl.
- Rozners 2004 **[PT]**: 6–12 bp, 300 mM NaCl, z zależności Tm. Dla DNA z **EG 1.8–4.1 H₂O/bp**, z glicerolem 1.8–2.6, z acetamidem 3.2–5.2.
- Spink & Chaires 1999 **[ABS]**: długi DNA, małe osmolity, **ok. 4 H₂O/bp**. W EG wiązanie przeciwjonów „modestly decreases” wraz ze spadkiem a_w.
- Spink, Garbett, Chaires 2007 **[ABS]**: z EG, glicerolem i PEG 200 ΔH topnienia maleje (kompensacja entalpia–entropia). Dla PEG 1450 ΔH się nie zmienia.

**(d) Oddziaływania preferencyjne i mechanizmy**

- Stanley & Rau 2006 **[PT]**: osmotic stress na uporządkowanych macierzach DNA, 20 °C. **EG jest wykluczony tylko w ok. 5 %** (Π_excess/Π₀ = 0.05 ± 0.01), bez zależności od odległości. Glicerol 0.00 ± 0.02, metanol 0.13 (ΔN_w = 11 ± 4/bp), MPD 0.49. Wykluczenie skaluje się z nadmiarem atomów C nad grupami OH. Wniosek autorów: wpływ na ε i elektrostatykę DNA jest „secondary”. Wykluczenie MPD jest prawie takie samo w macierzach Co³⁺, Spd³⁺ i Na⁺-DNA, czyli działa bezpośrednio na powierzchnię, a nie przez ε.
- Nordstrom i in. 2006 **[ABS]**: EG gromadzi się lokalnie przy ssDNA odsłanianym podczas topnienia, niezależnie od składu GC.
- Lambert & Draper 2007 **[PT]**: wszystkie badane osmolity (w tym EG, metanol, glicerol) destabilizują struktury drugorzędowe RNA. Autorzy tłumaczą to akumulacją przy zasadach. Zależność od soli monowalentnej zmienia się w EG tylko „na granicy istotności” (cytują Nordstrom 2006 i Spink & Chaires).
- Hickey & Turner 1985 **[ABS]**: A7U7p, 1 M NaCl, 10 mol% kosolwentu. Kolejność stabilności: EG > glicerol > EtOH > 2-PrOH > DMSO > 1-PrOH > formamid > DMF > mocznik > dioksan. Kolejność nie śledzi ε: formamid, mimo ε wyższej od wody (wartość z wiedzy ogólnej, nie z tej pracy), należy do najsilniej destabilizujących. To jakościowy argument przeciw dominacji efektu dielektrycznego.
- Levine, Gordon, Jencks 1963 oraz Herskovits i in. 1961 **[META]**: klasyczne prace o denaturacji alkoholami. Nie miałem dostępu do treści, więc niczego z nich nie cytuję liczbowo.
- Jaiswal i in. 2018 **[PT]**: MD dupleksu 23 bp, 0–30 % obj. EG, 0.1 M NaCl. EG wchodzi w pierwszą powłokę hydratacyjną, łączna liczba wiązań H się nie zmienia, a gęstość Na⁺ przy DNA rośnie z EG. **Słowo „groove” nie pada w tekście** (grep). Zakomentowane zdanie w `Manuscript/main.tex` (l. 201), że EG „penetrate the DNA grooves”, jest szersze niż to, co praca pokazuje. Praca symuluje tylko dupleks, bez ss, więc nie daje ΔG hybrydyzacji.
- Feng i in. 2019 **[PT, XML z PMC]**: etery glikolu etylenowego (PEG 400, diglyme, do 50 wt%) obniżają energię stackingu zasad. Hiperchromizm wynosi 3–6 %, CD się nie zmienia, częstość rozstackowania rośnie 10–100× („open hole” ok. 10⁻² zamiast 10⁻⁴), a parowanie zasad ulega wzmocnieniu. PEG 40 % nie zmienia CD. **Badano PEG 400 i diglyme, nie sam EG.**
- Parsegian, Rand, Rau 2000 **[PT]**: osmotic stress, preferential hydration i crowding to równoważne zapisy relacji Gibbsa–Duhema. Pojedynczy kosolut mierzy tylko ΔN_ew = ΔN_w − (n_w/n_s)·ΔN_s. Rozdzielenie hydratacji od wiązania kosolutu wymaga **kilku kosolutów**.
- Knowles i in. 2011 **[PT lokalnie; analizowany osobno w `analysis/knowles_reframe/`]**: rozkład m-value PEG na oddziaływania grup końcowych i wewnętrznych z powierzchnią DNA oraz na objętość wykluczoną.

Opublikowanej wartości d ln K/d(1/ε) dla krótkiego dupleksu przy stałej soli **nie znalazłem**. Tak samo nie znalazłem eksperymentu, w którym zmieniano by ε przy stałym a_w i stałych oddziaływaniach preferencyjnych. **Luka.**

### 5.2 Wielkości użyteczne dla naszych danych i co policzyłem

**A. Wkład dielektryczny, `q5_dielectric_PB.py`**

Model: nieliniowy PB dla cylindrów w rezerwuarze soli 1:1 przy 35 mM, z całką ładowania.
- dsDNA: promień 1.0 nm, 24 ładunki na długości 13 × 0.34 nm.
- ssDNA (dwie nici): promień a_ss, 12 ładunków; skok b_ss **kalibrowany** tak, żeby model odtwarzał zmierzone nachylenie d ln K/d ln[Na⁺] = **3.49** (z `Na_FRET.csv`, 17.5–52.6 mM).
- Następnie zmieniam tylko ε.

| a_ss | b_ss (kal.) | ΔΔG_diel, EG 40 % (ε = 67.5) | ΔΔG_diel, PEG 400 40 % (ε = 59) |
|---|---|---|---|
| 0.5 nm | 0.39 nm | −0.9 kJ/mol | −1.8 kJ/mol |
| 0.7 nm | 0.35 nm | +0.3 kJ/mol | +0.4 kJ/mol |
| 0.9 nm | 0.32 nm | +1.0 kJ/mol | +1.8 kJ/mol |

Wniosek: przy stałej soli efekt dielektryczny w średnim polu wynosi **|ΔΔG| ≲ 2 kJ/mol**, a nawet znak zależy od nieznanej geometrii ssDNA. Dla porównania obserwujemy +14.7 kJ/mol (EG 40 %) i +9.6 kJ/mol (PEG 400 40 %).

Założenia: jednorodne ε aż do powierzchni DNA, brak zmian solwatacji (energii Borna) Na⁺, aktywność równa stężeniu, pominięte efekty końców. Model przewiduje, że nachylenie solne spadnie z 3.49 do ok. 3.2–3.3 w EG 40 % i do ok. 3.0–3.15 w PEG 400 40 %. **To można sprawdzić eksperymentalnie** (pkt 5.3).

Uboczna obserwacja: kolumna `Na conc. [mM]` w danych K maleje z 35 do 22.5 mM przy 40 wt%. Jeśli Na⁺ rzeczywiście tak spada, samo rozcieńczenie soli daje 3.49 × ln(35/22.5) = 1.55 jednostki ln K, czyli **3.8 kJ/mol**. To więcej niż szacowany wkład ε, więc trzeba jasno ustalić, czy ten spadek jest realny i jak jest uwzględniony w analizie.

**B. Aktywność wody, `q5_water_activity.py`**

Pozorne n_app = Δln K/Δln a_w, liczone z naszych K przy 40 wt%:

| kosolut (40 wt%) | Δln a_w (zakres) | ΔΔG_obs [kJ/mol] | n_app [H₂O/bp] | ΔΔG, gdyby działało 15 H₂O/bp (Nakano) |
|---|---|---|---|---|
| EG | −0.14 do −0.19 | +14.7 | **2.4–3.2** | 68–94 |
| PEG 200 | −0.051 do −0.060 | +13.6 | 7.0–8.2 | 25–29 |
| PEG 400 | −0.030 do −0.044 | +9.6 | 6.8–9.9 | 15–21 |
| PEG 1000 / 1500 | ok. −0.03 | +4.4 / +4.6 | 4–5 | 14–16 |
| PEG ≥ 3000 | ok. −0.024 | od −0.3 do +1.4 | ≈ 0–1.7 | 11–12 |

Interpretacja:
1. Dla EG pozorna wartość 2.4–3.2 H₂O/bp **zgadza się z opublikowanymi „Δn_w” dla EG** (Rozners: 1.8–4.1, Spink & Chaires: ok. 4). Samą aktywnością wody da się więc „wyjaśnić” EG. Jednak Stanley & Rau pokazują, że EG jest prawie niewykluczony z DNA, a Nordstrom, że EG akumuluje się na ssDNA. Liczba dla EG jest więc wielkością pozorną (Parsegian 2000), w której mieszają się hydratacja i wiązanie EG z zasadami. **Nie da się jej rozdzielić tymi danymi.**
2. Wartość 15 H₂O/bp otrzymana z PEG (Nakano) przeszacowuje efekt EG 5–6-krotnie. Wartości Δn_w nie są przenośne między kosolutami.
3. Dla dużych PEG samo a_w przewiduje destabilizację o 11–12 kJ/mol, a obserwujemy ok. 0. Wykluczanie objętości (i/lub inne efekty) kompensuje więc rzędu +11 kJ/mol. Tych dwóch członów nie wolno jednak dodawać jako niezależnych, bo dla polimeru wykluczonego z powierzchni DNA „woda niedostępna dla kosolutu” i objętość wykluczona opisują to samo zjawisko (Parsegian 2000).
4. Dla PEG 200 i 400 n_app wynosi 7–10/bp, ok. połowy wartości Nakano. Pasuje to do częściowej kompensacji przez crowding już dla krótkich PEG albo do różnic warunków (1 M zamiast 35 mM soli, 8-mer zamiast 13-mera).

### 5.3 Czy da się rozdzielić te wkłady istniejącymi danymi?

- **(i) dielektryczny / elektrostatyczny:** jako górne ograniczenie tak. Daje ≲ 2 kJ/mol przy stałej soli (PB). Uzupełniające argumenty literaturowe: Stanley & Rau (efekt bezpośredni, nie przez ε), Hickey & Turner (brak korelacji z ε) oraz Lambert & Draper, Nordstrom i Spink & Chaires (EG prawie nie zmienia zależności od soli). **Test krytyczny bez nowych założeń:** zmierzyć d ln K/d ln[Na⁺] w 40 wt% EG i PEG 400. Jeśli nachylenie spadnie najwyżej o ok. 10 %, efekt ε jest mały. Duża zmiana nachylenia wskazywałaby na ε albo na kompleksowanie Na⁺.
- **(ii) aktywność wody vs (iii) oddziaływania preferencyjne:** **nie da się** z pojedynczym kosolutem i z a_w zapożyczonym z literatury, bo to ta sama wielkość termodynamiczna (Parsegian 2000). Możliwa jest częściowa separacja:
  1. zmierzyć osmolalność **naszych** roztworów (VPO dla PEG, VPO lub FPD dla EG) w 20 mM PB przy 25 °C, zamiast używać tabel Nakano z 1 M NaCl;
  2. wykorzystać szereg homologiczny EG → PEG 35k jako zestaw „wielu kosolutów” w duchu Knowlesa (oddziaływania grup końcowych i wewnętrznych z ASA oraz objętość wykluczona);
  3. porównać z dupleksem o innej długości lub z innym ΔASA.
- Wartości Δn_w z literatury nadają się tylko do kontroli rzędu wielkości, a nie do odejmowania.

### 5.4 Luki

1. Brak dostępnego pomiaru ε dla EG–woda w 25 °C w wt% (Åkerlöf 1932 i Kumbharkhane 1992 są płatne, DOI zweryfikowane). Moja wartość 67.5 jest przeskalowana z serii mierzonej najpewniej w 20 °C.
2. Brak ε dla PEG ≥ 1000 przy 20–40 wt% (Kaatze 1978 jest płatny).
3. Brak a_w dla PEG/EG w buforze o niskiej soli w 25 °C (Ninni i in. 1999 mają a_w PEG/woda, ale znam tylko metadane).
4. Brak opublikowanego d ln K/d(1/ε) przy stałej soli dla krótkich dupleksów.
5. Feng 2019 badał stacking w glikoeterach, nie w EG, a w PEG tylko PEG 400.

---

## Q6. Fotofizyka Atto488/Atto647N, oddziaływania barwnik–DNA i barwnik–barwnik

### 6.1 Nasz układ: co wiemy o barwnikach

- **Metoda w obecnym SI** (`manuscript/SI/main.tex`, l. 84–169): MBA w kanale donora. Atto488 jest na jednej nici, Atto647N na drugiej, oba na tym samym końcu dupleksu. α (jasność wolnego donora) pochodzi z FCS, γ z pomiaru przy dużym nadmiarze akceptora, V₀ jest korygowane na RI. Stężenie akceptora C_B jest **nominalne** (r × C_A,eff). Z tekstu nie wynika jasno, czy α i γ wyznaczano **osobno dla każdego stężenia każdego crowdera** (wynika to pośrednio, ale warto to potwierdzić).
- **Bielec i in. 2020** **[PT, PMC]** (25 °C, 20 mM PB):
  - K = (3.5 ± 1.9)·10⁹ M⁻¹ dla par z barwnikami na tym samym końcu („3′488/647N”) wobec (1.1 ± 0.5)·10⁹ M⁻¹ dla barwników na przeciwnych końcach;
  - pary z jednym znacznikiem: (1.11 ± 0.9)·10⁹ i (0.9 ± 0.4)·10⁹ M⁻¹;
  - interpretacja autorów: π–π stacking barwników, ΔG ≈ −2.75 kJ/mol;
  - jasność Atto488 rośnie o **22 %** po hybrydyzacji, a Atto647N maleje o 5 %.
  **Krytycznie:** różnica 3.5 ± 1.9 wobec 1.1 ± 0.5 to ok. 1.2σ łącznie. Stacking nie był testowany strukturalnie (brak anizotropii, absorpcji H-dimeru, NMR). Bielec 2022 SI, sekcja 4 **[PT, lokalny]**: stosunek ok. 3× utrzymuje się do 400 mM siły jonowej, powyżej trend się odwraca.
- **Hydrofobowość i ładunek:**
  - Zanetti-Domingues 2013 **[PT]**: Atto 647N ma obliczony ładunek netto +0.61 i logD +1.96 przy pH 7.41. Wśród barwników „czerwonych” daje dużo niespecyficznych przyklejeń w SPT.
  - Hughes, Rawle, Boxer 2014 **[PT]**: Atto 647N-maleimid ma MIF = **1361** (najwyższa klasa oddziaływania z dwuwarstwą egg-PC), logD 3.82 (hyd.) / 3.26. Atto 488-SE ma MIF = **0.007**, logD −7.6 / −4.67. Ujemnie naładowane lipidy *zwiększają* asocjację Atto 647N.
  - Mamy więc donor skrajnie hydrofilowy i akceptor kationowy oraz hydrofobowy.
- **Hellenkamp i in. 2018** **[PT]**: Atto 647N to karbopironina. Barwniki cyjaninowe są bardziej wrażliwe na otoczenie. Nawet przy niskiej anizotropii „the possibility of dyes interacting with the DNA cannot be fully excluded”. Autorzy używają n = 1.40 i zalecają mierzyć Φ_D, J, n i κ² w badanym układzie, bo wartości producenta zwiększają niepewność.

### 6.2 Literatura o stackingu barwników na końcach dupleksu

- Iqbal i in. 2008 **[PT]**: Cy3 i Cy5 na końcach 5′ (linker 3-C) stackują się głównie na końcowej parze zasad „jak dodatkowa para”. FRET jest modulowany przez κ² z okresem helisy. Do 31 % emisji może pochodzić z form rozstackowanych. Uwaga: tam barwniki są na **przeciwnych** końcach, a u nas na tym samym.
- Moreira, You, Owczarzy 2015 **[ABS]**: Cy3 lub Cy5 na końcu 5′ stabilizuje dupleks średnio o **1.2 kcal/mol (ok. 5.0 kJ/mol)**, zależnie od sekwencji. Warunki solne nie są podane w abstrakcie. Dla Atto 647N analogicznych danych **nie znalazłem**.
- Peterson i in. 2019 **[ABS]**: pięć różnych znaczników na 11-merze spowalnia k_on i k_off nawet 5×.
- Sanborn i in. 2007 **[ABS]**: wydajność kwantowa Cy3 spada 2.4× po utworzeniu dupleksu; fotoizomeryzacja silnie zależy od mikrootoczenia.
- Kretschy i in. 2016 **[ABS]**: fluorescencja Cy3 i Cy5 zależy od sekwencji 5 sąsiednich zasad.
- Urnavicius i in. 2012, Ouellet i in. 2011 **[META]**: struktura i orientacja cyjanin na DNA.
- Wpływ PEG/EG na stacking barwników lub końcowej pary zasad: bezpośrednich badań **nie znalazłem**. Pośrednio: Feng 2019 (PEG 400 i glikoetery obniżają energię stackingu zasad) **[PT]**. Rozpuszczalniki organiczne ogólnie osłabiają oddziaływania hydrofobowo-aromatyczne, ale to wiedza ogólna, bez źródła z DOI.

### 6.3 Fotofizyka w crowderach

- Atto488 (rodaminowy, sztywny ksanten) i Atto647N (karbopironina) nie mają izomeryzacji cis–trans typu cyjaninowego. Nie należy więc oczekiwać zależności Φ od lepkości typu PIFE. Hellenkamp podaje tylko kontrast z cyjaninami; bezpośredniego pomiaru Φ(η) dla tych dwóch barwników **nie znalazłem** (luka).
- Buschmann, Weston, Sauer 2003 **[ABS]**: dla ATTO655 i ATTO680 (oksazyny) Φ zależy głównie od polarności, dla barwników DY od lepkości. Atto488 i Atto647N nie były badane.
- Toptygin 2003 **[ABS]**: szybkość radiacyjna zależy od współczynnika załamania n (Strickler–Berg, k_r ∝ n²). Przy n = 1.333 → ok. 1.39 (PEG 40 %) k_r rośnie o ok. 9 %, czas życia skraca się o kilka procent, a Φ Atto488 (wysokie) rośnie o ≤ 2 %.
- R₀ ∝ n^(−2/3), więc spada o ok. 3 %. Przy barwnikach na tym samym końcu E ≈ 1, więc ten efekt jest pomijalny.
- Danych dla Atto488 ani Atto647N w PEG, EG ani glicerolu z DOI **nie znalazłem**. Wartości producenta (Φ, τ) nie mają DOI, więc są NIEZWERYFIKOWANE.

### 6.4 Czy efekty barwnikowe mogą dawać część naszego ΔΔG?

1. **Modulacja oddziaływania barwnik–barwnik: realne ryzyko.** Jeśli stacking Atto488/Atto647N (ΔG ≈ −2.75 kJ/mol, przy dużej niepewności, realnie 0–5 kJ/mol) zostanie przez crowder osłabiony, do ΔΔG trafi **do ok. +2.7 kJ/mol** sztucznej destabilizacji.
   - Dla EG 40 % (+14.7 kJ/mol) to ≤ ok. 20 %.
   - Dla długich PEG, gdzie |ΔΔG| wynosi kilka kJ/mol (np. PEG 35k 20 wt%: −4.4 kJ/mol), efekt barwnikowy jest **tego samego rzędu co cały mierzony efekt**.
   - Kierunek: glikole i glikoetery osłabiają stacking (Feng 2019), więc artefakt zawyżałby destabilizację w EG i krótkich PEG. Dla długich PEG kierunek jest nieznany: z jednej strony chemiczne osłabienie stackingu, z drugiej crowding sprzyjający formom zwartym.
2. **Stacking barwnik–końcowa para zasad:** dla cyjanin ok. 5 kJ/mol (Moreira 2015). Dla Atto647N nieznany, a ten człon też może zależeć od kosolwentu. Pary z jednym znacznikiem i pary z barwnikami na przeciwnych końcach mają u Bielca to samo K, co sugeruje mały wkład, ale bez nieznakowanej referencji to nie jest dowód.
3. **α i γ:** jeśli są mierzone w każdym roztworze, zmiany Φ, n i τ znoszą się w pierwszym rzędzie. Jeśli γ/α przyjęto z buforu, każda zależna od crowdera zmiana stosunku jasności przesuwa K. Wzrost jasności Atto488 o 22 % po hybrydyzacji pokazuje, że otoczenie wpływa na Φ donora.
4. **Strata nici z akceptorem (adsorpcja Atto647N–DNA na szkle):** C_B jest nominalne. Kationowy i hydrofobowy Atto647N (MIF 1361, logD +2 do +3.8) może się adsorbować. Jeśli adsorpcja zależy od crowdera (PEG pasywuje powierzchnie), K przesuwa się systematycznie, np. pozorny wzrost K w PEG. W Bielec 2022 bufor zawierał 0.002 % Tween 20; sprawdzić, czy obecne pomiary też.
5. **Ewentualna asocjacja Atto647N z PEG:** hipoteza, brak źródła. Obniżyłaby efektywny stacking i dostępność akceptora.

**Proponowane kontrole:**
1. K w EG 40 %, PEG 400 40 % i PEG 35k 20 % dla pary z jednym znacznikiem (tylko Atto488, metoda jasności, Bielec 2020) **równolegle** z parą FRET. Stosunek K_FRET/K_single w funkcji crowdera mierzy bezpośrednio modulację stackingu barwników.
2. FCS w kanale czerwonym, żeby zweryfikować rzeczywiste C_B w każdym roztworze.
3. Czas życia i anizotropia Atto488 (ss i ds) oraz absorpcja pary z barwnikami na jednym końcu (pasmo H-dimeru) w buforze, EG i PEG.

---

## Bibliografia

| # | Cytowanie | DOI | Crossref | Dostęp / co przeczytałem |
|---|---|---|---|---|
| 1 | Nakano S., Karimata H., Ohmichi T., Kawakami J., Sugimoto N. J. Am. Chem. Soc. 126, 14330 (2004) | 10.1021/ja0463029 | tak | pełny tekst + SI (lokalnie `literature/pdf/`) |
| 2 | Nakano S., Karimata H., Kitagawa Y., Sugimoto N. J. Am. Chem. Soc. 131, 16881 (2009). *Uwaga: to praca o rybozymie typu hammerhead, nie o dupleksie DNA.* | 10.1021/ja9066628 | tak | tylko abstrakt |
| 3 | Nakano S., Yamaguchi D., Tateishi-Karimata H., Miyoshi D., Sugimoto N. Biophys. J. 102, 2808 (2012) | 10.1016/j.bpj.2012.05.019 | tak | pełny tekst (PMC3379618) |
| 4 | Spink C.H., Chaires J.B. Biochemistry 38, 496 (1999) (Crossref podaje rok wydania online 1998) | 10.1021/bi9820154 | tak | tylko abstrakt |
| 5 | Spink C.H., Garbett N., Chaires J.B. Biophys. Chem. 126, 176 (2007) | 10.1016/j.bpc.2006.07.013 | tak | tylko abstrakt |
| 6 | Rozners E., Moulder J. Nucleic Acids Res. 32, 248 (2004) | 10.1093/nar/gkh175 | tak | pełny tekst (PMC373285) |
| 7 | Parsegian V.A., Rand R.P., Rau D.C. PNAS 97, 3987 (2000) | 10.1073/pnas.97.8.3987 | tak | pełny tekst (PMC18129) |
| 8 | Stanley C., Rau D.C. Biophys. J. 91, 912 (2006) | 10.1529/biophysj.106.086579 | tak | pełny tekst (PMC1563772) |
| 9 | Nordstrom L.J., Clark C.A., Andersen B., Champlin S.M., Schwinefus J.J. Biochemistry 45, 9604 (2006) | 10.1021/bi052469i | tak | tylko abstrakt |
| 10 | Lambert D., Draper D.E. J. Mol. Biol. 370, 993 (2007) | 10.1016/j.jmb.2007.03.080 | tak | pełny tekst (PMC1995082) |
| 11 | Knowles D.B., LaCroix A.S., Deines N.F., Shkel I., Record M.T. PNAS 108, 12699 (2011) | 10.1073/pnas.1103382108 | tak | pełny tekst (lokalnie) |
| 12 | Feng B., Sosa R.P., Mårtensson A.K.F., Jiang K., … Nordén B. PNAS 116, 17169 (2019) | 10.1073/pnas.1909122116 | tak | pełny tekst (Europe PMC XML, PMC6717297) |
| 13 | Jaiswal A.K., Srivastava R., Pandey P., Bandyopadhyay P. PLoS ONE 13, e0206359 (2018) | 10.1371/journal.pone.0206359 | tak | pełny tekst (PDF `q5_q6/`) |
| 14 | Hickey D.R., Turner D.H. Biochemistry 24, 2086 (1985) | 10.1021/bi00329a042 | tak | tylko abstrakt |
| 15 | Levine L., Gordon J.A., Jencks W.P. Biochemistry 2, 168 (1963) | 10.1021/bi00901a030 | tak | tylko metadane |
| 16 | Herskovits T.T., Singer S.J., Geiduschek E.P. Arch. Biochem. Biophys. 94, 99 (1961) | 10.1016/0003-9861(61)90016-9 | tak | tylko metadane |
| 17 | Cohen J.A., Podgornik R., Hansen P.L., Parsegian V.A. J. Phys. Chem. B 113, 3709 (2009) | 10.1021/jp806893a | tak | pełny tekst (lokalnie) |
| 18 | Ninni L., Camargo M.S., Meirelles A.J.A. Thermochim. Acta 328, 169 (1999) | 10.1016/S0040-6031(98)00638-8 | tak | tylko metadane |
| 19 | Åkerlöf G. J. Am. Chem. Soc. 54, 4125 (1932) | 10.1021/ja01350a001 | tak | tylko metadane (płatny) |
| 20 | Kumbharkhane A.C., Puranik S.M., Mehrotra S.C. J. Solution Chem. 21, 201 (1992) | 10.1007/BF00647008 | tak | tylko metadane (płatny) |
| 21 | Zagrebin P.A., Buchner R., Nazmutdinov R.R., Tsirlina G.A. J. Phys. Chem. B 114, 311 (2010) | 10.1021/jp907479z | tak | abstrakt; wartości ε wzięte z [22] |
| 22 | Kaiser A., Ritter M., Nazmutdinov R., Probst M. J. Phys. Chem. B 120, 10515 (2016) | 10.1021/acs.jpcb.6b05236 | tak | pełny tekst (PMC5066130), Tabela 1 |
| 23 | Yurquina A., Manzur M.E., Brito P., Manzo R., Molina M.A.A. J. Mol. Liq. 133, 47 (2007) | 10.1016/j.molliq.2006.05.005 | tak | pełny tekst (PDF z repozytorium autorów, `q5_q6/`) |
| 24 | Panraksa P., Tipduangta P., Jantanasakulwong K., Jantrawut P. Membranes 10, 376 (2020). *Źródło wartości 56.74 w Bielec 2022 SI; wartość obliczona, nie zmierzona.* | 10.3390/membranes10120376 | tak | pełny tekst (PMC7759778) |
| 25 | Kaatze U., Göttmann O., Podbielski R., Pottel R., Terveer U. J. Phys. Chem. 82, 112 (1978) | 10.1021/j100490a025 | tak | tylko metadane |
| 26 | Shikata T., Takahashi R., Sakamoto A. J. Phys. Chem. B 110, 8941 (2006) | 10.1021/jp060356i | tak | tylko abstrakt |
| 27 | Bielec K., Bubak G., Kalwarczyk T., Hołyst R. J. Phys. Chem. B 124, 1941 (2020) | 10.1021/acs.jpcb.0c00770 | tak | pełny tekst (PMC7497653) |
| 28 | Bielec K., Kowalski A., Bubak G., Witkowska Nery E., Hołyst R. J. Phys. Chem. Lett. 13, 112 (2022) (Crossref podaje rok wydania online 2021) | 10.1021/acs.jpclett.1c03596 | tak | SI pełny tekst (lokalnie `jz1c03596_si_001.pdf`) |
| 29 | Iqbal A., Arslan S., Okumus B., Wilson T.J., Giraud G., Norman D.G., Ha T., Lilley D.M.J. PNAS 105, 11176 (2008) | 10.1073/pnas.0801707105 | tak | pełny tekst (PMC2516210) |
| 30 | Hellenkamp B. i in. Nat. Methods 15, 669 (2018) | 10.1038/s41592-018-0085-0 | tak | pełny tekst (PDF `q5_q6/`) |
| 31 | Zanetti-Domingues L.C., Tynan C.J., Rolfe D.J., Clarke D.T., Martin-Fernandez M. PLoS ONE 8, e74200 (2013) | 10.1371/journal.pone.0074200 | tak | pełny tekst (PDF `q5_q6/`) |
| 32 | Hughes L.D., Rawle R.J., Boxer S.G. PLoS ONE 9, e87649 (2014) | 10.1371/journal.pone.0087649 | tak | pełny tekst (PDF `q5_q6/`) |
| 33 | Moreira B.G., You Y., Owczarzy R. Biophys. Chem. 198, 36 (2015) | 10.1016/j.bpc.2015.01.001 | tak | tylko abstrakt |
| 34 | Peterson E.M., Reece E.J., Li W., Harris J.M. J. Phys. Chem. B 123, 10746 (2019) | 10.1021/acs.jpcb.9b09736 | tak | tylko abstrakt |
| 35 | Sanborn M.E., Connolly B.K., Gurunathan K., Levitus M. J. Phys. Chem. B 111, 11064 (2007) | 10.1021/jp072912u | tak | tylko abstrakt |
| 36 | Kretschy N., Sack M., Somoza M.M. Bioconjugate Chem. 27, 840 (2016) | 10.1021/acs.bioconjchem.6b00053 | tak | tylko abstrakt |
| 37 | Urnavicius L., McPhee S.A., Lilley D.M.J., Norman D.G. Biophys. J. 102, 561 (2012) | 10.1016/j.bpj.2012.01.001 | tak | tylko metadane |
| 38 | Ouellet J., Schorr S., Iqbal A., Wilson T.J., Lilley D.M.J. Biophys. J. 101, 1148 (2011) | 10.1016/j.bpj.2011.07.007 | tak | tylko metadane |
| 39 | Buschmann V., Weston K.D., Sauer M. Bioconjugate Chem. 14, 195 (2003) (Crossref podaje rok wydania online 2002) | 10.1021/bc025600x | tak | tylko abstrakt |
| 40 | Toptygin D. J. Fluoresc. 13, 201 (2003) | 10.1023/A:1025033731377 | tak | tylko metadane |
| 41 | Stennett E.M.S., Ciuba M.A., Levitus M. Chem. Soc. Rev. 43, 1057 (2014) | 10.1039/c3cs60211g | tak | tylko metadane |
| — | „Lee et al.” (osmotic stress), wymieniony w zadaniu: nie udało się jednoznacznie zidentyfikować pracy | — | NIEZWERYFIKOWANE | — |
| — | Dane producenta ATTO-TEC (Φ, τ Atto488 i Atto647N) | brak DOI | NIEZWERYFIKOWANE | nieużyte liczbowo |
