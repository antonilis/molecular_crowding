# Q1 – Bezpośrednie dowody oddziaływań EG / glikoli / PEG z zasadami, nukleozydami i ssDNA

Data przeglądu: 2026-10-09. Wszystkie DOI sprawdzone w Crossref API (`api.crossref.org/works/<DOI>`);
status przy każdej pozycji w Bibliografii. „Pełny tekst” oznacza, że przeczytałem cały artykuł (PMC
XML przez NCBI efetch, PDF wydawcy/repozytorium albo lokalny PDF w `literature/`). „Tylko abstrakt”
oznacza, że nie miałem dostępu do treści, więc **liczb z tych prac nie podaję**, chyba że wyraźnie
oznaczam je jako pochodzące z abstraktu.
Pobrane PDF-y OA: `literature/review_2026-10/Jaiswal2018_PLoSOne_MD_ethylene_glycol_near_DNA.pdf`,
`literature/review_2026-10/Trajkovski2018_NAR_EG_PEG8000_Gquadruplex_NMR.pdf`. Pozostałe pełne
teksty czytałem z PMC (XML), bo PMC/Europe PMC blokują pobieranie PDF-ów skryptem.

Nasz punkt odniesienia: m-wartość EG = +0.31 kcal mol⁻¹ molal⁻¹ (znak dla asocjacji), 13 bp, 25 °C,
Na⁺ ≈ 35 mM. Knowles 2011 mierzył 12 bp dupleks w **0.02 m K₂HPO₄ + 0.1 m KCl + 1 mM K₂EDTA,
pH 7.3**, analizę m-wartości robił przy **40 °C** (SI Knowles 2011, „Materials and Methods”; pełny
tekst, lokalny PDF). Porównanie naszej wartości z jego −0.32…−0.34 kcal mol⁻¹ (molal reszt)⁻¹
(topnienie) zestawia więc dane z innej temperatury, innego kationu i ok. 4× wyższej siły jonowej.

---

## 0. Skrót odpowiedzi

1. **Nie ma żadnej bezpośredniej obserwacji strukturalnej kontaktu EG lub PEG z zasadą w ssDNA ani
   w dupleksie** (NOE, STD-NMR, krystalografia). Jedyne bezpośrednie dane NMR dotyczą
   G-kwadrupleksów. W tych pracach **brak międzycząsteczkowych NOE** zarówno dla EG (20 wt%),
   jak i dla PEG 8000 (Trajkovski 2018). Są za to perturbacje przesunięć chemicznych zasad
   eksponowanych (końcowy T20 i zewnętrzna G-tetrada: do 0.52 ppm przy PEG 8000, ≤ 0.03 ppm dla
   rdzenia przy EG). Kontakt glikol–zasada (CH–π, lone pair–π) pokazano wyłącznie dla TEG
   przyłączonego **kowalencyjnie** do tymidyny (Tateishi-Karimata 2017).
2. **Klasyczne dane o rozpuszczalności zasad w wodnych roztworach glikoli istnieją, ale są dla nas
   niedostępne.** Chodzi o Herskovits & Harrington 1972, *Biochemistry* 11, 4800 („Solubility in
   aqueous alcohol and glycol solutions”). Praca jest zamknięta, nie ma abstraktu w PubMed ani w
   Europe PMC, a nikt z przeczytanych autorów nie cytuje z niej liczb. **Tej pracy trzeba szukać przez
   bibliotekę IChF**, bo daje bezpośredni, niezależny od DNA test obrazu Knowlesa. Jedyna liczba, jaką
   mam z pełnego tekstu, pochodzi od Ts'o 1963 i dotyczy poliolu (adonitolu), nie EG.
3. **Record i współpracownicy nigdy nie opublikowali α-wartości glikol/PEG–zasada.** W 2015 r.
   piszą, że nukleozasady „are in progress” i że w analizie PEG400 je pominięto (Knowles 2015). W
   2018 r. publikują α-wartości dla zasad, ale tylko z alkilomocznikami (Cheng 2018). Pracy o
   zależności χ od rozmiaru DNA („experiments in progress” w Knowles 2011) **nie znalazłem**. Shkel
   2015 pisze wprost, że zależność χ od N₃ i od rozmiaru biopolimeru „is unknown”. **Uwaga:** w tej
   samej pracy ta sama grupa ponownie przeanalizowała dane DNA 12-mer i dostała **χ∞ = 0.6 ± 0.06**,
   a nie 0.45 ± 0.10 jak w 2011 r. Jeśli cytujemy χ∞ ≈ 0.45, musimy to uwzględnić.
4. **Pośrednie dane termodynamiczne są spójne z obrazem Knowlesa.**
   * PEG (jednostki wewnętrzne i końcowe) silnie i korzystnie oddziałuje z aromatycznym C w
     związkach modelowych białek: α_I = −0.765, α_2E = −0.364 cal mol⁻¹ molal⁻¹ Å⁻² (Knowles 2015).
     Per Å² to tyle samo co mocznik z pierścieniem zasady (Guinn 2013).
   * Wszystkie osmolity destabilizują drugorzędową strukturę RNA, „probably because they favor
     solubilization of base surfaces” (Lambert & Draper 2007; tylko abstrakt).
   * Sugimoto i współpracownicy, którzy interpretują dane przez aktywność wody, sami dostają
     **różne „liczby wody” dla różnych kosolutów** (15 H₂O/bp dla PEG, 4.7 dla glicerolu). Przy tej
     samej a_w nie leżą one na jednej prostej (Nakano 2004, Ghosh 2020). Sam model aktywności wody
     tego nie tłumaczy; potrzebne są oddziaływania specyficzne dla kosolutu.
5. **Co obrazowi Knowlesa przeczy lub go komplikuje:**
   * Buscaglia 2013 i Trajkovski 2018 traktują EG jako czysty osmolit, a efekt EG na G4 przypisują
     dehydratacji. Trajkovski dodaje stałą dielektryczną.
   * Nakano 2014: zależność stabilności dupleksu RNA od [NaCl] koreluje ze stałą dielektryczną
     kosolutu, a nie z a_w.
   * MD (Jaiswal 2018, GAFF + ff99bsc0) pokazuje, że EG gromadzi się przy dupleksie (lokalnie
     ~2× nadmiar względem objętości, głównie przy fosforanach) i wypiera wodę z rowków w małym
     stopniu. Akumulacja przy dupleksie zmniejszałaby, a nie zwiększała, różnicę ss–ds, którą
     zakłada Knowles.
   * Odwrotnie Knowles 2015 (VPO): grupy PEG/EG są silnie **wykluczane** z dianionu fosforanowego
     (β_I = +234 cal mol⁻¹ molal⁻¹ na jon).
   * Ts'o 1963: adonitol (poliol) „nieefektywny” zarówno jako czynnik rozpuszczający adeninę, jak i
     jako czynnik obniżający Tm DNA.
6. **Poza Q1, ale ważne dla naszego artykułu.** Te same pomiary VPO (Knowles 2015, Tabela 2)
   dają **niekorzystne** (dodatnie) β dla Na⁺ z tetraEG (+18.5 ± 5.2) i z grupą wewnętrzną PEG
   (+4.66 ± 2 cal mol⁻¹ molal⁻¹ jon⁻¹). Na poziomie współczynników preferencyjnych oddziaływań
   Record nie widzi więc korzystnego wiązania Na⁺ przez PEG. Rozkład μ₂₃ soli na jony opiera się na
   założeniach, ale recenzent może to podnieść przeciw naszej tezie o kompleksowaniu PEG–Na⁺ (DOSY).

---

## (a) Rozpuszczalność i energie przeniesienia zasad i nukleozydów: woda → EG / glicerol / PEG / alkohole

### Co wiadomo (liczby + warunki)

* **Ts'o, Melvin, Olson 1963** (pełny tekst; PDF z kursu Georgia Tech, nie OA wydawcy, więc nie
  kopiowałem go do `review_2026-10`). Tabela VII(A), 25.5 °C, woda bez buforu, oznaczenia
  radioznacznikowe ¹⁴C:
  * rozpuszczalność adeniny: S₀ = 8.25×10⁻³ M;
  * 0.60 M **adonitol** (rybitol, poliol C5): S = 9.84×10⁻³ M, log(S/S₀) = 0.076;
  * 0.60 M mocznik: 8.88×10⁻³ M, log = 0.033;
  * 0.20 M cykloheksanol: 9.47×10⁻³ M, log = 0.061.

  **Moje przeliczenie** (ΔG_tr = −2.303RT·log(S/S₀), T = 298.65 K, założenie γ = 1 w fazie
  nasyconej, skala molowa):
  * adonitol: −0.10 kcal/mol przy 0.6 M, czyli **≈ −0.17 kcal mol⁻¹ M⁻¹**;
  * mocznik: ≈ −0.075 kcal mol⁻¹ M⁻¹;
  * cykloheksanol: ≈ −0.42 kcal mol⁻¹ M⁻¹.

  Autorzy oceniają adonitol i mocznik jako „ineffective in raising the solubility of adenine”. W
  Dyskusji piszą też, że adonitol (i rybozyd metylu) mają „negligible” wpływ na Tm poli(A) i DNA
  (pomiar ORD). EG nie badano.
* **Herskovits & Harrington 1972**, *Biochemistry* 11, 4800–4811: rozpuszczalność zasad w wodnych
  alkoholach i glikolach. To kluczowa praca dla Q1. **Brak dostępu** (ACS, brak abstraktu w
  PubMed/EPMC, 403 na stronie wydawcy). **Liczb nie podaję.**
* **Herskovits & Bowen 1974**, *Biochemistry* 13, 5474: to samo dla mocznika i amidów. Brak
  dostępu. Guinn 2013 (pełny tekst) cytuje z niej tylko pochodne α dla mocznika: adenina
  −8.9×10⁻⁴ m⁻¹Å⁻², cytozyna −7.6×10⁻⁴ m⁻¹Å⁻² (μ₂₃/RT na Å², bez korekty na samoasocjację).
* **Scruggs, Achter, Ross 1972**, *Biopolymers* 11, 1961: rozpuszczalność zasad w wodzie i
  rozpuszczalnikach organicznych w funkcji T. Brak abstraktu i brak dostępu. Wg opisu z wyszukiwarki
  (nie z tekstu) wniosek brzmi „interakcja zasada–woda nie jest hydrofobowa”. **NIE czytałem; nie
  cytować bez lektury.**
* **Levine, Gordon, Jencks 1963**, *Biochemistry* 2, 168: zależność skuteczności denaturantów DNA
  od struktury. Brak abstraktu i dostępu.
* **Nowoczesne dane:**
  * jedyna praca z glikolem to **Xu, Zheng, Farajtabar, Zhao 2018**, *J. Chem. Thermodyn.* 125, 225:
    adenina w mieszaninach woda + **glikol propylenowy** (oraz DMF, NMP, DMSO), z analizą
    preferencyjnej solwatacji. Brak dostępu do tekstu i abstraktu przez API. Zakres stężeń
    (prawdopodobnie do czystego współrozpuszczalnika) jest daleki od naszego 0–40 wt%, liczb nie
    znam;
  * Arakawa i in. 2025 (przegląd, pełny tekst) odnotowuje „monotone increase in the solubility of
    adenine” w tych mieszaninach i rozpuszczalność uracylu w etanolu < ~50 %;
  * **danych dla zasad w wodnym EG ani w PEG 200–35k w Crossref/EPMC nie znalazłem.**
* **Analogie z aminokwasami (białka):**
  * Hirano, Shiraki, Arakawa 2012 (tylko abstrakt): PEG4000 „significantly decreased the free energy
    of the aromatic side chains”, nieznacznie podniósł energię swobodną łańcuchów alifatycznych i
    obniżył rozpuszczalność glicyny („PEG behaves like weak organic solvent”);
  * Knowles 2015 (pełny tekst): PEG400 zwiększa rozpuszczalność 28 aromatów nawet o 10⁴ przy
    75 % PEG400 (dane Rytting i in. 2005). Dla 12 steroidów obserwowane μ₂₃ są o ok.
    −1.5 kcal mol⁻¹ molal⁻¹ korzystniejsze, niż przewiduje addytywność. Autorzy przypisują to
    „concerted” oddziaływaniu wielu reszt PEG z płaskim, skondensowanym układem pierścieni.
    **Nukleozasady (heteroaromaty) wyłączono z tej analizy.**

### Wielkości użyteczne dla naszych danych

* **Test rzędu wielkości (moje oszacowanie).** Nasza m-wartość EG to 0.31 kcal mol⁻¹ molal⁻¹ dla
  13 bp, czyli 26 zasad odsłanianych przy topnieniu. Jeżeli w ssDNA odsłonięty jest ułamek f
  powierzchni zasad (f ≈ 0.3–0.5 przy 50–90 % stackingu ss; por. Guinn 2013), to potrzebna energia
  przeniesienia **na zasadę** do 1 molal EG wynosi tylko **ΔG_tr ≈ −(0.31/26f) ≈ −0.024…−0.040 kcal
  mol⁻¹ molal⁻¹** (−0.012 przy f = 1).
  * Adonitol daje dla adeniny ≈ −0.17 kcal mol⁻¹ M⁻¹, czyli kilkakrotnie więcej, niż trzeba.
  * Wymagany efekt jest więc **mały i realistyczny**: obraz Knowlesa nie wymaga silnego wiązania,
    tylko ~1 % zmiany rozpuszczalności zasady na molal EG.
  * Jednocześnie oznacza to, że **pomiar rozpuszczalności z dokładnością ±1–2 % może go nie
    rozstrzygnąć**. To argument, żeby zdobyć Herskovits & Harrington 1972, który mierzył do wysokich
    stężeń glikolu.
* Gdy uzyskamy tabelę ΔG_tr(EG) dla A, T, G, C z Herskovitsa:
  1. przeliczyć na skalę molalną, μ₂₃ ≈ ∂ΔG_tr/∂m₃;
  2. podzielić przez ASA zasady, żeby dostać α per Å²;
  3. przemnożyć przez ΔASA zasad dla naszego 13-meru (modele ss: stacked / half-stacked /
     unstacked, jak u Guinna 2013);
  4. **porównać z m = 0.31 bez żadnego dopasowania** do danych DNA.

  To byłby niezależny test, którego Knowles nie wykonał.

### Luki

* Brak nowoczesnych pomiarów (VPO, rozpuszczalność, podział hexanol/woda) μ₂₃ dla EG, di/triEG,
  PEG z A/T/G/C i nukleozydami.
* Brak danych temperaturowych: entalpii i entropii przeniesienia zasad do EG.
* Brak danych dla PEG > 400 z jakąkolwiek zasadą.

---

## (b) NMR, fluorescencja, kalorymetria – „bezpośrednie” obserwacje

### Co wiadomo

* **Trajkovski et al. 2018, NAR 46, 4301** (pełny tekst; PDF w `review_2026-10`).
  * Układ: G-kwadrupleks M2 (równoległy, 3 tetrady; nawisy T1–A2 i T20). Warunki NMR: 25 °C,
    0.2 mM nici, 100 mM KCl, 20 mM fosforan K, pH 7.0, 20 wt% EG albo 10/20 wt% PEG 8000.
  * **EG:**
    * Δδ protonów aromatycznych i metylowych T1–G19 mieści się w −0.01…+0.03 ppm („very close to
      the limits of the experimental error”);
    * wyjątek to T20: CH₃ +0.08 ppm, H6 +0.11 ppm;
    * NOESY (mixing 80–300 ms) **„did not reveal any long-lived interactions” EG–DNA**;
    * interpretacja autorów: dehydratacja (EG jako osmolit) i ewentualnie niższa ε.
  * **PEG 8000:**
    * T20 H6 przesuwa się o +0.43 ppm (10 wt%) i +0.52 ppm (20 wt%);
    * H8 zewnętrznej tetrady G5/G10/G14/G19: +0.13/0.07/0.07/0.13 ppm (20 wt%);
    * rdzeń: −0.02…0.04 ppm;
    * **brak NOE PEG–DNA** (< 5 Å). Autorzy tłumaczą to szybką wymianą przy słabym wiązaniu
      (K_B rzędu M do sub-M wg wcześniejszych prac) i dlatego nie wykluczają kontaktu.
  * Termodynamika (UV, 20 mM fosforan K, pH 7, ΔG°₂₅, kcal mol⁻¹):

    | warunki | ΔG°₂₅ (kcal mol⁻¹) |
    |---|---|
    | bez kosolutu | −5.33 ± 0.04 |
    | 20 wt% EG | −8.76 ± 0.10 |
    | 20 wt% PEG 8000 | −12.7 ± 0.40 |

  * **Dla nas:** to jedyne systematyczne NMR z EG i dużym PEG. Pokazuje, że kontakty glikol–DNA są
    za słabe lub za krótkotrwałe, by dały NOE. Najwyraźniejsze perturbacje dotyczą zasad
    **odsłoniętych lub końcowych**. Jakościowo pasuje to do obrazu „PEG lubi eksponowane
    płaszczyzny zasad”, ale ilościowo niczego nie dowodzi.
* **Buscaglia et al. 2013, NAR 41, 7934** (pełny tekst, PMC).
  * Układ: hTel22 G4, 25 mM KCl, 10 mM fosforan TBA, 1 mM EDTA, pH 7.0, 20 °C. Miareczkowania CD
    EG → PEG 10 000 (0–42.5 % v/v).
  * PEG ≥ 400 przestawia konformację hybrydową w śmigłową przy a_w ≈ 0.98, czyli **nie** przez
    osmozę. Śmigłowa jest hydrodynamicznie większa, więc to też **nie** crowding.
  * Model selekcji konformacyjnej (MWC):
    * PEG 600: K_B = 0.41 ± 0.01 M przy n = 4 (n niezależnie wyznaczone tylko jako ≥ 2), czyli
      ΔG ≈ −0.5 kcal mol⁻¹ na cząsteczkę;
    * Hill: h = 4.2 ± 0.5, K_d = 0.65 ± 0.02.
  * **MD 300 ns:** PEG 200 i 600 przechodzą z rowków na **płaszczyznę tetrady G** i zostają tam
    > 300 ns. W formie hybrydowej uciekają do roztworu.
  * Autorzy przyznają, że nie mają „direct” dowodu wiązania.
  * EG i acetonitryl traktują jako czyste czynniki osmotyczne: konwersja uwalnia ok. 17 H₂O na
    kwadrupleks.
* **Tateishi-Karimata et al. 2017, NAR 45, 7021** (pełny tekst, PMC).
  * TEG/OEG/DEG/HEG przyłączone kowalencyjnie do C5 dT w pętlach.
  * NMR: przesunięcia iminowe sąsiednich G. MD/FMO: TEG oddziałuje z zasadami tetrady i pętli przez
    **CH–π i lone pair–π**. Autorzy: „although it was previously assumed that oligoethylene glycols
    do not directly interact with DNAs”.
  * **Spinka D1** (100 mM KCl, 10 mM K₂HPO₄, pH 7.0, 1 mM K₂EDTA; ΔG°₂₅):

    | wariant | ΔG°₂₅ (kcal mol⁻¹) |
    |---|---|
    | niemodyfikowana | −6.6 |
    | X4 w pozycji 7 | −5.1 |
    | X4 w pozycji 10 | −5.9 |

    TEG **nie stabilizuje** dupleksu, a G4 owszem.
  * Ograniczenie: lokalne stężenie przyłączonego TEG jest ogromne, więc to nie jest pomiar
    oddziaływania wolnego EG/PEG.
* **Jia et al. 2017, Angew. Chem.** 56, 1239 (pełny tekst, PMC, rękopis autorski).
  * 10-mer w PBS, koniugaty PEG–DNA. Liniowy PEG 40 kDa: K_eq rośnie ~14–16×. Szczotka PEG daje
    tylko ~2× przy podobnej lokalnej gęstości PEG.
  * Autorzy rozdzielają to na „local excluded volume” (stabilizuje) i „chemical interaction”
    (destabilizuje), silniejszy dla rozciągniętych łańcuchów szczotki.
  * To interpretacja, nie obserwacja. Ciekawe dla χ: **wydłużone łańcuchy = więcej dostępnych
    segmentów**.
* **Fluorescencja (2-AP itp.) i kalorymetria z EG/PEG wobec zasad: nic nie znalazłem.**
  * Spink & Chaires 1999 (tylko abstrakt): DSC/UV, EG, glicerol, acetamid, sukroza, PEG 400–10 000.
    Interpretacja przez „waters released”: 4 H₂O/bp dla dupleksu, plus wykluczanie objętości przez
    PEG. Wiązanie Na⁺ maleje „modestly” ze spadkiem a_w w EG.
  * Stanley & Rau 2006 (tylko abstrakt): stres osmotyczny i SAXS na upakowanym DNA (dupleks).
    12 niepolarnych alkoholi jest wykluczanych przez odpychanie hydratacyjne od naładowanej
    powierzchni, a **„polyols are included at very close spacings”**, z inną zależnością od odległości.
    To jedyny eksperyment, który bezpośrednio mierzy położenie poliolu względem DNA, ale dotyczy
    dupleksu, nie ss, i nie rozdziela zasad od fosforanów.
  * Brewood, Aliwarga, Schurr 2008 (tylko abstrakt): EG 0–40 w/v % indukuje kooperatywne przejście
    strukturalne w dupleksie p30δ. Przy dopasowaniu nie-kooperatywnym ΔΓ wynosi −30 na bp, ale
    inne wyniki wskazują ΔΓ w zakresie −1…0 (z abstraktu, 0.1 M NaCl). Stan 2 ma „less water and/or
    more EG in its neighborhood”. **EG zmienia więc sam stan referencyjny dupleksu**, co komplikuje
    proste ss↔ds.

### Wielkości użyteczne dla naszych danych

* Rząd siły wiązania PEG z eksponowaną płaszczyzną zasad/tetrady: K_B ≈ 0.4 M dla PEG 600, czyli
  ≈ −0.5 kcal mol⁻¹ na cząsteczkę (Buscaglia; z modelu, nie z bezpośredniego pomiaru). Można to
  zestawić z naszym α na resztę (Knowles: ≈ −0.32 kcal mol⁻¹ (molal reszt)⁻¹ dla 12 bp), ale tylko
  jako rząd wielkości.
* Brak NOE nakłada górne ograniczenie: kontakty są krótkotrwałe. Jest to zgodne z
  „preferencyjnym oddziaływaniem” w sensie Kirkwooda–Buffa (lokalny nadmiar), a nie z wiązaniem
  stechiometrycznym.

### Luki

* Brak **jakichkolwiek** danych NMR (Δδ, NOE, STD, PRE) dla ssDNA lub dupleksu B z wolnym EG lub
  PEG.
* Brak wpływu EG/PEG na fluorescencję 2-AP w ss vs ds.
* Brak ITC/DSC rozdzielającego entalpię oddziaływania glikol–zasada.
* **Proponowany eksperyment** (naturalne uzupełnienie naszej pracy): ¹H/¹³C HSQC lub STD-NMR
  krótkiej ssDNA (np. dT₁₃ vs dA₁₃) albo nukleozydów w 0–40 wt% EG-d₄/PEG, z kontrolą lepkości i ε.

---

## (c) Symulacje MD z EG/PEG (preferencyjne wiązanie, Kirkwood–Buff)

### Co wiadomo

* **Jaiswal, Srivastava, Pandey, Bandyopadhyay 2018, PLoS ONE 13, e0206359** (pełny tekst; PDF w
  `review_2026-10`).
  * Układ: 23 bp dupleks B; AMBER14, DNA ff12SB (ff99bsc0), EG GAFF + ładunki RESP, TIP3P, 0.1 M
    NaCl (jony Joung–Cheatham), 300 K, NPT; 0/10/20/30 % v/v EG.
  * Pierwsza powłoka (3.5 Å), Tabela 4:

    | EG (v/v) | H₂O w powłoce | EG w powłoce |
    |---|---|---|
    | 0 % | 743 | 0 |
    | 10 % | 579 | 81 |
    | 20 % | 470 | 134 |
    | 30 % | 396 | 171 |

    Jedna cząsteczka EG wypiera 2.0 H₂O.
  * Wiązania H DNA–EG: 0/93/154/194; DNA–woda: 553/425/352/303.
  * **Moje przeliczenie** z ich Tabel 2 i 4: stosunek EG/woda w powłoce do stosunku „w objętości”
    wynosi **2.2 / 2.0 / 1.8** (10/20/30 %). EG jest więc lokalnie wzbogacony przy dupleksie.
    „Objętość” liczą autorzy jako średnią po całym pudle, więc to oszacowanie zaniżone, a nie G_ij.
  * Mapy gęstości: woda jest wypierana głównie przy **fosforanach** (OP1/OP2/P). Gęstość wody w
    rowkach „not much affected”.
  * Gęstość Na⁺ przy DNA rośnie z EG.
  * Nie liczą współczynników Γ ani całek KB. Nie symulują ssDNA.
* **Buscaglia 2013** (pełny tekst): MD 300 ns, PEG 200/600 osiada na płaszczyźnie tetrady G
  (opisane w (b)).
* **Tateishi-Karimata 2017** (pełny tekst): MD 35 ns, kowalencyjny TEG, CH–π z zasadami pętli.
* **Mathur & Singh 2025, ACS Omega** 10, 41425 (pełny tekst, PMC).
  * Układ: Dickerson-podobny 12-mer CGCAAATTTGCG (PDB 264D), 10/20/50 cząsteczek PEG-200, 500 ns,
    pojedyncze trajektorie.
  * PEG-200 „preferentially accumulate near the termini”, nie wiąże się w rowkach ani przy
    fosforanach.
  * Autorzy interpretują to jako efekt pośredni (aktywność wody).
  * **Niska jakość statystyczna** (kilkadziesiąt cząsteczek, brak KB, brak replik). Wynik
    jakościowo zgodny z lokalizacją PEG na **odsłoniętych płaszczyznach zasad końcowych par**.
* **Zhang, Li, Zhang 2018, Chin. Phys. B** 27, 113102 (tylko opis z wyszukiwarki; strona wydawcy
  niedostępna, ECONNREFUSED). EG tworzy „pierścienie” wokół fosforanów, woda siedzi w rowkach.
  **Liczb nie podaję.**
* **Diaz, Jothiraman, Ramakrishnan 2022, J. Mol. Graph. Model.** (tylko abstrakt): glicerol wokół
  dupleksu. Energia oddziaływania na cząsteczkę glicerolu jest silniejsza niż na cząsteczkę wody,
  zależna od sekwencji.
* **Norberg & Nilsson 1998, Biophys. J.** (tylko abstrakt): MD dinukleozydomonofosforanów. Stacking
  jest najsilniejszy w wodzie, słabszy w metanolu i DMSO, najsłabszy w CHCl₃. Odnosi się do
  **stackingu w stanie ss**: kosolut obniżający ε osłabia stacking ss, co zwiększa ΔASA topnienia
  i może naśladować „oddziaływanie z zasadami”.
* **Pincus, Hyeon, Thirumalai 2008, JACS** 130, 7364 (tylko abstrakt): TMAO w MD wiąże zasadę
  jednym wiązaniem H. Model gruboziarnisty pokazuje, że kosolut, który wiąże tylko jedną zasadę
  naraz, daje ΔTm ≈ 0.
* **Kirkwood–Buff dla glikol–zasada lub glikol–ssDNA: brak publikacji.** Najbliższe:
  * KB dla mocznik–ssDNA (Biophys. J. 2018; tylko wzmianka w wynikach wyszukiwania, nie
    weryfikowałem, **NIEZWERYFIKOWANE**, nie cytować);
  * Priyakumar i in. 2019, *Sci. Rep.* (mocznik–nukleozasada). Tylko w wynikach EPMC, nie czytałem.

### Wielkości użyteczne dla naszych danych

* Z Jaiswala: lokalny nadmiar EG przy dupleksie ≈ 2× (GAFF). Gdyby podobny lub większy nadmiar był
  przy ssDNA, różnica Γ(ss) − Γ(ds) decydowałaby o znaku m-wartości. **Ta różnica nie była liczona.**
  Jest to wykonalne (2 × ~200 ns, 13-mer ss i ds w 10–30 wt% EG; całki KB). Bezpośrednio
  przetestowałoby Knowlesa na naszym układzie.

### Luki

* Brak symulacji ssDNA w EG/PEG.
* Brak współczynników preferencyjnego wiązania (Γ₂₃, G_DNA–EG − G_DNA–W) dla ss i ds.
* Brak walidacji pól siłowych EG/PEG na danych rozpuszczalności zasad (Herskovits). Wyniki MD
  zależą od pola siłowego; GAFF dla glikoli przeszacowuje oddziaływania z polarnymi grupami
  (ogólnie znany problem, nie weryfikowałem dla tej pracy).

---

## (d) Prace grupy Recorda po 2011 r.

### Co wiadomo

* **Knowles et al. 2015, Biochemistry 54, 3528** (pełny tekst, PMC).
  * VPO i rozpuszczalność, tetraEG i glicerol z > 30 związkami modelowymi białek, 25 °C (VPO).
  * **Tabela 2**, α (cal mol⁻¹ molal⁻¹ Å⁻²):

    | grupa | tetraEG | glicerol | 2E (końce PEG) | I (–CH₂OCH₂–) |
    |---|---|---|---|---|
    | C alifatyczny | −0.349 | +0.055 | −0.003 | −0.115 |
    | **C aromatyczny** | **−2.66** | −0.431 | **−0.364** | **−0.765** |
    | O hydroksylowy | +0.843 | +0.031 | +0.026 | +0.272 |
    | O amidowy | +2.94 | +0.826 | +0.698 | +0.747 |
    | O karboksylanowy | +3.59 | +0.467 | +0.395 | +1.07 |
    | N amidowy | −1.67 | −0.491 | −0.415 | −0.418 |
    | N kationowy | −0.805 | −0.245 | −0.207 | −0.199 |

  * β (cal mol⁻¹ molal⁻¹ jon⁻¹):

    | jon | tetraEG | glicerol | 2E | I |
    |---|---|---|---|---|
    | HPO₄²⁻ | +780 | +93.9 | +79.3 | **+234** |
    | Cl⁻ | +132 | +22.8 | +19.3 | +37.6 |
    | Na⁺ | +18.5 | +5.34 | +4.51 | **+4.66** |
    | K⁺ | +6.19 | −4.11 | −3.47 | +3.22 |

  * m-wartości dla rozwijania lacDBD przy 40 °C (cal mol⁻¹ molal⁻¹):

    | kosolut | m |
    |---|---|
    | EG | +56.9 ± 11 |
    | diEG | +67.0 ± 34 |
    | triEG | −21 ± 29 |
    | tetraEG | −111 ± 41 |
    | PEG300 | −253 ± 64 |
    | glicerol | +203 ± 18 |

  * **Nukleozasady pominięto**: „Nucleobases and other heterocyclic aromatics … are omitted”.
    Kolejny etap zapowiedziano: „Studies of the interactions of alcohols, polyols and small PEGs
    with aromatic hydrocarbons and nucleobases are in progress”.
  * **EG nie da się przewidzieć z α końcówek PEG**: m(EG) dla lacDBD jest 2× mniejsze niż wkład
    2E. Dla nas oznacza to, że **α_EG ≠ α_2E**, więc EG nie jest „dwoma końcami PEG”, a
    założenie Knowlesa 2011, że końce i EG oddziałują z DNA tak samo, nie ma potwierdzenia na
    białkach.
* **Shkel, Knowles, Record 2015, Biopolymers 103, 517** (pełny tekst, PMC). Reanaliza danych DNA
  12-mer z Knowles 2011 (40 °C).
  * Tabela 3, kcal mol⁻¹ (molal reszt PEG)⁻¹:

    | wielkość | dupleks 12 bp | spinka 4 bp |
    |---|---|---|
    | obserwowane oddziaływanie chemiczne (EG–triEG) | −0.322 ± 0.02 | −0.160 ± 0.02 |
    | przewidywany efekt EV (Hermans), duże PEG | +0.34 ± 0.03 | +0.058 ± 0.006 |
    | redukcja przez osłanianie | 0.129 ± 0.02 | 0.064 ± 0.01 |
    | obserwowane, PEG ≥ 1450/2000 | +0.168 ± 0.01 | −0.033 ± 0.003 |
    | przewidywane | 0.164 ± 0.037 | −0.0346 ± 0.013 |

  * **χ∞ = 0.6 ± 0.06** („40 % of PEG residues are inaccessible to the 12-mer DNA oligomers”).
    Dla białek χ∞ = 0.74.
  * Cytat: „This shielded fraction (1–χ) has not yet been investigated by theory or simulations,
    and its dependence on PEG N₃ and protein size is unknown.”
* **Guinn et al. 2013, JACS 135, 5828** (pełny tekst, PMC). To szablon metodologiczny.
  * Rozpuszczalność zasad jest za mała na VPO, więc autorzy opracowali **test podziału
    heksanol/woda** (K_DWH) dla zasad i nukleozydów, a VPO zastosowali dla Na₂NMP. Na tej podstawie
    dopasowali α mocznika dla 6 typów powierzchni (×10⁴ m⁻¹ Å⁻²):

    | powierzchnia | α |
    |---|---|
    | pierścień aromatyczny (C, N) | −10.9 ± 0.7 |
    | metyl pierścienia | −12.0 ± 1.0 |
    | O karbonylowy | −6.2 ± 1.9 |
    | N aminowy | −2.8 ± 1.9 |
    | O fosforanowy | −5.8 ± 1.2 |
    | cukier | −6.6 ± 0.7 |

  * m-wartości mocznika dla 12-merów DNA/RNA odtwarzają się tylko przy założeniu **70–90 %
    stackingu ss (DNA)** i 60–70 % (RNA). Przy modelu „unstacked” są przeszacowane.
* **Cheng et al. 2018, Biochemistry 57, 2227** (pełny tekst, PMC): α dla alkilomoczników z 5 typami
  atomów zasad (sp²O, sp³N, sp²N, sp³C, sp²C), metodą rozpuszczalności. **Bez glikoli.**
* **Record, Guinn, Pegram, Capp 2013, Faraday Discuss.** 160, 9 (pełny tekst, PMC). Przegląd;
  wylicza „glycols, polyols, sugars” jako solutes w trakcie charakteryzacji.
* **Lista prac Recorda w EPMC 2016–2026** (26 pozycji) **nie zawiera** żadnej pracy o α
  glikol/PEG/glicerol–nukleozasada ani o χ(rozmiar DNA). Najbliższa to Zytkiewicz 2023 (amid–aromat).
  Wniosek: **zapowiedzi z 2011 i 2015 r. nie zostały opublikowane** (stan na 2026-10).

### Wielkości użyteczne dla naszych danych

* **Porównanie skal.** α_I(aromat C) = −0.765 cal mol⁻¹ molal⁻¹ Å⁻² odpowiada α/RT = −1.29×10⁻³
  m⁻¹ Å⁻² (25 °C). Dla 2E: −6.1×10⁻⁴. To **tyle samo albo więcej co mocznik z pierścieniem zasady**
  (−1.09×10⁻³), a mocznik jest uznanym denaturantem DNA. Grupa wewnętrzna PEG mogłaby więc
  destabilizować dupleks per Å² porównywalnie z mocznikiem.
  * Zastrzeżenie: aromat białkowy (benzenowy) ≠ heteroaromat zasady. Knowles 2015 zamienia O
    karbonylowy na amidowy, a N aminowy na amidowy.
* **Wykonalne przewidywanie a priori** dla PEG 200 (N₃ = 4–5) na naszym 13-merze:
  1. m_chem = Σ_i α_i(2E + (N₃−1)·I) × ΔASA_i, gdzie ΔASA_i to rozkład ΔASA na typy atomów
     (aromat C/N, O karbonylowy, N aminowy, metyl, cukier, fosforan) dla modeli ss stacked /
     half-stacked;
  2. porównać z naszym PEG 200;
  3. dla fosforanu użyć β(HPO₄²⁻) jako górnej granicy wykluczenia.

  Brakuje α dla N pierścienia, O cukru i fosfodiestru (monoanion), więc przewidywanie będzie
  półilościowe.
* **χ:** cytując χ∞, podać obie wartości: 0.45 ± 0.10 (2011) i 0.6 ± 0.06 (2015, reanaliza tej
  samej grupy). Obie pochodzą z 12-merów przy 40 °C i 0.1 m KCl. Ekstrapolacja na nasz 13-mer jest
  rozsądna (podobny rozmiar), ale **zależności od rozmiaru DNA nikt nie zmierzył**.

### Luki

* Brak α dla EG, glicerolu i PEG z nukleozasadami.
* Brak χ(N₃, rozmiar DNA).
* Brak danych PEG–DNA w niskiej soli (Knowles: ~0.14 m K⁺; my: ~35 mM Na⁺) i brak sprzężenia
  sól–PEG w tej analizie.

---

## (e) Co przeczy obrazowi preferencyjnych oddziaływań z zasadami (albo go osłabia)

1. **Szkoła „aktywności wody / stresu osmotycznego”** (Spink & Chaires 1999; Nakano 2004, 2012;
   Ghosh 2020; Buscaglia 2013; Trajkovski 2018).
   * Destabilizację krótkich dupleksów przez EG i małe PEG przypisuje się wychwytowi wody przy
     tworzeniu par: ~4 H₂O/bp (Spink, abstrakt), 15.0 ± 0.6 H₂O/bp dla PEG 200 vs 4.7 ± 0.3 dla
     glicerolu (Nakano 2004, pełny tekst, 1 M NaCl).
   * **Słabość tej interpretacji:** „liczba wody” zależy od kosolutu. Dane EG/glicerol, PDO/2-ME i
     PEG/DME tworzą **trzy różne proste** ΔΔG°₃₇ vs Δa_w (Ghosh 2020, Fig. 3, pełny tekst lokalny,
     d(ATGCGCAT), 1 M i 100 mM NaCl). Nakano 2004 sam proponuje, że wicynalne OH w EG i glicerolu
     „solvate the nucleotides”.
   * Teoria (Shimizu 2004, pełny abstrakt; Parsegian, Rand, Rau 2000) pokazuje, że analiza
     osmotyczna bez KB miesza wodę z preferencyjnym wiązaniem kosolutu.
   * **Wniosek:** dane tej szkoły nie obalają Knowlesa. Są z nim zgodne, jeśli „uwolnioną wodę”
     przeczytać jako ΔΓ kosolutu. Sam model a_w nie tłumaczy jednak zależności od struktury
     kosolutu.
2. **Stała dielektryczna** (Nakano 2014, pełny tekst).
   * Warunki: 11-mer RNA, 1 M NaCl, ΔG°₃₇, 20 wt% kosolutu. PEG 200 destabilizuje o 1.6 kcal
     mol⁻¹, PEG 600 o 0.9.
   * Nachylenie log K vs log[NaCl] spada z 1.8 do 1.1 w PEG 8000 i **koreluje z ε_r, nie z a_w ani
     lepkością**.
   * Dla nas to ważne, bo nasza zależność K od Na⁺ i PEG może mieć składową dielektryczną, której
     ani Knowles, ani DOSY nie rozdzielają.
   * Ghosh 2020 argumentuje odwrotnie: spadek ε powinien stabilizować, a obserwujemy destabilizację.
     Spór jest nierozstrzygnięty.
3. **Wiązanie PEG z grzbietem cukrowo-fosforanowym: raczej NIE.**
   * VPO: β_I(HPO₄²⁻) = +234 i β_2E = +79 cal mol⁻¹ molal⁻¹ jon⁻¹, czyli silne wykluczanie
     (Knowles 2015).
   * MD Mathur 2025: brak kontaktu PEG-200 z fosforanami.
   * Przeciwnie MD EG (Jaiswal 2018; Zhang 2018): EG wypiera wodę głównie przy fosforanach. To
     **rozbieżność między VPO a MD** (pole siłowe?). Jeśli EG faktycznie gromadzi się przy
     fosforanach dupleksu, część efektu EG może pochodzić z grzbietu, a nie z zasad. Wtedy skalowanie
     z ΔASA zasad (Knowles) byłoby częściowo przypadkowe.
4. **Bezpośrednie wiązanie dużego PEG z eksponowanymi zasadami** (Buscaglia 2013; Trajkovski 2018;
   Tateishi-Karimata 2017).
   * Wspiera ideę chemicznego oddziaływania PEG z płaszczyznami zasad.
   * Podważa jednak „crowding” jako opis PEG.
   * Dla dużego PEG oddziaływanie jest **specyficzne dla geometrii** (płaska tetrada, końce). To
     nie to samo co addytywny model ASA z jednym α.
5. **Stacking ssDNA jako mechanizm alternatywny.**
   * Kosolut, który osłabia stacking ss (niższe ε; Norberg & Nilsson 1998, abstrakt), zwiększa
     ΔASA i entropię stanu ss.
   * Daje ten sam znak i podobne skalowanie z długością co „oddziaływanie z zasadami”, choć mechanizm
     jest inny: zmiana stanu odniesienia, a nie solwatacja.
   * Guinn 2013 pokazuje, że m-wartości mocznika są bardzo czułe na założony stopień stackingu ss
     (70–90 %).
   * Termodynamicznie oba opisy są równoważne na poziomie m-wartości. Odróżnić je może tylko pomiar
     strukturalny (CD/2-AP ssDNA w EG).
6. **Polyole „nieefektywne” wobec zasad** (Ts'o 1963). Adonitol prawie nie zmienia rozpuszczalności
   adeniny i Tm DNA. To zgodne z małym α dla glicerolu w Knowles 2015 (α_aromat −0.43 vs −2.66 dla
   tetraEG). Sugeruje, że **dla EG (bez grupy eterowej) oddziaływanie z zasadami może być słabe**, a
   jego destabilizujący efekt musi wtedy mieć inne źródło (a_w/ε/stacking). Wymagana wielkość jest
   jednak mała (§a), więc to nie wyklucza obrazu Knowlesa.
7. **Lambert & Draper 2007** (tylko abstrakt) wspiera Knowlesa: dziewięć osmolitów destabilizuje
   strukturę drugorzędową RNA, „probably because they favor solubilization of base surfaces”.
   Hipoteza, nie pomiar. Czy wśród nich był EG: **NIEZWERYFIKOWANE** (słowa kluczowe: prolina,
   mocznik, glicerol, TMAO, betaina).

**Ocena krytyczna.**

* Obraz Knowlesa jest **spójny** z całą termodynamiką:
  * skalowanie z ΔASA (dupleks/spinka ≈ 2);
  * liniowość w N₃ dla EG–triEG;
  * korzystne α PEG z aromatami;
  * zależność efektu od struktury kosolutu.
* Jest jednak **niedowiedziony mikroskopowo**:
  * brak NOE;
  * brak danych rozpuszczalności zasad w EG w dostępnej literaturze;
  * brak MD ss vs ds z KB.
* Konkurencyjne opisy (a_w, ε, stacking ss) **nie są wzajemnie wykluczające**. Nie umiem wskazać
  pracy, która by obraz Knowlesa **obaliła**.
* Najpoważniejsze zastrzeżenia dla naszej pracy:
  1. χ∞ ma dwie różne wartości od tych samych autorów (0.45 vs 0.6);
  2. α_EG ≠ α_2E na białkach;
  3. MD sugeruje akumulację EG także przy dupleksie (fosforany), co podważa założenie, że liczy się
     tylko ΔASA zasad.

---

## Bibliografia

Format: autorzy, tytuł (oryginalny), czasopismo, tom, strony, rok, DOI. Weryfikacja: Crossref
(„CR ✓”) albo brak. Dostęp: pełny tekst / tylko abstrakt / brak (tylko metadane).

1. Knowles DB, LaCroix AS, Deines NF, Shkel I, Record MT Jr. Separation of preferential interaction
   and excluded volume effects on DNA duplex and hairpin stability. *Proc. Natl. Acad. Sci. USA*
   108, 12699–12704 (2011). DOI 10.1073/pnas.1103382108. CR ✓. Pełny tekst + SI (lokalne PDF
   `literature/theoretical_models/Knowles2011_*`).
2. Knowles DB, Shkel IA, Phan NM, Sternke M, Lingeman E, Cheng X, Cheng L, O'Connor K, Record MT Jr.
   Chemical Interactions of Polyethylene Glycols (PEGs) and Glycerol with Protein Functional Groups:
   Applications to Effects of PEG and Glycerol on Protein Processes. *Biochemistry* 54, 3528–3542
   (2015). DOI 10.1021/acs.biochem.5b00246. CR ✓. Pełny tekst (PMC4464906, rękopis autorski).
3. Shkel IA, Knowles DB, Record MT Jr. Separating chemical and excluded volume interactions of
   polyethylene glycols with native proteins: Comparison with PEG effects on DNA helix formation.
   *Biopolymers* 103, 517–527 (2015). DOI 10.1002/bip.22662. CR ✓. Pełny tekst (PMC4485569).
4. Guinn EJ, Schwinefus JJ, Cha HK, McDevitt JL, Merker WE, Ritzer R, Muth GW, Engelsgjerd SW, et al.
   Quantifying Functional Group Interactions That Determine Urea Effects on Nucleic Acid Helix
   Formation. *J. Am. Chem. Soc.* 135, 5828–5838 (2013). DOI 10.1021/ja400965n. CR ✓. Pełny tekst
   (PMC3655208).
5. Cheng X, Shkel IA, Molzahn C, Lambert D, Karim R, Record MT Jr. Quantifying Interactions of
   Nucleobase Atoms with Model Compounds for the Peptide Backbone and Glutamine and Asparagine Side
   Chains in Water. *Biochemistry* 57, 2227–2237 (2018). DOI 10.1021/acs.biochem.8b00087. CR ✓.
   Pełny tekst (PMC5903953).
6. Record MT Jr, Guinn E, Pegram L, Capp M. Introductory Lecture: Interpreting and predicting
   Hofmeister salt ion and solute effects on biopolymer and model processes using the solute
   partitioning model. *Faraday Discuss.* 160, 9–44 (2013). DOI 10.1039/c2fd20128c. CR ✓. Pełny
   tekst (PMC3694758).
7. Herskovits TT, Harrington JP. Solution studies of the nucleic acid bases and related model
   compounds. Solubility in aqueous alcohol and glycol solutions. *Biochemistry* 11, 4800–4811
   (1972). DOI 10.1021/bi00775a025. CR ✓. **Brak dostępu (tylko metadane, brak abstraktu).**
8. Herskovits TT, Bowen JJ. Solution studies of the nucleic acid bases and related model compounds.
   Solubility in aqueous urea and amide solutions. *Biochemistry* 13, 5474–5483 (1974). DOI
   10.1021/bi00724a004. CR ✓. Brak dostępu (liczby tylko pośrednio przez Guinn 2013).
9. Herskovits TT. Nonaqueous solutions of DNA: Factors determining the stability of the helical
   configuration in solution. *Arch. Biochem. Biophys.* 97, 474–484 (1962). DOI
   10.1016/0003-9861(62)90110-8. CR ✓. Brak dostępu.
10. Ts'o POP, Melvin IS, Olson AC. Interaction and Association of Bases and Nucleosides in Aqueous
    Solutions. *J. Am. Chem. Soc.* 85, 1289–1296 (1963). DOI 10.1021/ja00892a016. CR ✓. Pełny tekst
    (PDF udostępniony na stronie kursu Georgia Tech; nie OA wydawcy, nie skopiowany).
11. Ts'o POP, Helmkamp GK, Sander C. Interaction of nucleosides and related compounds with nucleic
    acids as indicated by the change of helix-coil transition temperature. *Proc. Natl. Acad. Sci.
    USA* 48, 686–698 (1962). DOI 10.1073/pnas.48.4.686. CR ✓. Brak (PMC220834 to skan, efetch
    zwraca tylko metadane). Wyniki znane tylko z cytowania w Ts'o 1963.
12. Levine L, Gordon JA, Jencks WP. The Relationship of Structure to the Effectiveness of Denaturing
    Agents for Deoxyribonucleic Acid. *Biochemistry* 2, 168–175 (1963). DOI 10.1021/bi00901a030.
    CR ✓. Brak dostępu.
13. Scruggs RL, Achter EK, Ross PD. The thermodynamic effects of exposing nucleic acid bases to
    water: Solubility measurements in water and organic solvents. *Biopolymers* 11, 1961–1972
    (1972). DOI 10.1002/bip.1972.360110915. CR ✓. Brak dostępu (nie cytować treści).
14. Xu R, Zheng M, Farajtabar A, Zhao H. Solubility modelling and preferential solvation of adenine
    in solvent mixtures of (N,N-dimethylformamide, N-methyl pyrrolidone, propylene glycol and
    dimethyl sulfoxide) plus water. *J. Chem. Thermodyn.* 125, 225–234 (2018). DOI
    10.1016/j.jct.2018.05.030. CR ✓. Brak dostępu (znany z cytowania w Arakawa 2025).
15. Arakawa T, Oyama T, Ura T, Nishinami S, Shiraki K, Akuta T. Differences and Similarities in
    Protein and Nucleic Acid Structures and Their Biological Interactions. *Curr. Issues Mol. Biol.*
    47, 1019 (2025). DOI 10.3390/cimb47121019. CR ✓. Pełny tekst (PMC12731849).
16. Hirano A, Shiraki K, Arakawa T. Polyethylene glycol behaves like weak organic solvent.
    *Biopolymers* 97, 117–122 (2012; Crossref podaje datę wydania 2011). DOI 10.1002/bip.21708.
    CR ✓. Tylko abstrakt.
17. Trajkovski M, Endoh T, Tateishi-Karimata H, Ohyama T, Tanaka S, Plavec J, Sugimoto N. Pursuing
    origins of (poly)ethylene glycol-induced G-quadruplex structural modulations. *Nucleic Acids
    Res.* 46, 4301–4315 (2018). DOI 10.1093/nar/gky250. CR ✓. Pełny tekst (PMC5934638; PDF
    `review_2026-10/Trajkovski2018_NAR_EG_PEG8000_Gquadruplex_NMR.pdf`).
18. Buscaglia R, Miller MC, Dean WL, Gray RD, Lane AN, Trent JO, Chaires JB. Polyethylene glycol
    binding alters human telomere G-quadruplex structure by conformational selection. *Nucleic
    Acids Res.* 41, 7934–7946 (2013). DOI 10.1093/nar/gkt440. CR ✓. Pełny tekst (PMC3763525, XML;
    PDF niepobieralny skryptem).
19. Tateishi-Karimata H, Ohyama T, Muraoka T, Podbevsek P, Wawro AM, Tanaka S, Nakano S, Kinbara K,
    Plavec J, Sugimoto N. Newly characterized interaction stabilizes DNA structure: oligoethylene
    glycols stabilize G-quadruplexes CH–π interactions. *Nucleic Acids Res.* 45, 7021–7030 (2017).
    DOI 10.1093/nar/gkx299. CR ✓. Pełny tekst (PMC5499538).
20. Jia F, Lu X, Tan X, Wang D, Cao X, Zhang K. Effect of PEG Architecture on the Hybridization
    Thermodynamics and Protein Accessibility of PEGylated Oligonucleotides. *Angew. Chem. Int. Ed.*
    56, 1239–1243 (2017). DOI 10.1002/anie.201610753. CR ✓. Pełny tekst (PMC9018076, rękopis
    autorski).
21. Spink CH, Chaires JB. Effects of hydration, ion release, and excluded volume on the melting of
    triplex and duplex DNA. *Biochemistry* 38, 496–508 (1999; Crossref: 1998 online). DOI
    10.1021/bi9820154. CR ✓. Tylko abstrakt.
22. Stanley C, Rau DC. Preferential Hydration of DNA: The Magnitude and Distance Dependence of
    Alcohol and Polyol Interactions. *Biophys. J.* 91, 912–920 (2006). DOI
    10.1529/biophysj.106.086579. CR ✓. Tylko abstrakt (PMC1563772 to skan; kopia CORE zablokowana).
23. Brewood GP, Aliwarga T, Schurr JM. A structural transition in duplex DNA induced by ethylene
    glycol. *J. Phys. Chem. B* 112, 13367–13380 (2008). DOI 10.1021/jp802139a. CR ✓. Tylko
    abstrakt.
24. Jaiswal AK, Srivastava R, Pandey P, Bandyopadhyay P. Microscopic picture of water-ethylene glycol
    interaction near a model DNA by computer simulation: Concentration dependence, structure, and
    localized thermodynamics. *PLoS ONE* 13, e0206359 (2018). DOI 10.1371/journal.pone.0206359.
    CR ✓. Pełny tekst (PDF `review_2026-10/Jaiswal2018_PLoSOne_MD_ethylene_glycol_near_DNA.pdf`).
25. Mathur N, Singh N. Molecular Crowder-Induced Structural Transformation of the DNA Dodecamer.
    *ACS Omega* 10, 41425–41435 (2025). DOI 10.1021/acsomega.5c04572. CR ✓. Pełny tekst
    (PMC12444521).
26. Zhang N, Li MR, Zhang FS. Ethylene glycol solution-induced DNA conformational transitions.
    *Chin. Phys. B* 27, 113102 (2018). DOI 10.1088/1674-1056/27/11/113102. CR ✓. **Brak dostępu**
    (opis tylko z wyszukiwarki; nie cytować treści bez lektury).
27. Diaz A, Jothiraman HB, Ramakrishnan V. Effect of glycerol on free DNA: A molecular dynamics
    simulation study. *J. Mol. Graph. Model.* 114, 108169 (2022). DOI 10.1016/j.jmgm.2022.108169.
    CR ✓. Tylko abstrakt.
28. Norberg J, Nilsson L. Solvent influence on base stacking. *Biophys. J.* 74, 394–402 (1998). DOI
    10.1016/S0006-3495(98)77796-3. CR ✓. Tylko abstrakt.
29. Pincus DL, Hyeon C, Thirumalai D. Effects of Trimethylamine N-Oxide (TMAO) and Crowding Agents on
    the Stability of RNA Hairpins. *J. Am. Chem. Soc.* 130, 7364–7372 (2008). DOI 10.1021/ja078326w.
    CR ✓. Tylko abstrakt.
30. Lambert D, Draper DE. Effects of Osmolytes on RNA Secondary and Tertiary Structure Stabilities
    and RNA-Mg²⁺ Interactions. *J. Mol. Biol.* 370, 993–1005 (2007). DOI 10.1016/j.jmb.2007.03.080.
    CR ✓. Tylko abstrakt.
31. Nakano S, Karimata H, Ohmichi T, Kawakami J, Sugimoto N. The Effect of Molecular Crowding with
    Nucleotide Length and Cosolute Structure on DNA Duplex Stability. *J. Am. Chem. Soc.* 126,
    14330–14331 (2004). DOI 10.1021/ja0463029. CR ✓. Pełny tekst (lokalny PDF
    `literature/pdf/JACS Sugimoto - 2004 - …`).
32. Nakano S, Yamaguchi D, Tateishi-Karimata H, Miyoshi D, Sugimoto N. Hydration Changes upon DNA
    Folding Studied by Osmotic Stress Experiments. *Biophys. J.* 102, 2808–2817 (2012). DOI
    10.1016/j.bpj.2012.05.019. CR ✓. Tylko abstrakt.
33. Nakano S, Kitagawa Y, Miyoshi D, Sugimoto N. Hammerhead ribozyme activity and oligonucleotide
    duplex stability in mixed solutions of water and organic compounds. *FEBS Open Bio* 4, 643–650
    (2014). DOI 10.1016/j.fob.2014.06.009. CR ✓. Pełny tekst (PMC4141205).
34. Ghosh S, Takahashi S, Ohyama T, Endoh T, Tateishi-Karimata H, Sugimoto N. Nearest-neighbor
    parameters for predicting DNA duplex stability in diverse molecular crowding conditions. *Proc.
    Natl. Acad. Sci. USA* 117, 14194–14201 (2020). DOI 10.1073/pnas.1920886117. CR ✓. Pełny tekst
    (lokalny PDF `literature/pdf/PNAS2 ghosh-et-al-2020-…`).
35. Shimizu S. Estimating hydration changes upon biomolecular reactions from osmotic stress, high
    pressure, and preferential hydration experiments. *Proc. Natl. Acad. Sci. USA* 101, 1195–1199
    (2004). DOI 10.1073/pnas.0305836101. CR ✓. Tylko abstrakt.
36. Parsegian VA, Rand RP, Rau DC. Osmotic stress, crowding, preferential hydration, and binding: A
    comparison of perspectives. *Proc. Natl. Acad. Sci. USA* 97, 3987–3992 (2000). DOI
    10.1073/pnas.97.8.3987. CR ✓. Brak (tylko metadane; cytowana jako ramy teoretyczne).
37. Hong J, Capp MW, Anderson CF, Saecker RM, Felitsky DJ, Anderson MW, Record MT Jr. Preferential
    Interactions of Glycine Betaine and of Urea with DNA: Implications for DNA Hydration and for
    Effects of These Solutes on DNA Stability. *Biochemistry* 43, 14744–14758 (2004). DOI
    10.1021/bi049096q. CR ✓. Brak dostępu (szablon VPO dla DNA–kosolut; treści nie cytuję).
38. Zohra FT, Al-Zuhairi H, Reinoza J, Kim H, Hanke A. Probing the effect of PEG-DNA interactions and
    buffer viscosity on tethered DNA in shear flow. *PLoS ONE* 20, e0329961 (2025). DOI
    10.1371/journal.pone.0329961. CR ✓. Abstrakt przeczytany (pełny tekst w PMC12377632; PEG
    modelowany wyłącznie jako efektywne przyciąganie EV, nieistotne dla Q1).
