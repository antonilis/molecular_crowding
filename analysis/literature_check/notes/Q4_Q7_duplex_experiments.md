# Q4 + Q7: eksperymenty z dupleksami i spinkami w EG/PEG; cytowania Bielec 2022; weryfikacja Henning-Knechtel 2026

Stan na 2026-10-09. Wszystkie DOI sprawdzone przez Crossref API (`api.crossref.org/works/<DOI>`), chyba że zaznaczono inaczej.
Oznaczenia dostępu: **pełny tekst** (przeczytany pełny tekst, z podaniem źródła) / **tylko abstrakt**.
Pobrane pliki OA: `literature/review_2026-10/q4_q7/` (PDF lub pełnotekstowy XML z Europe PMC, gdy wydawca blokował PDF).
Przeliczenia oznaczone „[nasze przeliczenie]” są moje, nie autorów. 1 kcal = 4.184 kJ.
„Monomolal” = mol jednostek EO na kg wody (skala Knowlesa i nasza skala „mol EO/kg”).

---

## Q4. Dupleksy i spinki DNA/RNA w EG/PEG o różnej masie molowej

### 4.1 Główna linia doświadczalna: znak efektu długiego PEG zależy od soli

| Praca | Układ, warunki | Wynik (liczby z pełnego tekstu, chyba że zaznaczono) | Dostęp |
|---|---|---|---|
| Nakano et al. 2004 JACS (+SI) | 8-mer ATGCGCAT (samokomplementarny), 17-mer, 30-mer; 1 M NaCl + 10 mM Na2HPO4, pH 7; UV-melting; ΔG przy 25 °C | **Przy 1 M NaCl żaden PEG nie stabilizuje 8-meru.** ΔG°25 [kcal/mol]: brak −11.2; PEG 8000 10/15/20 wt% −10.5/−10.0/−10.7; PEG 6000 20% −10.3; PEG 2000 20/40% −10.1/−9.1; PEG 1000 20/40% −9.9/−7.6; PEG 200 10–50% −9.8/−8.7/−7.6/−6.8/−5.9; EG 20/30/40% −9.3/−8.3/−7.4. Przy 100 mM NaCl: brak −10.2; PEG 200 10–50% −9.6/−8.8/−7.8/−7.2/−5.3 (SI Tab. S1). Tm przy 20 wt% (5 µM): PEG 8000 daje −5.1 °C (8-mer), −0.8 °C (17-mer), +0.8 °C (30-mer przy 10 wt%). Efekt zależy od długości: krótsze dupleksy i mniejsze PEG są bardziej destabilizowane. Autorzy przypisują efekt aktywności wody (ln K vs ln aw liniowe, wszystkie PEG na jednej prostej; −Δnw = 15.0 ± 0.6 przy 1 M i 14.4 ± 0.5 przy 0.1 M NaCl). | pełny tekst + SI (lokalnie `literature/pdf/JACS Sugimoto 2004*`) |
| Karimata, Nakano, Sugimoto 2007 Bull Chem Soc Jpn | spinka DNA i dupleks; PEG 200 i PEG 8000; różne stężenia NaCl | Według abstraktu: PEG 200 i 8000 destabilizują przy wysokim NaCl, a **PEG 8000 wyraźnie stabilizuje przy niskim NaCl**. PEG zmniejsza −ΔnNa+ i nie zmienia −Δnw („PEG buffers sodium ion binding”). Liczby (próg NaCl, ΔG) są w pełnym tekście, do którego nie mamy dostępu (OUP, tylko PDF dla subskrybentów). | **tylko abstrakt** |
| Karimata et al. 2004 Nucleic Acids Symp Ser | 8-mer ATGCGCAT | Według abstraktu: ΔTm = −11.8 °C (20 wt% PEG 200) i −1.5 °C (20 wt% PEG 8000). Po obniżeniu NaCl z 1 M do 400 mM „Tm increased in presence of PEGs” (sformułowanie dwuznaczne). | **tylko abstrakt** |
| Markarian & Schlenoff 2010 JPCB | 15-, 25-, 35-mer i molecular beacon; „PEG 4000 (Mw ≈ 3000)” 1–30 wt%; 1 mM fosforan, pH 7, Na+ 2–225 mM; 35/55/67/75 °C; 1 µM każdej nici; hipochromizm | **Dwa reżimy.** Przy a(Na+) < 0.05 Kobs rośnie z PEG (woda uwalniana, −4.0 ± 0.6 H2O/bp po ekstrapolacji). Przy a(Na+) ≥ 0.05 Kobs maleje z PEG (4.5 ± 0.5 H2O/bp wiązane przy aNa+ = 0.4), mocniej dla 25- i 35-meru niż dla 15-meru. „The pairing of the oligonucleotides requires a lower [NaCl] with increasing PEG3 concentration.” Pobór kontrjonów Δψ maleje ze wzrostem PEG i %GC. Przy 2 mM Na+ nie ma parowania nawet przy 30% PEG. Aktywność Na+ wyznaczali VPO, zakładając brak wiązania Na+–PEG w wodzie. | pełny tekst (manuskrypt autorski PMC2925268 przez NCBI BioC) |
| Goobes et al. 2003 Biochemistry | DNA–DNA, ITC i UV-melting, crowding | Według abstraktu: efekt crowdingu jest marginalny w warunkach sprzyjających asocjacji i rośnie, gdy warunki się pogarszają („buffering”), z korzystnym członem entalpowym. Knowles 2011 streszcza to (ref. 10) jako: „at low salt (0.15 M), high molecular weight PEGs stabilize small duplexes”. | **tylko abstrakt** |
| Spink & Chaires 1999 Biochemistry | E. coli DNA, poly(dA)·poly(dT), tripleks; EG i PEG | Według abstraktu: ok. 4 H2O/bp uwalniane przy topnieniu dupleksu. W EG wiązanie Na+ (kontrjonów) nieznacznie maleje ze spadkiem aw. Duże PEG stabilizują przez objętość wyłączoną (dotyczy polimerowego DNA). | **tylko abstrakt** |
| Spink & Chaires 1995 JACS | tripleks vs dupleks, PEG o różnej M | Brak abstraktu w Crossref/PubMed. Tylko streszczenie wtórne (Nakano 2004): Tm tripleksu, ale nie dupleksu, rośnie z M PEG; EG i małe PEG obniżają Tm dupleksu. | **nie czytano** (tylko opis wtórny) |

**Wniosek 4.1.** Literatura jest zgodna co do znaku: duży PEG stabilizuje krótkie dupleksy przy niskiej soli (≲0.1–0.15 M jednowartościowego kationu) i destabilizuje je przy wysokiej soli (1 M NaCl). Nasz wynik (35 mM Na+, stabilizacja −4 do −6 kJ/mol przez długie PEG) mieści się w reżimie „niskiej soli” według Markariana i Karimaty. Jest ilościowo przeciwny danym Nakano 2004 przy 1 M NaCl, gdzie PEG 8000 przy 10–20 wt% destabilizuje 8-mer o +0.5…+1.2 kcal/mol (+2…+5 kJ/mol) [nasze przeliczenie z SI Tab. S1].

### 4.2 Nasycenie i odwrócenie stabilizacji; skala monomeru; niezależność od M

| Praca | Warunki | Wynik | Dostęp |
|---|---|---|---|
| Knowles et al. 2011 PNAS (+SI) | 12-bp dupleks (58% AT) i 12-nt spinka (4-bp trzon); 0.02 m K2HPO4 + 0.1 m KCl + 0.001 m K2EDTA (K+ ≈ 0.14 m), pH 7.3; 313 K; UV-melting; EG, diEG, triEG, PEG 200 … 8000 | **Jedyna praca z wyraźną niezależnością od M na skali monomolalnej.** m-value/N3 [kcal/(molal monomeru)], znak dodatni = stabilizacja: EG–triEG: dupleks −0.340 ± 0.01, spinka −0.144 ± 0.01. PEG 200: −0.21 i −0.12. Plateau dla wysokiej M (PEG ≥ 2000, N3 ≳ 45): **dupleks +0.156 ± 0.020** (= +0.65 kJ/mol na monomolal), spinka −0.040 ± 0.010. Przejście między plateau następuje między PEG 200 a 2000. PEG 600 „stabilizes the duplex slightly”. Stabilność zmienia się w przybliżeniu liniowo do 5 monomolal. **Szerokie przejście z centrum przy 6 monomolal.** Od 6 do 12 monomolal m-values dla PEG 2000/3350/8000 są porównywalne z PEG 200, czyli następuje odwrócenie. Odwrócenia nie ma dla spinki ani dla małych PEG. Autorzy przypisują je nakładaniu się i splątaniu kłębków PEG (zachowanie „jak niezależne segmenty PEG 200”). | pełny tekst + SI (lokalnie `literature/theoretical_models/Knowles2011_*`) |
| Nakano 2004 / Ghosh 2020 | 1 M NaCl, 8-mer | Na skali **Δaw** (nie monomeru) PEG 200, 2000 i 8000 oraz 1,2-DME leżą na jednej prostej ΔΔG vs Δaw. Inny rodzaj uniwersalności: osmotyczny, nie objętość wyłączona. | pełny tekst |

**Porównanie z naszymi danymi.**
1. Położenie odwrócenia zgadza się: u nas 6–7 mol EO/kg, u Knowlesa centrum przy 6 monomolal.
2. Kształt się nie zgadza. U Knowlesa stabilizacja rośnie **liniowo do ~5 monomolal**. Maksimum to ok. 0.156 × 5–6 ≈ 0.8–0.9 kcal/mol ≈ 3.3–3.9 kJ/mol [nasze przeliczenie ze zboczy; Fig. 4 nie była odczytywana numerycznie]. U nas podobna amplituda (−4 do −6 kJ/mol) pojawia się już przy kilku wt%, czyli ≲1 monomolal, a potem się nasyca. Zbocze początkowe jest więc u nas ok. 5–10× większe niż u Knowlesa. Możliwe przyczyny, nierozstrzygnięte w literaturze: (a) niższa sól (35 mM Na+ vs 0.14 m K+), zgodnie z Markarianem i Karimatą; (b) Na+ vs K+; (c) 298 vs 313 K; (d) nM (FCS) vs µM (UV).
3. Plateau M zaczyna się u nas od ~600 Da, u Knowlesa między 200 a 2000 (PEG 600 jest tam pośredni). To drobna, ale realna różnica.

**Napięcie wewnątrz naszej grupy.** Bielec 2022 (Fig. 2) pisze, że poniżej kilku wt% zmiana K jest „nearly negligible”. Nasz obecny wynik to −4 do −6 kJ/mol (K × 5–11) przy kilku wt% długiego PEG. Trzeba to w manuskrypcie wyjaśnić (inny zestaw PEG/M? inna sól, 100 mM PB?).

### 4.3 Zależność od soli: czy ilościowo wspiera mechanizm kompleksowania Na+ (Bielec 2022)?

Mechanizm Bielca: κ ≈ 0.49 M⁻¹ na monomer, K ∝ [Na+]^2.51 dla 13-bp, spadek wolnego Na+ o ~73% w 40 wt% PEG 400. Dowody z literatury, od najmocniejszych:

1. **Knowles 2011 SI, Tab. S3 (VPO, pełny tekst).** Oddziaływanie PEG–KCl jest **niekorzystne**: μ3,4 = +0.0378 ± 0.001 (PEG 1000) i +0.0324 ± 0.001 (PEG 2000) kcal/(molal monomeru). PEG *podnosi* potencjał chemiczny (aktywność) KCl, zamiast go obniżać, jak wymagałoby kompleksowanie. Przy 6 monomolal daje to Δμ_KCl ≈ +0.2 kcal/mol, czyli Δln a± ≈ +0.16 (≈ +17% aktywności średniej) [nasze przeliczenie, przy założeniu stałego μ3,4]. Zastrzeżenia: dotyczy K+, nie Na+; μ3,4 to wielkość netto dla obu jonów, więc silne wykluczenie Cl⁻ mogłoby maskować słabe wiązanie kationu. Mimo to jest to bezpośrednia termodynamiczna miara, która **przeczy netto obniżeniu aktywności soli przez PEG**.
2. **Nakano 2004 SI (pełny tekst).** Destabilizacja przez 40 wt% PEG 200 wynosi **+4.4 kcal/mol przy 1 M NaCl i +3.0 kcal/mol przy 0.1 M NaCl**, tj. 0.11 vs 0.075 kcal/mol/wt% [nasze przeliczenie]. Ghosh 2020: PEG 200 przy 1 M i 0.1 M NaCl leży na tej samej prostej ΔΔG vs Δaw („independent of the ionic concentration”). Przy 1 M NaCl wiązanie Na+ przez PEG prawie nie powinno zmieniać K: według samego Bielca K ma maksimum przy IS ≈ 0.7 M i maleje powyżej. Destabilizacja przez PEG 200 przy wysokiej soli jest więc co najmniej w dużej części **niezależna od Na+**: osmotyczna/preferencyjne oddziaływanie według Knowlesa i Sugimoto. Zastrzeżenia: inny oligo (8-mer), µM, UV.
3. **Markarian 2010 (pełny tekst).** Długi PEG *obniża* [NaCl] potrzebne do parowania, odwrotnie niż przewiduje kompleksowanie. Autorzy powołują się na brak wiązania Na+–PEG w wodzie, ale bez własnego pomiaru.
4. **Karimata 2007, Nakano 2008 Mol BioSyst, Spink & Chaires 1999 (abstrakty).** W PEG/EG zmniejsza się liczba Na+ pobieranych przy tworzeniu dupleksu/spinki (−ΔnNa+ ↓). Proste kompleksowanie (wolny Na+ = c/(1+κ[EO]), niezależnie od c) przesuwa ln K równolegle i **nie zmienia** nachylenia d ln K/d ln[Na+]. Spadek nachylenia wymaga innego mechanizmu (stała dielektryczna, aktywności, preferencyjne oddziaływanie). Te dane nie są więc dowodem na mechanizm Bielca, choć często są tak cytowane.
5. **Kilburn 2010 JACS (pełny tekst).** Rybozym Azoarcus, PEG 1000 0–20% w/v, 20 mM Tris. Punkt środkowy zwijania spada z 0.7 do 0.21 mM MgCl2, czyli PEG zmniejsza potrzebne Mg2+. Zmianę aktywności jonów tylko *oszacowano* (~3.5% z molalności i Gibbsa–Duhema), nie zmierzono. Brak sygnału sekwestracji Mg2+, ale dominuje objętość wyłączona.

**Ocena.** Literatura przypisuje stabilizację długimi PEG przy niskiej soli i destabilizację przy wysokiej soli objętości wyłączonej, aktywności wody i preferencyjnemu oddziaływaniu PEG z odsłoniętymi zasadami. **Żadna niezależna praca doświadczalna nie potwierdza wiązania Na+ przez PEG w wodzie o sile κ ~0.5 M⁻¹/monomer.** Bezpośredni pomiar termodynamiczny (Knowles, VPO, K+) wskazuje przeciwny znak oddziaływania sól–PEG. Mechanizm Na+ może co najwyżej być małą poprawką. Nasze DOSY-NMR to jedyny bezpośredni pomiar sprzężenia PEG–Na+ w tym kontekście i musi być porównany z μ3,4 Knowlesa (zob. też notatki Q2). Obraz Knowlesa jest spójny z naszym położeniem odwrócenia, ale nie z szybkim nasyceniem.

### 4.4 Spinki, single-molecule, kinetyka (2014–2025)

| Praca | Układ | Wynik | Dostęp |
|---|---|---|---|
| Baltierra-Jasso et al. 2015 JACS | spinka DNA, smFRET (MFD dyfuzyjny + TIRF), PEG 400, PEG 8000, EG | Przy 10% w/w: k_close 1.52 → 5.02 s⁻¹ (PEG 400) i 5.81 s⁻¹ (PEG 8000); k_open 2.01 → 1.14 i 1.13 s⁻¹. **ΔΔG = −1.23 ± 0.06 (PEG 400) i −1.73 ± 0.03 kcal/mol (PEG 8000)**, tj. −5.1 i −7.2 kJ/mol. Ekstrapolacja do 30%: −3.72 i −5.19 kcal/mol. EG prawie bez efektu. Pomiar zespołowy: zamknięty stan w PEG 8000 bardziej faworyzowany przy niższej soli (Fig. S9; SI nieczytane, bufor niepodany w tekście głównym). Duży efekt przy niskich wt%, podobnie jak u nas. | pełny tekst (eprint Glasgow, CC-BY), SI nie |
| Dupuis, Holmstrom, Nesbitt 2014 PNAS | RNA, tertiary tetraloop–receptor, smFRET, PEG 8000 | 4% wt/wt: K × 6.3, ΔΔG ≈ −1.1 kcal/mol. 16%: K × 62 (0.34 → 21), ΔΔG ≈ −2.5 kcal/mol. Efekt kinetyczny głównie przez przyspieszenie zwijania. Bufor w SI (nieczytane). | pełny tekst (NSF PAR) |
| Sung & Nesbitt 2021 JPCB | smFRET, PEG vs dekstran/Ficoll, T-zależność | PEG: stabilizacja głównie entropowa (k_fold ↑, k_unfold ↓). Dekstran/Ficoll: brak efektu przy RT, za to preferencyjne oddziaływanie z formą rozwiniętą. | **tylko abstrakt** |
| Sung, Sengupta, Nesbitt 2021 J Chem Phys | RNA tertiary, crowdery 90–3000 Da | „Smaller molecules crowd better” na jednostkę objętości, opisane modelem deplecji. | **tylko abstrakt** |
| Martini et al. 2025 JPCB | spinki CAG/CTG, smFRET | Crowding przyspiesza zamykanie, a otwieranie zależnie od rozmiaru crowdera. **Przesuwa równowagę od spinki wraz ze wzrostem NaCl**, czyli ten sam trend sól × crowder co Karimata/Markarian. Crowdery i liczby niepodane w abstrakcie. | **tylko abstrakt** |
| Hudoba et al. 2017 ACS Nano | origami „NanoDyn”, smFRET; 0.5×TBE + 11 mM MgCl2; PEG 0.3/1/3/10/35 kDa, 5% w/w; PEG 35k 0–5% | Wszystkie PEG oprócz 0.3 kDa przesuwają K w stronę stanu zamkniętego. PEG 0.3 wchodzi w szczelinę 4 nm. ΔΔG_PEG jest liniowe w [PEG35] (deplecja), rozdzielczość siły ~40 fN. Pokazuje próg M dla deplecji przy danej geometrii, ale nie dotyczy hybrydyzacji. | pełny tekst (lokalnie `literature/pdf/not done/dynamic-dna-origami*`) |
| Kilburn 2010 | zob. 4.3 | — | pełny tekst |
| Nakano et al. 2012 Biophys J | osmotic stress, małe kosolutanty, dupleksy/spinki z mismatchami | Hydratacja przy zwijaniu. Brak dużych PEG. | **tylko abstrakt** |

Nie znaleziono pracy 2015–2026 mierzącej hybrydyzację *dwucząsteczkową* krótkiego dupleksu DNA w PEG o wielu M przy niskiej soli przez smFRET/FCS (poza naszymi pracami). Najbliższe są spinki (Baltierra-Jasso, Martini) i struktury RNA (Nesbitt).

### 4.5 RNA i NN-parametry w małych PEG (kontekst destabilizacji liniowej)

- Ghosh et al. 2020 PNAS (pełny tekst + SI, lokalnie): 40 wt% PEG 200, 100 mM NaCl, 37 °C, NN-parametry DNA. Model ΔG_crowder = m_cs·Δaw, plus człon objętości wyłączonej Hermansa dla długich PEG: 30-mer w 10% PEG 8000, −1.3 kcal/mol.
- Ghosh et al. 2023 NAR (pełny tekst): RNA, 40 wt% PEG 200, 100 mM NaCl. Tylko PEG 200/400/600. W 40% PEG 200 dupleksy RNA są stabilniejsze w Na+ niż w K+ (średnio o 12.1%). Stała dielektryczna spada do ~50%.
- Adams & Znosko 2019 NAR (pełny tekst): RNA, 20% (v/v) PEG 200, 1 M NaCl. Średnia destabilizacja +1.02 kcal/mol (zakres 0.39–2.27).
- Gu et al. 2013 JPCB (lokalnie, przejrzane): RNA z końcowymi mismatchami A, PEG 200, różne NaCl.

Te prace wspierają liniową destabilizację przez EG/PEG 200, którą obserwujemy, i jej osmotyczno-solwatacyjną interpretację. Nie zawierają danych dla długich PEG przy wysokich stężeniach.

### 4.6 Luki
- Brak pełnego tekstu Karimata 2007 (próg NaCl dla zmiany znaku efektu PEG 8000; −ΔnNa+ z i bez PEG). **To najważniejsza brakująca liczba dla naszej dyskusji o soli.** Warto zamówić PDF przez bibliotekę IChF.
- Brak pomiaru całej krzywej (0–40 wt%) dla wielu M przy niskiej soli poza Knowlesem (0.14 m K+) i nami.
- Nikt poza Knowlesem nie testował explicite skali monomolalnej.
- Nie ma niezależnego potwierdzenia κ(Na+–EO) ~0.5 M⁻¹ w wodzie z pomiaru aktywności (ISE, VPO, potencjometria). Bielec 2022 sam raportuje trudności z ISE (SI S7).

---

## Cytowania Bielec et al. 2022 (JPCL 13, 112; DOI 10.1021/acs.jpclett.1c03596)

Źródła: Crossref `is-referenced-by-count` = 11; OpenAlex (W4200066301, `filter=cites:`) = 11 rekordów (w tym 2 preprinty SSRN); Semantic Scholar = 9; Europe PMC = 7. Lista po scaleniu (DOI zweryfikowane w Crossref):

| # | Praca | DOI | Związek z mechanizmem Na+ | Dostęp / podstawa oceny |
|---|---|---|---|---|
| 1 | Kowalski, Bielec, Bubak … Hołyst 2022, Nat Commun 13, 6451 | 10.1038/s41467-022-34182-z | **Używa** (autocytowanie). Cytuje Bielca jako przykład jonowo czułej hybrydyzacji DNA (13 bp), nie testuje kompleksowania. | pełny tekst (`q4_q7/Kowalski2022_*.pdf`) |
| 2 | Karpińska, Zgorzelska, Kwapiszewska, Hołyst 2022, J Colloid Interface Sci 627, 270 | 10.1016/j.jcis.2022.07.040 | Ta sama instytucja. Kontekst niesprawdzony (dostarczanie DNA w medium hipertonicznym z polimerami). | nie czytano |
| 3 | Kwapiszewska 2024, ACS Phys Chem Au 4, 314 | 10.1021/acsphyschemau.3c00079 | **Wspiera bezkrytycznie** (ta sama instytucja): „adding polymers to the buffer can significantly change ionic strength”. | pełny tekst (PMC11274282) |
| 4 | Karoń, Drozd, Malinowska 2024, Biosensors 14, 136 | 10.3390/bios14030136 | **Neutralne/użycie ogólne**: cytuje za tezą, że kationy ekranują ładunek i potrzebne są bufory o wysokiej sile jonowej. | pełny tekst (PMC10968432) |
| 5 | Bai et al. 2025, Signal Transduct Target Ther 10, 94 | 10.1038/s41392-025-02186-y | **Nieistotne**: cytuje przy wzorze na stałą kompleksowania Ca2+–alginian. | pełny tekst (PMC11920195) |
| 6 | Kowalski, Piros … Hołyst 2025, Chem Eng J 524, 169512 | 10.1016/j.cej.2025.169512 | **Używa** (autocytowanie): porównanie wartości K_DNA-DNA = 3.6·10⁹ M⁻¹ i metody FCS/MBA. Mechanizmu nie testuje. | pełny tekst (manuskrypt, repozytorium Radboud) |
| 7 | (preprint pracy 6) SSRN 2025 | 10.2139/ssrn.5382834 | jak 6 | nie czytano |
| 8 | Pilarski & Meints 2026, Biophys Chem 329, 107548 | 10.1016/j.bpc.2025.107548 | Kontekst niesprawdzony. Abstrakt: PEG ma „minimal effect” (< 5%) na równowagę BI/BII szkieletu (³¹P NMR), a K+ i Mg2+ ją przesuwają. Nie testuje kompleksowania. | **tylko abstrakt** |
| 9 | (preprint pracy 8) SSRN 2025 | 10.2139/ssrn.5445680 | jak 8 | Crossref nie zwraca metadanych: **NIEZWERYFIKOWANE** w Crossref (jest w OpenAlex) |
| 10 | Silva & Weber 2026, ACS Omega 11, 29787 | 10.1021/acsomega.6c00104 | **Przyjmuje jako fakt (wspiera, bez testu)**: „PEG also results in ion complexation, effectively reducing the ionic strength”. Używa tego jako argumentu, że sama aw nie wyjaśnia efektu PEG 200. Model mezoskopowy (Peyrard–Bishop) na danych Tm Sugimoto (40 wt% PEG 200, 122 mM Na+). | pełny tekst (PMC13216965) |
| 11 | Shevtsov et al. 2026, Kataliz v promyshlennosti 26(2), 3 | 10.18412/1816-0387-2026-2-3-17 | Sorbenty CO2 K2CO3/CNT. Prawdopodobnie cytowanie poboczne; kontekst niesprawdzony. | nie czytano |

**Wniosek.** Po 4 latach żadna praca nie przetestowała ani nie potwierdziła niezależnie kompleksowania Na+ przez PEG/dekstran w wodzie. 4 z 9 unikalnych prac to autocytowania lub ta sama instytucja (IChF PAN). Jedyna zewnętrzna praca, która przyjmuje mechanizm (Silva & Weber 2026), robi to bez weryfikacji. **Brak prac, które mu jawnie przeczą przez cytowanie**, ale niezależne wcześniejsze dane (Knowles μ3,4 > 0; Nakano/Ghosh: brak zależności od NaCl; Markarian) są z nim w napięciu (zob. 4.3). Henning-Knechtel 2026 (Q7) **nie cytuje** Bielca.

---

## Q7. Henning-Knechtel, Brnović, He, Kirmizialtin, Nucleic Acids Res. 54, gkag752 (2026)

**Weryfikacja.** Praca istnieje. Crossref: „Protein crowders remodel RNA electrostatics, hydration, and dynamics: a challenge to steric crowding models”, NAR 54(14), gkag752, published-print 2026-07-17, CC BY-NC 4.0, 89 referencji, 0 cytowań (stan na 2026-10-09). Preprint bioRxiv 10.1101/2025.09.25.678182 (posted 2025-09-27) zweryfikowany w Crossref. **Pełny tekst przeczytany**: wersja wydawnicza przez Europe PMC (PMC13425243, XML) oraz PDF preprintu, oba w `q4_q7/`. Suplementu (Tab. S2–S7, Fig. S15–S18) nie przeczytano.

**Metody (z pełnego tekstu):**
- All-atom MD, GROMACS 2018.8, NPT, 310 K.
- RNA: HIV-1 TAR (PDB 1ANR, model 1) przedłużony dupleksem do 52 nt; spinka z wybrzuszeniem i pętlą.
- Pola siłowe: AMBER ff14SB (białka) / OL3 (RNA), z poprawkami **CUFIX** dla par białko–RNA. **Na+ i Cl⁻: parametry Joung–Cheatham.** Woda **TIP4P**. Sami autorzy zaznaczają, że modele 4-punktowe mogą wpływać na stabilność RNA.
- **PEG 6000: GAFF z ładunkami RESP.** Walidacja tylko względem Rg i gęstości. **Brak walidacji oddziaływania Na+–eter z danymi doświadczalnymi.**
- Stężenia: **150 mM NaCl**, bez Mg2+ (świadomie). PEG: 72 cząsteczki PEG 6000, **165.22 g/L** (≈ 14 wt%; ≈ 3.7 M EO, ≈ 4.3 mol EO/kg wody [nasze przeliczenie]). Białka: mieszanina S18, TCTP, L23, tubulina α1b, 164.97 g/L. Udział objętościowy podobny dla obu (v̄ = 0.83 i 0.73 mL/g).
- Protokół: 200 ns z zamrożonymi cząsteczkami rozpuszczonymi (równowaga jonów), potem 200 ns z zamrożonym RNA. Produkcja: 1 µs (rozcieńczony), **2 × 2 µs (PEG)**, 2 × 4 µs (białka, reżimy HC/LC wybrane z 10 konfiguracji startowych). Niepewność z różnicy dwóch replik.

**Wyniki dotyczące Na+:**
- Liczba koordynacyjna kationów ≤ 6 Å od RNA: **41 jonów (~80% neutralizacji) w roztworze rozcieńczonym i „podobnie” w PEG.** W białkach neutralizacja spada do ~31%. Abstrakt podaje spadek kondensacji Na+ o „nearly 60%”, dyskusja o „~40%”: **niespójność wewnętrzna**.
- „PEG exhibits limited association with Na+, consistent with its inert and neutral character” (Supp. Fig. S17). Główny tekst nie podaje liczby (brak κ, brak CN Na+–O eteru).
- RDF Na+ względem fosforanów i bruzd w PEG takie jak bez crowdera. Białka sekwestrują Na+ i bezpośrednio kontaktują RNA (Arg/Lys).
- PEG kontaktuje bezpośrednio rybozy (O2′, O4′) i odsłonięte zasady w pętlach. Liczba cząsteczek wody w promieniu 3.5 nm od RNA spada o 14% (PEG) i 16% (białka). RNA kompaktuje się w PEG, a ekspanduje w białkach. Dyfuzja RNA spada 8× w PEG (158 → 19 µm²/s).

**Znaczenie i solidność dla pytania o Na+:**
1. *Przemawia przeciw dużemu κ.* Przy κ = 0.49 M⁻¹/monomer (Bielec) i 3.7 M EO ok. 65% Na+ byłoby związane z PEG [nasze przeliczenie]. Tak duży efekt byłby widoczny nawet w surowym polu siłowym, a autorzy raportują „limited association”. To jakościowy argument przeciw sile kompleksowania postulowanej przez Bielca.
2. *Ale słabo rozstrzyga.* (a) GAFF/RESP + JC + TIP4P są niepolaryzowalne i nie były kalibrowane na asocjację Na+–polieter w wodzie. Wynik jest więc predykcją pola siłowego, nie pomiarem. (b) Liczba jonów skondensowanych w odległości ≤ 6 Å jest z natury mało czuła na aktywność jonów w objętości (kondensacja Manninga), a ΔG hybrydyzacji zależy od różnicy ln a_Na± między ss i ds. Niezmieniona CN nie wyklucza zmiany wolnego Na+ o kilkadziesiąt %. (c) Badany jest jeden PEG (6000), jedno stężenie (~14 wt%, poniżej naszego odwrócenia przy 6–7 mol EO/kg), 150 mM NaCl (u nas 35 mM), 310 K. Układ to jednocząsteczkowa spinka RNA, bez hybrydyzacji i bez energii swobodnej. (d) Tylko 2 repliki po 2 µs, a jony wokół PEG 6000 relaksują się wolno.
3. *Dla obrazu Knowlesa:* bezpośrednie kontakty PEG z rybozami i zasadami w pętlach oraz kompaktowanie RNA są jakościowo zgodne z modelem „preferencyjne oddziaływanie + objętość wyłączona”, a nie z modelem „czysto steryczny PEG”. Autorzy jednak nazywają PEG „predominantly steric”.

**Wniosek Q7.** Praca jest prawdziwa i rzetelnie opisana. Można ją przytaczać jako *symulacyjne* wskazanie, że w polu siłowym AMBER/GAFF PEG 6000 nie zmienia atmosfery Na+ wokół RNA przy 150 mM NaCl, ale **nie jako dowód** przeciw (ani za) kompleksowaniu Na+ przez PEG. Do naszego pytania potrzebne jest porównanie z danymi doświadczalnymi: nasze DOSY, μ3,4 Knowlesa (VPO), ewentualnie ISE.

---

## Bibliografia

Format: autorzy, rok, tytuł, czasopismo tom, strony. DOI. [weryfikacja; dostęp]

1. Nakano S., Karimata H., Ohmichi T., Kawakami J., Sugimoto N. (2004) The Effect of Molecular Crowding with Nucleotide Length and Cosolute Structure on DNA Duplex Stability. J. Am. Chem. Soc. 126, 14330–14331. DOI 10.1021/ja0463029 [Crossref OK; pełny tekst + SI, lokalnie]
2. Karimata H., Nakano S., Sugimoto N. (2007) Effects of Polyethylene Glycol on DNA Duplex Stability at Different NaCl Concentrations. Bull. Chem. Soc. Jpn. 80, 1987–1994. DOI 10.1246/bcsj.80.1987 [Crossref OK; tylko abstrakt]
3. Karimata H., Nakano S., Ohmichi T., Kawakami J., Sugimoto N. (2004) Stabilization of a DNA duplex under molecular crowding conditions of PEG. Nucleic Acids Symp. Ser. 48, 107–108. DOI 10.1093/nass/48.1.107 [Crossref OK; tylko abstrakt]
4. Nakano S., Wu L., Oka H., Karimata H.T., Kirihata T., Sato Y., Fujii S., Sakai H., Kuwahara M., Sawai H., Sugimoto N. (2008) Conformation and the sodium ion condensation on DNA and RNA structures in the presence of a neutral cosolute as a mimic of the intracellular media. Mol. BioSyst. 4, 579–588. DOI 10.1039/B718806D [Crossref OK; tylko abstrakt]
5. Spink C.H., Chaires J.B. (1999) Effects of Hydration, Ion Release, and Excluded Volume on the Melting of Triplex and Duplex DNA. Biochemistry 38, 496–508. DOI 10.1021/bi9820154 [Crossref OK; tylko abstrakt]
6. Spink C.H., Chaires J.B. (1995) Selective Stabilization of Triplex DNA by Poly(ethylene glycols). J. Am. Chem. Soc. 117, 12887–12888. DOI 10.1021/ja00156a038 [Crossref OK; nie czytano, tylko opis wtórny w Nakano 2004]
7. Goobes R., Kahana N., Cohen O., Minsky A. (2003) Metabolic Buffering Exerted by Macromolecular Crowding on DNA–DNA Interactions: Origin and Physiological Significance. Biochemistry 42, 2431–2440. DOI 10.1021/bi026775x [Crossref OK; tylko abstrakt]
8. Markarian M.Z., Schlenoff J.B. (2010) Effect of Molecular Crowding and Ionic Strength on the Isothermal Hybridization of Oligonucleotides. J. Phys. Chem. B 114, 10620–10627. DOI 10.1021/jp103213w [Crossref OK; pełny tekst, manuskrypt PMC2925268]
9. Knowles D.B., LaCroix A.S., Deines N.F., Shkel I., Record M.T. Jr. (2011) Separation of preferential interaction and excluded volume effects on DNA duplex and hairpin stability. Proc. Natl. Acad. Sci. USA 108, 12699–12704. DOI 10.1073/pnas.1103382108 [Crossref OK; pełny tekst + SI, lokalnie]
10. Ghosh S., Takahashi S., Ohyama T., Endoh T., Tateishi-Karimata H., Sugimoto N. (2020) Nearest-neighbor parameters for predicting DNA duplex stability in diverse molecular crowding conditions. Proc. Natl. Acad. Sci. USA 117, 14194–14201. DOI 10.1073/pnas.1920886117 [Crossref OK; pełny tekst + SI, lokalnie]
11. Ghosh S., Takahashi S., Banerjee D., Ohyama T., Endoh T., Tateishi-Karimata H., Sugimoto N. (2023) Nearest-neighbor parameters for the prediction of RNA duplex stability in diverse in vitro and cellular-like crowding conditions. Nucleic Acids Res. 51, 4101–4111. DOI 10.1093/nar/gkad020 [Crossref OK; pełny tekst, PMC10201447]
12. Adams M.S., Znosko B.M. (2019) Thermodynamic characterization and nearest neighbor parameters for RNA duplexes under molecular crowding conditions. Nucleic Acids Res. 47, 3658–3666. DOI 10.1093/nar/gkz019 [Crossref OK; pełny tekst, PMC6468164]
13. Nakano S., Yamaguchi D., Tateishi-Karimata H., Miyoshi D., Sugimoto N. (2012) Hydration Changes upon DNA Folding Studied by Osmotic Stress Experiments. Biophys. J. 102, 2808–2817. DOI 10.1016/j.bpj.2012.05.019 [Crossref OK; tylko abstrakt]
14. Miyoshi D., Sugimoto N. (2008) Molecular crowding effects on structure and stability of DNA. Biochimie 90, 1040–1051. DOI 10.1016/j.biochi.2008.02.009 [Crossref OK; nie czytano]
15. Nakano S., Miyoshi D., Sugimoto N. (2014) Effects of Molecular Crowding on the Structures, Interactions, and Functions of Nucleic Acids. Chem. Rev. 114, 2733–2758. DOI 10.1021/cr400113m [Crossref OK; nie czytano]
16. Baltierra-Jasso L.E., Morten M.J., Laflör L., Quinn S.D., Magennis S.W. (2015) Crowding-Induced Hybridization of Single DNA Hairpins. J. Am. Chem. Soc. 137, 16020–16023. DOI 10.1021/jacs.5b11829 [Crossref OK; pełny tekst (bez SI)]
17. Dupuis N.F., Holmstrom E.D., Nesbitt D.J. (2014) Molecular-crowding effects on single-molecule RNA folding/unfolding thermodynamics and kinetics. Proc. Natl. Acad. Sci. USA 111, 8464–8469. DOI 10.1073/pnas.1316039111 [Crossref OK; pełny tekst (bez SI)]
18. Sung H.-L., Nesbitt D.J. (2021) Effects of Molecular Crowders on Single-Molecule Nucleic Acid Folding: Temperature-Dependent Studies Reveal True Crowding vs Enthalpic Interactions. J. Phys. Chem. B 125, 13147–13157. DOI 10.1021/acs.jpcb.1c07852 [Crossref OK; tylko abstrakt]
19. Sung H.-L., Sengupta A., Nesbitt D.J. (2021) Smaller molecules crowd better: Crowder size dependence revealed by single-molecule FRET studies and depletion force modeling analysis. J. Chem. Phys. 154, 155101. DOI 10.1063/5.0045492 [Crossref OK; tylko abstrakt]
20. Martini D., Anderson J., Garcia G., Cannon B. (2025) Contrasting Effects of Molecular Crowding on Trinucleotide-Repeat DNA Hairpins. J. Phys. Chem. B 129, 11490–11499. DOI 10.1021/acs.jpcb.5c06024 [Crossref OK; tylko abstrakt]
21. Hudoba M.W., Luo Y., Zacharias A., Poirier M.G., Castro C.E. (2017) Dynamic DNA Origami Device for Measuring Compressive Depletion Forces. ACS Nano 11, 6566–6573. DOI 10.1021/acsnano.6b07097 [Crossref OK; pełny tekst, lokalnie]
22. Kilburn D., Roh J.H., Guo L., Briber R.M., Woodson S.A. (2010) Molecular Crowding Stabilizes Folded RNA Structure by the Excluded Volume Effect. J. Am. Chem. Soc. 132, 8690–8696. DOI 10.1021/ja101500g [Crossref OK; pełny tekst, manuskrypt PMC2906142]
23. Gu X., Nguyen M.-T., Overacre A., Seaton S., Schroeder S.J. (2013) Effects of Salt, Polyethylene Glycol, and Locked Nucleic Acids on the Thermodynamic Stabilities of Consecutive Terminal Adenosine Mismatches in RNA Duplexes. J. Phys. Chem. B 117, 3531–3540. DOI 10.1021/jp312154d [Crossref OK; pełny tekst lokalnie, przejrzany]
24. Bielec K., Kowalski A., Bubak G., Witkowska Nery E., Hołyst R. (2022) Ion Complexation Explains Orders of Magnitude Changes in the Equilibrium Constant of Biochemical Reactions in Buffers Crowded by Nonionic Compounds. J. Phys. Chem. Lett. 13, 112–117. DOI 10.1021/acs.jpclett.1c03596 [Crossref OK; pełny tekst + SI, lokalnie]
25. Henning-Knechtel A., Brnović M., He W., Kirmizialtin S. (2026) Protein crowders remodel RNA electrostatics, hydration, and dynamics: a challenge to steric crowding models. Nucleic Acids Res. 54, gkag752. DOI 10.1093/nar/gkag752 [Crossref OK; pełny tekst, PMC13425243 + preprint 10.1101/2025.09.25.678182, Crossref OK]
26. Silva L.G., Weber G. (2026) Effects of Polyethylene Glycol on the Hydrogen Bonds and Stacking Interactions of DNA/DNA, DNA/RNA, and RNA/RNA Calculated by a Mesoscopic Model. ACS Omega 11, 29787–29795. DOI 10.1021/acsomega.6c00104 [Crossref OK; pełny tekst, PMC13216965]
27. Kowalski A., Bielec K., Bubak G., Żuk P.J., Czajkowski M., Sashuk V., Huck W.T.S., Antosiewicz J.M., Hołyst R. (2022) Effective screening of Coulomb repulsions in water accelerates reactions of like-charged compounds by orders of magnitude. Nat. Commun. 13, 6451. DOI 10.1038/s41467-022-34182-z [Crossref OK; pełny tekst]
28. Kowalski A.K., Piros …, Hołyst R. (2025) Microfluidic-based platform for the semi-automated study of concentration, diffusion, and equilibrium binding of biomolecules. Chem. Eng. J. 524, 169512. DOI 10.1016/j.cej.2025.169512 [Crossref OK; pełny tekst (manuskrypt)]
29. Kwapiszewska K. (2024) Physicochemical Perspective of Biological Heterogeneity. ACS Phys. Chem. Au 4, 314–321. DOI 10.1021/acsphyschemau.3c00079 [Crossref OK; pełny tekst]
30. Karpińska A., Zgorzelska A., Kwapiszewska K., Hołyst R. (2022) Entanglement of polymer chains in hypertonic medium enhances the delivery of DNA and other biomacromolecules into cells. J. Colloid Interface Sci. 627, 270–282. DOI 10.1016/j.jcis.2022.07.040 [Crossref OK; nie czytano]
31. Karoń S., Drozd M., Malinowska E. (2024) A Careful Insight into DDI-Type Receptor Layers on the Way to Improvement of Click-Biology-Based Immunosensors. Biosensors 14, 136. DOI 10.3390/bios14030136 [Crossref OK; pełny tekst, kontekst cytowania]
32. Bai … (2025) Photodynamic gel-bombs enhance tumor penetration and downstream synergistic therapies. Signal Transduct. Target. Ther. 10, 94. DOI 10.1038/s41392-025-02186-y [Crossref OK; pełny tekst, kontekst cytowania]
33. Pilarski A., Meints G.A. (2026) Backbone equilibrium in mismatched DNA influenced by solution conditions. Biophys. Chem. 329, 107548. DOI 10.1016/j.bpc.2025.107548 [Crossref OK; tylko abstrakt]
34. Shevtsov … (2026) Regenerable CO2 sorbents based on potassium carbonate and carbon nanotubes. Kataliz v promyshlennosti 26, 3–17. DOI 10.18412/1816-0387-2026-2-3-17 [Crossref OK; nie czytano]
35. Pilarski & Meints, preprint SSRN, DOI 10.2139/ssrn.5445680 — **NIEZWERYFIKOWANE** (Crossref nie zwraca metadanych; widoczne tylko w OpenAlex)
