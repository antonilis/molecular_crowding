# Wykluczanie soli z otoczki PEG (μ_x > 0): jakie inne zjawiska to tłumaczy?

Data: 2026-10-11. Kontynuacja `analysis/literature_check/notes/Q2_peg_salt.md` (tam: Knowles 2011/2015 VPO,
Tan 2008, McAfee 2014/2015, Morales ISE, Florin 1984, Ataman & Boucher 1982, nanopory). Nie powtarzam
tamtych opisów, tylko odsyłam do nich jako „Q2”.

Konwencja jak w Q2: μ_x ≡ (∂μ_sól/∂m_EO)_{m_sól}. μ_x > 0 oznacza wykluczanie soli (preferencyjną hydratację PEG).
Wartości referencyjne z VPO (Knowles 2015, Tab. 2B, na grupę EO): Na⁺ +4.7, Cl⁻ +37.6, HPO₄²⁻ +234 cal mol⁻¹ molal⁻¹.
Daje to μ_x(NaCl) ≈ +42 cal, a μ_x(Na₂HPO₄) ≈ +243 cal.

Weryfikacja: wszystkie DOI sprawdzono w Crossref API (`crossref.py doi …`) 2026-10-11 (tytuł, autorzy i rok się zgadzają).
**[PT]** oznacza przeczytany pełny tekst (lub jego odpowiednie sekcje), **[ABS]** tylko abstrakt (OpenAlex, Europe PMC,
Semantic Scholar lub strona wydawcy), **[META]** same metadane Crossref. Pobrane pliki leżą w
`literature/review_2026-10/salt_exclusion/`. Pliki `*_blocked.html` i `epmc_error_*.json` to strony blokady
(Cloudflare lub OUP) albo błędy API, a nie artykuły. Zostawiłem je, bo nie wolno było nic kasować.
PMC (pmc.ncbi.nlm.nih.gov) zwracało stronę weryfikacji przeglądarki, a pubs.acs.org, pubs.rsc.org, sciencedirect.com
i mdpi.com zwracały 403 lub Cloudflare. Pełne teksty czytałem więc tylko z Europe PMC XML i z repozytoriów.

Skala statusu:
- **ustalone**: wykluczanie soli jest standardowym wyjaśnieniem w literaturze;
- **jedno z kilku**: współistnieje z innymi mechanizmami;
- **sporne**: istnieją prace wprost przeczące;
- **nie ten mechanizm**: zjawisko z PEG i solą, ale rola soli jest inna (np. ekranowanie).

---

## 1. Nieaddytywne ciśnienie osmotyczne i aktywność wody (VPO, izopiestyka, potencjał wodny)

(a) Gdy sól jest wykluczana z otoczki PEG, ta sama woda jest „zużywana” dwa razy. Woda w otoczce hydratacyjnej PEG nie jest
dostępna dla soli, więc efektywna molalność soli w pozostałej wodzie rośnie. W efekcie osmolalność mieszaniny PEG + sól jest
większa niż suma osmolalności roztworów dwuskładnikowych (ΔOsm > 0), a to jest wprost definicja μ_x > 0
(Knowles 2011, równ. S1–S2). To nie jest zjawisko „wyjaśniane” przez μ_x, tylko jego bezpośredni pomiar.
(b) Michel (1983): VPO dla PEG 8000 z czterema solami. Synergizm NaCl–PEG (potencjał wodny niższy niż addytywny)
rośnie liniowo ze stężeniem soli dodanej do PEG (abstrakt). Liczb nie odczytałem, bo pełny tekst był zablokowany.
Knowles 2011 i 2015 oraz Sadeghi & Ziamajidi 2006 są omówione w Q2 (A1, A2, A5).
(c) **Ustalone**. Znak jest zgodny z μ_x > 0 i to jest ta sama wielkość co w VPO.
(d)
- Michel B.E. „Evaluation of the Water Potentials of Solutions of Polyethylene Glycol 8000 Both in the Absence and Presence of Other Solutes”. *Plant Physiol.* 72, 66–70 (1983). DOI 10.1104/pp.72.1.66. [ABS]
- Knowles D.B. et al. *PNAS* 108, 12699–12704 (2011). DOI 10.1073/pnas.1103382108. [PT] (Q2)
- Sadeghi R., Ziamajidi F. *J. Chem. Thermodyn.* 38, 1335–1343 (2006). DOI 10.1016/j.jct.2006.02.003. [META] (Q2)

## 2. Synergiczne obniżenie temperatury krzepnięcia i przesunięcie eutektyku PEG–woda

(a) Obniżenie temperatury krzepnięcia jest miarą aktywności wody (punkt 1), mierzoną w innej temperaturze inną metodą.
Jeśli jony silnie hydratowane „odbierają” wodę z otoczki PEG (sól jest wykluczana), obniżenie dla PEG + sól
przekracza sumę efektów składowych. Jony słabo hydratowane, które asocjują z PEG (SCN⁻), dają addytywność.
W pomiarach DSC eutektyku dodatek NaCl obniża likwidus wody, podwyższa likwidus PEG (wysalanie)
i obniża temperaturę eutektyczną. Autorzy dopasowali model Flory'ego–Hugginsa i przypisują ten efekt deplecji NaCl wokół PEG
(odpychanie od obrazu ładunku w ośrodku o niskiej przenikalności).
(b) Sørensen & Thormann 2026: NaF, NaCl, NaI dają obniżenie „exceeding the sum of the individual PEG and salt
contributions”, a NaSCN „nearly additive” (abstrakt). Liczb nie mam. Gosh et al. 2009 [PT], PEG 35k (Tab. 2):
T_eutektyczna = 264, 262, 259, 250 K przy 0, 1, 3, 5 wt% NaCl. Dopasowany parametr χ_a = χ(sól–woda) − χ(sól–PEG) < 0,
czyli „NaCl is more attracted by water than PEG”.
(c) **Ustalone** (jedna praca z 2026 r. i klasyczna DSC). Szereg jest zgodny z Hofmeisterem: NaI też daje synergizm,
a tylko SCN⁻ nie. Zgadza się to z Q2: μ_x < 0 dla anionów najbardziej chaotropowych (SCN⁻, ClO₄⁻).
(d)
- Sørensen I.E., Thormann E. „Quantifying Hofmeister Effects on Polymer–Water Interactions through Freezing Point Depression”. *Langmuir* 42, 6261–6267 (2026). DOI 10.1021/acs.langmuir.5c05841. [ABS]
- Gosh R.C., Toda A., Tanaka S. „The effect of NaCl on the eutectic phase behavior of aqueous poly(ethylene glycol) solutions”. *Polymer* 50, 1304–1310 (2009). DOI 10.1016/j.polymer.2008.12.044. [PT] (rękopis z repozytorium Hiroshima, `gosh2009_polymer50_1304.pdf`)

## 3. Obniżanie punktu zmętnienia (LCST) PEO przez sole, szereg Hofmeistera

(a) Gdy łańcuchy PEG się stykają, nakładające się strefy zubożone w sól oddają wodę i jony do roztworu.
Przy μ_x > 0 obniża to energię swobodną stanu zagregowanego, więc rozdział faz następuje w niższej temperaturze.
Model Florina et al. zakłada wprost „a zone with decreased salt concentration” wokół łańcucha. Siła efektu zależy głównie
od anionu (kosmotropy obniżają T_cp najsilniej). Bailey & Callard (1959) opisują już „salting-out effects”
PEO o wydajności zależnej od soli „similar to … small neutral solute molecules”.
(b) Madeira et al. 2025 [PT]: T_cp maleje liniowo z molalnością soli (T_cp = T⁰ − α[sól]) dla 8 soli sodowych
(m.in. Na₃PO₄, Na₂SO₄, Na₂CO₃) przy 0.5–30 wt% PEG. ln α koreluje liniowo z entropią strukturalną hydratacji anionu.
Liczbowych nachyleń α nie podaję, bo sens jednostek ΔS w równ. (2) nie jest jednoznaczny w tekście.
Florin 1984 nie podaje liczb w abstrakcie.
(c) **Ustalone jako fenomenologia, mechanizm molekularny dyskutowany.** Madeira et al. piszą, że mechanizm wysalania
„remains unknown”, i proponują opis entropią struktury wody. Na poziomie termodynamiki makroskopowej obie interpretacje
wymagają jednak tego samego znaku współczynnika krzyżowego (μ_x > 0). Przeciwko prostemu obrazowi przemawia fakt,
że PEG tworzy ATPS także z NaClO₄ (Madeira cytuje [34,35]), a Q2 (Morales 2020) daje dla NaClO₄ μ_x < 0 przy
niskim stężeniu. Przy wysokich stężeniach znak może się więc zmieniać, a samo tworzenie dwóch faz nie dowodzi μ_x > 0.
(d)
- Bailey F.E. Jr., Callard R.W. „Some properties of poly(ethylene oxide) in aqueous solution”. *J. Appl. Polym. Sci.* 1, 56–62 (1959). DOI 10.1002/app.1959.070010110. [ABS]
- Florin E., Kjellander R., Eriksson J.C. „Salt effects on the cloud point of the poly(ethylene oxide)+water system”. *J. Chem. Soc., Faraday Trans. 1* 80, 2889 (1984). DOI 10.1039/F19848002889. [ABS]
- Ataman M. „Properties of aqueous salt solutions of poly(ethylene oxide). Cloud points, θ temperatures”. *Colloid Polym. Sci.* 265, 19–25 (1987). DOI 10.1007/BF01422658. [META] (oraz Ataman & Boucher 1982, Q2)
- Madeira P.P., Uversky V.N., Zaslavsky B.Y. „Effects of Inorganic Salts on Phase Separation in Aqueous Solutions of Poly(ethylene glycol)”. *Int. J. Mol. Sci.* 26, 4545 (2025). DOI 10.3390/ijms26104545. [PT] (Europe PMC XML, `madeira2025_ijms26_4545.xml`)

## 4. Wodne układy dwufazowe (ATPS) PEG–fosforan, siarczan, cytrynian

(a) To skrajna, wysokostężeniowa wersja punktu 3. Przy dużym dodatnim μ_x (fosforan, siarczan, cytrynian)
koszt współistnienia PEG i soli w jednej fazie jest na tyle duży, że układ rozpada się na fazę bogatą w PEG
i fazę bogatą w sól. NaCl, z ~6× mniejszym μ_x, sam ATPS nie tworzy. da Silva & Loh: siłą napędową jest wzrost
entropii, a NaCl „does not induce phase separation”, w odróżnieniu od Na₂SO₄ i Li₂SO₄.
(b) Huddleston 2003: binodale PEG 1000–3400 + K₃PO₄, K₂CO₃, (NH₄)₂SO₄ itd. w 25 °C. Liczb nie odczytałem.
Binodale leżą przy ≳0.5 m soli (Q2 b).
(c) **Fenomen ustalony, mechanizm sporny.** Bonnassieux et al. 2024 (NMR, Raman, mikrokalorymetria, PEG/K₂HPO₄)
nie widzą różnicy w strukturze solwatacji wody między fazami i odrzucają dużą zmianę entalpii. Raportują przy tym,
że „PEG interacts with salt ions in the PEG-rich solution” (przesunięcie ¹H NMR). Madeira 2025 uważa mechanizm za
otwarty. Wykluczanie soli jest więc najprostszym opisem termodynamicznym, ale nie jest jednoznacznie potwierdzone na poziomie molekularnym.
(d)
- Ananthapadmanabhan K.P., Goddard E.D. *Langmuir* 3, 25–31 (1987). DOI 10.1021/la00073a005. [META] (Q2)
- Zafarani-Moattar M.T., Sadeghi R. (PEG + NaH₂PO₄/Na₂HPO₄). *Fluid Phase Equilib.* 181, 95–112 (2001). DOI 10.1016/S0378-3812(01)00373-9. [META] (Q2)
- Huddleston J.G., Willauer H.D., Rogers R.D. *J. Chem. Eng. Data* 48, 1230–1236 (2003). DOI 10.1021/je034042p. [ABS]
- da Silva L.H.M., Loh W. „Calorimetric Investigation of the Formation of Aqueous Two-Phase Systems in Ternary Mixtures of Water, Poly(ethylene oxide) and Electrolytes (Or Dextran)”. *J. Phys. Chem. B* 104, 10069–10073 (2000). DOI 10.1021/jp000719t. [ABS]
- Bonnassieux S., Pandya R., Skiba D.A., … Gallant B.M., Grimaud A. „Revisiting the driving force inducing phase separation in PEG–phosphate aqueous biphasic systems”. *Faraday Discuss.* 253, 181–192 (2024). DOI 10.1039/D4FD00058G. [ABS] (pełny abstrakt z Semantic Scholar; PDF OA zablokowany przez RSC)

## 5. Krystalizacja soli przez „wysalanie” PEG-iem (drowning-out)

(a) Skoro PEG podnosi współczynnik aktywności soli (μ_x > 0), to przy stałej aktywności nasycenia
spada rozpuszczalność soli, a sól wykrystalizowuje. Jest to odwrotność wysalania PEG przez sól, bo z relacji Maxwella
(∂μ_sól/∂m_PEG) = (∂μ_PEG/∂m_sól). Przemysłowo stosuje się to do otrzymywania bezwodnego Na₂SO₄ i Na₂CO₃.
Powstają przy tym trzy fazy: kryształy, ciecz bogata w PEG i ciecz bogata w sól.
(b) Taboada 2000: układ woda–Na₂SO₄–PEG 3350, 28 °C. Taboada 2004: przy 313 K jedna faza jest „free of sodium carbonate”
(wolna od soli), a druga wolna od PEG. Krzywych rozpuszczalności liczbowo nie odczytałem.
(c) **Ustalone jako technologia.** Wyjaśnienie przez wysalanie jest naturalne, ale prace inżynierskie nie analizują μ_x.
Jest to zakres wysokich stężeń.
(d)
- Taboada M.E., Graber T.A., Asenjo J.A., Andrews B.A. „Drowning-out crystallisation of sodium sulphate using aqueous two-phase systems”. *J. Chromatogr. B* 743, 101–105 (2000). DOI 10.1016/S0378-4347(00)00110-9. [ABS]
- Taboada M.E., Rocha O.A., Graber T.A., Andrews B.A. „Liquid−Liquid and Solid−Liquid Equilibria of the Poly(ethylene glycol) + Sodium Sulfate + Water System at 298.15 K”. *J. Chem. Eng. Data* 46, 308–311 (2001). DOI 10.1021/je000242e. [ABS]
- Taboada M.E., Graber T.A., Cisternas L.A. „Sodium Carbonate Extractive Crystallization with Poly(ethylene glycol) Equilibrium Data and Conceptual Process Design”. *Ind. Eng. Chem. Res.* 43, 835–838 (2004). DOI 10.1021/ie030692a. [ABS]

## 6. Dyfuzjoforeza PEG w gradiencie soli i „osmotyczna dyfuzja” soli w gradiencie PEG

(a) Gdy sól jest wykluczana z otoczki, μ_PEG rośnie ze stężeniem soli. PEG wędruje więc z obszaru o wysokim
stężeniu soli do niskiego (D̂_PS > 0), a sól z obszaru o wysokim stężeniu PEG do niskiego. Oba współczynniki krzyżowe
macierzy dyfuzji są proporcjonalne do nadmiaru wody ν_W. Dla NaSCN i KSCN znak się odwraca (wiązanie anionu z PEG).
(b) Annunziata 2024 [PT], PEG 20k, 25 °C, interferometria Rayleigha. ν_W = 3500 (**7.7 H₂O/EO**) dla Na₂SO₄,
1100 (**2.4/EO**) dla NaCl, 990 (2.2/EO) dla KCl, −550 (−1.2/EO) dla NaSCN i −690 (−1.5/EO) dla KSCN.
Cytat: „D̂_PS data are positive, thereby implying that PEG diffusiophoresis occurs from high to low salt
concentrations”. Zamiana kationu (Na⁺ → K⁺) prawie nic nie zmienia.
(c) **Ustalone** (grupa Annunziaty, walidacja izopiestyczna w Tan 2008, Q2). Znak i rząd wielkości są zgodne z VPO (patrz niżej).
(d)
- Annunziata O. „Diffusiophoresis of Macromolecules within the Framework of Multicomponent Diffusion”. *Molecules* 29, 1367 (2024). DOI 10.3390/molecules29061367. [PT] (Europe PMC XML, `annunziata2024_review_PMC10974165.xml`)
- McAfee M.S., Zhang H., Annunziata O. *Langmuir* 30, 12210–12219 (2014). DOI 10.1021/la503214b. [ABS] (Q2)
- Tan C., Albright J.G., Annunziata O. *J. Phys. Chem. B* 112, 4967–4974 (2008). DOI 10.1021/jp711725e. [ABS] (Q2)

## 7. Wysalanie i oddziaływania PEG–sól w symulacjach i modelach

(a) W atomistycznych symulacjach ogólnego, nienaładowanego polimeru, bez specyficznych oddziaływań polimer–anion,
odtworzono deplecję silnie hydratowanego SO₄²⁻ i akumulację SCN⁻ przy łańcuchu oraz nieaddytywne efekty w mieszaninach soli.
Wynika z tego, że wykluczanie kosmotropowych anionów jest właściwością ogólną (oddziaływania jon–woda w objętości),
a nie cechą chemiczną PEG. Gosh 2009 podaje osobny argument: niska przenikalność PEG, czyli odpychanie od obrazu ładunku.
(b) Liczb dla PEG nie ma, bo model jest generyczny.
(c) **Jedno z kilku** (wsparcie mechanistyczne, nie pomiar).
(d)
- Goel K., Choudhary M., Bharadwaj S. „Non-additive ion effects on the coil–globule equilibrium of a generic polymer in aqueous salt solutions”. *J. Chem. Phys.* 165 (2026). DOI 10.1063/5.0336301. [ABS] (preprint arXiv 2603.26555, niepobrany)

## 8. Elektrolity „molecular crowding” w bateriach wodnych

(a) PEG (lub PEGDME) w dużym stężeniu obniża aktywność wody, co poszerza okno elektrochemiczne do 3.2 V przy zaledwie 2 m LiTFSI.
Mechanizm podawany przez autorów to wiązanie wody przez PEG (zmiana sieci wiązań wodorowych), a nie wykluczanie soli.
Łączy się to z punktem 1 przez konkurencję PEG i jonów o wodę. TFSI⁻ jest jednak anionem silnie chaotropowym,
więc znak μ_x dla tego układu nie musi być dodatni.
(b) Xie 2020: 2 m LiTFSI, okno 3.2 V. Dong 2021: PEGDME 450 daje przewodnictwo 2.4 mS/cm wobec 0.8 dla PEG 400.
(c) **Nie ten mechanizm** (pokrewny: obniżona a_w). Nie należy przytaczać tego jako dowodu wykluczania soli.
(d)
- Xie J., Liang Z., Lu Y.-C. „Molecular crowding electrolytes for high-voltage aqueous batteries”. *Nat. Mater.* 19, 1006–1011 (2020). DOI 10.1038/s41563-020-0667-y. [ABS] (Europe PMC)
- Dong D., Xie J., Liang Z., Lu Y.-C. „Tuning Intermolecular Interactions of Molecular Crowding Electrolyte for High-Performance Aqueous Batteries”. *ACS Energy Lett.* 7, 123–130 (2021/2022). DOI 10.1021/acsenergylett.1c02064. [ABS]

## 9. Kondensacja ψ DNA i strącanie DNA przez PEG + NaCl

(a) Neutralny PEG z solą powyżej progu kompaktuje DNA (przejście kłębek–globula) lub je strąca. Krytyczne stężenie PEG
maleje ze wzrostem stężenia soli i z masą PEG. W teorii Vasilevskiej et al. rola soli to **ekranowanie** ładunku DNA
(efekt polielektrolitowy) plus redystrybucja PEG, a PEG–DNA jest odpychaniem (Lerman: „nonspecifically repulsive”).
Wykluczanie soli z PEG nie występuje w tych modelach. Mogłoby dodatkowo podnosić a± soli i wzmacniać ekranowanie,
ale tego nikt nie wydzielił.
(b) Lis & Schleif: zależność od PEG i NaCl zbadana, liczby tylko w pełnym tekście (zablokowany).
(c) **Nie ten mechanizm, o ile nie zostanie wykazane inaczej.** Dominuje ekranowanie i deplecja.
(d)
- Lerman L.S. „A Transition to a Compact Form of DNA in Polymer Solutions”. *PNAS* 68, 1886–1890 (1971). DOI 10.1073/pnas.68.8.1886. [ABS]
- Vasilevskaya V.V., Khokhlov A.R., Matsuzawa Y., Yoshikawa K. „Collapse of single DNA molecule in poly(ethylene glycol) solutions”. *J. Chem. Phys.* 102, 6595–6602 (1995). DOI 10.1063/1.469375. [ABS]
- Lis J.T., Schleif R. „Size fractionation of double-stranded DNA by precipitation with polyethylene glycol”. *Nucleic Acids Res.* 2, 383–390 (1975). DOI 10.1093/nar/2.3.383. [ABS]

## 10. Strącanie i krystalizacja białek oraz wirusów PEG-iem w obecności soli

(a) Sól w mieszankach krystalizacyjnych PEG + sól tłumaczy się zwykle ekranowaniem ładunku białka, a PEG deplecją
(Gosh 2009, wstęp). Bončina: NaCl obniża rozpuszczalność lizozymu z PEG i bez PEG, „explained by the screening”.
Efekty jonowo-specyficzne zależą od przeciwjonu białka, a szereg anionów jest odwrócony. Yamamoto 1970: strącanie
fagów PEG-iem jest „relatively insensitive to … ionic strength”.
(b) Bončina 2008, pH 4, kolejność anionów: F⁻ < H₂PO₄⁻ < Cl⁻ < Br⁻ (odwrócony szereg Hofmeistera).
(c) **Nie ten mechanizm.** Wykluczanie soli z PEG nie jest tu wyjaśnieniem w literaturze. Gosh 2009 wprost zauważa,
że oddziaływanie PEG–sól w krystalizacji białek „received little attention”.
(d)
- Bončina M., Reščič J., Vlachy V. „Solubility of Lysozyme in Polyethylene Glycol-Electrolyte Mixtures: The Depletion Interaction and Ion-Specific Effects”. *Biophys. J.* 95, 1285–1294 (2008). DOI 10.1529/biophysj.108.128694. [ABS]
- Yamamoto K.R., Alberts B.M., Benzinger R., Lawhorne L., Treiber G. *Virology* 40, 734–744 (1970). DOI 10.1016/0042-6822(70)90218-7. [ABS]

## 11. Technika stresu osmotycznego (PEG jako osmolit w buforach solnych)

(a) Jeśli sól jest wykluczana z PEG, ciśnienie osmotyczne roztworu PEG w buforze solnym jest większe niż suma składowych
(punkt 1). Ciśnienia osmotycznego PEG w soli nie można więc liczyć addytywnie z kalibracji w czystej wodzie, tylko trzeba je zmierzyć.
Michel 1983 pokazuje to dla PEG 8000 + NaCl. Parsegian, Rand & Rau 2000 dają ramę (Gibbs–Duhem), w której
stres osmotyczny i preferencyjne oddziaływanie są równoważne.
(b) Brak liczb (pełny tekst Michela zablokowany).
(c) **Ustalone jako fakt pomiarowy (nieaddytywność).** Nie znalazłem pracy z grupy Rau/Parsegian, która wprost
podawałaby poprawkę na wykluczanie soli; tego NIE twierdzę.
(d)
- Parsegian V.A., Rand R.P., Rau D.C. „Osmotic stress, crowding, preferential hydration, and binding: A comparison of perspectives”. *PNAS* 97, 3987–3992 (2000). DOI 10.1073/pnas.97.8.3987. [ABS]
- Michel 1983 (jak w punkcie 1).

## 12. Nanopory: sól zwiększa wchodzenie PEG do α-hemolizyny (KONTRPRZYKŁAD lub interpretacja przeciwna)

(a) W α-HL przy 1–4 M soli PEG zachowuje się „as if charged”, a jego podział do poru rośnie z KCl.
Literatura nanoporowa tłumaczy to **kompleksowaniem kationów** przez PEG (wszystkie poza Li⁺), czyli efektem przeciwnym
do wykluczania. Alternatywa (wysalanie PEG z roztworu do poru) nie została tam wydzielona. Szczegóły w Q2 (c).
(b) Rodrigues 2008 [PT, Q2]: energia oddziaływania PEG–por 0.03 kT na monomer przy 1 M i 0.27 kT przy 4 M KCl.
(c) **Sporne.** Warunki to ograniczona geometria i bardzo wysoka sól. W objętości VPO daje dla K⁺ i Na⁺ β ≈ 0 do +5 cal (Q2).
(d)
- Breton M.F. et al. *J. Phys. Chem. Lett.* 4, 2202–2208 (2013). DOI 10.1021/jz400938q. [ABS]
- Rodrigues C.G. et al. *Biophys. J.* 95, 5186–5192 (2008). DOI 10.1529/biophysj.108.140814. [PT] (Q2)

## 13. Hydrożele i membrany PEG/PEGDA: sorpcja soli (LUKA)

(a) Hipoteza: obojętne sieci PEO powinny sorbować sól słabiej, niż wynika z zawartości wody (K_s < K_w),
bez udziału efektu Donnana, z tego samego powodu co μ_x > 0. Grupa Freemana/Ganesana bada to w XLPEGDA
(LiCl, NaCl, KCl, mieszaniny soli).
(b) **NIEZWERYFIKOWANE.** Abstraktów nie udało się pobrać (Elsevier 403, brak w OpenAlex i Crossref). Semantic Scholar
zwrócił dla części 1 Janga abstrakt części 2 (o dyfuzji), czyli błędne przypisanie. Snippety z wyszukiwarki wspominają
o „polymer–ion specific interactions” i nasycaniu „ethylene oxide binding sites” dla soli kompleksujących.
Może to więc być raczej kontrprzykład niż przykład.
(c) Status nieznany. Nie cytować bez przeczytania.
(d)
- Jang E.-S., Kamcev J., Kobayashi K., Yan N., Sujanani R., Dilenschneider T.J., Park H.B., Paul D.R., Freeman B.D. „Influence of water content on alkali metal chloride transport in cross-linked Poly(ethylene glycol) Diacrylate. 1. Ion sorption”. *Polymer* 178, 121554 (2019). DOI 10.1016/j.polymer.2019.121554. [META]
- Zofchak E.S., Quigley A.R., Yoh J., Sachar H.S., Reimund K.K., Milner S.T., Freeman B.D., Ganesan V. „Molecular and electrostatic origins of mixed salt partitioning phenomena in uncharged poly(ethylene oxide)-based membranes”. *J. Membr. Sci.* 702, 122800 (2024). DOI 10.1016/j.memsci.2024.122800. [META]

## Inne sprawdzone, ale odrzucone lub poboczne
- Loo et al. 2019 Macromolecules (SANS PEO + LiTFSI) dotyczy **stopu** PEO, a nie roztworu wodnego, więc jest nieistotne.
- Kondensaty i LLPS: nie znalazłem pracy, która wiązałaby partycjonowanie PEG do kondensatów z wykluczaniem soli.
  Snippety pokazują sprzeczne wyniki (PEG wykluczany z koacerwatów vs. współkondensacja), bez analizy roli soli. Pomijam.
- Spink & Chaires 1998 (Biochemistry 38, 496; DOI 10.1021/bi9820154; [ABS]): w EG wiązanie przeciwjonów Na⁺
  przez DNA „modest decrease” przy spadku a_w. Dotyczy DNA–Na⁺, a nie PEG–sól, więc nie jest to test μ_x.

---

## Które zjawiska dają niezależną *ilościową* kontrolę μ_x (znak i wielkość)?

1. **Dyfuzja wieloskładnikowa i dyfuzjoforeza (punkt 6) to najlepszy, w pełni niezależny test.** Z ν_W na EO
   (μ_x ≈ ν_W·ν·RT·(1+ε)/55.51, przeliczenie z Q2):
   - NaCl: 2.4/EO daje **+47 do +50 cal/(mol·molal EO)**, a VPO daje +42 ± 3. Zgodność w ~15%.
   - Na₂SO₄: 7.7/EO daje ≈ **+170 do +210 cal** (dla (1+ε) = 0.70–0.85, szacunek). To ten sam rząd co Na₂HPO₄ z VPO (+243),
     zgodnie z tym, że HPO₄²⁻ i SO₄²⁻ są podobnie silnymi kosmotropami. Dla fosforanu sodu bezpośrednich danych dyfuzyjnych nie ma.
   - Znak odwraca się dla SCN⁻, zgodnie z NaClO₄ < 0 (Q2). Szereg anionów i brak wpływu kationu są zgodne z
     β(Na⁺) ≈ +5 i β(Cl⁻) ≈ +38 cal z Knowles 2015.
2. **Synergiczne obniżenie temperatury krzepnięcia (Sørensen & Thormann 2026)** to pomiar aktywności wody w innej
   temperaturze. Z pełnego tekstu da się wyliczyć μ_x(NaCl) i porównać z VPO (dostęp płatny, embargo AAM do 02.2027).
   Znak zgadza się już na poziomie abstraktu.
3. **VPO Michela 1983 i izopiestyka Lin 1996 (K-fosforany) oraz Sadeghi 2006 (NaCl)** to ta sama wielkość co Knowles,
   ale z innych laboratoriów. Wymagają odczytu tabel z pełnego tekstu. Lin 1996 to jedyne źródło dla **fosforanu**
   (potasowego) niezależne od Knowlesa.
4. **Tylko jakościowo lub półilościowo (wysokie stężenia, model-zależne):** punkt zmętnienia, ATPS, drowning-out, eutektyk
   (Gosh: χ_a < 0). Potwierdzają znak (μ_x > 0) i porządek Hofmeistera (fosforan, siarczan ≫ chlorek > jodek > SCN⁻, ClO₄⁻),
   ale ekstrapolacja do 35 mM wymaga modelu.
5. **Bez wkładu do μ_x lub z przeciwnym obrazem:** kondensacja ψ, strącanie białek i wirusów (ekranowanie), baterie
   (obniżona a_w, inny mechanizm), nanopory (kompleksowanie kationów w porze przy 1–4 M; konflikt z VPO w objętości, Q2).

Wniosek: dla **NaCl** mamy dwa niezależne pomiary ilościowe (VPO i dyfuzja) zgodne co do znaku i wielkości
(+42 do +50 cal/(mol·molal EO)) oraz trzecie, jakościowe potwierdzenie (synergizm w krzepnięciu i VPO Michela).
Dla **fosforanu sodu** ilościowa jest tylko liczba z VPO (Knowles 2015). Najbliższym niezależnym analogiem jest Na₂SO₄
z dyfuzji (≈ +170 do +210 cal), a jakościowo potwierdza to silne tworzenie ATPS i obniżanie T_cp przez fosforany.
