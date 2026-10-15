# Q3 – Teoria deplecji polimerowej: ΔG_dep(x) dla hybrydyzacji 13 bp w PEG

Skrypt: `analysis/literature_check/q3_depletion_exponents.py` (uruchamiany przez `.venv/bin/python`).
Wyniki: `q3_table.csv`, `q3_lengths.csv`, wykresy `q3_dG_dep.pdf`, `q3_local_slope.pdf`, `q3_lengths.pdf`
(wszystko w `analysis/literature_check/`). Strony książki L&T podaję jako numer drukowany / strona PDF
(przesunięcie +14).

## 0. Skrót odpowiedzi

* **Granica białkowa** (cząstka ≪ ξ, R_g): energia wstawienia jest **liniowa w stężeniu monomeru i
  nie zależy od M** ani od reżimu (rozcieńczony / półrozcieńczony). Dla kuli
  μ/kT ∝ (R/ξ)^{3−1/ν} = (R/ξ)^{4/3} (Odijk; de Gennes), a ponieważ ξ ∝ c^{−ν/(3ν−1)}, wychodzi
  μ ∝ c^{(3ν−1)/(3ν−1)} = c^1 **dokładnie**. To samo dla cienkiego pręta: μ/L ∝ r^{2−1/ν}ξ^{−(3−1/ν)} ∝ c^1.
  Wykładnik nie wynosi 1/ν ani 5/3: 1/ν opisuje zależność od R_x (rozmiaru łańcucha) w rozcieńczonym
  roztworze, a 3−1/ν ≈ 4/3 zależność od promienia cząstki.
* **Granica koloidalna** (cząstka ≫ ξ): μ = ΠV + γA, Π ∝ c^{3ν/(3ν−1)} ≈ c^{2.31} (prawo 9/4),
  γ ≈ (3/2)Πξ ∝ c^{2ν/(3ν−1)} ≈ c^{1.54}. Zatem p ∈ [1.54, 2.31].
* **Żadna wersja czystej deplecji nie daje p < 1 ani nasycenia.** |ΔG_dep| rośnie monotonicznie
  z x. W naszym zakresie (5–40 wt%) lokalne nachylenia na skali **monomolalnej** wynoszą
  p ≈ 0.64–1.6. Wartości p < 1 w granicy białkowej (≈ 0.83) biorą się wyłącznie z nieliniowości
  x = 1000w/[(1−w)·44] względem stężenia masowego, a nie z fizyki.
* Wielkość: granica białkowa daje −2 (ssDNA jako pręt) do −5.4 kJ/mol (ssDNA jako kula R = 1.5 nm)
  przy 5 wt%, czyli **tyle, ile wynosi nasze plateau**. Ta sama teoria przewiduje jednak dalszy liniowy
  wzrost do −18…−46 kJ/mol przy 40 wt%, a obserwujemy lekką destabilizację (+0.42 kJ/mol na molal
  monomeru). Sama deplecja nie wyjaśnia więc kształtu krzywej. Potrzebny jest człon dodatni,
  w przybliżeniu liniowy w x, który kompensuje deplecję już od ~5 wt%, albo deplecja jest w
  rzeczywistości dużo słabsza, niż zakładają modele sztywnych cząstek (patrz §5).

## 1. Wzory

Oznaczenia: c – stężenie monomerów (liczbowe), φ = c/c* (c* = M/[N_A·(4π/3)R_g³], tak jak w `utils.py`
i w L&T Eq. 4.11), ν – wykładnik Flory'ego (0.588 RG; projekt dla PEG używa 0.583), R_x² = ⟨R_E²⟩/3.

### 1a. Mała kula, granica białkowa (R ≪ ξ, R_x)

1. **Hanke, Eisenriegler, Dietrich 1999**, PRE 59, 6853 (pełny tekst, lokalnie `Hanke_meso.pdf`).
   Roztwór rozcieńczony, łańcuchy z wyłączoną objętością, „generalized cylinder” o wymiarze osi δ
   (kula δ = 0 → d = D = 3; nieskończony walec d = 2):
   - Eq. (1.5), s. 6854: $n_p f^{(1)}_K = n_p R^{d}\,Y_{d,D}(x)$, $x = R_x/R$
   - Eq. (1.8)–(1.9), s. 6855: $Y_{d,D}(x\to\infty) \to A_{d,D}\,x^{1/\nu}$, ważne dla $d > 1/\nu$ (spełnione dla kuli i walca w 3D, nie dla walca przy łańcuchach idealnych, gdzie zależność jest logarytmiczna)
   - stąd dla kuli: $\displaystyle \frac{\mu_\mathrm{sph}}{kT} = A_{3,3}\, n_p R_x^{1/\nu}\, R^{\,3-1/\nu}$, a dla walca na jednostkę długości: $\displaystyle \frac{\mu_\mathrm{cyl}}{kT\,L} = A_{2,3}\, n_p R_x^{1/\nu}\, r^{\,2-1/\nu}$ (2 − 1/ν ≈ 0.30, s. 6856)
   - Eq. (2.54), (2.55), s. 6865: $A_{3,3} = 9.82 \pm 0.3$, $A_{2,3} = 1.23 \pm 0.2$
   - Eq. (1.11)–(1.13), s. 6855–6856 (rozwinięcie dla małego promienia, SRE): cząstkę zastępuje się
     operatorem $1 - A_{d,D} R^{d-1/\nu} w_K$ działającym na zmodyfikowaną gęstość monomerów
     $\rho(\mathbf r) = (R_x^{1/\nu}/N)\sum_i \delta(\mathbf y_i - \mathbf r)$. Uwaga: $n_p R_x^{1/\nu} = c\,(R_x^{1/\nu}/N)$, a $R_x^{1/\nu}/N$ nie zależy od N, więc wyrażenie jest liniowe w c i niezależne od M.
   - Ważność: R, R_x ≪ długości walca l, R ≫ rozmiar monomeru, roztwór rozcieńczony (n_p R_x³ ≲ 1).
2. **Odijk 1996**, Macromolecules 29, 1842 (pełny tekst, wersja autorska z TU Delft). Roztwór
   półrozcieńczony, dobry rozpuszczalnik, $\xi = A_K^{-5/4}c_o^{-3/4}$:
   - Eq. (4): profil wokół nieskończenie małej kuli $f = 1 - a/r$, czyli warstwa zubożona ma
     grubość ~a i nie zależy od ξ
   - Eq. (5): $\displaystyle \frac{w}{kT} = \left(\frac{a}{\xi}\right)^{4/3}\quad(\xi > a)$, z prefaktorem O(1) nieznanym. Odijk zauważa, że w ∝ c_o, bo ξ^{−4/3} ∝ c_o.
   - Eq. (6): dla rozcieńczonego roztworu ta sama postać z R zamiast ξ; Eq. (9): łańcuch idealny
     $w/kT = a/R_o$ (wykładnik 1 = 3 − 1/ν przy ν = 1/2)
3. **de Gennes 1979** (C. R. Acad. Sci. B 288, 359; cytowany przez Odijka). L&T Eq. (1.28), s. 26/40:
   $W_\mathrm{dep}(h=0)/kT \simeq R/\xi$ dla R ≫ ξ oraz $\simeq (R/\xi)^{4/3}$ dla R ≪ ξ (potencjał
   kontaktowy dwóch kul, prefaktor O(1)). NIEZWERYFIKOWANE (brak DOI; znane tylko z cytowań u L&T i Odijka).
4. **Hermans 1982**, JCP 77, 2193 (pełny tekst lokalnie). Łańcuchy gaussowskie, długie:
   Eq. (26) $V_{12}/n_1 = 5\pi R_2 l^2/18$. Objętość wyłączona na segment jest ∝ R (wykładnik 3 − 1/ν = 1
   dla ν = 1/2) i nie zależy od M. Eq. (29) podaje wersję dla walca. Autor sam przyznaje, że prefaktor
   jest zaniżony 2–2.5 razy w porównaniu z Monte Carlo.
5. **Eisenriegler 2000**, JCP 113, 5091, *Small mesoscopic particles in dilute and semidilute solutions
   of nonadsorbing polymers*: DOI zweryfikowane, ale tekstu nie udało się pobrać (repozytorium FZ Jülich
   blokuje boty, strona AIP zwraca 403). To jest właściwe źródło przejścia SRE do reżimu
   półrozcieńczonego. Według L&T, s. 32/46 (o pracy [135]): B₂ małych koloidów zmienia się nieznacznie
   aż do c*, a powyżej rośnie. **Tylko tytuł i streszczenie u L&T.** Wniosek „μ = A R^{3−1/ν}⟨ρ⟩ także
   w roztworze półrozcieńczonym” jest moją interpretacją Eq. (1.11) Hanke i in. (ρ jest lokalnym
   operatorem, a ⟨ρ⟩ ∝ c w każdym reżimie). Zgadza się ona ze skalowaniem z Eq. (5) Odijka.

**Ważność dla nas.** ξ(PEG 35k) = 2.3 nm przy 10 wt% i 0.76 nm przy 40 wt% (`q3_lengths.csv`). ssDNA
jako pręt (r = 0.5 nm) jest na granicy, dsDNA (r = 1 nm) i ssDNA jako kula (R = 1.5 nm) są w
przejściu R ~ ξ powyżej ~10–20 wt%. Granica białkowa jest więc ściśle stosowalna tylko przy niskich
stężeniach. Kontrola wewnętrzna: przy 10 wt% SRE daje μ(kula 1.5 nm) = 2.6 kT, a (R/ξ)^{4/3} = 0.57.
Efektywny prefaktor Odijka wynosi więc ≈ 4.5, czyli rozsądne O(1).

### 1b. Cienki pręt / walec

- Granica białkowa: $\mu_\mathrm{rod}/kT = A_{2,3}\,L\,r^{2-1/\nu}\langle\rho\rangle$ (Hanke i in.,
  s. 6855–6856). Zależność od promienia jest bardzo słaba (r^{0.30}), a od długości liniowa.
  W półrozcieńczonym (skalowanie blobów): $\mu/(kT L) \sim r^{2-1/\nu}\xi^{-(3-1/\nu)} \propto c^1$.
  Końce pręta pominięte (wymóg L ≫ R_x lub ξ; u nas L ≈ 4–8 nm ~ ξ, więc wynik jest szacunkiem).
- Łańcuchy idealne (θ): d = 2 = 1/ν jest przypadkiem marginalnym. A_{2,D} → 0 przy D → 4 (Eq. 2.52c,
  s. 6864), więc energia na długość zależy logarytmicznie od r (Hanke s. 6856; Eisenriegler, Hanke,
  Dietrich PRE 54, 1134 (1996) – DOI zweryfikowane, tylko abstrakt).
- Odijk 1996 Eq. (13)–(16) dotyczy odwrotnej sytuacji (mała kula oddziałująca z półsztywnym łańcuchem
  o długości L), np. $\tilde B = L a^2$ dla P > a > D (Eq. 15).
- GFVT dla prętów (L&T rozdz. 6.4.2): grubość warstwy wokół walca w roztworze oddziałujących polimerów
  (dobry rozpuszczalnik), Eq. (6.48), s. 214/228: $\delta_\mathrm{flat} = 1.071R_g/\sqrt{1+3.95\phi^{1.54}}$; Eq. (6.49)–(6.50), s. 215/229, z $y = \delta_\mathrm{flat}/r$:
  $$\frac{\delta_c}{r} = \sqrt{1 + 2y - 0.51215\,y^{1.6} + 1.214\,y^{1.77}} - 1 .$$
  Autorzy zaznaczają, że wpływ krzywizny przy oddziałujących polimerach „is unknown yet”; jest to
  krzywizna policzona dla łańcuchów idealnych z poprawionym δ_flat. Eq. (6.44) (idealne, q = 2R_g/D ≤ 100,
  dokładność 1%).

### 1c. Granica koloidalna, δ(c), ξ(c), Π(c)

- **ξ i c***, L&T Eq. (4.11)–(4.16), s. 143–144/157–158: $\xi/R_g \sim \phi^{-\gamma}$, γ = 0.77
  (dobry rozpuszczalnik), 1/γ + 1/ν = 3. Zosel i in. stosują to samo: $\xi \approx R_g(c/c^*)^{-0.77}$.
- **Profil przy ścianie** w półrozcieńczonym (de Gennes, GSA), L&T Eq. (4.17), s. 144/158:
  $f = \tanh^2(x/\xi)$, co daje δ = ξ.
- **Grubość warstwy, interpolacja rozcieńczony–półrozcieńczony** (Fleer, Skvortsov, Tuinier 2007),
  L&T Eq. (4.19)–(4.21), s. 146/160: $\delta_0 = pR_g$ (p = 1.07 dla dobrego rozpuszczalnika, Hanke i in.);
  $(\delta_0/\delta)^2 = 1 + \beta\phi^{2\gamma}$, β = 3.95 (dobry), czyli $\delta = 1.07R_g(1+3.95\phi^{1.54})^{-1/2}$ (asymptotycznie $\delta \to 0.54R_g\phi^{-0.77}$).
- **Wokół kuli**, L&T Eq. (4.23), (4.24), (4.33), s. 146–150/160–164:
  $\delta_s/R = 0.865\,[q/\sqrt{1+3.95\phi^{1.54}}]^{0.88}$, q = R_g/R. Jest to fit potęgowy, więc nie
  odtwarza asymptotyki q → ∞: daje μ ∝ (R/ξ)^{0.36} zamiast (R/ξ)^{1.30}.
- **Π**: L&T Eq. (4.27)–(4.28), s. 148/162: $\Pi v_p/(\phi_p kT) = 1 + 1.62\,\phi_p^{1.31}$ (dobry
  rozpuszczalnik); w półrozcieńczonym $\Pi \sim kT/\xi^3 \sim \phi^{3\gamma}$.
  **Cohen i in. 2009** (pełny tekst lokalnie), Eq. (4), s. 3711:
  $\Pi N^{9/5} = \frac{RT}{M_m\bar V}\big[(C/C^*_N) + \alpha (C/C^*_N)^{9/4}\big]$, $C^*_N = N^{-4/5}/\bar V$,
  α_PEG = 0.49 ± 0.01 (Fig. 3, s. 3712), $\bar V = 0.825$ mL/g, M_m = 44. Po rozpisaniu:
  $$\Pi = RT\left[\frac{C}{M} + \alpha\,\frac{(C\bar V)^{5/4} C}{M_m}\right],$$
  gdzie człon des Cloizeaux nie zależy od M. Domena van 't Hoffa: C ≲ 0.2C#, domena des Cloizeaux:
  C ≳ 5C#, C# = α^{−4/5}C*_N (s. 3712). Długość ξ_Π ≡ (kT/Π_dC)^{1/3} wynosi 3.2 nm przy 10 wt%
  i 1.1 nm przy 40 wt% (niezależnie od M).
  Stanley & Strey 2003 (Macromolecules 36, 6888, pełny tekst lokalnie) to dane Π dla PEG 400 i 8000
  w zakresie 0–50 wt%. Lokalnie pliki `pi_experimental.pdf` i `measuring-osmotic-pressure-...pdf`
  to **ta sama praca**.
- **Napięcie powierzchniowe / potencjał kontaktowy**, Louis, Bolhuis, Meijer, Hansen 2002, JCP 117,
  1893 (pełny tekst, arXiv). Eq. (1): $W(0) = -2\gamma_w$; Eq. (2):
  $\gamma_w = -\int_0^\rho (\partial\Pi/\partial\rho')\hat\Gamma(\rho')d\rho'$; Eq. (6) w półrozcieńczonym
  ($\hat\Gamma \approx -\xi$): $W_{sd}(0) = 3\Pi\hat\Gamma \propto \rho^{2\nu/(3\nu-1)} \approx \rho^{1.539}$,
  czyli $\gamma_w \simeq \tfrac32\Pi\xi$; Eq. (8): zasięg $D_{sd} = -3\hat\Gamma \propto \rho^{-0.770}$.
  Ważne dla ρ/ρ* ≳ 2.
- **Wstawienie dużego koloidu** (Helfrich, Hanke i in. Eq. (1.7a)): $F_K = \Pi v_K + \int_S(\gamma + \ldots)dS$.
  W GFVT (L&T Eq. (3.27), s. 120/134; rozdz. 4.3) $\mu_\mathrm{dep} = \Pi\, V_\mathrm{excl}(R+\delta_s)$.
  Pierwszy rząd rozwinięcia daje $\Pi V + \Pi\delta A$, czyli γ = Πδ, o czynnik 2/3 mniej niż u Louisa.
- **Skalowanie w granicy białkowej w GFVT**, L&T Eq. (4.37)–(4.39), s. 153/167: $Y = \phi_p q^{-1/\gamma}$,
  binodale ∝ q^{1.3}, potwierdzone symulacjami i eksperymentem (Fig. 4.14).

## 2. Model dla naszego układu i założenia

ΔG_dep = μ(dsDNA) − 2μ(ssDNA). Ujemne wartości oznaczają stabilizację dupleksu. Interakcje między
dwoma ssDNA w kompleksie przejściowym i entropię konformacyjną pomijam. ΔG_dep w tym ujęciu to po
prostu różnica energii wstawienia stanu końcowego i stanów początkowych w roztworze PEG.

- dsDNA: walec r = 1 nm, L = 4.4 nm. ssDNA: (i) walec r = 0.5 nm, L = 8 nm; (ii) kula R = 1.5 nm.
  Kula jest twardą górną granicą; rzeczywisty kłębek jest przenikalny, patrz §5.
- PEG: R_g = 0.0215·M^{0.583} nm; c* jak w projekcie; C = wt%·ρ/100, ρ = 0.99707 + 0.0017441·wt%;
  x = 1000w/[(1−w)·44.05] (mol EO/kg H₂O).
- P (granica białkowa): $\langle\rho\rangle = c_\mathrm{mon}\,R_x^{1/\nu}/N$, przy czym
  $R_x = R_g\sqrt{6.25/3}$. Stosunek ⟨R_E²⟩/⟨R_g²⟩ ≈ 6.25 dla SAW to założenie, którego nie
  weryfikowałem. A_{3,3}, A_{2,3} według Hanke i in.; ν = 0.583.
- G (GFVT): Π z równania Cohena; kula: $\Pi\frac{4\pi}{3}(R+\delta_s)^3$ (Eq. 4.33); walec:
  $\Pi\pi(r+\delta_c)^2(L+2\delta_c)$ (Eq. 6.50). Łączę Π z Cohena z δ z Fleera, choć u L&T δ jest
  spójne z Eq. 4.27.
- C (asymptota koloidalna): $\Pi(V + A\,\delta_\mathrm{flat})$, walec z płaskimi końcami. Wersja Louisa
  (γ = 1.5Πξ) podnosi tylko człon powierzchniowy.

## 3. Wyniki (`q3_table.csv`)

ΔG_dep w kJ/mol. p_fit to nachylenie ln|ΔG| względem ln x dopasowane w zakresie 5–40 wt%;
p(w) to nachylenie lokalne.

| PEG | model | ssDNA | p_fit 5–40 | p(10%) | p(20%) | p(40%) | ΔG 5% | ΔG 10% | ΔG 20% | ΔG 40% |
|---|---|---|---|---|---|---|---|---|---|---|
| dowolne (600/6k/35k) | P granica białkowa | pręt | 0.83 | 0.92 | 0.83 | 0.64 | −2.1 | −4.2 | −8.6 | −17.7 |
| dowolne | P | kula | 0.83 | 0.92 | 0.83 | 0.64 | −5.4 | −10.8 | −22.0 | −45.6 |
| 600 | G GFVT | pręt | 0.67 | 0.76 | 0.64 | 0.53 | −4.9 | −9.1 | −16.0 | −28.4 |
| 600 | G | kula | 1.07 | 1.05 | 1.09 | 1.02 | −4.8 | −10.4 | −24.8 | −71.3 |
| 600 | C koloid. | pręt | 0.80 | 0.88 | 0.79 | 0.64 | −2.2 | −4.4 | −8.6 | −17.6 |
| 600 | C | kula | 1.05 | 1.03 | 1.07 | 1.00 | −4.8 | −10.3 | −24.1 | −67.9 |
| 6000 | G | pręt | 0.77 | 0.75 | 0.80 | 0.67 | −3.3 | −5.6 | −10.5 | −22.0 |
| 6000 | G | kula | 1.42 | 1.42 | 1.46 | 1.23 | −1.5 | −4.0 | −13.0 | −49.9 |
| 6000 | C | pręt | 1.04 | 1.10 | 1.06 | 0.82 | −1.0 | −2.2 | −5.2 | −13.3 |
| 6000 | C | kula | 1.38 | 1.37 | 1.42 | 1.21 | −1.6 | −4.1 | −12.8 | −47.7 |
| 35000 | G | pręt | 0.85 | 0.87 | 0.87 | 0.70 | −2.6 | −4.8 | −9.8 | −21.6 |
| 35000 | G | kula | 1.53 | 1.61 | 1.54 | 1.26 | −1.0 | −3.3 | −11.9 | −48.1 |
| 35000 | C | pręt | 1.15 | 1.26 | 1.14 | 0.85 | −0.7 | −1.8 | −4.8 | −13.0 |
| 35000 | C | kula | 1.49 | 1.54 | 1.50 | 1.24 | −1.1 | −3.4 | −11.7 | −45.9 |

Obserwacje: −4 do −6 kJ/mol przy ~5 wt% (x ≈ 1.2 m), potem +0.10 kcal/mol = +0.42 kJ/mol na molal,
czyli ΔG ≈ −5 + 5.8 ≈ +1 kJ/mol przy 40 wt% (x ≈ 15 m).

Uwaga o p: model P jest dokładnie liniowy w stężeniu masowym C. Ponieważ x ∝ w/(1−w), na skali
monomolalnej ta sama liniowość daje p od 0.96 (5%) do 0.64 (40%). Asymptotyczne wartości
półrozcieńczone (C ≫ c*, R ≫ ξ) to 2.31 dla członu objętościowego i 1.54 dla powierzchniowego. Widać je
tylko częściowo, bo w naszym zakresie R ~ ξ.

## 4. Interpretacja

1. **Wczesne nasycenie (~5 wt%).** Czysta deplecja go nie daje. We wszystkich modelach
   d|ΔG_dep|/dx > 0 w całym zakresie, a najmniejsze p ≈ 0.5–0.6 (GFVT, pręty, PEG 600 przy 40 wt%) wciąż
   oznacza wyraźny wzrost: ΔG_dep(40%)/ΔG_dep(5%) ≈ 6–45. Plateau i następującej po nim destabilizacji
   nie da się odtworzyć bez członu dodatniego (o przeciwnym znaku) rzędu
   ≈ +1.6 (ssDNA jako pręt) do +3.4 kJ/mol·m⁻¹ (ssDNA jako kula) w modelu P (= obserwowane +0.42 minus nachylenie deplecji ≈ −1.2…−3.0 kJ/mol·m⁻¹) (np. preferencyjne oddziaływanie PEG–DNA lub efekty kompleksowania
   Na⁺, liczone w projekcie osobno) albo bez założenia, że deplecja jest znacznie słabsza (§5).
   Krzywizna ln|ΔG_dep| w modelach G/C (p rośnie, a potem maleje) ma przeciwny kierunek do obserwacji.
2. **Niezależność od M na skali monomerowej.** Ten punkt przemawia za **granicą białkową**: SRE daje
   μ ∝ c_mon·R^{3−1/ν} bez zależności od M, zarówno w roztworze rozcieńczonym, jak i półrozcieńczonym.
   Modele G/C dla PEG 600 vs 35k różnią się ~2–5× przy 5–10 wt%, bo w dużym PEG δ ≈ R_g jest zastępowane
   przez ξ dopiero powyżej c* (c* = 0.33, 0.059, 0.016 g/mL dla 600/6k/35k). Dla M ≳ 6k wszystkie modele
   zbiegają się na skali monomerowej powyżej ~20 wt% (w półrozcieńczonym obowiązuje des Cloizeaux, Π i ξ
   nie zależą od M).
3. **Wykładnik.** Pytanie „>1 vs <1” należy przeformułować. Granica białkowa daje **p = 1 względem
   stężenia monomeru** (efektywne 0.6–0.95 na skali monomolalnej), a granica koloidalna p = 1.54–2.31
   (efektywne 1.0–1.6 w naszym oknie). Liniowa w x destabilizacja powyżej 5 wt% jest **zgodna co do
   kształtu z granicą białkową** (też liniową), ale ma przeciwny znak. Jeśli więc w danych dominuje człon
   liniowy, depletion w granicy białkowej i oddziaływanie preferencyjne są nierozróżnialne na podstawie
   samej zależności od x. Rozróżnić je może tylko znak i zależność od geometrii (r^{0.30}L dla prętów).
4. **Wielkość.** Przy 5 wt% granica białkowa daje −2…−5 kJ/mol, czyli rząd naszego plateau. Zosel i in.
   (PNAS 2020) przy 13–17 wt% PEG 4.6k–35k dla dwóch IDP mierzą |ΔΔG| ≈ 2 kT ≈ 5 kJ/mol i dopasowują
   liniową zależność od c (Eq. 4 oraz Fig. 2E), stosując przenikalny „segment” R₂ = 0.4 nm zamiast R_H.
   Przy pełnych twardych kulach przeszacowanie jest wyraźne (ich Fig. 2F).

## 5. Luki w teorii dla naszego przypadku

- **Brak ścisłego wyniku dla obiektów o rozmiarze ~ξ.** Przy 10–40 wt% mamy r, R, L ~ ξ ≈ 0.8–2.3 nm.
  Granica białkowa i koloidalna są wtedy tylko asymptotami. GFVT (Eq. 4.33) nie ma poprawnej asymptotyki
  q → ∞ i nie ma poprawki krzywizny dla walców przy łańcuchach oddziałujących (L&T s. 215).
- **Krótkie pręty.** Wynik Hanke i in. dotyczy walca nieskończonego (L ≫ R_x). Brak amplitudy SRE dla
  walca o L/r ≈ 4–16 w półrozcieńczonym roztworze z wyłączoną objętością. Eisenriegler (2005, 2006,
  2007) daje tylko anizotropowe cząstki przy łańcuchach idealnych.
- **ssDNA nie jest twardą cząstką.** To przenikalny, naładowany kłębek (R_g ≈ 1.5–2 nm). Efektywna
  objętość wyłączona jest o rząd mniejsza (Zosel: R₂ ≈ 0.4 nm). Wtedy ΔG_dep jest bliższe wartościom
  „pręt r = 0.5 nm”, czyli −2 kJ/mol przy 5 wt%. Do tego dochodzi kompakcja ssDNA w PEG (zmiana μ_ss
  wynikająca z deplecji wewnątrzcząsteczkowej), której tu nie liczę.
- **Rozmiar monomeru PEG jest porównywalny z r ssDNA.** Teoria ciągła (R ≫ długość segmentu) przestaje
  działać przy r = 0.5 nm i dla PEG 600 (N ≈ 13). Brak też efektów grup końcowych.
- **Niespójność Π–δ.** Cohen (empiryczne Π dla PEG) jest łączony z δ Fleera, zaprojektowanym pod
  własne Π (Eq. 4.27). Różnica prefaktorów w γ (Πδ vs 1.5Πξ) wynosi czynnik ~1.5–3.
- **Sprzężenie z solą i warstwą hydratacyjną/jonową** (Na⁺ kompleksowany przez PEG, warstwa
  przeciwjonów wokół DNA zwiększa efektywny promień) nie występuje w żadnej z tych teorii.

## Bibliografia

DOI sprawdzone przez Crossref API (`api.crossref.org/works`), 2026‑10‑09.

1. Lekkerkerker H.N.W., Tuinier R. *Colloids and the Depletion Interaction*, Lecture Notes in Physics 833, Springer 2011. DOI 10.1007/978-94-007-1223-2. **Pełny tekst** (`literature/theoretical_models/978-94-007-1223-2.pdf`).
2. Hanke A., Eisenriegler E., Dietrich S. Polymer depletion effects near mesoscopic particles. *Phys. Rev. E* 59, 6853–6878 (1999). DOI 10.1103/PhysRevE.59.6853. **Pełny tekst** (`literature/theoretical_models/Hanke_meso.pdf`).
3. Odijk T. Protein–Macromolecule Interactions. *Macromolecules* 29, 1842–1843 (1996). DOI 10.1021/ma951467a. **Pełny tekst** (wersja autorska z TU Delft; `literature/review_2026-10/Odijk1996_Macromolecules_protein_macromolecule_interactions.pdf`).
4. Odijk T. Depletion Theory of Protein Transport in Semi-Dilute Polymer Solutions. *Biophys. J.* 79, 2314–2321 (2000). DOI 10.1016/S0006-3495(00)76477-0. **Tylko abstrakt** (PubMed 11053111; pobranie PDF z PMC i cell.com zablokowane). Dotyczy transportu, nie energii wstawienia.
5. Eisenriegler E. Small mesoscopic particles in dilute and semidilute solutions of nonadsorbing polymers. *J. Chem. Phys.* 113, 5091–5097 (2000). DOI 10.1063/1.1289239. **Tylko tytuł** (abstrakt niedostępny, opis u L&T s. 32).
6. Eisenriegler E., Hanke A., Dietrich S. Polymers interacting with spherical and rodlike particles. *Phys. Rev. E* 54, 1134–1152 (1996). DOI 10.1103/PhysRevE.54.1134. **Tylko metadane.**
7. Eisenriegler E., Bringer A. Depletion interactions of non-spherical colloidal particles in polymer solutions. *J. Phys.: Condens. Matter* 17, S1711–S1729 (2005). DOI 10.1088/0953-8984/17/20/005. **Pełny tekst** (arXiv cond-mat/0501432; `literature/review_2026-10/Eisenriegler2005_JPhysCondensMatter_depletion_nonspherical_particles.pdf`). Tylko łańcuchy idealne.
8. Louis A.A., Bolhuis P.G., Meijer E.J., Hansen J.-P. Polymer induced depletion potentials in polymer–colloid mixtures. *J. Chem. Phys.* 117, 1893–1907 (2002). DOI 10.1063/1.1483299. **Pełny tekst** (arXiv cond-mat/0203144; `literature/review_2026-10/Louis2002_JChemPhys_polymer_induced_depletion_potentials.pdf`).
9. Fleer G.J., Skvortsov A.M., Tuinier R. A Simple Relation for the Concentration Dependence of Osmotic Pressure and Depletion Thickness in Polymer Solutions. *Macromol. Theory Simul.* 16, 531–540 (2007). DOI 10.1002/mats.200700022. **Tylko metadane** (wzory cytowane za L&T Eq. 4.20–4.21, 4.27).
10. Fleer G.J., Tuinier R. Analytical phase diagrams for colloids and non-adsorbing polymer. *Adv. Colloid Interface Sci.* 143, 1–47 (2008). DOI 10.1016/j.cis.2008.07.001. **Tylko metadane** (wzory za L&T Eq. 4.24–4.25).
11. Tuinier R., Fleer G.J. Concentration and Solvency Effects on the Excess Amount and Surface Free Energy of a Colloidal Particle in a Solution of Nonadsorbing Polymer. *Macromolecules* 37, 8754–8763 (2004). DOI 10.1021/ma0485742. **Tylko metadane** (OA w FZ Jülich, ale repozytorium blokuje pobieranie).
12. Cohen J.A., Podgornik R., Hansen P.L., Parsegian V.A. A Phenomenological One-Parameter Equation of State for Osmotic Pressures of PEG and Other Neutral Flexible Polymers in Good Solvents. *J. Phys. Chem. B* 113, 3709–3714 (2009). DOI 10.1021/jp806893a. **Pełny tekst** (lokalnie).
13. Stanley C.B., Strey H.H. Measuring Osmotic Pressure of Poly(ethylene glycol) Solutions by Sedimentation Equilibrium Ultracentrifugation. *Macromolecules* 36, 6888–6893 (2003). DOI 10.1021/ma034079e. **Pełny tekst** (lokalnie, dwie kopie).
14. Hermans J. Excluded-volume theory of polymer–protein interactions based on polymer chain statistics. *J. Chem. Phys.* 77, 2193–2203 (1982). DOI 10.1063/1.444026. **Pełny tekst** (lokalnie).
15. Zosel F., Soranno A., Buholzer K.J., Nettels D., Schuler B. Depletion interactions modulate the binding between disordered proteins in crowded environments. *PNAS* 117, 13480–13489 (2020). DOI 10.1073/pnas.1921617117. **Pełny tekst** (lokalnie `depletion_disordered_proteins.pdf`).
16. de Gennes P.G. *C. R. Acad. Sci. Paris B* 288, 359 (1979) (u Odijka; L&T cytują s. 203). **NIEZWERYFIKOWANE** (brak DOI, rozbieżne strony).
17. de Gennes P.G. *Scaling Concepts in Polymer Physics*, Cornell Univ. Press 1979. **NIEZWERYFIKOWANE** (cytowane za L&T i Cohenem).
