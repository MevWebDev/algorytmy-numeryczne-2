# Analiza Numeryczna Kodu Symulacji Fal Morskich

**Student:** [Twoje Imię i Nazwisko]
**Data:** 10 maja 2025

## 1. Wstęp

Ten kod napisany w Pythonie służy do symulacji komputerowej i rysowania dwuwymiarowych fal na powierzchni wody o stałej głębokości. Głównym zadaniem jest rozwiązanie równania Laplace'a dla potencjału prędkości płynu, przy uwzględnieniu odpowiednich warunków na brzegach. Kod wykorzystuje metody numeryczne (komputerowe metody obliczeniowe) do:

1.  Przekształcenia głównego równania różniczkowego i warunków brzegowych na postać, którą komputer może obliczyć.
2.  Przechowywania powstałego układu równań liniowych w specjalny sposób (macierz rzadka), aby oszczędzić pamięć.
3.  Rozwiązania tego układu równań.
4.  Sprawdzenia, jak dokładne są uzyskane wyniki.
5.  Narysowania ruchu fal i pola potencjału.

## 2. Model Matematyczny (Jak opisujemy falę za pomocą matematyki)

### 2.1. Główne Równanie

Zakładamy, że płyn jest idealny (nielepki, nieściśliwy) i jego ruch jest bezwirowy. Wtedy ruch płynu możemy opisać za pomocą funkcji zwanej **potencjałem prędkości** $\phi(x,z,t)$. Funkcja ta spełnia **równanie Laplace'a** w obszarze, gdzie znajduje się płyn $\Omega$:
$$ \nabla^2 \phi = \frac{\partial^2 \phi}{\partial x^2} + \frac{\partial^2 \phi}{\partial z^2} = 0 \quad \text{dla } (x,z) \in \Omega $$
Tutaj $x$ to współrzędna pozioma, $z$ to współrzędna pionowa (skierowana w górę, $z=0$ oznacza średni poziom swobodnej powierzchni wody), a $t$ to czas.

### 2.2. Warunki Brzegowe (Co się dzieje na krawędziach obszaru z wodą)

(Uproszczone dla fal o małej amplitudzie)

1.  **Na dnie morza ($z=-h$):** Woda nie może przepływać przez dno.
    $$ \frac{\partial \phi}{\partial z} = 0 \quad \text{dla } z = -h $$
2.  **Na swobodnej powierzchni wody ($z \approx 0$):**
    Łącząc warunki dotyczące tego, że cząstki wody pozostają na powierzchni i że ciśnienie na powierzchni jest stałe, otrzymujemy (po uproszczeniach dla małych fal):
    $$ \frac{\partial^2 \phi}{\partial t^2} + g \frac{\partial \phi}{\partial z} = 0 \quad \text{dla } z = 0 $$
    gdzie $g$ to przyspieszenie ziemskie.
3.  **Na bocznych granicach (np. $x=0, x=L_x$):**
    Możemy założyć, że fala się powtarza (warunki okresowe) lub, jak często w tym kodzie, narzucić wartości potencjału obliczone ze znanego rozwiązania analitycznego (dokładnego wzoru).

### 2.3. Rozwiązanie Analityczne (Dokładny wzór dla fali sinusoidalnej)

Dla prostej fali sinusoidalnej o małej wysokości $H$, długości $L$ i okresie $T$, potencjał prędkości można opisać wzorem:
$$ \phi(x,z,t) = \frac{gH}{2\omega} \frac{\cosh(k(z+h))}{\cosh(kh)} \sin(kx - \omega t) $$
gdzie $k = 2\pi/L$ to liczba falowa, a $\omega = 2\pi/T$ to częstość kołowa. Ten wzór jest bardzo ważny, ponieważ:

- Pozwala sprawdzić, czy nasze obliczenia komputerowe są poprawne.
- Może być użyty do ustawienia wartości na bocznych granicach.
- Pomaga w płynnym rysowaniu animacji.
  Funkcja `analytic_solution(x, z, t, h, L, T, H, g)` w kodzie oblicza wartości według tego wzoru.

## 3. Zastosowane Metody Numeryczne (Jak komputer to liczy)

### 3.1. Podział Przestrzeni na Siatkę: Metoda Różnic Skończonych (FDM)

Ciągły obszar z wodą dzielimy na małe prostokąty, tworząc siatkę punktów $(x_i, z_j)$. Pochodne (zmiany funkcji) w równaniu Laplace'a i warunkach brzegowych zastępujemy przybliżeniami liczbowymi.

- **`build_wave_equations(N, h, L, T, H, g, t=0)`**: Ta funkcja tworzy układ równań liniowych $A\Phi = b$.
  - Tworzy siatkę $(N+1) \times (N+1)$ punktów.
  - Dla **punktów wewnętrznych siatki**, równanie Laplace'a $\frac{\partial^2 \phi}{\partial x^2} + \frac{\partial^2 \phi}{\partial z^2} = 0$ jest przybliżane za pomocą wartości potencjału w danym punkcie i jego czterech sąsiadach (tzw. szablon 5-punktowy):
    $$ \frac{\phi*{i+1,j} - 2\phi*{i,j} + \phi*{i-1,j}}{(\Delta x)^2} + \frac{\phi*{i,j+1} - 2\phi*{i,j} + \phi*{i,j-1}}{(\Delta z)^2} = 0 $$
    Kod używa prostszej formy, gdzie współczynniki dla sąsiadów wynoszą 1, a dla centralnego punktu `-liczba_sąsiadów`.
  - Dla **punktów brzegowych**:
    - **Dno ($z=-h$):** Warunek $\frac{\partial \phi}{\partial z} = 0$ jest przybliżany. Kod implementuje to jako $A_{idx,idx} = -1, A_{idx,idx\_powyżej}=1, b_{idx}=0$.
    - **Powierzchnia ($z=0$):** Warunek $\frac{\partial^2 \phi}{\partial t^2} + g \frac{\partial \phi}{\partial z} = 0$. Składnik $\frac{\partial^2 \phi}{\partial t^2}$ jest często traktowany jako znany (np. z rozwiązania analitycznego dla danego czasu $t$). $\frac{\partial \phi}{\partial z}$ jest przybliżane. Kod implementuje to jako $A_{idx,idx}=g, A_{idx,idx\_poniżej}=-g, b_{idx} = \text{wyrażenie_analityczne} \cdot dz$.
    - **Boki ($x=0, x=L_x$):** Potencjał $\phi$ jest ustawiany bezpośrednio z rozwiązania analitycznego: $A_{idx,idx}=1, b_{idx} = \phi_{analityczne}$.
  - Mała wartość `epsilon` jest dodawana do przekątnej macierzy $A$, aby poprawić stabilność obliczeń i uniknąć problemów z macierzą osobliwą (nie mającą jednoznacznego rozwiązania).

### 3.2. Reprezentacja Macierzy Rzadkiej

Macierz $A$ powstała z metody różnic skończonych jest **rzadka** (większość jej elementów to zera). Przechowywanie jej jako zwykłej, pełnej macierzy byłoby marnotrawstwem pamięci.

- **`class SparseMatrix`**:
  - `__init__(self, n_rows, n_cols)`: Tworzy macierz o podanych wymiarach.
  - `elements`: Słownik Pythona `{(r, c): wartość}` przechowuje tylko elementy niezerowe.
  - `__getitem__((r,c))`: Zwraca wartość elementu, lub 0 jeśli element nie jest przechowany (jest zerem).
  - `__setitem__((r,c), value)`: Zapisuje wartość, jeśli nie jest bliska zeru. Usuwa element, jeśli jego wartość staje się zerowa.
  - `to_numpy()`: Konwertuje do zwykłej tablicy NumPy (dla małych macierzy lub testów).
  - `to_scipy_sparse()`: Konwertuje do formatu `scipy.sparse.csr_matrix`, który jest zoptymalizowany do operacji na macierzach rzadkich.
- **`sparse_matrix_stats(matrix)`**: Oblicza i wyświetla statystyki, takie jak wymiary, liczba elementów niezerowych, procent rzadkości i oszczędność pamięci.

### 3.3. Rozwiązywanie Układu Równań Liniowych $A\Phi = b$

- **`gauss_elimination_with_partial_pivoting(A, b, use_sparse=False)`**:
  - Implementuje metodę eliminacji Gaussa do rozwiązania $A\Phi=b$.
  - **Częściowy wybór elementu głównego (pivoting)**: W każdym kroku $k$ eliminacji, znajduje wiersz $m \ge k$ z największą co do wartości bezwzględnej liczbą w kolumnie $k$. Następnie zamienia wiersz $k$ z wierszem $m$. Jest to kluczowe dla stabilności numerycznej, zapobiegając dzieleniu przez bardzo małe liczby.
  - **Eliminacja w przód**: Przekształca macierz $A$ w macierz górnotrójkątną $U$ (zera poniżej głównej przekątnej).
  - **Podstawienie wsteczne**: Rozwiązuje układ $U\Phi = b'$ (gdzie $b'$ to przekształcone $b$), zaczynając od ostatniego równania.
  - Flaga `use_sparse` pozwala działać na własnej klasie `SparseMatrix` lub na zwykłej tablicy NumPy.
- **Rozwiązania z Bibliotek**: Kod pozwala także użyć funkcji z biblioteki SciPy (np. `scipy.sparse.linalg.spsolve`), które są bardzo wydajne dla macierzy rzadkich.
- **`solve_wave_potential(...)`**: Ta funkcja zarządza budowaniem układu równań i wyborem metody jego rozwiązania.

## 4. Analiza Błędów i Weryfikacja (Sprawdzanie dokładności)

Aby ocenić dokładność rozwiązania numerycznego $\phi_{num}$:

- **`analytic_solution(...)`**: Dostarcza dokładne rozwiązanie $\phi_{analityczne}$ jako punkt odniesienia.
- **`compare_with_analytic(phi, ...)`**:
  - Oblicza błąd bezwzględny $| \phi_{num}(x_i, z_j) - \phi_{analityczne}(x_i, z_j) |$ w każdym punkcie siatki.
  - Zwraca:
    - Maksymalny błąd bezwzględny.
    - Średni błąd bezwzględny.
- **`analyze_errors(phi, ...)`**:
  - Przeprowadza bardziej szczegółową analizę błędów.
  - Znajduje punkt, w którym błąd jest największy.
  - Grupuje błędy według głębokości $z$ i oblicza średni błąd na każdym poziomie głębokości.
  - Rysuje wykres średniego błędu w funkcji głębokości. Może to pomóc zidentyfikować, czy błędy są większe np. przy powierzchni wody czy przy dnie.

## 5. Wizualizacja Wyników (Rysowanie)

- **`visualize_potential(phi, h, L, title)`**:
  - Tworzy statyczny wykres konturowy (mapę ciepła) obliczonego komputerowo pola potencjału $\phi(x,z)$ w danym momencie.
  - Pomaga to wizualnie sprawdzić, czy rozwiązanie jest gładkie i wygląda sensownie.
- **`animate_wave_with_particles(N, h, L, T, H, g, ...)`**:
  - Generuje animację ruchu fali.
  - **Powierzchnia Fali**: Kształt powierzchni $\eta(x,t) = (H/2) \cos(kx - \omega t)$ jest rysowany bezpośrednio z wzoru analitycznego, aby animacja była płynna.
  - **Ruch Cząstek**: Animowane są trajektorie wybranych cząstek płynu. Ich pozycje $(x_p(t), z_p(t))$ są obliczane na podstawie prędkości $u = \partial\phi/\partial x$ i $w = \partial\phi/\partial z$. Dla prostoty i płynności, te prędkości (a więc i przemieszczenia) są często wyliczane z _analitycznego_ potencjału $\phi_{analitycznego}$.
  - Animacja składa się z dwóch części: jedna pokazuje falę i ruch cząstek, a druga profil powierzchni fali.
  - Wersja tej funkcji w dostarczonym kodzie skupia się _tylko_ na powierzchni fali i ruchu cząstek, pomijając wizualizację pola potencjału w samej animacji dla większej przejrzystości.

## 6. Porównanie Metod Rozwiązywania

- **`compare_methods(N_values, ...)`**:
  - Systematycznie ocenia różne sposoby rozwiązywania układu równań dla różnych rozmiarów siatki $N$:
    1.  Własna eliminacja Gaussa z pełnymi macierzami NumPy (próba dla małych $N$).
    2.  Własna eliminacja Gaussa z własną klasą `SparseMatrix`.
    3.  Funkcja z biblioteki SciPy (`spsolve`) z macierzą `csr_matrix`.
  - Dla każdej metody i rozmiaru $N$ zapisuje:
    - Czas wykonania.
    - Maksymalny błąd w porównaniu do rozwiązania analitycznego.
  - Wyniki są następnie rysowane na wykresach, aby porównać wydajność (czas vs. $N$) i dokładność (błąd vs. $N$) tych podejść.

## 7. Testowanie i Główny Przepływ Programu

- **`test_gauss_elimination()`**: Zawiera kilka przypadków testowych dla funkcji `gauss_elimination_with_partial_pivoting`, w tym prosty układ 2x2, macierz osobliwą (powinna zgłosić błąd) oraz większą, losową macierz, porównując wynik z `np.linalg.solve`. Pomaga to upewnić się, że główny algorytm rozwiązywania działa poprawnie.
- **`main()`**:
  - Ustawia fizyczne parametry problemu fali.
  - Demonstruje eliminację Gaussa (Zadanie 1).
  - Buduje i rozwiązuje równania falowe dla określonego $N$ (Zadanie 2).
  - Analizuje rzadkość macierzy układu (Zadanie 3).
  - Porównuje metody rozwiązywania (Zadanie 4).
  - Rysuje statyczne pole potencjału.
  - Opcjonalnie generuje animację fali (Zadanie 5).

## 8. Podsumowanie

Kod Pythona dostarcza kompleksowe narzędzia do symulacji liniowych fal wodnych. Poprawnie wskazuje na potrzebę użycia technik dla macierzy rzadkich ze względu na charakter dyskretyzacji równania Laplace'a metodą różnic skończonych. Implementacja eliminacji Gaussa z częściowym wyborem elementu głównego, choć pouczająca, podkreśla korzyści wydajnościowe płynące z używania zoptymalizowanych funkcji bibliotecznych dla większych układów. Komponenty do analizy błędów i wizualizacji są niezbędne do walidacji modelu i zrozumienia dynamiki fal. Porównanie metod oferuje cenne spostrzeżenia dotyczące praktycznych aspektów rozwiązywania problemów numerycznych.
