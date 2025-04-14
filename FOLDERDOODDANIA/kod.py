import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.sparse.linalg as spla


class SparseMatrix:
   
    def __init__(self, n_rows, n_cols):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.elements = {}  # słownik: klucz (r,c), wartość: wartość elementu
    
    def __getitem__(self, key):
        return self.elements.get(key, 0.0)
    
    def __setitem__(self, key, value):
        if abs(value) != 0: 
            self.elements[key] = value
        elif key in self.elements:
            del self.elements[key]
    
    def to_numpy(self):
        result = np.zeros((self.n_rows, self.n_cols))
        for (r, c), value in self.elements.items():
            result[r, c] = value
        return result
    
    def to_scipy_sparse(self):
    
        rows, cols, data = [], [], []
        for (r, c), value in self.elements.items():
            rows.append(r)
            cols.append(c)
            data.append(value)
        return csr_matrix((data, (rows, cols)), shape=(self.n_rows, self.n_cols))
    
    def copy(self):
       
        new_matrix = SparseMatrix(self.n_rows, self.n_cols)
        new_matrix.elements = self.elements.copy()
        return new_matrix
    
def sparse_matrix_stats(matrix):
    total_elements = matrix.n_rows * matrix.n_cols
    nonzero_elements = len(matrix.elements)
    sparsity = 100 * (1 - nonzero_elements / total_elements)
    
    print(f"Wymiary macierzy: {matrix.n_rows} x {matrix.n_cols}")
    print(f"Liczba elementów całkowita: {total_elements}")
    print(f"Liczba elementów niezerowych: {nonzero_elements}")
    print(f"Współczynnik rzadkości: {sparsity:.2f}%")
    print(f"Oszczędność pamięci: {total_elements - nonzero_elements} elementów")
    
    row_counts = {}
    for (r, _) in matrix.elements:
        row_counts[r] = row_counts.get(r, 0) + 1
    
    if row_counts:
        avg_per_row = sum(row_counts.values()) / len(row_counts)
        print(f"Średnia liczba niezerowych elementów na wiersz: {avg_per_row:.2f}")

def gauss_elimination_with_partial_pivoting(A, b, use_sparse=False):
    if use_sparse and not isinstance(A, SparseMatrix):
        # Konwersja do formatu rzadkiego jeśli potrzeba
        sparse_A = SparseMatrix(len(b), len(b))
        for i in range(len(b)):
            for j in range(len(b)):
                if isinstance(A, np.ndarray) and A.ndim == 2:
                    if abs(A[i, j]) > 1e-10:
                        sparse_A[i, j] = A[i, j]
                else:
                    try:
                        val = A[i, j]
                        if abs(val) > 1e-10:
                            sparse_A[i, j] = val
                    except:
                        continue
        A = sparse_A
        
    n = len(b)
    if not use_sparse:
        # Sprawdzamy wymiary A
        if isinstance(A, np.ndarray):
            if A.ndim == 1:
                #case maciez 1 wymiar
                if A.shape[1] != n:
                    raise ValueError(f"Niekompatybilne wymiary: Wektor A ma długość {A.shape[1]}, a wektor b ma długość {n}")
                # Tworzymy macierz o odpowiednich wymiarach
                A_reshaped = np.zeros((n, n))
                A_reshaped[0, :min(n, A.shape[1])] = A[0, :min(n, A.shape[1])]
                A = A_reshaped
            elif A.shape[0] != n or A.shape[1] != n:
                raise ValueError(f"Niekompatybilne wymiary: A ma kształt {A.shape}, a wektor b ma długość {n}")
        
        # Tworzenie macierzy rozszerzonej [A|b]
        Ab = np.column_stack((A.copy(), b.copy()))
    else:
        b_copy = b.copy()
    
    for i in range(n):
        # Znajdź wiersz z największym elementem w kolumnie i
        if use_sparse:
            max_val = abs(A[i, i])
            max_row = i
            for k in range(i+1, n):
                val = abs(A[k, i])
                if val > max_val:
                    max_val = val
                    max_row = k
        else:
            max_row = i + np.argmax(np.abs(Ab[i:, i]))
    
        if max_row != i:
            if use_sparse:
                # Zamiana w macierzy rzadkiej - zamieniamy tylko niezerowe elementy
                for j in range(n+1):
                    if j < n:  # macierz A
                        temp = A[i, j]
                        A[i, j] = A[max_row, j]
                        A[max_row, j] = temp
                    else:  # wektor b
                        temp = b_copy[i]
                        b_copy[i] = b_copy[max_row]
                        b_copy[max_row] = temp
            else:
                Ab[[i, max_row]] = Ab[[max_row, i]]
        
        if use_sparse:
            if abs(A[i, i]) < 1e-10:
                raise ValueError("Macierz jest osobliwa")
        else:
            if abs(Ab[i, i]) < 1e-10:
                raise ValueError("Macierz jest osobliwa")
        for j in range(i+1, n):
            if use_sparse:
                # Obliczenie współczynnika
                if abs(A[j, i]) < 1e-10:  # jesli 0 to skip
                    continue
                factor = A[j, i] / A[i, i]
                
                # Aktualizacja wiersza j
                A[j, i] = 0.0  # zerujemy element pod przekatna
                for k in range(i+1, n):
                    A[j, k] -= factor * A[i, k]
                b_copy[j] -= factor * b_copy[i]
            else:
                factor = Ab[j, i] / Ab[i, i]
                Ab[j, i:] -= factor * Ab[i, i:]
    
    # Podstawienie wsteczne
    x = np.zeros(n)
    if use_sparse:
        for i in range(n-1, -1, -1):
            sum_val = 0
            for j in range(i+1, n):
                sum_val += A[i, j] * x[j]
            x[i] = (b_copy[i] - sum_val) / A[i, i]
    else:
        for i in range(n-1, -1, -1):
            x[i] = (Ab[i, -1] - np.sum(Ab[i, i+1:-1] * x[i+1:])) / Ab[i, i]
    
    return x
def build_wave_equations(N, h, L, T, H, g, t=0):
    
    # Liczba falowa i częstotliwość kołowa
    k = 2 * np.pi / L
    omega = 2 * np.pi / T
    
    # Krok siatki
    dx = L / N
    dz = h / N
    
    # Tworzenie siatki
    points = []
    for i in range(N+1):  # punkty w kierunku x
        for j in range(N+1):  # punkty w kierunku z
            x = i * dx
            z = -h + j * dz
            # skip poza
            if z <= 0 and z >= -h and x >= 0 and x <= L:
                points.append((x, z))
    
    # Mapowanie punktów na indeksy w układzie równań
    point_to_index = {point: idx for idx, point in enumerate(points)}
    n = len(points)
    
    # tworzymy matrix i wektor b 
    A = SparseMatrix(n, n)
    b = np.zeros(n)

    epsilon = 1e-10
    
  # budowanie dla punktow 
    for point, idx in point_to_index.items():
        x, z = point
        
        # Sprawdzenie gdzie punkt
        is_surface = abs(z) < 1e-10  
        is_bottom = abs(z + h) < 1e-10 
        is_side = abs(x) < 1e-10 or abs(x - L) < 1e-10  
        
        if is_surface:
            # Warunek brzegowy na powierzchni: (∂²φ/∂t²) + g(∂φ/∂z) = 0
            # aproksymujemy pochodna  z dolem 
            point_below = (x, z - dz)
            if point_below in point_to_index:
                idx_below = point_to_index[point_below]
                # g*(φ(x,0) - φ(x,-dz))/dz = analytic_val
                A[idx, idx] = g + epsilon  # Dodajemy regularyzację
                A[idx, idx_below] = -g
                # Wartość analityczna drugiej pochodnej
                analytic_val = -g * (g * H / (2 * omega)) * np.sin(k * x - omega * t)
                b[idx] = analytic_val * dz
            else:
                #  nie ma ponizej to analityic
                A[idx, idx] = 1.0 + epsilon
                analytic_val = (g * H / (2 * omega)) * np.sin(k * x - omega * t)
                b[idx] = analytic_val
            
        elif is_bottom:
            # Warunek brzegowy na dnie: ∂φ/∂z = 0
            point_above = (x, z + dz)
            if point_above in point_to_index:
                idx_above = point_to_index[point_above]
                A[idx, idx] = -1.0 + epsilon 
                A[idx, idx_above] = 1.0
                b[idx] = 0.0
            else:
                A[idx, idx] = 1.0 + epsilon
                analytic_val = (g * H / (2 * omega)) * (np.cosh(k * (z + h)) / np.cosh(k * h)) * np.sin(k * x - omega * t)
                b[idx] = analytic_val
                
        elif is_side:
            A[idx, idx] = 1.0 + epsilon  
            analytic_val = (g * H / (2 * omega)) * (np.cosh(k * (z + h)) / np.cosh(k * h)) * np.sin(k * x - omega * t)
            b[idx] = analytic_val
            
        else:
            # Punkt wewnętrzny - równanie Laplace'a: (∂²φ/∂x²) + (∂²φ/∂z²) = 0
            point_left = (x - dx, z)
            point_right = (x + dx, z)
            point_down = (x, z - dz)
            point_up = (x, z + dz)
            
            # Sprawdzamy czy sasieady w siatce
            neighbors = []
            if point_left in point_to_index:
                neighbors.append((point_to_index[point_left], 1.0))
            if point_right in point_to_index:
                neighbors.append((point_to_index[point_right], 1.0))
            if point_down in point_to_index:
                neighbors.append((point_to_index[point_down], 1.0))
            if point_up in point_to_index:
                neighbors.append((point_to_index[point_up], 1.0))
                
            # Współczynnik przy centralnym punkcie
            A[idx, idx] = -len(neighbors) + epsilon  # Dodajemy regularyzację
            
            # sasiady
            for neighbor_idx, coef in neighbors:
                A[idx, neighbor_idx] = coef
                
            # wyniki
            b[idx] = 0.0
    
    return A, b, point_to_index

def solve_wave_potential(N, h, L, T, H, g, t=0, use_sparse=True, use_library=False):

    start_time = time.time()
    
    # budowanie ukladu rownan
    A, b, point_to_index = build_wave_equations(N, h, L, T, H, g, t)
    
    # rozwiazanie
    if use_library:
        
        A_scipy = A.to_scipy_sparse()
        try:
            phi_values = spla.spsolve(A_scipy, b)
        except Warning:
            phi_values = spla.lsqr(A_scipy, b)[0]
    else:
        # U
        phi_values = gauss_elimination_with_partial_pivoting(A, b, use_sparse=use_sparse)
    
    # mapowanie na siatke
    phi = {point: phi_values[idx] for point, idx in point_to_index.items()}
    
    time_taken = time.time() - start_time
    return phi, time_taken

def analytic_solution(x, z, t, h, L, T, H, g):
    k = 2 * np.pi / L
    omega = 2 * np.pi / T
    
    return (g * H / (2 * omega)) * (np.cosh(k * (z + h)) / np.cosh(k * h)) * np.sin(k * x - omega * t)

def compare_with_analytic(phi, h, L, T, H, g, t=0):
    errors = []
    
    for (x, z), phi_num in phi.items():
        phi_analytic = analytic_solution(x, z, t, h, L, T, H, g)
        error = abs(phi_num - phi_analytic)
        errors.append(error)
    
    max_error = max(errors)
    mean_error = sum(errors) / len(errors)
    
    return max_error, mean_error

def analyze_errors(phi, h, L, T, H, g, t=0):
   
    errors = {}
    max_error = 0
    max_error_point = None
    for (x, z), phi_num in phi.items():
        phi_analytic = analytic_solution(x, z, t, h, L, T, H, g)
        error = abs(phi_num - phi_analytic)
        errors[(x, z)] = error
        
        if error > max_error:
            max_error = error
            max_error_point = (x, z)
    
    print(f"Maksymalny błąd: {max_error:.6e} w punkcie {max_error_point}")
    
    
    depth_errors = {}
    for (x, z), error in errors.items():
        depth = round(-z, 3)  #
        if depth not in depth_errors:
            depth_errors[depth] = []
        depth_errors[depth].append(error)
    
    avg_errors = {depth: sum(errs)/len(errs) for depth, errs in depth_errors.items()}
    
    
    depths = sorted(avg_errors.keys())
    avg_error_values = [avg_errors[d] for d in depths]
    
    plt.figure(figsize=(8, 6))
    plt.plot(depths, avg_error_values, 'o-')
    plt.xlabel('Głębokość [m]')
    plt.ylabel('Średni błąd')
    plt.title('Rozkład błędów w zależności od głębokości')
    plt.grid(alpha=0.3)
    plt.show()
    
    return errors

def visualize_potential(phi, h, L, title="Funkcja potencjału"):
    
    x_values = []
    z_values = []
    phi_values = []
    
    for (x, z), val in phi.items():
        x_values.append(x)
        z_values.append(z)
        phi_values.append(val)
    
    # siatka
    x_unique = sorted(list(set(x_values)))
    z_unique = sorted(list(set(z_values)))
    X, Z = np.meshgrid(x_unique, z_unique)
    
    
    PHI = np.zeros_like(X)
    for i, z in enumerate(z_unique):
        for j, x in enumerate(x_unique):
            if (x, z) in phi:
                PHI[i, j] = phi[(x, z)]
    
    plt.figure(figsize=(10, 6))
    contour = plt.contourf(X, Z, PHI, 50, cmap='viridis')
    plt.colorbar(contour, label='Potencjał φ')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('z')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)  
    plt.axhline(y=-h, color='black', linestyle='-', alpha=0.7) 
    plt.xlim(0, L)
    plt.ylim(-h, 0)
    plt.grid(alpha=0.3)
    plt.savefig('potentials.png')
    plt.show()

def compare_methods(N_values, h, L, T, H, g):
 
    results = []
    
    for N in N_values:
        print(f"\nTestowanie dla N = {N}...")
    
        if N <= 10:  
            try:
                start_time = time.time()
                phi_dense, _ = solve_wave_potential(N, h, L, T, H, g, 
                                                  use_sparse=False, use_library=False)
                time_dense = time.time() - start_time
                max_error_dense, mean_error_dense = compare_with_analytic(phi_dense, h, L, T, H, g)
                print(f"Własna implementacja (gęsta) - czas: {time_dense:.4f}s, " 
                      f"max błąd: {max_error_dense:.6e}")
            except Exception as e:
                print(f"Błąd dla gęstej macierzy: {e}")
                time_dense = float('inf')
                max_error_dense = float('inf')
                mean_error_dense = float('inf')
        else:
            time_dense = float('inf')
            max_error_dense = float('inf')
            mean_error_dense = float('inf')
        
        # wlasna
        try:
            start_time = time.time()
            phi_sparse, _ = solve_wave_potential(N, h, L, T, H, g, 
                                               use_sparse=True, use_library=False)
            time_sparse = time.time() - start_time
            max_error_sparse, mean_error_sparse = compare_with_analytic(phi_sparse, h, L, T, H, g)
            print(f"Własna implementacja (rzadka) - czas: {time_sparse:.4f}s, " 
                  f"max błąd: {max_error_sparse:.6e}")
        except Exception as e:
            print(f"Błąd dla rzadkiej macierzy: {e}")
            time_sparse = float('inf')
            max_error_sparse = float('inf')
            mean_error_sparse = float('inf')
        
        # Implementacja biblioteczna
        try:
            start_time = time.time()
            phi_lib, _ = solve_wave_potential(N, h, L, T, H, g, 
                                            use_sparse=True, use_library=True)
            time_lib = time.time() - start_time
            max_error_lib, mean_error_lib = compare_with_analytic(phi_lib, h, L, T, H, g)
            print(f"Implementacja biblioteczna - czas: {time_lib:.4f}s, " 
                  f"max błąd: {max_error_lib:.6e}")
        except Exception as e:
            print(f"Błąd dla implementacji bibliotecznej: {e}")
            time_lib = float('inf')
            max_error_lib = float('inf')
            mean_error_lib = float('inf')
        
        results.append({
            'N': N,
            'time_dense': time_dense,
            'time_sparse': time_sparse,
            'time_lib': time_lib,
            'error_dense': max_error_dense,
            'error_sparse': max_error_sparse,
            'error_lib': max_error_lib
        })
  
    return results

def animate_wave_with_particles(N, h, L, T, H, g, duration=2.0, fps=15, particles=20):
    num_frames = int(duration * fps)
    k = 2 * np.pi / L
    omega = 2 * np.pi / T
    
    particle_x = np.linspace(0, L, particles)
    particle_z = np.linspace(-h, -0.05, 5)  
    particle_X, particle_Z = np.meshgrid(particle_x, particle_z)
    particle_X = particle_X.flatten()
    particle_Z = particle_Z.flatten()
    
    # Początkowe pozycje cząstek
    initial_positions = np.column_stack((particle_X, particle_Z))
    
    # Inicjalizacja figury
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), 
                                  gridspec_kw={'height_ratios': [3, 1]})
    
    def update(frame):
        t = (frame / num_frames) * T  # czas w sekundach

        ax1.clear()
        ax2.clear()
        particle_positions = []
        
        for x0, z0 in initial_positions:
            #funkcje do liczenia ruchu czasteczek
            amplitude_x = (g * H * k / (2 * omega**2)) * (np.cosh(k * (z0 + h)) / np.cosh(k * h))
            amplitude_z = (g * H * k / (2 * omega**2)) * (np.sinh(k * (z0 + h)) / np.cosh(k * h))
            
            x = x0 + amplitude_x * np.cos(k * x0 - omega * t)
            z = z0 + amplitude_z * np.sin(k * x0 - omega * t)
            
            # Zawijanie cząstek
            if x < 0:
                x += L
            elif x > L:
                x -= L
                
            particle_positions.append((x, z))
        
        # Rysowanie powierzchni wody
        surface_x = np.linspace(0, L, 100)
        # Wzór na przesunięcie powierzchni: ζ(x,t) = (H/2) * cos(kx - ωt)
        surface_z = (H/2) * np.cos(k * surface_x - omega * t)
        
        # Obszar wody
        ax1.fill_between(surface_x, surface_z, -h, color='lightblue', alpha=0.4)
        
        ax1.plot(surface_x, surface_z, 'b-', linewidth=3)
        
    
        particle_x, particle_z = zip(*particle_positions)
        ax1.scatter(particle_x, particle_z, c='red', s=25, alpha=0.8)
        

        ax1.set_title(f'Falowanie i ruch cząstek, t = {t:.2f} s')
        ax1.set_xlabel('x [m]')
        ax1.set_ylabel('z [m]')
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)  
        ax1.axhline(y=-h, color='black', linestyle='-', alpha=0.7)  # dno morza
        ax1.set_xlim(0, L)
        ax1.set_ylim(-h, H)
        ax1.grid(alpha=0.3)
        
        ax2.plot(surface_x, surface_z, 'b-', linewidth=3)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)  # woda
        ax2.fill_between(surface_x, surface_z, -H, color='lightblue', alpha=0.4)
        ax2.set_title('Profil powierzchni fali')
        ax2.set_xlabel('x [m]')
        ax2.set_ylabel('ζ [m]')
        ax2.set_xlim(0, L)
        ax2.set_ylim(-H, H)
        ax2.grid(alpha=0.3)
        
        return ax1, ax2
    
    # Tworzenie animacji
    ani = FuncAnimation(fig, update, frames=num_frames, blit=False)
    plt.tight_layout()
    
    ani.save('wave_animation.gif', writer='pillow', fps=fps)
    print("Animacja zapisana jako 'wave_animation.gif'")
    
    plt.show()
    return ani
def test_gauss_elimination():

    # prosta maceiz 
    A1 = np.array([[2, 1], [1, 2]], dtype=float)
    b1 = np.array([4, 5], dtype=float)
    expected1 = np.array([1, 2])
    print("\nTest 1: Prosta macierz 2x2")
    x1 = gauss_elimination_with_partial_pivoting(A1, b1)
    print("Rozwiązanie:", x1)
    print("Oczekiwane:", expected1)
    print("Błąd:", np.linalg.norm(x1 - expected1))
    

    
    # test maciez osobliwa
    A3 = np.array([[1, 2], [2, 4]], dtype=float)
    b3 = np.array([3, 6], dtype=float)
    print("\nTest 3: Macierz osobliwa")
    try:
        x3 = gauss_elimination_with_partial_pivoting(A3, b3)
        print("BŁĄD: Powinien zgłosić wyjątek dla macierzy osobliwej!")
    except ValueError as e:
        print("SUKCES: Zgłoszono wyjątek:", str(e))
    
    # test duza maciez
    np.random.seed(42)
    size = 20 
    A4 = np.random.rand(size, size) + 10*np.eye(size)
    b4 = np.random.rand(size)
    expected4 = np.linalg.solve(A4, b4)
    print("\nTest 4: Duża macierz losowa 20x20")
    x4 = gauss_elimination_with_partial_pivoting(A4.copy(), b4.copy())
    error = np.linalg.norm(x4 - expected4)
    print("Błąd:", error)
    print("Czy błąd < 1e-10?", error < 1e-10)

def main():
    h = 10.0      # głębokość morza [m]
    L = 50.0      # długość fali [m]
    T = 5.0       # okres fali [s]
    H = 0.5        # wysokość fali [m]
    g = 9.81       # przyspieszenie ziemskie [m/s²]
    
   
    # Zadanie 1: Demonstracja eliminacji Gaussa
    print("\n[Zadanie 1] test Gaussa")
    
    test_A = np.array([
        [2.0, 1.0, 4.0],
        [3.0, 4.0, -1.0],
        [1.0, -1.0, 1.0]
    ])
    test_b = np.array([12.0, 7.0, 3.0])
    
    print("Przykładowa macierz A:")
    print(test_A)
    print("Wektor b:")
    print(test_b)
    
    x = gauss_elimination_with_partial_pivoting(test_A, test_b)
    print("\nRozwiązanie x:")
    print(x)
    print("Weryfikacja Ax - b:")
    print(np.dot(test_A, x) - test_b)
    
    # Rozszerzone testy
    print("\n[Zadanie 1] Rozszerzone testy poprawności")
    test_gauss_elimination()
    
    # Reszta istniejącego kodu...
    print("\n[Zadanie 2] Rozwiązanie układu równań falowania")
    N = 10  # mniejsze N dla szybszej demonstracji
    
    print(f"Parametry: N={N}, h={h}m, L={L}m, T={T}s, H={H}m, g={g}m/s²")

    A, b, point_to_index = build_wave_equations(N, h, L, T, H, g)
    

    print("\n[Zadanie 3] Analiza macierzy rzadkiej")
    sparse_matrix_stats(A)
    
   
    phi_values = gauss_elimination_with_partial_pivoting(A, b, use_sparse=True)
    phi = {point: phi_values[idx] for point, idx in point_to_index.items()}
    
   
    max_error, mean_error = compare_with_analytic(phi, h, L, T, H, g)
    print(f"\nBłąd maksymalny: {max_error:.6e}")
    print(f"Błąd średni: {mean_error:.6e}")
    
  
    print("\nAnaliza rozkładu błędów:")
    analyze_errors(phi, h, L, T, H, g)
    
   
    print("\n[Z4] porownanie")
    compare_methods([5, 8, 10], h, L, T, H, g)
    
    visualize_potential(phi,h,L)
    
    choice = input("Czy chcesz utworzyć animację? (t/n): ")
    if choice.lower() == 't':
        animate_wave_with_particles(N, h, L, T, H, g, duration=2.0, fps=15)
    
    print("\nProgram zakończony.")

if __name__ == "__main__":
    main()

#  Attribution:
# - Fragmenty kodu  związane z tworzeniem rownan fal, czesc wizualizacji (funkcje ruchu czasteczek w animacji) zostały zainspirowane przez kod DeepseekAI
# zakres pracy, mniej wiecej po rowno razem kodowalismy lapalismy swoje bledy etc. kazdy mial jakis udzial w kazdym z zadan 

