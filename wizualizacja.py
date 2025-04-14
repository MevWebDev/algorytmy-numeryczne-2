import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.sparse.linalg as spla
from wave_functions import *

def visualize_potential(phi, h, L, title="Funkcja potencjału"):
    """
    Wizualizuje funkcję potencjału na siatce.
    
    Parametry:
    ----------
    phi : dict
        Słownik mapujący punkty siatki (x, z) na wartości potencjału.
    h, L : float
        Głębokość morza i długość fali.
    title : str, optional
        Tytuł wykresu.
    """
    # Przygotowanie danych do wizualizacji
    x_values = []
    z_values = []
    phi_values = []
    
    for (x, z), val in phi.items():
        x_values.append(x)
        z_values.append(z)
        phi_values.append(val)
    
    # Tworzenie siatki dla wykresu konturowego
    x_unique = sorted(list(set(x_values)))
    z_unique = sorted(list(set(z_values)))
    X, Z = np.meshgrid(x_unique, z_unique)
    
    # Wypełnianie siatki wartościami potencjału
    PHI = np.zeros_like(X)
    for i, z in enumerate(z_unique):
        for j, x in enumerate(x_unique):
            if (x, z) in phi:
                PHI[i, j] = phi[(x, z)]
    
    # Tworzenie wykresu
    plt.figure(figsize=(10, 6))
    contour = plt.contourf(X, Z, PHI, 50, cmap='viridis')
    plt.colorbar(contour, label='Potencjał φ')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('z')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)  # powierzchnia morza
    plt.axhline(y=-h, color='black', linestyle='-', alpha=0.7)  # dno morza
    plt.xlim(0, L)
    plt.ylim(-h, 0)
    plt.grid(alpha=0.3)
    plt.savefig('potentials.png')
    plt.show()

def compare_methods(N_values, h, L, T, H, g):
    """
    Porównuje różne metody rozwiązywania układu równań.
    
    Parametry:
    ----------
    N_values : list
        Lista wartości N (liczba podziałów siatki) do przetestowania.
    h, L, T, H, g : float
        Parametry fizyczne fali.
    """
    results = []
    
    for N in N_values:
        print(f"\nTestowanie dla N = {N}...")
        
        # Własna implementacja bez optymalizacji dla macierzy rzadkich
        # Tylko dla małych N próbujemy gęstej implementacji
        if N <= 10:  # Ustal odpowiedni próg
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
            print(f"Pomijam gęstą implementację dla N={N} (zbyt duże wymiary)")
            time_dense = float('inf')
            max_error_dense = float('inf')
            mean_error_dense = float('inf')
        
        # Własna implementacja z optymalizacją dla macierzy rzadkich
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
    
    # Wizualizacja wyników
    plt.figure(figsize=(12, 10))
    
    # Wykres czasu wykonania
    plt.subplot(2, 1, 1)
    plt.plot([r['N'] for r in results], [r['time_dense'] for r in results], 'o-', label='Własna (gęsta)')
    plt.plot([r['N'] for r in results], [r['time_sparse'] for r in results], 's-', label='Własna (rzadka)')
    plt.plot([r['N'] for r in results], [r['time_lib'] for r in results], '^-', label='Biblioteczna')
    plt.xlabel('Rozmiar siatki (N)')
    plt.ylabel('Czas wykonania [s]')
    plt.title('Porównanie czasu wykonania różnych implementacji')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Wykres błędu
    plt.subplot(2, 1, 2)
    plt.plot([r['N'] for r in results], [r['error_dense'] for r in results], 'o-', label='Własna (gęsta)')
    plt.plot([r['N'] for r in results], [r['error_sparse'] for r in results], 's-', label='Własna (rzadka)')
    plt.plot([r['N'] for r in results], [r['error_lib'] for r in results], '^-', label='Biblioteczna')
    plt.xlabel('Rozmiar siatki (N)')
    plt.ylabel('Maksymalny błąd')
    plt.yscale('log')
    plt.title('Porównanie dokładności różnych implementacji')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comparison_methods.png')
    plt.show()
    
    return results

def animate_wave_with_particles(N, h, L, T, H, g, duration=2.0, fps=15, particles=20):
    """
    Tworzy animację fali z wizualizacją ruchu cząstek płynu, bez pokazywania potencjału.
    
    Parametry:
    ----------
    N : int
        Liczba podziałów siatki.
    h, L, T, H, g : float
        Parametry fizyczne fali.
    duration : float, optional
        Czas trwania animacji w sekundach.
    fps : int, optional
        Liczba klatek na sekundę.
    particles : int, optional
        Liczba cząstek do wizualizacji.
    """
    # Liczba klatek i parametry fali
    num_frames = int(duration * fps)
    k = 2 * np.pi / L
    omega = 2 * np.pi / T
    
    # Inicjalizacja cząstek płynu
    particle_x = np.linspace(0, L, particles)
    particle_z = np.linspace(-h, -0.05, 5)  # Różne głębokości
    particle_X, particle_Z = np.meshgrid(particle_x, particle_z)
    particle_X = particle_X.flatten()
    particle_Z = particle_Z.flatten()
    
    # Początkowe pozycje cząstek
    initial_positions = np.column_stack((particle_X, particle_Z))
    
    # Inicjalizacja figury
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), 
                                  gridspec_kw={'height_ratios': [3, 1]})
    
    # Funkcja aktualizująca animację
    def update(frame):
        t = (frame / num_frames) * T  # czas w sekundach
        
        # Czyszczenie wykresów
        ax1.clear()
        ax2.clear()
        
        # Obliczanie pozycji cząstek w czasie t
        particle_positions = []
        
        for x0, z0 in initial_positions:
            # Przesunięcie cząstki zgodnie z polem prędkości
            # Używamy rozwiązania analitycznego dla prędkości
            amplitude_x = (g * H * k / (2 * omega**2)) * (np.cosh(k * (z0 + h)) / np.cosh(k * h))
            amplitude_z = (g * H * k / (2 * omega**2)) * (np.sinh(k * (z0 + h)) / np.cosh(k * h))
            
            x = x0 + amplitude_x * np.cos(k * x0 - omega * t)
            z = z0 + amplitude_z * np.sin(k * x0 - omega * t)
            
            # Zawijanie cząstek, które wychodzą poza obszar
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
        
        # Rysowanie powierzchni wody
        ax1.plot(surface_x, surface_z, 'b-', linewidth=3)
        
        # Rysowanie cząstek
        particle_x, particle_z = zip(*particle_positions)
        ax1.scatter(particle_x, particle_z, c='red', s=25, alpha=0.8)
        
        # Ustawienia wykresu fali
        ax1.set_title(f'Falowanie i ruch cząstek, t = {t:.2f} s')
        ax1.set_xlabel('x [m]')
        ax1.set_ylabel('z [m]')
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)  # powierzchnia morza (średni poziom)
        ax1.axhline(y=-h, color='black', linestyle='-', alpha=0.7)  # dno morza
        ax1.set_xlim(0, L)
        ax1.set_ylim(-h, H)
        ax1.grid(alpha=0.3)
        
        # Wykres profilu powierzchni fali
        ax2.plot(surface_x, surface_z, 'b-', linewidth=3)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)  # średni poziom wody
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
    
    # Zapisanie animacji do pliku
    try:
        ani.save('wave_animation.gif', writer='pillow', fps=fps)
        print("Animacja zapisana jako 'wave_animation.gif'")
    except Exception as e:
        print(f"Nie udało się zapisać animacji: {e}")
    
    plt.show()
    return ani
def test_gauss_elimination():
    """Rozbudowane testy eliminacji Gaussa dla zadania 1"""
    print("\n" + "="*80)
    print("ROZSZERZONE TESTY IMPLEMENTACJI ELIMINACJI GAUSSA")
    print("="*80)
    
    # Test 1: Prosta macierz 2x2
    A1 = np.array([[2, 1], [1, 2]], dtype=float)
    b1 = np.array([4, 5], dtype=float)
    expected1 = np.array([1, 2])
    print("\nTest 1: Prosta macierz 2x2")
    x1 = gauss_elimination_with_partial_pivoting(A1, b1)
    print("Rozwiązanie:", x1)
    print("Oczekiwane:", expected1)
    print("Błąd:", np.linalg.norm(x1 - expected1))
    
    # Test 2: Wymagająca zamiany wierszy
    A2 = np.array([[0, 1, 1], [1, 1, 1], [2, 3, 4]], dtype=float)
    b2 = np.array([3, 4, 9], dtype=float)
    expected2 = np.array([1, 1, 2])
    print("\nTest 2: Wymagająca zamiany wierszy")
    x2 = gauss_elimination_with_partial_pivoting(A2, b2)
    print("Rozwiązanie:", x2)
    print("Oczekiwane:", expected2)
    print("Błąd:", np.linalg.norm(x2 - expected2))
    
    # Test 3: Macierz osobliwa
    A3 = np.array([[1, 2], [2, 4]], dtype=float)
    b3 = np.array([3, 6], dtype=float)
    print("\nTest 3: Macierz osobliwa")
    try:
        x3 = gauss_elimination_with_partial_pivoting(A3, b3)
        print("BŁĄD: Powinien zgłosić wyjątek dla macierzy osobliwej!")
    except ValueError as e:
        print("SUKCES: Zgłoszono wyjątek:", str(e))
    
    # Test 4: Duża macierz losowa
    np.random.seed(42)
    size = 20  # Zmniejszono dla szybkości testów
    A4 = np.random.rand(size, size) + 10*np.eye(size)
    b4 = np.random.rand(size)
    expected4 = np.linalg.solve(A4, b4)
    print("\nTest 4: Duża macierz losowa 20x20")
    x4 = gauss_elimination_with_partial_pivoting(A4.copy(), b4.copy())
    error = np.linalg.norm(x4 - expected4)
    print("Błąd:", error)
    print("Czy błąd < 1e-10?", error < 1e-10)

def main():
    # Parametry fizyczne
    h = 10.0      # głębokość morza [m]
    L = 50.0      # długość fali [m]
    T = 5.0       # okres fali [s]
    H = 0.5        # wysokość fali [m]
    g = 9.81       # przyspieszenie ziemskie [m/s²]
    
    print("=== IMPLEMENTACJA METODY ELIMINACJI GAUSSA Z CZĘŚCIOWYM WYBOREM ELEMENTU PODSTAWOWEGO ===")
    
    # Zadanie 1: Demonstracja eliminacji Gaussa
    print("\n[Zadanie 1] Demonstracja eliminacji Gaussa")
    
    # Przykład z treści zadania
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
    N = 30  # mniejsze N dla szybszej demonstracji
    
    print(f"Parametry: N={N}, h={h}m, L={L}m, T={T}s, H={H}m, g={g}m/s²")
    
    # Budowanie i rozwiązywanie układu równań
    A, b, point_to_index = build_wave_equations(N, h, L, T, H, g)
    
    # Zadanie 3: Analiza macierzy rzadkiej
    print("\n[Zadanie 3] Analiza macierzy rzadkiej")
    sparse_matrix_stats(A)
    
    # Rozwiązanie układu
    phi_values = gauss_elimination_with_partial_pivoting(A, b, use_sparse=True)
    phi = {point: phi_values[idx] for point, idx in point_to_index.items()}
    
    # Porównanie z rozwiązaniem analitycznym
    max_error, mean_error = compare_with_analytic(phi, h, L, T, H, g)
    print(f"\nBłąd maksymalny: {max_error:.6e}")
    print(f"Błąd średni: {mean_error:.6e}")
    
    # Analiza błędów
    print("\nAnaliza rozkładu błędów:")
    analyze_errors(phi, h, L, T, H, g)
    
    # Zadanie 4: Porównanie metod
    print("\n[Zadanie 4] Porównanie z implementacją biblioteczną")
    compare_methods([5, 8, 10], h, L, T, H, g)  # Zmniejszamy N do bezpiecznych wartości
    
    visualize_potential(phi,h,L)
    
    # Zadanie 5: Animacja
    print("\n[Zadanie 5] Tworzenie animacji")
    choice = input("Czy chcesz utworzyć animację? (t/n): ")
    if choice.lower() == 't':
        animate_wave_with_particles(N, h, L, T, H, g, duration=2.0, fps=15)
    
    print("\nProgram zakończony.")

if __name__ == "__main__":
    main()



