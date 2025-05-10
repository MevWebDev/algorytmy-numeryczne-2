import numpy as np

def gauss_elimination_with_partial_pivoting(equations):
    """Rozwiązuje układ równań liniowych metodą eliminacji Gaussa z częściowym wyborem elementu podstawowego.
    
    Args:
        equations: Lista stringów reprezentujących równania (np. ["1 2 3", "4 5 6"])
        
    Returns:
        Wektor rozwiązań lub None jeśli układ jest osobliwy
    """
    # Sprawdzenie czy wszystkie równania mają tę samą liczbę współczynników
    n = len(equations)
    rows = [list(map(float, eq.split())) for eq in equations]
    lengths = [len(row) for row in rows]
    
    if len(set(lengths)) != 1:
        raise ValueError("Wszystkie równania muszą mieć tę samą liczbę współczynników")
    
    if lengths[0] != n + 1:
        raise ValueError(f"Oczekiwano {n} niewiadomych - każde równanie powinno mieć {n+1} współczynników")
    
    # Utworzenie macierzy rozszerzonej
    a = np.array(rows)
    original_a = a.copy()  # Zachowaj oryginalną macierz dla późniejszej weryfikacji
    
    # Eliminacja Gaussa z częściowym wyborem elementu podstawowego
    for i in range(n):
        # Częściowy wybór elementu podstawowego
        max_row = i
        for k in range(i+1, n):
            if abs(a[k][i]) > abs(a[max_row][i]):
                max_row = k
        
        # Zamiana wierszy
        a[[i, max_row]] = a[[max_row, i]]
        
        if abs(a[i][i]) < 1e-10:  # Mała tolerancja dla zer
            print("Macierz jest osobliwa - układ może nie mieć rozwiązania")
            return None, None, None
        
        # Eliminacja
        for j in range(i+1, n):
            ratio = a[j][i]/a[i][i]
            a[j] -= ratio * a[i]
    
    # Podstawienie wsteczne
    x = np.zeros(n)
    x[n-1] = a[n-1][n]/a[n-1][n-1]
    
    for i in range(n-2, -1, -1):
        x[i] = a[i][n]
        for j in range(i+1, n):
            x[i] -= a[i][j] * x[j]
        x[i] /= a[i][i]
    
    # Oblicz błędy przez podstawienie rozwiązań do oryginalnych równań
    errors = []
    for i in range(n):
        lhs = np.sum(original_a[i, :-1] * x)
        rhs = original_a[i, -1]
        errors.append(abs(lhs - rhs))
    
    return x, original_a, errors

def print_verification(solutions, original_matrix, errors):
    """Wyświetla weryfikację rozwiązań przez podstawienie do oryginalnych równań."""
    print("\nWeryfikacja rozwiązań:")
    for i in range(len(original_matrix)):
        equation_str = " + ".join([f"{original_matrix[i,j]:.1f}x{j+1}" for j in range(len(solutions))])
        equation_str += f" = {original_matrix[i,-1]:.1f}"
        print(f"Równanie {i+1}: {equation_str}")
        
        substituted_str = " + ".join([f"{original_matrix[i,j]:.1f}·{solutions[j]:.4f}" 
                                    for j in range(len(solutions))])
        calculated = np.sum(original_matrix[i,:-1] * solutions)
        print(f"Podstawienie: {substituted_str} = {calculated:.4f}")
        print(f"Oczekiwano: {original_matrix[i,-1]:.1f}, Błąd: {errors[i]:.10f}")
        print()
    
# Przykład użycia:
if __name__ == "__main__":
    # Przykładowe równania:
    # 1x + 1y + 2z = 8
    # 2x + 3y + 1z = 11
    # 1x - 1y + 1z = 3
    equations = [
        "1 1 2 8",
        "2 3 1 11",
        "1 -1 1 3"
    ]
    
    solution, original_matrix, errors = gauss_elimination_with_partial_pivoting(equations)
    if solution is not None:
        print("\nRozwiązanie:")
        for i, val in enumerate(solution):
            print(f"x{i+1} = {val:.4f}")
        
        print_verification(solution, original_matrix, errors)
        
        # Podsumowanie błędów
        print("\nPodsumowanie błędów:")
        print(f"Maksymalny błąd: {max(errors):.10f}")
        print(f"Średni błąd: {np.mean(errors):.10f}")