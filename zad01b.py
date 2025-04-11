import numpy as np

def gauss_elimination_with_partial_pivoting():
    # Wczytanie liczby niewiadomych
    n = int(input("Podaj liczbę niewiadomych: "))
    
    # Inicjalizacja macierzy rozszerzonej
    a = np.zeros((n, n+1))
    
    # Wczytanie współczynników
    print("Wprowadź współczynniki równań (oddzielone spacją, np. '1 2 3' dla 1x + 2y = 3):")
    for i in range(n):
        while True:
            try:
                row = list(map(float, input(f"Równanie {i+1}: ").split()))
                if len(row) != n + 1:
                    print(f"Błąd: Wprowadź dokładnie {n+1} liczby (współczynniki + wynik)")
                    continue
                a[i] = row
                break
            except ValueError:
                print("Błąd: Wprowadź tylko liczby oddzielone spacjami")
    
    # Eliminacja Gaussa z częściowym wyborem elementu podstawowego
    for i in range(n):
        # Częściowy wybór elementu podstawowego
        max_row = i
        for k in range(i+1, n):
            if abs(a[k][i]) > abs(a[max_row][i]):
                max_row = k
        
        # Zamiana wierszy
        a[[i, max_row]] = a[[max_row, i]]
        
        if a[i][i] == 0.0:
            print("Macierz jest osobliwa - układ może nie mieć rozwiązania")
            return None
        
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
    
    return x

# Uruchomienie funkcji i wyświetlenie wyniku
solution = gauss_elimination_with_partial_pivoting()
if solution is not None:
    print("\nRozwiązanie:")
    for i, val in enumerate(solution):
        print(f"x{i+1} = {val:.4f}")