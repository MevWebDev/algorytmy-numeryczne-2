import numpy as np

def gauss_elimination_with_partial_pivoting(A, b):
    """
    Rozwiązuje układ równań Ax = b metodą eliminacji Gaussa z częściowym wyborem elementu podstawowego
    
    Parametry:
    A - macierz współczynników (n x n)
    b - wektor wyrazów wolnych (n)
    
    Zwraca:
    x - wektor rozwiązania (n)
    """
    n = len(b)
    
    # Połączenie macierzy A i wektora b w macierz rozszerzoną
    Ab = np.hstack([A.astype(float), b.reshape(n, 1).astype(float)])
    
    # Eliminacja Gaussa
    for i in range(n):
        # Częściowy wybór elementu podstawowego
        max_row = i + np.argmax(np.abs(Ab[i:, i]))
        Ab[[i, max_row]] = Ab[[max_row, i]]  # Zamiana wierszy
        
        # Sprawdzenie czy macierz jest osobliwa
        if np.abs(Ab[i, i]) < 1e-12:
            raise ValueError("Macierz jest osobliwa lub źle uwarunkowana")
            
        # Eliminacja dla wierszy poniżej i-tego
        for j in range(i + 1, n):
            factor = Ab[j, i] / Ab[i, i]
            Ab[j, i:] -= factor * Ab[i, i:]
    
    # Postępowanie odwrotne (back substitution)
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:n])) / Ab[i, i]
    
    return x

# Przykład użycia
if __name__ == "__main__":
    # Przykład z równania (1) z PDF: sin(t)z'(t) - cos(t)z(t) = -1
    # Dla N=6 mamy układ równań przedstawiony w PDF (tutaj uproszczony przykład)
    
    # Tworzymy przykładową macierz rzadką (tridiagonalną)
    N = 6
    A = np.zeros((N+1, N+1))
    
    # Wypełnienie macierzy zgodnie z przykładem z PDF (uproszczone)
    h = np.pi/2 / N
    for k in range(1, N):
        t_k = k*h
        A[k, k-1] = np.sin(t_k)
        A[k, k] = 2*h*np.cos(t_k)
        A[k, k+1] = -np.sin(t_k)
    
    # Warunki brzegowe
    A[0, 0] = 1  # z(0) = 1
    A[N, N] = 1   # z(pi/2) = 0
    
    # Wektor prawych stron
    b = np.zeros(N+1)
    b[1:N] = 2*h  # Dla równań środkowych
    b[0] = 1      # Warunek brzegowy z(0) = 1
    b[N] = 0      # Warunek brzegowy z(pi/2) = 0
    
    # Rozwiązanie układu
    try:
        z = gauss_elimination_with_partial_pivoting(A, b)
        print("Rozwiązanie z w punktach siatki:")
        print(z)
        
        # Porównanie z rozwiązaniem analitycznym z(t) = cos(t)
        t_points = np.linspace(0, np.pi/2, N+1)
        z_analytic = np.cos(t_points)
        print("\nRozwiązanie analityczne cos(t):")
        print(z_analytic)
        
        print("\nBłąd maksymalny:", np.max(np.abs(z - z_analytic)))
    except ValueError as e:
        print(e)