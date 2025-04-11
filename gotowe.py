# -*- coding: utf-8 -*-
import numpy as np
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

# =======================================================================
# ZADANIE 1: Eliminacja Gaussa z częściowym pivotem
# =======================================================================
def gauss_elimination(A, b):
    """Rozwiązuje Ax = b z częściowym pivotem
    
    Args:
        A: Macierz współczynników (n x n)
        b: Wektor prawych stron (n)
        
    Returns:
        x: Wektor rozwiązania
    """
    n = len(b)
    Ab = np.hstack([A.astype(float), b.reshape(-1, 1)])  # Łączymy macierz i wektor
    
    for i in range(n):
        # Częściowy pivot
        max_row = np.argmax(np.abs(Ab[i:, i])) + i
        Ab[[i, max_row]] = Ab[[max_row, i]]
        
        if np.abs(Ab[i, i]) < 1e-12:
            raise ValueError("Macierz osobliwa")
            
        # Eliminacja
        for j in range(i+1, n):
            factor = Ab[j, i] / Ab[i, i]
            Ab[j, i:] -= factor * Ab[i, i:]
    
    # Substitucja wsteczna
    x = np.zeros(n)
    for i in reversed(range(n)):
        x[i] = (Ab[i, -1] - Ab[i, i+1:n] @ x[i+1:n]) / Ab[i, i]
    return x

# =======================================================================
# ZADANIE 2: Równanie Laplace'a dla falowania
# =======================================================================
def build_laplace_system(N):
    """Buduje układ równań dla równania Laplace'a na trójkącie"""
    h = 1/N
    size = (N+1)*(N+2)//2  # Liczba punktów w siatce trójkątnej
    A = lil_matrix((size, size))
    b = np.zeros(size)
    
    # Mapowanie punktów (i,j) do indeksu
    node_idx = {}
    idx = 0
    for i in range(N+1):
        for j in range(N+1 - i):
            node_idx[(i, j)] = idx
            idx += 1
            
    # Wypełnianie macierzy A i wektora b
    for (i, j), idx in node_idx.items():
        if i == 0 or j == 0 or i + j == N:  # Warunki brzegowe
            A[idx, idx] = 1
            if i == 0:
                b[idx] = -(j*h)**2  # z(0, y) = -y²
            elif j == 0:
                b[idx] = (i*h)**2   # z(x, 0) = x²
            else:
                b[idx] = 2*i*h - 1  # z(x, 1-x) = 2x -1
        else:
            # Równanie Laplace'a: (u_xx + u_yy = 0)
            A[idx, node_idx[(i+1, j)]] = 1/h**2
            A[idx, node_idx[(i-1, j)]] = 1/h**2
            A[idx, node_idx[(i, j+1)]] = 1/h**2
            A[idx, node_idx[(i, j-1)]] = 1/h**2
            A[idx, idx] = -4/h**2
    return A.tocsr(), b

# =======================================================================
# ZADANIE 3: Optymalizacja macierzy rzadkich (użyto scipy.sparse)
# =======================================================================
# W powyższej funkcji build_laplace_system użyto macierzy rzadkiej lil_matrix
# co automatycznie optymalizuje pamięć

# =======================================================================
# ZADANIE 4: Porównanie z numpy.linalg.solve
# =======================================================================
def compare_solvers(A, b):
    """Porównuje czas i dokładność"""
    # Moja implementacja
    start = time.time()
    x_custom = gauss_elimination(A.toarray(), b)
    t_custom = time.time() - start
    
    # Numpy
    start = time.time()
    x_numpy = np.linalg.solve(A.toarray(), b)
    t_numpy = time.time() - start
    
    error = np.max(np.abs(x_custom - x_numpy))
    return t_custom, t_numpy, error

# =======================================================================
# ZADANIE 5: Animacja
# =======================================================================
def animate_wave(N=10, frames=50):
    """Animacja potencjału w czasie"""
    A, b = build_laplace_system(N)
    x = np.linalg.solve(A.toarray(), b)  # Dla szybkości używamy numpy
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Przygotowanie siatki
    nodes = [(i, j) for i in range(N+1) for j in range(N+1 - i)]
    X, Y = np.array([(i/N, j/N) for (i, j) in nodes]).T
    Z = x
    
    def update(frame):
        ax.cla()
        ax.plot_trisurf(X, Y, Z + 0.1*np.sin(frame/10))  # Prostą animacja
        ax.set_zlim(-1, 1)
        return ax
    
    ani = FuncAnimation(fig, update, frames=frames, interval=100)
    plt.show()

# =======================================================================
# SPRAWDZENIE DZIAŁANIA
# =======================================================================
if __name__ == "__main__":
    # Test Z1-Z4
    A, b = build_laplace_system(N=5)
    t1, t2, err = compare_solvers(A, b)
    print(f"Czas własny: {t1:.4f}s, Czas numpy: {t2:.4f}s, Błąd: {err:.2e}")
    
    # Animacja Z5
    animate_wave(N=5)