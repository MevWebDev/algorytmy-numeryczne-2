import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cg

# Parametry zadania
L = 1.0
h_depth = 1.0
N = 50
M = 50
g = 9.81
omega = 2.0
H = 0.1  # Amplituda fali
k = 2 * np.pi / L  # Liczba falowa

# Budowa siatki
x_points = np.linspace(0, L, N + 1)
z_points = np.linspace(-h_depth, 0, M + 1)
X, Z = np.meshgrid(x_points, z_points)
total_points = (N + 1) * (M + 1)

# Budowa macierzy A i wektora b (użycie macierzy rzadkich)
rows = []
cols = []
data = []
b = np.zeros(total_points)

h_x = L / N
h_z = h_depth / M

for i in range(N + 1):
    for j in range(M + 1):
        idx = j * (N + 1) + i

        if j == 0:  # Warunek na dnie
            rows.append(idx)
            cols.append(idx)
            data.append(1.0)
            rows.append(idx)
            cols.append(idx + (N + 1))
            data.append(-1.0)
            b[idx] = 0.0
        elif j == M:  # Warunek na powierzchni (z warunkiem początkowym)
            rows.append(idx)
            cols.append(idx)
            data.append(-omega**2 + g / h_z)
            rows.append(idx)
            cols.append(idx - (N + 1))
            data.append(-g / h_z)
            b[idx] = -omega**2 * H * np.sin(k * x_points[i])
        elif i == 0 or i == N:  # Warunek na bokach
            rows.append(idx)
            cols.append(idx)
            data.append(1.0)
            b[idx] = 0.0
        elif 0 < i < N and 0 < j < M:  # Równanie Laplace'a wewnątrz
            rows.append(idx)
            cols.append(idx)
            data.append(-2 / h_x**2 - 2 / h_z**2)
            rows.append(idx)
            cols.append(idx + 1)
            data.append(1 / h_x**2)
            rows.append(idx)
            cols.append(idx - 1)
            data.append(1 / h_x**2)
            rows.append(idx)
            cols.append(idx + (N + 1))
            data.append(1 / h_z**2)
            rows.append(idx)
            cols.append(idx - (N + 1))
            data.append(1 / h_z**2)
            b[idx] = 0.0

A = csr_matrix((data, (rows, cols)), shape=(total_points, total_points))

# Rozwiązanie układu równań (użycie metody gradientów sprzężonych)
solution, info = cg(A, b)

# Wizualizacja wyniku
if info == 0:
    phi_grid = solution.reshape((M + 1, N + 1))

    plt.contourf(X, Z, phi_grid, levels=20, cmap='viridis')
    plt.colorbar(label='Potencjał φ(x,z)')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title('Rozwiązanie równania falowego')
    plt.show()
else:
    print("Nie udało się rozwiązać układu równań!")