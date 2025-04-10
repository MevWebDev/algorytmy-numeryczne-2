import numpy as np
import matplotlib.pyplot as plt
from zad1WERSJA2 import gauss_elimination_with_partial_pivoting

# Parametry zadania
L = 1.0
h_depth = 1.0
N = 10
M = 10
g = 9.81
omega = 2.0
H = 1  # Amplituda fali
k = 4 * np.pi / L  # Liczba falowa

# Budowa siatki
x_points = np.linspace(0, L, N + 1)
z_points = np.linspace(-h_depth, 0, M + 1)
X, Z = np.meshgrid(x_points, z_points)
total_points = (N + 1) * (M + 1)

# Budowa macierzy A i wektora b
A = np.zeros((total_points, total_points))
b = np.zeros(total_points)

h_x = L / N
h_z = h_depth / M

for i in range(N + 1):
    for j in range(M + 1):
        idx = j * (N + 1) + i

        if j == 0:  # Warunek na dnie
            A[idx, idx] = 1.0
            A[idx, idx + (N + 1)] = -1.0
            b[idx] = 0.0
        elif j == M:  # Warunek na powierzchni (z warunkiem początkowym)
            A[idx, idx] = -omega**2 + g / h_z
            A[idx, idx - (N + 1)] = -g / h_z
            b[idx] = -omega**2 * H * np.sin(k * x_points[i])
        elif i == 0 or i == N:  # Warunek na bokach
            A[idx, idx] = 1.0
            b[idx] = 0.0
        elif 0 < i < N and 0 < j < M:  # Równanie Laplace'a wewnątrz
            A[idx, idx] = -2 / h_x**2 - 2 / h_z**2
            A[idx, idx + 1] = 1 / h_x**2
            A[idx, idx - 1] = 1 / h_x**2
            A[idx, idx + (N + 1)] = 1 / h_z**2
            A[idx, idx - (N + 1)] = 1 / h_z**2
            b[idx] = 0.0

# Rozwiązanie układu równań
equations = []
for i in range(total_points):
    equation = " ".join(map(str, A[i])) + " " + str(b[i])
    equations.append(equation)

solution, original_matrix, errors = gauss_elimination_with_partial_pivoting(equations)

# Wizualizacja wyniku
if solution is not None:
    phi_grid = solution.reshape((M + 1, N + 1))

    plt.contourf(X, Z, phi_grid, levels=20, cmap='viridis')
    plt.colorbar(label='Potencjał φ(x,z)')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title('Rozwiązanie równania falowego')
    plt.show()
else:
    print("Nie udało się rozwiązać układu równań!")