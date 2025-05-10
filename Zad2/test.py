import numpy as np
from scipy.sparse import csr_matrix, spsolve
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class SparseMatrix:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.data = {}
    
    def set(self, i, j, val):
        if val:
            self.data[(i,j)] = val
        elif (i,j) in self.data:
            del self.data[(i,j)]
    
    def to_csr(self):
        r, c, v = [], [], []
        for (i,j), val in self.data.items():
            r.append(i)
            c.append(j)
            v.append(val)
        return csr_matrix((v, (r, c)), shape=(self.rows, self.cols))

def solve_linear_system(A, b, sparse=False):
    n = len(b)
    
    if sparse:
        if not isinstance(A, SparseMatrix):
            A_sparse = SparseMatrix(n, n)
            for i in range(n):
                for j in range(n):
                    val = A[i,j]
                    if val:
                        A_sparse.set(i, j, val)
            A = A_sparse
        
        for i in range(n):
            pivot_row = i
            for k in range(i+1, n):
                if abs(A.data.get((k,i), 0)) > abs(A.data.get((pivot_row,i), 0)):
                    pivot_row = k
            
            if pivot_row != i:
                for j in range(n):
                    A.data[(i,j)], A.data[(pivot_row,j)] = A.data.get((pivot_row,j), 0), A.data.get((i,j), 0)
                b[i], b[pivot_row] = b[pivot_row], b[i]
            
            pivot = A.data.get((i,i), 0)
            if not pivot:
                raise ValueError("Matrix is singular")
            
            for j in range(i+1, n):
                factor = A.data.get((j,i), 0) / pivot
                if factor:
                    for k in range(i+1, n):
                        A.data[(j,k)] = A.data.get((j,k), 0) - factor * A.data.get((i,k), 0)
                    b[j] -= factor * b[i]
        
        x = np.zeros(n)
        for i in reversed(range(n)):
            x[i] = (b[i] - sum(A.data.get((i,j), 0)*x[j] for j in range(i+1, n))) / A.data.get((i,i), 1)
        return x
    
    else:
        return np.linalg.solve(A, b)

def setup_wave_problem(grid_size, depth, wavelength, period, height, gravity, time=0):
    k = 2 * np.pi / wavelength
    ω = 2 * np.pi / period
    dx = wavelength / grid_size
    dz = depth / grid_size
    
    points = [(i*dx, -depth + j*dz) 
             for i in range(grid_size+1) 
             for j in range(grid_size+1) 
             if -depth <= -depth + j*dz <= 0 and 0 <= i*dx <= wavelength]
    
    pt_idx = {pt:i for i,pt in enumerate(points)}
    n = len(points)
    A = SparseMatrix(n, n)
    b = np.zeros(n)
    
    for pt, i in pt_idx.items():
        x, z = pt
        ε = 1e-10
        
        if z == 0:  # Surface
            below = (x, z-dz)
            if below in pt_idx:
                j = pt_idx[below]
                A.set(i, i, gravity + ε)
                A.set(i, j, -gravity)
                b[i] = -gravity * (gravity * height / (2 * ω)) * np.sin(k*x - ω*time) * dz
            else:
                A.set(i, i, 1 + ε)
                b[i] = (gravity * height / (2 * ω)) * np.sin(k*x - ω*time)
        
        elif z == -depth:  # Bottom
            above = (x, z+dz)
            if above in pt_idx:
                j = pt_idx[above]
                A.set(i, i, -1 + ε)
                A.set(i, j, 1)
            else:
                A.set(i, i, 1 + ε)
                b[i] = (gravity * height / (2 * ω)) * (np.cosh(k*(z+depth))/np.cosh(k*depth)) * np.sin(k*x - ω*time)
        
        elif x == 0 or x == wavelength:  # Sides
            A.set(i, i, 1 + ε)
            b[i] = (gravity * height / (2 * ω)) * (np.cosh(k*(z+depth))/np.cosh(k*depth)) * np.sin(k*x - ω*time)
        
        else:  # Interior
            neighbors = []
            for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                neighbor = (x + di*dx, z + dj*dz)
                if neighbor in pt_idx:
                    neighbors.append(pt_idx[neighbor])
            
            A.set(i, i, -len(neighbors) + ε)
            for j in neighbors:
                A.set(i, j, 1)
    
    return A, b, pt_idx

def animate_waves(grid_size=30, depth=10, wavelength=50, period=5, height=0.5, gravity=9.81, frames=30):
    k = 2 * np.pi / wavelength
    ω = 2 * np.pi / period
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    particles = np.column_stack((
        np.random.uniform(0, wavelength, 100),
        np.random.uniform(-depth, -0.1, 100)
    ))
    
    def update(t):
        ax1.clear()
        ax2.clear()
        
        t = t / frames * period
        surface_x = np.linspace(0, wavelength, 100)
        surface_z = (height/2) * np.cos(k*surface_x - ω*t)
        
        # Update particle positions
        x = particles[:,0]
        z = particles[:,1]
        x += (gravity * height * k / (2 * ω**2)) * (np.cosh(k*(z+depth))/np.cosh(k*depth)) * np.cos(k*x - ω*t)
        z += (gravity * height * k / (2 * ω**2)) * (np.sinh(k*(z+depth))/np.cosh(k*depth)) * np.sin(k*x - ω*t)
        
        # Boundary conditions
        x[x < 0] += wavelength
        x[x > wavelength] -= wavelength
        z[z < -depth] = -depth
        z[z > 0] = 0
        
        ax1.fill_between(surface_x, surface_z, -depth, color='lightblue')
        ax1.scatter(x, z, color='red', s=10)
        ax1.set_xlim(0, wavelength)
        ax1.set_ylim(-depth, height)
        
        ax2.plot(surface_x, surface_z)
        ax2.set_xlim(0, wavelength)
        ax2.set_ylim(-height, height)
        
        return ax1, ax2
    
    ani = FuncAnimation(fig, update, frames=frames, blit=False)
    plt.tight_layout()
    return ani

if __name__ == "__main__":
    # Quick demo
    A, b, mapping = setup_wave_problem(20, 10, 50, 5, 0.5, 9.81)
    solution = solve_linear_system(A, b, sparse=True)
    
    # Visualize
    x = [pt[0] for pt in mapping]
    z = [pt[1] for pt in mapping]
    plt.tricontourf(x, z, solution, levels=20)
    plt.colorbar()
    plt.show()
    
    # Animate
    anim = animate_waves()
    plt.show()