# approximate solution to an equation of the form: p(t)z'(t) + q(t)z(t) = f(t)
# on the interval [a, b] using uniform grid size with n-1 intermediate points, h = (B-A)/n

import numpy as np

n = 6
a = 0
b = np.pi/2
za = 1
zb = 0

def f(t):
    return -1

def p(t):
    return np.sin(t)

def q(t):
    return -np.cos(t)

def eqsystem(a, b, za, zb, n, p, q, f):
    h = (b-a)/n
    A = np.zeros((n+1, n+1))
    B = np.zeros(n+1)
    A[0,0]=1
    B[0]=za            # boundary condition
    A[n,n]=1
    B[n]=zb            # boundary condition
    for k in range(1, n):
        tk = a+k*h
        A[k, k-1] = -p(tk)
        A[k, k]   =  2*h*q(tk)
        A[k, k+1] =  p(tk)
        B[k] = f(tk)*2*h
    return (A, B)

(A, B) = eqsystem(a, b, za, zb, n, p, q, f)
Y = np.linalg.solve(A, B)


# compare to cosine function
T = np.linspace(a, b, n+1)
error = np.abs(Y - np.cos(T))
print(np.max(error))
