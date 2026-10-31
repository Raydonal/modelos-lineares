"""Cap. 2 -- Fatoração LU com e sem pivotamento; substituições progressiva e regressiva."""
import numpy as np
import scipy.linalg as sla


def lu_sem_pivo(A):
    U = np.array(A, dtype=float); n = len(U); L = np.eye(n)
    for k in range(n - 1):
        if U[k, k] == 0:
            raise ZeroDivisionError(f"pivô nulo na etapa {k + 1}")
        L[k + 1:, k] = U[k + 1:, k] / U[k, k]              # multiplicadores
        U[k + 1:] -= np.outer(L[k + 1:, k], U[k])          # elimina abaixo do pivô
    return L, U


def progressiva(L, b):
    z = np.zeros(len(b))
    for i in range(len(b)):
        z[i] = (b[i] - L[i, :i] @ z[:i]) / L[i, i]
    return z


def regressiva(U, z):
    n = len(z); x = np.zeros(n)
    for i in reversed(range(n)):
        x[i] = (z[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


A = np.array([[2, 1, 1, 0], [4, 3, 3, 1], [8, 7, 9, 5], [6, 7, 9, 8]], dtype=float)
L, U = lu_sem_pivo(A)
print(L, U, np.abs(L @ U - A).max(), sep="\n")
b = np.array([1.0, 2, 3, 4])
x = regressiva(U, progressiva(L, b))
print(x, np.abs(A @ x - b).max())

E = np.array([[1e-20, 1], [1, 1]]); bE = np.array([1.0, 2])
Lg, Ug = lu_sem_pivo(E)
P, Lp, Up = sla.lu(E)                                       # scipy: E = P L U
print("sem pivô:", regressiva(Ug, progressiva(Lg, bE)))
print("com pivô:", regressiva(Up, progressiva(Lp, P.T @ bE)))
print("solve:   ", np.linalg.solve(E, bE))
