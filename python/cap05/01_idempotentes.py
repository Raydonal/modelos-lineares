"""Cap. 5 -- Matrizes idempotentes: posto = traço, identidade de posto, potências."""
import numpy as np

posto = lambda M: np.linalg.matrix_rank(M, tol=1e-10)

A1 = np.array([[1.0, 1.0], [0.0, 0.0]])
C5 = np.eye(5) - np.full((5, 5), 1 / 5)
A = np.array([[1, 2, 0, 1], [2, 4, 1, 1], [3, 6, 1, 2]], float)
G = np.zeros((4, 3)); G[np.ix_([0, 2], [0, 1])] = np.linalg.inv(A[np.ix_([0, 1], [0, 2])])
for nome, M in {"oblíqua": A1, "centragem": C5, "AG": A @ G}.items():
    print(f"{nome:10s} erro_idemp={np.abs(M @ M - M).max():.1e} simétrica={np.allclose(M, M.T)!s:5s} "
          f"posto={posto(M)} traço={np.trace(M):.0f}")

rng = np.random.default_rng(5)
n, r = 6, 2
B = rng.normal(size=(n, r)); C = rng.normal(size=(r, n))
P = B @ np.linalg.solve(C @ B, C)                     # B (C B)^{-1} C
print("idempotente construída:", posto(P), round(np.trace(P), 10), posto(np.eye(n) - P))

M = P + 0.1 * rng.normal(size=(n, n))
print("identidade de posto:", posto(M) + posto(np.eye(n) - M), "=", n + posto(M - M @ M))

Pm = np.array([[0.9, 0.1], [0.3, 0.7]])
L = np.linalg.matrix_power(Pm, 200)
print(L.round(6), "\nerro L^2 - L:", np.abs(L @ L - L).max(), " posto:", posto(L))
