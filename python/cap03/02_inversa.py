"""Cap. 3 -- Inversa por Gauss-Jordan, pela adjunta e regra de Cramer."""
import numpy as np


def gauss_jordan_inv(A):
    n = len(A); M = np.hstack([A.astype(float), np.eye(n)])
    for j in range(n):
        k = j + np.argmax(np.abs(M[j:, j]))
        if abs(M[k, j]) < 1e-12:
            raise np.linalg.LinAlgError("matriz singular")
        M[[j, k]] = M[[k, j]]
        M[j] /= M[j, j]
        for i in range(n):
            if i != j:
                M[i] -= M[i, j] * M[j]
    return M[:, n:]


def adjunta(A):
    n = len(A); C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            menor = np.delete(np.delete(A, i, 0), j, 1)
            C[i, j] = (-1) ** (i + j) * np.linalg.det(menor)
    return C.T


A = np.array([[2, 1, 1], [1, 3, 2], [1, 0, 0]], dtype=float)
Ainv = gauss_jordan_inv(A)
print(np.round(Ainv, 4))
print(np.abs(Ainv - np.linalg.inv(A)).max(), np.abs(adjunta(A) / np.linalg.det(A) - Ainv).max())
b = np.array([4.0, 5, 6])
x = [np.linalg.det(np.column_stack([b if k == j else A[:, k] for k in range(3)])) / np.linalg.det(A)
     for j in range(3)]
print(np.round(x, 10), np.linalg.solve(A, b))
