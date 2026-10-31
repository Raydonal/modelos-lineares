"""Cap. 4 -- Compatibilidade e solução geral de A x = b (e de A X C = B)."""
import numpy as np

A = np.array([[1, 2, 0, 1], [2, 4, 1, 1], [3, 6, 1, 2]], dtype=float)
G = np.linalg.pinv(A)
compativel = lambda b: np.allclose(A @ G @ b, b)
b1, b2 = np.array([4.0, 8, 12]), np.array([1.0, 0, 0])
print("compatível:", compativel(b1), compativel(b2))

rng = np.random.default_rng(1)
Xsol = np.column_stack([G @ b1 + (np.eye(4) - G @ A) @ rng.normal(size=4) for _ in range(5)])
print(np.round(Xsol, 3))
print("erro máximo:", np.abs(A @ Xsol - b1[:, None]).max(), " dim N(A):",
      np.linalg.matrix_rank(np.eye(4) - G @ A, tol=1e-10))
print("k no espaço linha:", np.round(np.array([2, 4, 1, 1]) @ Xsol, 6))
print("k fora          :", np.round(np.array([1, 0, 0, 0]) @ Xsol, 3))

C = np.array([[1, 2], [2, 4]], dtype=float)
B = A @ np.arange(1, 9, dtype=float).reshape(2, 4).T @ C
Gc = np.linalg.pinv(C)
Z = rng.normal(size=(4, 2))
Xs = G @ B @ Gc + Z - G @ A @ Z @ C @ Gc
print("A X C = B:", np.abs(A @ G @ B @ Gc @ C - B).max(), np.abs(A @ Xs @ C - B).max())
