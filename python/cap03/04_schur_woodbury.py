"""Cap. 3: Complemento de Schur, inversa particionada e identidade de Woodbury."""
import time

import numpy as np

rng = np.random.default_rng(2)
G = rng.normal(size=(6, 6)); M = G.T @ G + np.eye(6)
A, B, C, D = M[:2, :2], M[:2, 2:], M[2:, :2], M[2:, 2:]
S = D - C @ np.linalg.solve(A, B)
print(np.linalg.det(M), np.linalg.det(A) * np.linalg.det(S))
Ai, Si = np.linalg.inv(A), np.linalg.inv(S)
Minv = np.block([[Ai + Ai @ B @ Si @ C @ Ai, -Ai @ B @ Si], [-Si @ C @ Ai, Si]])
print(np.abs(Minv - np.linalg.inv(M)).max())

n, k = 2000, 5
a0 = rng.uniform(1, 2, n); U = rng.normal(size=(n, k)); V = U; W = np.eye(k)
b = rng.normal(size=n)
t0 = time.perf_counter(); x1 = np.linalg.solve(np.diag(a0) + U @ W @ V.T, b); t1 = time.perf_counter()
AU, Ab = U / a0[:, None], b / a0
x2 = Ab - AU @ np.linalg.solve(np.linalg.inv(W) + V.T @ AU, V.T @ Ab); t2 = time.perf_counter()
print(f"diferença = {np.abs(x1 - x2).max():.1e}; direto {t1 - t0:.3f}s; Woodbury {t2 - t1:.4f}s")
s, ld = np.linalg.slogdet(np.diag(a0) + U @ V.T)
print(ld, np.sum(np.log(a0)) + np.linalg.slogdet(np.eye(k) + V.T @ AU)[1])
