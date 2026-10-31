"""Cap. 6: Autovalores: traço, determinante, AB e BA, diagonalização."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.gerador import Gerador  # noqa: E402

A = np.array([[2.0, 1, 0], [1, 3, 1], [0, 1, 4]])
lam = np.linalg.eigvals(A).real
print("autovalores:", np.sort(lam)[::-1].round(6))
print(f"soma = {lam.sum():.6f}, traço = {np.trace(A)}, produto = {lam.prod():.6f}, det = {np.linalg.det(A):.6f}")
print("rotação:", np.linalg.eigvals(np.array([[0.0, 1], [-1, 0]])))
g = Gerador(6)
B, C = g.matriz_normal(3, 5), g.matriz_normal(5, 3)
print("BC:", np.sort(np.linalg.eigvals(B @ C).real)[::-1].round(6))
print("CB:", np.sort(np.linalg.eigvals(C @ B).real)[::-1].round(6))
J = np.array([[1.0, 1], [0, 1]])
val, vec = np.linalg.eig(J)
print("Jordan: autovalores", val, " posto dos autovetores:", np.linalg.matrix_rank(vec, tol=1e-6))
P = np.array([[2.0, -2, -4], [-1, 3, 4], [1, -2, -3]])
print("idempotente:", np.sort(np.linalg.eigvals(P).real).round(10))
