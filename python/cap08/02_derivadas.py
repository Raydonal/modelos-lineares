"""Cap. 8: Derivadas matriciais conferidas por diferenças finitas."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.gerador import Gerador  # noqa: E402

g = Gerador(80); n = 4
B = g.matriz_normal(6, n); X = B.T @ B + np.eye(n); dX = g.matriz_normal(n, n); h = 1e-6
df = lambda F: (F(X + h * dX) - F(X - h * dX)) / (2 * h)
Xi = np.linalg.inv(X)
print("log det:", df(lambda M: np.linalg.slogdet(M)[1]) - np.trace(Xi @ dX))
print("det (relativo):", (df(np.linalg.det) - np.linalg.det(X) * np.trace(Xi @ dX)) / np.linalg.det(X))
print("inversa:", np.abs(df(np.linalg.inv) + Xi @ dX @ Xi).max())
S, dS = X, dX + dX.T
lam = lambda M: np.linalg.eigvalsh(M)[-1]
q1 = np.linalg.eigh(S)[1][:, -1]
print(f"d lambda_1: numérica = {(lam(S + h * dS) - lam(S - h * dS)) / (2 * h):.6f}, fórmula = {q1 @ dS @ q1:.6f}")
