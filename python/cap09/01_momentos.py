"""Cap. 9: Momentos: Var(Ax), viés de A/n e E(x'Ax)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.gerador import Gerador  # noqa: E402

mu = np.array([1.0, 2, 0])
Sigma = np.array([[4.0, 2, 1], [2, 3, -1], [1, -1, 2]])
L = np.linalg.cholesky(Sigma)          # Sigma = L L'
A = np.array([[1.0, 1, 0], [0, 1, -1]])
print(A @ Sigma @ A.T)                 # Var(Ax) = A Sigma A'

# 20 000 amostras de tamanho n = 5 (normais, via x = mu + L z)
g = Gerador(9); R, n = 20000, 5
soma_S = np.zeros((3, 3)); soma_q = 0.0; B = np.diag([1.0, 2, 3])
for r in range(R):
    X = g.matriz_normal(n, 3) @ L.T + mu   # linhas: x_i'
    Xc = X - X.mean(0)
    soma_S += Xc.T @ Xc / n            # estimador de MV: A / n
    soma_q += np.sum((X @ B) * X) / n  # média de x_i' B x_i
print((soma_S / R).round(3))           # perto de (n - 1)/n Sigma
print((n - 1) / n * Sigma)
print("simulado:", soma_q / R,
      " teórico:", np.trace(B @ Sigma) + mu @ B @ mu)
