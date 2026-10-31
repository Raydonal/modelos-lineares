"""Cap. 11: Modelo multivariado: a forma vetorizada com Kronecker"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.gerador import Gerador  # noqa: E402

g = Gerador(11); n = 50
X = np.column_stack([np.ones(n), g.matriz_normal(n, 2)])
B = np.array([[1, 0], [2, 0.5], [-1, 3.0]])   # duas respostas
L = np.linalg.cholesky(np.array([[1, 0.8], [0.8, 1]]))
E = g.matriz_normal(n, 2) @ L.T               # linhas ~ N(0, Sigma)
Y = X @ B + E
B_mq = np.linalg.solve(X.T @ X, X.T @ Y)      # coluna a coluna
Xv = np.kron(np.eye(2), X)                    # vec Y = (I(x)X) vec B
S = E.T @ E / n                               # qualquer Sigma > 0
W = np.kron(np.linalg.inv(S), np.eye(n))      # Cov(vec E)^{-1}
yv = Y.T.reshape(-1)                          # vec empilha colunas
b_g = np.linalg.solve(Xv.T @ W @ Xv, Xv.T @ W @ yv)
dif = np.max(np.abs(b_g - B_mq.T.reshape(-1)))
print(f"dif. MQG - MQ: {dif:.1e}")
print(B_mq.round(3))
