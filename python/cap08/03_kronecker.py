"""Cap. 8: Produto de Kronecker, vec e vech."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.gerador import Gerador  # noqa: E402

g = Gerador(81)
A, B, C, D = g.matriz_normal(2, 3), g.matriz_normal(4, 2), g.matriz_normal(3, 5), g.matriz_normal(2, 3)
print("produto misto:", np.abs(np.kron(A, B) @ np.kron(C, D) - np.kron(A @ C, B @ D)).max())
vec = lambda M: M.reshape(-1, order="F")                     # empilha colunas
X, A1, C1 = g.matriz_normal(3, 3), g.matriz_normal(4, 3), g.matriz_normal(3, 2)
print("vec(AXC) = (C' kron A) vec X:", np.abs(vec(A1 @ X @ C1) - np.kron(C1.T, A1) @ vec(X)).max())
Bm = A1 @ X @ C1; Xs = np.linalg.pinv(A1) @ Bm @ np.linalg.pinv(C1)
print("A Xs C = B:", np.abs(A1 @ Xs @ C1 - Bm).max(), " igual à via vec:", np.abs(vec(Xs) - np.linalg.pinv(np.kron(C1.T, A1)) @ vec(Bm)).max())
P = g.matriz_normal(3, 3); P = P.T @ P + np.eye(3); Q = g.matriz_normal(2, 2); Q = Q.T @ Q + np.eye(2)
ev = np.sort(np.linalg.eigvalsh(np.kron(P, Q)))
print("autovalores do Kronecker = produtos:", np.abs(ev - np.sort(np.outer(np.linalg.eigvalsh(P), np.linalg.eigvalsh(Q)).ravel())).max())
