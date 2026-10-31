"""Cap. 5 -- A atenção como suavizador: temperatura, projeções e colapso de posto."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.gerador import Gerador  # noqa: E402


def softmax(M):
    E = np.exp(M - M.max(1, keepdims=True))
    return E / E.sum(1, keepdims=True)


g = Gerador(2026)                                            # mesmos sorteios que o R
d = 8; grupo = np.repeat([0, 1, 2], [5, 4, 3]); n = len(grupo)
K = g.matriz_normal(3, d, 1.5)[grupo]                       # chaves iguais dentro do grupo
Q = K + g.matriz_normal(n, d, 0.3)                          # consultas próximas das chaves
G = np.eye(3)[grupo]; PG = G @ np.linalg.solve(G.T @ G, G.T)
for tau in (10, 3, 1, 0.3, 0.1):
    A = softmax(Q @ K.T / (tau * np.sqrt(d)))
    print(f"tau = {tau:4}: |A - P_G| = {np.abs(A - PG).max():.1e}, |A^2 - A| = {np.abs(A @ A - A).max():.1e}, "
          f"tr A = {np.trace(A):.3f}, posto = {np.linalg.matrix_rank(A, tol=1e-8)}")

X = g.matriz_normal(n, d)
A = softmax(X @ X.T / np.sqrt(d))
val, vec = np.linalg.eig(A.T)
pi = np.real(vec[:, np.argmax(np.real(val))]); pi = pi / pi.sum()
L = np.outer(np.ones(n), pi)
print("L idempotente:", np.allclose(L @ L, L), " posto:", np.linalg.matrix_rank(L))
for k in (1, 2, 4, 8, 16, 32, 64):
    print(f"k = {k:2d}: max|A^k - 1 pi'| = {np.abs(np.linalg.matrix_power(A, k) - L).max():.2e}")

# Normalização de camada: projeção sobre 1-perp seguida de reescala
ln = lambda x: (x - x.mean()) / np.sqrt(((x - x.mean()) ** 2).mean())
x = g.matriz_normal(1, d)[0]
print(f"LayerNorm: soma = {ln(x).sum():.1e}, norma = {np.linalg.norm(ln(x)):.4f} = sqrt(d) = {np.sqrt(d):.4f}, "
      f"invariância a 3x + 5: {np.abs(ln(3 * x + 5) - ln(x)).max():.1e}")
