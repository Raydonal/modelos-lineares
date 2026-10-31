"""Cap. 8 -- Retropropagação numa camada de atenção, conferida por diferenças finitas."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.gerador import Gerador  # noqa: E402

g = Gerador(88); n, d, dk = 6, 5, 3
X, T = g.matriz_normal(n, d), g.matriz_normal(n, d)
W = {"q": g.matriz_normal(d, dk), "k": g.matriz_normal(d, dk), "v": g.matriz_normal(d, d)}


def softmax(S):
    E = np.exp(S - S.max(1, keepdims=True)); return E / E.sum(1, keepdims=True)


def frente(W):
    Q, K, V = X @ W["q"], X @ W["k"], X @ W["v"]
    A = softmax(Q @ K.T / np.sqrt(dk)); O = A @ V
    return Q, K, V, A, O, 0.5 * ((O - T) ** 2).sum()


Q, K, V, A, O, L = frente(W)
GO = O - T; GA = GO @ V.T
GS = A * (GA - (GA * A).sum(1, keepdims=True))
grad = {"q": X.T @ (GS @ K) / np.sqrt(dk), "k": X.T @ (GS.T @ Q) / np.sqrt(dk), "v": X.T @ (A.T @ GO)}
h = 1e-6
for c in "qkv":
    num = np.zeros_like(W[c])
    for idx in np.ndindex(W[c].shape):
        Wp = {k: v.copy() for k, v in W.items()}; Wm = {k: v.copy() for k, v in W.items()}
        Wp[c][idx] += h; Wm[c][idx] -= h
        num[idx] = (frente(Wp)[-1] - frente(Wm)[-1]) / (2 * h)
    print(f"W_{c}: max |retropropagação - diferenças finitas| = {np.abs(num - grad[c]).max():.2e}")
J = [np.diag(a) - np.outer(a, a) for a in A]
print("maior autovalor do jacobiano da softmax:", round(max(np.linalg.eigvalsh(M)[-1] for M in J), 6), "(<= 1/2)")
