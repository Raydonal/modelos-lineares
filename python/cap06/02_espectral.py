"""Cap. 6: Teorema espectral, projeções espectrais, quociente de Rayleigh e Ky Fan."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.gerador import Gerador  # noqa: E402

A = np.array([[4.0, 1, 1, 0], [1, 3, 0, 1], [1, 0, 3, 1], [0, 1, 1, 2]])
lam, Q = np.linalg.eigh(A)                                  # ordem crescente
print("autovalores:", lam.round(6))
print("Q'Q = I:", np.allclose(Q.T @ Q, np.eye(4)), " reconstrução:", np.abs(Q @ np.diag(lam) @ Q.T - A).max())
raiz = Q @ np.diag(np.sqrt(lam)) @ Q.T
print("raiz^2 - A:", np.abs(raiz @ raiz - A).max())
g = Gerador(62)
X = g.matriz_normal(10000, 4)
r = np.einsum("ij,ij->i", X @ A, X) / (X ** 2).sum(1)
print(f"Rayleigh em [{r.min():.4f}, {r.max():.4f}] contido em [{lam.min():.4f}, {lam.max():.4f}]")
tr2 = [np.trace(V.T @ A @ V) for V in (np.linalg.qr(g.matriz_normal(4, 2))[0] for _ in range(2000))]
print(f"Ky Fan: max tr(V'AV) = {max(tr2):.4f} <= {lam[-2:].sum():.4f}")
