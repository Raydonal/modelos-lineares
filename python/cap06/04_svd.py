"""Cap. 6 -- Decomposição em valores singulares, subespaços fundamentais e Eckart-Young."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.gerador import Gerador  # noqa: E402

A = np.array([[1.0, 2, 0, 1], [2, 4, 1, 1], [3, 6, 1, 2]])
U, s, Vt = np.linalg.svd(A)
print("valores singulares:", s.round(6))
r = (s > 1e-10 * s[0]).sum()
print("A V_nulo:", np.abs(A @ Vt[r:].T).max(), "  A' U_nulo:", np.abs(A.T @ U[:, r:]).max())
g = Gerador(64)
X = g.matriz_normal(30, 8)
U, s, Vt = np.linalg.svd(X, full_matrices=False); k = 2
Xk = U[:, :k] * s[:k] @ Vt[:k]
print(f"Eckart-Young: erro = {((X - Xk) ** 2).sum():.4f}, soma descartados = {(s[k:] ** 2).sum():.4f}")
outras = [((X - (W := np.linalg.qr(g.matriz_normal(30, k))[0]) @ W.T @ X) ** 2).sum() for _ in range(2000)]
print(f"melhor de 2000 aproximações ao acaso: {min(outras):.4f}")
