"""Cap. 7 -- Perturbação de autovalores e valores singulares; Eckart-Young na norma espectral."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.gerador import Gerador  # noqa: E402

g = Gerador(74)
S = g.matriz_normal(6, 6); A = S + S.T
E = g.matriz_normal(6, 6); E = 1e-3 * (E + E.T)
print(f"max |Delta lambda| = {np.abs(np.linalg.eigvalsh(A + E) - np.linalg.eigvalsh(A)).max():.6f} <= ||E||_2 = {np.linalg.norm(E, 2):.6f}")
X = g.matriz_normal(8, 5); F = 1e-3 * g.matriz_normal(8, 5)
sv = lambda M: np.linalg.svd(M, compute_uv=False)
print(f"max |Delta sigma| = {np.abs(sv(X + F) - sv(X)).max():.6f} <= ||F||_2 = {np.linalg.norm(F, 2):.6f}")
U, s, Vt = np.linalg.svd(X, full_matrices=False)
for k in range(1, 5):
    print(f"k = {k}: ||X - X_k||_2 = {np.linalg.norm(X - U[:, :k] * s[:k] @ Vt[:k], 2):.6f}, sigma_k+1 = {s[k]:.6f}")
