"""Cap. 7 -- Normas matriciais: fórmulas das normas induzidas, Frobenius e raio espectral."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.gerador import Gerador  # noqa: E402

A = np.array([[1.0, -2, 3], [0, 4, -1], [2, 1, 1]])
s = np.linalg.svd(A, compute_uv=False)
print(f"||A||_1 = {np.linalg.norm(A, 1):.6f} (máx. soma de colunas {np.abs(A).sum(0).max():.6f})")
print(f"||A||_inf = {np.linalg.norm(A, np.inf):.6f} (máx. soma de linhas {np.abs(A).sum(1).max():.6f})")
print(f"||A||_2 = {np.linalg.norm(A, 2):.6f} = sigma_1;  ||A||_F = {np.linalg.norm(A):.6f} = {np.sqrt((s ** 2).sum()):.6f}")
print(f"raio espectral = {np.abs(np.linalg.eigvals(A)).max():.6f};  ||A^-1||_2 = {np.linalg.norm(np.linalg.inv(A), 2):.6f} = 1/sigma_min")
g = Gerador(70)
X = g.matriz_normal(3, 10000)
print("máx ||Ax||/||x|| em 10 000 vetores:", round((np.linalg.norm(A @ X, axis=0) / np.linalg.norm(X, axis=0)).max(), 6))
