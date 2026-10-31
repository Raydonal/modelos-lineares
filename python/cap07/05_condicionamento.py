"""Cap. 7: Número de condição: sistemas lineares e mínimos quadrados."""
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import hilbert

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.gerador import Gerador  # noqa: E402

H = hilbert(8); x = np.ones(8)
print(f"kappa(H) = {np.linalg.cond(H):.4e}, erro relativo de solve = {np.linalg.norm(np.linalg.solve(H, H @ x) - x) / np.sqrt(8):.2e}")
g = Gerador(75)
U = np.linalg.qr(g.matriz_normal(50, 5))[0]; V = np.linalg.qr(g.matriz_normal(5, 5))[0]
for k in (1e2, 1e4, 1e6, 1e7):
    X = U @ np.diag(k ** -(np.arange(5) / 4)) @ V.T; y = X @ np.ones(5)
    b_ne = np.linalg.solve(X.T @ X, X.T @ y); b_qr = np.linalg.lstsq(X, y, rcond=None)[0]
    print(f"kappa = {np.linalg.cond(X):.1e}: erro equações normais = {np.abs(b_ne - 1).max():.2e}, erro QR/SVD = {np.abs(b_qr - 1).max():.2e}")
