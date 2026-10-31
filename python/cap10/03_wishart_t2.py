"""Cap. 10: Wishart, complemento de Schur e o T^2 de Hotelling."""
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular
from scipy.stats import chi2, f

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.gerador import Gerador  # noqa: E402

p, n, R = 3, 12, 20000
mu = np.array([1.0, 2, 0])
Sigma = np.array([[4.0, 2, 1], [2, 3, -1], [1, -1, 2]])
L = np.linalg.cholesky(Sigma); g = Gerador(37)
T2 = np.empty(R); schur = np.empty(R)
for r in range(R):
    X = g.matriz_normal(n, p) @ L.T + mu
    xb = X.mean(0); Xc = X - xb; S = Xc.T @ Xc / (n - 1)
    T2[r] = n * (xb - mu) @ np.linalg.solve(S, xb - mu)
    M = solve_triangular(L, Xc.T, lower=True)
    W = M @ M.T                                # ~ W(n-1, I)
    schur[r] = 1 / np.linalg.inv(W)[0, 0]      # W11 - W12 W22^-1 W21
# 1 / (W^{-1})_11 ~ qui-quadrado com m - p + 1 = 9 gl (aqui m = n - 1)
print(f"média do complemento de Schur: {schur.mean():.4f}"
      f" (teórica {n - p})")
Fst = (n - p) / (p * (n - 1)) * T2
q = [0.5, 0.9, 0.95, 0.99]
print("quantis:", np.quantile(Fst, q).round(4))
print("F:      ", f.ppf(q, p, n - p).round(4))
print(f"rejeição a 5%: F {np.mean(Fst > f.ppf(0.95, p, n - p)):.4f},"
      f" qui2 {np.mean(T2 > chi2.ppf(0.95, p)):.4f}")
