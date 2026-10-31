"""Cap. 10 -- Cochran na análise de variância com um fator."""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import f, ncf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.gerador import Gerador  # noqa: E402

a, r = 3, 5; n = a * r; sigma = 2.0
D = np.kron(np.eye(a), np.ones((r, 1)))
J = np.full((n, n), 1 / n); P = D @ np.linalg.inv(D.T @ D) @ D.T
A1, A2 = P - J, np.eye(n) - P                # entre e dentro
print("postos:", np.trace(A1).round(), np.trace(A2).round(),
      " max|A1 A2| =", np.abs(A1 @ A2).max())
mu = np.repeat([10.0, 11, 13], r)           # médias dos grupos
lam = mu @ A1 @ mu / sigma**2
g = Gerador(31); R = 20000
Y = g.matriz_normal(R, n, sigma) + mu
q1 = np.sum((Y @ A1) * Y, 1) / sigma**2     # SQ entre / sigma^2
q2 = np.sum((Y @ A2) * Y, 1) / sigma**2     # SQ dentro / sigma^2
F = (q1 / (a - 1)) / (q2 / (n - a)); c = f.ppf(0.95, a - 1, n - a)
print(f"lambda = {lam:.4f}; médias {q1.mean():.4f} (teórica"
      f" {a - 1 + lam:.4f}) e {q2.mean():.4f} (teórica {n - a})")
print(f"correlação = {np.corrcoef(q1, q2)[0, 1]:.4f}; poder simulado"
      f" {np.mean(F > c):.4f},"
      f" teórico {ncf.sf(c, a - 1, n - a, lam):.4f}")
