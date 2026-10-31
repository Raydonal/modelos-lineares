"""Cap. 10: Ataques de pertencimento e o mecanismo gaussiano."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, ncx2, norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.gerador import Gerador  # noqa: E402

raiz = Path(__file__).resolve().parents[2]
db = pd.read_csv(raiz / "dados" / "diabetes.csv")
X = db.iloc[:, :10].to_numpy(); p = X.shape[1]
X = (X - X.min(0)) / (X.max(0) - X.min(0))
Delta = np.sqrt(p); D = np.linalg.norm(X[0])  # ||a - b|| = ||x_1||


def sigma_suf(eps, delta=1e-5):               # Cap. 9
    z = norm.ppf(delta)
    u = (np.sqrt(2) * z + np.sqrt(2 * z**2 + 4 * eps)) / 2
    return Delta / (np.sqrt(2) * u)


alfa = 0.05; q = chi2.ppf(1 - alfa, p)
print(" eps   sigma  poder_NP  NP_pior  poder_qui2  cota_DP")
for eps in (0.5, 1, 2, 4, 8):
    s = sigma_suf(eps)
    print(f"{eps:4g} {s:7.3f} {norm.cdf(norm.ppf(alfa) + D / s):9.4f}"
          f" {norm.cdf(norm.ppf(alfa) + Delta / s):8.4f}"
          f" {ncx2.sf(q, p, (D / s)**2):11.4f}"
          f" {min(1, np.exp(eps) * alfa + 1e-5):8.4f}")
# Simulação com eps = 4: 10 000 publicações com e sem o paciente 1
s = sigma_suf(4); a, b = X.sum(0), X[1:].sum(0)
g = Gerador(43); R = 10000
Y1 = g.matriz_normal(R, p, s) + a             # presente
Y0 = g.matriz_normal(R, p, s) + b             # ausente
np_est = lambda Y: (Y - b) @ (a - b) / (s * D)
q2 = lambda Y: np.sum((Y - b)**2, 1) / s**2
z = norm.ppf(1 - alfa)
print(f"NP: alfa {np.mean(np_est(Y0) > z):.4f},"
      f" poder {np.mean(np_est(Y1) > z):.4f};"
      f" qui2: alfa {np.mean(q2(Y0) > q):.4f},"
      f" poder {np.mean(q2(Y1) > q):.4f}")
print(f"||x_1|| = {D:.4f}, Delta = {Delta:.4f}")
