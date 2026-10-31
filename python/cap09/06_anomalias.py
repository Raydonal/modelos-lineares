"""Cap. 9: Detecção de anomalias pela distância de Mahalanobis."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, rankdata

raiz = Path(__file__).resolve().parents[2]
dig = pd.read_csv(raiz / "dados" / "digitos.csv")
X = dig.iloc[:, 1:].to_numpy() / 16; y = dig["digito"].to_numpy()
tr, te = np.arange(1000), np.arange(1000, len(y))
Z = X[tr][y[tr] == 1]                        # o "normal": dígitos 1
mu = Z.mean(0); S = np.cov(Z.T)
print("n =", len(Z), " posto =", np.linalg.matrix_rank(S),
      " constantes =", (Z.std(0) == 0).sum())


def auc(s, anom):                            # Mann-Whitney
    r = rankdata(s); n1 = anom.sum(); n0 = len(anom) - n1
    return (r[anom].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


D = X[te] - mu; anom = y[te] != 1; oito = np.isin(y[te], [1, 8])
d2 = lambda M, Si: np.sum((M @ Si) * M, 1)
eu = d2(D, np.eye(64))
print(f"euclidiana:           AUC = {auc(eu, anom):.4f},"
      f" contra 8 = {auc(eu[oito], anom[oito]):.4f}")
for lam in (0.001, 0.01, 0.1, 1):
    m = d2(D, np.linalg.inv(S + lam * np.eye(64)))
    print(f"mahalanobis {lam:6g}:  AUC = {auc(m, anom):.4f},"
          f" contra 8 = {auc(m[oito], anom[oito]):.4f}")

# Calibração com lambda = 0,01: o limiar qui-quadrado não vale
Si = np.linalg.inv(S + 0.01 * np.eye(64))
dt = d2(Z - mu, Si); d1 = d2(D[~anom], Si)
print(f"média treino = {dt.mean():.4f}, média teste (1) ="
      f" {d1.mean():.4f}")
print(f"falsos positivos: limiar qui2 = "
      f"{np.mean(d1 > chi2.ppf(0.99, 64)):.4f}, quantil 99% do treino"
      f" = {np.mean(d1 > np.quantile(dt, 0.99)):.4f}")
