"""Cap. 7 -- Quanto os dados de Longley podem ser arredondados? (Beaton, Rubin e Barone, 1976)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.width", 120)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.gerador import Gerador  # noqa: E402

L = pd.read_csv(Path(__file__).resolve().parents[2] / "dados" / "longley.csv")
nomes = ["GNP.deflator", "GNP", "Unemployed", "Armed.Forces", "Population", "Year"]
X = np.column_stack([np.ones(16), L[nomes]]); y = L["Employed"].to_numpy()
print(f"kappa(X) = {np.linalg.cond(X):.4e}, kappa(X'X) = {np.linalg.cond(X.T @ X):.4e}")
mq = lambda A: np.linalg.lstsq(A, y, rcond=None)[0]
b = mq(X)
casas = np.array([1, 3, 1, 1, 3, 0])
g = Gerador(76)
R = []
for _ in range(1000):
    Xp = X.copy()
    Xp[:, 1:] += (g.unif(16 * 6) - 0.5).reshape(6, 16).T * 10.0 ** -casas   # mesma ordem (por colunas) do R
    R.append(mq(Xp))
R = np.array(R)
rel = np.linalg.norm(R - b, axis=1) / np.linalg.norm(b)
print("variação relativa dos coeficientes (5, 50, 95%):", np.quantile(rel, [0.05, 0.5, 0.95]).round(3))
print(pd.DataFrame({"mediana": np.median(np.abs(R - b) / np.abs(b), 0), "muda_sinal": (np.sign(R) != np.sign(b)).mean(0)},
                   index=["intercepto"] + nomes).round(3).T)
