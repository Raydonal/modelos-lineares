"""Cap. 11: Sazonalidade: indicadoras ou harmônicos de Fourier"""
from pathlib import Path

import numpy as np
import pandas as pd

raiz = Path(__file__).resolve().parents[2]
s = pd.read_csv(raiz / "dados" / "seatbelts.csv")
y = np.log(s["motoristas"].to_numpy()); n = len(y)
t = np.arange(1, n + 1.0); lei = s["lei"].to_numpy(float)
D = np.eye(12)[s["mes"].to_numpy() - 1]       # 12 indicadoras


def fourier(K):                               # K harmônicos (K <= 6)
    F = []
    for k in range(1, K + 1):
        F.append(np.cos(2 * np.pi * k * t / 12))
        if k < 6:
            F.append(np.sin(2 * np.pi * k * t / 12))
    return np.column_stack(F) if F else np.empty((n, 0))


F6 = fourier(6)                               # 11 colunas
G = F6.T @ F6                                 # F'F diagonal
fora = np.max(np.abs(G - np.diag(np.diag(G))))
print(f"fora da diagonal {fora:.1e}, diagonal de"
      f" {G.diagonal().min():.0f} a {G.diagonal().max():.0f}")


def coef(X): return np.linalg.lstsq(X, y, rcond=None)[0]
def fit(X): return X @ coef(X)


def modelo(K):
    return np.column_stack([np.ones(n), fourier(K), t, lei])


Xd = np.column_stack([D, t, lei])             # indicadoras
dif = np.max(np.abs(fit(Xd) - fit(modelo(6))))
print(f"dif. ajustados {dif:.1e}")
sqr = np.array([np.sum((y - fit(modelo(K)))**2) for K in range(7)])
frac = (sqr[0] - sqr) / (sqr[0] - sqr[6])
col = [0, 2, 4, 6, 8, 10, 11]
print(pd.DataFrame({"K": range(7), "colunas": col,
                    "SQR": sqr.round(3), "frac": frac.round(3)}))
lei_K = [coef(modelo(K))[-1] for K in (2, 6)]
print("efeito da lei, K = 2 e K = 6:", np.round(lei_K, 4))
