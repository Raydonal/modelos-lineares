"""Cap. 8 -- Influência de cada imagem no primeiro autovalor (derivada de autovalores)."""
from pathlib import Path

import numpy as np
import pandas as pd

dig = pd.read_csv(Path(__file__).resolve().parents[2] / "dados" / "digitos.csv")
X = dig.drop(columns="digito").to_numpy(float); n = len(X)
S = np.cov(X.T); lam, Q = np.linalg.eigh(S); l1, q1 = lam[-1], Q[:, -1]
z = (X - X.mean(0)) @ q1
aprox = (l1 - n / (n - 1) * z ** 2) / (n - 2)
exato = np.array([np.linalg.eigvalsh(np.cov(np.delete(X, i, 0).T))[-1] - l1 for i in range(n)])
print(f"lambda_1 = {l1:.4f}, correlação aproximação x exato = {np.corrcoef(aprox, exato)[0, 1]:.6f}")
o = np.argsort(exato)[:5]
print(pd.DataFrame({"imagem": o + 1, "dígito": dig["digito"].to_numpy()[o], "exato": exato[o].round(3), "aprox": aprox[o].round(3)}))
