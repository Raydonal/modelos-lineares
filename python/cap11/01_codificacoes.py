"""Cap. 11: Codificações de um fator: o que cada coeficiente mede"""
from pathlib import Path

import numpy as np
import pandas as pd

raiz = Path(__file__).resolve().parents[2]
a = pd.read_csv(raiz / "dados" / "yates_oats.csv")
niv = np.sort(a["nitro"].unique())
j = np.searchsorted(niv, a["nitro"])          # nível de cada parcela
y = a["yield"].to_numpy(float)
mu = np.array([y[j == k].mean() for k in range(4)])


def poli(k):                                  # como contr.poly do R
    V = np.vander(np.arange(1, k + 1), k, increasing=True)
    Q = np.linalg.qr(V)[0][:, 1:]
    return Q * np.sign(Q[-1])                 # último elemento > 0


helm = np.array([[-1, -1, -1], [1, -1, -1], [0, 2, -1], [0, 0, 3]])
cods = {"tratamento": np.eye(4)[:, 1:],
        "soma": np.vstack([np.eye(3), -np.ones(3)]),
        "helmert": helm, "polinomial": poli(4)}
for nome, C in cods.items():
    M = np.linalg.inv(np.column_stack([np.ones(4), C]))
    X = np.column_stack([np.ones(len(y)), C[j]])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    print(f"{nome}: max |b - M mu| = "
          f"{np.max(np.abs(b - M @ mu)):.1e}")
    print(pd.DataFrame(M.round(3) + 0.0, index=[f"b{i}" for i in
          range(4)], columns=[f"mu{i}" for i in range(1, 5)]))
print("médias:", mu.round(2))
