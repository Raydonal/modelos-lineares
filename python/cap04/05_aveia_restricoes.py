"""Cap. 4: Restrições de identificação no experimento de aveia e inversas generalizadas."""
from pathlib import Path

import numpy as np
import pandas as pd

aveia = pd.read_csv(Path(__file__).resolve().parents[2] / "dados" / "yates_oats.csv")
ind = lambda s: pd.get_dummies(s).to_numpy(float)
X = np.column_stack([np.ones(len(aveia)), ind(aveia["block"]), ind(aveia["nitro"]), ind(aveia["gen"])])
nomes = ["mu"] + [f"b{i}" for i in range(1, 7)] + ["n0", "n2", "n4", "n6", "GR", "Ma", "Vi"]
y = aveia["yield"].to_numpy(float)
XtX, Xty = X.T @ X, X.T @ y
e = lambda *js: np.array([float(n in js) for n in nomes])
B, N, V = nomes[1:7], nomes[7:11], nomes[11:]

K = {"tratamento": np.array([e("b1"), e("n0"), e("GR")]),
     "soma_zero": np.array([e(*B), e(*N), e(*V)]),
     "internas": np.array([e("mu") - e(*B), e("mu") - e(*N), e("mu") - e(*V)])}
sol = {}
for nome, Kk in K.items():
    assert np.linalg.matrix_rank(np.vstack([X, Kk])) == 14
    G = np.linalg.inv(XtX + Kk.T @ Kk)
    assert np.abs(XtX @ G @ XtX - XtX).max() < 1e-8
    sol[nome] = G @ Xty
sol["pinv"] = np.linalg.pinv(XtX) @ Xty
tab = pd.DataFrame(sol, index=nomes)
print(tab.loc[["mu", "b1", "n0", "n6", "GR", "Vi"]].round(2) + 0.0)

L = {"n6 - n0": e("n6") - e("n0"), "Vi - GR": e("Vi") - e("GR"),
     "mu + b1 + n0 + GR": e("mu", "b1", "n0", "GR"), "mu": e("mu"), "n6": e("n6")}
Gmp = np.linalg.pinv(XtX)
for nome, l in L.items():
    est = np.abs(l @ Gmp @ XtX - l).max() < 1e-8
    print(f"{nome:18s}", np.round(l @ tab.to_numpy(), 3), "estimável" if est else "não estimável")
