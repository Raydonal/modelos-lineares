"""Cap. 4: Ratings de Massey na NFL: o que é estimável num campeonato."""
from pathlib import Path

import numpy as np
import pandas as pd

nfl = pd.read_csv(Path(__file__).resolve().parents[2] / "dados" / "nfl.csv")
s = nfl[nfl["ano"] == 1981]
times = sorted(set(s["mandante"]) | set(s["visitante"]))
idx = {t: i for i, t in enumerate(times)}


def delineamento(j):
    X = np.zeros((len(j), len(times) + 1)); X[:, -1] = 1          # última coluna: mando
    X[np.arange(len(j)), j["mandante"].map(idx)] = 1
    X[np.arange(len(j)), j["visitante"].map(idx)] = -1
    return X


X = delineamento(s); y = (s["pontos_mandante"] - s["pontos_visitante"]).to_numpy(float)
print("posto:", np.linalg.matrix_rank(X), " posto dos times:", np.linalg.matrix_rank(X[:, :28]))
b_mp = np.linalg.pinv(X) @ y                                     # norma mínima
b_ref = np.linalg.solve(X.T @ X + np.outer(np.eye(29)[0], np.eye(29)[0]), X.T @ y)   # ATL = 0
print(f"mando = {b_mp[-1]:.4f} (média das margens = {y.mean():.4f}); soma dos ratings = {b_mp[:28].sum():.1e}")
i, j = idx["SF"], idx["DAL"]
print(f"SF - DAL: {b_mp[i] - b_mp[j]:.3f} (MP) e {b_ref[i] - b_ref[j]:.3f} (ATL = 0)")
top = np.argsort(-b_mp[:28])[:5]
print("melhores:", ", ".join(f"{times[k]} {b_mp[k]:.1f}" for k in top))

for w in range(1, 5):
    Xw = delineamento(s[s["semana"] <= w])[:, :28]
    print(f"semana {w}: componentes = {28 - np.linalg.matrix_rank(Xw)}")

tr, te = s[s["semana"] <= 8], s[s["semana"] > 8]
b8 = np.linalg.pinv(delineamento(tr)) @ (tr["pontos_mandante"] - tr["pontos_visitante"]).to_numpy(float)
m = (te["pontos_mandante"] - te["pontos_visitante"]).to_numpy(float); prev = delineamento(te) @ b8
print(f"EAM Massey = {np.abs(m - prev).mean():.2f}; apostas = {np.abs(m - te['linha']).mean():.2f}; "
      f"zero = {np.abs(m).mean():.2f}")
