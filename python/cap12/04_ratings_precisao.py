"""Cap. 12 -- Ratings de Massey: estimabilidade e precisão"""
from pathlib import Path

import numpy as np
import pandas as pd

raiz = Path(__file__).resolve().parents[2]
nfl = pd.read_csv(raiz / "dados" / "nfl.csv")
s = nfl[nfl["ano"] == 1981]
times = sorted(set(s["mandante"]) | set(s["visitante"]))
idx = {t: i for i, t in enumerate(times)}


def delineamento(j):                  # +1 mandante, -1 visitante
    X = np.zeros((len(j), 29)); X[:, 28] = 1
    lin = np.arange(len(j))
    X[lin, j["mandante"].map(idx)] = 1
    X[lin, j["visitante"].map(idx)] = -1
    return X


def lam(i, j):
    l = np.zeros(29); l[idx[i]] = 1; l[idx[j]] = -1
    return l


def ajuste(j):
    X = delineamento(j)
    y = (j["pontos_mandante"] - j["pontos_visitante"]).to_numpy(float)
    G = np.linalg.pinv(X.T @ X, rcond=1.5e-8)   # tol. do ginv
    b = G @ X.T @ y
    gl = len(y) - np.linalg.matrix_rank(X)
    return X, G, b, np.sum((y - X @ b) ** 2) / gl, gl


X, G, b, s2, gl = ajuste(s); l = lam("SF", "DAL")
ep = np.sqrt(s2 * l @ G @ l)
print(f"s {np.sqrt(s2):.3f}, gl {gl}, SF-DAL {l @ b:.3f},"
      f" ep {ep:.3f}, t {l @ b / ep:.3f}")
# Precisão de todas as 378 diferenças e da diferença 1o - 2o
v = [lam(ti, tj) @ G @ lam(ti, tj)
     for a, ti in enumerate(times) for tj in times[a + 1:]]
e = np.sqrt(s2 * np.array(v))
print(f"ep min {e.min():.2f}, médio {e.mean():.2f},"
      f" max {e.max():.2f}")
l2 = lam("PHA", "SF")
print(f"PHA-SF {l2 @ b:.2f}, ep {np.sqrt(s2 * l2 @ G @ l2):.2f}")
# Semana a semana: SF - DAL é estimável? Com que variância (em s2)?
for w in range(2, 17):
    Xw, Gw, _, _, _ = ajuste(s[s["semana"] <= w])
    P = np.eye(29) - Gw @ Xw.T @ Xw      # l estimável <=> l'P = 0
    est = np.max(np.abs(l @ P)) < 1e-8
    vv = f"{l @ Gw @ l:.3f}" if est else "NA"
    print(f"semana {w:2d}, jogos {len(Xw):3d}, estimável {int(est)},"
          f" var/s2 {vv}")
