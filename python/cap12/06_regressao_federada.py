"""Cap. 12: Regressão federada: regiões sem juntar os dados"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

raiz = Path(__file__).resolve().parents[2]
w = pd.read_csv(raiz / "dados" / "wage1.csv")
w["regiao"] = np.select([w.northcen == 1, w.south == 1, w.west == 1],
                        ["centro-norte", "sul", "oeste"], "nordeste")
X = np.column_stack([np.ones(len(w)), w.educ, w.exper,
                     w.exper**2 / 100, w.tenure, w.female])
y = np.log(w.wage.to_numpy()); n, p = X.shape
# Cada sítio envia só X_k'X_k, X_k'y_k, y_k'y_k e n_k
env = {}
for k, i in w.groupby("regiao").indices.items():
    env[k] = dict(XtX=X[i].T @ X[i], Xty=X[i].T @ y[i],
                  yty=y[i] @ y[i], n=len(i))
print({k: e["n"] for k, e in env.items()})
S = sum(e["XtX"] for e in env.values())
Xty = sum(e["Xty"] for e in env.values())
b_fed = np.linalg.solve(S, Xty)
b_mq = np.linalg.lstsq(X, y, rcond=None)[0]
print(f"max dif. {np.max(np.abs(b_fed - b_mq)):.1e}")  # idênticos
sqr = sum(e["yty"] for e in env.values()) - b_fed @ Xty
s2 = sqr / (n - p)
# Alternativa: média simples das estimativas de cada região
bk = {k: np.linalg.solve(e["XtX"], e["Xty"]) for k, e in env.items()}
b_med = np.mean(list(bk.values()), axis=0)
C_fed = s2 * np.linalg.inv(S)
C_med = s2 * sum(np.linalg.inv(e["XtX"]) for e in env.values()) / 16
j = 1                                         # educação
print(f"educ fed {b_fed[j]:.4f} (ep {np.sqrt(C_fed[j, j]):.4f}),"
      f" média {b_med[j]:.4f} (ep {np.sqrt(C_med[j, j]):.4f})")
print(f"razão das variâncias {C_med[j, j] / C_fed[j, j]:.4f}")
print(" ".join(f"{k} {b[j]:.4f}" for k, b in bk.items()))
ev = np.linalg.eigvalsh(C_med - C_fed)[::-1] / s2   # Loewner: >= 0
print(np.array2string(ev, formatter={"float": lambda v: f"{v:.2e}"}))
# H: mesmo beta nas 4 regiões? (teste de Chow, Cap. 10)
sqr_k = sum(e["yty"] - bk[k] @ e["Xty"] for k, e in env.items())
F = ((sqr - sqr_k) / (3 * p)) / (sqr_k / (n - 4 * p))
print(f"F {F:.3f}, gl {3 * p} e {n - 4 * p},"
      f" p-valor {stats.f.sf(F, 3 * p, n - 4 * p):.3f}")
