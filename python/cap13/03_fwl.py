"""Cap. 13: Frisch-Waugh-Lovell e o viés de variável omitida"""
from pathlib import Path

import numpy as np
import pandas as pd

raiz = Path(__file__).resolve().parents[2]
w = pd.read_csv(raiz / "dados" / "wage1.csv")
y = np.log(w.wage.to_numpy())
n = len(y)
X1 = np.column_stack([np.ones(n), w.exper, w.tenure])  # controles
x2 = w.educ.to_numpy(float)                           # de interesse


def mq(X, v): return np.linalg.lstsq(X, v, rcond=None)[0]
def M1(v): return v - X1 @ mq(X1, v)                  # resíduo em X1


X = np.column_stack([X1, x2]); b = mq(X, y)
f = mq(M1(x2)[:, None], M1(y))[0]                     # FWL
f2 = mq(M1(x2)[:, None], y)[0]
print(f"longo {b[3]:.6f}, fwl {f:.6f}, só x2 residualizado {f2:.6f}")
e_l = y - X @ b; e_f = M1(y) - f * M1(x2)
print(f"dif. resíduos {np.max(np.abs(e_l - e_f)):.1e}")
# Erro-padrão: o do FWL usa gl n - 1 em vez de n - 4
ep_l = np.sqrt(e_l @ e_l / (n - 4) * np.linalg.inv(X.T @ X)[3, 3])
ep_f = np.sqrt(e_f @ e_f / (n - 1) / (M1(x2) @ M1(x2)))
print(f"ep longo {ep_l:.6f}, fwl {ep_f:.6f},"
      f" fwl corrigido {ep_f * np.sqrt((n - 1) / (n - 4)):.6f}")
# Viés de omissão: curto = longo + delta * (coef. omitido)
Z = np.column_stack([np.ones(n), x2])
curto = mq(Z, y)[1]
delta = mq(Z, X1[:, 1:])[1]                           # exper, tenure
print(f"curto {curto:.6f}, longo {b[3]:.6f},"
      f" identidade {b[3] + delta @ b[1:3]:.6f}")
print("delta (exper, tenure):", delta.round(4))
