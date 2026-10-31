"""Cap. 3: Validação cruzada leave-one-out em forma fechada e mínimos quadrados recursivos."""
import time
from pathlib import Path

import numpy as np
import pandas as pd

db = pd.read_csv(Path(__file__).resolve().parents[2] / "dados" / "diabetes.csv")
X = np.column_stack([np.ones(len(db)), db.iloc[:, :10].to_numpy()])
y = db["y"].to_numpy(float)
n, p = X.shape

Q, R = np.linalg.qr(X)
b_mqo = np.linalg.solve(R, Q.T @ y)
e = y - X @ b_mqo
h = np.sum(Q**2, axis=1)                         # alavancas
loo_fechado = e / (1 - h)

t0 = time.perf_counter()
loo_bruto = np.array([y[i] - X[i] @ np.linalg.lstsq(np.delete(X, i, 0), np.delete(y, i), rcond=None)[0]
                      for i in range(n)])
print(f"max dif = {np.abs(loo_fechado - loo_bruto).max():.1e}, bruto {time.perf_counter() - t0:.2f}s")
print(f"PRESS = {np.sum(loo_fechado**2):.0f}, SQres = {np.sum(e**2):.0f}")

m0 = 20
P = np.linalg.inv(X[:m0].T @ X[:m0]); b = P @ X[:m0].T @ y[:m0]
for t in range(m0, n):
    x = X[t]; Px = P @ x
    P -= np.outer(Px, Px) / (1 + x @ Px)
    b += P @ x * (y[t] - x @ b)
print(np.abs(b - b_mqo).max())
