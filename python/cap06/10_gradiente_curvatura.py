"""Cap. 6: Gradiente descendente: taxa de aprendizado, autovalores da hessiana e padronização."""
from pathlib import Path

import numpy as np
import pandas as pd

db = pd.read_csv(Path(__file__).resolve().parents[2] / "dados" / "diabetes.csv")
y = db["y"].to_numpy(float); n = len(y); Z = db.iloc[:, :10].to_numpy(float)
Xb = np.column_stack([np.ones(n), Z])
Xp = np.column_stack([np.ones(n), (Z - Z.mean(0)) / Z.std(0, ddof=1)])
for nome, X in (("bruto", Xb), ("padronizado", Xp)):
    ev = np.linalg.eigvalsh(X.T @ X / n)
    print(f"{nome:11s} lambda_max = {ev.max():.4g}, lambda_min = {ev.min():.4g}, kappa = {ev.max() / ev.min():.4g}")
ev = np.linalg.eigvalsh(Xp.T @ Xp / n); Lm, mu = ev.max(), ev.min()
bs = np.linalg.solve(Xp.T @ Xp, Xp.T @ y)
for nome, eta in {"1/L": 1 / Lm, "ótimo": 2 / (Lm + mu), "1,98/L": 1.98 / Lm, "2,02/L": 2.02 / Lm}.items():
    b = np.zeros(11); d = []
    for _ in range(500):
        b = b - eta * Xp.T @ (Xp @ b - y) / n; d.append(np.linalg.norm(b - bs))
    print(f"eta = {nome:7s} erro após 1, 10, 100, 500 passos: {np.round([d[0], d[9], d[99], d[499]], 3)}; "
          f"fator previsto {np.abs(1 - eta * ev).max():.5f}")
