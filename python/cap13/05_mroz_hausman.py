"""Cap. 13: MQ2E como MQG e o teste de Hausman (dados de Mroz)"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

raiz = Path(__file__).resolve().parents[2]
m = pd.read_csv(raiz / "dados" / "mroz.csv")
y = m.lwage.to_numpy(); n = len(y); um = np.ones(n)
X = np.column_stack([um, m.educ])
Zs = {"pai": np.column_stack([um, m.fatheduc]),
      "pai_mae": np.column_stack([um, m.fatheduc, m.motheduc])}
bmq = np.linalg.solve(X.T @ X, X.T @ y)
s2 = np.sum((y - X @ bmq) ** 2) / (n - 2)     # MQ é eficiente sob H0
for nome, Z in Zs.items():
    PZ = Z @ np.linalg.solve(Z.T @ Z, Z.T)
    biv = np.linalg.solve(X.T @ PZ @ X, X.T @ PZ @ y)
    # MQG no modelo Z'y = Z'X b + Z'e, com Cov(Z'e) = s2 Z'Z
    W = np.linalg.inv(Z.T @ Z)
    bg = np.linalg.solve(X.T @ Z @ W @ Z.T @ X, X.T @ Z @ W @ Z.T @ y)
    vd = s2 * (np.linalg.inv(X.T @ PZ @ X)
               - np.linalg.inv(X.T @ X))[1, 1]
    H = (biv[1] - bmq[1]) ** 2 / vd
    print(f"{nome:8} dif_mqg {np.max(np.abs(bg - biv)):.0e}"
          f" mq {bmq[1]:.4f} mq2e {biv[1]:.4f}")
    print(f"{'':8} ep_dif {np.sqrt(vd):.4f} H {H:.4f}"
          f" p {stats.chi2.sf(H, 1):.4f}")
