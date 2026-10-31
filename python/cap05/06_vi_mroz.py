"""Cap. 5: Variáveis instrumentais como projeção oblíqua (dados de Mroz)."""
from pathlib import Path

import numpy as np
import pandas as pd

m = pd.read_csv(Path(__file__).resolve().parents[2] / "dados" / "mroz.csv")
y = m["lwage"].to_numpy(); n = len(y)
X = np.column_stack([np.ones(n), m["educ"]]); Z = np.column_stack([np.ones(n), m["fatheduc"]])
P = X @ np.linalg.solve(Z.T @ X, Z.T)
print("idempotente:", np.allclose(P @ P, P), " simétrica:", np.allclose(P, P.T), " traço:", np.trace(P).round(10))
b_mq = np.linalg.solve(X.T @ X, X.T @ y); b_vi = np.linalg.solve(Z.T @ X, Z.T @ y)
print("MQ:", b_mq.round(4), " VI:", b_vi.round(4))
e = y - P @ y
print("Z'e =", np.abs(Z.T @ e).max().round(10), "  X'e =", (X.T @ e).round(2))
r = np.corrcoef(m["educ"], m["fatheduc"])[0, 1]
print(f"|P| = {np.linalg.norm(P, 2):.4f},  1/|r| = {1 / abs(r):.4f}")
Z2 = np.column_stack([Z, m["motheduc"]]); PZ = Z2 @ np.linalg.solve(Z2.T @ Z2, Z2.T)
b_2e = np.linalg.solve(X.T @ PZ @ X, X.T @ PZ @ y)
P2 = X @ np.linalg.solve(X.T @ PZ @ X, X.T @ PZ)
print("MQ2E:", b_2e.round(4), " idempotente:", np.allclose(P2 @ P2, P2), f" |P2| = {np.linalg.norm(P2, 2):.4f}")
