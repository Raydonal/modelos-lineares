"""Cap. 5: Subespaços encaixados, somas de quadrados e restrições (dados wage1)."""
from pathlib import Path

import numpy as np
import pandas as pd

w = pd.read_csv(Path(__file__).resolve().parents[2] / "dados" / "wage1.csv")
y = np.log(w["wage"].to_numpy()); n = len(y)
proj = lambda X: X @ np.linalg.solve(X.T @ X, X.T)
sq = lambda v: float(v @ v)
um = np.ones((n, 1))
X1 = np.column_stack([um, w["educ"]])
X2 = np.column_stack([X1, w["exper"], w["tenure"]])
P0, P1, P2 = proj(um), proj(X1), proj(X2)
print("P2 P1 = P1:", np.allclose(P2 @ P1, P1), " P2 - P1 idempotente:", np.allclose((P2 - P1) @ (P2 - P1), P2 - P1))
for nome, M in {"educ": P1 - P0, "exper+tenure": P2 - P1, "resíduo": np.eye(n) - P2}.items():
    print(f"{nome:13s} gl = {np.trace(M):5.0f}   SQ = {sq(M @ y):9.4f}")
print(f"SQ total = {sq(y - y.mean()):.4f}")

Z = (np.eye(n) - P1) @ np.column_stack([w["exper"], w["tenure"]])
print("P_[X1 X2] = P1 + P_{M1 X2}:", np.abs(P2 - P1 - proj(Z)).max())

K = np.array([[0.0, 0, 1, -1]])
PR = proj(np.column_stack([um, w["educ"], w["exper"] + w["tenure"]]))
U = X2 @ np.linalg.solve(X2.T @ X2, K.T)
print("P_X - P_S = P_U:", np.abs(P2 - PR - proj(U)).max(), "  SQ da restrição =", round(sq((P2 - PR) @ y), 4))
