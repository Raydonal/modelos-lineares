"""Cap. 3 -- Coeficiente parcial pela inversa particionada (prévia de Frisch-Waugh-Lovell)."""
from pathlib import Path

import numpy as np
import pandas as pd

d = pd.read_csv(Path(__file__).resolve().parents[2] / "dados" / "wage1.csv")
y = np.log(d["wage"].to_numpy())
X1 = np.column_stack([np.ones(len(d)), d["exper"]])
x2 = d["educ"].to_numpy(float)
X = np.column_stack([X1, x2])

b = np.linalg.solve(X.T @ X, X.T @ y)
s = x2 @ x2 - x2 @ X1 @ np.linalg.solve(X1.T @ X1, X1.T @ x2)
print(np.linalg.inv(X.T @ X)[2, 2], 1 / s)
e2 = x2 - X1 @ np.linalg.solve(X1.T @ X1, X1.T @ x2)
print(s, e2 @ e2)
print(b[2], (e2 @ y) / (e2 @ e2), np.cov(x2, y)[0, 1] / np.var(x2, ddof=1))
