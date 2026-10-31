"""Cap. 5: A diagonal da matriz chapéu (alavancas) nos dados wage1."""
from pathlib import Path

import numpy as np
import pandas as pd

w = pd.read_csv(Path(__file__).resolve().parents[2] / "dados" / "wage1.csv")
X = np.column_stack([np.ones(len(w)), w[["educ", "exper", "tenure"]]]); y = np.log(w["wage"].to_numpy())
n, p = X.shape
H = X @ np.linalg.solve(X.T @ X, X.T); h = np.diag(H)
print(f"soma = {h.sum():.4f}, mín = {h.min():.6f} (1/n = {1 / n:.6f}), máx = {h.max():.6f}")
Zc = X[:, 1:] - X[:, 1:].mean(0)
S = Zc.T @ Zc / (n - 1)
d2 = np.einsum("ij,ij->i", Zc @ np.linalg.inv(S), Zc)
print("h = 1/n + d2/(n-1):", np.abs(h - (1 / n + d2 / (n - 1))).max())
i = h.argmax()
print(f"obs. {i + 1}: h_ii = {h[i]:.6f}, h(1-h) = {h[i] * (1 - h[i]):.6f}, soma h_ij^2 = {(np.delete(H[i], i) ** 2).sum():.6f}")
print(w.loc[i, ["wage", "educ", "exper", "tenure"]].to_dict())
print("acima de 2p/n:", (h > 2 * p / n).sum(), " acima de 3p/n:", (h > 3 * p / n).sum())
