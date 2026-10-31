"""Cap. 7 -- Pseudo-inversa, tolerância e filtros espectrais num classificador de dígitos."""
from pathlib import Path

import numpy as np
import pandas as pd

dig = pd.read_csv(Path(__file__).resolve().parents[2] / "dados" / "digitos.csv")
X = np.column_stack([np.ones(len(dig)), dig.drop(columns="digito").to_numpy() / 16]); y = dig["digito"].to_numpy()
tr, te = np.arange(1000), np.arange(1000, len(y))
Y = (y[tr, None] == np.arange(10)).astype(float)
U, s, Vt = np.linalg.svd(X[tr], full_matrices=False)
print("valores singulares:", s[[0, 9, 29, 49, 59, 61, 62, 63, 64]])
acc = lambda W: np.mean((X[te] @ W).argmax(1) == y[te])
filtro = lambda f: Vt.T @ (f[:, None] * (U.T @ Y))
for k in (5, 10, 20, 30, 40, 50, 62, 63):
    f = np.r_[1 / s[:k], np.zeros(65 - k)]
    W = filtro(f); print(f"SVD truncada k = {k:2d}: acurácia = {acc(W):.4f}, ||W||_F = {np.linalg.norm(W):.4g}")
for lam in 10.0 ** np.arange(-3, 3):
    W = filtro(s / (s ** 2 + lam))
    print(f"ridge lambda = {lam:g}: gl = {(s ** 2 / (s ** 2 + lam)).sum():.2f}, acurácia = {acc(W):.4f}, ||W||_F = {np.linalg.norm(W):.4f}")
