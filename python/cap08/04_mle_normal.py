"""Cap. 8: Máxima verossimilhança da normal multivariada pelo cálculo matricial (dados wage1)."""
from pathlib import Path

import numpy as np
import pandas as pd

w = pd.read_csv(Path(__file__).resolve().parents[2] / "dados" / "wage1.csv")
X = np.column_stack([np.log(w["wage"]), w["educ"], w["exper"], w["tenure"]]); n, p = X.shape


def loglik(mu, S):
    L = np.linalg.cholesky(S); Z = np.linalg.solve(L, (X - mu).T)
    return -n * p / 2 * np.log(2 * np.pi) - n * np.log(np.diag(L)).sum() - (Z ** 2).sum() / 2


mu = X.mean(0); Ac = (X - mu).T @ (X - mu); Sh = Ac / n
grad = lambda S: 0.5 * np.linalg.inv(S) @ (Ac - n * S) @ np.linalg.inv(S)
print(Sh.round(3))
print(f"||grad|| no EMV = {np.linalg.norm(grad(Sh)):.2e}, na covariância com n-1 = {np.linalg.norm(grad(np.cov(X.T))):.4f}")
print(f"log-verossimilhança máxima = {loglik(mu, Sh):.4f}")
