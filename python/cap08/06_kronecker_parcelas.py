"""Cap. 8: Parcelas subdivididas no experimento de aveia: a covariância como produto de Kronecker."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

a = pd.read_csv(Path(__file__).resolve().parents[2] / "dados" / "yates_oats.csv").sort_values(["block", "gen", "nitro"])
y = a["yield"].to_numpy(float); n = len(y)
X = np.column_stack([np.ones(n), pd.get_dummies(a["block"], drop_first=True), pd.get_dummies(a["gen"], drop_first=True),
                     pd.get_dummies(a["nitro"], drop_first=True)]).astype(float)
inv_bloco = lambda s2, s2p: (np.eye(4) - 0.25) / s2 + 0.25 / (s2 + 4 * s2p)


def loglik(s2, s2p):
    Vi = np.kron(np.eye(18), inv_bloco(s2, s2p))
    b = np.linalg.solve(X.T @ Vi @ X, X.T @ Vi @ y); r = y - X @ b
    return -0.5 * (n * np.log(2 * np.pi) + 18 * (3 * np.log(s2) + np.log(s2 + 4 * s2p)) + r @ Vi @ r)


V0 = np.kron(np.eye(18), 150 * np.eye(4) + 200 * np.ones((4, 4)))
print("inversa por Kronecker:", np.abs(np.linalg.inv(V0) - np.kron(np.eye(18), inv_bloco(150, 200))).max())
opt = minimize(lambda t: -loglik(*np.exp(t)), np.log([150, 200]), method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-10})
s2, s2p = np.exp(opt.x); print(f"s2 (subparcela) = {s2:.1f}, s2p (parcela) = {s2p:.1f}")
Vi = np.kron(np.eye(18), inv_bloco(s2, s2p)); covb = np.linalg.inv(X.T @ Vi @ X)
res = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]; covo = res @ res / (n - X.shape[1]) * np.linalg.inv(X.T @ X)
idx = [6, 7, 10]                                             # Marvellous, Victory, nitro 0,6
print("EP MQG:", np.sqrt(np.diag(covb))[idx].round(2), " EP MQO ingênuo:", np.sqrt(np.diag(covo))[idx].round(2))
