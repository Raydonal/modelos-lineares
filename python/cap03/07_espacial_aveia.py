"""Cap. 3 -- Verossimilhança gaussiana com correlação espacial: log-determinantes e sistemas."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import cdist

a = pd.read_csv(Path(__file__).resolve().parents[2] / "dados" / "yates_oats.csv")
y = a["yield"].to_numpy(float)
dummies = lambda s: pd.get_dummies(s, drop_first=True).to_numpy(float)
X = np.column_stack([np.ones(len(a)), dummies(a["block"]), dummies(a["nitro"]), dummies(a["gen"])])
coord = np.column_stack([(a["col"] - 1) * 44 * 0.2012, (a["row"] - 1) * 28.4 * 0.2012])
D = cdist(coord, coord)
n = len(y)


def logvero(phi):
    R = np.eye(n) if phi == 0 else np.exp(-D / phi)
    RiX, Riy = np.linalg.solve(R, X), np.linalg.solve(R, y)
    beta = np.linalg.solve(X.T @ RiX, X.T @ Riy)
    r = y - X @ beta
    sigma2 = r @ np.linalg.solve(R, r) / n
    ld = np.linalg.slogdet(R)[1]
    return -n / 2 * np.log(2 * np.pi * sigma2) - ld / 2 - n / 2


for phi in (0, 1, 2, 4, 8, 15, 30, 45, 60):
    print(f"phi = {phi:4.0f}  logvero = {logvero(phi):8.2f}")
opt = minimize_scalar(lambda f: -logvero(f), bounds=(0.5, 60), method="bounded")
print(f"phi_hat = {opt.x:.2f}, ganho = {-opt.fun - logvero(0):.2f}")
