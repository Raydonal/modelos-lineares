"""Cap. 2 -- Mínimos quadrados nos dados de Longley: equações normais versus QR."""
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.linalg as sla

lon = pd.read_csv(Path(__file__).resolve().parents[2] / "dados" / "longley.csv")
cols = ["GNP.deflator", "GNP", "Unemployed", "Armed.Forces", "Population", "Year"]
X = np.column_stack([np.ones(len(lon)), lon[cols].to_numpy()])
y = lon["Employed"].to_numpy()

cert = np.array([-3482.25863459582, 0.0150618722713733, -0.0358191792925910,
                 -0.0202022980381683, -0.0103322686717359, -0.0511041056535807,
                 1.82915146461355])                  # NIST StRD (unidades do R)

b_normais = np.linalg.solve(X.T @ X, X.T @ y)
Q, R = np.linalg.qr(X)
b_qr = sla.solve_triangular(R, Q.T @ y)

lre = lambda b: np.minimum(-np.log10(np.abs(b - cert) / np.abs(cert)), 15)
print(pd.DataFrame({"normais": lre(b_normais), "qr": lre(b_qr)}, index=["(Intercepto)"] + cols).round(1))
print(f"cond(X) = {np.linalg.cond(X):.2e}, cond(X'X) = {np.linalg.cond(X.T @ X):.2e}")
