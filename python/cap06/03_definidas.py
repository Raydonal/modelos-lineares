"""Cap. 6: Matrizes definidas positivas: critérios, Cholesky, ordem de Loewner."""
from pathlib import Path

import pandas as pd

import numpy as np
from scipy.interpolate import BSpline

A = np.array([[4.0, 2, -2], [2, 10, 4], [-2, 4, 9]])
print("autovalores:", np.linalg.eigvalsh(A).round(6))
print("menores líderes:", [round(np.linalg.det(A[:k, :k]), 6) for k in (1, 2, 3)])
L = np.linalg.cholesky(A)                                  # A = L L', L triangular inferior
print(L.round(6))
print("log det:", 2 * np.log(np.diag(L)).sum(), np.linalg.slogdet(A)[1])
C = A - np.diag([1.0, 2, 3])
print("min autovalor de C^-1 - A^-1:", np.linalg.eigvalsh(np.linalg.inv(C) - np.linalg.inv(A)).min())
mc = pd.read_csv(Path(__file__).resolve().parents[2] / "dados" / "mcycle.csv")
x = mc["times"].to_numpy(); h = (x.max() - x.min()) / 21
Bs = BSpline.design_matrix(x, x.min() + h * np.arange(-3, 25), 3).toarray()
D2 = np.diff(np.eye(Bs.shape[1]), n=2, axis=0)
S = Bs @ np.linalg.solve(Bs.T @ Bs + 0.7943282 * D2.T @ D2, Bs.T)
ev = np.linalg.eigvalsh(S)
print(f"P-spline: autovalores em [{ev.min():.2e}, {ev.max():.6f}], iguais a 1: {(ev > 1 - 1e-8).sum()}, traço = {ev.sum():.4f}")
