"""Cap. 5 -- Suavizadores lineares: projeções e não projeções (dados mcycle)."""
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.width", 120)
from scipy.interpolate import BSpline

mc = pd.read_csv(Path(__file__).resolve().parents[2] / "dados" / "mcycle.csv")
x = mc["times"].to_numpy(); y = mc["accel"].to_numpy(); n = len(y)


def base(K):
    h = (x.max() - x.min()) / (K + 1)
    nos = x.min() + h * np.arange(-3, K + 5)
    return BSpline.design_matrix(x, nos, 3).toarray()


def resumo(S):
    r = y - S @ y; d = np.diag(S); gl = np.trace(S)
    return {"gl": gl, "tr_StS": np.trace(S.T @ S), "erro_idemp": np.abs(S @ S - S).max(),
            "assimetria": np.abs(S - S.T).max(), "max_Sii": d.max(),
            "CV": np.mean((r / (1 - d)) ** 2), "GCV": np.mean(r ** 2) / (1 - gl / n) ** 2}


proj = lambda B: B @ np.linalg.solve(B.T @ B, B.T)
Ks = np.arange(2, 21)
K_opt = Ks[np.argmin([resumo(proj(base(K)))["CV"] for K in Ks])]
B = base(20); Dd = np.diff(np.eye(B.shape[1]), n=2, axis=0)
S_pen = lambda l: B @ np.linalg.solve(B.T @ B + l * Dd.T @ Dd, B.T)
lams = 10 ** np.linspace(-2, 4, 121)
l_opt = lams[np.argmin([resumo(S_pen(l))["GCV"] for l in lams])]


def nw(b):
    W = np.exp(-0.5 * (x[:, None] - x[None, :]) ** 2 / b ** 2)
    return W / W.sum(1, keepdims=True)


bs = np.round(np.arange(0.5, 4.0001, 0.05), 2)
b_opt = bs[np.argmin([resumo(nw(b))["CV"] for b in bs])]
print(f"K = {K_opt}, lambda = {l_opt:.4f}, janela = {b_opt}")
tab = pd.DataFrame({"spline de regressão": resumo(proj(base(K_opt))), "P-spline": resumo(S_pen(l_opt)),
                    "Nadaraya-Watson": resumo(nw(b_opt))}).T
print(tab.round(4))
