"""Cap. 13: Erros AR(1) na regressão de acidentes (Seatbelts)"""
from pathlib import Path

import numpy as np
import pandas as pd

raiz = Path(__file__).resolve().parents[2]
s = pd.read_csv(raiz / "dados" / "seatbelts.csv"); n = len(s)
y = np.log(s.motoristas.to_numpy(float))
X = np.column_stack([np.log(s.km), np.log(s.preco_gasolina), s.lei,
                     np.eye(12)[s.mes - 1]])
p = X.shape[1]; A = np.linalg.inv(X.T @ X)
b = A @ X.T @ y; e = y - X @ b
print(f"r1 {e[1:] @ e[:-1] / (e @ e):.7f}")   # autocorrelação lag 1


def pw(M, r):                                 # Prais-Winsten: L^{-1}
    M = M.reshape(len(M), -1)
    return np.vstack([np.sqrt(1 - r**2) * M[:1], M[1:] - r * M[:-1]])


def perfil(r):                                # log-veross. perfilada
    z = pw(y, r).ravel(); W = pw(X, r)
    bz = np.linalg.lstsq(W, z, rcond=None)[0]
    return -n / 2 * np.log(np.sum((z - W @ bz) ** 2) / n) \
        + np.log(1 - r**2) / 2


grade = np.round(np.arange(0, 0.9505, 0.001), 3)
rho = grade[np.argmax([perfil(r) for r in grade])]
z = pw(y, rho).ravel(); W = pw(X, rho); Ag = np.linalg.inv(W.T @ W)
bg = Ag @ W.T @ z; s2g = np.sum((z - W @ bg) ** 2) / (n - p)
# Erros-padrão do coeficiente da lei
i = np.arange(n); V = rho ** np.abs(i[:, None] - i) / (1 - rho**2)
s2 = e @ e / (n - p)
s2V = e @ e / np.trace((np.eye(n) - X @ A @ X.T) @ V)
ep_sand = np.sqrt(np.diag(s2V * A @ X.T @ V @ X @ A))
j = 2
print(f"rho {rho:.4f}, mq {b[j]:.4f}, ep ingênuo"
      f" {np.sqrt(s2 * A[j, j]):.4f}, ep AR(1) {ep_sand[j]:.4f}")
print(f"mqg {bg[j]:.4f}, ep mqg {np.sqrt(s2g * Ag[j, j]):.4f}")
print(f"efeito em %: mq {100 * (np.exp(b[j]) - 1):.1f},"
      f" mqg {100 * (np.exp(bg[j]) - 1):.1f}")
