"""Cap. 9: Regressão por processo gaussiano (mcycle)."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cho_solve, cholesky, solve_triangular

raiz = Path(__file__).resolve().parents[2]
mc = pd.read_csv(raiz / "dados" / "mcycle.csv")
t = mc["times"].to_numpy(); y = mc["accel"].to_numpy(); n = len(y)
r = y - y.mean()
nucleo = lambda a, b, l: np.exp(-np.subtract.outer(a, b)**2
                                / (2 * l**2))


def perfil(l, tau):
    """Var(y) = s2 (C_l + tau I), com s2 de forma fechada."""
    R = cholesky(nucleo(t, t, l) + tau * np.eye(n))   # A = R'R
    u = solve_triangular(R.T, r, lower=True); s2 = u @ u / n
    return -n / 2 * np.log(2 * np.pi * s2) \
        - np.log(np.diag(R)).sum() - n / 2


ls = np.arange(1, 10.001, 0.25)
taus = 10 ** np.arange(-2, 1.001, 0.05)
ll = np.array([[perfil(l, tau) for l in ls] for tau in taus])
k = np.unravel_index(ll.argmax(), ll.shape)
l, tau = ls[k[1]], taus[k[0]]

A = nucleo(t, t, l) + tau * np.eye(n); Ac = cholesky(A)
s2 = r @ cho_solve((Ac, False), r) / n
gl = np.trace(cho_solve((Ac, False), nucleo(t, t, l)))
print(f"l = {l}, tau = {tau:.4f}, sigma_f = {np.sqrt(s2):.3f},"
      f" sigma = {np.sqrt(s2 * tau):.3f}, logver = {ll.max():.3f},"
      f" gl = {gl:.3f}")

# Posterior de f nos tempos t* (condicional da normal conjunta)
ts = np.array([10.0, 15, 20, 30, 40, 50])
Cs = nucleo(ts, t, l)
media = y.mean() + Cs @ cho_solve((Ac, False), r)
quad = np.sum(Cs * cho_solve((Ac, False), Cs.T).T, 1)
dp = np.sqrt(s2 * (1 - quad))
print(np.vstack([ts, media, dp]).round(2))
