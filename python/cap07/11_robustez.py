"""Cap. 7 -- Robustez adversarial de um classificador linear."""
from pathlib import Path

import numpy as np
import pandas as pd

raiz = Path(__file__).resolve().parents[2]
dig = pd.read_csv(raiz / "dados" / "digitos.csv")
X = np.column_stack([np.ones(len(dig)),
                     dig.drop(columns="digito").to_numpy() / 16])
y = dig["digito"].to_numpy()
tr, te = np.arange(1000), np.arange(1000, len(y))
Y = (y[tr, None] == np.arange(10)).astype(float)
U, s, Vt = np.linalg.svd(X[tr], full_matrices=False)
filtro = lambda f: Vt.T @ (f[:, None] * (U.T @ Y))
corte = lambda k: np.r_[1 / s[:k], np.zeros(65 - k)]
modelos = {"pseudo-inversa": filtro(corte(62)),
           "SVD k = 30": filtro(corte(30)),
           "ridge 10": filtro(s / (s**2 + 10))}
n = len(te); ii = np.arange(n)


def analisar(W):
    """Margens e raios certificados contra cada classe rival j."""
    S = X[te] @ W; c = S.argmax(1); P = W[1:]  # sem intercepto
    D = P[:, c].T[:, :, None] - P[None]  # w_c - w_j: n x 64 x 10
    M = S[ii, c][:, None] - S                   # margens: n x 10
    l1, l2 = np.abs(D).sum(1), np.linalg.norm(D, axis=1)
    l1[ii, c] = l2[ii, c] = 1; M[ii, c] = np.inf
    return c, M / l1, M / l2, D


def ataque(W, D, R1, eps):
    """Pior perturbação l_inf contra a rival mais próxima."""
    j = R1.argmin(1)
    Xa = X[te].copy()
    Xa[:, 1:] = np.clip(Xa[:, 1:] - eps * np.sign(D[ii, :, j]), 0, 1)
    return np.mean((Xa @ W).argmax(1) == y[te])


print("modelo         norma2 acur. raio_inf raio_2 cota_2"
      " rob_0,05 rob_0,1 ataque_0,1")
for nome, W in modelos.items():
    c, R1, R2, D = analisar(W)
    r_inf, r_2 = R1.min(1), R2.min(1)
    ok = c == y[te]
    n2 = np.linalg.norm(W[1:], 2)              # norma espectral
    m = np.sort(X[te] @ W, axis=1)
    cota = (m[:, -1] - m[:, -2]) / (np.sqrt(2) * n2)
    print(f"{nome:14s} {n2:6.2f} {ok.mean():5.3f}"
          f" {np.median(r_inf[ok]):8.4f} {np.median(r_2[ok]):6.3f}"
          f" {np.median(cota[ok]):6.3f}"
          f" {np.mean(ok & (r_inf > 0.05)):8.3f}"
          f" {np.mean(ok & (r_inf > 0.1)):7.3f}"
          f" {ataque(W, D, R1, 0.1):10.3f}")
