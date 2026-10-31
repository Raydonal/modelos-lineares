"""Cap. 2 -- Perda de ortogonalidade: Gram-Schmidt clássico, modificado e Householder."""
import numpy as np


def gs_classico(X):
    Q = X.copy()
    for j in range(X.shape[1]):
        v = X[:, j] - Q[:, :j] @ (Q[:, :j].T @ X[:, j])
        Q[:, j] = v / np.linalg.norm(v)
    return Q


def gs_modificado(X):
    Q = X.copy()
    for j in range(X.shape[1]):
        Q[:, j] /= np.linalg.norm(Q[:, j])
        Q[:, j + 1:] -= np.outer(Q[:, j], Q[:, j] @ Q[:, j + 1:])
    return Q


def matriz_cond(kappa, m=60, n=12, semente=1):
    rng = np.random.default_rng(semente)
    U = np.linalg.qr(rng.normal(size=(m, n)))[0]
    V = np.linalg.qr(rng.normal(size=(n, n)))[0]
    return U @ np.diag(kappa ** (-np.arange(n) / (n - 1))) @ V.T


perda = lambda Q: np.abs(Q.T @ Q - np.eye(Q.shape[1])).max()
print(f"{'kappa':>8} {'clássico':>10} {'modificado':>11} {'Householder':>12}")
for k in 10.0 ** np.array([2, 5, 8, 11]):
    X = matriz_cond(k)
    print(f"{k:8.0e} {perda(gs_classico(X)):10.1e} {perda(gs_modificado(X)):11.1e}"
          f" {perda(np.linalg.qr(X)[0]):12.1e}")
