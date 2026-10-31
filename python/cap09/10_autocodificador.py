"""Cap. 9 -- Autocodificador linear, PPCA e VAE linear."""
from pathlib import Path

import numpy as np
import pandas as pd

raiz = Path(__file__).resolve().parents[2]
dig = pd.read_csv(raiz / "dados" / "digitos.csv")
X = dig.iloc[:, 1:].to_numpy() / 16
tr, te = X[:1000], X[1000:]
n, p = tr.shape
mu = tr.mean(0); S = (tr - mu).T @ (tr - mu) / n     # EMV
d, Q = np.linalg.eigh(S); d, Q = d[::-1], Q[:, ::-1]
d = np.maximum(d, 0)                   # zeros numéricos


def ppca(k, s2=None):
    """EMV de Tipping e Bishop (s2 livre) ou W ótimo com s2 fixo."""
    if s2 is None:
        s2 = d[k:].mean()
    return Q[:, :k] * np.sqrt(np.maximum(d[:k] - s2, 0)), s2


def logver(W, s2, R):
    """Log-densidade média de N(mu, WW' + s2 I) (Woodbury)."""
    k = W.shape[1]; G = np.eye(k) + W.T @ W / s2
    U = R @ W
    quad = np.sum(R**2, 1) - np.sum(U @ np.linalg.inv(G) * U, 1) / s2
    quad /= s2
    logdet = p * np.log(s2) + np.linalg.slogdet(G)[1]
    return np.mean(-0.5 * (p * np.log(2 * np.pi) + logdet + quad))


Rtr, Rte = tr - mu, te - mu
print("  k  erro_pca_teste  sigma2  logver_treino  logver_teste")
for k in (1, 2, 5, 10, 20, 30, 40, 50, 55):
    P = Q[:, :k] @ Q[:, :k].T
    erro = np.mean(np.sum((Rte - Rte @ P)**2, 1))
    W, s2 = ppca(k)
    print(f"{k:3d} {erro:14.3f} {s2:8.1e} {logver(W, s2, Rtr):13.2f}"
          f" {logver(W, s2, Rte):13.2f}")


# ELBO = log p(x) com o codificador exato; folga = KL(q || post.)
def kl_normal(m1, C1, m2, C2):
    """KL(N(m1, C1) || N(m2, C2))."""
    C2i = np.linalg.inv(C2); dm = m2 - m1
    ld = np.linalg.slogdet(C2)[1] - np.linalg.slogdet(C1)[1]
    return 0.5 * (np.trace(C2i @ C1) + dm @ C2i @ dm - len(m1) + ld)


def elbo(r, W, s2, m, C):
    k = W.shape[1]; res = r - W @ m
    rec = -0.5 * (p * np.log(2 * np.pi * s2)
                  + (res @ res + np.trace(W @ C @ W.T)) / s2)
    return rec - kl_normal(m, C, np.zeros(k), np.eye(k))


W, s2 = ppca(10); r = Rte[0]; k = 10
G = np.eye(k) + W.T @ W / s2
m_post = np.linalg.solve(G, W.T @ r) / s2; C_post = np.linalg.inv(G)
m_pca = np.linalg.lstsq(W, r, rcond=None)[0]   # outro codificador
print(f"log p(x) = {logver(W, s2, r[None]):.6f},"
      f" ELBO exata = {elbo(r, W, s2, m_post, C_post):.6f}")
folga = logver(W, s2, r[None]) - elbo(r, W, s2, m_pca, C_post)
print(f"folga do codificador pseudo-inversa = {folga:.6f},"
      f" KL = {kl_normal(m_pca, C_post, m_post, C_post):.6f}")

# Colapso posterior: k = 20 dimensões latentes, sigma2 fixo
for s2 in (0.002, 0.01, 0.05, 0.1, 0.3):
    W, _ = ppca(20, s2)
    ativas = int(np.sum(np.linalg.norm(W, axis=0) > 0))
    print(f"sigma2 = {s2}: dimensões ativas = {ativas},"
          f" logver teste = {logver(W, s2, Rte):.2f}")
print("autovalores:", d[[0, 4, 9, 19, 29, 39]].round(4))
