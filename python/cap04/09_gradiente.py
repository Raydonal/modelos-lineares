"""Cap. 4: O gradiente descendente converge para a solução de norma mínima."""
import numpy as np

rng = np.random.default_rng(7)
n, p = 20, 100
X = rng.normal(size=(n, p)); y = rng.normal(size=n)
Xp = np.linalg.pinv(X); b_mn = Xp @ y
eta = 1 / np.linalg.norm(X, 2) ** 2


def gd(b, iter=3000):
    for _ in range(iter):
        b = b - eta * X.T @ (X @ b - y)
    return b


b_zero = gd(np.zeros(p)); b0 = rng.normal(size=p); b_alea = gd(b0)
Pn = np.eye(p) - Xp @ X
print(f"início 0:          distância à norma mínima = {np.linalg.norm(b_zero - b_mn):.1e}")
print(f"início aleatório:  distância à norma mínima = {np.linalg.norm(b_alea - b_mn):.2f}")
print(f"                   distância a b_mn + P_N b0 = {np.linalg.norm(b_alea - b_mn - Pn @ b0):.1e}")
