"""Cap. 1 -- Uma rede linear de duas camadas treinada por gradiente aprende
a mesma função que a regressão de mínimos quadrados."""
import numpy as np

rng = np.random.default_rng(42)
n, p, h = 500, 5, 8
X = rng.normal(size=(n, p))
beta = np.array([2, -1, 0.5, 0, 3])
y = X @ beta + rng.normal(scale=0.5, size=n)

W1 = rng.normal(scale=0.3, size=(h, p))          # camada 1: R^5 -> R^8
w2 = rng.normal(scale=0.3, size=(1, h))          # camada 2: R^8 -> R
eta = 0.02
for it in range(3000):
    r = X @ (w2 @ W1).ravel() - y                # resíduos
    g_prod = (X.T @ r / n)[None, :]              # gradiente em relação a W = w2 W1
    g_w2, g_W1 = g_prod @ W1.T, w2.T @ g_prod    # regra da cadeia
    w2, W1 = w2 - eta * g_w2, W1 - eta * g_W1

mqo = np.linalg.lstsq(X, y, rcond=None)[0]
print("rede:", np.round((w2 @ W1).ravel(), 4))
print("mqo: ", np.round(mqo, 4))
