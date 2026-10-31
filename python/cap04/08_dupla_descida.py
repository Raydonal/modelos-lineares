"""Cap. 4 -- Interpolação de norma mínima e dupla descida (modelo de "features fracas")."""
import numpy as np

rng = np.random.default_rng(2026)
D, n, sigma = 500, 50, 0.5
beta = np.full(D, 2 / np.sqrt(D))
ps = [5, 10, 20, 30, 40, 45, 48, 50, 52, 55, 60, 80, 100, 200, 300, 500]
R = np.empty((200, len(ps)))
for r in range(200):
    X = rng.normal(size=(n, D)); y = X @ beta + sigma * rng.normal(size=n)
    for j, p in enumerate(ps):
        b = np.linalg.pinv(X[:, :p]) @ y
        R[r, j] = np.sum((beta[:p] - b) ** 2) + np.sum(beta[p:] ** 2) + sigma**2
for p, v in zip(ps, np.median(R, axis=0)):
    print(f"p = {p:3d}: risco mediano = {v:7.2f}")
print("risco do preditor nulo:", beta @ beta + sigma**2)
