"""Cap. 9: Normal multivariada: densidade, condicionais."""
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.gerador import Gerador  # noqa: E402

mu = np.array([1.0, 2, 0])
Sigma = np.array([[4.0, 2, 1], [2, 3, -1], [1, -1, 2]])
L = np.linalg.cholesky(Sigma)


def logdens(x):
    """Pela Cholesky: z = L^{-1}(x - mu), log det = 2 sum log L_ii."""
    z = solve_triangular(L, x - mu, lower=True)
    return -0.5 * (3 * np.log(2 * np.pi) + z @ z) \
        - np.log(np.diag(L)).sum()


x0 = np.array([2.0, 1, 1])
d = x0 - mu
print("cholesky:", logdens(x0), " fórmula:",
      -0.5 * (3 * np.log(2 * np.pi) + np.log(np.linalg.det(Sigma))
              + d @ np.linalg.solve(Sigma, d)))

# Condicional de x1 dado (x2, x3) = (3, -1): complemento de Schur
i1, i2, x2 = 0, [1, 2], np.array([3.0, -1])
B = Sigma[i1, i2] @ np.linalg.inv(Sigma[np.ix_(i2, i2)])
print("média:", mu[i1] + B @ (x2 - mu[i2]),
      " variância:", Sigma[i1, i1] - B @ Sigma[i2, i1])

# Simulação: 100 000 vetores; regressão de x1 em (x2, x3)
g = Gerador(11); N = 100_000
X = g.matriz_normal(N, 3) @ L.T + mu
C = np.cov(X.T)
print(np.r_[X.mean(0), C[np.tril_indices(3)]].round(3))
Z = np.column_stack([np.ones(N), X[:, 1:]])
coef, *_ = np.linalg.lstsq(Z, X[:, 0], rcond=None)
res = X[:, 0] - Z @ coef
print("coeficientes:", coef.round(3), " var. resid.:",
      round(res.var(ddof=1), 3), " B:", B.round(3))

# Distância de Mahalanobis: d2 = z'z ~ qui-quadrado com 3 gl
d2 = (solve_triangular(L, (X - mu).T, lower=True) ** 2).sum(0)
print("média:", d2.mean().round(4), " quantil 95%:",
      np.quantile(d2, 0.95).round(4),
      " qui2:", chi2.ppf(0.95, 3).round(4))

# Normal singular: x = (z1, z2, z1 + z2), no plano x1 + x2 = x3
S2 = np.array([[1.0, 0, 1], [0, 1, 1], [1, 1, 2]])
lam, V = np.linalg.eigh(S2)
print(lam[::-1].round(12), V[:, 0].round(4))  # base de N(Sigma)

# Não correlacionadas mas dependentes: Y = S X, S = +-1 ao acaso
xs = g.normal(N); s = np.where(g.unif(N) < 0.5, -1, 1); ys = s * xs
print("correlação:", np.corrcoef(xs, ys)[0, 1].round(4),
      " corr. dos quadrados:", np.corrcoef(xs**2, ys**2)[0, 1])
