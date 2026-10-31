"""Cap. 13: Mínimos quadrados generalizados: três exemplos"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.gerador import Gerador  # noqa: E402

# (a) Média comum de duas medidas com variâncias 1 e 4
V = np.diag([1., 4.]); Vi = np.linalg.inv(V); X = np.ones((2, 1))
w = np.linalg.solve(X.T @ Vi @ X, X.T @ Vi).ravel()  # pesos de MQG
print(f"pesos {w}, var_mqg {1 / Vi.trace():.2f},"
      f" var_media {V.trace() / 4:.2f}")
# (b) MQG = MQ nos dados "branqueados" por L^{-1}, V = LL'
g = Gerador(13)
n = 6; x = np.arange(1., n + 1); X = np.column_stack([np.ones(n), x])
y = 1 + 0.5 * x + g.normal(n)
i = np.arange(n); V = 0.6 ** np.abs(i[:, None] - i)  # AR(1), rho 0,6
L = np.linalg.cholesky(V)
b_form = np.linalg.solve(X.T @ np.linalg.solve(V, X),
                         X.T @ np.linalg.solve(V, y))
b_branq = np.linalg.lstsq(np.linalg.solve(L, X),
                          np.linalg.solve(L, y), rcond=None)[0]
print(np.column_stack([b_form, b_branq]).round(6))
# (c) Erros equicorrelacionados: com intercepto, MQ = MQG
V = 0.5 * np.eye(n) + 0.5                     # rho = 0,5


def mqg(X):
    return np.linalg.solve(X.T @ np.linalg.solve(V, X),
                           X.T @ np.linalg.solve(V, y))


def mq(X):
    return np.linalg.lstsq(X, y, rcond=None)[0]


print(f"com intercepto {np.max(np.abs(mqg(X) - mq(X))):.3e},"
      f" sem {np.max(np.abs(mqg(X[:, 1:]) - mq(X[:, 1:]))):.4f}")


# Eficiência de MQ para a média sob AR(1), e a cota de Kantorovich
def ef(n, rho):
    i = np.arange(n); V = rho ** np.abs(i[:, None] - i)
    l = np.linalg.eigvalsh(V)
    e = (n**2 / V.sum()) / np.linalg.solve(V, np.ones(n)).sum()
    k = 4 * l.max() * l.min() / (l.max() + l.min()) ** 2
    return e, k


for rho in [0.5, 0.9]:
    print("n=50, rho=%.1f: eficiência %.4f, Kantorovich %.4f"
          % (rho, *ef(50, rho)))
