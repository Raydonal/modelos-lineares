"""Cap. 3 -- Fluxo normalizante com uma camada de acoplamento afim (Jacobiano triangular)."""
import numpy as np
from scipy.stats import norm

s_fun = lambda x1: np.log(0.5) + 0.3 * x1
t_fun = lambda x1: 0.5 * x1**2 - 1


def inversa(x):
    return np.column_stack([x[:, 0], (x[:, 1] - t_fun(x[:, 0])) / np.exp(s_fun(x[:, 0]))])


def log_dens(x):
    return norm.logpdf(inversa(x)).sum(axis=1) - s_fun(x[:, 0])


x0, h = np.array([0.7, 0.4]), 1e-6
J = np.column_stack([(inversa((x0 + h * e)[None]) - inversa((x0 - h * e)[None]))[0] / (2 * h)
                     for e in np.eye(2)])
print(np.log(abs(np.linalg.det(J))), -s_fun(x0[0]))
g1, g2 = np.linspace(-6, 6, 401), np.linspace(-4, 20, 801)
G = np.array(np.meshgrid(g1, g2)).reshape(2, -1).T
print(np.exp(log_dens(G)).sum() * (g1[1] - g1[0]) * (g2[1] - g2[0]))
