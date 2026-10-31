"""Comparações múltiplas: Dunnett e Benjamini-Hochberg.

Mesmas contas de R/comum/multiplas.R.
"""
import numpy as np
from scipy import stats


def conf_dunnett(d, k, nu, M=400):
    """P(|T_i| <= d, i = 1..k): k tratamentos, n iguais."""
    z = np.arange(-160, 161) * 0.05
    w = stats.norm.pdf(z) * 0.05
    u = (np.arange(1, M + 1) - 0.5) / M
    s = np.sqrt(stats.chi2.ppf(u, nu) / nu)
    c = np.sqrt(2) * d * s[:, None]
    return np.mean(np.sum(w * (stats.norm.cdf(z + c)
                               - stats.norm.cdf(z - c)) ** k, axis=1))


def q_dunnett(k, nu, alfa=0.05):
    a, b = 1.0, 12.0
    for _ in range(30):
        d = (a + b) / 2
        if conf_dunnett(d, k, nu) < 1 - alfa:
            a = d
        else:
            b = d
    return (a + b) / 2


def bh(p, alfa=0.05):
    """Número de rejeições de Benjamini-Hochberg."""
    o = np.sort(np.asarray(p)); m = len(o)
    k = np.nonzero(o <= alfa * np.arange(1, m + 1) / m)[0]
    return int(k[-1] + 1) if len(k) else 0
