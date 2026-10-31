"""Cap. 14: Poder do teste F e tamanho de amostra."""
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.gerador import Gerador  # noqa: E402


def poder(lam, q, nu, a=0.05):
    return stats.ncf.sf(stats.f.ppf(1 - a, q, nu), q, nu, lam)


# Dois grupos de r unidades, delta/sigma = 1: lambda = r/2
for r in [10, 15, 16, 17, 18, 20]:
    print(f"r {r:2d} lambda {r / 2:.1f}"
          f" poder {poder(r / 2, 1, 2 * r - 2):.4f}")
za, zb = stats.norm.ppf(0.975), stats.norm.ppf(0.8)
print(f"aproximação normal: r = {2 * (za + zb) ** 2:.2f}")
# Mesmo lambda, mais graus de liberdade no numerador: menos poder
for q in [1, 2, 3, 6, 12]:
    print(f"q {q:2d} poder(lambda = 10, nu = 30)"
          f" {poder(10, q, 30):.4f}")
# Conferência por simulação: 3 grupos de 5, sigma = 2 (Cap. 10)
g = Gerador(1414); N = 20000
mu = np.repeat([10.0, 11, 13], 5); cr = stats.f.ppf(0.95, 2, 12)
Y = mu + g.matriz_normal(N, 15, sd=2)
mg = Y.reshape(N, 3, 5).mean(axis=2)
qe = ((Y - np.repeat(mg, 5, axis=1)) ** 2).sum(axis=1) / 12
qa = 5 * ((mg - Y.mean(axis=1, keepdims=True)) ** 2).sum(axis=1) / 2
rej = np.sum(qa / qe > cr)
lam = 5 * np.sum((np.array([10, 11, 13]) - 34 / 3) ** 2) / 4
print(f"lambda {lam:.3f} poder {poder(lam, 2, 12):.4f}"
      f" simulado {rej / N:.4f}")
