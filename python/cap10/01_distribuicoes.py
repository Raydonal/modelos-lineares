"""Cap. 10 -- Qui-quadrado não central, mistura de Poisson e a F."""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2, f, ncx2, poisson, t

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.gerador import Gerador  # noqa: E402

g = Gerador(29); N, k = 20000, 3
mu = np.array([2.0, 0, 0])                    # lambda = ||mu||^2 = 4
u = np.sum((g.matriz_normal(N, k) + mu)**2, 1)
q95 = chi2.ppf(0.95, k); j = np.arange(201)
mistura = np.sum(poisson.pmf(j, 2) * chi2.sf(q95, k + 2 * j))
print(f"média {u.mean():.4f} (teórica {k + 4}), variância"
      f" {u.var(ddof=1):.4f} (teórica {2 * k + 16})")
print(f"P(U > q95): simulada {np.mean(u > q95):.4f}, ncx2"
      f" {ncx2.sf(q95, k, 4):.4f}, mistura {mistura:.4f}")
# Poder do teste qui-quadrado: cresce com lambda
lam = np.array([0, 1, 4, 9, 16])
poder = [chi2.sf(q95, k) if l == 0 else ncx2.sf(q95, k, l)
         for l in lam]
print("poder:", np.round(poder, 4))
# t^2 com nu gl é F(1, nu)
print("t^2 =", t.ppf(0.975, 10)**2, " F =", f.ppf(0.95, 1, 10))
