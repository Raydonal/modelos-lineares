"""Cap. 12: Gauss-Markov: três estimadores da inclinação"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.gerador import Gerador  # noqa: E402

x = np.arange(5.); X = np.column_stack([np.ones(5), x]); n = 5
A = np.vstack([np.linalg.solve(X.T @ X, X.T)[1],   # a'y de cada um
               np.array([-1, 0, 0, 0, 1]) / 4,
               np.array([-1, -1, 0, 1, 1]) / 6])
nomes = ["mq", "pontas", "pares"]
print(A.round(3) + 0.0)
for nm, a in zip(nomes, A):                   # a'X = (0, 1)?
    print(f"{nm:7} a'X = {(a @ X).round(4) + 0.0},"
          f" var/s2 = {a @ a:.4f}")
# Monte Carlo com o gerador portátil: beta = (2, 0,5), sigma = 1
g = Gerador(12); R = 20000
E = g.matriz_normal(R, n)
Y = 2 + 0.5 * x + E
est = Y @ A.T
print("média", est.mean(0).round(4))
print("var  ", est.var(0, ddof=1).round(4))
# Versão matricial: Cov(Ay) - Cov(b_mq) é semidefinida positiva
Ap = np.vstack([[1, 0, 0, 0, 0], A[1]])       # (y_1, inclinação)
Cmq = np.linalg.inv(X.T @ X); Cp = Ap @ Ap.T
print(np.linalg.eigvalsh(Cp - Cmq)[::-1].round(4))
