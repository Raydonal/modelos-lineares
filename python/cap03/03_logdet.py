"""Cap. 3: Determinantes grandes: LU, log-determinante e estouro numérico."""
import numpy as np
import scipy.linalg as sla

rng = np.random.default_rng(1)
n = 1200
A = 0.5 * np.eye(n) + rng.normal(scale=0.01, size=(n, n))
print(np.linalg.det(A))                          # subfluxo
print(np.linalg.slogdet(A))                      # (sinal, log|det|), via LU
P, L, U = sla.lu(A)
print(np.sum(np.log(np.abs(np.diag(U)))), n * np.log(0.5))
