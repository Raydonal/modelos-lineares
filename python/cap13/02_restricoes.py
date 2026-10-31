"""Cap. 13 -- Mínimos quadrados restritos: retornos constantes"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.gerador import Gerador  # noqa: E402

g = Gerador(1928)
n = 30                                        # 30 firmas simuladas
lK = 2 + g.normal(n); lL = 3 + 0.8 * lK + 0.6 * g.normal(n)
lQ = 1 + 0.3 * lK + 0.7 * lL + 0.4 * g.normal(n)  # alfa + gama = 1
X = np.column_stack([np.ones(n), lK, lL]); A = np.linalg.inv(X.T @ X)
b = A @ X.T @ lQ                              # irrestrito
K = np.array([0., 1, 1]); m = 1.              # restrição K'b = m
bK = b - A @ K * (K @ b - m) / (K @ A @ K)
# Conferência: impor a restrição reescrevendo o modelo
Z = np.column_stack([np.ones(n), lK - lL])
r = np.linalg.lstsq(Z, lQ - lL, rcond=None)[0]
print("irrestrito", b.round(4))
print("restrito  ", bK.round(4))
print("reescrito ", np.array([r[0], r[1], 1 - r[1]]).round(4))
s2 = np.sum((lQ - X @ b) ** 2) / (n - 3)
CK = A - np.outer(A @ K, K @ A) / (K @ A @ K)
print("ep irrestrito", np.sqrt(s2 * np.diag(A)).round(4))
print("ep restrito  ", np.sqrt(s2 * np.diag(CK)).round(4))
# Aumento da SQRes = (K'b - m)^2 / K'AK
d = np.sum((lQ - X @ bK) ** 2) - np.sum((lQ - X @ b) ** 2)
f = (K @ b - m) ** 2 / (K @ A @ K)
print(f"dif SQRes {d:.8f}, fórmula {f:.8f}")
