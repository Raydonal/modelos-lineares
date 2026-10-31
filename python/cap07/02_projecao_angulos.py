"""Cap. 7: Norma de projeções oblíquas e ângulos principais."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.gerador import Gerador  # noqa: E402

n2 = lambda M: np.linalg.norm(M, 2)


def angulos(A, B):
    Qa, Qb = np.linalg.qr(A)[0], np.linalg.qr(B)[0]
    return np.degrees(np.arccos(np.minimum(1, np.linalg.svd(Qa.T @ Qb, compute_uv=False))))


P = np.array([[1.0, 0, -1], [0, 1, -1], [0, 0, 0]])
th = angulos(P[:, :2], P.T[:, :2])
print(f"||P|| = {n2(P):.6f}, ||I-P|| = {n2(np.eye(3) - P):.6f}, 1/cos(theta_max) = {1 / np.cos(np.radians(th.max())):.6f}")
g = Gerador(72)
B, C = g.matriz_normal(8, 3), g.matriz_normal(8, 3)
Pq = B @ np.linalg.solve(C.T @ B, C.T)
thq = angulos(B, C)
print(f"8 x 8: ||P|| = {n2(Pq):.6f}, ||I-P|| = {n2(np.eye(8) - Pq):.6f}, 1/cos = {1 / np.cos(np.radians(thq.max())):.6f}; ângulos {thq.round(2)}")
