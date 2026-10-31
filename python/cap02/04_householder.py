"""Cap. 2 -- Fatoração QR por reflexões de Householder."""
import numpy as np


def householder_qr(A):
    R = np.array(A, dtype=float); m, n = R.shape; Q = np.eye(m)
    for k in range(min(n, m - 1)):
        x = R[k:, k]
        alfa = -np.copysign(np.linalg.norm(x), x[0])       # sinal que evita cancelamento
        v = x.copy(); v[0] -= alfa                          # v = x - alfa e1
        if v @ v == 0:
            continue
        v /= np.linalg.norm(v)
        R[k:] -= 2 * np.outer(v, v @ R[k:])                 # aplica H sem formá-la
        Q[:, k:] -= 2 * np.outer(Q[:, k:] @ v, v)           # Q = H1 H2 ... Hn
    R[np.abs(R) < 1e-14] = 0
    return Q, R


t = np.arange(1.0, 6)
X = np.column_stack([np.ones(5), t, t**2])
Q, R = householder_qr(X)
print(np.round(R, 4))
print(np.abs(Q @ R - X).max(), np.abs(Q.T @ Q - np.eye(5)).max())
print(np.round(np.linalg.qr(X)[1], 4))                      # R do numpy
