"""Cap. 1: Processo de Gram-Schmidt (versão clássica)."""
import numpy as np


def gram_schmidt(X):
    n, k = X.shape
    Q = np.zeros((n, k))
    for j in range(k):
        v = X[:, j].astype(float)
        for i in range(j):
            v -= (Q[:, i] @ X[:, j]) * Q[:, i]   # remove a componente em u_i
        nv = np.linalg.norm(v)
        if nv < 1e-12:
            raise ValueError("colunas linearmente dependentes")
        Q[:, j] = v / nv                        # normaliza
    return Q


t = np.arange(1, 5)
X = np.column_stack([np.ones(4), t, t**2])      # 1, t, t^2
Q = gram_schmidt(X)
print(np.round(Q, 4))
print(np.round(Q.T @ Q, 12))                    # Q'Q = I
R = Q.T @ X                                     # triangular superior
print(np.round(R, 4))
print(np.abs(Q @ R - X).max())                  # X = QR
