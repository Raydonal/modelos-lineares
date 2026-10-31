"""Cap. 1: Operações básicas com vetores e matrizes."""
import numpy as np

A = np.array([[2, 1],
              [0, 3],
              [1, -1]])                  # 3 x 2
B = np.array([[1, 4, 2],
              [0, 1, 5]])                # 2 x 3
x = np.array([2, -1])

print(A @ B)                             # produto 3 x 3
print((A @ B).T - B.T @ A.T)             # (AB)' = B'A'  -> matriz nula
print(np.trace(A @ B) - np.trace(B @ A)) # tr(AB) = tr(BA) -> 0

# Ax como combinação linear das colunas de A
print(A @ x, x[0] * A[:, 0] + x[1] * A[:, 1])

# Produto como soma de produtos externos: AB = sum_k a_k b_k'
print(sum(np.outer(A[:, k], B[k, :]) for k in range(2)))
