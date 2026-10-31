"""Cap. 2: Posto de matrizes e desigualdades de posto."""
import numpy as np

posto = np.linalg.matrix_rank                   # posto via SVD (tolerância padrão)

A = np.array([[1, 2, 0, 3],
              [2, 4, 1, 7],
              [3, 6, 1, 10]])                    # linha 3 = linha 1 + linha 2
B = np.array([[1, 0], [0, 1], [1, 1], [2, -1]])
print("posto A, A', A'A:", posto(A), posto(A.T), posto(A.T @ A))
print("posto AB:", posto(A @ B), " cota:", min(posto(A), posto(B)),
      " Sylvester:", posto(A) + posto(B) - A.shape[1])

C = np.outer([1, -1, 2], [0, 1, 1, 0])           # posto 1
print("posto(A + C):", posto(A + C), " cota:", posto(A) + posto(C))
