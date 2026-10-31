"""Cap. 1 -- Representação matricial do operador derivada em P_3 e mudança de base."""
import numpy as np

D = np.zeros((4, 4))
for k in range(1, 4):
    D[k - 1, k] = k                              # d/dt t^k = k t^(k-1)
print(D)
p = np.array([1, -2, 0, 5])                      # p(t) = 1 - 2t + 5t^3
print(D @ p, D @ D @ p)                          # p' e p''
print(np.linalg.matrix_power(D, 4))              # D^4 = 0

posto = np.linalg.matrix_rank(D)
print(f"dim imagem = {posto}, dim núcleo = {4 - posto}")

P = np.array([[1, -1, 1, -1],                    # colunas: 1, t-1, (t-1)^2, (t-1)^3
              [0, 1, -2, 3],
              [0, 0, 1, -3],
              [0, 0, 0, 1]], dtype=float)
print(np.linalg.solve(P, D @ P))                 # P^{-1} D P
