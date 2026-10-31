"""Cap. 4: Cadeia de Markov do mercado de trabalho: sistemas singulares compatíveis."""
import numpy as np

P = np.array([[0.960, 0.015, 0.025], [0.250, 0.600, 0.150], [0.040, 0.030, 0.930]])
r = np.array([3000.0, 600.0, 300.0])                 # E, D, F
A = np.eye(3) - P
print("posto(I - P):", np.linalg.matrix_rank(A))

M, d = np.vstack([A.T, np.ones(3)]), np.array([0.0, 0, 0, 1])
Gm = np.linalg.pinv(M)
print("compatível:", np.allclose(M @ Gm @ d, d))
pi = Gm @ d
print("pi =", np.round(pi, 4), " desemprego =", round(pi[1] / (pi[0] + pi[1]), 4))

G = np.linalg.pinv(A)
g = pi @ r
compat = lambda g: np.allclose(A @ G @ (r - g), r - g)
print(f"g = {g:.2f}; compatível com g: {compat(g)}; com 2500: {compat(2500)}")
rng = np.random.default_rng(3)
H = np.column_stack([G @ (r - g) + (np.eye(3) - G @ A) @ rng.normal(0, 1e4, 3) for _ in range(3)])
print("h_E - h_D:", np.round(H[0] - H[1], 2), " h_D - h_F:", np.round(H[1] - H[2], 2))
