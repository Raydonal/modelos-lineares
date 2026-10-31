"""Cap. 4: Existência e não unicidade de inversas generalizadas."""
import numpy as np

A = np.array([[1, 2, 0, 1], [2, 4, 1, 1], [3, 6, 1, 2]], dtype=float)   # 3 x 4, posto 2
print("posto:", np.linalg.matrix_rank(A))

# (1) Fatoração de posto A = B C
B = A[:, [0, 2]]
C = np.array([[1, 2, 0, 1], [0, 0, 1, -1]], dtype=float)
G1 = C.T @ np.linalg.inv(C @ C.T) @ np.linalg.inv(B.T @ B) @ B.T

# (2) Submatriz 2 x 2 não singular (linhas 1-2, colunas 1 e 3)
G2 = np.zeros((4, 3))
G2[np.ix_([0, 2], [0, 1])] = np.linalg.inv(A[np.ix_([0, 1], [0, 2])])

# (3) Forma geral G = G0 + U - G0 A U A G0
rng = np.random.default_rng(4)
U = np.round(rng.normal(size=(4, 3)), 1)
G3 = G2 + U - G2 @ A @ U @ A @ G2

for nome, G in (("G1", G1), ("G2", G2), ("G3", G3)):
    print(nome, "erro A G A - A =", f"{np.abs(A @ G @ A - A).max():.1e}",
          " posto(G) =", np.linalg.matrix_rank(G))
print("G1 coincide com a de Moore-Penrose:", np.allclose(G1, np.linalg.pinv(A)))
