"""Cap. 4: Equações normais de um modelo superparametrizado: o que muda e o que não muda."""
import numpy as np

g = np.array([1, 1, 2, 2, 2, 3, 3])
y = np.array([12, 14, 18, 17, 19, 10, 11], dtype=float)
X = np.column_stack([np.ones(7), (g[:, None] == np.arange(1, 4)).astype(float)])
XtX, Xty = X.T @ X, X.T @ y

Gmp = np.linalg.pinv(XtX)
Gsub = np.zeros((4, 4)); Gsub[1:, 1:] = np.linalg.inv(XtX[1:, 1:])
b_mp, b_sub = Gmp @ Xty, Gsub @ Xty
print("soluções:\n", np.round(np.column_stack([b_mp, b_sub]), 4))

H_mp, H_sub = X @ Gmp @ X.T, X @ Gsub @ X.T
print(f"dif. entre as matrizes H: {np.abs(H_mp - H_sub).max():.1e}; traço = {np.trace(H_mp):.4f}")

L = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0], [0, 1, -1, 0]], dtype=float)
for nome, l in zip(["mu", "alfa1", "mu+alfa1", "alfa1-alfa2"], L):
    print(f"{nome:12s} {l @ b_mp:9.4f} {l @ b_sub:9.4f}")
