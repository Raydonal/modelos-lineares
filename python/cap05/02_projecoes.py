"""Cap. 5: Projeções oblíquas e ortogonais em R^3."""
import numpy as np
from scipy.linalg import null_space, qr

B = np.array([[1.0, 0], [0, 1], [0, 0]])
w = np.ones(3)
C = null_space(w[None, :])                             # N(C') = ger{w}
P = B @ np.linalg.solve(C.T @ B, C.T)
print(P.round(4) + 0.0)
y = np.array([0.0, 0, 1])
print("P w =", (P @ w).round(12) + 0.0, " P y =", P @ y + 0.0)
print(f"|y| = 1, |Py| = {np.linalg.norm(P @ y):.4f}, |P| = {np.linalg.norm(P, 2):.4f}")
Po = B @ np.linalg.solve(B.T @ B, B.T)
print(f"ortogonal: |Po y| = {np.linalg.norm(Po @ y):.1f}, |Po| = {np.linalg.norm(Po, 2):.1f}")

rng = np.random.default_rng(2)
X = np.column_stack([np.ones(8), rng.normal(size=(8, 2))]); X = np.column_stack([X, X[:, 1] + X[:, 2]])
H1 = X @ np.linalg.pinv(X.T @ X) @ X.T
Q, R, piv = qr(X, mode="economic", pivoting=True)
H2 = Q[:, :3] @ Q[:, :3].T
H3 = X @ np.linalg.pinv(X)
print("diferenças entre fórmulas:", np.abs(H1 - H2).max(), np.abs(H1 - H3).max(), " traço:", np.trace(H1).round(10))

W = np.diag(np.arange(1.0, 9))
PW = X @ np.linalg.pinv(X.T @ W @ X) @ X.T @ W
print("P_W idempotente:", np.allclose(PW @ PW, PW), " simétrica:", np.allclose(PW, PW.T),
      " W P_W simétrica:", np.allclose(W @ PW, (W @ PW).T))
