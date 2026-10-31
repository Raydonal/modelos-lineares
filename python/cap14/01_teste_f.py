"""Cap. 14: Teste F da hipótese linear geral: três contas iguais."""
import numpy as np
from scipy import stats

# Dois grupos, modelo superparametrizado (posto 2 em 3 colunas)
y = np.array([1.0, 3, 4, 6, 5]); n = 5
X = np.column_stack([np.ones(5), [1, 1, 0, 0, 0], [0, 0, 1, 1, 1]])
K = np.array([[0.0, 1, -1]]); m = 0; q = 1
r = np.linalg.matrix_rank(X)
Gs = [np.diag([0, 1 / 2, 1 / 3]), np.linalg.pinv(X.T @ X)]
for G in Gs:
    b = G @ X.T @ y
    d = (K @ b).item() - m
    KGK = (K @ G @ K.T).item()
    Q = d**2 / KGK
    s2 = np.sum((y - X @ b) ** 2) / (n - r)
    print(f"Kb {d:.4f} KGK' {KGK:.4f} Q {Q:.4f} s2 {s2:.4f}"
          f" F {Q / q / s2:.4f}")
# Modelos encaixados: SQR restrita - SQR irrestrita
G = Gs[0]; b = G @ X.T @ y
KGK = K @ G @ K.T
bK = b - G @ K.T @ np.linalg.solve(KGK, K @ b - m)
sqr0 = np.sum((y - X @ bK) ** 2); sqr1 = np.sum((y - X @ b) ** 2)
Fobs = (sqr0 - sqr1) / q / (sqr1 / (n - r))
print(f"SQR0 {sqr0:.4f} SQR1 {sqr1:.4f} F {Fobs:.4f}")
print(f"p {stats.f.sf(Fobs, q, n - r):.4f}"
      f" crit5% {stats.f.ppf(0.95, q, n - r):.3f}")
# Geometria: P1 - P0 é a projeção em Col(XGK')
A = X @ G @ K.T
PA = A @ np.linalg.solve(A.T @ A, A.T)
P1 = X @ G @ X.T
P0 = np.full((n, n), 1 / n)
print(f"max|P1-P0-PA| {np.abs(P1 - P0 - PA).max():.1e}"
      f"  y'PAy {y @ PA @ y:.4f}")
# Hipótese não estimável: alfa_1 = 0 não muda o ajuste
Z = X[:, [0, 2]]
ez = y - Z @ np.linalg.solve(Z.T @ Z, Z.T @ y)
print(f"SQR sob alfa1=0 {ez @ ez:.4f} (igual a SQR1 {sqr1:.4f})")
# Três pontos: H0 beta_1 = 0 e H0 (beta_0, beta_1) = (1, 1)
x = np.arange(3.0); y3 = np.array([1.0, 2, 4])
X3 = np.column_stack([np.ones(3), x]); A3 = np.linalg.inv(X3.T @ X3)
b3 = A3 @ X3.T @ y3
s23 = np.sum((y3 - X3 @ b3) ** 2) / 1
d = b3 - np.array([1.0, 1])
F2 = d @ np.linalg.solve(A3, d) / 2 / s23
print(f"3 pontos: b ({b3[0]:.4f}, {b3[1]:.4f}) s2 {s23:.4f}"
      f" F(q=1) {b3[1]**2 / A3[1, 1] / s23:.1f} F(q=2) {F2:.2f}")
