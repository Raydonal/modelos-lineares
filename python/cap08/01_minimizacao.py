"""Cap. 8: Minimização de funções quadráticas, com e sem restrições lineares."""
import numpy as np
from scipy.optimize import minimize_scalar

A = np.array([[2.0, -1, -1], [-1, 2, -1], [-1, -1, 2]])
f = lambda x, b: 0.5 * x @ A @ x - b @ x
b1, b2 = np.array([1.0, -2, 1]), np.ones(3)
x1 = np.linalg.pinv(A) @ b1
print("autovalores:", np.linalg.eigvalsh(A).round(10), " mínimo:", f(x1, b1), " fórmula:", -0.5 * b1 @ np.linalg.pinv(A) @ b1)
print("f(x1 + t1):", [round(float(f(x1 + t, b1)), 10) for t in (0, 5, -3)], " f(t1) com b fora de C(A):", [float(f(t * np.ones(3), b2)) for t in (1, 10, 100)])
A2 = np.diag([1.0, 2, 3]); bb = np.ones(3); C = np.ones((1, 3))
M = np.block([[A2, C.T], [C, np.zeros((1, 1))]])
sol = np.linalg.solve(M, np.r_[bb, 1]); x = sol[:3]
print("restrito:", x.round(6), "multiplicador:", round(sol[3], 6), "gradiente/C:", (A2 @ x - bb).round(6))
S = np.array([[4.0, 2, 1], [2, 3, 1], [1, 1, 2]]); x1v = np.array([1.0, 2])
schur = x1v @ (S[:2, :2] - np.outer(S[:2, 2], S[2, :2]) / S[2, 2]) @ x1v
num = minimize_scalar(lambda t: np.r_[x1v, t] @ S @ np.r_[x1v, t]).fun
print(f"mínimo em x2: numérico = {num:.6f}, complemento de Schur = {schur:.6f}")
