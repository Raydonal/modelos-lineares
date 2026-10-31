"""Cap. 6 -- Raio espectral, série de Neumann e matrizes estocásticas positivas."""
import numpy as np

raio = lambda M: np.abs(np.linalg.eigvals(M)).max()
A = np.array([[0.5, 4.0], [0.0, 0.6]])
print("raio:", raio(A), " max|A^k|:", [round(np.abs(np.linalg.matrix_power(A, k)).max(), 6) for k in (1, 2, 5, 10, 20, 50)])
T = np.array([[0.20, 0.30, 0.10], [0.10, 0.25, 0.20], [0.15, 0.10, 0.30]])
Lt = np.linalg.inv(np.eye(3) - T)
soma = sum(np.linalg.matrix_power(T, k) for k in range(61))
print(f"raio(T) = {raio(T):.4f}, erro de Neumann = {np.abs(soma - Lt).max():.1e}")
print(Lt.round(4), "\nmultiplicadores:", Lt.sum(0).round(4))
P = np.array([[0.960, 0.015, 0.025], [0.250, 0.600, 0.150], [0.040, 0.030, 0.930]])
osc = lambda v: v.max() - v.min()
v = np.array([1.0, 0, 0]); razoes = []
for _ in range(6):
    w = P @ v; razoes.append(osc(w) / osc(v)); v = w
print("1 - n delta =", 1 - 3 * P.min(), " razões:", np.round(razoes, 4))
