"""Cap. 7: A inversa de Moore-Penrose pela SVD: propriedades, limite ridge e descontinuidade."""
import numpy as np

A = np.array([[1.0, 2, 0, 1], [2, 4, 1, 1], [3, 6, 1, 2]])
U, s, Vt = np.linalg.svd(A, full_matrices=False)
r = (s > 1e-10 * s[0]).sum()
Ap = Vt[:r].T @ np.diag(1 / s[:r]) @ U[:, :r].T
print("condições de Penrose:", [float(np.abs(M).max()) for M in (A @ Ap @ A - A, Ap @ A @ Ap - Ap,
                                                                  A @ Ap - (A @ Ap).T, Ap @ A - (Ap @ A).T)])
print("igual a pinv:", np.allclose(Ap, np.linalg.pinv(A)))
for lam in 10.0 ** -np.arange(1, 7):
    e = np.linalg.norm(np.linalg.solve(A.T @ A + lam * np.eye(4), A.T) - Ap, 2)
    print(f"lambda = {lam:.0e}: erro espectral = {e:.4e}, fórmula = {(lam / (s[:r] * (s[:r] ** 2 + lam))).max():.4e}, cota = {lam / s[r - 1] ** 3:.4e}")
print("(A_eps^+)_22:", [f"{np.linalg.pinv(np.diag([1, e]))[1, 1]:.4g}" for e in (1e-1, 1e-4, 1e-7, 1e-9, 0.0)])
