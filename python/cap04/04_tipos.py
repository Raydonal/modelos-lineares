"""Cap. 4: Inversas reflexivas, de mínimos quadrados, de norma mínima e de Moore-Penrose."""
import numpy as np

A = np.array([[1, 2, 0, 1], [2, 4, 1, 1], [3, 6, 1, 2]], dtype=float)


def ginv_sub(M, i, j):
    G = np.zeros(M.T.shape); G[np.ix_(j, i)] = np.linalg.inv(M[np.ix_(i, j)]); return G


G = {"sub": ginv_sub(A, [0, 1], [0, 2]),
     "mq": ginv_sub(A.T @ A, [0, 2], [0, 2]) @ A.T,
     "nm": A.T @ ginv_sub(A @ A.T, [0, 1], [0, 1]),
     "mp": np.linalg.pinv(A)}
ok = lambda M: "sim" if np.abs(M).max() < 1e-10 else "não"
print("      AGA=A GAG=G (AG)'=AG (GA)'=GA")
for k, g in G.items():
    print(f"{k:5s}", *(f"{ok(M):>6s}" for M in (A @ g @ A - A, g @ A @ g - g, A @ g - (A @ g).T, g @ A - (g @ A).T)))

b = np.array([1.0, 0, 0])
for k in ("sub", "mq", "mp"):
    x = G[k] @ b
    print(f"{k:4s} resíduo = {np.linalg.norm(A @ x - b):.4f}  norma = {np.linalg.norm(x):.4f}")
x_ls = np.linalg.lstsq(A, b, rcond=None)[0]     # lstsq devolve a solução de norma mínima
print("lstsq = pinv b:", np.allclose(x_ls, G["mp"] @ b))
for lam in 10.0 ** -np.arange(1, 7):
    print(f"lambda = {lam:.0e}: dif. = {np.abs(np.linalg.solve(A.T @ A + lam * np.eye(4), A.T) - G['mp']).max():.1e}")
