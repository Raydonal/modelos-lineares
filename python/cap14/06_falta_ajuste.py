"""Cap. 14: Falta de ajuste e o teste de aditividade de Tukey."""
import numpy as np
from scipy import stats

x = np.array([0.0, 0, 1, 1, 2, 2]); y = np.array([1.0, 3, 2, 4, 7, 7])
X = np.column_stack([np.ones(6), x])
b = np.linalg.solve(X.T @ X, X.T @ y)
ybar = np.repeat(y.reshape(3, 2).mean(axis=1), 2); yh = X @ b
sqp = np.sum((y - ybar) ** 2); sqf = np.sum((ybar - yh) ** 2)
Fl = (sqf / 1) / (sqp / 3)
print(f"b {b[0]:.2f} {b[1]:.2f} SQ falta {sqf:.2f} SQ puro {sqp:.2f}"
      f" F {Fl:.2f} p {stats.f.sf(Fl, 1, 3):.3f}")


# Tukey (1949): um grau de liberdade para não aditividade
def tukey(Y):
    a = Y.mean(axis=1) - Y.mean(); bb = Y.mean(axis=0) - Y.mean()
    E = Y - np.add.outer(a, bb) - Y.mean()
    D = np.sum(np.outer(a, bb) * Y) ** 2 / (a @ a * (bb @ bb))
    gl = (Y.shape[0] - 1) * (Y.shape[1] - 1) - 1
    Fv = D / ((np.sum(E**2) - D) / gl)
    return a, bb, np.sum(E**2), D, Fv, gl, stats.f.sf(Fv, 1, gl)


Y = np.array([[2.0, 1, 3], [1, 5, 6], [3, 6, 9]])
a, bb, sq, D, Fv, gl, p = tukey(Y)
print(f"a {a[0]:.3f} {a[1]:.3f} {a[2]:.3f}"
      f" b {bb[0]:.3f} {bb[1]:.3f} {bb[2]:.3f}")
print(f"SQRes {sq:.4f} D {D:.4f} F {Fv:.2f} gl {gl} p {p:.3f}")
