"""Cap. 14: Somas de quadrados sequenciais e parciais."""
import numpy as np
from scipy import stats

y = np.array([3.0, 1, -2, -2]); n = 4
um = np.ones(4); x1 = np.array([1.0, 1, -1, -1])
x2 = np.array([1.0, 0, 0, -1])


def sqr(*cols):
    Z = np.column_stack(cols)
    e = y - Z @ np.linalg.solve(Z.T @ Z, Z.T @ y)
    return e @ e


t0 = sqr(um); r1 = sqr(um, x1); r2 = sqr(um, x2)
r12 = sqr(um, x1, x2)
print(f"SQ total {t0:.2f} residual {r12:.2f}")
print(f"x1 primeiro: {t0 - r1:.2f} + {r1 - r12:.2f}")
print(f"x2 primeiro: {t0 - r2:.2f} + {r2 - r12:.2f}")
print(f"parciais: x1 {r2 - r12:.2f} x2 {r1 - r12:.2f}"
      f" (soma {r2 - r12 + r1 - r12:.2f})")
# Parcial = t^2 s^2 = b^2 / [(X'X)^-1]_jj
X = np.column_stack([um, x1, x2]); A = np.linalg.inv(X.T @ X)
b = A @ X.T @ y
print(f"b {b[0]:.3f} {b[1]:.3f} {b[2]:.3f}")
print(f"b1^2/A11 {b[1]**2 / A[1, 1]:.2f}"
      f" b2^2/A22 {b[2]**2 / A[2, 2]:.2f}")
# Testes: sequencial e parcial com o mesmo denominador
for s in [16, 4.5, 12.5, 1]:
    print(f"SQ {s:5.2f} F {s / r12:5.2f}"
          f" p {stats.f.sf(s / r12, 1, 1):.3f}")
