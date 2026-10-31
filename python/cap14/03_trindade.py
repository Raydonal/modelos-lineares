"""Cap. 14: Razão de verossimilhanças, Wald e escore."""
import numpy as np
from scipy import stats


def est(Fv, q, n, r):
    x = q * Fv / (n - r)
    return n * x, n * np.log1p(x), n * x / (1 + x)


# Os dois grupos: F = 8,1 com n = 5, r = 2, q = 1
s = est(8.1, 1, 5, 2)
print(f"W {s[0]:.3f} RV {s[1]:.3f} E {s[2]:.3f}")
ps = [stats.chi2.sf(v, 1) for v in s]
print(f"p W {ps[0]:.4f} RV {ps[1]:.4f} E {ps[2]:.4f}"
      f" F {stats.f.sf(8.1, 1, 3):.4f}")
# Conta direta a partir dos resíduos
y = np.array([1.0, 3, 4, 6, 5])
e1 = y - np.array([2.0, 2, 5, 5, 5]); e0 = y - y.mean()
s1 = np.sum(e1**2) / 5; s0 = np.sum(e0**2) / 5
X = np.column_stack([[1, 1, 0, 0, 0], [0, 0, 1, 1, 1]]).astype(float)
P = X @ np.linalg.solve(X.T @ X, X.T)
print(f"s1 {s1:.3f} s0 {s0:.3f}"
      f" W {(e0 @ e0 - e1 @ e1) / s1:.3f}"
      f" RV {5 * np.log(s0 / s1):.3f} E {e0 @ P @ e0 / s0:.3f}")


# Tamanho real a 5% (calibração qui-quadrado): exato, pois cada
# estatística é função crescente de F
def tam(q, n, r):
    cc = stats.chi2.ppf(0.95, q); nu = n - r
    w = stats.f.sf(cc * nu / (n * q), q, nu)
    rv = stats.f.sf(np.expm1(cc / n) * nu / q, q, nu)
    e = stats.f.sf(cc / (1 - cc / n) * nu / (n * q), q, nu) \
        if cc < n else 0.0
    return w, rv, e


print(f"{'n':>5} {'r':>5} {'W':>7} {'RV':>7} {'E':>7}")
for n in [10, 20, 50, 100, 500]:
    t = tam(3, n, 5)
    print(f"{n:5d} {5:5d} {t[0]:7.3f} {t[1]:7.3f} {t[2]:7.3f}")
