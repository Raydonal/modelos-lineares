"""Cap. 14: Sobreidentificação: a estatística de Sargan."""
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.gerador import Gerador  # noqa: E402


def proj(Z, v):
    return Z @ np.linalg.solve(Z.T @ Z, Z.T @ v)


def sargan(y, X, Z):
    Xh = proj(Z, X)
    b = np.linalg.solve(Xh.T @ X, Xh.T @ y)
    e = y - X @ b
    num = np.sum(proj(Z, e) ** 2)
    return b, e, num, num / (e @ e / len(y))


# Exemplo à mão: Z ortonormal, Z'x = (2, 1), Z'y = (1, 1)
Z = np.column_stack([np.ones(4), [1, 1, -1, -1]]) / 2
x = np.array([1.5, 1.5, 0.5, 0.5])
y = np.array([1.5, 0.5, 0.5, -0.5])
b, e, num, S = sargan(y, x[:, None], Z)
print(f"Z'x {Z[:, 0] @ x:.1f} {Z[:, 1] @ x:.1f}"
      f"  Z'y {Z[:, 0] @ y:.1f} {Z[:, 1] @ y:.1f}")
print(f"b {b[0]:.2f}  e " + " ".join(f"{v:.1f}" for v in e))
print(f"e'PZe {num:.2f}  e'e/n {e @ e / 4:.2f}  S {S:.4f}"
      f" p {stats.chi2.sf(S, 1):.3f}")
# Identidade: e'PZe = eps'(PZ - P_Xh)eps
g = Gerador(77); n = 12
Z = np.column_stack([np.ones(n), g.matriz_normal(n, 3)])
X = np.column_stack([np.ones(n), g.normal(n)])
eps = g.normal(n); beta = np.array([1.0, 2]); y = X @ beta + eps
num = sargan(y, X, Z)[2]; Xh = proj(Z, X)
print(f"identidade: {num:.10f}"
      f" {np.sum(proj(Z, eps)**2) - np.sum(proj(Xh, eps)**2):.10f}")


# Simulação: n = 100, k = 3, p = 2 (um grau de sobreidentificação)
def simula(c1, c2, N=4000, n=100):
    g = Gerador(1958); rej = 0; bb = 0.0
    cr = stats.chi2.ppf(0.95, 1); um = np.ones(n)
    for _ in range(N):
        z1 = g.normal(n); z2 = g.normal(n)
        v = g.normal(n); e = g.normal(n)
        eps = 0.7 * v + np.sqrt(0.51) * e
        x = 0.5 * z1 + 0.5 * z2 + v
        y = x + c1 * z1 + c2 * z2 + eps
        b, _, _, S = sargan(y, np.column_stack([um, x]),
                            np.column_stack([um, z1, z2]))
        rej += S > cr; bb += b[1]
    return rej / N, bb / N


for c1, c2 in [(0, 0), (0, 0.3), (0.3, 0.3)]:
    v = simula(c1, c2)
    print(f"efeito direto ({c1:.1f}, {c2:.1f}): rejeita {v[0]:.3f}"
          f"  média beta {v[1]:.3f}")
