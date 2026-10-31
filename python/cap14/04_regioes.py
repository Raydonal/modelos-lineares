"""Cap. 14: Regiões de confiança e intervalos de predição."""
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.gerador import Gerador  # noqa: E402

# Elipse para as duas médias: (2, 5), s2 = 4/3, gl = 3
F23 = stats.f.ppf(0.95, 2, 3); s2 = 4 / 3
print(f"F(2,3) {F23:.3f} c {2 * s2 * F23:.3f}")
print(f"semieixos {np.sqrt(2 * s2 * F23 / 2):.3f}"
      f" {np.sqrt(2 * s2 * F23 / 3):.3f}")
for v in [(3, 3), (0, 0)]:
    print(f"({v[0]}, {v[1]}): forma"
          f" {2 * (v[0] - 2) ** 2 + 3 * (v[1] - 5) ** 2:.1f}")
# Sombra da elipse: Scheffé contra t para cada média
t3 = stats.t.ppf(0.975, 3)
for nj in [2, 3]:
    print(f"n {nj}: t {t3 * np.sqrt(s2 / nj):.3f}"
          f"  sombra {np.sqrt(2 * F23 * s2 / nj):.3f}")
# Três pontos: média e nova observação em x0 = 3
x = np.arange(3.0); y = np.array([1.0, 2, 4])
X = np.column_stack([np.ones(3), x])
A = np.linalg.inv(X.T @ X); b = A @ X.T @ y
s2 = np.sum((y - X @ b) ** 2); x0 = np.array([1.0, 3])
y0 = x0 @ b; h = x0 @ A @ x0; tt = stats.t.ppf(0.975, 1)
print(f"y0 {y0:.4f} h {h:.4f} t {tt:.3f}")
print(f"IC +- {tt * np.sqrt(s2 * h):.3f}"
      f"  IP +- {tt * np.sqrt(s2 * (1 + h)):.3f}")
# Cobertura: regressão com 30 pontos, erros de variância 1
X = np.column_stack([np.ones(30), np.linspace(0, 1, 30)])
A = np.linalg.inv(X.T @ X)
H = A @ X.T; M = np.eye(30) - X @ H; x0 = np.array([1, 1.2])
h = x0 @ A @ x0; tt = stats.t.ppf(0.975, 28)
g = Gerador(1405); N = 20000
erro = {"normal": lambda k: g.normal(k),
        "uniforme": lambda k: np.sqrt(12) * (g.unif(k) - 0.5),
        "exponencial": lambda k: -np.log(g.unif(k)) - 1}
print(f"{'erro':<12} {'cob.IC':>7} {'falhaE':>7} {'falhaD':>7}")
for nm, f in erro.items():
    E = f(30 * N).reshape(N, 30)
    e0 = f(N)
    dif = E @ H.T @ x0                      # ajuste - média
    s = np.sqrt(((E @ M) ** 2).sum(axis=1) / 28)
    ic = np.mean(np.abs(dif) <= tt * s * np.sqrt(h))
    z = (e0 - dif) / (s * np.sqrt(1 + h))   # y0 - ajuste
    print(f"{nm:<12} {ic:7.4f} {np.mean(z < -tt):7.4f}"
          f" {np.mean(z > tt):7.4f}")
