"""Cap. 14: Comparações múltiplas: multiplicadores e cobertura."""
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.gerador import Gerador  # noqa: E402
from comum.multiplas import bh, q_dunnett  # noqa: E402


def mult(a, nu, al=0.05):
    m = a * (a - 1) / 2
    return (stats.t.ppf(1 - al / 2, nu),
            stats.t.ppf(1 - al / (2 * m), nu),
            stats.studentized_range.ppf(1 - al, a, nu) / np.sqrt(2),
            q_dunnett(a - 1, nu),
            np.sqrt((a - 1) * stats.f.ppf(1 - al, a - 1, nu)))


print(f"{'a':>3} {'nu':>3} {'t':>6} {'Bonf':>6} {'Tukey':>6}"
      f" {'Dunn':>6} {'Sch':>6}")
for a in [3, 5, 10, 20]:
    v = mult(a, 4 * a)
    print(f"{a:3d} {4 * a:3d} {v[0]:6.3f} {v[1]:6.3f} {v[2]:6.3f}"
          f" {v[3]:6.3f} {v[4]:6.3f}")
# Exemplo: 3 grupos de 5, s = 2, médias 10, 12, 15
v = mult(3, 12); e3 = 2 * np.sqrt(2 / 5)
print(f"ep {e3:.4f} limites: t {v[0] * e3:.3f} Bonf {v[1] * e3:.3f}"
      f" Tukey {v[2] * e3:.3f} Dunn {v[3] * e3:.3f}"
      f" Sch {v[4] * e3:.3f}")
# Cobertura simultânea das diferenças de pares: a = 5, r = 5
a = 5; r = 5; nu = a * (r - 1); N = 20000
g = Gerador(1414); v = mult(a, nu); ep = np.sqrt(2 / r)
Z = g.matriz_normal(N, a * r).reshape(N, a, r)
md = Z.mean(axis=2)
s = np.sqrt(((Z - md[:, :, None]) ** 2).sum(axis=(1, 2)) / nu)
mx = (md.max(axis=1) - md.min(axis=1)) / (s * ep)
cob = [np.mean(mx <= v[j]) for j in (0, 1, 2, 4)]
print(f"cobertura: t {cob[0]:.4f} Bonf {cob[1]:.4f}"
      f" Tukey {cob[2]:.4f} Scheffé {cob[3]:.4f}")
# Benjamini-Hochberg e Holm à mão
p = np.array([0.001, 0.012, 0.024, 0.035, 0.6]); m = 5; al = 0.05
print("limiares BH " + " ".join(f"{al * k / m:.3f}"
                                for k in range(1, m + 1)))
holm = 0
while holm < m and p[holm] <= al / (m - holm):
    holm += 1
print(f"rejeições: Bonferroni {np.sum(p <= al / m)} Holm {holm}"
      f" BH {bh(p, al)}")
