"""Cap. 12 -- Teste A/B/n: quantas unidades dar ao controle?"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.gerador import Gerador  # noqa: E402

k = 4; N = 600                                # 4 variantes + controle


def v(n0):                           # Var(ybar_t - ybar_0)/s2
    nt = (N - n0) / k
    return n0, nt, 1 / nt + 1 / n0, (1 / n0) / (1 / nt + 1 / n0)


n0_otimo = N * np.sqrt(k) / (k + np.sqrt(k))
for nome, n0 in [("igual", N / (k + 1)), ("ótima", n0_otimo)]:
    print(nome, "n0 %.0f nt %.0f var %.4f cor %.4f" % v(n0))
print(f"redução {1 - v(n0_otimo)[2] / v(N / (k + 1))[2]:.3f}")
# Simulação com o gerador portátil: efeitos 0 no controle e
# (0, 0,1, 0,2, 0,3) nas variantes, sigma = 1, 3000 réplicas
g = Gerador(2026); R = 3000; ef = np.array([0, 0, .1, .2, .3])
for n0 in [int(N / (k + 1)), int(n0_otimo)]:
    nt = (N - n0) // k
    grupo = np.repeat(np.arange(5), [n0] + [nt] * k)
    d = np.empty((R, 4))
    for r in range(R):
        z = ef[grupo] + g.normal(N)
        ybar = np.bincount(grupo, z) / np.bincount(grupo)
        d[r] = ybar[1:] - ybar[0]
    print(f"n0 {n0}, média d4 {d[:, 3].mean():.4f}, var média"
          f" {d.var(0, ddof=1).mean():.4f},"
          f" cor_12 {np.corrcoef(d[:, 0], d[:, 1])[0, 1]:.4f}")
