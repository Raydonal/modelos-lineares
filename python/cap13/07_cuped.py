"""Cap. 13: Ajuste por covariável em testes A/B (CUPED) e FWL"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.gerador import Gerador  # noqa: E402


def um(g, n):                            # um experimento simulado
    x = g.normal(n)                           # métrica pré-teste
    t = (g.unif(n) < 0.5).astype(float)       # sorteio do tratamento
    y = 1 + 0.1 * t + 0.8 * x + 0.6 * g.normal(n)
    return x, t, y


def mq(X, v): return np.linalg.lstsq(X, v, rcond=None)[0]
def res(v, x): return v - np.column_stack([np.ones(len(x)), x]) \
    @ mq(np.column_stack([np.ones(len(x)), x]), v)


def ajustado(x, t, y):                        # regressão em (1, t, x)
    return mq(np.column_stack([np.ones(len(x)), t, x]), y)[1]


g = Gerador(2013); x, t, y = um(g, 500)
dif = y[t == 1].mean() - y[t == 0].mean()
th = np.cov(y, x)[0, 1] / x.var(ddof=1)       # CUPED
yc = y - th * (x - x.mean())
cup = yc[t == 1].mean() - yc[t == 0].mean()
ry, rt = res(y, x), res(t, x)
fwl = mq(np.column_stack([np.ones(500), rt]), ry)[1]
print(f"diferença {dif:.4f}, ajustado {ajustado(x, t, y):.4f},"
      f" cuped {cup:.4f}")
print(f"fwl {fwl:.4f}, cor_xy {np.corrcoef(x, y)[0, 1]:.4f}")
# 2000 experimentos: variância de cada estimador do efeito
R = 2000; est = np.empty((R, 2))
for r in range(R):
    x, t, y = um(g, 500)
    est[r] = [y[t == 1].mean() - y[t == 0].mean(), ajustado(x, t, y)]
v = est.var(0, ddof=1)
print(f"médias {est.mean(0).round(4)}, var {v.round(4)}")
print(f"razão {v[1] / v[0]:.4f}, teórica {0.36 / 1.0:.4f}")
