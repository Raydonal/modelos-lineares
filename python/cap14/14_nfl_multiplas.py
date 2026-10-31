"""Cap. 14: Ratings da NFL: 378 comparações ao mesmo tempo."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.gerador import Gerador  # noqa: E402
from comum.multiplas import bh  # noqa: E402

raiz = Path(__file__).resolve().parents[2]
nfl = pd.read_csv(raiz / "dados" / "nfl.csv")
s = nfl[nfl.ano == 1981]
tm = sorted(set(s.mandante) | set(s.visitante)); k = len(tm)
n = len(s)
y = (s.pontos_mandante - s.pontos_visitante).to_numpy(float)
idx = {t: i for i, t in enumerate(tm)}
X = np.zeros((n, k + 1)); X[:, k] = 1
X[np.arange(n), s.mandante.map(idx)] = 1
X[np.arange(n), s.visitante.map(idx)] = -1
G = np.linalg.pinv(X.T @ X); b = G @ X.T @ y
r = np.linalg.matrix_rank(X); nu = n - r
s2 = np.sum((y - X @ b) ** 2) / nu
print(f"jogos {n} posto {r} gl {nu} s {np.sqrt(s2):.3f}")
print(f"mando: {b[k]:.2f} ep {np.sqrt(s2 * G[k, k]):.2f}")
# Teste F global: todos os times iguais (q = k - 1)
sq0 = np.sum((y - y.mean()) ** 2)
q = k - 1; Fg = ((sq0 - s2 * nu) / q) / s2
print(f"F({q},{nu}) {Fg:.3f} p {stats.f.sf(Fg, q, nu):.2e}")
# Todas as diferenças
i1, i2 = np.triu_indices(k, 1); m = len(i1)
L = np.zeros((m, k + 1))
L[np.arange(m), i1] = 1; L[np.arange(m), i2] = -1
est = L @ b; LG = L @ G
se2 = (LG * L).sum(axis=1); tt = est / np.sqrt(s2 * se2)
pv = 2 * stats.t.sf(np.abs(tt), nu)
# Valor crítico conjunto: máximo de |t| (lei simulada)
N = 10000; g = Gerador(1981)
E = g.matriz_normal(N, n)
ss = np.sqrt(stats.chi2.ppf(g.unif(N), nu) / nu)
D = (LG @ X.T) / np.sqrt(se2)[:, None]
mx = np.abs(E @ D.T).max(axis=1) / ss
cmax = np.sort(mx)[int(0.95 * N) - 1]
cs = (stats.t.ppf(0.975, nu), stats.t.ppf(1 - 0.025 / m, nu), cmax,
      np.sqrt(q * stats.f.ppf(0.95, q, nu)))
print(f"m {m} t {cs[0]:.3f} Bonf {cs[1]:.3f} max-t {cs[2]:.3f}"
      f" Scheffé {cs[3]:.3f}")
rb = bh(pv, 0.05)
print(f"pares: sem ajuste {np.sum(np.abs(tt) > cs[0])}"
      f" Bonf {np.sum(np.abs(tt) > cs[1])}"
      f" max-t {np.sum(np.abs(tt) > cs[2])}"
      f" Scheffé {np.sum(np.abs(tt) > cs[3])} BH {rb}")
print(f"contraste máximo: |t| = sqrt(q F) = {np.sqrt(q * Fg):.3f}")
print(f"limiar BH: p <= {0.05 * rb / m:.5f},"
      f" |t| >= {stats.t.ppf(1 - 0.025 * rb / m, nu):.3f}")
print(f"sob ausência de efeito: esperadas {0.05 * m:.1f}"
      " falsas sem ajuste")
o = np.argsort(-b[:k])
print("melhores: " + " ".join(tm[i] for i in o[:3]))
print("piores: " + " ".join(tm[i] for i in o[-3:]))
print(f"rating 1o - 2o {b[o[0]] - b[o[1]]:.2f};"
      f" 1o - 28o {b[o[0]] - b[o[-1]]:.2f}")
for i in (2, 5, 10, 14):
    l = np.zeros(k + 1); l[o[0]] = 1; l[o[i - 1]] = -1
    e = l @ b; v = np.sqrt(s2 * (l @ G @ l))
    print(f"1o - {i:2d}o: {e:.2f} ep {v:.2f} t {e / v:.2f}")
print(f"ep mínimo {np.sqrt(s2 * se2.min()):.2f}"
      f" médio {np.mean(np.sqrt(s2 * se2)):.2f}"
      f" máximo {np.sqrt(s2 * se2.max()):.2f}")
