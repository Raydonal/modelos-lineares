"""Cap. 14: Testes na fronteira do espaço paramétrico."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.gerador import Gerador  # noqa: E402

raiz = Path(__file__).resolve().parents[2]
# (a) Parâmetro com sinal conhecido: t = 2, nu = 20
tt = 2; nu = 20
print(f"bilateral {2 * stats.t.sf(tt, nu):.4f}"
      f" unilateral {stats.t.sf(tt, nu):.4f}")
print(f"mistura com F {0.5 * stats.f.sf(tt**2, 1, nu):.4f}"
      f" mistura com chi2 {0.5 * stats.chi2.sf(tt**2, 1):.4f}")


# (b) Componente de variância: a grupos de n; LR por ML
# O quantil c95 sai com um decimal: é o que o erro de Monte Carlo
# permite com N = 200000 (erro-padrão de cerca de 0,02)
def rv(sqb, sqd, a, n):
    Fv = (sqb / (a - 1)) / (sqd / (a * (n - 1)))
    sqt = sqb + sqd
    v = (a * n * np.log(sqt / (a * n))
         - a * (n - 1) * np.log(sqd / (a * (n - 1)))
         - a * np.log(sqb / a))
    return np.where(Fv <= a / (a - 1), 0.0, v)


print(f"{'a':>3} {'n':>3} {'P(RV=0)':>8} {'t(chi2)':>8}"
      f" {'t(mist)':>8} {'c95':>8}")
N = 200000
for a, n in [(6, 3), (10, 4), (30, 3), (200, 3)]:
    g = Gerador(1987)
    sqb = stats.chi2.ppf(g.unif(N), a - 1)
    sqd = stats.chi2.ppf(g.unif(N), a * (n - 1))
    L = rv(sqb, sqd, a, n)
    print(f"{a:3d} {n:3d}"
          f" {stats.f.cdf(a / (a - 1), a - 1, a * (n - 1)):8.4f}"
          f" {np.mean(L > stats.chi2.ppf(0.95, 1)):8.4f}"
          f" {np.mean(L > stats.chi2.ppf(0.90, 1)):8.4f}"
          f" {np.sort(L)[int(0.95 * N) - 1]:8.1f}")
# (c) Alcance da correlação na aveia: lei exata de RV por simulação.
# RV não depende de beta nem de sigma: basta simular ruído branco.
av = pd.read_csv(raiz / "dados" / "yates_oats.csv")
y = av["yield"].to_numpy(float); n = len(y); link = 0.2012
blocos = sorted(av.block.unique()); doses = sorted(av.nitro.unique())
gens = sorted(av.gen.unique())
X = np.column_stack(
    [np.ones(n)]
    + [(av.block == b).to_numpy(float) for b in blocos[1:]]
    + [(av.nitro == d).to_numpy(float) for d in doses[1:]]
    + [(av.gen == h).to_numpy(float) for h in gens[1:]])
cx = (av.col.to_numpy() - 1) * 44 * link
cy = (av.row.to_numpy() - 1) * 28.4 * link
D = np.hypot(cx[:, None] - cx, cy[:, None] - cy)
fases = [0, 1, 2, 3, 4, 6, 8, 11, 15, 20, 25, 29, 35, 45, 60]
pre = []
for f in fases:
    R = np.eye(n) if f == 0 else np.exp(-D / f)
    Ri = np.linalg.inv(R); RiX = Ri @ X
    A = Ri - RiX @ np.linalg.solve(X.T @ RiX, RiX.T)
    pre.append(((A + A.T) / 2, np.linalg.slogdet(R)[1]))


def perf(Y):
    return np.column_stack(
        [-n / 2 * np.log(((Y @ A) * Y).sum(axis=1) / n) - ld / 2
         for A, ld in pre])


obs = perf(y[None, :])[0]
rvobs = 2 * (obs.max() - obs[0])
N = 20000; g = Gerador(2014)
Z = g.matriz_normal(N, n)
L = perf(Z); rv0 = 2 * (L.max(axis=1) - L[:, 0])
print(f"RV obs {rvobs:.3f} em phi = {fases[int(obs.argmax())]}")
print(f"P(RV<0.01) {np.mean(rv0 < 0.01):.4f}"
      f" c95 {np.sort(rv0)[int(0.95 * N) - 1]:.3f}"
      f" max {rv0.max():.3f}"
      f" p sim {(1 + np.sum(rv0 >= rvobs)) / (N + 1):.5f}")
print(f"p chi2 {stats.chi2.sf(rvobs, 1):.5f}"
      f" p mistura {0.5 * stats.chi2.sf(rvobs, 1):.5f}")
# Por que tanta massa em zero: os resíduos de MQ de parcelas vizinhas
# já são negativamente correlacionados (blocos no modelo)
W = ((D > 0) & (D < 9)) * 1.0
M = np.eye(n) - X @ np.linalg.solve(X.T @ X, X.T)
e = M @ y
print(f"correlação de vizinhos em média sob H0"
      f" {n * np.sum(M * W) / ((n - X.shape[1]) * W.sum()):.3f}"
      f" (obs. {n * (e @ W @ e) / (W.sum() * (e @ e)):.3f})")
