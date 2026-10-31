"""Cap. 14: Salários: falta de ajuste e somas de quadrados."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

raiz = Path(__file__).resolve().parents[2]
w = pd.read_csv(raiz / "dados" / "wage1.csv")
y = np.log(w.wage.to_numpy()); e = w.educ.to_numpy(float); n = len(y)
um = np.ones(n); ej = np.unique(e); c0 = len(ej)
D = (e[:, None] == ej) * 1.0


def sqr(*cols):
    Z = np.column_stack(cols)
    r = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    return r @ r


pur = sqr(D); glp = n - c0
print(f"caselas {c0} SQ puro {pur:.3f} gl {glp} QM {pur / glp:.4f}")


def lof(nome, *cols):
    s = sqr(*cols); r = np.linalg.matrix_rank(np.column_stack(cols))
    Fv = ((s - pur) / (c0 - r)) / (pur / glp)
    print(f"{nome}: SQ falta {s - pur:.3f} gl {c0 - r} F {Fv:.2f}"
          f" p {stats.f.sf(Fv, c0 - r, glp):.2e}")


q12 = np.maximum(e - 12, 0)
lof("reta", um, e); lof("parábola", um, e, e**2)
lof("quebrada", um, e, q12)
# Dois encaixados dentro do modelo: a quebra é necessária?
s1 = sqr(um, e); s2 = sqr(um, e, q12)
Fq = (s1 - s2) / (s2 / (n - 3))
print(f"quebra: F(1,{n - 3}) {Fq:.2f}"
      f" p {stats.f.sf(Fq, 1, n - 3):.1e}")
# Só com caselas de 8 anos ou mais (poucos dados abaixo)
print(f"caselas abaixo de 8 anos: {np.sum(e < 8)} pessoas")
# Somas sequenciais e parciais: mulher, educ, exper, exper2, tenure
f = w.female.to_numpy(float); x = w.exper.to_numpy(float)
t = w.tenure.to_numpy(float)
nomes = ["mulher", "educ", "exper", "tenure"]
cols = {"mulher": f, "educ": e, "exper": np.column_stack([x, x**2]),
        "tenure": t}
gl = [1, 1, 2, 1]
tudo = np.column_stack([um] + list(cols.values()))
glr = n - tudo.shape[1]
sr = sqr(tudo); qmr = sr / glr
print(f"SQRes {sr:.3f} gl {glr} QM {qmr:.4f}")


def seqs(ordem):
    s0 = sqr(um); Z = um; out = np.zeros(4)
    for nm in ordem:
        Z = np.column_stack([Z, cols[nm]]); s = sqr(Z)
        out[nomes.index(nm)] = s0 - s; s0 = s
    return out


a = seqs(nomes); b = seqs(nomes[::-1])
pc = [sqr(*([um] + [cols[k] for k in nomes if k != nm])) - sr
      for nm in nomes]
print(f"{'termo':<7} {'gl':>2} {'direta':>8} {'inversa':>8}"
      f" {'parcial':>8} {'F parc':>7}")
for j, nm in enumerate(nomes):
    print(f"{nm:<7} {gl[j]:2d} {a[j]:8.3f} {b[j]:8.3f} {pc[j]:8.3f}"
          f" {pc[j] / gl[j] / qmr:7.2f}")
print(f"somas: {a.sum():.3f} {b.sum():.3f} {sum(pc):.3f}"
      f" (total {sqr(um) - sr:.3f})")
# Hiato de gênero: bruto e ajustado
Z = np.column_stack([um, f])
br = np.linalg.solve(Z.T @ Z, Z.T @ y)
A = np.linalg.inv(tudo.T @ tudo); bt = A @ tudo.T @ y
print(f"hiato bruto {br[1]:.3f} ajustado {bt[1]:.3f}"
      f" ep {np.sqrt(qmr * A[1, 1]):.3f}")
# Intervalos em educ = 16 no modelo da reta quebrada
X = np.column_stack([um, e, q12]); A = np.linalg.inv(X.T @ X)
b = A @ X.T @ y; s2 = np.sum((y - X @ b) ** 2) / (n - 3)
x0 = np.array([1.0, 16, 4]); y0 = x0 @ b; h = x0 @ A @ x0
tq = stats.t.ppf(0.975, n - 3)
mw = np.sqrt(3 * stats.f.ppf(0.95, 3, n - 3))
print(f"educ 16: log {y0:.3f} IC +- {tq * np.sqrt(s2 * h):.3f}"
      f" IP +- {tq * np.sqrt(s2 * (1 + h)):.3f}"
      f" WH +- {mw * np.sqrt(s2 * h):.3f}")
print(f"salário: mediana {np.exp(y0):.2f}"
      f" IP ({np.exp(y0 - tq * np.sqrt(s2 * (1 + h))):.2f},"
      f" {np.exp(y0 + tq * np.sqrt(s2 * (1 + h))):.2f})")
