"""Cap. 9: Imputação pela distribuição condicional."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.gerador import Gerador  # noqa: E402

raiz = Path(__file__).resolve().parents[2]
aq = pd.read_csv(raiz / "dados" / "airquality.csv")
V = np.column_stack([np.log(aq["Ozone"]), aq["Solar.R"],
                     aq["Wind"], aq["Temp"]])
cc = ~np.isnan(V).any(1)
print("completas:", cc.sum(), " incompletas:", (~cc).sum())


def condicional(x, mu, S):
    """Condicional dos ausentes (m) dados os observados (o)."""
    m = np.isnan(x); o = ~m
    B = S[np.ix_(m, o)] @ np.linalg.inv(S[np.ix_(o, o)])
    return (m, mu[m] + B @ (x[o] - mu[o]),
            S[np.ix_(m, m)] - B @ S[np.ix_(o, m)])


# Validação: esconde o ozônio de cada linha completa (sem usá-la)
Vc = V[cc]; z90 = norm.ppf(0.95); val = []
for i in range(len(Vc)):
    resto = np.delete(Vc, i, 0)
    mu, S = resto.mean(0), np.cov(resto.T)
    x = Vc[i].copy(); x[0] = np.nan
    _, med, cv = condicional(x, mu, S)
    val.append((Vc[i, 0] - med[0], Vc[i, 0] - mu[0],
                abs(Vc[i, 0] - med[0]) < z90 * np.sqrt(cv[0, 0])))
val = np.array(val)
print("rmse condicional:", np.sqrt(np.mean(val[:, 0]**2)).round(4),
      " rmse média:", np.sqrt(np.mean(val[:, 1]**2)).round(4),
      " cobertura 90%:", val[:, 2].mean().round(4))

# Imputação das 42 linhas incompletas: média condicional e sorteio
mu, S = Vc.mean(0), np.cov(Vc.T); g = Gerador(13)
Vm, Vs = V.copy(), V.copy()
for i in np.where(~cc)[0]:
    m, med, cv = condicional(V[i], mu, S)
    Vm[i, m] = med
    Vs[i, m] = med + np.linalg.cholesky(cv) @ g.normal(m.sum())
print(np.column_stack([Vm[~cc, 0], Vs[~cc, 0]])[:6].round(3))
simples = np.where(np.isnan(V[:, 0]), mu[0], V[:, 0])
print("variâncias: completas", V[cc, 0].var(ddof=1).round(4),
      " média simples", simples.var(ddof=1).round(4),
      " condicional", Vm[:, 0].var(ddof=1).round(4),
      " sorteio", Vs[:, 0].var(ddof=1).round(4))
