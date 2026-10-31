"""Estatísticas descritivas com a mesma formatação em R e em Python
(R/comum/descritiva.R): uma linha por variável, <= 70 colunas."""
import numpy as np

_CAB = ["média", "dp", "mín", "Q1", "mediana", "Q3", "máx"]


def _fmt(s, w):
    """Casas decimais pela ordem de grandeza da linha (igual no R)."""
    m = max(abs(t) for t in s)
    d = 0 if m >= 1000 else 1 if m >= 100 else 2 if m >= 10 else 3
    return "".join(f"%{w}.{d}f" % t for t in s)


def resumo(d, vars=None):
    vars = list(d.columns) if vars is None else vars
    print(f"{'':<9}{'n':>5}" + "".join(f"{h:>8}" for h in _CAB))
    for v in vars:
        x = d[v].dropna().to_numpy(float)
        q = np.quantile(x, [0, 0.25, 0.5, 0.75, 1])
        s = [x.mean(), x.std(ddof=1), *q]
        print(f"{v[:9]:<9}{len(x):5d}" + _fmt(s, 8))


def correlacoes(d, vars=None):
    vars = list(d.columns) if vars is None else vars
    r = d[vars].corr().to_numpy()
    print(f"{'':<9}" + "".join(f"{v[:7]:>8}" for v in vars))
    for i, v in enumerate(vars):
        print(f"{v[:9]:<9}" + "".join("%8.3f" % t for t in r[i]))


def por_grupo(y, g, rotulo="grupo"):
    """Médias de y por grupos (um fator), com n e desvio-padrão."""
    y = np.asarray(y, float); g = np.asarray(g)
    print(f"{rotulo:<12}{'n':>5}"
          + "".join(f"{h:>9}" for h in ["média", "dp", "mín", "máx"]))
    for k in np.unique(g):
        x = y[(g == k) & ~np.isnan(y)]
        print(f"{str(k)[:12]:<12}{len(x):5d}"
              + _fmt([x.mean(), x.std(ddof=1), x.min(), x.max()], 9))
