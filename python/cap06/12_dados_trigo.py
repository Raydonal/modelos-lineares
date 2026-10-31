"""Cap. 6: Conhecendo o ensaio de trigo: 18 cultivares, 9 locais"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.descritiva import por_grupo, resumo  # noqa: E402

d = pd.read_csv(Path(__file__).resolve().parents[2] / "dados"
                / "yan_trigo.csv")
resumo(d, ["producao"])
por_grupo(d["producao"], d["ambiente"], "ambiente")
Y = d.pivot_table("producao", "genotipo", "ambiente")
g = Y.mean(axis=1).sort_values(ascending=False, kind="stable")
print(" ".join(f"{k}:{v:.2f}" for k, v in
               pd.concat([g.head(3), g.tail(3)]).items()))
# Quanto da variação é de ambiente, de genótipo e de interação?
y = Y.to_numpy(); m = y.mean()
A = y.mean(0) - m; G = y.mean(1) - m
I = y - m - G[:, None] - A[None, :]
sq = np.array([18 * (A**2).sum(), 9 * (G**2).sum(), (I**2).sum()])
for k, v in zip(["ambiente", "genotipo", "interacao"],
                100 * sq / sq.sum()):
    print(f"{k} {v:.1f}%", end="  ")
print()
# A melhor cultivar muda de ambiente para ambiente
print(list(Y.index[y.argmax(axis=0)]))
