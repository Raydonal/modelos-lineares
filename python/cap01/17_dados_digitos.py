"""Cap. 1 -- Conhecendo os dígitos manuscritos"""
from pathlib import Path

import numpy as np
import pandas as pd

d = pd.read_csv(Path(__file__).resolve().parents[2] / "dados"
                / "digitos.csv")
X = d.drop(columns="digito").to_numpy(float)
print(X.shape)                           # 1797 imagens, 64 pixels
c = d["digito"].value_counts().sort_index()
print(" ".join(f"{k}:{v}" for k, v in c.items()))
print(f"média por pixel {X.mean():.4f}, fração de zeros "
      f"{(X == 0).mean():.4f}")
# Tinta por imagem (soma das intensidades) em cada dígito
tinta = X.sum(axis=1)
m = pd.Series(tinta).groupby(d["digito"]).mean().round().astype(int)
print(" ".join(f"{k}:{v}" for k, v in m.items()))
# Pixels que nunca acendem e pixels mais variáveis
print("sempre zero:", int((X.max(axis=0) == 0).sum()))
v = X.var(axis=0, ddof=1)
print(list(d.columns[1:][np.argsort(-v, kind="stable")[:5]]))
