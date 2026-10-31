"""Cap. 11 -- Atributos aleatórios: uma rede de uma camada congelada"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.gerador import Gerador  # noqa: E402

raiz = Path(__file__).resolve().parents[2]
d = pd.read_csv(raiz / "dados" / "digitos.csv")
dig = d["digito"].to_numpy()
X = np.column_stack([np.ones(len(d)), d.iloc[:, 1:] / 16])
Y = np.eye(10)[dig]                           # 10 respostas (one-hot)
tr, te = slice(0, 1000), slice(1000, None)
g = Gerador(7); M = 2000
W = g.matriz_normal(M, 65, sd=1 / 8)          # pesos congelados
Phi = np.maximum(X @ W.T, 0)                  # ReLU: n x M


def acerto(m):                                # B = A^+ Y (norma mínima)
    A = Phi[:, :m]
    U, s, Vt = np.linalg.svd(A[tr], full_matrices=False)
    r = np.sum(s > np.sqrt(np.finfo(float).eps) * s[0])  # como ginv
    B = Vt[:r].T @ ((U[:, :r].T @ Y[tr]) / s[:r, None])
    prev = np.argmax(A[te] @ B, axis=1)
    return m, r, np.sqrt(np.sum(B**2)), np.mean(prev == dig[te])


ms = [10, 50, 100, 300, 600, 900, 1000, 1100, 1500, 2000]
res = pd.DataFrame([acerto(m) for m in ms],
                   columns=["m", "posto", "norma", "acerto"])
print(res.round(3).to_string(index=False))
# Atributos que nunca se acendem no treino (colunas nulas)
print("colunas nulas:", np.sum(Phi[tr].sum(0) == 0))
