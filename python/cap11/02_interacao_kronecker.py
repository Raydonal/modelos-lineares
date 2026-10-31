"""Cap. 11: Interações como produtos de Kronecker linha a linha"""
from pathlib import Path

import numpy as np
import pandas as pd

raiz = Path(__file__).resolve().parents[2]
a = pd.read_csv(raiz / "dados" / "yates_oats.csv")
ga = pd.factorize(a["gen"], sort=True)[0]
nb = pd.factorize(a["nitro"], sort=True)[0]
DA, DB = np.eye(3)[ga], np.eye(4)[nb]         # indicadoras
DAB = np.eye(12)[ga * 4 + nb]                 # caselas, e_a (x) e_b
TA = np.column_stack([np.ones(3), np.eye(3)[:, 1:]])
TB = np.column_stack([np.ones(4), np.eye(4)[:, 1:]])
FA, FB = DA @ TA, DB @ TB                     # [1 C] por observação
X = (FA[:, :, None] * FB[:, None, :]).reshape(len(a), 12)
dif = np.max(np.abs(X - DAB @ np.kron(TA, TB)))
print(f"dif. Kronecker {dif:.1e}, posto {np.linalg.matrix_rank(X)}")


def P(Z):                                     # projeção sobre C(Z)
    return Z @ np.linalg.pinv(Z)


print(f"max |P(X) - P(DAB)| = {np.max(np.abs(P(X) - P(DAB))):.1e}")
# Perder uma casela inteira (Victory, dose 0,6): 6 parcelas
fora = ((a["gen"] == "Victory") & (a["nitro"] == 0.6)).to_numpy()
print(f"colunas 12, posto {np.linalg.matrix_rank(X[~fora])},"
      f" caselas cheias {np.sum(DAB[~fora].sum(0) > 0)}")
