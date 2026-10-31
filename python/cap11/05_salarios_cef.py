"""Cap. 11: Salários: regressão e média condicional"""
from pathlib import Path

import numpy as np
import pandas as pd

raiz = Path(__file__).resolve().parents[2]
w = pd.read_csv(raiz / "dados" / "wage1.csv")
y = np.log(w["wage"].to_numpy()); e = w["educ"].to_numpy(float)
ej, j, nj = np.unique(e, return_inverse=True, return_counts=True)
mj = np.bincount(j, y) / nj                   # média condicional
D = np.eye(len(ej))[j]                        # modelo saturado
um = np.ones(len(y))
X1 = np.column_stack([um, e])                 # reta
X2 = np.column_stack([X1, e**2])              # parábola
X3 = np.column_stack([X1, np.maximum(e - 12, 0)])  # quebra em 12


def coef(Z, v=y): return np.linalg.lstsq(Z, v, rcond=None)[0]
def P(Z, v=y): return Z @ coef(Z, v)


# MQ com os indivíduos = MQ ponderado nas médias das caselas
b_ind = coef(X1)
r = np.sqrt(nj)                               # pesos n_j
b_cel = coef(np.column_stack([r, r * ej]), r * mj)
dif = np.max(np.abs(b_ind - b_cel))
print(f"dif {dif:.1e}, b = {b_ind.round(5)}")
# Que fração da variação entre caselas cada forma capta?
entre = np.sum((P(D) - y.mean())**2)
capta = [np.sum((P(Z) - y.mean())**2) / entre for Z in (X1, X2, X3)]
print("reta, parábola, quebrada:", np.round(capta, 3))
b3 = coef(X3)
print(f"inclinações: até 12 {b3[1]:.4f}, depois {b3[1] + b3[2]:.4f}")
b2 = coef(X2)
print("retornos marginais em 8, 12, 16:",
      np.round(b2[1] + 2 * b2[2] * np.array([8, 12, 16]), 4))
