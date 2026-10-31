"""Cap. 12: Aveia sem a casela Victory/0,6: o que é estimável"""
from pathlib import Path

import numpy as np
import pandas as pd

raiz = Path(__file__).resolve().parents[2]
a = pd.read_csv(raiz / "dados" / "yates_oats.csv")
gv, ga = np.unique(a["gen"], return_inverse=True)   # GoldenRain, ...
dv, da = np.unique(a["nitro"], return_inverse=True)
bv, ba = np.unique(a["block"], return_inverse=True)
y = a["yield"].to_numpy(float)
Bs = np.vstack([np.eye(5), -np.ones(5)])[ba]  # blocos, soma zero


def efeitos(g, d, b):                         # ~ block + gen * dose
    A = np.eye(3)[g][:, 1:]; N = np.eye(4)[d][:, 1:]
    AN = (A[:, :, None] * N[:, None, :]).reshape(len(g), -1)
    return np.column_stack([np.ones(len(g)), Bs[b], A, N, AN])


ok = ~((gv[ga] == "Victory") & (dv[da] == 0.6))
cel = ga * 4 + da
cheias = np.unique(cel[ok])
Xc = np.column_stack([np.eye(12)[cel[ok]][:, cheias], Bs[ba[ok]]])
yb = y[ok]; n = len(yb)
r = np.linalg.matrix_rank(Xc)
res = yb - Xc @ np.linalg.lstsq(Xc, yb, rcond=None)[0]
s2 = res @ res / (n - r)
print(f"n {n}, posto {r}, gl {n - r}, s {np.sqrt(s2):.5f}")
# Modelo de efeitos com interação: posto incompleto
Xe = efeitos(ga[ok], da[ok], ba[ok])
XtX = Xe.T @ Xe; G = np.linalg.pinv(XtX, rcond=1.5e-8)
bh = G @ Xe.T @ yb
P = np.eye(Xe.shape[1]) - G @ XtX
# linha de lambda para cada casela: média das linhas sobre os blocos
Xa = efeitos(ga, da, ba)
M = np.array([Xa[cel == c].mean(0) for c in range(12)])
gi = {g: i for i, g in enumerate(gv)}


def dif(ds):
    v, g = gi["Victory"] * 4, gi["GoldenRain"] * 4
    return np.mean([M[v + d] - M[g + d] for d in ds], axis=0)


L = {"doses_0_a_0.4": dif([0, 1, 2]),
     "todas_as_doses": dif([0, 1, 2, 3]), "so_dose_0.6": dif([3])}
print(f"{'':15}{'desvio':>8}{'estim.':>9}{'ep':>7}")
for nome, l in L.items():
    print(f"{nome:15}{np.max(np.abs(l @ P)):8.2f}{l @ bh:9.3f}"
          f"{np.sqrt(s2 * l @ G @ l):7.3f}")
# Sob aditividade (sem interação), a média sobre as doses é estimável
A = np.eye(3)[ga[ok]][:, 1:]; N = np.eye(4)[da[ok]][:, 1:]
Xa2 = np.column_stack([np.ones(n), np.eye(6)[ba[ok]][:, 1:], A, N])
Ci = np.linalg.inv(Xa2.T @ Xa2); ba2 = Ci @ Xa2.T @ yb
e = yb - Xa2 @ ba2; s2a = e @ e / (n - Xa2.shape[1])
j = 6 + gi["Victory"] - 1
print(f"aditivo sem casela {ba2[j]:.3f}"
      f"  ep {np.sqrt(s2a * Ci[j, j]):.3f}")
# Dados completos, modelo com interação (referência)
Xf = np.column_stack([np.eye(12)[cel], Bs[ba]])
Cf = np.linalg.inv(Xf.T @ Xf); bf = Cf @ Xf.T @ y
e = y - Xf @ bf; s2f = e @ e / (len(y) - 17)
lf = np.zeros(17)
lf[gi["Victory"] * 4:gi["Victory"] * 4 + 4] = 1 / 4
lf[gi["GoldenRain"] * 4:gi["GoldenRain"] * 4 + 4] = -1 / 4
print(f"completo interação {lf @ bf:.3f}"
      f"  ep {np.sqrt(s2f * lf @ Cf @ lf):.3f}")
