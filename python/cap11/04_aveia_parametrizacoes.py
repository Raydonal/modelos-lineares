"""Cap. 11 -- Aveia: parametrizações do modelo com interação"""
from pathlib import Path

import numpy as np
import pandas as pd

raiz = Path(__file__).resolve().parents[2]
a = pd.read_csv(raiz / "dados" / "yates_oats.csv")
gv, ga = np.unique(a["gen"], return_inverse=True)
dv, nb = np.unique(a["nitro"], return_inverse=True)
y = a["yield"].to_numpy(float)


def trat(k): return np.eye(k)[:, 1:]
def soma(k): return np.vstack([np.eye(k - 1), -np.ones(k - 1)])
def helm(k):
    C = [[-1] * (j + 1) + [j + 1] + [0] * (k - 2 - j)
         for j in range(k - 1)]
    return np.array(C, float).T


def poli(k):
    Q = np.linalg.qr(np.vander(np.arange(1, k + 1), k, True))[0]
    return Q[:, 1:] * np.sign(Q[-1, 1:])


def matriz(CA, CB, g=ga, d=nb):               # [1, A, B, A:B]
    FA = np.column_stack([np.ones(3), CA])[g]
    FB = np.column_stack([np.ones(4), CB])[d]
    X = FA[:, :, None] * FB[:, None, :]       # ordem de Kronecker
    return X.reshape(len(g), -1)


def ajuste(X, v=y): return np.linalg.lstsq(X, v, rcond=None)[0]


cel = ga * 4 + nb
Xc = np.eye(12)[cel]                          # médias das caselas
mods = [Xc, matriz(trat(3), trat(4)), matriz(soma(3), soma(4)),
        matriz(helm(3), poli(4))]
f = np.column_stack([X @ ajuste(X) for X in mods])
print(f"max dif. ajustados: {np.max(np.abs(f - f[:, [0]])):.1e}")
mu = ajuste(Xc).reshape(3, 4)
print(pd.DataFrame(mu.round(2), index=gv, columns=dv))
bt = ajuste(mods[1]); bs = ajuste(mods[2])    # Marvellous: 4 e 8
print(f"tratamento {bt[4]:.3f}, dif. na dose 0"
      f" {mu[1, 0] - mu[0, 0]:.3f}")
print(f"soma {bs[8]:.3f}, dif. de médias"
      f" {mu[1].mean() - mu.mean():.3f}")
# Uma casela inteira perdida: Victory na dose 0,6
ok = ~((a["gen"] == "Victory") & (a["nitro"] == 0.6)).to_numpy()
cheias = np.unique(cel[ok])
Xt2 = matriz(trat(3), trat(4))[ok]
print(f"médias {len(cheias)}, colunas efeitos {Xt2.shape[1]},"
      f" posto {np.linalg.matrix_rank(Xt2)}")
m2 = dict(zip(cheias, ajuste(np.eye(12)[cel[ok]][:, cheias], y[ok])))
dif = [m2[2 * 4 + d] - m2[0 * 4 + d] for d in range(3)]
print(f"Victory - GoldenRain nas doses 0 a 0,4: {np.mean(dif):.3f}")
