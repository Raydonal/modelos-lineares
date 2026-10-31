"""Cap. 13 -- Parcelas subdivididas: MQ = MQG, os erros-padrão não"""
from pathlib import Path

import numpy as np
import pandas as pd

raiz = Path(__file__).resolve().parents[2]
a = pd.read_csv(raiz / "dados" / "yates_oats.csv")
a = a.sort_values(["gen", "block", "nitro"]).reset_index(drop=True)
gv, ga = np.unique(a.gen, return_inverse=True)
dv, da = np.unique(a.nitro, return_inverse=True)
bv, ba = np.unique(a.block, return_inverse=True)
y = a["yield"].to_numpy(float); n = len(y)
G = np.eye(3)[ga][:, 1:]; D = np.eye(4)[da][:, 1:]
GD = (G[:, :, None] * D[:, None, :]).reshape(n, -1)
X = np.column_stack([np.ones(n), np.eye(6)[ba][:, 1:], G, D, GD])
A = np.linalg.inv(X.T @ X)
parc = ga * 6 + ba                            # 18 parcelas


def sqres(M):
    r = y - M @ np.linalg.lstsq(M, y, rcond=None)[0]
    return r @ r


Xp = np.column_stack([np.eye(18)[parc], D, GD])  # parcelas como fator
qm_sub = sqres(Xp) / 45                       # E = s2
qm_par = (sqres(X) - sqres(Xp)) / 10          # E = s2 + 4 sp2
s2 = qm_sub; sp2 = (qm_par - qm_sub) / 4
print(f"QM parcelas {qm_par:.2f}, QM subparcelas {qm_sub:.2f}")
print(f"sigma2 {s2:.2f}, sigma2_p {sp2:.2f}")
V = np.kron(np.eye(18), s2 * np.eye(4) + sp2)  # Cov(y) estimada
bmq = A @ X.T @ y
bg = np.linalg.solve(X.T @ np.linalg.solve(V, X),
                     X.T @ np.linalg.solve(V, y))
incl = np.max(np.abs(V @ X - X @ A @ X.T @ V @ X))
print(f"max dif MQ-MQG {np.max(np.abs(bmq - bg)):.1e},"
      f" inclusão {incl:.1e}")
# Erros-padrão: ingênuo (s2 único) e correto (sanduíche com V)
s2u = sqres(X) / (n - X.shape[1])


def med(g, d=None):
    ok = (gv[ga] == g) & (True if d is None else (dv[da] == d))
    return X[ok].mean(0)


Lv = med("Marvellous") - med("GoldenRain")
Ld = sum(med(g, 0.6) - med(g, 0.0) for g in gv) / 3
C = A @ X.T @ V @ X @ A
for nome, l in [("variedade", Lv), ("dose", Ld)]:
    print(f"{nome:9} {l @ bmq:7.3f}  ep ingênuo"
          f" {np.sqrt(s2u * l @ A @ l):.3f}  ep correto"
          f" {np.sqrt(l @ C @ l):.3f}")
