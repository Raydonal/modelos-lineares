"""Cap. 14: Mroz: sobreidentificação dos instrumentos."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

raiz = Path(__file__).resolve().parents[2]
m = pd.read_csv(raiz / "dados" / "mroz.csv")
y = m.lwage.to_numpy(); n = len(y); um = np.ones(n)
ed = m.educ.to_numpy(float); pa = m.fatheduc.to_numpy(float)
ma = m.motheduc.to_numpy(float)
ex = m[["exper", "expersq"]].to_numpy(float)


def proj(Z, v):
    return Z @ np.linalg.solve(Z.T @ Z, Z.T @ v)


def fit(X, Z):
    Xh = proj(Z, X); A = np.linalg.inv(Xh.T @ X)
    b = A @ Xh.T @ y; e = y - X @ b
    S = np.sum(proj(Z, e) ** 2) / (e @ e / n)
    gl = Z.shape[1] - X.shape[1]
    ep = np.sqrt(np.diag(A) * (e @ e) / (n - X.shape[1]))
    p = stats.chi2.sf(S, gl) if gl > 0 else np.nan
    return b, ep, S, gl, p


# Primeiro estágio: teste F de pai e mãe na escolaridade
def fse(W):
    r0 = ed - proj(W, ed); Z = np.column_stack([W, pa, ma])
    r1 = ed - proj(Z, ed); gl = n - Z.shape[1]
    Fv = ((r0 @ r0 - r1 @ r1) / 2) / (r1 @ r1 / gl)
    print(f"primeiro estágio: F(2,{gl}) {Fv:.2f}")


X1 = np.column_stack([um, ed]); X2 = np.column_stack([um, ed, ex])
for k in (1, 2):
    X = X1 if k == 1 else X2
    W = um[:, None] if k == 1 else np.column_stack([um, ex])
    print(f"modelo {k}"); fse(W)
    mq = np.linalg.solve(X.T @ X, X.T @ y)[1]
    fp = fit(X, np.column_stack([W, pa]))
    fm = fit(X, np.column_stack([W, ma]))
    f2 = fit(X, np.column_stack([W, pa, ma]))
    print(f"MQ {mq:.4f} pai {fp[0][1]:.4f} ({fp[1][1]:.4f})"
          f" mãe {fm[0][1]:.4f} ({fm[1][1]:.4f})")
    print(f"ambos {f2[0][1]:.4f} ({f2[1][1]:.4f})")
    print(f"Sargan S {f2[2]:.4f} gl {f2[3]} p {f2[4]:.3f}")
