"""Cap. 10 -- Teste de Chow: a mesma equação de salários?"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f

raiz = Path(__file__).resolve().parents[2]
w = pd.read_csv(raiz / "dados" / "wage1.csv")
y = np.log(w["wage"].to_numpy()); n = len(y)
m = (w["female"] == 1).to_numpy()
X = np.column_stack([np.ones(n), w[["educ", "exper", "tenure"]]])
k = X.shape[1]
Xu = np.column_stack([X * ~m[:, None], X * m[:, None]])  # separados
Xd = np.column_stack([X, m])                  # só o intercepto muda


def sqr(Z, v=y):
    b = np.linalg.lstsq(Z, v, rcond=None)[0]
    return np.sum((v - Z @ b)**2)


r, u, d = sqr(X), sqr(Xu), sqr(Xd); gl_u = n - 2 * k
F_chow = ((r - u) / k) / (u / gl_u)           # tudo igual?
F_incl = ((d - u) / (k - 1)) / (u / gl_u)     # inclinações iguais?
print(f"SQR: restrito {r:.5f}, dummy {d:.5f}, separado {u:.5f}")
print(f"Chow: F = {F_chow:.5f}, p = {f.sf(F_chow, k, gl_u):.2e};"
      f" inclinações: F = {F_incl:.5f},"
      f" p = {f.sf(F_incl, k - 1, gl_u):.5f}")
# A hipótese de variâncias iguais, que o teste supõe
s2 = [sqr(X[m == g], y[m == g]) / (np.sum(m == g) - k)
      for g in (False, True)]
Fv = s2[0] / s2[1]; g1, g2 = np.sum(~m) - k, np.sum(m) - k
pv = 2 * min(f.cdf(Fv, g1, g2), f.sf(Fv, g1, g2))
print(f"s2 homens {s2[0]:.4f}, mulheres {s2[1]:.4f},"
      f" razão {Fv:.4f}, p = {pv:.4f}")
