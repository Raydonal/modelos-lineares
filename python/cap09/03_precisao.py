"""Cap. 9 -- Melhor preditor linear e correlações parciais."""
from pathlib import Path

import numpy as np
import pandas as pd

raiz = Path(__file__).resolve().parents[2]
w = pd.read_csv(raiz / "dados" / "wage1.csv")
D = pd.DataFrame({"lsal": np.log(w["wage"]), "educ": w["educ"],
                  "exper": w["exper"], "perman": w["tenure"]})
S = D.cov().to_numpy(); m = D.mean().to_numpy()

# Melhor preditor linear de lsal: b = S_xx^{-1} S_xy (= coef. de MQ)
b = np.linalg.solve(S[1:, 1:], S[1:, 0])
Z = np.column_stack([np.ones(len(D)), D.iloc[:, 1:]])
mq, *_ = np.linalg.lstsq(Z, D["lsal"], rcond=None)
print("momentos:", np.r_[m[0] - b @ m[1:], b].round(6))
print("MQ:      ", mq.round(6))
print("R2 =", S[0, 1:] @ b / S[0, 0])

# Correlações parciais pela precisão K = S^{-1}: -k_ij/sqrt(k_ii k_jj)
K = np.linalg.inv(S)
P = -K / np.sqrt(np.outer(np.diag(K), np.diag(K)))
np.fill_diagonal(P, 1)
print(D.corr().round(3))
print(pd.DataFrame(P, D.columns, D.columns).round(3))
# Conferência: resíduos de lsal e educ após (exper, perman)
Z2 = np.column_stack([np.ones(len(D)), D[["exper", "perman"]]])
res = lambda v: v - Z2 @ np.linalg.lstsq(Z2, v, rcond=None)[0]
print(np.corrcoef(res(D["lsal"]), res(D["educ"]))[0, 1])
