"""Cap. 10 -- Parcelas subdivididas: dois estratos de erro (aveia)."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f, ncf

raiz = Path(__file__).resolve().parents[2]
av = pd.read_csv(raiz / "dados" / "yates_oats.csv")
y = av["yield"].to_numpy(float); n = len(y)


def proj(fator):
    D = pd.get_dummies(fator).to_numpy(float)
    return D @ np.linalg.inv(D.T @ D) @ D.T


J = np.full((n, n), 1 / n)
PB, PV, PN = proj(av["block"]), proj(av["gen"]), proj(av["nitro"])
PW = proj(av["block"] + av["gen"])           # parcelas principais
PNV = proj(av["nitro"].astype(str) + av["gen"])
A = {"bloco": PB - J, "variedade": PV - J,
     "erro_a": PW - PB - PV + J,             # erro da parcela
     "nitrogenio": PN - J, "NxV": PNV - PN - PV + J,
     "erro_b": np.eye(n) - PW - PNV + PV}    # erro da subparcela
# Ortogonalidade e a estrutura de Sigma = s2 I + sp2 D_W D_W'
DD = 4 * PW                                  # D_W D_W' = 4 P_W
M = list(A.values())
ort = max(np.abs(M[i] @ M[j]).max() for i in range(6)
          for j in range(6) if i != j)
print("max|AiAj| =", ort, " estratos:",
      [abs(round(float(np.trace(Ai @ DD) / np.trace(Ai)), 10))
       for Ai in M])
gl = np.array([round(np.trace(Ai)) for Ai in M])
SQ = np.array([y @ Ai @ y for Ai in M]); QM = SQ / gl
F = np.r_[QM[:2] / QM[2], np.nan, QM[3:5] / QM[5], np.nan]
den = np.array([gl[2], gl[2], 1, gl[5], gl[5], 1])
tab = pd.DataFrame({"gl": gl, "SQ": SQ.round(1), "QM": QM.round(1),
                    "F": F.round(2),
                    "p": f.sf(F, gl, den)}, index=list(A))
print(tab.to_string(formatters={"p": lambda v: f"{v:.3g}"}))
s2 = QM[5]; sp2 = (QM[2] - s2) / 4
print(f"sigma2 = {s2:.4f}, sigma2_parcela = {sp2:.4f}")

# Poder para uma diferença de 10 entre o melhor e o pior nível
tv = np.array([-5, 0, 5.0]); tn = np.array([-5, -5 / 3, 5 / 3, 5])
print("blocos  variedade  nitrogênio")
for b in (3, 6, 9, 12):
    lv = 4 * b * np.sum(tv**2) / (s2 + 4 * sp2)
    ln = 3 * b * np.sum(tn**2) / s2
    pv = ncf.sf(f.ppf(0.95, 2, 2 * (b - 1)), 2, 2 * (b - 1), lv)
    pn = ncf.sf(f.ppf(0.95, 3, 9 * (b - 1)), 3, 9 * (b - 1), ln)
    print(f"{b:6d} {pv:10.3f} {pn:11.3f}")
