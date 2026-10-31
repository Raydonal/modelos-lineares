"""Cap. 1: Decomposição ortogonal da produção de aveia (Yates, 1935)."""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

aveia = pd.read_csv(Path(__file__).resolve().parents[2] / "dados" / "yates_oats.csv")
y = aveia["yield"].to_numpy(float)
n = len(y)

# Base ortogonal: indicadores das doses de nitrogênio
D = pd.get_dummies(aveia["nitro"]).to_numpy(float)
print(D.T @ D)                                  # matriz diagonal

y_med = np.full(n, y.mean())                    # projeção em span{1}
y_trt = D @ ((D.T @ y) / D.sum(axis=0))         # projeção em span{d_1,...,d_4}

sq_tot = np.sum((y - y_med) ** 2)
sq_trt = np.sum((y_trt - y_med) ** 2)
sq_res = np.sum((y - y_trt) ** 2)
print(f"SQ total = {sq_tot:.2f}, SQ trat. = {sq_trt:.2f}, SQ res. = {sq_res:.2f}")
print(f"Pitágoras: {sq_trt + sq_res - sq_tot:.2e}")
print(f"Ortogonalidade: {(y_trt - y_med) @ (y - y_trt):.2e}")

modelo = smf.ols("Q('yield') ~ C(nitro)", data=aveia).fit()
print(sm.stats.anova_lm(modelo))                # mesmas somas de quadrados
