"""Cap. 1 -- Retorno da educação: regressão simples como projeção ortogonal."""
from pathlib import Path

import numpy as np
import pandas as pd

dados = pd.read_csv(Path(__file__).resolve().parents[2] / "dados" / "wage1.csv")
y = np.log(dados["wage"].to_numpy())
x = dados["educ"].to_numpy(float)

u1 = np.ones_like(x)                    # base ortogonal de span{1, x}
u2 = x - x.mean()
proj = lambda v, u: (v @ u) / (u @ u) * u
y_hat = proj(y, u1) + proj(y, u2)

b1 = (y @ u2) / (u2 @ u2)
b0 = y.mean() - b1 * x.mean()
print(f"b0 = {b0:.4f}, b1 = {b1:.4f}, retorno = {100 * (np.exp(b1) - 1):.2f}% ao ano")

sq_tot = np.sum((y - y.mean()) ** 2)
sq_reg = np.sum((y_hat - y.mean()) ** 2)
print(f"R2 = {sq_reg / sq_tot:.4f}, cor^2 = {np.corrcoef(x, y)[0, 1] ** 2:.4f}")
print(np.polyfit(x, y, 1)[::-1])        # conferência: (b0, b1)
