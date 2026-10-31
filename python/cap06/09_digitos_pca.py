"""Cap. 6 -- Componentes principais dos dígitos manuscritos e o teorema de Eckart-Young."""
from pathlib import Path

import numpy as np
import pandas as pd

dig = pd.read_csv(Path(__file__).resolve().parents[2] / "dados" / "digitos.csv")
X = dig.drop(columns="digito").to_numpy(float); y = dig["digito"].to_numpy()
Xc = X - X.mean(0)
U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
ac = np.cumsum(s ** 2) / (s ** 2).sum()
print("variância acumulada (k = 1, 2, 5, 10, 20, 30, 40):", ac[[0, 1, 4, 9, 19, 29, 39]].round(3))
print("k para 80, 90, 95, 99%:", [int(np.argmax(ac >= q)) + 1 for q in (0.8, 0.9, 0.95, 0.99)])
for k in (2, 10, 20, 40):
    erro = ((Xc - U[:, :k] * s[:k] @ Vt[:k]) ** 2).sum()
    print(f"k = {k:2d}: erro = {erro:.1f}, descartados = {(s[k:] ** 2).sum():.1f}")
tr, te = np.arange(1000), np.arange(1000, len(y))
for k in (2, 5, 10, 20, 40, 64):
    Z = Xc @ Vt[:k].T
    d2 = (Z[te] ** 2).sum(1)[:, None] + (Z[tr] ** 2).sum(1)[None, :] - 2 * Z[te] @ Z[tr].T
    print(f"k = {k:2d}: acurácia 1-NN = {np.mean(y[tr][d2.argmin(1)] == y[te]):.4f}")
