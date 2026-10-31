"""Cap. 2: Posto exato e posto numérico da matriz de imagens de dígitos."""
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.linalg as sla

dig = pd.read_csv(Path(__file__).resolve().parents[2] / "dados" / "digitos.csv")
X = dig.drop(columns="digito").to_numpy(float)
print(list(dig.columns[1:][(X != 0).sum(axis=0) == 0]))     # pixels sempre apagados
Q, R, piv = sla.qr(X, mode="economic", pivoting=True)        # QR com pivotamento
d = np.abs(np.diag(R))
print(np.round(d[[0, 9, 29, 49, 59, 60, 61, 63]], 3))
for tau in (1e-12, 1e-3, 1e-2, 5e-2):              # tolerâncias relativas
    print(tau, (d > tau * d[0]).sum())                # posto numérico
print(np.linalg.matrix_rank(X))
