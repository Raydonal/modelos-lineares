"""Cap. 2 -- Posto da matriz de delineamento superparametrizada do experimento de aveia."""
from pathlib import Path

import numpy as np
import pandas as pd

a = pd.read_csv(Path(__file__).resolve().parents[2] / "dados" / "yates_oats.csv")
ind = lambda s: pd.get_dummies(s).to_numpy(float)
X = np.column_stack([np.ones(len(a)), ind(a["block"]), ind(a["nitro"]), ind(a["gen"])])
r = np.linalg.matrix_rank(X)
print(X.shape, "posto:", r, "dim núcleo:", X.shape[1] - r)

N = np.zeros((14, 3)); N[0] = 1
N[1:7, 0] = -1; N[7:11, 1] = -1; N[11:14, 2] = -1
print(np.abs(X @ N).max(), np.linalg.matrix_rank(N))

Xr = np.column_stack([np.ones(len(a)), ind(a["block"])[:, 1:], ind(a["nitro"])[:, 1:],
                      ind(a["gen"])[:, 1:]])                  # contrastes de tratamento
print(Xr.shape[1], np.linalg.matrix_rank(Xr))
y = a["yield"].to_numpy(float)
aj = lambda M: M @ np.linalg.lstsq(M, y, rcond=None)[0]
print(np.abs(aj(X) - aj(Xr)).max())
