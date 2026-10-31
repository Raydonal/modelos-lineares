"""Cap. 1 -- Decomposição da SQ de tratamentos em contrastes polinomiais ortogonais."""
from pathlib import Path

import numpy as np
import pandas as pd

aveia = pd.read_csv(Path(__file__).resolve().parents[2] / "dados" / "yates_oats.csv")
medias = aveia.groupby("nitro")["yield"].mean().to_numpy()
r = 18

C = np.array([[-3, -1, 1, 3],      # linear
              [1, -1, -1, 1],      # quadrático
              [-1, 3, -3, 1]]).T   # cúbico
print(C.T @ C)

sq = r * (C.T @ medias) ** 2 / (C ** 2).sum(axis=0)
for nome, v in zip(["linear", "quadrático", "cúbico"], sq):
    print(f"{nome:<11} SQ = {v:9.2f}  fração = {v / sq.sum():.4f}")
print(sq.sum())
