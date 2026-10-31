"""Cap. 1 -- A média móvel como operador linear: núcleo, imagem e sazonalidade."""
from pathlib import Path

import numpy as np
import pandas as pd

gas = pd.read_csv(Path(__file__).resolve().parents[2] / "dados" / "ukgas.csv")
y = np.log(gas["consumo"].to_numpy())
n = len(y)


def media_movel(n, k=4):
    """Matriz (n-k+1) x n da média móvel de k termos."""
    M = np.zeros((n - k + 1, n))
    for t in range(n - k + 1):
        M[t, t:t + k] = 1 / k
    return M


M = media_movel(n)
print(M[:3, :6])

s = np.resize([0.5, -0.2, -0.6, 0.3], n)     # padrão sazonal de soma zero
print(np.abs(M @ s).max())                   # ~ 0
posto = np.linalg.matrix_rank(M)
print(f"dim imagem = {posto}, dim núcleo = {n - posto}")

tendencia = M @ y
desvio = y[1:n - 2] - tendencia
tri = gas["trimestre"].to_numpy()[1:n - 2]
sazonal = np.array([desvio[tri == q].mean() for q in range(1, 5)])
print(np.round(sazonal - sazonal.mean(), 3))
