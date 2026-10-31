"""Cap. 2: A armadilha das variáveis indicadoras: posto incompleto e identificação."""
from pathlib import Path

import numpy as np
import pandas as pd

d = pd.read_csv(Path(__file__).resolve().parents[2] / "dados" / "wage1.csv")
y = np.log(d["wage"].to_numpy())
nordeste = 1 - d["northcen"] - d["south"] - d["west"]
X = np.column_stack([np.ones(len(d)), nordeste, d["northcen"], d["south"], d["west"]])
print("colunas:", X.shape[1], " posto:", np.linalg.matrix_rank(X))
print(np.abs(X @ np.array([1, -1, -1, -1, -1])).max())     # vetor do núcleo

b1 = np.linalg.lstsq(X[:, [0, 2, 3, 4]], y, rcond=None)[0]   # nordeste como referência
b2 = np.linalg.lstsq(X[:, 1:], y, rcond=None)[0]             # médias por região
print(np.round(b1, 4), np.round(b2, 4), sep="\n")
print(np.abs(X[:, [0, 2, 3, 4]] @ b1 - X[:, 1:] @ b2).max())  # mesmos ajustados

# lstsq com a matriz de posto incompleto devolve UMA das infinitas soluções
b_min = np.linalg.lstsq(X, y, rcond=None)[0]
print(np.round(b_min, 4), np.abs(X @ b_min - X[:, 1:] @ b2).max())
