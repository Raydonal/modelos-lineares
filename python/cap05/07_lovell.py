"""Cap. 5: Ajuste sazonal e regressão: o teorema de Lovell (dados Seatbelts)."""
from pathlib import Path

import numpy as np
import pandas as pd

s = pd.read_csv(Path(__file__).resolve().parents[2] / "dados" / "seatbelts.csv"); n = len(s)
y = np.log(s["motoristas"].to_numpy())
X = np.column_stack([np.log(s["km"]), np.log(s["preco_gasolina"]), s["lei"]])
D = pd.get_dummies(s["mes"]).to_numpy(float)
MS = np.eye(n) - D @ np.linalg.solve(D.T @ D, D.T)
mq = lambda A, b: np.linalg.lstsq(A, b, rcond=None)[0]
um = np.ones((n, 1))
tab = {"completo": mq(np.column_stack([X, D]), y)[:3],
       "ambos": mq(MS @ X, MS @ y),
       "só y": mq(np.column_stack([um, X]), MS @ y)[1:],
       "nenhum": mq(np.column_stack([um, X]), y)[1:]}
print(pd.DataFrame(tab, index=["km", "preço", "lei"]).T.round(4))
e1 = y - np.column_stack([X, D]) @ mq(np.column_stack([X, D]), y)
e2 = MS @ y - MS @ X @ tab["ambos"]
print("resíduos iguais:", np.abs(e1 - e2).max())
print(f"s correto = {np.sqrt(e1 @ e1 / (n - 15)):.5f},  s ingênuo = {np.sqrt(e2 @ e2 / (n - 3)):.5f}")
