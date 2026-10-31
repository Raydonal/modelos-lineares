"""Cap. 1: Similaridade cosseno e classificação pelo centróide mais próximo."""
from pathlib import Path

import numpy as np
import pandas as pd

dig = pd.read_csv(Path(__file__).resolve().parents[2] / "dados" / "digitos.csv")
X = dig.drop(columns="digito").to_numpy(float)   # 1797 imagens x 64 pixels
rotulo = dig["digito"].to_numpy()

treino, teste = np.arange(1200), np.arange(1200, len(X))   # partição fixa (igual à do R)


def normalizar(M):
    return M / np.linalg.norm(M, axis=1, keepdims=True)   # linhas com norma 1


centr = np.vstack([X[treino][rotulo[treino] == k].mean(axis=0) for k in range(10)])
S = normalizar(centr) @ normalizar(centr).T                # cossenos 10 x 10
print(np.round(S[:4, :4], 3))

cos_teste = normalizar(X[teste]) @ normalizar(centr).T
pred = cos_teste.argmax(axis=1)
print(f"acurácia no teste: {np.mean(pred == rotulo[teste]):.4f}")

# matriz de confusão (linhas: verdadeiro; colunas: previsto)
conf = pd.crosstab(pd.Series(rotulo[teste], name="verdadeiro"), pd.Series(pred, name="previsto"))
print(conf)
