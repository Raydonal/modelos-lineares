"""Cap. 9: Inicialização de redes: variância das ativações."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.gerador import Gerador  # noqa: E402

g = Gerador(17); n, L, B = 128, 30, 16
X0 = g.matriz_normal(n, B)                   # 16 entradas N(0, I)
G = [g.matriz_normal(n, n) for _ in range(L)]
relu = lambda z: np.maximum(z, 0)
redes = {"linear_s2_1": (lambda z: z, 1), "relu_s2_1": (relu, 1),
         "relu_s2_2_he": (relu, 2), "tanh_s2_1": (np.tanh, 1)}
camadas = [0, 1, 2, 5, 10, 20, 30]
q = {}
for nome, (f, s2) in redes.items():
    X = X0; qs = [np.mean(X**2)]
    for l in range(L):
        X = f(np.sqrt(s2 / n) * G[l] @ X)
        qs.append(np.mean(X**2))
    q[nome] = [qs[c] for c in camadas] + [min(qs[1:]), max(qs[1:])]
with pd.option_context("display.float_format", "{:.3g}".format):
    print(pd.DataFrame(q, index=camadas + ["min_1a30", "max_1a30"]))
prev = np.mean(X0**2) / 2.0 ** np.array(camadas)   # ReLU, s2 = 1
print(" ".join(f"{v:.3g}" for v in prev))
