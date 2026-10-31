"""Cap. 2 -- Adaptação de posto baixo (LoRA): fatoração de posto e contagem de parâmetros."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
d, r = 64, 4
W0 = rng.normal(size=(d, d))
B, A = rng.normal(size=(d, r)), rng.normal(size=(r, d))
Delta = B @ A
posto = np.linalg.matrix_rank
print(posto(W0), posto(Delta), posto(W0 + Delta))

Q, R = np.linalg.qr(Delta)
B2 = Q[:, :r]                                 # (sem pivotamento basta aqui: colunas genéricas)
A2 = B2.T @ Delta
print(np.abs(B2 @ A2 - Delta).max())

tab = pd.DataFrame([(dd, rr) for dd in (768, 4096) for rr in (4, 8, 16)], columns=["d", "r"])
tab["completo"] = tab.d ** 2
tab["lora"] = 2 * tab.d * tab.r
tab["fracao_pct"] = (100 * tab.lora / tab.completo).round(2)
print(tab)
