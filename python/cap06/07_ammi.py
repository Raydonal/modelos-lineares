"""Cap. 6 -- Interação genótipo x ambiente pela SVD: o modelo AMMI."""
from pathlib import Path

import numpy as np
import pandas as pd

w = pd.read_csv(Path(__file__).resolve().parents[2] / "dados" / "yan_trigo.csv")
Y = w.pivot_table(index="genotipo", columns="ambiente", values="producao")
m = Y.to_numpy().mean(); a = Y.mean(axis=1) - m; b = Y.mean(axis=0) - m
I = Y.sub(a, axis=0).sub(b, axis=1) - m
SQ = {"genótipos": Y.shape[1] * (a ** 2).sum(), "ambientes": Y.shape[0] * (b ** 2).sum(), "interação": (I ** 2).to_numpy().sum()}
tot = ((Y - m) ** 2).to_numpy().sum()
for k, v in SQ.items():
    print(f"{k:10s} SQ = {v:8.3f}  proporção = {v / tot:.3f}")
U, s, Vt = np.linalg.svd(I.to_numpy())
print("valores singulares:", s.round(3))
print("proporção acumulada:", (np.cumsum(s ** 2) / (s ** 2).sum()).round(3))
G = U[:, :2] * np.sqrt(s[:2]); E = Vt[:2].T * np.sqrt(s[:2])
print(pd.DataFrame(E, index=Y.columns, columns=["CP1", "CP2"]).round(2))
I2 = G @ E.T
print(f"Eckart-Young: erro = {((I.to_numpy() - I2) ** 2).sum():.4f}, descartados = {(s[2:] ** 2).sum():.4f}")
