"""Cap. 6 -- Componentes principais e quase colinearidade nos dados de Longley."""
from pathlib import Path

import numpy as np
import pandas as pd

L = pd.read_csv(Path(__file__).resolve().parents[2] / "dados" / "longley.csv")
nomes = ["GNP.deflator", "GNP", "Unemployed", "Armed.Forces", "Population", "Year"]
R = np.corrcoef(L[nomes].to_numpy().T)
lam, V = np.linalg.eigh(R); lam, V = lam[::-1], V[:, ::-1]
print("autovalores:", lam.round(5), "\nacumulada:", (lam.cumsum() / 6).round(4))
print(f"kappa = {lam[0] / lam[-1]:.1f}")
print(pd.DataFrame(V[:, [0, 1, 5]], index=nomes, columns=["CP1", "CP2", "CP6"]).round(3))
vif = np.diag(np.linalg.inv(R))
print("VIF:", vif.round(1))
print("VIF espectral:", ((V ** 2) @ (1 / lam)).round(1))
