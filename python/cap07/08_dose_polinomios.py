"""Cap. 7: Resposta da aveia ao nitrogênio: condicionamento de bases polinomiais."""
from pathlib import Path

import numpy as np
import pandas as pd

a = pd.read_csv(Path(__file__).resolve().parents[2] / "dados" / "yates_oats.csv")
dose = a["nitro"].to_numpy() * 50.80235 / 0.4046856; y = a["yield"].to_numpy(float); n = len(y)
c = dose - dose.mean()
Q = np.linalg.qr(np.column_stack([np.ones(n), c, c ** 2, c ** 3]))[0]          # base ortonormal
bases = {"monômios": np.vander(dose, 4, increasing=True), "centrados": np.vander(c, 4, increasing=True),
         "padronizados": np.vander(c / dose.std(ddof=1), 4, increasing=True), "ortogonais": Q}
for nome, B in bases.items():
    Rc = np.linalg.inv(B.T @ B); d = np.sqrt(np.diag(Rc)); Rc = Rc / np.outer(d, d)
    print(f"{nome:12s} kappa(X) = {np.linalg.cond(B):.4g}, kappa(X'X) = {np.linalg.cond(B.T @ B):.4g}, "
          f"máx |corr| entre estimativas = {np.abs(Rc[np.triu_indices(4, 1)]).max():.4f}")
