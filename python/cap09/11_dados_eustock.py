"""Cap. 9: Conhecendo os índices europeus: retornos diários"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.descritiva import correlacoes, resumo  # noqa: E402

eu = pd.read_csv(Path(__file__).resolve().parents[2] / "dados"
                 / "eustock.csv")
r = 100 * np.log(eu.iloc[:, 1:]).diff().dropna()     # em %
resumo(r)
correlacoes(r)
# Caudas pesadas: curtose (3 na normal) e dias além de 4 dp
z = r - r.mean()
curt = (z**4).mean() / (z**2).mean() ** 2
alem4 = (z.abs() > 4 * r.std()).sum()
print(pd.DataFrame({"curtose": curt, "dias_alem_4dp": alem4})
      .T.round(2).to_string())
