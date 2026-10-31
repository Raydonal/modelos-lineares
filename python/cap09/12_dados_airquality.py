"""Cap. 9: Conhecendo os dados airquality"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.descritiva import correlacoes, resumo  # noqa: E402

a = pd.read_csv(Path(__file__).resolve().parents[2] / "dados"
                / "airquality.csv")
v = ["Ozone", "Solar.R", "Wind", "Temp"]
resumo(a, v)                                 # n conta os observados
print(a[v].isna().sum().to_dict())
correlacoes(a, v)                            # pares completos
# O ozônio falta mais em junho
print(a["Ozone"].isna().groupby(a["Month"]).mean().round(2).to_dict())
