"""Cap. 2 -- Conhecendo os dados de Longley"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.descritiva import correlacoes, resumo  # noqa: E402

l = pd.read_csv(Path(__file__).resolve().parents[2] / "dados"
                / "longley.csv")
v = ["Employed", "GNP.deflator", "GNP", "Unemployed",
     "Armed.Forces", "Population", "Year"]
resumo(l, v)
correlacoes(l, v)
