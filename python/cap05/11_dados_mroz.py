"""Cap. 5 -- Conhecendo os dados de Mroz: estatísticas descritivas"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.descritiva import correlacoes, resumo  # noqa: E402

m = pd.read_csv(Path(__file__).resolve().parents[2] / "dados"
                / "mroz.csv")
v = ["lwage", "educ", "exper", "fatheduc", "motheduc"]
resumo(m, v)
correlacoes(m, v)
