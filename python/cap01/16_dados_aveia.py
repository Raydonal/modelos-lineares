"""Cap. 1 -- Conhecendo o experimento de aveia"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.descritiva import por_grupo, resumo  # noqa: E402

a = pd.read_csv(Path(__file__).resolve().parents[2] / "dados"
                / "yates_oats.csv")
resumo(a, ["yield"])
print(pd.crosstab(a["gen"], a["nitro"]))  # 6 parcelas por casela
por_grupo(a["yield"], a["gen"], "variedade")
por_grupo(a["yield"], a["nitro"], "dose")
por_grupo(a["yield"], a["block"], "bloco")
# Médias das caselas variedade x dose
print(a.pivot_table("yield", "gen", "nitro").round(1).to_string())
