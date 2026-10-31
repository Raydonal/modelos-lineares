"""Cap. 5: Conhecendo os dados mcycle: estatísticas descritivas"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.descritiva import por_grupo, resumo  # noqa: E402

d = pd.read_csv(Path(__file__).resolve().parents[2] / "dados"
                / "mcycle.csv")
resumo(d)
print("instantes distintos", d["times"].nunique(),
      " repetidos", int(d["times"].duplicated().sum()))
# A dispersão muda ao longo do tempo (heterocedasticidade)
fase = pd.cut(d["times"], [0, 14, 25, 40, 60],
              labels=["a:antes", "b:queda", "c:oscila", "d:fim"])
por_grupo(d["accel"], fase.astype(str), "fase (ms)")
