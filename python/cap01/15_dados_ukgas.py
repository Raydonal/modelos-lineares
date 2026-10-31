"""Cap. 1: Conhecendo os dados UKgas: estatísticas descritivas"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.descritiva import por_grupo, resumo  # noqa: E402

g = pd.read_csv(Path(__file__).resolve().parents[2] / "dados"
                / "ukgas.csv")
g["log_cons"] = np.log(g["consumo"])
resumo(g, ["consumo", "log_cons"])
tri = "tri" + g["trimestre"].astype(str)
por_grupo(g["consumo"], tri, "consumo")
por_grupo(g["log_cons"], tri, "log(consumo)")
# Amplitude sazonal (máx - mín dentro do ano), média por década
amp = g.groupby("ano")[["consumo", "log_cons"]].agg(np.ptp)
amp = amp.groupby((10 * (amp.index // 10)).astype(str) + "s").mean()
amp.columns = ["nivel", "log"]
print(amp.T.round(2).to_string())
