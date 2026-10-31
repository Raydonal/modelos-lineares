"""Cap. 5: Conhecendo os dados Seatbelts"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.descritiva import correlacoes, por_grupo, resumo  # noqa

s = pd.read_csv(Path(__file__).resolve().parents[2] / "dados"
                / "seatbelts.csv")
resumo(s, ["motoristas", "km", "preco_gasolina"])
# Antes e depois da lei (fevereiro de 1983)
por_grupo(s["motoristas"], np.where(s["lei"] == 1, "depois", "antes"),
          "lei")
# Perfil sazonal: média de mortos e feridos graves em cada mês
m = s.groupby("mes")["motoristas"].mean().round().astype(int)
t = [f"{k}:{v}" for k, v in m.items()]
print(" ".join(t[:6])); print(" ".join(t[6:]))
s = s.assign(lmot=np.log(s["motoristas"]), lkm=np.log(s["km"]),
             lpreco=np.log(s["preco_gasolina"]))
correlacoes(s, ["lmot", "lkm", "lpreco", "lei"])
