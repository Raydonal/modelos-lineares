"""Cap. 1 -- Conhecendo os dados wage1: estatísticas descritivas"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.descritiva import correlacoes, por_grupo, resumo  # noqa

w = pd.read_csv(Path(__file__).resolve().parents[2] / "dados"
                / "wage1.csv")
w["lwage"] = np.log(w["wage"])          # log do salário-hora
resumo(w, ["wage", "lwage", "educ", "exper", "tenure", "female"])
correlacoes(w, ["lwage", "educ", "exper", "tenure"])
# Log do salário por faixa de escolaridade
faixa = pd.cut(w["educ"], [-1, 11, 12, 15, 18],
               labels=["a:0-11", "b:12", "c:13-15", "d:16-18"])
por_grupo(w["lwage"], faixa.astype(str), "escolaridade")
