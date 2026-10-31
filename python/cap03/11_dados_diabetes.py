"""Cap. 3 -- Conhecendo os dados de diabetes"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.descritiva import correlacoes, resumo  # noqa: E402

d = pd.read_csv(Path(__file__).resolve().parents[2] / "dados"
                / "diabetes.csv")
resumo(d)
# Correlação de cada covariável com a progressão y
r = d.iloc[:, :10].corrwith(d["y"]).round(3)
t = [f"{k}:{v:.3f}" for k, v in r.items()]
print(" ".join(t[:5])); print(" ".join(t[5:]))
# As medidas séricas s1 a s4 são fortemente correlacionadas
correlacoes(d, ["s1", "s2", "s3", "s4", "s5"])
