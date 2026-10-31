"""Cap. 4: Conhecendo a temporada de 1981 da NFL"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.descritiva import resumo  # noqa: E402

j = pd.read_csv(Path(__file__).resolve().parents[2] / "dados"
                / "nfl.csv").query("ano == 1981")
j["margem"] = j["pontos_mandante"] - j["pontos_visitante"]
resumo(j, ["pontos_mandante", "pontos_visitante", "margem", "linha"])
print(f"vitórias do mandante {np.mean(j['margem'] > 0):.4f}, empates "
      f"{int((j['margem'] == 0).sum())}, cor(margem, linha) "
      f"{np.corrcoef(j['margem'], j['linha'])[0, 1]:.4f}")
# Saldo de pontos (marcados - sofridos): 3 melhores e 3 piores
saldo = pd.concat([pd.Series(j["margem"].values, j["mandante"]),
                   pd.Series(-j["margem"].values, j["visitante"])])
s = saldo.groupby(level=0).sum().sort_values(ascending=False,
                                              kind="stable")
print(" ".join(f"{k}:{v}" for k, v in
               pd.concat([s.head(3), s.tail(3)]).items()))
