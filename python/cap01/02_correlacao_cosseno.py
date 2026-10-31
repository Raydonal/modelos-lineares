"""Cap. 1: Correlação amostral como cosseno entre vetores centrados."""
from pathlib import Path

import numpy as np
import pandas as pd

dados = pd.read_csv(Path(__file__).resolve().parents[2] / "dados" / "wage1.csv")
y = np.log(dados["wage"].to_numpy())   # log do salário-hora
x = dados["educ"].to_numpy(float)      # anos de escolaridade


def cosseno(u, v):
    return u @ v / np.sqrt((u @ u) * (v @ v))


r = np.corrcoef(x, y)[0, 1]
print(f"n = {len(y)}")
print(f"cosseno (não centrado) = {cosseno(x, y):.4f}")
print(f"cosseno (centrado)     = {cosseno(x - x.mean(), y - y.mean()):.4f}")
print(f"correlação             = {r:.4f}")
print(f"ângulo                 = {np.degrees(np.arccos(r)):.2f} graus")
