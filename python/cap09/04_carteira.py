"""Cap. 9 -- Carteira de variância mínima e valor em risco."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

raiz = Path(__file__).resolve().parents[2]
eu = pd.read_csv(raiz / "dados" / "eustock.csv")
r = 100 * np.diff(np.log(eu.iloc[:, 1:].to_numpy()), axis=0)
n = len(r); um = np.ones(4)
m = r.mean(0); S = np.cov(r.T)
print(pd.DataFrame([m, np.sqrt(np.diag(S))], ["média", "dp"],
                   eu.columns[1:]).round(4))
print(np.corrcoef(r.T).round(3))

# Variância mínima: w = S^{-1} 1 / (1' S^{-1} 1)
Si1 = np.linalg.solve(S, um); w = Si1 / Si1.sum()
print("pesos:", w.round(4), " dp:", np.sqrt(1 / Si1.sum()).round(4),
      " dp pesos iguais:", np.sqrt(um @ S @ um / 16).round(4))

# Fronteira: var(alvo) = (a t^2 - 2 b t + c) / (a c - b^2)
a = Si1.sum(); b = np.linalg.solve(S, m).sum()
c = m @ np.linalg.solve(S, m)
fronteira = lambda t: np.sqrt((a * t**2 - 2 * b * t + c)
                              / (a * c - b**2))
print("retorno da variância mínima:", round(b / a, 4),
      " dp para alvo 0,1:", fronteira(0.1).round(4))

# Valor em risco a 1% da carteira de variância mínima
rp = r @ w
var_normal = -(rp.mean() + norm.ppf(0.01) * rp.std(ddof=1))
print(f"VaR normal = {var_normal:.4f}, empírico ="
      f" {-np.quantile(rp, 0.01):.4f}, excedências ="
      f" {np.mean(rp < -var_normal):.4f}, curtose ="
      f" {np.mean((rp - rp.mean())**4) / rp.var(ddof=1)**2:.4f}")

# Fora da amostra: estima na 1a metade, avalia na 2a
h = n // 2
w1 = np.linalg.solve(np.cov(r[:h].T), um); w1 /= w1.sum()
print("dp fora da amostra: variância mínima",
      (r[h:] @ w1).std(ddof=1).round(4), " pesos iguais",
      (r[h:] @ um / 4).std(ddof=1).round(4))
