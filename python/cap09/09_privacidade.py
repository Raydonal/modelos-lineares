"""Cap. 9: Privacidade diferencial: o mecanismo gaussiano."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.gerador import Gerador  # noqa: E402

raiz = Path(__file__).resolve().parents[2]
db = pd.read_csv(raiz / "dados" / "diabetes.csv")
X = db.iloc[:, :10].to_numpy(); n, p = X.shape
X = (X - X.min(0)) / (X.max(0) - X.min(0))  # em [0, 1]
Delta = np.sqrt(p)                     # sensibilidade da soma em l2


def sigma_suf(eps, delta):
    """Proposição do texto: Phi((eta - eps)/sqrt(2 eta)) = delta."""
    z = norm.ppf(delta)
    u = (np.sqrt(2) * z + np.sqrt(2 * z**2 + 4 * eps)) / 2
    return 1 / (np.sqrt(2) * u)        # em unidades de Delta


sigma_cla = lambda eps, delta: np.sqrt(2 * np.log(1.25 / delta)) / eps


def sigma_exato(eps, delta):           # calibração de Balle e Wang
    h = lambda s: norm.cdf(1 / (2 * s) - eps * s) \
        - np.exp(eps) * norm.cdf(-1 / (2 * s) - eps * s) - delta
    return brentq(h, 1e-3, 1e3, xtol=1e-12)


delta = 1e-5; eps = [0.1, 0.5, 1, 2, 4]
print(pd.DataFrame({"suficiente": [sigma_suf(e, delta) for e in eps],
                    "clássico": [sigma_cla(e, delta) for e in eps],
                    "exato": [sigma_exato(e, delta) for e in eps]},
                   index=eps).round(3))

# Duas publicações da soma: com todos e sem o paciente 1 (eps = 1)
sigma = Delta * sigma_suf(1, delta); g = Gerador(19)
s_tudo, s_sem = X.sum(0), X[1:].sum(0)
r_tudo = s_tudo + sigma * g.normal(p)
r_sem = s_sem + sigma * g.normal(p)
print(f"sigma = {sigma:.4f}, erro máx. da média ="
      f" {np.abs(r_tudo - s_tudo).max() / n:.4f}")
print(f"erro exato = {np.abs((s_tudo - s_sem) - X[0]).max():.1e},"
      f" com ruído = {np.abs((r_tudo - r_sem) - X[0]).max():.4f}")
print(np.vstack([X[0], r_tudo - r_sem]).round(2))
