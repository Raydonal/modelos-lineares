"""Cap. 7 -- Instrumentos fracos: a amplificação 1/|r| da projeção oblíqua (simulação)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.gerador import Gerador  # noqa: E402

g = Gerador(77)
n, beta = 428, 0.06


def simula(pi1):
    z, u, v = g.normal(n), g.normal(n), g.normal(n)
    x = pi1 * z + 0.6 * u + 0.8 * v
    y = beta * x + 0.5 * u + 0.3 * g.normal(n)
    zc, xc, yc = z - z.mean(), x - x.mean(), y - y.mean()
    r = zc @ xc / np.sqrt((zc @ zc) * (xc @ xc))
    return zc @ yc / (zc @ xc), xc @ yc / (xc @ xc), r


for pi1 in (0.4, 0.2, 0.1, 0.05):
    s = np.array([simula(pi1) for _ in range(2000)])
    q = np.quantile(s[:, 0], [0.05, 0.25, 0.5, 0.75, 0.95])
    r = np.median(s[:, 2])
    print(f"força {pi1}: r = {r:.3f}, F = {r ** 2 * (n - 2) / (1 - r ** 2):.1f}, VI mediana = {q[2]:.3f}, "
          f"IQR = {q[3] - q[1]:.3f} (IQR*r = {(q[3] - q[1]) * r:.3f}), 5-95% = [{q[0]:.2f}, {q[4]:.2f}], MQ = {np.median(s[:, 1]):.3f}")
