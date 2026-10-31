"""Cap. 14: Benjamini-Hochberg: FDR em simulação."""
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.gerador import Gerador  # noqa: E402
from comum.multiplas import bh  # noqa: E402

m = 50; N = 4000; al = 0.10; delta = 3
g = Gerador(2026)


def sim(m0, rho):
    Z = g.matriz_normal(N, m); u = g.normal(N)
    W = np.sqrt(rho) * u[:, None] + np.sqrt(1 - rho) * Z
    W[:, m0:] += delta
    P = 2 * stats.norm.cdf(-np.abs(W))
    fdp = np.zeros(N); fwb = np.zeros(N)
    pdb = np.zeros(N); pwb = np.zeros(N)
    for i in range(N):
        p = P[i]
        R = bh(p, al)
        rej = p <= al * R / m if R > 0 else np.zeros(m, bool)
        V = rej[:m0].sum(); S = R - V
        fdp[i] = V / R if R > 0 else 0
        bo = p <= al / m
        fwb[i] = bo[:m0].any() if m0 > 0 else 0
        if m0 < m:
            pdb[i] = S / (m - m0); pwb[i] = bo[m0:].sum() / (m - m0)
    return fdp.mean(), fwb.mean(), pdb.mean(), pwb.mean()


print(f"{'m0':>4} {'rho':>4} {'FDR':>7} {'lim':>7} {'FWER_B':>7}"
      f" {'pod_BH':>7} {'pod_B':>7}")
for rho in [0, 0.5]:
    for m0 in [50, 45, 25, 5]:
        v = sim(m0, rho)
        print(f"{m0:4d} {rho:4.1f} {v[0]:7.4f} {al * m0 / m:7.4f}"
              f" {v[1]:7.4f} {v[2]:7.4f} {v[3]:7.4f}")
