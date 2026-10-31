"""Cap. 5: Somas de quadrados como projeções no experimento de aveia."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import qr
from scipy.stats import f as f_dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.gerador import Gerador  # noqa: E402

aveia = pd.read_csv(Path(__file__).resolve().parents[2] / "dados" / "yates_oats.csv")


def proj(X):
    """Projeção ortogonal sobre C(X), com X possivelmente de posto incompleto (QR com pivotamento)."""
    Q, R, _ = qr(X, mode="economic", pivoting=True)
    k = (np.abs(np.diag(R)) > 1e-10 * np.abs(R[0, 0])).sum()
    return Q[:, :k] @ Q[:, :k].T


ind = lambda s: pd.get_dummies(s).to_numpy(float)
sq = lambda P, y: float(y @ P @ y)


def pecas(d):
    n = len(d); J = np.full((n, n), 1 / n)
    return {"B": proj(ind(d["block"])) - J, "N": proj(ind(d["nitro"])) - J, "V": proj(ind(d["gen"])) - J, "J": J}


y = aveia["yield"].to_numpy(float); n = len(y); p = pecas(aveia)
print("P_B P_N, P_B P_V, P_N P_V:", [float(np.abs(p[a] @ p[b]).max().round(15)) for a, b in ("BN", "BV", "NV")])
R = np.eye(n) - p["J"] - p["B"] - p["N"] - p["V"]
for nome, M in {"blocos": p["B"], "nitrogênio": p["N"], "variedade": p["V"], "resíduo": R}.items():
    print(f"{nome:11s} gl = {np.trace(M):3.0f}  SQ = {sq(M, y):10.3f}")

g = Gerador(5)                                             # mesmos sorteios que o R
sorteia = lambda: np.sort(np.argsort(g.unif(n))[:6])       # 6 subparcelas ao acaso
perdidas = sorteia()
print("perdidas:", perdidas + 1)


def sq_var(d):
    y2 = d["yield"].to_numpy(float); um = np.ones((len(d), 1))
    B, N, V = ind(d["block"]), ind(d["nitro"]), ind(d["gen"])
    P = lambda *M: proj(np.column_stack((um,) + M))
    Pc = P(B, N, V); gl = len(d) - 11
    sq_res = sq(np.eye(len(d)) - Pc, y2)
    ss = {"V primeiro": sq(P(V) - P(), y2), "V após B": sq(P(B, V) - P(B), y2),
          "V após N": sq(P(N, V) - P(N), y2), "V por último": sq(Pc - P(B, N), y2)}
    pv = [f_dist.sf(ss[k] / 2 / (sq_res / gl), 2, gl) for k in ("V primeiro", "V por último")]
    return ss, pv


ss, pv = sq_var(aveia.drop(index=perdidas))
for k, v in ss.items():
    print(f"{k:13s} SQ = {v:9.3f}")
print(f"valor-p: entrando primeiro {pv[0]:.4f}, por último {pv[1]:.4f}")

res = [sq_var(aveia.drop(index=sorteia())) for _ in range(1000)]
razao = np.array([r[0]["V por último"] / r[0]["V primeiro"] for r in res])
pv = np.array([r[1] for r in res])
print("razão último/primeiro, quantis 0, 5, 50, 95, 100%:", np.quantile(razao, [0, .05, .5, .95, 1]).round(3))
print("fração em que a conclusão a 5% muda:", np.mean((pv[:, 0] < 0.05) != (pv[:, 1] < 0.05)))
