"""Cap. 13: Figuras de dados do capítulo (geradas em figuras/cap13/)."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.estilo_livro import CATEG, COR, DADOS, aplicar_estilo, salvar  # noqa: E402
from comum.gerador import Gerador  # noqa: E402

aplicar_estilo()
CAP = "cap13"
LARG = 5.3
_fmt = FuncFormatter(lambda v, _: f"{v:.10g}".replace(".", ",").replace("-", "−"))


def virgula(ax, eixos="xy"):
    if "x" in eixos:
        ax.xaxis.set_major_formatter(_fmt)
    if "y" in eixos:
        ax.yaxis.set_major_formatter(_fmt)


# 1. Seatbelts: perfil em rho e intervalos para o efeito da lei ----------------------
def fig_seatbelts():
    s = pd.read_csv(DADOS / "seatbelts.csv"); n = len(s)
    y = np.log(s.motoristas.to_numpy(float))
    X = np.column_stack([np.log(s.km), np.log(s.preco_gasolina), s.lei, np.eye(12)[s.mes - 1]])
    p = X.shape[1]

    def pw(M, r):
        M = M.reshape(len(M), -1)
        return np.vstack([np.sqrt(1 - r**2) * M[:1], M[1:] - r * M[:-1]])

    def perfil(r):
        z = pw(y, r).ravel(); W = pw(X, r)
        bz = np.linalg.lstsq(W, z, rcond=None)[0]
        return -n / 2 * np.log(np.sum((z - W @ bz) ** 2) / n) + np.log(1 - r**2) / 2

    grade = np.round(np.arange(0, 0.9505, 0.001), 3)
    ll = np.array([perfil(r) for r in grade]); rho = grade[ll.argmax()]
    A = np.linalg.inv(X.T @ X); b = A @ X.T @ y; e = y - X @ b
    i = np.arange(n); V = rho ** np.abs(i[:, None] - i) / (1 - rho**2)
    s2 = e @ e / (n - p); s2V = e @ e / np.trace((np.eye(n) - X @ A @ X.T) @ V)
    ep_s = np.sqrt((s2V * A @ X.T @ V @ X @ A)[2, 2])
    z = pw(y, rho).ravel(); W = pw(X, rho); Ag = np.linalg.inv(W.T @ W); bg = Ag @ W.T @ z
    s2g = np.sum((z - W @ bg) ** 2) / (n - p)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(LARG, 2.3), gridspec_kw={"width_ratios": [1, 1.1]})
    a1.plot(grade, ll - ll.max(), color=CATEG[0])
    a1.axvline(rho, color=COR["destaque"], lw=0.8, ls="--")
    a1.axhline(-1.92, color=COR["cinza"], lw=0.6, ls=":")
    a1.set_ylim(-12, 0.8); a1.set_xlabel(r"$\rho$"); a1.set_ylabel("log-veross. perfilada")
    a1.text(rho + 0.03, -10, rf"$\hat\rho={rho:.3f}".replace(".", "{,}") + "$", fontsize=7, color=COR["destaque"])
    est = [b[2], b[2], bg[2]]; ep = [np.sqrt(s2 * A[2, 2]), ep_s, np.sqrt(s2g * Ag[2, 2])]
    nomes = ["MQ, EP ingênuo", "MQ, EP sob AR(1)", "MQG (AR(1))"]
    for k, (m_, e_, c) in enumerate(zip(est, ep, [COR["cinza"], CATEG[1], CATEG[0]])):
        a2.errorbar(100 * m_, 2 - k, xerr=200 * e_, fmt="o", color=c, ms=4, capsize=2)
    a2.set_yticks([2, 1, 0]); a2.set_yticklabels(nomes, fontsize=7); a2.set_ylim(-0.6, 2.6)
    a2.axvline(0, color=COR["cinza"], lw=0.5)
    a2.set_xlabel("efeito da lei (×100, log), ± 2 EP")
    virgula(a1); virgula(a2, "x")
    salvar(fig, CAP, "seatbelts_ar1")


# 2. CUPED: distribuição dos dois estimadores em 2000 experimentos --------------------
def fig_cuped():
    g = Gerador(2013)

    def um(n):
        x = g.normal(n); t = (g.unif(n) < 0.5).astype(float)
        return x, t, 1 + 0.1 * t + 0.8 * x + 0.6 * g.normal(n)

    um(500)                                   # o mesmo primeiro experimento do script
    est = []
    for _ in range(2000):
        x, t, y = um(500)
        Z = np.column_stack([np.ones(500), t, x])
        est.append([y[t == 1].mean() - y[t == 0].mean(), np.linalg.lstsq(Z, y, rcond=None)[0][1]])
    est = np.array(est)
    fig, ax = plt.subplots(figsize=(0.72 * LARG, 2.4))
    bins = np.linspace(-0.25, 0.45, 50)
    ax.hist(est[:, 0], bins=bins, color=COR["cinza"], alpha=0.55, label="diferença de médias")
    ax.hist(est[:, 1], bins=bins, color=CATEG[0], alpha=0.65, label="ajustada pela covariável")
    ax.axvline(0.1, color=COR["destaque"], lw=1, ls="--")
    ax.set_xlabel("efeito estimado (verdadeiro: 0,1)"); ax.set_ylabel("experimentos")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    virgula(ax)
    salvar(fig, CAP, "cuped")


if __name__ == "__main__":
    fig_seatbelts()
    fig_cuped()
