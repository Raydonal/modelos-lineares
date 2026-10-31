"""Cap. 11: Figuras de dados do capítulo (geradas em figuras/cap11/)."""
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
CAP = "cap11"
LARG = 5.3
_fmt = FuncFormatter(lambda v, _: f"{v:.10g}".replace(".", ",").replace("-", "−"))


def virgula(ax, eixos="xy"):
    if "x" in eixos:
        ax.xaxis.set_major_formatter(_fmt)
    if "y" in eixos:
        ax.yaxis.set_major_formatter(_fmt)


def coef(Z, v):
    return np.linalg.lstsq(Z, v, rcond=None)[0]


# 1. Médias das caselas da aveia ---------------------------------------------------
def fig_aveia():
    a = pd.read_csv(DADOS / "yates_oats.csv")
    mu = a.groupby(["gen", "nitro"])["yield"].mean().unstack()
    doses = mu.columns.to_numpy()
    fig, ax = plt.subplots(figsize=(0.72 * LARG, 2.6))
    for k, g in enumerate(mu.index):
        ax.plot(doses, mu.loc[g], "o-", ms=3.5, color=CATEG[k], label=g)
    ax.plot([0.6], [mu.loc["Victory", 0.6]], "o", ms=11, mfc="none", mec=COR["destaque"], mew=1.3)
    ax.annotate("casela retirada\nna aplicação", (0.6, mu.loc["Victory", 0.6]), xytext=(0.36, 72),
                fontsize=6.5, color=COR["destaque"], arrowprops=dict(arrowstyle="->", color=COR["destaque"], lw=0.8))
    ax.set_xticks(doses); ax.set_xlabel("dose de nitrogênio (cwt/acre)")
    ax.set_ylabel("produção média (1/4 lb)"); ax.legend(fontsize=6.5, loc="upper left"); virgula(ax)
    fig.tight_layout(); salvar(fig, CAP, "aveia_caselas")


# 2. Salário e escolaridade: média condicional e formas lineares nos parâmetros -----
def fig_salarios():
    w = pd.read_csv(DADOS / "wage1.csv")
    y = np.log(w["wage"].to_numpy()); e = w["educ"].to_numpy(float)
    ej, j, nj = np.unique(e, return_inverse=True, return_counts=True)
    mj = np.bincount(j, y) / nj
    um = np.ones(len(y)); g = np.linspace(0, 18, 200)
    formas = {"reta": (np.column_stack([um, e]), lambda x: np.column_stack([np.ones_like(x), x])),
              "parábola": (np.column_stack([um, e, e**2]), lambda x: np.column_stack([np.ones_like(x), x, x**2])),
              "quebrada em 12": (np.column_stack([um, e, np.maximum(e - 12, 0)]),
                                 lambda x: np.column_stack([np.ones_like(x), x, np.maximum(x - 12, 0)]))}
    fig, ax = plt.subplots(figsize=(0.8 * LARG, 2.7))
    ax.scatter(ej, mj, s=4 + 1.2 * nj, color=COR["cinza"], alpha=0.55, lw=0, label="média por casela")
    for k, (nome, (X, f)) in enumerate(formas.items()):
        ax.plot(g, f(g) @ coef(X, y), color=CATEG[k], label=nome)
    ax.set_xlabel("anos de escolaridade"); ax.set_ylabel("log do salário-hora")
    ax.legend(fontsize=6.5, loc="upper left"); virgula(ax)
    fig.tight_layout(); salvar(fig, CAP, "salarios_cef")


# 3. Sazonalidade: indicadoras e harmônicos ----------------------------------------
def fig_sazonalidade():
    s = pd.read_csv(DADOS / "seatbelts.csv")
    y = np.log(s["motoristas"].to_numpy()); n = len(y); t = np.arange(1, n + 1.0)
    lei = s["lei"].to_numpy(float)

    def fourier(K, tt):
        F = [np.ones_like(tt)]
        for k in range(1, K + 1):
            F.append(np.cos(2 * np.pi * k * tt / 12))
            if k < 6:
                F.append(np.sin(2 * np.pi * k * tt / 12))
        return np.column_stack(F)

    meses = np.arange(1, 13.0); fino = np.linspace(1, 12, 200)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(LARG, 2.5), gridspec_kw={"width_ratios": [1.5, 1]})
    sqr = []
    for K in range(7):
        X = np.column_stack([fourier(K, t), t, lei]); b = coef(X, y)
        sqr.append(np.sum((y - X @ b) ** 2))
        saz = fourier(K, meses) @ b[: X.shape[1] - 2]
        if K == 6:
            a1.plot(meses, saz - saz.mean(), "o", ms=3.5, color=COR["cinza"], label="12 indicadoras ($K=6$)", zorder=3)
        elif K in (1, 2):
            sf = fourier(K, fino) @ b[: X.shape[1] - 2]
            a1.plot(fino, sf - saz.mean(), color=CATEG[K - 1], label=f"$K={K}$ harmônico" + ("s" if K > 1 else ""))
    a1.set_xticks(range(1, 13)); a1.set_xlabel("mês"); a1.set_ylabel("efeito sazonal (log)")
    a1.legend(fontsize=6.3, loc="upper left"); a1.set_title("(a) padrão sazonal", fontsize=8.5, loc="left"); virgula(a1)
    sqr = np.array(sqr); frac = (sqr[0] - sqr) / (sqr[0] - sqr[6])
    a2.bar(range(7), frac, color=COR["primaria"], width=0.65)
    a2.set_xlabel("harmônicos $K$"); a2.set_ylabel("fração da redução da SQR"); a2.set_ylim(0, 1.05)
    a2.set_title("(b) quanto cada $K$ capta", fontsize=8.5, loc="left"); virgula(a2, "y")
    fig.tight_layout(); salvar(fig, CAP, "sazonalidade_fourier")


# 4. Atributos aleatórios ---------------------------------------------------------
def fig_atributos():
    d = pd.read_csv(DADOS / "digitos.csv"); dig = d["digito"].to_numpy()
    X = np.column_stack([np.ones(len(d)), d.iloc[:, 1:] / 16]); Y = np.eye(10)[dig]
    W = Gerador(7).matriz_normal(2000, 65, sd=1 / 8)
    Phi = np.maximum(X @ W.T, 0)
    ms = np.unique(np.round(np.geomspace(10, 2000, 36)).astype(int))
    ms = np.unique(np.concatenate([ms, [900, 950, 1000, 1050, 1100, 1200]]))
    ac, nr = [], []
    for m in ms:
        A = Phi[:, :m]; U, s, Vt = np.linalg.svd(A[:1000], full_matrices=False)
        r = np.sum(s > np.sqrt(np.finfo(float).eps) * s[0])
        B = Vt[:r].T @ ((U[:, :r].T @ Y[:1000]) / s[:r, None])
        ac.append(np.mean(np.argmax(A[1000:] @ B, 1) == dig[1000:])); nr.append(np.sqrt(np.sum(B ** 2)))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(LARG, 2.4))
    for ax, v, rot in ((a1, ac, "acerto no teste"), (a2, nr, "norma de $\\widehat{\\mathbf{B}}$")):
        ax.plot(ms, v, "o-", ms=2.2, color=COR["primaria"])
        ax.axvline(1000, color=COR["destaque"], ls=":", lw=1)
        ax.set_xscale("log"); ax.set_xlabel("número de atributos $m$"); ax.set_ylabel(rot)
    a1.set_title("(a) acerto", fontsize=8.5, loc="left"); a2.set_title("(b) norma", fontsize=8.5, loc="left")
    a2.set_yscale("log"); virgula(a1, "y")
    a1.text(1080, 0.9, "$m=n$", fontsize=6.5, color=COR["destaque"])
    fig.tight_layout(); salvar(fig, CAP, "atributos_aleatorios")


if __name__ == "__main__":
    fig_aveia()
    fig_salarios()
    fig_sazonalidade()
    fig_atributos()
