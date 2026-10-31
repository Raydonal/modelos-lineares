"""Cap. 1: Figuras descritivas dos conjuntos de dados (figuras/cap01/dados_*)."""
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.estilo_livro import CATEG, COR, DADOS, aplicar_estilo, salvar  # noqa: E402

aplicar_estilo()
CAP = "cap01"
LARG = 5.3
_fmt = FuncFormatter(lambda v, _: f"{v:.10g}".replace(".", ",").replace("-", "−"))


def virgula(*axs):
    for ax in axs:
        ax.xaxis.set_major_formatter(_fmt); ax.yaxis.set_major_formatter(_fmt)


def titulo(ax, t):
    ax.set_title(t, fontsize=8.5, loc="left")


# 1. wage1: distribuição do salário, efeito do log e média por escolaridade
def fig_wage1():
    w = pd.read_csv(DADOS / "wage1.csv"); lw = np.log(w["wage"])
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(LARG, 2.0))
    a1.hist(w["wage"], bins=30, color=CATEG[0], alpha=0.8)
    a1.axvline(w["wage"].mean(), color=COR["destaque"], lw=1)
    a1.axvline(w["wage"].median(), color=COR["destaque"], lw=1, ls="--")
    a1.set_xlabel("salário-hora (US\\$)"); a1.set_ylabel("trabalhadores")
    titulo(a1, "(a) salário")
    a2.hist(lw, bins=30, color=CATEG[2], alpha=0.8)
    a2.axvline(lw.mean(), color=COR["destaque"], lw=1)
    a2.set_xlabel("log(salário-hora)"); titulo(a2, "(b) log do salário")
    g = w.assign(lw=lw).groupby("educ")["lw"].agg(["mean", "size"])
    a3.scatter(g.index, g["mean"], s=4 + 1.2 * g["size"], color=CATEG[0], alpha=0.7, lw=0)
    b = np.polyfit(w["educ"], lw, 1)
    xs = np.array([0, 18]); a3.plot(xs, b[1] + b[0] * xs, color=CATEG[1], lw=1.2)
    a3.set_xlabel("anos de escolaridade"); a3.set_ylabel("média do log")
    titulo(a3, "(c) média por escolaridade")
    virgula(a1, a2, a3); fig.tight_layout(w_pad=0.6)
    salvar(fig, CAP, "dados_wage1")


# 2. UKgas: série em nível e em log, e perfil sazonal por década
def fig_ukgas():
    g = pd.read_csv(DADOS / "ukgas.csv"); t = g["ano"] + (g["trimestre"] - 1) / 4
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(LARG, 2.0))
    a1.plot(t, g["consumo"], color=CATEG[0], lw=1); a1.set_xlabel("ano")
    a1.set_ylabel("consumo"); titulo(a1, "(a) nível")
    a2.plot(t, np.log(g["consumo"]), color=CATEG[2], lw=1); a2.set_xlabel("ano")
    titulo(a2, "(b) logaritmo")
    for k, (dec, cor) in enumerate(zip([1960, 1970, 1980], CATEG)):
        s = g[(g["ano"] >= dec) & (g["ano"] < dec + 10)]
        m = np.log(s["consumo"]).groupby(s["trimestre"]).mean()
        a3.plot(m.index, m - m.mean(), "o-", ms=3, color=cor, label=f"anos {dec % 100}")
    a3.set_xticks([1, 2, 3, 4]); a3.set_xlabel("trimestre")
    a3.set_ylabel("desvio do log"); a3.legend(fontsize=6.5)
    titulo(a3, "(c) perfil sazonal")
    virgula(a1, a2, a3); fig.tight_layout(w_pad=0.6)
    salvar(fig, CAP, "dados_ukgas")


# 3. Aveia: parcelas por dose e variedade, e efeito de bloco
def fig_aveia():
    a = pd.read_csv(DADOS / "yates_oats.csv")
    vars_ = ["GoldenRain", "Marvellous", "Victory"]; doses = [0, 0.2, 0.4, 0.6]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(LARG, 2.3), gridspec_kw={"width_ratios": [1.3, 1]})
    for k, v in enumerate(vars_):
        s = a[a["gen"] == v]; dx = (k - 1) * 0.022
        a1.scatter(s["nitro"] + dx, s["yield"], s=7, color=CATEG[k], alpha=0.45, lw=0)
        m = s.groupby("nitro")["yield"].mean()
        a1.plot(m.index + dx, m.values, "o-", ms=3.5, color=CATEG[k], label=v)
    a1.set_xticks(doses); a1.set_xlabel("dose de nitrogênio (cwt/acre)")
    a1.set_ylabel("produção (1/4 lb por subparcela)"); a1.legend(fontsize=6.5)
    titulo(a1, "(a) produção, dose e variedade")
    bl = sorted(a["block"].unique())
    for j, b in enumerate(bl):
        s = a[a["block"] == b]
        a2.scatter(np.full(len(s), j) + np.linspace(-0.2, 0.2, len(s)), s["yield"], s=7,
                   color=CATEG[0], alpha=0.5, lw=0)
        a2.plot([j - 0.3, j + 0.3], [s["yield"].mean()] * 2, color=COR["destaque"], lw=1.5)
    a2.set_xticks(range(len(bl))); a2.set_xticklabels(bl); a2.set_xlabel("bloco")
    titulo(a2, "(b) variação entre blocos")
    virgula(a1); a2.yaxis.set_major_formatter(_fmt)
    fig.tight_layout(w_pad=0.6); salvar(fig, CAP, "dados_aveia")


# 4. Dígitos: um exemplo de cada classe, imagem média e variância por pixel
def fig_digitos():
    d = pd.read_csv(DADOS / "digitos.csv"); X = d.drop(columns="digito").to_numpy(float)
    y = d["digito"].to_numpy()
    fig = plt.figure(figsize=(LARG, 2.2))
    gs = fig.add_gridspec(2, 7, width_ratios=[1] * 5 + [1.5, 1.5], wspace=0.15, hspace=0.25)
    for k in range(10):
        ax = fig.add_subplot(gs[k // 5, k % 5])
        ax.imshow(X[np.argmax(y == k)].reshape(8, 8), cmap="Greys", vmin=0, vmax=16)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    a1 = fig.add_subplot(gs[:, 5]); a2 = fig.add_subplot(gs[:, 6])
    a1.imshow(X.mean(0).reshape(8, 8), cmap="Greys"); a1.set_title("média", fontsize=8)
    im = a2.imshow(X.var(0, ddof=1).reshape(8, 8), cmap="viridis")
    a2.set_title("variância", fontsize=8)
    for ax in (a1, a2):
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    cb = fig.colorbar(im, ax=a2, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=6); cb.ax.yaxis.set_major_formatter(_fmt)
    salvar(fig, CAP, "dados_digitos")


# 5. Austen: lei de Zipf e tamanho dos romances
def fig_austen():
    d = pd.read_csv(DADOS / "austen.csv", keep_default_na=False)
    f = np.array(sorted(Counter(re.findall(r"[a-z]+", " ".join(d["texto"]).lower())).values(),
                        reverse=True))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(LARG, 2.2), gridspec_kw={"width_ratios": [1, 1.2]})
    r = np.arange(1, len(f) + 1)
    a1.loglog(r, f, color=CATEG[0], lw=1.2, label="observado")
    a1.loglog(r, f[9] * 10 / r, color=COR["destaque"], lw=0.9, ls="--", label=r"freq. $\propto$ 1/posto")
    a1.set_xlabel("posto da palavra"); a1.set_ylabel("frequência"); a1.legend(fontsize=6.5)
    titulo(a1, "(a) lei de Zipf")
    liv = d.groupby("livro")["texto"].apply(lambda t: len(re.findall(r"[a-z]+", " ".join(t).lower())))
    a2.barh(range(len(liv)), liv.values / 1000, color=CATEG[2])
    a2.set_yticks(range(len(liv))); a2.set_yticklabels(liv.index, fontsize=6.5)
    a2.set_xlabel("palavras (milhares)"); titulo(a2, "(b) tamanho dos romances")
    a2.xaxis.set_major_formatter(_fmt)
    fig.tight_layout(w_pad=0.6); salvar(fig, CAP, "dados_austen")


if __name__ == "__main__":
    fig_wage1()
    fig_ukgas()
    fig_aveia()
    fig_digitos()
    fig_austen()
