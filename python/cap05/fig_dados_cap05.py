"""Cap. 5: Figuras descritivas: Mroz, Seatbelts e mcycle (figuras/cap05/dados_*)."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.estilo_livro import CATEG, COR, DADOS, aplicar_estilo, salvar  # noqa: E402

aplicar_estilo()
CAP = "cap05"
LARG = 5.3
_fmt = FuncFormatter(lambda v, _: f"{v:.10g}".replace(".", ",").replace("-", "−"))


def virgula(*axs):
    for ax in axs:
        ax.xaxis.set_major_formatter(_fmt); ax.yaxis.set_major_formatter(_fmt)


def titulo(ax, t):
    ax.set_title(t, fontsize=8.5, loc="left")


def fig_mroz():
    m = pd.read_csv(DADOS / "mroz.csv")
    rng = np.random.default_rng(1)
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(LARG, 2.0))
    a1.hist(m["lwage"], bins=25, color=CATEG[0], alpha=0.8)
    a1.set_xlabel("log(salário-hora)"); a1.set_ylabel("mulheres"); titulo(a1, "(a) resposta")
    j = lambda v: v + rng.uniform(-0.3, 0.3, len(v))
    a2.scatter(j(m["educ"]), m["lwage"], s=4, alpha=0.45, color=CATEG[0], lw=0)
    a2.set_xlabel("escolaridade"); titulo(a2, "(b) $r=0{,}34$")
    a3.scatter(j(m["fatheduc"]), j(m["educ"]), s=4, alpha=0.45, color=CATEG[2], lw=0)
    a3.set_xlabel("escolaridade do pai"); a3.set_ylabel("escolaridade")
    titulo(a3, "(c) instrumento, $r=0{,}42$")
    virgula(a1, a2, a3); fig.tight_layout(w_pad=0.5)
    salvar(fig, CAP, "dados_mroz")


def fig_seatbelts():
    s = pd.read_csv(DADOS / "seatbelts.csv"); t = s["ano"] + (s["mes"] - 1) / 12
    fig, axs = plt.subplots(1, 3, figsize=(LARG, 2.0))
    lei = t[s["lei"] == 1].min()
    for ax, c, rot, cor in [(axs[0], "motoristas", "mortos e feridos graves", CATEG[1]),
                            (axs[1], "km", "distância percorrida", CATEG[0])]:
        ax.plot(t, s[c], color=cor, lw=0.8); ax.axvline(lei, color=COR["cinza"], lw=0.8, ls="--")
        ax.set_xlabel("ano")
    titulo(axs[0], "(a) motoristas"); titulo(axs[1], "(b) tráfego")
    axs[0].text(lei - 0.3, s["motoristas"].max(), "lei", fontsize=7, ha="right", va="top")
    m = s.groupby("mes")["motoristas"].mean()
    axs[2].bar(m.index, m.values, color=CATEG[2]); axs[2].set_ylim(1200, m.max() * 1.05)
    axs[2].set_xticks([1, 4, 7, 10, 12]); axs[2].set_xlabel("mês")
    titulo(axs[2], "(c) média por mês")
    virgula(*axs); fig.tight_layout(w_pad=0.5)
    salvar(fig, CAP, "dados_seatbelts")


def fig_mcycle():
    d = pd.read_csv(DADOS / "mcycle.csv")
    fig, ax = plt.subplots(figsize=(0.7 * LARG, 2.1))
    ax.scatter(d["times"], d["accel"], s=7, color=CATEG[0], alpha=0.7, lw=0)
    for x in [14, 25, 40]:
        ax.axvline(x, color=COR["cinza"], lw=0.6, ls=":")
    for x, r in [(7, "antes"), (19.5, "queda"), (32.5, "oscilação"), (49, "fim")]:
        ax.text(x, 68, r, fontsize=7, ha="center", color=COR["cinza"])
    ax.set_xlabel("tempo após o impacto (ms)"); ax.set_ylabel("aceleração (g)")
    virgula(ax); fig.tight_layout()
    salvar(fig, CAP, "dados_mcycle")


if __name__ == "__main__":
    fig_mroz()
    fig_seatbelts()
    fig_mcycle()
