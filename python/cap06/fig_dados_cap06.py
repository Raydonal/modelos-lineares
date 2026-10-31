"""Cap. 6 -- Figura descritiva do ensaio de trigo (figuras/cap06/dados_trigo)."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.estilo_livro import CATEG, COR, DADOS, aplicar_estilo, salvar  # noqa: E402

aplicar_estilo()
LARG = 5.3
_fmt = FuncFormatter(lambda v, _: f"{v:.10g}".replace(".", ",").replace("-", "−"))


def fig_trigo():
    d = pd.read_csv(DADOS / "yan_trigo.csv")
    Y = d.pivot_table("producao", "genotipo", "ambiente")
    Y = Y[Y.mean().sort_values().index]                 # ambientes do pior ao melhor
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(LARG, 2.6), gridspec_kw={"width_ratios": [1.25, 1]})
    x = np.arange(Y.shape[1])
    for g in Y.index:
        a1.plot(x, Y.loc[g], color="#C9D2DC", lw=0.7)
    for g, c in zip(["Fun", "Har", "Kat"], [CATEG[0], CATEG[1], CATEG[2]]):
        a1.plot(x, Y.loc[g], "o-", ms=2.5, lw=1.2, color=c, label=g)
    a1.plot(x, Y.mean(), "k--", lw=1, label="média")
    a1.set_xticks(x); a1.set_xticklabels(Y.columns, rotation=60, fontsize=6.5)
    a1.set_ylabel("produção (t/ha)"); a1.legend(fontsize=6.5, ncol=2)
    a1.set_title("(a) cultivares nos ambientes", fontsize=8.5, loc="left")
    a1.yaxis.set_major_formatter(_fmt)
    y = Y.to_numpy(); I = y - y.mean(1, keepdims=True) - y.mean(0, keepdims=True) + y.mean()
    o = np.argsort(-y.mean(1))
    im = a2.imshow(I[o], cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    a2.set_yticks(range(len(o))); a2.set_yticklabels(Y.index[o], fontsize=5.5)
    a2.set_xticks(x); a2.set_xticklabels(Y.columns, rotation=60, fontsize=6); a2.grid(False)
    a2.set_title("(b) interação (t/ha)", fontsize=8.5, loc="left")
    cb = fig.colorbar(im, ax=a2, fraction=0.05, pad=0.03)
    cb.ax.tick_params(labelsize=6); cb.ax.yaxis.set_major_formatter(_fmt)
    fig.tight_layout(w_pad=0.5)
    salvar(fig, "cap06", "dados_trigo")


if __name__ == "__main__":
    fig_trigo()
