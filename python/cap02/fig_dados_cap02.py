"""Cap. 2 -- Figura descritiva dos dados de Longley (figuras/cap02/dados_longley)."""
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


def fig_longley():
    l = pd.read_csv(DADOS / "longley.csv")
    v = ["Employed", "GNP.deflator", "GNP", "Unemployed", "Armed.Forces", "Population"]
    rot = ["emprego", "deflator", "PIB", "desemprego", "Forças Arm.", "população"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(LARG, 2.5), gridspec_kw={"width_ratios": [1.2, 1]})
    for k, (c, r) in enumerate(zip(v, rot)):
        z = (l[c] - l[c].mean()) / l[c].std()
        cor = COR["cinza"] if k == 5 else CATEG[k % 5]
        a1.plot(l["Year"], z, lw=1.1, color=cor, ls="--" if k == 5 else "-", label=r)
    a1.set_xlabel("ano"); a1.set_ylabel("valor padronizado")
    a1.legend(fontsize=6, ncol=2, loc="upper left")
    a1.set_title("(a) as séries padronizadas", fontsize=8.5, loc="left")
    a1.yaxis.set_major_formatter(_fmt)
    R = l[v + ["Year"]].corr().to_numpy()
    im = a2.imshow(R, cmap="RdBu_r", vmin=-1, vmax=1)
    lab = rot + ["ano"]
    a2.set_xticks(range(7)); a2.set_xticklabels(lab, rotation=60, ha="right", fontsize=6)
    a2.set_yticks(range(7)); a2.set_yticklabels(lab, fontsize=6); a2.grid(False)
    for i in range(7):
        for j in range(7):
            a2.text(j, i, f"{R[i, j]:.2f}".replace(".", ",").replace("-", "−"), ha="center",
                    va="center", fontsize=4.8, color="white" if abs(R[i, j]) > 0.7 else "#222")
    a2.set_title("(b) correlações", fontsize=8.5, loc="left")
    cb = fig.colorbar(im, ax=a2, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=6); cb.ax.yaxis.set_major_formatter(_fmt)
    fig.tight_layout(w_pad=0.5)
    salvar(fig, "cap02", "dados_longley")


if __name__ == "__main__":
    fig_longley()
