"""Cap. 3 -- Figura descritiva dos dados de diabetes (figuras/cap03/dados_diabetes)."""
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


def fig_diabetes():
    d = pd.read_csv(DADOS / "diabetes.csv")
    fig, axs = plt.subplots(1, 3, figsize=(LARG, 2.0))
    axs[0].hist(d["y"], bins=25, color=CATEG[0], alpha=0.8)
    axs[0].set_xlabel("progressão $y$"); axs[0].set_ylabel("pacientes")
    axs[0].set_title("(a) resposta", fontsize=8.5, loc="left")
    for ax, c, rot, cor in [(axs[1], "bmi", "IMC", CATEG[1]), (axs[2], "s5", "s5 (triglicerídeos, log)", CATEG[2])]:
        ax.scatter(d[c], d["y"], s=4, alpha=0.45, color=cor, lw=0)
        b = np.polyfit(d[c], d["y"], 1); xs = np.linspace(d[c].min(), d[c].max(), 2)
        ax.plot(xs, b[1] + b[0] * xs, color=COR["primaria"], lw=1.2)
        r = np.corrcoef(d[c], d["y"])[0, 1]
        ax.set_title(f"({'b' if c == 'bmi' else 'c'}) $r={r:.2f}$".replace(".", "{,}"), fontsize=8.5, loc="left")
        ax.set_xlabel(rot)
    for ax in axs:
        ax.xaxis.set_major_formatter(_fmt); ax.yaxis.set_major_formatter(_fmt)
    fig.tight_layout(w_pad=0.6)
    salvar(fig, "cap03", "dados_diabetes")


if __name__ == "__main__":
    fig_diabetes()
