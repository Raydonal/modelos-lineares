"""Cap. 9: Figuras descritivas: índices europeus e airquality (figuras/cap09/dados_*)."""
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


def virgula(*axs):
    for ax in axs:
        ax.xaxis.set_major_formatter(_fmt); ax.yaxis.set_major_formatter(_fmt)


def fig_eustock():
    eu = pd.read_csv(DADOS / "eustock.csv")
    r = 100 * np.log(eu.iloc[:, 1:]).diff().dropna()
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(LARG, 2.0))
    t = 1991.496 + (eu["dia"].to_numpy() - 1) / 260    # 260 dias úteis por ano
    for k, c in enumerate(eu.columns[1:]):
        a1.plot(t, eu[c] / eu[c].iloc[0], lw=0.8, color=CATEG[k], label=c)
    a1.set_ylabel("índice (início = 1)"); a1.legend(fontsize=6); a1.set_xlabel("ano")
    a1.set_xticks([1992, 1994, 1996, 1998])
    a1.set_title("(a) índices", fontsize=8.5, loc="left")
    x = r["DAX"]; z = (x - x.mean()) / x.std()
    a2.hist(z, bins=60, density=True, color=CATEG[0], alpha=0.75)
    u = np.linspace(-6, 6, 200); a2.plot(u, np.exp(-u**2 / 2) / np.sqrt(2 * np.pi), color=COR["destaque"], lw=1)
    a2.set_xlim(-6, 6); a2.set_xlabel("retorno do DAX padronizado")
    a2.set_title("(b) caudas pesadas", fontsize=8.5, loc="left")
    a3.scatter(r["DAX"], r["CAC"], s=2, alpha=0.4, color=CATEG[2], lw=0)
    a3.set_xlabel("DAX (%)"); a3.set_ylabel("CAC (%)")
    a3.set_title("(c) $r=0{,}73$", fontsize=8.5, loc="left")
    virgula(a1, a2, a3); fig.tight_layout(w_pad=0.5)
    salvar(fig, "cap09", "dados_eustock")


def fig_airquality():
    a = pd.read_csv(DADOS / "airquality.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(LARG, 2.2), gridspec_kw={"width_ratios": [1.3, 1]})
    dia = np.arange(len(a))
    a1.plot(dia, a["Ozone"], "o-", ms=2, lw=0.7, color=CATEG[0])
    falta = a["Ozone"].isna()
    a1.plot(dia[falta], np.full(falta.sum(), -8), "|", ms=6, color=COR["destaque"])
    a1.set_xlabel("dia (1 de maio a 30 de setembro)"); a1.set_ylabel("ozônio (ppb)")
    a1.set_title("(a) ozônio; traços: dias sem medida", fontsize=8.5, loc="left")
    a2.scatter(a["Temp"], a["Ozone"], s=5, alpha=0.6, color=CATEG[1], lw=0)
    a2.set_xlabel("temperatura (°F)"); a2.set_title("(b) $r=0{,}70$", fontsize=8.5, loc="left")
    virgula(a1, a2); fig.tight_layout(w_pad=0.5)
    salvar(fig, "cap09", "dados_airquality")


if __name__ == "__main__":
    fig_eustock()
    fig_airquality()
