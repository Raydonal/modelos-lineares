"""Cap. 4: Figura descritiva da temporada de 1981 da NFL (figuras/cap04/dados_nfl)."""
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


def fig_nfl():
    j = pd.read_csv(DADOS / "nfl.csv").query("ano == 1981")
    m = j["pontos_mandante"] - j["pontos_visitante"]
    saldo = pd.concat([pd.Series(m.values, j["mandante"]), pd.Series(-m.values, j["visitante"])])
    s = saldo.groupby(level=0).sum().sort_values()
    fig = plt.figure(figsize=(LARG, 3.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.75, wspace=0.3)
    a1 = fig.add_subplot(gs[0, 0]); a2 = fig.add_subplot(gs[0, 1]); a3 = fig.add_subplot(gs[1, :])
    a1.hist(m, bins=np.arange(-45, 46, 5), color=CATEG[0], alpha=0.8)
    a1.axvline(0, color=COR["cinza"], lw=0.6); a1.axvline(m.mean(), color=COR["destaque"], lw=1.2)
    a1.set_xlabel("margem do mandante (pontos)"); a1.set_ylabel("jogos")
    a1.set_title("(a) margens", fontsize=8.5, loc="left")
    a2.scatter(j["linha"], m, s=6, alpha=0.5, color=CATEG[2], lw=0)
    a2.axhline(0, color=COR["cinza"], lw=0.6)
    a2.set_xlabel("linha de apostas (pontos)"); a2.set_ylabel("margem")
    r = np.corrcoef(j["linha"], m)[0, 1]
    a2.set_title(f"(b) margem e linha, $r={r:.2f}$".replace(".", "{,}"), fontsize=8.5, loc="left")
    a3.bar(range(len(s)), s.values, color=[CATEG[1] if v < 0 else CATEG[0] for v in s.values])
    a3.set_xticks(range(len(s))); a3.set_xticklabels(s.index, rotation=90, fontsize=6)
    a3.set_ylabel("saldo de pontos"); a3.set_title("(c) saldo de pontos por time", fontsize=8.5, loc="left")
    for ax in (a1, a2, a3):
        ax.yaxis.set_major_formatter(_fmt)
    a1.xaxis.set_major_formatter(_fmt); a2.xaxis.set_major_formatter(_fmt)
    salvar(fig, "cap04", "dados_nfl")


if __name__ == "__main__":
    fig_nfl()
