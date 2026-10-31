"""Estilo gráfico comum às figuras do livro (paleta, fontes e salvamento)."""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

RAIZ = Path(__file__).resolve().parents[2]          # pasta codigo/
DADOS = RAIZ / "dados"
FIGURAS = RAIZ / "figuras"

# Paleta do livro (mesmas cores do preâmbulo LaTeX)
COR = {
    "primaria":   "#1F4E79",
    "secundaria": "#2A9D8F",
    "destaque":   "#E76F51",
    "ouro":       "#E9A23B",
    "violeta":    "#6D597A",
    "cinza":      "#5C6770",
    "fundo":      "#F4F7FB",
}
# Paleta categórica para séries de dados, nesta ordem fixa (validada para
# daltonismo: separação CVD >= 8 entre vizinhas). O âmbar tem contraste baixo
# sobre branco: usá-lo sempre com rótulo direto.
CATEG = ["#3A6FB0", "#E06A4E", "#2A9D8F", "#C98A1A", "#9063B5"]
CICLO = CATEG


def aplicar_estilo():
    """Configura o matplotlib com o estilo do livro."""
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Lato", "DejaVu Sans"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.edgecolor": COR["cinza"],
        "axes.labelcolor": "#222222",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#DDE3EA",
        "grid.linewidth": 0.6,
        "axes.prop_cycle": mpl.cycler(color=CICLO),
        "xtick.color": COR["cinza"],
        "ytick.color": COR["cinza"],
        "legend.frameon": False,
        "lines.linewidth": 1.6,
    })


def salvar(fig, capitulo, nome):
    """Salva a figura em figuras/capXX/nome.pdf (formato vetorial para o LaTeX)."""
    destino = FIGURAS / capitulo
    destino.mkdir(parents=True, exist_ok=True)
    caminho = destino / f"{nome}.pdf"
    fig.savefig(caminho)
    plt.close(fig)
    print(f"figura salva: {caminho.relative_to(RAIZ)}")
    return caminho
