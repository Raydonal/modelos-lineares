"""Cap. 8 -- Figuras de dados do capítulo (geradas em figuras/cap08/)."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.estilo_livro import CATEG, COR, DADOS, aplicar_estilo, salvar  # noqa: E402

aplicar_estilo()
CAP = "cap08"
LARG = 5.3
_fmt = FuncFormatter(lambda v, _: f"{v:.10g}".replace(".", ",").replace("-", "−"))


def virgula(ax, eixos="xy"):
    if "x" in eixos:
        ax.xaxis.set_major_formatter(_fmt)
    if "y" in eixos:
        ax.yaxis.set_major_formatter(_fmt)


# 1. IS-LM não linear ---------------------------------------------------------
def fig_islm():
    c0, c1, T = 100, 0.75, 200
    F = lambda x, M: np.array([x[0] - c0 - c1 * (x[0] - T) - 800 * (x[1] / 5) ** -0.3125 - 250,
                               M - 500 * (x[0] / 4000) ** 2 * (x[1] / 5) ** -1])
    Jx = lambda x: np.array([[1 - c1, 0.3125 * 800 / 5 * (x[1] / 5) ** -1.3125],
                             [-1000 / 4000 * (x[0] / 4000) * (x[1] / 5) ** -1, 500 * (x[0] / 4000) ** 2 * (x[1] / 5) ** -2 / 5]])

    def newton(M):
        z = np.array([4000.0, np.log(5.0)])
        for _ in range(100):
            x = np.array([z[0], np.exp(z[1])]); z = z - np.linalg.solve(Jx(x) @ np.diag([1, x[1]]), F(x, M))
        return np.array([z[0], np.exp(z[1])])
    Ms = np.linspace(300, 3000, 60); sol = np.array([newton(M) for M in Ms])
    dY = [(-np.linalg.solve(Jx(x), np.array([[-1.0, 0], [0, 1]])))[0, 1] for x in sol]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(LARG, 2.5))
    a1.plot(Ms, sol[:, 1], color=CATEG[0], label="não linear")
    a1.plot(Ms, 5 - (Ms - 500) * 0.25 / 37.5, color=CATEG[1], ls="--", label="linear (Cap. 3)")
    a1.axhline(0, color=COR["cinza"], lw=0.5); a1.set_ylim(-3, 7)
    a1.set_xlabel("oferta real de moeda $M/P$"); a1.set_ylabel("juros de equilíbrio $r$ (%)")
    a1.legend(fontsize=6.5); a1.set_title("(a) juros", fontsize=8.5, loc="left"); virgula(a1, "y")
    a2.plot(Ms, dY, color=CATEG[0]); a2.axhline(50 / 37.5, color=CATEG[1], ls="--")
    a2.set_xlabel("oferta real de moeda $M/P$"); a2.set_ylabel(r"$\partial Y/\partial(M/P)$")
    a2.set_title("(b) potência da política monetária", fontsize=8.5, loc="left"); virgula(a2, "y")
    fig.tight_layout(); salvar(fig, CAP, "islm_nao_linear")


# 2. Parcelas subdivididas -----------------------------------------------------
def fig_parcelas():
    s2, s2p = 153.5, 45.1
    V = np.kron(np.eye(18), s2 * np.eye(4) + s2p * np.ones((4, 4)))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(LARG, 2.6), gridspec_kw={"width_ratios": [1, 1.2]})
    im = a1.imshow(V[:24, :24], cmap="Blues", vmin=0)
    a1.set_xticks([]); a1.set_yticks([]); a1.grid(False)
    a1.set_title("(a) $\\mathbf{V}$ (24 primeiras subparcelas)", fontsize=8.5, loc="left")
    ep = {"variedade\n(entre parcelas)": (5.28, 4.42), "nitrogênio\n(dentro das parcelas)": (4.13, 5.10)}
    pos = np.arange(2)
    a2.bar(pos - 0.18, [v[0] for v in ep.values()], 0.34, color=CATEG[0], label="MQG (covariância de Kronecker)")
    a2.bar(pos + 0.18, [v[1] for v in ep.values()], 0.34, color=CATEG[1], label="MQO (independência)")
    a2.set_xticks(pos, list(ep.keys()), fontsize=7); a2.set_ylabel("erro-padrão do contraste")
    a2.legend(fontsize=6.2, loc="upper right"); a2.set_ylim(0, 7.5); a2.grid(axis="x", visible=False)
    a2.set_title("(b) erros-padrão", fontsize=8.5, loc="left")
    fig.tight_layout(); salvar(fig, CAP, "parcelas_kronecker")


# 3. Influência nas componentes principais -------------------------------------
def fig_influencia():
    dig = pd.read_csv(DADOS / "digitos.csv"); X = dig.drop(columns="digito").to_numpy(float); n = len(X)
    lam, Q = np.linalg.eigh(np.cov(X.T)); l1, q1 = lam[-1], Q[:, -1]
    z = (X - X.mean(0)) @ q1; aprox = (l1 - n / (n - 1) * z ** 2) / (n - 2)
    exato = np.array([np.linalg.eigvalsh(np.cov(np.delete(X, i, 0).T))[-1] - l1 for i in range(n)])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(LARG, 2.5))
    a1.plot(exato, aprox, ".", ms=2, color=CATEG[0]); m = [exato.min(), exato.max()]
    a1.plot(m, m, color=COR["cinza"], lw=0.7, ls=":")
    a1.set_xlabel("variação exata de $\\lambda_1$"); a1.set_ylabel("aproximação $\\mathbf{q}_1^\\top\\Delta\\mathbf{S}\\,\\mathbf{q}_1$")
    a1.set_title("(a) derivada vs. recálculo", fontsize=8.5, loc="left"); virgula(a1)
    for k, c in enumerate(range(10)):
        sel = dig["digito"].to_numpy() == c
        a2.scatter(z[sel], exato[sel], s=2, color=plt.cm.tab10(k), label=str(c))
    a2.set_xlabel("escore na 1ª componente"); a2.set_ylabel("variação exata de $\\lambda_1$")
    a2.legend(fontsize=5.5, ncol=2, markerscale=3, loc="lower center", handletextpad=0.1, columnspacing=0.5)
    a2.set_title("(b) influência e escore", fontsize=8.5, loc="left"); virgula(a2)
    fig.tight_layout(); salvar(fig, CAP, "influencia_pca")


if __name__ == "__main__":
    fig_islm()
    fig_parcelas()
    fig_influencia()
