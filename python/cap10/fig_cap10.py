"""Cap. 10 -- Figuras de dados do capítulo (geradas em figuras/cap10/)."""
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from scipy.stats import chi2, f, ncf, ncx2, norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.austen import embeddings_austen  # noqa: E402
from comum.estilo_livro import CATEG, COR, DADOS, aplicar_estilo, salvar  # noqa: E402
from comum.gerador import Gerador  # noqa: E402

aplicar_estilo()
CAP = "cap10"
LARG = 5.3
_fmt = FuncFormatter(lambda v, _: f"{v:.10g}".replace(".", ",").replace("-", "−"))


def virgula(ax, eixos="xy"):
    if "x" in eixos:
        ax.xaxis.set_major_formatter(_fmt)
    if "y" in eixos:
        ax.yaxis.set_major_formatter(_fmt)


# 1. Qui-quadrado não central -----------------------------------------------------
def fig_qui2():
    u = np.linspace(0.01, 30, 400); q = chi2.ppf(0.95, 3)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(LARG, 2.5))
    for k, lam in enumerate((0, 2, 6, 12)):
        dens = chi2.pdf(u, 3) if lam == 0 else ncx2.pdf(u, 3, lam)
        a1.plot(u, dens, color=CATEG[k], label=f"$\\lambda={lam}$")
    a1.axvline(q, color=COR["cinza"], ls=":", lw=1)
    a1.set_xlabel("$u$"); a1.set_ylabel("densidade"); a1.legend(fontsize=6.3)
    a1.set_title("(a) $\\chi^2_3(\\lambda)$", fontsize=8.5, loc="left"); virgula(a1)
    lam = np.linspace(0.01, 30, 300)
    for k, gl in enumerate((1, 3, 10, 30)):
        a2.plot(lam, ncx2.sf(chi2.ppf(0.95, gl), gl, lam), color=CATEG[k], label=f"{gl} gl")
    a2.set_xlabel("$\\lambda$"); a2.set_ylabel("poder do teste a 5%"); a2.legend(fontsize=6.3, loc="lower right")
    a2.set_title("(b) poder e graus de liberdade", fontsize=8.5, loc="left"); virgula(a2)
    fig.tight_layout(); salvar(fig, CAP, "qui2_nao_central")


# 2. Poder nas parcelas subdivididas ----------------------------------------------
def fig_parcelas():
    s2, sp2 = 177.0833, 106.0618                 # da saída 04_parcelas
    tv = np.array([-5, 0, 5.0]); tn = np.array([-5, -5 / 3, 5 / 3, 5])
    b = np.arange(2, 25)
    pv = [ncf.sf(f.ppf(0.95, 2, 2 * (k - 1)), 2, 2 * (k - 1), 4 * k * np.sum(tv ** 2) / (s2 + 4 * sp2)) for k in b]
    pn = [ncf.sf(f.ppf(0.95, 3, 9 * (k - 1)), 3, 9 * (k - 1), 3 * k * np.sum(tn ** 2) / s2) for k in b]
    fig, ax = plt.subplots(figsize=(0.7 * LARG, 2.5))
    ax.plot(b, pn, "o-", ms=2.5, color=CATEG[0], label="nitrogênio (subparcela)")
    ax.plot(b, pv, "s-", ms=2.5, color=CATEG[1], label="variedade (parcela)")
    ax.axhline(0.8, color=COR["cinza"], ls=":", lw=1)
    ax.set_xlabel("número de blocos"); ax.set_ylabel("poder a 5%"); ax.set_ylim(0, 1)
    ax.legend(fontsize=6.3, loc="lower right"); virgula(ax)
    fig.tight_layout(); salvar(fig, CAP, "parcelas_poder")


# 3. Deriva de embeddings ---------------------------------------------------------
def fig_deriva():
    vocab, _, E = embeddings_austen(); ind = {w: i for i, w in enumerate(vocab)}
    d = pd.read_csv(DADOS / "austen.csv", keep_default_na=False)

    def blocos(livro, L=200):
        ids = [ind[w] for w in re.findall(r"[a-z]+", " ".join(d["texto"][d["livro"] == livro]).lower()) if w in ind]
        m = len(ids) // L
        return E[ids[:m * L]].reshape(m, L, -1).mean(1)
    pp = blocos("Pride & Prejudice"); h = len(pp) // 2; ref = pp[:h]; mu = ref.mean(0); p = 10
    V = np.linalg.svd(ref - mu, full_matrices=False)[2][:p].T

    def fst(X1, X2):
        Z1, Z2 = (X1 - mu) @ V, (X2 - mu) @ V; n1, n2 = len(Z1), len(Z2); g = n1 + n2 - 2
        dm = Z1.mean(0) - Z2.mean(0); Sp = ((n1 - 1) * np.cov(Z1.T) + (n2 - 1) * np.cov(Z2.T)) / g
        return n1 * n2 / (n1 + n2) * dm @ np.linalg.solve(Sp, dm) * (g - p + 1) / (p * g), g
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(LARG, 2.6))
    lotes = {"Orgulho (2ª metade)": pp[h:h + 30], "Emma": blocos("Emma")[:30], "Persuasão": blocos("Persuasion")[:30]}
    Zr = (ref - mu) @ V
    a1.plot(Zr[:, 0], Zr[:, 1], ".", ms=2.5, color=COR["cinza"], alpha=0.5, label="referência")
    for k, (nome, X) in enumerate(lotes.items()):
        Z = (X - mu) @ V; a1.plot(Z[:, 0], Z[:, 1], "o", ms=2.5, color=CATEG[k], label=nome)
    a1.set_xlabel("CP 1"); a1.set_ylabel("CP 2"); virgula(a1)
    a1.legend(fontsize=5.6, loc="lower left", frameon=True, framealpha=0.9, edgecolor="none")
    a1.set_title("(a) blocos de 200 palavras", fontsize=8.5, loc="left")
    g = Gerador(41); k = h // 2; nulos = []
    for _ in range(500):
        o = np.argsort(g.unif(h)); nulos.append(fst(ref[o[:k]], ref[o[k:]])[0])
    gl = fst(ref[:k], ref[k:])[1]
    a2.hist(nulos, bins=30, density=True, color=CATEG[0], alpha=0.35, label="partições ao acaso")
    x = np.linspace(0.01, 12, 300); a2.plot(x, f.pdf(x, p, gl - p + 1), color=CATEG[0], label=f"$F({p},{gl - p + 1})$")
    for k, (nome, X) in enumerate(lotes.items()):
        a2.axvline(fst(ref, X)[0], color=CATEG[k], lw=1.2)
    a2.set_xlabel("estatística $F$"); a2.set_ylabel("densidade"); virgula(a2)
    a2.legend(fontsize=6.3, loc="center right", frameon=True, framealpha=0.9, edgecolor="none")
    a2.set_title("(b) distribuição nula e lotes", fontsize=8.5, loc="left")
    fig.tight_layout(); salvar(fig, CAP, "deriva")


# 4. Ataques de pertencimento -----------------------------------------------------
def fig_pertencimento():
    db = pd.read_csv(DADOS / "diabetes.csv"); X = db.iloc[:, :10].to_numpy(); p = 10
    X = (X - X.min(0)) / (X.max(0) - X.min(0)); Delta = np.sqrt(p); D = np.linalg.norm(X[0])
    eps = np.linspace(0.1, 10, 200); z = norm.ppf(1e-5)
    u = (np.sqrt(2) * z + np.sqrt(2 * z ** 2 + 4 * eps)) / 2; s = Delta / (np.sqrt(2) * u)
    fig, ax = plt.subplots(figsize=(0.75 * LARG, 2.6))
    ax.plot(eps, np.minimum(1, np.exp(eps) * 0.05 + 1e-5), ":", color=COR["cinza"], label="cota $e^\\varepsilon\\alpha+\\delta$")
    ax.plot(eps, norm.cdf(norm.ppf(0.05) + Delta / s), color=CATEG[1], label="NP, registro extremo")
    ax.plot(eps, norm.cdf(norm.ppf(0.05) + D / s), color=CATEG[0], label="NP, paciente 1")
    ax.plot(eps, ncx2.sf(chi2.ppf(0.95, p), p, (D / s) ** 2), color=CATEG[2], label="qui-quadrado, paciente 1")
    ax.set_xlabel("$\\varepsilon$ (com $\\delta=10^{-5}$)"); ax.set_ylabel("poder do ataque ($\\alpha=0{,}05$)")
    ax.set_ylim(0, 1); ax.legend(fontsize=6.3, loc="upper left"); virgula(ax)
    fig.tight_layout(); salvar(fig, CAP, "pertencimento")


if __name__ == "__main__":
    fig_qui2()
    fig_parcelas()
    fig_deriva()
    fig_pertencimento()
