"""Cap. 6: Figuras de dados do capítulo (geradas em figuras/cap06/)."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum.estilo_livro import CATEG, COR, DADOS, aplicar_estilo, salvar  # noqa: E402

aplicar_estilo()
CAP = "cap06"
LARG = 5.3
_fmt = FuncFormatter(lambda v, _: f"{v:.10g}".replace(".", ",").replace("-", "−"))


def virgula(ax, eixos="xy"):
    if "x" in eixos:
        ax.xaxis.set_major_formatter(_fmt)
    if "y" in eixos:
        ax.yaxis.set_major_formatter(_fmt)


# ---------------------------------------------------------------------------
# 1. Longley: autovalores da matriz de correlações e a última componente
# ---------------------------------------------------------------------------
def fig_longley():
    L = pd.read_csv(DADOS / "longley.csv")
    nomes = ["GNP.deflator", "GNP", "Unemployed", "Armed.Forces", "Population", "Year"]
    rot = ["deflator", "PIB", "desemprego", "forças arm.", "população", "ano"]
    lam, V = np.linalg.eigh(np.corrcoef(L[nomes].to_numpy().T)); lam, V = lam[::-1], V[:, ::-1]
    v6 = V[:, 5] * np.sign(V[1, 5]) * -1
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(LARG, 2.4))
    a1.semilogy(np.arange(1, 7), lam, "o-", color=CATEG[0], ms=4)
    for k, l in enumerate(lam):
        a1.annotate(f"{l:.4g}".replace(".", ","), (k + 1, l), xytext=(5, 2), textcoords="offset points", fontsize=6.3)
    a1.set_xlabel("componente $j$"); a1.set_ylabel(r"autovalor $\lambda_j$")
    a1.set_title("(a) autovalores de $\\mathbf{R}$", fontsize=8.5, loc="left")
    a2.barh(np.arange(6), v6, color=[CATEG[1] if abs(t) > 0.25 else COR["cinza"] for t in v6])
    a2.set_yticks(np.arange(6), rot, fontsize=7); a2.invert_yaxis()
    a2.axvline(0, color=COR["cinza"], lw=0.5)
    a2.set_xlabel("coeficiente em $\\mathbf{v}_6$")
    a2.set_title("(b) a direção quase constante", fontsize=8.5, loc="left")
    a2.grid(axis="y", visible=False)
    virgula(a2, "x")
    fig.tight_layout()
    salvar(fig, CAP, "longley_pca")


# ---------------------------------------------------------------------------
# 2. AMMI: biplot da interação genótipo x ambiente
# ---------------------------------------------------------------------------
def fig_ammi():
    w = pd.read_csv(DADOS / "yan_trigo.csv")
    Y = w.pivot_table(index="genotipo", columns="ambiente", values="producao")
    m = Y.to_numpy().mean()
    I = (Y.sub(Y.mean(axis=1), axis=0).sub(Y.mean(axis=0), axis=1) + m).to_numpy()
    U, s, Vt = np.linalg.svd(I)
    G = U[:, :2] * np.sqrt(s[:2]); E = Vt[:2].T * np.sqrt(s[:2])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(LARG, 2.7), gridspec_kw={"width_ratios": [1, 1.5]})
    prop = s ** 2 / (s ** 2).sum()
    a1.bar(np.arange(1, len(s)), prop[:-1], color=CATEG[0]); a1.set_xticks(range(1, 9))
    a1.set_xlabel("componente"); a1.set_ylabel("fração da SQ de interação")
    a1.set_title("(a) $\\sigma_k^2/\\sum\\sigma_j^2$", fontsize=8.5, loc="left")
    a1.grid(axis="x", visible=False); virgula(a1, "y")
    a2.axhline(0, color=COR["cinza"], lw=0.5); a2.axvline(0, color=COR["cinza"], lw=0.5)
    a2.scatter(G[:, 0], G[:, 1], s=10, color=CATEG[0], zorder=3)
    for k, nome in enumerate(Y.index):
        a2.annotate(nome, G[k], xytext=(2, 2), textcoords="offset points", fontsize=5.8, color=CATEG[0])
    for k, nome in enumerate(Y.columns):
        a2.annotate("", E[k], (0, 0), arrowprops=dict(arrowstyle="->", color=CATEG[1], lw=0.8))
        a2.annotate(nome, E[k], xytext=(2, -6 if E[k, 1] < 0 else 2), textcoords="offset points", fontsize=5.8, color=CATEG[1])
    a2.set_xlabel(f"CP1 ({100 * prop[0]:.1f}%)".replace(".", ","))
    a2.set_ylabel(f"CP2 ({100 * prop[1]:.1f}%)".replace(".", ","))
    a2.set_title("(b) biplot AMMI2", fontsize=8.5, loc="left")
    virgula(a2)
    fig.tight_layout()
    salvar(fig, CAP, "ammi_trigo")


# ---------------------------------------------------------------------------
# 3. Markov: convergência para o equilíbrio
# ---------------------------------------------------------------------------
def fig_markov():
    P = np.array([[0.960, 0.015, 0.025], [0.250, 0.600, 0.150], [0.040, 0.030, 0.930]])
    P2 = P.copy(); P2[1] = [0.50, 0.35, 0.15]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(LARG, 2.4))
    ts = np.arange(0, 61)
    for c, (nome, M) in enumerate((("atual", P), ("saída do desemprego dobrada", P2))):
        val, vec = np.linalg.eig(M.T); pi = np.real(vec[:, np.argmax(val.real)]); pi /= pi.sum()
        lam2 = np.sort(np.abs(val))[-2]
        x = np.array([0.0, 1, 0]); d = []
        for _ in ts:
            d.append(np.abs(x - pi).sum()); x = x @ M
        a1.semilogy(ts, d, color=CATEG[c], label=nome)
        a1.semilogy(ts, d[24] * lam2 ** (ts - 24.0), color=CATEG[c], ls=":", lw=0.9)
        if c == 0:
            traj = np.array([np.array([0.0, 1, 0]) @ np.linalg.matrix_power(M, t) for t in ts])
            for j, rot in enumerate(("empregado", "desempregado", "fora da força")):
                a2.plot(ts, traj[:, j], color=CATEG[j], label=rot)
                a2.axhline(pi[j], color=CATEG[j], ls=":", lw=0.8)
    a1.set_xlabel("meses"); a1.set_ylabel("distância ao equilíbrio")
    a1.legend(fontsize=6.2, loc="upper right")
    a1.set_title("(a) decaimento $\\approx\\lambda_2^t$", fontsize=8.5, loc="left")
    a2.set_xlabel("meses"); a2.set_ylabel("probabilidade")
    a2.legend(fontsize=6.2, loc="lower right", bbox_to_anchor=(1.0, 0.38))
    a2.set_title("(b) começando desempregado", fontsize=8.5, loc="left")
    virgula(a2, "y")
    fig.tight_layout()
    salvar(fig, CAP, "markov_convergencia")


# ---------------------------------------------------------------------------
# 4. Dígitos: variância explicada e reconstruções
# ---------------------------------------------------------------------------
def fig_digitos():
    dig = pd.read_csv(DADOS / "digitos.csv")
    X = dig.drop(columns="digito").to_numpy(float); mu = X.mean(0); Xc = X - mu
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    ac = np.cumsum(s ** 2) / (s ** 2).sum()
    fig = plt.figure(figsize=(LARG, 2.6))
    a1 = fig.add_axes([0.08, 0.17, 0.33, 0.72])
    a1.plot(np.arange(1, 65), ac, color=CATEG[0])
    for q in (0.8, 0.9, 0.95):
        k = int(np.argmax(ac >= q)) + 1
        a1.plot(k, ac[k - 1], "o", ms=3, color=CATEG[1])
        a1.annotate(f"{k}", (k, ac[k - 1]), xytext=(3, -9), textcoords="offset points", fontsize=6.5)
    a1.set_xlabel("número de componentes $k$"); a1.set_ylabel("variância acumulada")
    a1.set_title("(a) fração da variância", fontsize=8.5, loc="left")
    virgula(a1, "y")
    ks = [1, 5, 10, 20, 64]; ex = [0, 3, 7]
    fig.text(0.50, 0.92, "(b) reconstruções com $k$ componentes", fontsize=8.5)
    for i, idx in enumerate(ex):
        for j, k in enumerate(ks):
            ax = fig.add_axes([0.50 + j * 0.095, 0.62 - i * 0.26, 0.085, 0.22])
            img = mu + U[idx, :k] * s[:k] @ Vt[:k]
            ax.imshow(img.reshape(8, 8), cmap="Greys", vmin=0, vmax=16)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            if i == 2:
                ax.set_xlabel(f"$k={k}$", fontsize=6.5)
    salvar(fig, CAP, "digitos_pca")


# ---------------------------------------------------------------------------
# 5. Gradiente descendente e espectro da hessiana
# ---------------------------------------------------------------------------
def fig_gradiente():
    db = pd.read_csv(DADOS / "diabetes.csv")
    y = db["y"].to_numpy(float); n = len(y); Z = db.iloc[:, :10].to_numpy(float)
    Xp = np.column_stack([np.ones(n), (Z - Z.mean(0)) / Z.std(0, ddof=1)])
    Xb = np.column_stack([np.ones(n), Z])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(LARG, 2.5))
    for c, (nome, X) in enumerate((("padronizado", Xp), ("bruto", Xb))):
        ev = np.linalg.eigvalsh(X.T @ X / n)
        a1.semilogy(np.arange(1, 12), ev[::-1], "o-", ms=3, color=CATEG[c], label=nome)
    a1.set_xlabel("$j$"); a1.set_ylabel("autovalor da hessiana"); a1.set_xticks([1, 3, 5, 7, 9, 11])
    a1.legend(fontsize=6.5); a1.set_title("(a) espectro de $\\mathbf{X}^\\top\\mathbf{X}/n$", fontsize=8.5, loc="left")
    ev = np.linalg.eigvalsh(Xp.T @ Xp / n); Lm, mu = ev.max(), ev.min()
    bs = np.linalg.solve(Xp.T @ Xp, Xp.T @ y)
    for c, (nome, eta) in enumerate({"$\\eta=1/L$": 1 / Lm, "$\\eta=2/(L+\\mu)$": 2 / (Lm + mu),
                                     "$\\eta=2{,}02/L$": 2.02 / Lm}.items()):
        b = np.zeros(11); d = []
        for _ in range(400):
            b = b - eta * Xp.T @ (Xp @ b - y) / n; d.append(np.linalg.norm(b - bs))
        a2.semilogy(np.arange(1, 401), d, color=CATEG[c], label=nome)
    a2.set_ylim(1, 1e4)
    a2.set_xlabel("iteração"); a2.set_ylabel(r"$\|\mathbf{b}_k-\widehat{\mathbf{b}}\|$")
    a2.legend(fontsize=6.5, loc="upper left")
    a2.set_title("(b) gradiente (dados padronizados)", fontsize=8.5, loc="left")
    fig.tight_layout()
    salvar(fig, CAP, "gradiente_curvatura")


# ---------------------------------------------------------------------------
# 6. Embeddings de palavras: espectro da PPMI e mapa 2D
# ---------------------------------------------------------------------------
def fig_embeddings():
    import re
    from collections import Counter
    d = pd.read_csv(DADOS / "austen.csv", keep_default_na=False)
    p = re.findall(r"[a-z]+", " ".join(d["texto"]).lower())
    fr = sorted(Counter(p).items(), key=lambda kv: (-kv[1], kv[0]))
    V = 2000; vocab = [w for w, _ in fr[:V]]; pos = {w: i for i, w in enumerate(vocab)}
    ids = np.array([pos.get(w, -1) for w in p])
    C = np.zeros((V, V))
    for o in (1, 2):
        i, j = ids[:-o], ids[o:]; ok = (i >= 0) & (j >= 0)
        np.add.at(C, (i[ok], j[ok]), 1)
    C = C + C.T; N = C.sum(); pi = C.sum(1) / N
    pc = C.sum(0) ** 0.75; pc /= pc.sum()
    with np.errstate(divide="ignore"):
        M = np.log((C / N) / np.outer(pi, pc))
    M[~np.isfinite(M) | (M < 0)] = 0
    U, s, _ = np.linalg.svd(M)
    W = U[:, :100] * np.sqrt(s[:100]); W /= np.linalg.norm(W, axis=1, keepdims=True)
    grupos = {"personagens": ["elizabeth", "emma", "elinor", "anne", "fanny", "catherine", "darcy", "edmund"],
              "família": ["mother", "father", "sister", "brother", "aunt", "uncle", "daughter"],
              "números": ["two", "three", "four", "five", "six", "ten", "twenty"],
              "sentimentos": ["happy", "glad", "miserable", "pleased", "sorry", "comfortable"],
              "movimento": ["walked", "went", "ran", "came", "drove", "returned"]}
    sel = [pos[w] for g in grupos.values() for w in g]
    Z = W[sel] - W[sel].mean(0)
    _, _, Vt = np.linalg.svd(Z, full_matrices=False); P = Z @ Vt[:2].T
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(LARG, 2.7), gridspec_kw={"width_ratios": [1, 1.5]})
    a1.loglog(np.arange(1, V + 1), s, color=CATEG[0])
    a1.axvline(100, color=CATEG[1], ls=":", lw=0.9)
    a1.set_xlabel("$j$"); a1.set_ylabel("$\\sigma_j$")
    a1.set_title("(a) valores singulares da PPMI", fontsize=8.5, loc="left")
    k = 0
    for c, (g, ws) in enumerate(grupos.items()):
        pts = P[k:k + len(ws)]; k += len(ws)
        a2.scatter(pts[:, 0], pts[:, 1], s=9, color=CATEG[c], label=g)
        for (x, y), w in zip(pts, ws):
            a2.annotate(w, (x, y), xytext=(2, 2), textcoords="offset points", fontsize=5.6, color=CATEG[c])
    a2.legend(fontsize=5.8, loc="best", handletextpad=0.1)
    a2.set_xticks([]); a2.set_yticks([])
    a2.set_title("(b) 35 palavras: duas primeiras componentes", fontsize=8.5, loc="left")
    fig.tight_layout()
    salvar(fig, CAP, "embeddings")


if __name__ == "__main__":
    fig_longley()
    fig_ammi()
    fig_markov()
    fig_digitos()
    fig_gradiente()
    fig_embeddings()
